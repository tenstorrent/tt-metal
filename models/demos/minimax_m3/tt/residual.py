# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Residual-stream layout selector for MiniMax-M3 prefill.

Two schemes, one env var:

  * **replicated** (``M3_SHARDED_RESIDUAL=0``) — the residual is FULL emb on every TP column:
    ``[1, 1, s_local, 6144]`` everywhere. Attention closes with an all-reduce (RS + AG), the MoE
    all-gathers its reduce-scattered output back to full emb, and the shared expert all-reduces its
    own output. Norms are single-op (each column normalizes the same full vector).

  * **sharded** (DEFAULT) — the residual is ``emb/tp``: ``[1, 1, s_local, 1536]``
    per column, the DeepSeek/Kimi/GLM layout. Full width is reconstituted only where a
    column-parallel projection needs it as its K dim: ONE all-gather per norm output, shared by every
    consumer downstream of that norm (q/k/v/index for attention; router + shared expert + dispatch
    for the MoE). Attention and both MLPs then close with a reduce-scatter and no all-gather.
    ``M3_SHARDED_RESIDUAL_NORM`` picks whether that gather sits after a distributed norm or before a
    single-pass one (see DEFAULT_NORM_MODE).

The all-gathers RELOCATE rather than vanish — 3 removed, 2 added of the same 7.9 MB shape — so the
genuine win is one net all-gather per layer (the shared expert stops paying its own because it shares
the pre-MLP one) plus elementwise adds on ``emb/tp`` instead of full emb. See
M3_SHARDED_RESIDUAL_PROMPT.md §2.

Every consumer reads this ONE helper so a run cannot end up half-sharded: the embedding, both norms,
attention's closing collective, both MLPs and the LM-head tail all branch on it. ``log_scheme_once``
prints the active scheme at model build, because a silently-disabled layout change is invisible in
every metric except op count.
"""

import os

from loguru import logger

# Default ON (with DEFAULT_NORM_MODE = "gather_first"). Measured against the 46a19f8 replicated
# baseline on the 8x4 galaxy: device time -2.2% (max-across-chips, ~-165 us/layer), op launches -1.0%,
# wall-clock -0.1% (inside the 1-2% run-to-run band), KV PCC bit-identical. So it is better on the two
# axes that will matter once a traced runtime converts the device saving, and neutral on the one that
# does not. It is also the layout the sibling models (DeepSeek/Kimi/GLM) use, which keeps the shared EP
# machinery on a single well-trodden shape.
#
# Set M3_SHARDED_RESIDUAL=0 for the replicated layout (the path the older baselines were measured on,
# so it stays available for bisecting a regression against them).
DEFAULT_USE_SHARDED_RESIDUAL = True

# WHERE the per-norm all-gather sits, under a sharded residual. Both variants gather exactly once per
# norm and hand full emb to the column-parallel consumers:
#
#   "gather_first" (DEFAULT) — all-gather the residual shard to full emb FIRST, then one single-pass
#       ttnn.rms_norm over the full width. 2 ops per norm. Every TP column redundantly normalizes the
#       same 6144-wide vector, which sounds wasteful and measures free.
#
#   "distributed" (DeepSeek/Kimi/GLM's shape) — norm the emb/tp shard with the 3-op distributed
#       RMSNorm (pre_all_gather -> tiny stats AG -> post_all_gather), THEN all-gather the normed
#       result. 4 ops per norm; the big ops touch only emb/tp.
#
# MEASURED on the 8x4 galaxy (Tracy capture, LEVEL=3 LAYERS=6 CACHE=25600, 42 layer-visits, against
# the 46a19f8 baseline capture 2026_08_05_18_07_13) — "distributed" loses on BOTH axes:
#
#                       device-op launches      DEVICE kernel time (max-across-chips)   wall-clock
#   baseline                  2785                        309.53 ms                        --
#   gather_first              2757  (-1.0%)               302.61 ms  (-2.2%)             -0.1%
#   distributed               2925  (+5.0%)               308.90 ms  (-0.2%)             -4.6%
#
# The reason is that narrowing the norm to emb/tp saves ~27 us per norm, while the tiny (40 KB) stats
# all-gather it needs costs ~50 us of device time -- so the distributed norm is ~2 us/norm net-negative
# on device, and then pays +2 op launches on top. There is no regime in which it wins, including after
# a traced runtime makes launches free. Kept selectable because it is the shape the sibling models use,
# so an A/B stays one env var away.
DEFAULT_NORM_MODE = "gather_first"
_NORM_MODES = ("distributed", "gather_first")

_scheme_logged = False


def use_sharded_residual() -> bool:
    """True -> ``emb/tp``-sharded residual stream; False -> full-emb replicated. See module docstring."""
    v = os.environ.get("M3_SHARDED_RESIDUAL")
    if v is None:
        return DEFAULT_USE_SHARDED_RESIDUAL
    return v.strip().lower() in ("1", "true", "yes", "on")


def norm_mode() -> str:
    """Either ``gather_first`` (default) or ``distributed`` — see DEFAULT_NORM_MODE. Only meaningful
    when the residual is sharded; the replicated scheme always norms full emb in one pass."""
    v = (os.environ.get("M3_SHARDED_RESIDUAL_NORM") or DEFAULT_NORM_MODE).strip().lower()
    if v not in _NORM_MODES:
        raise ValueError(f"M3_SHARDED_RESIDUAL_NORM={v!r} must be one of {_NORM_MODES}")
    return v


def use_distributed_norm() -> bool:
    """True -> the decoder norms run the 3-op distributed form on the emb/tp shard. False -> they run
    single-pass on a gathered full-emb input (either the replicated scheme, or gather_first)."""
    return use_sharded_residual() and norm_mode() == "distributed"


def gather_before_norm() -> bool:
    """True -> the layer's per-norm all-gather runs BEFORE the norm (gather_first); False -> after it."""
    return use_sharded_residual() and norm_mode() == "gather_first"


def log_scheme_once(mesh_config=None) -> bool:
    """Log the active residual scheme exactly once per process and return it.

    Called from the model build. A sharded residual requested on a TP=1 mesh degenerates to the
    replicated scheme (nothing to shard); that is logged as a WARNING rather than accepted silently,
    since it makes every downstream measurement look 'layout-neutral' when the layout never changed.
    """
    global _scheme_logged
    requested = use_sharded_residual()
    tp = getattr(mesh_config, "tp", None)
    active = requested and (tp is None or tp > 1)
    if not _scheme_logged:
        _scheme_logged = True
        if requested and not active:
            logger.warning(
                f"[residual] SHARDED residual REQUESTED (M3_SHARDED_RESIDUAL) but the mesh has tp={tp}: "
                f"there is nothing to shard, so the REPLICATED scheme is active. Every collective and op "
                f"count below is the baseline's."
            )
        elif active:
            logger.info(
                f"[residual] residual stream: SHARDED emb/tp (tp={tp}, norm={norm_mode()}) — one TP "
                f"all-gather per norm {'after' if norm_mode() == 'distributed' else 'before'} it; "
                f"attention and both MLPs close with a reduce-scatter only"
            )
        else:
            logger.info(f"[residual] residual stream: REPLICATED full emb (tp={tp}, M3_SHARDED_RESIDUAL=0)")
    return active
