# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Residual-stream layout selector for Llama-3.1-8B prefill.

Two schemes, selected by ``LLAMA31_8B_SHARDED_RESIDUAL``:

  * **sharded** (DEFAULT) — the residual is ``emb/tp`` per TP column (the DeepSeek/Kimi/GLM/M3 layout).
    Full width is reconstituted only where a column-parallel projection needs it: one all-gather per
    norm output, shared by every consumer downstream of that norm. Attention and both MLPs close with
    a reduce-scatter and no all-gather. ``LLAMA31_8B_RESIDUAL_NORM`` picks whether that gather sits
    before a single-pass norm or after a distributed one (see DEFAULT_NORM_MODE).

  * **replicated** (``LLAMA31_8B_SHARDED_RESIDUAL=0``) — the residual is full emb on every TP column.
    Attention and the MLP close with an all-reduce; norms are single-op.

Every consumer (embedding, norms, attention's closing collective, the MLP, the LM-head tail) reads
these helpers so a run cannot end up half-sharded.

Borrowed as-is from ``minimax_m3/tt/residual.py``: the donor measured the sharded default better on
device time and op launches at this same mesh with KV PCC bit-identical, and it is the layout the
sibling models use. This model's hidden 4096 / tp 4 = 1024 is tile-aligned, so the sharded residual
carries no output-dim padding into a TP column's slice.
"""

import os

# Default ON: measured better on device time and op launches than the replicated layout (KV PCC
# bit-identical), and it is the layout the sibling models use. LLAMA31_8B_SHARDED_RESIDUAL=0 restores the
# replicated layout, kept for bisecting against baselines measured on it.
DEFAULT_USE_SHARDED_RESIDUAL = True

# Where the per-norm all-gather sits under a sharded residual. "gather_first" (DEFAULT): all-gather
# the residual shard to full emb, then one single-pass ttnn.rms_norm — measured better on both device
# time and op launches. "distributed": the 3-op distributed RMSNorm on the emb/tp shard, then
# all-gather the normed result — the sibling models' shape, kept selectable for an A/B.
DEFAULT_NORM_MODE = "gather_first"
_NORM_MODES = ("distributed", "gather_first")


def use_sharded_residual() -> bool:
    """True -> ``emb/tp``-sharded residual stream; False -> full-emb replicated. See module docstring."""
    v = os.environ.get("LLAMA31_8B_SHARDED_RESIDUAL")
    if v is None:
        return DEFAULT_USE_SHARDED_RESIDUAL
    return v.strip().lower() in ("1", "true", "yes", "on")


def norm_mode() -> str:
    """Either ``gather_first`` (default) or ``distributed`` — see DEFAULT_NORM_MODE. Only meaningful
    when the residual is sharded; the replicated scheme always norms full emb in one pass."""
    v = (os.environ.get("LLAMA31_8B_RESIDUAL_NORM") or DEFAULT_NORM_MODE).strip().lower()
    if v not in _NORM_MODES:
        raise ValueError(f"LLAMA31_8B_RESIDUAL_NORM={v!r} must be one of {_NORM_MODES}")
    return v


def use_distributed_norm() -> bool:
    """True -> the decoder norms run the 3-op distributed form on the emb/tp shard. False -> they run
    single-pass on a gathered full-emb input (either the replicated scheme, or gather_first)."""
    return use_sharded_residual() and norm_mode() == "distributed"


def gather_before_norm() -> bool:
    """True -> the layer's per-norm all-gather runs BEFORE the norm (gather_first); False -> after it."""
    return use_sharded_residual() and norm_mode() == "gather_first"
