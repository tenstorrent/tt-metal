# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Wormhole adjustments to the shared experimental Gated DeltaNet module.

This model runs on Blackhole (P150) and Wormhole (n300). The shared kernels in
``models/experimental/gated_attention_gated_deltanet`` were tuned for Blackhole, which has
substantially more L1 to spend:

    Blackhole  140 worker cores x 1,572,864 B L1  ~= 210 MB,  80 interleave banks
    Wormhole    80 worker cores x 1,499,136 B L1  ~= 114 MB,  64 interleave banks

Blackhole therefore has ~1.84x the total L1 (the delta is core count, not L1 per core), and with
64 banks instead of 80 the same interleaved tensor costs ~1.25x more per bank on Wormhole.
Working sets that fit comfortably there do not fit here, and the Tensix circular buffers those
kernels allocate are L1-only by hardware -- compute cannot read DRAM -- so the only things that
can move are the surrounding activations.

The shared module is NOT edited: both adjustments are applied from inside this model's folder.
``apply()`` is idempotent and is called by the qwen36 GDN entry points before any GDN forward.

Two adjustments, both no-ops on Blackhole:

1. ``_seq_memory_config`` -> DRAM. Upstream keeps short sequences in L1 for speed; on Wormhole
   those activations no longer fit beside the chunk-seq kernel's circular buffers and fail as
   "clash with L1 buffers" for T <= 512.

2. ``chunk_gated_delta_rule_seq`` -> the bf16 variant in ``chunk_seq_wh.py``. Its
   L1-resident ``[BH, L, V]`` fp32 relayout needs 33,554,432 B at L=2048, which does not fit;
   bf16 halves it to 16,777,216 B. That dispatch delegates to the upstream function whenever
   ``is_blackhole()``, so Blackhole always runs upstream code.

NOTE on blast radius: these rebind module-globals in the shared module, so within a process that
imports it the change is visible to any other model using it. Both are guarded by
``is_blackhole()`` evaluated per call (not at import, which would need an open device), so
Blackhole behaviour is bit-for-bit unchanged and only Wormhole takes the new paths.
"""
import inspect

import models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_seq as _shared_seq
import models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_deltanet as _shared
from models.common.utility_functions import is_blackhole
from models.demos.blackhole.qwen36.tt.chunk_seq_wh import chunk_gated_delta_rule_seq_dispatch

_FLAG = "_qwen36_wh_compat_applied"

# chunk_seq_wh.py is a verbatim copy of this upstream function with one dtype change. If
# upstream edits it, the copy is stale -- fail loudly rather than silently running old kernel code.
_UPSTREAM_ANCHOR = "ttnn.typecast(out_4d, ttnn.float32, memory_config=_out_l1)"


def _check_fork_is_current():
    try:
        upstream_src = inspect.getsource(_shared_seq.chunk_gated_delta_rule_seq)
    except (OSError, TypeError):  # source unavailable (zipimport etc.) -- skip the check
        return
    if _UPSTREAM_ANCHOR not in upstream_src:
        raise RuntimeError(
            "models/demos/blackhole/qwen36/tt/chunk_seq_wh.py is a copy of "
            "chunk_gated_delta_rule_seq() from ttnn_delta_rule_seq.py, and upstream has changed: "
            f"the anchor {_UPSTREAM_ANCHOR!r} is gone. Re-copy that function into chunk_seq_wh.py "
            "and re-apply the single bf16 edit (marked 'THE ONE CHANGE vs upstream')."
        )


def apply():
    """Install the Wormhole GDN adjustments on the shared module. Idempotent."""
    if getattr(_shared, _FLAG, False):
        return

    _check_fork_is_current()

    # --- 1. chunk-seq activations: DRAM on Wormhole -------------------------------------- #
    _orig_seq_memory_config = _shared._seq_memory_config

    def _seq_memory_config(seq_len):
        """Wormhole: always DRAM for the chunk-seq activations.

        Upstream is ``L1 if seq_len <= _L1_SEQ_THRESHOLD else None``. On Wormhole those L1
        activations collide with the chunk-seq kernel's statically allocated circular buffers
        ("clash with L1 buffers", T <= 512). Returning None selects DRAM -- the same escape hatch
        upstream already uses for long sequences.

        Why not force ``valid_len`` instead (which upstream also routes to DRAM): passing
        valid_len makes the module build the conv-tail one-hot selector on the host via
        ``ttnn.from_torch(...)``. That is a host->device write, and inside begin_trace_capture it
        raises "TT_FATAL: Writes are not supported during trace capture" and wedges the device,
        because the fatal fires before end_trace_capture can run. A memory_config carries no host
        op, so it is trace-safe.
        """
        if not is_blackhole():
            return None
        return _orig_seq_memory_config(seq_len)

    _shared._seq_memory_config = _seq_memory_config

    # --- 2. chunk-seq kernel wrapper: bf16 output relayout on Wormhole -------------------- #
    # The adapter calls this as a module global, so rebinding it here takes effect.
    _shared_seq.chunk_gated_delta_rule_seq = chunk_gated_delta_rule_seq_dispatch

    setattr(_shared, _FLAG, True)


apply()
