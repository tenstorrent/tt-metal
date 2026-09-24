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

    Blackhole: no exposure. Both overrides delegate to the upstream implementation whenever
    ``is_blackhole()``, so any model in the process -- qwen36 or not -- runs upstream code.

    Wormhole: real exposure, and it is process-wide, not qwen36-scoped. Importing any qwen36
    GDN module (``tt/gdn/tp.py``, ``tt/gdn/decode.py``, or this module) runs ``apply()`` as an
    import side effect, which rebinds ``_seq_memory_config`` and
    ``chunk_gated_delta_rule_seq`` on the SHARED module. From that point any *other* Wormhole
    model that imports ``models/experimental/gated_attention_gated_deltanet`` in the same
    process silently gets DRAM chunk-seq activations and the bf16 output relayout, even though
    it never imported anything under qwen36. It is not opt-in and there is no per-caller
    scoping.

    Why that is acceptable today: qwen36 is the only Wormhole consumer of the shared GDN
    module, so no other model can observe the rebind. Both overrides are also strictly
    L1-relief changes on a path that OOMs without them -- the alternative for a co-resident
    Wormhole model is not "upstream numerics", it is a failed allocation.

    What to do if that stops holding: if a second Wormhole model starts using the shared GDN
    module, do NOT leave this as an import side effect. Drop the module-level ``apply()`` call
    at the bottom of this file, keep the explicit calls at the qwen36 GDN entry points, and
    push the dtype/memory-config choice down into the shared module as a parameter so each
    caller picks its own. The pytest process is the case to watch: a single session that
    collects both qwen36 and another Wormhole GDN model would share one interpreter, and
    collection-time imports alone are enough to flip the globals -- test order, not the model
    under test, would decide which kernels run.
"""
import inspect

import models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops as _shared_ops
import models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_seq as _shared_seq
import models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_deltanet as _shared
import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.blackhole.qwen36.tt.chunk_seq_wh import chunk_gated_delta_rule_seq_dispatch

_FLAG = "_qwen36_wh_compat_applied"

# Largest [B,H,K,V] state tensor this model will keep L1-resident on Wormhole. Mirrors
# recurrent_decode_wh._OUTER_L1_BUDGET_BYTES, which gates the same tensor on the fork side.
_STATE_L1_BUDGET_BYTES = 8 << 20

# chunk_seq_wh.py is a verbatim copy of this upstream function with one dtype change. If
# upstream edits it, the copy is stale -- fail loudly, not run old kernels.
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

    # --- 3. decode state write: DRAM for the [B,H,K,V] tensors that do not fit L1 --------- #
    # recurrent_gated_delta_rule_decode_ttnn calls this as a module global, so rebinding takes
    # effect for the upstream decode leg (the one the WH fork hands B=32 back to).
    _orig_fused_decay_and_write = _shared_ops.fused_decay_and_write_ttnn

    def _fused_decay_and_write(h, k_t, delta, decay_t, beta_t, device=None, apply_decay=True):
        """Wormhole: place the [B,H,K,V] state-write intermediates in DRAM when they miss L1.

        Upstream keeps every one of them in L1 -- "Decode opt: keep state-write operands in L1
        (tiny at B=1)" -- and there are FOUR of that shape: the k(x)delta outer product, its beta
        scaling, the decayed h, and the sum. At B=1 each is 512 KB and L1 is the right call.

        At B=32 with the fp32 state each is [32,8,128,128] = 16,777,216 B. Wormhole interleaves
        over 64 banks, so that is 262,144 B/bank against the 1,368,864 B a bank has -- and with the
        model resident only ~186 KB/bank is free, so the very first one dies with

            Out of Memory: Not enough space to allocate 16777216 B L1 buffer across 64 banks

        This is the decode leg the WH fork deliberately declines (wh_decode_fork_applies), so
        before this override B=32 decode had nowhere to go: the fork refused it for exactly this
        reason and upstream then tried to do it in L1 anyway.

        Blackhole spreads the same tensor over 80 banks of a larger L1 and never trips this, so it
        keeps the upstream function untouched.

        ONLY the memory configs change. The op sequence, dtypes and compute config are upstream's,
        so the result is bit-identical -- DRAM vs L1 is placement, not arithmetic.
        """
        if is_blackhole():
            return _orig_fused_decay_and_write(
                h=h, k_t=k_t, delta=delta, decay_t=decay_t, beta_t=beta_t, device=device, apply_decay=apply_decay
            )

        B, H, K, V = h.shape[0], h.shape[1], h.shape[2], h.shape[3]
        _itemsize = 4 if h.dtype == ttnn.float32 else 2
        _L1 = ttnn.L1_MEMORY_CONFIG
        # `big` is where every [B,H,K,V] tensor goes; the [B,H,1,1] and [B,H,K,1] operands are
        # orders of magnitude smaller and stay in L1 as upstream has them.
        big = _L1 if B * H * K * V * _itemsize <= _STATE_L1_BUDGET_BYTES else ttnn.DRAM_MEMORY_CONFIG

        decay = ttnn.reshape(decay_t, [B, H, 1, 1], memory_config=_L1)
        beta_expanded = ttnn.reshape(beta_t, [B, H, 1, 1], memory_config=_L1)
        k_col = ttnn.reshape(k_t, [B, H, K, 1], memory_config=_L1)
        d_row = ttnn.reshape(delta, [B, H, 1, V], memory_config=_L1)
        k_col = ttnn.to_memory_config(k_col, _L1)
        d_row = ttnn.to_memory_config(d_row, _L1)

        matmul_compute_cfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        outer = ttnn.matmul(
            k_col,
            d_row,
            memory_config=big,
            compute_kernel_config=matmul_compute_cfg,
            program_config=None,
        )
        outer = ttnn.multiply(outer, beta_expanded, memory_config=big)
        if apply_decay:
            h = ttnn.multiply(h, decay, memory_config=big)
        return ttnn.add(h, outer, memory_config=big)

    _shared_ops.fused_decay_and_write_ttnn = _fused_decay_and_write

    setattr(_shared, _FLAG, True)


apply()
