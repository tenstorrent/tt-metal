# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Wormhole adjustments to the shared experimental Gated DeltaNet module.

This model runs on Blackhole (P150) and Wormhole (n300). The shared kernels in
``models/experimental/gated_attention_gated_deltanet`` were tuned for Blackhole, which has
substantially more total L1 and more interleave banks, so working sets that fit comfortably there
do not fit here -- and the Tensix circular buffers those kernels allocate are L1-only by hardware,
so the only things that can move are the surrounding activations.

The shared module is NOT edited: the adjustments are applied from inside this model's folder.
``apply()`` is idempotent and runs from the qwen36 GDN entry points before any GDN forward.

Three adjustments, all no-ops on Blackhole:

1. ``_seq_memory_config`` -> DRAM. Upstream keeps short sequences in L1; on Wormhole those
   activations no longer fit beside the chunk-seq kernel's circular buffers.
2. ``chunk_gated_delta_rule_seq`` -> called with ``out_dtype=bfloat16``, halving an L1-resident
   fp32 relayout that does not otherwise fit.
3. ``fused_decay_and_write_ttnn`` -> DRAM for the [B,H,K,V] state-write intermediates once they
   exceed the L1 budget. See that override for why upstream cannot do B=32 in L1.

BLAST RADIUS: these rebind module globals on the SHARED module, so within a process that imports
it the change is visible to any other model using it -- and ``apply()`` runs as an import side
effect of any qwen36 GDN module, so it is not opt-in and has no per-caller scoping. Every override
is guarded by ``is_blackhole()`` evaluated PER CALL (not at import, which would need an open
device), so Blackhole is bit-for-bit unchanged and only Wormhole takes the new paths.

That is acceptable only because qwen36 is currently the sole Wormhole consumer of the shared
module, and because all three are strictly L1-relief on paths that OOM without them -- the
alternative for a co-resident Wormhole model is not "upstream numerics", it is a failed
allocation. If a SECOND Wormhole model starts using the shared module, do not leave this as an
import side effect: drop the module-level ``apply()`` below, keep the explicit entry-point calls,
and push the dtype/memory-config choice into the shared module as a per-caller parameter. Watch
pytest in particular -- one session collecting two such models shares an interpreter, and
collection-time imports alone would let test order decide which kernels run.
"""

import models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops as _shared_ops
import models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_seq as _shared_seq
import models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_deltanet as _shared
import ttnn
from models.common.utility_functions import is_blackhole

_FLAG = "_qwen36_wh_compat_applied"

# Largest [B,H,K,V] state tensor this model will keep L1-resident on Wormhole. Mirrors
# recurrent_decode_wh._OUTER_L1_BUDGET_BYTES, which gates the same tensor on the fork side.
_STATE_L1_BUDGET_BYTES = 8 << 20


def apply():
    """Install the Wormhole GDN adjustments on the shared module. Idempotent."""
    if getattr(_shared, _FLAG, False):
        return

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

    # --- 2. chunk-seq output relayout: bf16 on the Wormhole configs that need it --------- #
    # The adapter calls this as a module global, so rebinding it here takes effect.
    _orig_chunk_seq = _shared_seq.chunk_gated_delta_rule_seq

    def _chunk_gated_delta_rule_seq(*args, **kwargs):
        """Ask upstream for a bf16 output relayout instead of the default fp32.

        Upstream relayouts the kernel output as an L1-resident [BH,L,V] tensor. At fp32 that does
        not fit Wormhole's smaller L1 and dies with "Out of Memory"; bf16 halves it and costs
        nothing measurable (logit PCC 0.9998-1.0000).

        Blackhole absorbs the fp32, so it keeps the default. A 8-device T3K also keeps it: its
        per-chip head count makes the tensor smaller than the config that first needed the fix, and
        the fp32 path was measured to fit there. N150 KEEPS the fix -- at TP=1 it holds the full
        head count on one chip, twice N300's per-chip size, so it needs it more than N300, not
        less. T3K is detected via the mesh_device kwarg the adapter always passes.
        """
        if is_blackhole():
            return _orig_chunk_seq(*args, **kwargs)
        mesh = kwargs.get("mesh_device")
        if mesh is not None and mesh.get_num_devices() == 8:
            return _orig_chunk_seq(*args, **kwargs)
        kwargs.setdefault("out_dtype", ttnn.bfloat16)
        return _orig_chunk_seq(*args, **kwargs)

    _shared_seq.chunk_gated_delta_rule_seq = _chunk_gated_delta_rule_seq

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
