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
2. ``chunk_gated_delta_rule_seq`` -> the bf16 variant in ``chunk_seq_wh.py``, halving an
   L1-resident fp32 relayout that does not otherwise fit.
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
import inspect

import models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops as _shared_ops
import models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_seq as _shared_seq
import models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_deltanet as _shared
import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.blackhole.qwen36.tt.chunk_seq_wh import chunk_gated_delta_rule_seq_dispatch

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
    if getattr(_shared, "_qwen36_wh_compat_applied", False):
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

        Upstream keeps all FOUR of that shape in L1 -- the k(x)delta outer product, its beta
        scaling, the decayed h, and the sum -- which is right at B=1 and impossible at B=32, where
        each is far larger than the free L1 per bank. This is the decode leg the WH fork
        deliberately declines (wh_decode_fork_applies), so without this override B=32 had nowhere
        to go: the fork refused it for exactly this reason and upstream then tried L1 anyway.
        Blackhole spreads the same tensor over more banks of a larger L1 and keeps the upstream
        function untouched.

        ONLY the memory configs change -- op sequence, dtypes and compute config are upstream's, so
        the result is bit-identical. DRAM vs L1 is placement, not arithmetic.
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

    _shared._qwen36_wh_compat_applied = True


apply()
