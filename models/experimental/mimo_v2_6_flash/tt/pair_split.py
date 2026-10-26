# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device side of the global KV pair split on one TP=8 replica (MIMO_KV_PAIR_SPLIT, vLLM mode; the layout and the
math are in reference/kv_pair_split.py).

Each chip of a global layer's pair (chips 2g, 2g + 1) stores its 64-position halves of every 128-position global
block, attends with its 16 query heads over them and scales its partial output by the merge weight from the two LSEs:

* ``positions``: one one-core program (kernels/attn/pair_positions.cpp) at the start of every decode and verify step
  writes each chip's write positions, SDPA positions and serial (one slot each) write positions from the step's global
  positions, which the trace advances with ``plus_one``. So the per-chip positions follow the global ones inside the
  trace, with nothing uploaded. The program keys on nothing that changes (generic_op hashes the runtime arguments'
  count, not their values), so it compiles in the decode trace's warm-up and serves the verify trace too.
* ``merge``: the partner's LSE (an all_gather of the LSE over the replica's row, then a tensor-args slice at the
  partner's index, a per-chip start), and ``alpha * o`` with ``alpha = 1 / (1 + exp(clamp(lse_p - lse, -80, 80)))``
  (reference.kv_pair_split.merge_weights_device): the pair's two scaled partials then add up to full attention in the
  row all-reduce after the o projection.

Every tensor here is persistent and made before any trace capture (TtMiMoModel.setup_vllm_kv_cache).
"""

import torch

import ttnn

from ..reference import kv_pair_split as ref
from . import kv as kv_lib
from .device_sampler import CB_SCRATCH, _cb, _is_dram, _kernel, _ranges

PARITY_WORDS = 16  # the per-chip parity page (64 bytes): word 0 the chip's parity, the rest zero


def _align64(nbytes):
    return (nbytes + 63) // 64 * 64


class PairSplit:
    def __init__(self, plan, max_batch, serial_slots=0):
        """``serial_slots``: the serial write rows a verify step uses (0 without a drafter)."""
        tp = plan.tp
        assert tp % ref.REP == 0 and max_batch * 4 <= 1024 and serial_slots <= max_batch
        self.plan, self.max_batch, self.serial_slots = plan, max_batch, serial_slots
        chips = torch.arange(tp, dtype=torch.int32)
        parity = torch.zeros(tp, PARITY_WORDS, dtype=torch.int32)
        parity[:, 0] = chips % ref.REP
        # Per chip: int32 [PARITY_WORDS], word 0 its parity (the positions kernel reads one 64-byte page).
        self.parity = kv_lib.per_chip(plan, parity, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        # Per chip: int32 [1], its parity (pair-split chunked MLA prefill's kv_pair_parity_tensor).
        self.prefill_parity = kv_lib.per_chip(plan, parity[:, :1].contiguous(), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        partner = torch.zeros(tp, 4, dtype=torch.int32)
        partner[:, 0] = chips ^ 1
        # Per chip: the slice start (partner, 0, 0, 0) of its partner's LSE in the row's gathered LSEs.
        self.partner_start = kv_lib.per_chip(plan, partner, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        idle = torch.full((max_batch,), -1, dtype=torch.int32)
        self.write = kv_lib.replicated(plan, idle, ttnn.int32)
        self.sdpa = kv_lib.replicated(plan, idle, ttnn.int32)
        self.serial = [kv_lib.replicated(plan, idle, ttnn.int32) for _ in range(serial_slots)]
        self.core = ttnn.CoreCoord(0, 0)

    # ------------------------------------------------------------------ device (trace-capturable)
    def positions(self, cur_pos):
        """The step's global positions ``cur_pos`` (int32 [B], DRAM row-major, -1 an idle slot) -> each chip's
        (write [B], sdpa [B], serial rows), the persistent buffers rewritten in place."""
        b, rows = self.max_batch, self.serial_slots
        assert tuple(cur_pos.shape) == (b,) and cur_pos.dtype == ttnn.int32 and _is_dram(cur_pos)
        assert cur_pos.layout == ttnn.ROW_MAJOR_LAYOUT
        outs = [self.write, self.sdpa, *self.serial]
        args = [cur_pos.buffer_address(), self.parity.buffer_address()] + [t.buffer_address() for t in outs]
        scratch = _align64(b * 4) + _align64(PARITY_WORDS * 4) + (2 + rows) * _align64(b * 4)
        cores = [self.core]
        kernel = _kernel(
            "attn/pair_positions.cpp",
            cores,
            _ranges(cores),
            [b, rows, CB_SCRATCH, PARITY_WORDS * 4],
            [args],
            ttnn.ReaderConfigDescriptor(),
        )
        program = ttnn.ProgramDescriptor(
            kernels=[kernel], semaphores=[], cbs=[_cb(CB_SCRATCH, scratch, _ranges(cores))]
        )
        ttnn.generic_op([cur_pos, self.parity, *outs], program)
        return self.write, self.sdpa, self.serial

    def partner(self, lse):
        """The pair partner's LSE, of ``lse``'s shape: the row's LSEs gathered along dim 0, then the partner's entry
        (a tensor-args slice: the start is per chip; the program keys on the shape only). The LSE is a TILE tensor
        whose last dim (and in decode, the 16 heads) does not fill its tiles, and the tensor-args slice checks the
        logical shape against whole tiles, so it slices a view of the whole tiles (logical = padded shape) and views
        the result back."""
        assert lse.shape[0] == 1, f"the LSE's dim 0 gathers the row: {tuple(lse.shape)}"
        gathered = self.plan.all_gather(lse, dim=0)
        tp = self.plan.tp
        tiles = ttnn.Shape(list(gathered.padded_shape))
        whole = ttnn.reshape(gathered, tiles, tiles)  # a view: same buffer, the padding counted in
        out = ttnn.slice(whole, self.partner_start, self.partner_start, slice_dim=0, num_devices=tp)
        gathered.deallocate(True)
        return ttnn.reshape(out, lse.shape, ttnn.Shape(list(out.padded_shape)))  # a view back to lse's shape

    def merge(self, o, lse, memory_config):
        """``alpha * o`` (bf16, of o's shape): o the chip's normalized partial attention (bf16, [.., rows, dv]), lse
        its fp32 LSE ([.., rows, 1], broadcast along o's last dim), alpha this chip's merge weight in fp32."""
        lse_p = self.partner(lse)
        d = ttnn.subtract(lse_p, lse, dtype=ttnn.float32, memory_config=memory_config)
        lse_p.deallocate(True)
        clamped = ttnn.clamp(d, min=-ref.LSE_CLAMP, max=ref.LSE_CLAMP, memory_config=memory_config)
        d.deallocate(True)
        e = ttnn.exp(clamped, memory_config=memory_config)
        clamped.deallocate(True)
        den = ttnn.add(e, 1.0, memory_config=memory_config)
        e.deallocate(True)
        alpha = ttnn.reciprocal(den, memory_config=memory_config)
        den.deallocate(True)
        out = ttnn.multiply(o, alpha, dtype=ttnn.bfloat16, memory_config=memory_config)
        alpha.deallocate(True)
        return out
