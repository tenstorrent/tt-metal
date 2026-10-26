# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Packed speculative-step inputs on one TP=8 replica (reference/spec_inputs.py has the layout and the torch mirrors
of these programs).

A verify step and an MTP draft each copy one int32 tensor to the device; a small program at the start of the verify
(draft) trace writes the trace's existing input buffers in place from it, before the captured step reads them. The
captured step is unchanged: it reads the same buffers, with the same values, as when the host copied each one. A
draft chained behind a verify (tt/spec_chain.py) copies nothing: the chain step writes ``mtp_packed`` on device.

Every tensor here is persistent and made before any trace capture (TtMiMoModel.setup_mtp): the derive programs read
the constants on every replay, and a buffer allocated after a capture can sit where that trace's replays write. The
derived buffers themselves are the traces' inputs, rewritten on every replay before anything reads them.
"""

import torch

import ttnn

from ..reference.spec_decode import SPEC_SLOTS
from ..reference.spec_inputs import (
    MTP_ROWS,
    VERIFY_ROWS,
    owned_slot_caps,
    owned_slot_columns,
    pack_mtp,
    pack_verify,
    slot_caps,
)
from . import kv as kv_lib


class SpecInputs:
    def __init__(self, plan, max_batch, kv_rep):
        # A row of the upload starts at a multiple of B int32s: whole 64-byte DRAM words for the row-major slices.
        assert max_batch % 16 == 0 and max_batch >= SPEC_SLOTS
        self.plan, self.max_batch = plan, max_batch
        tp = plan.tp
        idle = torch.full((max_batch,), -1, dtype=torch.int32)
        self.verify_packed = kv_lib.replicated(plan, pack_verify(torch.zeros_like(idle), idle), ttnn.int32)
        self.mtp_packed = kv_lib.replicated(plan, pack_mtp(max_batch), ttnn.int32)
        self.own_cap = kv_lib.per_chip(plan, owned_slot_caps(max_batch, tp, kv_rep), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        self.own_col = kv_lib.per_chip(plan, owned_slot_columns(max_batch, tp, kv_rep), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        self.slot_caps = [kv_lib.replicated(plan, cap, ttnn.int32) for cap in slot_caps(max_batch)]
        self.zeros = kv_lib.replicated(plan, torch.zeros(max_batch, dtype=torch.int32), ttnn.int32)
        self.ones = kv_lib.replicated(plan, torch.ones(max_batch, dtype=torch.int32), ttnn.int32)

    # ------------------------------------------------------------------ host
    def verify_host(self, tokens, positions, chain=None):
        """The host tensor a verify copies into ``verify_packed``: tokens, positions [B] (-1 an idle slot), and the
        chain row (reference.spec_chain.chain_row; None: idle), which the accept trace's chain step reads."""
        return kv_lib.replicated(self.plan, pack_verify(tokens, positions, chain), ttnn.int32, device=False)

    def mtp_host(self, tokens=None, plan=None):
        """The host tensor a draft copies into ``mtp_packed`` (reference.spec_inputs.pack_mtp; None: idle)."""
        return kv_lib.replicated(self.plan, pack_mtp(self.max_batch, tokens, plan), ttnn.int32, device=False)

    def reset(self):
        """Idle uploads (every position -1): a trace's warm-up run and capture then write no K / V, whatever the
        last served step left in the buffers."""
        idle = torch.full((self.max_batch,), -1, dtype=torch.int32)
        ttnn.copy_host_to_device_tensor(self.verify_host(torch.zeros_like(idle), idle), self.verify_packed)
        ttnn.copy_host_to_device_tensor(self.mtp_host(), self.mtp_packed)

    # ------------------------------------------------------------------ device (trace-capturable)
    def _row(self, packed, k, out=None):
        b = self.max_batch
        return ttnn.slice(packed, (k * b,), ((k + 1) * b,), output_tensor=out)

    @staticmethod
    def _into(t, out):
        """Copy ``t`` into the existing buffer ``out`` and free it: the eltwise binary ops refuse a preallocated
        output when both inputs are row-major (binary.cpp ``!(output_preallocated && input_a_rm && input_b_rm)``)."""
        ttnn.copy(t, out)
        t.deallocate(True)

    def _to_uint32(self, t, out):
        """int32 [B] -> the uint32 [1, B] buffer ``out``, bit for bit (the values are non-negative)."""
        row = ttnn.reshape(t, (1, self.max_batch))
        ttnn.bitcast(row, ttnn.uint32, output_tensor=out)
        row.deallocate(True)

    def derive_verify(self, inputs, rope):
        """Write the verify trace's inputs (TtMiMoModel._decode_inputs order: tokens, cur_pos, 4 cos / sin, token
        mask, owned positions, owner mask, the RoPE table index when ``rope``, then SPEC_SLOTS serial positions)
        from ``verify_packed``. Leaves the cos / sin alone (host inputs, or unused with the RoPE tables)."""
        b, packed = self.max_batch, self.verify_packed
        assert VERIFY_ROWS * b == packed.shape[-1]
        tokens, cur_pos, mask, pos_owned, owner_mask = inputs[0], inputs[1], inputs[6], inputs[7], inputs[8]
        serial = inputs[10:] if rope else inputs[9:]
        assert len(serial) == len(self.slot_caps)
        self._to_uint32(self._row(packed, 0), tokens)
        self._row(packed, 1, out=cur_pos)
        self._into(ttnn.minimum(cur_pos, self.own_cap), pos_owned)
        for cap, out in zip(self.slot_caps, serial):
            self._into(ttnn.minimum(cur_pos, cap), out)
        if rope:
            self._to_uint32(ttnn.maximum(cur_pos, self.zeros), inputs[9])
        # 1 on the active slots, 0 on the idle ones, row-major; tilized it is padded with zeros like the host mask.
        low = ttnn.minimum(cur_pos, self.zeros)
        active = ttnn.add(low, self.ones)
        low.deallocate(True)
        tiled = ttnn.to_layout(ttnn.reshape(active, (1, 1, 1, b)), ttnn.TILE_LAYOUT)
        active.deallocate(True)
        row = ttnn.typecast(tiled, ttnn.bfloat16)
        tiled.deallocate(True)
        col = ttnn.transpose(row, -2, -1)  # [1, 1, B, 1]
        row.deallocate(True)
        ttnn.copy(col, mask)
        ttnn.multiply(col, self.own_col, output_tensor=owner_mask)
        col.deallocate(True)

    def derive_mtp(self, tokens, fill, serial, step2, step3, rope_idx=None):
        """Write the draft trace's token and position inputs (TtMiMoMTP) from ``mtp_packed``; ``rope_idx`` the
        three RoPE table index buffers (uint32 [1, B]) with the device RoPE tables."""
        packed = self.mtp_packed
        assert MTP_ROWS * self.max_batch == packed.shape[-1] and len(serial) == len(self.slot_caps)
        self._to_uint32(self._row(packed, 0), tokens)
        self._row(packed, 1, out=fill)
        for cap, out in zip(self.slot_caps, serial):
            self._into(ttnn.minimum(fill, cap), out)
        self._row(packed, 2, out=step2)
        self._row(packed, 3, out=step3)
        for k, out in enumerate(rope_idx or []):
            self._to_uint32(self._row(packed, 4 + k), out)
