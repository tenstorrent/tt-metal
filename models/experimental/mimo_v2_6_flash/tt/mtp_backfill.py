# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The target-hidden history and the MTP ring backfill on one TP=8 replica (reference/mtp_backfill.py has the design,
the layouts and the host mirror of kernels/mtp_history.cpp).

MtpHistory keeps, per decode slot that may speculate, the target's final normed hidden of its last HISTORY_ROWS
positions (the MTP window and a verify pass) with their positions and input tokens: persistent buffers (made in
TtMiMoMTP.setup_cache, before any capture) written by one-core steps of kernels/mtp_history.cpp:

* ``stage`` + ``decode``: in the decode traces (TtMiMoModel.capture_decode_trace), a copy of the step's input tokens
  at its start, then the slots' final norm rows (untilized) at its end;
* ``verify``: in the verify trace, the pass's slots into their requests' rings (the chain row's history slots);
* ``prefill``: after each prefill chunk (eager) or in a traced prefill batch, the final norm of the chunk's rows and
  the host's write list (its last HISTORY_ROWS rows, the invalidations).

The backfill program (TtMiMoMTP's backfill trace, ``plan`` then ``fill``): the plan step writes the fill's inputs from
the backfill upload and the history page; one gather of the history rows, one RoPE gather, then per MTP layer the
layer's input over the BACKFILL_ROWS rows (enorm / hnorm, eh_proj, the block's input norm) and its K / V written into
the whole ring by one paged_fill_cache each (TtMiMoAttention.fill_kv_rows): no attention, MLP or LM head.

Cost: per decode step one 8-row copy and an untilize of the final norm's tile ([32, 4096] bf16) and one kernel
(~10-20 us against a ~13 ms step); per verify pass one kernel; per prefill chunk a final norm and untilize of its
rows and one kernel; per backfill (once per transition) 3 x (256-row eh_proj, qkv projection, RoPE and 2 fills), about
1 ms on device (to measure). Memory per chip: SPEC_SLOTS x HISTORY_ROWS x hidden bf16 rows (4.3 MB), a 4 KB page, the
backfill's inputs (~6 KB) and the 2 x 256-row cos / sin tails with compact RoPE tables.
"""

import torch

import ttnn

from ..reference.mtp_backfill import (
    BACKFILL_ROWS,
    BLOCK,
    HIST_WORDS,
    HISTORY_ROWS,
    MTP_WINDOW,
    TABLE_WORDS,
    backfill_row,
    pack_writes,
    prefill_writes,
)
from ..reference.spec_chain import CHAIN_HSLOTS, CHAIN_ROW
from ..reference.spec_decode import GROUP_SLOTS, SPEC_SLOTS
from .device_sampler import CB_SCRATCH, _cb, _kernel, _ranges

DECODE, VERIFY, LIST, PLAN = 0, 1, 2, 3  # the kernel's modes
LIST_ENTRIES = SPEC_SLOTS * HISTORY_ROWS  # the longest write list: a prefill batch's first chunks


def _align64(nbytes):
    return (nbytes + 63) // 64 * 64


class MtpHistory:
    def __init__(self, plan, max_batch, hidden_size, layers, rope_tails=None):
        """``layers``: the drafter's MTP layers (their tokens rows; with more than one the plan also writes the stash).
        ``rope_tails``: the model's compact RoPE table tails (None with whole rows), whose 256-row versions the
        backfill's cos / sin append. Allocates every buffer (persistent)."""
        self.plan, self.max_batch, self.hidden, self.layers = plan, max_batch, hidden_size, layers
        self.row_bytes = hidden_size * 2
        rep = plan.replicate()
        mesh = plan.mesh_device

        def tensor(values, dtype):
            return ttnn.from_torch(
                values,
                device=mesh,
                dtype=dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=rep,
            )

        def ints(values, dtype=ttnn.int32):
            return tensor(torch.tensor(values, dtype=torch.int32), dtype)

        n = SPEC_SLOTS * HISTORY_ROWS
        self.rows = tensor(torch.zeros(n, hidden_size), ttnn.bfloat16)  # hidden rows, row e at entry e
        self.page = ints([-1] * n + [0] * n)
        assert self.page.shape[-1] == HIST_WORDS
        self.tokens_in = ints([[0] * max_batch], ttnn.uint32)  # the decode step's input tokens
        self.upload = ints(backfill_row(max_batch))
        self.hidx = ints([[0] * BACKFILL_ROWS], ttnn.uint32)
        self.rope = ints([list(range(BACKFILL_ROWS))], ttnn.uint32)
        self.tokens = [ints([[0] * BACKFILL_ROWS], ttnn.uint32) for _ in range(layers)]
        self.table = ints([[0] * TABLE_WORDS])
        self.tails = None
        if rope_tails is not None:
            self.tails = [
                ttnn.from_torch(
                    torch.full((1, 1, BACKFILL_ROWS, t.shape[-1]), value),
                    device=mesh,
                    dtype=t.dtype,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=rep,
                )
                for t, value in zip(rope_tails, (1.0, 0.0))
            ]
        self.core = ttnn.CoreCoord(0, 0)

    # ------------------------------------------------------------------ device (trace-capturable)
    def _run(self, mode, args, io, extra=0):
        cores = [self.core]
        compile_args = [
            mode,
            self.max_batch,
            SPEC_SLOTS,
            HISTORY_ROWS,
            self.row_bytes,
            BACKFILL_ROWS,
            self.layers,
            BLOCK,
            CHAIN_ROW,
            CHAIN_HSLOTS,
            TABLE_WORDS,
            CB_SCRATCH,
            MTP_WINDOW,
            GROUP_SLOTS,
        ]
        scratch = _align64(HIST_WORDS * 4) + _align64(self.row_bytes) + extra
        kernel = _kernel("mtp_history.cpp", cores, _ranges(cores), compile_args, [args], ttnn.ReaderConfigDescriptor())
        prog = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[_cb(CB_SCRATCH, scratch, _ranges(cores))])
        ttnn.generic_op(io, prog)

    def _base(self):
        return [self.page.buffer_address(), self.rows.buffer_address()]

    def stage(self, tokens_tt):
        """At a decode step's start: keep its input tokens (the step's GreedyHead / sampler overwrites them)."""
        ttnn.copy(tokens_tt, self.tokens_in)

    def _untilized(self, hidden):
        assert hidden.shape[-2] == self.max_batch, "one source row per slot"
        return ttnn.to_layout(hidden, ttnn.ROW_MAJOR_LAYOUT)

    def decode(self, cur_pos, hidden):
        """At a decode step's end (before its positions advance): the slots' final norm rows ``hidden``."""
        src = self._untilized(hidden)
        args = self._base() + [src.buffer_address(), cur_pos.buffer_address(), self.tokens_in.buffer_address()]
        self._run(DECODE, args, [cur_pos, self.tokens_in, src, self.page, self.rows], 2 * _align64(self.max_batch * 4))
        src.deallocate(True)

    def verify(self, verify_packed, hidden):
        """After a verify step: its slots' final norm rows into their requests' rings (the chain row's history
        slots)."""
        src = self._untilized(hidden)
        args = self._base() + [src.buffer_address(), verify_packed.buffer_address()]
        extra = _align64((CHAIN_ROW + 1) * self.max_batch * 4)
        self._run(VERIFY, args, [verify_packed, src, self.page, self.rows], extra)
        src.deallocate(True)

    def prefill(self, x, final_norm, write_list, entries):
        """After a prefill pass: the final norm of its rows ``x`` [1, 1, rows, hidden] (``final_norm``: the model's,
        in its row-invariant prefill form) and the write list ``write_list`` (int32 [4 * entries] on device, source
        rows into ``x``'s rows: reference.mtp_backfill.prefill_writes)."""
        h = final_norm(x, prefill=True)
        src = ttnn.reshape(ttnn.to_layout(h, ttnn.ROW_MAJOR_LAYOUT), (x.shape[-2], self.hidden))
        h.deallocate(True)
        # The list's L1 is sized for the longest list (a batch of SPEC_SLOTS first chunks): one program for every
        # list, the entry count a runtime argument (a program keyed on it would compile while serving).
        assert entries <= LIST_ENTRIES, f"{entries} history writes > {LIST_ENTRIES}"
        args = self._base() + [src.buffer_address(), write_list.buffer_address(), entries]
        self._run(LIST, args, [write_list, src, self.page, self.rows], _align64(16 * LIST_ENTRIES))
        src.deallocate(True)

    def prefill_list(self, writes, entries=None, device=True):
        """The write list of ``writes`` (reference.mtp_backfill.prefill_writes), padded to ``entries``: a device
        tensor, or with ``device`` False the host tensor to copy into a persistent one."""
        entries = entries or max(1, len(writes))
        values = torch.tensor(pack_writes(writes, entries), dtype=torch.int32)
        return ttnn.from_torch(
            values,
            device=self.plan.mesh_device if device else None,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG if device else None,
            mesh_mapper=self.plan.replicate(),
        )

    @staticmethod
    def chunk_writes(slot, start, tokens, src_first, continues):
        return prefill_writes(slot, start, [int(t) for t in tokens], src_first, continues)

    def backfill_plan(self, layer_inputs=None):
        """The backfill's plan step: its inputs from the upload and the page; with every layer (``layer_inputs``, the
        draft's LayerInputs) also the stash page and hidden rows."""
        li = layer_inputs
        stash = [li.stash, li.stash_hidden] if li is not None else [self.upload, self.upload]
        args = [self.upload.buffer_address(), self.page.buffer_address(), self.rows.buffer_address()]
        args += [self.hidx.buffer_address(), self.rope.buffer_address(), self.table.buffer_address()]
        args += [t.buffer_address() for t in stash] + [t.buffer_address() for t in self.tokens]
        row = _align64(self.max_batch * 4)
        extra = 2 * row + (2 + self.layers) * _align64(BACKFILL_ROWS * 4) + _align64(TABLE_WORDS * 4)
        io = [self.upload, self.page, self.rows, *(stash if li is not None else []), self.hidx, self.rope, self.table]
        self._run(PLAN, args, io + self.tokens, extra)

    def gather(self, model):
        """The backfill rows' hidden [1, 1, BACKFILL_ROWS, hidden] (tile) and their cos / sin (the SWA theta's
        tables at the plan's RoPE indices)."""
        hidden = ttnn.embedding(self.hidx, self.rows, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        hidden = ttnn.reshape(hidden, (1, 1, BACKFILL_ROWS, self.hidden))
        rope = []
        for k, t in enumerate(model.rope_tables[:2]):
            row = ttnn.embedding(self.rope, t, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
            row = ttnn.reshape(row, (1, 1, BACKFILL_ROWS, t.shape[-1]))
            if self.tails is not None:
                whole = ttnn.concat([row, self.tails[k]], dim=-1)
                row.deallocate(True)
                row = whole
            rope.append(row)
        return hidden, rope

    def warm(self, layer_inputs=None):
        """Compile the kernel's modes before any trace capture: they write nothing on the idle inputs they find (no
        decode slot active, a zero chain row, an empty list, an idle upload: the scratch blocks' table)."""
        idle = ttnn.from_torch(
            torch.full((self.max_batch,), -1, dtype=torch.int32),
            device=self.plan.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.plan.replicate(),
        )
        zero_rows = ttnn.from_torch(
            torch.zeros(1, 1, self.max_batch, self.hidden),
            device=self.plan.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.plan.replicate(),
        )
        self.decode(idle, zero_rows)
        verify = ttnn.from_torch(
            torch.cat(
                (
                    torch.zeros(self.max_batch, dtype=torch.int32),
                    torch.full((self.max_batch,), -1),
                    torch.zeros(self.max_batch, dtype=torch.int32),
                )
            ).to(torch.int32),
            device=self.plan.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.plan.replicate(),
        )
        self.verify(verify, zero_rows)
        self.upload_backfill()  # a recapture: the last served backfill's upload would rewrite the stash
        self.backfill_plan(layer_inputs)
        for t in (idle, zero_rows, verify):
            t.deallocate(True)

    # ------------------------------------------------------------------ host
    def upload_backfill(self, slot=None, a=None, group=0, keep_stash=False):
        """Copy the backfill upload (the request in lane slot ``slot``, the draft's first row ``a``, into draft group
        ``group``'s ring and stash slots, or its ring alone with ``keep_stash``; None: idle)."""
        host = ttnn.from_torch(
            torch.tensor(backfill_row(self.max_batch, slot, a, group, keep_stash), dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=self.plan.replicate(),
        )
        ttnn.copy_host_to_device_tensor(host, self.upload)
