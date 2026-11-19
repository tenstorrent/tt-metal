# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The greedy acceptance of one MTP verify and the drafter's inputs, on the device (one-enqueue steps).

On a step whose verify answers ``argmax_ids`` (greedy rows, and device-sampled rows whose ids are draws),
MTPBlockEngine enqueues the verify trace, the sampler trace (a device-sampled step), the step trace of this module
(StepAccept.body) and the drafter trace, and reads the results with one wait. The step trace continues from the
verify's ids:

  1. the ids: the verify's argmax lanes, with each drawn user's row replaced by the sampler's draws;
  2. the plugin's greedy walk (accept_greedy_drafts): m[u] = the leading drafts that equal the id of the column before
     them, among the user's valid drafts, so the user commits ids[u, 0 .. m[u]] (m[u] + 1 tokens);
  3. the drafter pack (spec_step.drafter_pack) of that commit, and the reseed table (spec_step.reseed_table),
     written into the drafter trace's input buffers.

What the host knows before the verify comes from the acceptance rows of the verify step buffer (spec_step): valid
drafts, drawn rows, column-0 positions, RoPE base rows, and the rows the propose declines whatever they accept. So the
drafter runs on exactly the inputs the host would stage after reading the ids. Every value is an integer below 2**24
(token ids, positions, table rows, block ids).

The step trace is one single-core kernel (kernels/step_accept.cpp; its arithmetic is accept_words in
kernels/spec_step_words.hpp). ``graph`` states the same arithmetic as the float32 op graph the kernel replaced, over an
op set: TorchOps runs it in torch, and tests/spec_op_graphs.py keeps the ttnn op set, which the device tests run
beside the kernel. Float32 is exact on integers below 2**24, so the kernel's int32 words equal the graph's;
step_accept_reference states the result in the terms of the host path.
"""

from pathlib import Path

import torch

import ttnn
from models.demos.blackhole.qwen36.tt.spec_step import (
    STEP_DECLINED,
    STEP_DRAWN,
    STEP_KEEP,
    STEP_NUM_VALID,
    STEP_P0,
    STEP_ROPE_BASE,
    VERIFY_STEP_ROWS,
    VERIFY_TOKENS,
    drafter_pack,
    drafter_pack_rows,
    reseed_real_rows,
    reseed_table,
)


def greedy_counts(ids, tokens, num_valid):
    """The plugin's greedy walk (accept_greedy_drafts) over ``ids`` [B, T] and the verify tokens [B, T] (drafts in
    columns 1..K): accepted drafts m [B], the leading valid drafts equal to the id of the column before them."""
    ids = torch.as_tensor(ids).to(torch.int64)
    tokens = torch.as_tensor(tokens).to(torch.int64)
    B, T = ids.shape
    valid = torch.arange(T - 1).reshape(1, T - 1) < torch.as_tensor(num_valid).reshape(B, 1)
    match = (ids[:, : T - 1] == tokens[:, 1:]) & valid
    return torch.cumprod(match.to(torch.int64), dim=1).sum(dim=1)


def step_accept_reference(ids, tokens, step, pt, scratch_block, K, table_rows, draws=None):
    """What StepAccept.body writes, in the host path's terms. ``ids`` [B * T] (the verify's argmax ids, user-major),
    ``tokens`` [B * T] (the verify's tokens), ``step`` {STEP_* row: [B]} (the acceptance rows), ``pt`` [B, nb] (the
    drafter's table), ``draws`` [B * T] or None. Returns (ids [B, T] with the drawn rows' draws, m [B], drafter pack,
    reseed table)."""
    T = K + 1
    B = len(step[STEP_P0])
    ids = torch.as_tensor(ids).reshape(-1)[: B * T].to(torch.int64).reshape(B, T)
    if draws is not None:
        drawn = torch.as_tensor(step[STEP_DRAWN]).to(torch.bool).reshape(B, 1)
        ids = torch.where(drawn, torch.as_tensor(draws).reshape(-1)[: B * T].to(torch.int64).reshape(B, T), ids)
    tokens = torch.as_tensor(tokens).reshape(-1)[: B * T].reshape(B, T)
    m = greedy_counts(ids, tokens, step[STEP_NUM_VALID])
    dec, keep = step[STEP_DECLINED], step[STEP_KEEP]
    pack = drafter_pack(ids, m, step[STEP_P0], dec, keep, step[STEP_ROPE_BASE], K, table_rows)
    rs = reseed_table(pt, reseed_real_rows(m, dec, keep, T), T, scratch_block)
    return ids, m, pack, rs


def graph(o, B, K, W):
    """The step trace's arithmetic over the op set ``o`` (TorchOps, or the ttnn op set of tests/spec_op_graphs.py):
    returns (pack [2K + 5, W], reseed table [B * T, nb]) as float32."""
    T, R = K + 1, B * (K + 1)
    ids = o.lanes(o.ids, B, T)  # [B, T]
    aux = o.transpose(o.rows(o.step, STEP_NUM_VALID, VERIFY_STEP_ROWS, W))  # [W, 6], column i = row STEP_NUM_VALID + i

    def column(row):
        return o.slice(aux, (0, row - STEP_NUM_VALID), (B, row - STEP_NUM_VALID + 1))  # [B, 1]

    nv, drawn, p0, rb, dec, keep = (column(r) for r in range(STEP_NUM_VALID, VERIFY_STEP_ROWS))
    if o.draws is not None:
        ids = o.add(ids, o.mul(drawn, o.sub(o.lanes(o.draws, B, T), ids)))
    tokens = o.row_block(o.step, VERIFY_TOKENS, B, T)
    # the greedy walk: leading valid drafts equal to the id of the column before them
    match = o.mul(o.eq(o.slice(ids, (0, 0), (B, K)), o.slice(tokens, (0, 1), (B, T))), o.lt(o.iota_k, nv))
    m = o.sum(o.cumprod(match))  # [B, 1]
    # spec_step.drafter_pack of ids[u, 0 .. m[u]]
    drafting = o.rsub1(dec)
    pending = o.mul(o.sum(o.mul(o.eq(o.iota_t, m), ids)), drafting)
    p = o.mul(o.add(p0, m), drafting)
    real = o.lt(o.iota_t, o.add(m, o.mul(dec, keep)))  # [B, T]
    rtok = o.mul(ids, real)
    rpos = o.mul(o.add(p0, o.iota_t), real)
    steps = o.add(p, o.iota_k)  # [B, K]
    anchor = o.add(o.user_rows, m)  # u * T + m
    rope_reseed = o.minimum(o.add(rb, o.iota_t), o.top)
    rope_steps = o.minimum(o.add(o.add(rb, m), o.iota_k), o.top)
    pack = o.concat(
        [
            o.pad_w(o.transpose(o.concat([pending, steps], 1)), B, W),  # pending, then step k's positions
            o.pad_w(o.reshape(o.concat([rtok, rpos], 0), (2, R)), R, W),
            o.pad_w(o.transpose(anchor), B, W),
            o.pad_w(o.reshape(rope_reseed, (1, R)), R, W),
            o.pad_w(o.transpose(rope_steps), B, W),  # step k's table rows
        ],
        0,
    )
    rs = o.add(o.mul(o.sub(o.table(o.pt_rows), o.scratch), o.reshape(real, (R, 1))), o.scratch)
    return pack, rs


class TorchOps:
    """``graph``'s op set in torch over host tensors: ``ids`` / ``draws`` [1, 1, 1, >= B*T], ``step`` [rows, W],
    ``pt_rows`` [B*T, nb]; the constants as the op graph staged them."""

    def __init__(self, ids, step, pt_rows, B, K, scratch_block, table_rows, draws=None):
        T = K + 1
        f = lambda t: torch.as_tensor(t).to(torch.float32)
        self.ids, self.step, self.pt_rows, self.draws = f(ids), f(step), f(pt_rows), None if draws is None else f(draws)
        self.iota_t = f(torch.arange(T).reshape(1, T))
        self.iota_k = f(torch.arange(K).reshape(1, K))
        self.user_rows = f((torch.arange(B) * T).reshape(B, 1))
        self.top = f(torch.tensor([[int(table_rows) - 1]]))
        self.scratch = f(torch.tensor([[int(scratch_block)]]))

    def lanes(self, t, B, T):
        return t.reshape(-1)[: B * T].reshape(B, T)

    def row_block(self, t, row, B, T):
        return t[row, : B * T].reshape(B, T)

    def rows(self, t, start, stop, n):
        return t[start:stop, :n]

    def table(self, t):
        return t

    transpose = staticmethod(lambda x: x.transpose(0, 1))
    slice = staticmethod(lambda x, s, e: x[s[0] : e[0], s[1] : e[1]])
    reshape = staticmethod(lambda x, shape: x.reshape(shape))
    add = staticmethod(lambda a, b: a + b)
    sub = staticmethod(lambda a, b: a - b)
    mul = staticmethod(lambda a, b: a * b)
    minimum = staticmethod(lambda a, b: torch.minimum(a, b))
    rsub1 = staticmethod(lambda x: 1.0 - x)
    eq = staticmethod(lambda a, b: (a == b).to(torch.float32))
    lt = staticmethod(lambda a, b: (a < b).to(torch.float32))
    cumprod = staticmethod(lambda x: torch.cumprod(x, dim=1))
    sum = staticmethod(lambda x: x.sum(dim=1, keepdim=True))
    concat = staticmethod(lambda xs, dim: torch.cat(xs, dim=dim))

    @staticmethod
    def pad_w(x, n, W):
        return torch.nn.functional.pad(x, (0, W - n))


_KERNELS = Path(__file__).resolve().parent / "kernels"
STEP_ACCEPT_KERNEL = str(_KERNELS / "step_accept.cpp")
IDS_LANES = 32  # words of the verify's ids row (the fused argmax lanes): B * T verify rows at most
SLOT = 64  # spec_step_words.hpp SLOT: every L1 slot of the kernels starts on it
RS_CHUNK_MAX = 1024  # bytes of one reseed table column chunk; L1 holds B * T + 1 chunks
STEP_USER_ROWS = VERIFY_STEP_ROWS - STEP_NUM_VALID
KERNEL_CB = 0


def slot_bytes(words):
    """spec_step_words.hpp slot_bytes: ``words`` uint32 words rounded up to whole L1 slots."""
    return -(-4 * int(words) // SLOT) * SLOT


def rs_chunk_bytes(nb):
    """Bytes of the reseed table column chunk kernels/step_accept.cpp moves per row: the whole slotted row up to
    RS_CHUNK_MAX."""
    return min(slot_bytes(nb), RS_CHUNK_MAX)


def step_kernel_l1_bytes(B, K, pack_words, chunk):
    """spec_step_words.hpp AcceptLayout(B, K, pack_words, chunk).bytes."""
    R = B * (K + 1)
    rows = 3 * slot_bytes(R) + STEP_USER_ROWS * slot_bytes(B) + drafter_pack_rows(K) * slot_bytes(pack_words)
    return rows + slot_bytes(R) + chunk + R * chunk


def accessor_args(t):
    return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())


def check_same_accessor(live, standin, what):
    """A live tensor bound after the warm call must keep the stand-in's accessor (layout, memory, aligned page size):
    the kernel's compile-time arguments carry it, so another one would compile a new program inside a trace capture."""
    old, new = accessor_args(standin), accessor_args(live)
    if old != new:
        raise ValueError(f"{what}: accessor {new} differs from the stand-in's {old} the program was compiled on")


def one_core_program(source, compile_time_args, runtime_args, cb_bytes):
    """A single-core (core (0, 0)) data-movement program with one L1 scratch CB of ``cb_bytes``. Buffer addresses
    travel as common runtime arguments, which the program cache does not hash."""
    core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    cb = ttnn.CBDescriptor(
        total_size=cb_bytes,
        core_ranges=core,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=KERNEL_CB, data_format=ttnn.uint32, page_size=cb_bytes)
        ],
    )
    kernel = ttnn.KernelDescriptor(
        kernel_source=source,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core,
        compile_time_args=compile_time_args,
        common_runtime_args=runtime_args,
        config=ttnn.ReaderConfigDescriptor(),
    )
    return ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[cb])


class StepAccept:
    """The step trace of a one-enqueue MTP step over B users and K drafts: one kernels/step_accept.cpp program. Inputs:
    ``ids`` (the verify's argmax lanes [1, 1, 1, 32] uint32; a stand-in until bind_ids), ``draws`` (the sampler's
    [1, 1, 1, B*T] draws, or None without a device sampler), ``step`` (the verify step buffer) and ``pt_rows``
    ([B*T, nb] int32: row u*T + i holds the user's drafter table, staged when it changes). Outputs, written in place:
    ``pack`` (the drafter pack) and ``rs_pt`` (the reseed table)."""

    def __init__(self, mesh, B, K, nb, table_rows, scratch_block, step, pack, rs_pt, draws=None):
        T, R = K + 1, B * (K + 1)
        if R > IDS_LANES:
            raise ValueError(f"{B} users x {T} columns exceed the {IDS_LANES} verify id lanes")
        self.mesh, self.B, self.K, self.nb, self.W = mesh, int(B), int(K), int(nb), int(pack.shape[-1])
        self.top, self.scratch = int(table_rows) - 1, int(scratch_block)
        self.step, self.pack, self.rs_pt, self.draws = step, pack, rs_pt, draws
        self.pt_rows = self._device(torch.zeros(R, int(nb), dtype=torch.int32), ttnn.int32)
        self.ids = self.standin = self._device(torch.zeros(1, 1, 1, IDS_LANES, dtype=torch.int32), ttnn.uint32)
        self.chunk = rs_chunk_bytes(nb)
        self.cb_bytes = step_kernel_l1_bytes(self.B, self.K, self.W, self.chunk) + SLOT
        self.buffers = [self.pt_rows]

    def _device(self, t, dtype):
        return ttnn.from_torch(
            t,
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )

    def bind_ids(self, ids):
        """Read the verify trace's ids output (it exists only after the verify capture) instead of the stand-in the
        program was compiled on; both are [1, 1, 1, 32] uint32 ROW_MAJOR in DRAM."""
        check_same_accessor(ids, self.standin, "the verify ids")
        self.ids = ids

    def stage_pt_rows(self, pt_rows):
        """Host -> device copy of the [B*T, nb] table rows."""
        h = ttnn.from_torch(
            torch.as_tensor(pt_rows).to(torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        ttnn.copy_host_to_device_tensor(h, self.pt_rows)

    def program(self):
        """The kernel's program over the bound tensors; without a device sampler the ids stand in for the draws."""
        draws = self.ids if self.draws is None else self.draws
        tensors = [self.ids, draws, self.step, self.pt_rows, self.pack, self.rs_pt]
        cta = [self.B, self.K, self.W, self.nb, self.top, self.scratch, int(self.draws is not None), self.chunk]
        cta += [KERNEL_CB, self.cb_bytes]
        for t in tensors:
            cta += accessor_args(t)
        return one_core_program(STEP_ACCEPT_KERNEL, cta, [t.buffer_address() for t in tensors], self.cb_bytes)

    def body(self):
        """Trace-capturable: writes ``pack`` and ``rs_pt`` in place."""
        io = [t for t in (self.ids, self.draws, self.step, self.pt_rows, self.pack, self.rs_pt) if t is not None]
        ttnn.generic_op(io, self.program())

    def release(self):
        for t in (*self.buffers, self.standin):
            ttnn.deallocate(t)
        self.buffers, self.standin = [], None
