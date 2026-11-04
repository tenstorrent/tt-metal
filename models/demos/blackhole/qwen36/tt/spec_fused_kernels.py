# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""QWEN36_SPEC_CHAIN_KERNEL / QWEN36_SPEC_ACCEPT_KERNEL (default 0): the chain trace body and the greedy accept trace body
as ONE generic_op kernel each.

What they replace. Every production spec step replays four traces: [chain, verify, accept, drafter]. Two of them are pure
integer glue on a few hundred words:
  * chain (spec_chain.ChainDevice.body, stage S3c): 126 ttnn ops (83 glue + 42 eltwise) that build the next verify's
    token block, kv positions / rope rows / GDN ctrl page and the next fused aux from the previous accept's result words,
    the drafter ids and the host overlay; 0.596 ms on the 32K / 8-user profile (decode_roofline.md, section 4);
  * accept (spec_fused.FusedGreedyDevice.body, stage S1): 115 ttnn ops (69 glue + 44 eltwise) for the greedy walk, the
    drafter input pack, the reseed page table and the result words; 0.548 ms.
Each op is 3-5 us of launch and tile glue for data that fits in a few hundred bytes.

What runs instead (same input and output buffers, same words in every output page, zero padding included):
  1. kernels/spec_chain_fused.cpp, one core: reads result rows 0-1, the aux p0 row, the 11 overlay rows and the K draft
     id pages, writes the token page, the 3 vaux pages and the 7 next-aux pages;
  2. kernels/spec_accept_fused.cpp, ``spec_accept_kernel_cores()`` cores (default 32): every core reads ids / tokens /
     aux and runs the walk; core 0 writes the pack and result pages; the reseed page-table rows [R, nb] (512 KB at the
     serving width nb = 4097) are cut into <= 1 KB column chunks spread over the cores.
Exactness. The op graphs compute in fp32 TILE on integers below 2^24 (token ids, positions, block ids, small flags), where
fp32 is exact, so the kernels' int32 arithmetic gives the same words. The only other rule is ttnn.typecast fp32 -> uint32
turning a negative into 0 (reachable only from out-of-contract words: the all-zero warm-up inputs with rope_rows = 0);
the kernels do the same. tests/test_spec_fused_kernels.py checks kernel == op graph == host reference word for word.
The chain kernel serves both chain variants (S1 / S2 result buffer).

QWEN36_SPEC_ACCEPT_KERNEL_SAMPLED (default 0): the SAMPLED accept (spec_fused_sampled.FusedSampledDevice.body, stage S2:
union top-M, temperature / top-k / top-p softmax, the lossless point-mass walk with the runner-drawn uniforms, the S1
drafter tail; ~115 ttnn ops, ~0.5 ms) as kernels/spec_accept_sampled_fused.cpp on R cores (one engine row per core;
the user's first row core walks it, core 0 writes pack / result, every core its reseed page-table row). Its float32
math follows the reference's operation order with correctly rounded soft-float calls, an integer exp and exact sums
(tt/spec_sampled_kernel_math.py is its bit-exact host model), so it differs from fused_sampled_reference only by float32
rounding: a decision flips only within ~1e-7 of its threshold (accept, inverse-CDF edge, top-p tail edge), as the op
graph's device float math does.

Adaptive K (QWEN36_SPEC_ADAPTIVE_K): the chain kernel lays the GDN ctrl page out at B users. A ChainDevice of the wide
family (state width Bs > B) needs [parity, mi[B], 0 x (Bs-B), ((mi*Bs+u)*Nv+h)], so ``ChainKernel.supports`` refuses it
(the op graph runs, which knows Bs); every kernel here refuses a device whose state width differs from its B.
Flag off: no kernel object exists and ChainDevice / FusedGreedyDevice / FusedSampledDevice run their op graphs exactly as
before.
"""

from pathlib import Path

from loguru import logger

import ttnn

_KERNELS = Path(__file__).resolve().parent / "kernels"
CHAIN_KERNEL = str(_KERNELS / "spec_chain_fused.cpp")
ACCEPT_KERNEL = str(_KERNELS / "spec_accept_fused.cpp")
SAMPLED_KERNEL = str(_KERNELS / "spec_accept_sampled_fused.cpp")

CB_L1 = 0
ACCEPT_CHUNK_MAX = 1024  # bytes of one rs_pt column chunk (L1 per accept core ~ (R + 1) x chunk)
FUSED_AUX_ROWS, OVR_ROWS = 7, 11
SAMPLED_RS_CHUNK_MAX = 16448  # bytes of one pt / rs_pt row piece of the sampled kernel (a 4112-word row in one read)
SAMPLED_SCR_MAX = 1024  # bytes of its scratch-word piece
SAMPLED_SEMS = 3  # row records -> leader, real rows -> rows, user records -> core 0
SAMPLED_PARAM_ROWS = 4  # spec_fused_sampled.SAMP_PARAM_ROWS


def _al64(x):
    return (int(x) + 63) // 64 * 64


def state_width_reason(dev):
    """None if ``dev`` (a ChainDevice / FusedGreedyDevice / FusedSampledDevice) lays its per-user words out at its own B
    users, else why the kernels must not take it: an adaptive-K wide-family ChainDevice carries the GDN state width
    ``Bs`` > B (spec_chain.chain_reference ``state_B``), and the chain kernel would write the ring-block indices of a
    B-wide state into the GDN ctrl page, silently corrupting the recurrent state of the users it does not own."""
    B = int(dev.B)
    Bs = int(getattr(dev, "Bs", B))
    if Bs != B:
        return f"GDN state width Bs={Bs} != B={B} (adaptive-K wide family): the kernel lays its pages out at B"
    return None


def sampled_plan(R, nb, grid_x):
    """(nx, ny, rs_chunk, scr_bytes) of the sampled kernel: R row cores, row-major on an nx-wide grid; the pt / rs_pt
    row in pieces of <= SAMPLED_RS_CHUNK_MAX bytes (64 B multiples), the scratch piece <= SAMPLED_SCR_MAX bytes."""
    nx = max(1, min(int(R), int(grid_x)))
    ny = -(-int(R) // nx)
    row_bytes = 4 * int(nb)
    return nx, ny, min(_al64(row_bytes), SAMPLED_RS_CHUNK_MAX), min(_al64(row_bytes), SAMPLED_SCR_MAX)


def sampled_l1_bytes(B, K, W, M, TP, rs_chunk, scr_bytes):
    """Bytes of the sampled kernel's L1 layout (kernels/spec_accept_sampled_fused.cpp, L1_BYTES), before the 64 B base
    alignment."""
    T = K + 1
    R = B * T
    row, mw, nt = _al64(4 * W), _al64(4 * M), -(-(TP * M) // 32)
    head = _al64(4 * R) + 128 + (FUSED_AUX_ROWS + 5) * row + 2 * nt * 128 + 5 * mw + _al64(8 * TP)
    recs = _al64(16 * T) + 64 + _al64(16 * B)
    return head + recs + (2 * K + 5 + 3) * row + scr_bytes + rs_chunk


def _accessor(t):
    return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())


def _rm(t, dtype, rows=None, width=None):
    """None if ``t`` is a ROW_MAJOR ``dtype`` tensor of ``rows`` x ``width`` (last two dims; None = any), else why not."""
    shape = tuple(int(d) for d in t.shape)
    if t.layout != ttnn.ROW_MAJOR_LAYOUT:
        return f"{shape} not ROW_MAJOR"
    if t.dtype != dtype:
        return f"{shape} dtype {t.dtype} != {dtype}"
    if t.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        return f"{shape} not interleaved"
    if rows is not None and (len(shape) < 2 or shape[-2] != rows):
        return f"{shape}: need {rows} rows"
    if width is not None and shape[-1] != width:
        return f"{shape}: need width {width}"
    return None


def _cb(core_ranges, size):
    return ttnn.CBDescriptor(
        total_size=size,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=CB_L1, data_format=ttnn.uint32, page_size=size)],
    )


class ChainKernel:
    """The chain body of one ChainDevice ``cd`` as one single-core program (core (0, 0)). The descriptor is rebuilt per call
    (buffer addresses are common runtime args, not program-hash inputs), so ``cd.bind_draft_ins`` needs nothing here."""

    _logged = set()

    def __init__(self, cd):
        self.cd = cd
        self.core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])

    @staticmethod
    def supports(cd):
        """None if the chain device's buffers fit the kernel's contract, else the reason."""
        B, K, Nv, R = cd.B, cd.K, cd.Nv, cd.R
        why = state_width_reason(cd)
        if why is not None:
            return why
        if R > 32 or K < 1:
            return f"B={B} K={K}: more than 32 rows"
        checks = (
            ("result_in", cd.result_in, ttnn.int32, None, cd.W_res),
            ("aux_prev_in", cd.aux_prev_in, ttnn.int32, FUSED_AUX_ROWS, cd.W_a),
            ("ovr_in", cd.ovr_in, ttnn.int32, OVR_ROWS, cd.W_o),
            ("tok_out", cd.tok_out, ttnn.uint32, 1, R),
            ("vaux_out", cd.vaux_out, ttnn.uint32, 3, cd.W_v),
        )
        for name, t, dt, rows, width in checks:
            why = _rm(t, dt, rows, width)
            if why is not None:
                return f"{name}: {why}"
        if int(cd.result_in.shape[0]) < 2:
            return "result_in has fewer than 2 rows"
        if 1 + B + B * Nv > cd.W_v or cd.W_a > cd.W_o or R > cd.W_o:
            return "widths"
        return ChainKernel._draft_reason(cd.draft_ins, B)

    @staticmethod
    def _draft_reason(draft_ins, B):
        if not draft_ins:
            return "no draft inputs"
        first = None
        for i, d in enumerate(draft_ins):
            why = _rm(d, ttnn.uint32, B, 1)
            if why is not None:
                return f"draft_ins[{i}]: {why}"
            acc = _accessor(d)
            if first is None:
                first = acc
            elif acc != first:
                return f"draft_ins[{i}]: accessor {acc} != draft_ins[0] {first}"
        page = int(draft_ins[0].buffer_aligned_page_size())
        if not 4 <= page <= 64:
            return f"draft id page {page} B"
        return None

    def check_rebind(self, draft_ins):
        """AssertionError (the engine then turns chain mode off) unless the new draft id tensors keep the kernel's contract
        and the stand-ins' accessor layout (a different layout would be a program-cache miss inside the trace capture)."""
        why = self._draft_reason(draft_ins, self.cd.B)
        assert why is None, f"QWEN36_SPEC_CHAIN_KERNEL: {why}"
        old, new = _accessor(self.cd.draft_ins[0]), _accessor(draft_ins[0])
        assert old == new, f"QWEN36_SPEC_CHAIN_KERNEL: live draft ids accessor {new} != stand-in {old}"

    def descriptor(self):
        cd = self.cd
        why = self._draft_reason(cd.draft_ins, cd.B)
        assert why is None, f"QWEN36_SPEC_CHAIN_KERNEL: {why}"
        B, K, Nv, R = cd.B, cd.K, cd.Nv, cd.R
        W_res = int(cd.result_in.shape[-1])
        W_a, W_o, W_v = cd.W_a, cd.W_o, cd.W_v
        page = int(cd.draft_ins[0].buffer_aligned_page_size())
        # kernel L1 layout (spec_chain_fused.cpp): 2 result rows, 1 aux row, 11 overlay rows, K*B 64 B draft slots,
        # the token row, 3 vaux rows, 7 next-aux rows; + 64 B for the base alignment
        l1 = 2 * _al64(4 * W_res) + _al64(4 * W_a) + OVR_ROWS * _al64(4 * W_o) + K * B * 64
        l1 += _al64(4 * R) + 3 * _al64(4 * W_v) + FUSED_AUX_ROWS * _al64(4 * W_a)
        cb_bytes = l1 + 64
        cta = [B, K, Nv, W_res, W_a, W_o, W_v, page, CB_L1, cb_bytes]
        for t in (cd.result_in, cd.aux_prev_in, cd.ovr_in, cd.tok_out, cd.vaux_out, cd.draft_ins[0]):
            cta += _accessor(t)
        common = [
            cd.result_in.buffer_address(),
            cd.aux_prev_in.buffer_address(),
            cd.ovr_in.buffer_address(),
            cd.tok_out.buffer_address(),
            cd.vaux_out.buffer_address(),
        ] + [d.buffer_address() for d in cd.draft_ins]
        kernel = ttnn.KernelDescriptor(
            kernel_source=CHAIN_KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=self.core,
            compile_time_args=cta,
            common_runtime_args=common,
            config=ttnn.ReaderConfigDescriptor(),
        )
        return ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[_cb(self.core, cb_bytes)])

    def run(self):
        """Enqueue the kernel (trace-capturable: no allocation, no host read). Returns the next-aux tensor."""
        cd = self.cd
        io = [cd.result_in, *cd.draft_ins, cd.ovr_in, cd.tok_out, cd.vaux_out, cd.aux_prev_in]
        return ttnn.generic_op(io, self.descriptor())


class AcceptKernel:
    """The greedy accept body of one FusedGreedyDevice ``fd`` as one program on ``n_cores`` cores (logical grid from (0, 0),
    row-major, x < grid width). Core 0 writes pack / result; every core writes its rs_pt column chunks."""

    def __init__(self, fd, n_cores=32):
        self.fd = fd
        grid = fd.mesh.compute_with_storage_grid_size()
        n = max(1, min(int(n_cores), int(grid.x) * int(grid.y)))
        # rs_pt column chunks: <= ACCEPT_CHUNK_MAX bytes, 64 B multiples, no core without a chunk (cores loop over
        # chunks c, c + n, ... when a row needs more than n chunks)
        row_bytes = 4 * int(fd.nb)
        n = max(1, min(n, -(-row_bytes // 64)))
        self.chunk = min(ACCEPT_CHUNK_MAX, _al64(-(-row_bytes // n)))
        n = min(n, -(-row_bytes // self.chunk))
        self.nx = min(n, int(grid.x))
        self.ny = -(-n // self.nx)
        self.n_cores = n
        self.cores = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.nx - 1, self.ny - 1))]
        )

    @staticmethod
    def supports(fd):
        """None if the accept device's buffers fit the kernel's contract, else the reason (ids_in is checked per call)."""
        B, K, R, W, nb = fd.B, fd.K, fd.R, fd.W, fd.nb
        why = state_width_reason(fd)
        if why is not None:
            return why
        if R > 32 or K < 1 or W < max(R, B, 5):
            return f"B={B} K={K} W={W}"
        checks = (
            ("tok_in", fd.tok_in, ttnn.uint32, 1, R),
            ("aux_in", fd.aux_in, ttnn.int32, FUSED_AUX_ROWS, W),
            ("pt_rows_in", fd.pt_rows_in, ttnn.int32, R, nb),
            ("pack_out", fd.pack_out, ttnn.uint32, 2 * K + 5, W),
            ("rs_pt_out", fd.rs_pt_out, ttnn.int32, R, nb),
            ("result_out", fd.result_out, ttnn.int32, 2, W),
        )
        for name, t, dt, rows, width in checks:
            why = _rm(t, dt, rows, width)
            if why is not None:
                return f"{name}: {why}"
        return None

    def check_rebind(self, ids):
        """AssertionError (the engine then turns the fused accept off) unless the live ids tensor keeps the stand-in's
        contract and accessor layout (a different layout would be a program-cache miss inside the trace capture)."""
        why = _rm(ids, ttnn.uint32, 1, 32)
        assert why is None, f"QWEN36_SPEC_ACCEPT_KERNEL: ids_in {why}"
        old, new = _accessor(self.fd.ids_in), _accessor(ids)
        assert old == new, f"QWEN36_SPEC_ACCEPT_KERNEL: live ids accessor {new} != stand-in {old}"

    def descriptor(self):
        fd = self.fd
        why = _rm(fd.ids_in, ttnn.uint32, 1, 32)
        assert why is None, f"QWEN36_SPEC_ACCEPT_KERNEL: ids_in {why}"
        B, K, R, W, nb = fd.B, fd.K, fd.R, fd.W, fd.nb
        chunk = self.chunk
        # kernel L1 layout (spec_accept_fused.cpp): ids 128 B, the token row, 7 aux rows, 2K+5 pack rows, 2 result rows,
        # one scratch chunk, R pt chunks; + 64 B for the base alignment
        row = _al64(4 * W)
        l1 = 128 + _al64(4 * R) + (FUSED_AUX_ROWS + 2 * K + 5 + 2) * row + chunk + R * chunk
        cb_bytes = l1 + 64
        cta = [B, K, W, nb, self.nx, self.n_cores, chunk, CB_L1, cb_bytes]
        tensors = (fd.ids_in, fd.tok_in, fd.aux_in, fd.pt_rows_in, fd.pack_out, fd.rs_pt_out, fd.result_out)
        for t in tensors:
            cta += _accessor(t)
        kernel = ttnn.KernelDescriptor(
            kernel_source=ACCEPT_KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=self.cores,
            compile_time_args=cta,
            common_runtime_args=[t.buffer_address() for t in tensors],
            config=ttnn.ReaderConfigDescriptor(),
        )
        return ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[_cb(self.cores, cb_bytes)])

    def run(self):
        """Enqueue the kernel (trace-capturable). Returns result_out."""
        fd = self.fd
        io = [fd.ids_in, fd.tok_in, fd.aux_in, fd.pt_rows_in, fd.pack_out, fd.rs_pt_out, fd.result_out]
        return ttnn.generic_op(io, self.descriptor())


def _tile(t, dtype, shape):
    """None if ``t`` is an interleaved TILE ``dtype`` tensor of exactly ``shape``, else why not."""
    got = tuple(int(d) for d in t.shape)
    if t.layout != ttnn.TILE_LAYOUT:
        return f"{got} not TILE"
    if t.dtype != dtype:
        return f"{got} dtype {t.dtype} != {dtype}"
    if t.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        return f"{got} not interleaved"
    if got != tuple(shape):
        return f"{got} != {tuple(shape)}"
    return None


class SampledAcceptKernel:
    """The SAMPLED accept body of one FusedSampledDevice ``fd`` (stage S2) as one program on R row cores (logical grid
    from (0, 0), row-major, x < nx; core c owns engine row c). kernels/spec_accept_sampled_fused.cpp has the phases and
    the post protocol; tt/spec_sampled_kernel_math.py is its bit-exact host model. The descriptor is rebuilt per call
    (buffer addresses and the cores' NOC coordinates are common runtime args), so rebinding the live ids / union needs
    only the accessor check."""

    def __init__(self, fd):
        self.fd = fd
        grid = fd.mesh.compute_with_storage_grid_size()
        self.nx, self.ny, self.rs_chunk, self.scr_bytes = sampled_plan(fd.R, fd.nb, grid.x)
        self.cores = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.nx - 1, self.ny - 1))]
        )
        self.noc = []  # (x, y) NOC coordinates of row core c (the same on every chip of the mesh)
        for c in range(fd.R):
            v = fd.mesh.worker_core_from_logical_core(ttnn.CoreCoord(c % self.nx, c // self.nx))
            self.noc += [int(v.x), int(v.y)]
        self.cb_bytes = sampled_l1_bytes(fd.B, fd.K, fd.W, fd.M, fd.TP, self.rs_chunk, self.scr_bytes) + 64

    @staticmethod
    def supports(fd):
        """None if the sampled accept device's buffers fit the kernel's contract, else the reason (ids_in and the union are
        checked again when the live tensors are bound)."""
        B, K, R, W, nb, TP, M = fd.B, fd.K, fd.R, fd.W, fd.nb, fd.TP, fd.M
        why = state_width_reason(fd)
        if why is not None:
            return why
        if R > 32 or K < 1:
            return f"B={B} K={K}: more than 32 rows"
        if W < max(R, B, K + 2, 5):
            return f"W={W} below max(R, B, K + 2, 5)"
        if not 2 <= M <= 64 or TP * M > 1024 or TP > 32:
            return f"TP={TP} M={M}: need 2 <= M <= 64, TP <= 32 and TP * M <= 1024"
        grid = fd.mesh.compute_with_storage_grid_size()
        if R > int(grid.x) * int(grid.y):
            return f"{R} rows > {int(grid.x) * int(grid.y)} cores"
        TPM = TP * M
        for name, t, dt in (("union_vals_in", fd.union_vals_in, ttnn.bfloat16), ("union_ids_in", fd.union_ids_in, ttnn.uint32)):
            why = _tile(t, dt, (1, 1, 32, TPM))
            if why is not None:
                return f"{name}: {why}"
        checks = (
            ("tok_in", fd.tok_in, ttnn.uint32, 1, R),
            ("aux_in", fd.aux_in, ttnn.int32, FUSED_AUX_ROWS, W),
            ("samp_in", fd.samp_in, ttnn.float32, SAMPLED_PARAM_ROWS + B, W),
            ("pt_rows_in", fd.pt_rows_in, ttnn.int32, R, nb),
            ("pack_out", fd.pack_out, ttnn.uint32, 2 * K + 5, W),
            ("rs_pt_out", fd.rs_pt_out, ttnn.int32, R, nb),
            ("result_out", fd.result_out, ttnn.int32, 3, W),
        )
        for name, t, dt, rows, width in checks:
            why = _rm(t, dt, rows, width)
            if why is not None:
                return f"{name}: {why}"
        return None

    def check_rebind(self, ids):
        """AssertionError (the engine then turns the sampled fused accept off) unless the live ids keep the stand-in's
        contract and accessor layout (a different layout would be a program-cache miss inside the trace capture)."""
        why = _rm(ids, ttnn.uint32, 1, 32)
        assert why is None, f"QWEN36_SPEC_ACCEPT_KERNEL_SAMPLED: ids_in {why}"
        old, new = _accessor(self.fd.ids_in), _accessor(ids)
        assert old == new, f"QWEN36_SPEC_ACCEPT_KERNEL_SAMPLED: live ids accessor {new} != stand-in {old}"

    def check_rebind_union(self, vals, ids):
        """The same for the live top-k union (the verify topk trace outputs)."""
        TPM = self.fd.TP * self.fd.M
        for name, t, dt, cur in (
            ("union_vals_in", vals, ttnn.bfloat16, self.fd.union_vals_in),
            ("union_ids_in", ids, ttnn.uint32, self.fd.union_ids_in),
        ):
            why = _tile(t, dt, (1, 1, 32, TPM))
            assert why is None, f"QWEN36_SPEC_ACCEPT_KERNEL_SAMPLED: {name} {why}"
            old, new = _accessor(cur), _accessor(t)
            assert old == new, f"QWEN36_SPEC_ACCEPT_KERNEL_SAMPLED: live {name} accessor {new} != stand-in {old}"

    def _io(self):
        fd = self.fd
        return [
            fd.ids_in,
            fd.tok_in,
            fd.aux_in,
            fd.samp_in,
            fd.union_vals_in,
            fd.union_ids_in,
            fd.pt_rows_in,
            fd.pack_out,
            fd.rs_pt_out,
            fd.result_out,
        ]

    def descriptor(self):
        fd = self.fd
        why = _rm(fd.ids_in, ttnn.uint32, 1, 32)
        assert why is None, f"QWEN36_SPEC_ACCEPT_KERNEL_SAMPLED: ids_in {why}"
        tensors = self._io()
        cta = [fd.B, fd.K, fd.W, fd.nb, self.nx, fd.M, fd.TP, fd.per_shard, self.rs_chunk, self.scr_bytes]
        cta += [CB_L1, self.cb_bytes]
        for t in tensors:
            cta += _accessor(t)
        kernel = ttnn.KernelDescriptor(
            kernel_source=SAMPLED_KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=self.cores,
            compile_time_args=cta,
            common_runtime_args=[t.buffer_address() for t in tensors] + self.noc,
            config=ttnn.ReaderConfigDescriptor(),
        )
        sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=self.cores, initial_value=0) for i in range(SAMPLED_SEMS)]
        return ttnn.ProgramDescriptor(kernels=[kernel], semaphores=sems, cbs=[_cb(self.cores, self.cb_bytes)])

    def run(self):
        """Enqueue the kernel (trace-capturable: no allocation, no host read). Returns result_out."""
        return ttnn.generic_op(self._io(), self.descriptor())


def attach_chain_kernel(cd):
    """The ChainKernel for ``cd``, or None (logged once per reason) when its buffers do not fit: the op graph then runs."""
    why = ChainKernel.supports(cd)
    if why is not None:
        if why not in ChainKernel._logged:
            ChainKernel._logged.add(why)
            logger.warning(f"QWEN36_SPEC_CHAIN_KERNEL: chain op graph kept ({why})")
        return None
    return ChainKernel(cd)


def attach_accept_kernel(fd, n_cores):
    """The AcceptKernel for ``fd``, or None (logged) when its buffers do not fit: the op graph then runs."""
    why = AcceptKernel.supports(fd)
    if why is not None:
        logger.warning(f"QWEN36_SPEC_ACCEPT_KERNEL: accept op graph kept ({why})")
        return None
    return AcceptKernel(fd, n_cores)


def attach_sampled_accept_kernel(fd):
    """The SampledAcceptKernel for ``fd``, or None (logged) when its buffers do not fit: the sampled op graph then runs."""
    why = SampledAcceptKernel.supports(fd)
    if why is not None:
        logger.warning(f"QWEN36_SPEC_ACCEPT_KERNEL_SAMPLED: sampled accept op graph kept ({why})")
        return None
    k = SampledAcceptKernel(fd)
    logger.info(
        f"QWEN36_SPEC_ACCEPT_KERNEL_SAMPLED: sampled accept kernel on {fd.R} cores ({k.nx}x{k.ny}), "
        f"B={fd.B} K={fd.K} TP={fd.TP} M={fd.M} nb={fd.nb}, {k.cb_bytes} B L1 per core"
    )
    return k
