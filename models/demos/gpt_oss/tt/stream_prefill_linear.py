# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill matmul that reads the fused decode stream copies directly: out[M, N] = x[M, K] @ W + bias.

In the batch-1 fused model the DRAM-streaming decode copies of the dense projections are the only stored copies: QKV,
o_proj and router (fused_decode/stream.py stream_linear_layout) and the LM head (fused_decode/terminal.py, the same
layout per device). This op (ttnn.generic_op, kernels fused_decode/kernels/stream_prefill_{reader,compute,writer}.cpp)
multiplies a prefill activation by such a stream without copying any weight.

Stream layout ([rows, banks * 32] per device, DRAM width-sharded, one 32-wide shard per bank): bank b holds the slots
[b * C, (b + 1) * C) (C = ceil(N / 32 / banks)), each a column of kt = K / 32 + 1 tiles (K / 32 weight tiles, then the
bias tile), contiguous; slot s holds output column tile order[s] (identity unless an order is given, e.g.
qkv_column_order); slots past the N / 32 real columns are zero padding (always a suffix of a bank's slots).

Work split (plan_work): every core owns `cols` consecutive real slots of one bank (one contiguous byte range of that
bank, read in one piece) and a range of output row tiles. Per M block (MB row tiles) the core reads its x rows (K / 32
tiles per row, per-core reads through the tensor accessor: no multicast) plus one "ones" tile per row (columns
0..NBIAS-1 = 1: multiplied by the bias tile it adds the bias, as the decode activation's extra tile does), then for each
of its columns the MB x 1 output tiles summed over all kt K tiles in DST (FP32, matmul_block, at most 4 rows per DST
block). Weights resident: the core's columns are read once and kept in L1 for all M blocks (QKV / o_proj / router).
Otherwise (LM head) they stream through a two-group ring per M block. No K blocking: a whole x row (K <= 90 tiles here)
is in L1. The writer writes output tile (row, order[slot]) of an [.., M, out_width] interleaved tensor: the logical
column order (QKV back in Q, K, V order); columns [N, out_width) are written as zeros.

Bias: the stream's own bias tile (NBIAS residual terms of the BF16 bias in the weight format, fused_decode/stream.py
bias_tiles), so the prefill adds exactly the bias decode adds (sum of the terms, ~1e-4 relative of the 16-value group
maximum from the BF16 bias); the o_proj stream's bias is on TP device 0 only, as the stored prefill bias was.

The pure parts (StreamSpec, plan_work, reference_linear, vocab_resplit_pieces) do not import ttnn: tests/unit/
test_stream_prefill_linear.py runs them with torch on the real stream_linear_layout.
"""

import math
from pathlib import Path

TILE = 32
KERNEL_DIR = Path(__file__).parent / "fused_decode" / "kernels"
NBIAS = 4  # fused_decode/stream.py NBIAS (the bias terms of a stream column; asserted equal on device)
MAX_RT = 4  # output rows per DST block (FP32 DST, half sync: 4 tiles)
MAX_MB = 8  # row tiles per M block (L1 permitting)
L1_CAP = 900 * 1024  # circular-buffer bytes per core at most
L1_MARGIN = 16 * 1024  # kept free below the lowest L1 buffer
MAX_WRITER_COLS = 300  # the writer's runtime args hold the core's column list
# plan_work cost model (rough, Blackhole per device): DRAM read bandwidth, one core's read bandwidth, one 32x32x32
# HiFi2 tile multiply
DRAM_BW = 400e9
CORE_BW = 30e9
TILE_MM_S = 50e-9


class StreamSpec:
    """One stream_linear_layout weight: k x n over `banks` DRAM banks, order = the output column tile of each slot
    (length C * banks), None for the identity."""

    def __init__(self, k, n, banks, order=None):
        assert k % TILE == 0 and n % TILE == 0, f"stream weights are whole tiles: {k} x {n}"
        self.k, self.n, self.banks = k, n, banks
        self.kx = k // TILE
        self.kt = self.kx + 1
        self.n_tiles = n // TILE
        self.C = -(-self.n_tiles // banks)
        self.slots = self.C * banks
        self.rows = self.C * self.kt * TILE  # fused_decode/stream.py linear_stream_rows
        order = list(range(self.slots)) if order is None else [int(c) for c in order]
        assert sorted(order) == list(range(self.slots)), f"order is not a permutation of the {self.slots} slots"
        self.order = order
        self.real = []  # real (non-padding) slots per bank: a prefix of the bank's slots
        for b in range(banks):
            real = [order[b * self.C + j] < self.n_tiles for j in range(self.C)]
            cnt = sum(real)
            assert all(real[:cnt]), f"bank {b}: padding slots before real ones ({real})"
            self.real.append(cnt)

    def columns(self, bank, j0, cnt):
        """Output column tiles of bank `bank`'s slots [j0, j0 + cnt)."""
        return [self.order[bank * self.C + j0 + i] for i in range(cnt)]


class WorkItem:
    """One core's share: slots [j0, j0 + cols) of bank `bank`, output row tiles [m0, m0 + m_cnt), and the zero output
    columns [z0, z1) it writes."""

    def __init__(self, bank, j0, cols, m0, m_cnt, z0=0, z1=0):
        self.bank, self.j0, self.cols, self.m0, self.m_cnt, self.z0, self.z1 = bank, j0, cols, m0, m_cnt, z0, z1

    def __repr__(self):
        return f"WorkItem(b{self.bank} j{self.j0}+{self.cols} m{self.m0}+{self.m_cnt} z[{self.z0},{self.z1}))"


class WorkPlan:
    """plan_work's result: the items (one per core), the block geometry and the circular-buffer sizes."""

    def __init__(self, items, MB, G, resident, x_bufs, cost):
        self.items, self.MB, self.G, self.resident, self.x_bufs, self.cost = items, MB, G, resident, x_bufs, cost

    @property
    def w_pages(self):
        """Weight circular-buffer pages: one group (resident) or a two-group ring, kt tiles per column."""
        return self.G * (1 if self.resident else 2)

    def __repr__(self):
        return (
            f"WorkPlan({len(self.items)} cores, MB {self.MB}, G {self.G}, resident {self.resident}, "
            f"x_bufs {self.x_bufs}, cost {self.cost * 1e6:.1f} us)"
        )


def _geometry(chunk, rows, kt, x_tile, w_tile, out_tile, budget):
    """(G, MB, resident, x_bufs, cb bytes) for a core of <= chunk columns and <= rows row tiles within budget bytes of
    circular buffers, or None. Resident weights first (read once), else streamed weight groups of G <= 4 columns. The x
    blocks: one block holding all rows, else double-buffered (the reader fetches the next block while the compute
    works), else single-buffered; the largest M block within each (fewer weight passes when streamed)."""
    out_bytes = 2 * MAX_RT * out_tile
    xs = [(mb, 1) for mb in range(min(rows, MAX_MB), 0, -1) if mb >= rows]
    xs += [(mb, 2) for mb in range(min(rows, MAX_MB), 0, -1) if mb < rows]
    xs += [(mb, 1) for mb in range(min(rows, MAX_MB), 0, -1) if mb < rows]
    w = chunk * kt * w_tile
    for mb, bufs in xs:
        xb = bufs * mb * kt * x_tile
        if w + xb + out_bytes <= budget:
            return chunk, mb, True, bufs, w + xb + out_bytes
    for mb, bufs in xs:
        xb = bufs * mb * kt * x_tile
        for g in range(min(chunk, 4), 0, -1):
            wb = 2 * g * kt * w_tile
            if wb + xb + out_bytes <= budget:
                return g, mb, False, bufs, wb + xb + out_bytes
    return None


def plan_work(spec, m_tiles, n_cores, x_tile, w_tile, out_tile, budget, out_tiles=None):
    """The work split of one call: per bank its real slots in `ns` chunks of <= chunk columns (one core each) times
    `pm` ranges of output row tiles, minimizing a rough time estimate max(DRAM bytes / bandwidth, the busiest bank's
    weight bytes / its share, the busiest core's reads / a core's bandwidth, its tile multiplies): x is read once per
    column chunk (P_N times), the weights once per row range (P_M times; per M block when not resident). out_tiles
    (> spec.n_tiles): the extra zero output columns, split over the chunks of each row range."""
    assert m_tiles >= 1
    out_tiles = spec.n_tiles if out_tiles is None else out_tiles
    assert out_tiles >= spec.n_tiles
    active = [b for b in range(spec.banks) if spec.real[b]]
    R = max(spec.real)
    kt = spec.kt
    w_total = sum(spec.real) * kt * w_tile
    x_total = m_tiles * spec.kx * x_tile
    best = None
    for ns in range(1, R + 1):
        chunk = -(-R // ns)
        if ns > 1 and chunk == -(-R // (ns - 1)):
            continue  # the same chunk size as ns - 1
        pn = sum(-(-spec.real[b] // chunk) for b in active)
        if pn > n_cores:
            break
        seen = set()
        for pm in range(1, min(m_tiles, n_cores // pn) + 1):
            rows = -(-m_tiles // pm)
            pm_eff = -(-m_tiles // rows)
            if pm_eff in seen:
                continue
            seen.add(pm_eff)
            geo = _geometry(chunk, rows, kt, x_tile, w_tile, out_tile, budget)
            if geo is None:
                continue
            G, MB, resident, x_bufs, _ = geo
            passes = 1 if resident else -(-rows // MB)
            w_reads = pm_eff * w_total * passes
            bank_reads = pm_eff * R * kt * w_tile * passes
            t = max(
                (w_reads + pn * x_total) / DRAM_BW,
                bank_reads / (DRAM_BW / spec.banks),
                (rows * spec.kx * x_tile + passes * chunk * kt * w_tile) / CORE_BW,
                rows * chunk * kt * TILE_MM_S,
            )
            key = (t, pn * pm_eff)
            if best is None or key < best[0]:
                best = (key, chunk, rows, geo)
    assert best is not None, f"no work split fits {budget} bytes of L1 (K {spec.k}, N {spec.n}, {m_tiles} row tiles)"
    (t, _), chunk, rows, (G, MB, resident, x_bufs, _) = best
    items = []
    for m0 in range(0, m_tiles, rows):
        m_cnt = min(rows, m_tiles - m0)
        row_items = []
        for b in active:
            for j0 in range(0, spec.real[b], chunk):
                row_items.append(WorkItem(b, j0, min(chunk, spec.real[b] - j0), m0, m_cnt))
        extra = out_tiles - spec.n_tiles  # zero columns, split evenly over this row range's cores
        for i, it in enumerate(row_items):
            it.z0 = spec.n_tiles + extra * i // len(row_items)
            it.z1 = spec.n_tiles + extra * (i + 1) // len(row_items)
        items += row_items
    assert len(items) <= n_cores
    assert all(it.cols <= MAX_WRITER_COLS for it in items)
    return WorkPlan(items, MB, G, resident, x_bufs, t)


# ---- the kernels' loops in torch (tests/unit/test_stream_prefill_linear.py) ----


def compute_order(item, plan):
    """(local row, local column) of each output tile in the order the compute kernel packs them."""
    out = []
    for mb in range(0, item.m_cnt, plan.MB):
        rows = min(plan.MB, item.m_cnt - mb)
        for g in range(0, item.cols, plan.G):
            for c in range(g, min(g + plan.G, item.cols)):
                for r0 in range(0, rows, MAX_RT):
                    for r in range(r0, min(r0 + MAX_RT, rows)):
                        out.append((mb + r, c))
    return out


def writer_pages(item, plan, columns, out_tiles):
    """Output tile pages in the order the writer writes the compute kernel's tiles (stream_prefill_writer.cpp), after
    its zero columns."""
    pages = [(item.m0 + r) * out_tiles + z for r in range(item.m_cnt) for z in range(item.z0, item.z1)]
    for mb in range(0, item.m_cnt, plan.MB):
        rows = min(plan.MB, item.m_cnt - mb)
        for g in range(0, item.cols, plan.G):
            for c in range(g, min(g + plan.G, item.cols)):
                for r0 in range(0, rows, MAX_RT):
                    for r in range(r0, min(r0 + MAX_RT, rows)):
                        pages.append((item.m0 + mb + r) * out_tiles + columns[c])
    return pages


def ones_tile(torch):
    t = torch.zeros(TILE, TILE)
    t[:, :NBIAS] = 1.0
    return t


def reference_linear(x, layout, spec, plan, out_tiles=None):
    """What the cores compute, in torch: x [M_pad, K] (M_pad = row tiles * 32), layout = one device's stream
    [rows, banks * 32] -> [M_pad, out_tiles * 32] (float64 sums). Each core: its x rows + the ones tile per row, its
    bank's slot tiles at the reader's byte offsets, per M block / weight group / column / DST block of rows the sums
    over the kt K tiles, written to (row, order[slot]); then the zero columns."""
    import torch

    out_tiles = spec.n_tiles if out_tiles is None else out_tiles
    M = x.shape[0]
    assert M % TILE == 0 and x.shape[1] == spec.k
    out = torch.full((M, out_tiles * TILE), float("nan"), dtype=torch.float64)
    ones = ones_tile(torch).double()
    xd = x.double()
    for it in plan.items:
        shard = layout[:, it.bank * TILE : (it.bank + 1) * TILE].double()
        columns = spec.columns(it.bank, it.j0, it.cols)
        for r in range(it.m_cnt):
            for z in range(it.z0, it.z1):
                row = it.m0 + r
                out[row * TILE : (row + 1) * TILE, z * TILE : (z + 1) * TILE] = 0

        # the weight group(s) as the reader lays them out in cb_w: tile (c, k) at c * kt + k
        def w_tile(c, k):
            t = (it.j0 + c) * spec.kt + k
            return shard[t * TILE : (t + 1) * TILE]

        for mb in range(0, it.m_cnt, plan.MB):
            rows = min(plan.MB, it.m_cnt - mb)
            # cb_x: row r's kx x tiles, then the ones tile
            block = []
            for r in range(rows):
                row = it.m0 + mb + r
                xr = xd[row * TILE : (row + 1) * TILE]
                block.append([xr[:, k * TILE : (k + 1) * TILE] for k in range(spec.kx)] + [ones])
            for g in range(0, it.cols, plan.G):
                for c in range(g, min(g + plan.G, it.cols)):
                    for r0 in range(0, rows, MAX_RT):
                        for r in range(r0, min(r0 + MAX_RT, rows)):
                            acc = torch.zeros(TILE, TILE, dtype=torch.float64)
                            for k in range(spec.kt):
                                acc += block[r][k] @ w_tile(c, k)
                            row, col = it.m0 + mb + r, columns[c]
                            out[row * TILE : (row + 1) * TILE, col * TILE : (col + 1) * TILE] = acc
    return out


def unpack_stream_weight(layout, spec):
    """One device's stream [rows, banks * 32] (host) -> the [K, N] weight it holds (the bias tiles left out)."""
    import torch

    w = torch.zeros(spec.k, spec.n, dtype=layout.dtype)
    for b in range(spec.banks):
        shard = layout[:, b * TILE : (b + 1) * TILE]
        for j, col in enumerate(spec.columns(b, 0, spec.real[b])):
            t0 = (j * spec.kt) * TILE
            w[:, col * TILE : (col + 1) * TILE] = shard[t0 : t0 + spec.k]
    return w


def vocab_resplit_pieces(tp, vd, per_device):
    """Column ranges of the all-gathered decode-split logits ([.., tp * per_device]: device d's block [d * per_device,
    (d + 1) * per_device) holds vocab ids d * vd + j for j < vd and zeros after) whose concatenation is the prefill
    split's [.., tp * per_device] logits (vocab id v at column v, zeros after the tp * vd ids): every device's vd real
    columns in device order, then every device's zero tail."""
    assert vd <= per_device and vd % TILE == 0 and per_device % TILE == 0, (vd, per_device)
    real = [(d * per_device, d * per_device + vd) for d in range(tp)]
    tails = [(d * per_device + vd, (d + 1) * per_device) for d in range(tp) if vd < per_device]
    return real + tails


# ---- device ----


_ONES = {}


def _ones_tensor(mesh_device, dtype):
    """The [32, 32] bias-multiplier tile (ones in columns 0..NBIAS-1) in dtype, DRAM, replicated; cached per mesh."""
    import torch

    import ttnn

    key = (id(mesh_device), str(dtype))
    cached = _ONES.get(key)
    if cached is None or cached[0] is not mesh_device:  # (the entry holds the mesh: its id is not reused meanwhile)
        t = ttnn.from_torch(
            ones_tile(torch).reshape(1, 1, TILE, TILE),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        cached = _ONES[key] = (mesh_device, t)
    return cached[1]


def _l1_budget(mesh_device):
    """Circular-buffer bytes per core: below the lowest L1 buffer now allocated (a program's circular buffers must end
    below it), at most L1_CAP."""
    base = mesh_device.l1_allocator_base()
    low = mesh_device.lowest_occupied_compute_l1_address()
    if not low:
        return L1_CAP
    return max(0, min(L1_CAP, low - base - L1_MARGIN))


def _core_ranges(ttnn, n, gx):
    """The first n cores of the grid, row-major (core i = (i % gx, i // gx)), as full rows + one partial row."""
    full, rest = divmod(n, gx)
    ranges = []
    if full:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, full - 1)))
    if rest:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full), ttnn.CoreCoord(rest - 1, full)))
    return ttnn.CoreRangeSet(ranges)


class StreamPrefillLinear:
    """out = x @ W + b for a fused decode stream W (module doc). One per weight shape / order; the plan and the
    kernel / circular-buffer descriptors are built once per (row tiles, dtypes, L1 budget, widths, memory configs)."""

    def __init__(self, mesh_device, k, n, order=None, math_fidelity="HiFi2"):
        self.mesh_device = mesh_device
        self.spec = StreamSpec(k, n, mesh_device.dram_grid_size().x, order)
        self.math_fidelity = math_fidelity
        self._programs = {}  # (row tiles, dtypes, L1 budget, widths, memory configs) -> (plan, (kernels, cbs))

    def __call__(self, x, stream, out_dtype, out_width=None, memory_config=None):
        """x: [..., M, K] TILE (interleaved or sharded, DRAM or L1; bfloat16 / bfloat8_b); stream: the decode stream
        tensor; -> a new [..., M, out_width or N] TILE tensor (DRAM interleaved by default), columns >= N zero."""
        import ttnn

        from .fused_decode.stream import NBIAS as STREAM_NBIAS
        from .fused_decode.stream import TILE_BYTES

        assert STREAM_NBIAS == NBIAS, (STREAM_NBIAS, NBIAS)
        spec, mesh = self.spec, self.mesh_device
        assert x.layout == ttnn.TILE_LAYOUT, "x must be in TILE layout"
        padded = list(x.padded_shape)
        assert padded[-1] == spec.k, f"x width {padded[-1]} != K {spec.k}"
        sshape = list(stream.padded_shape)
        assert sshape[-2:] == [spec.rows, spec.banks * TILE], (sshape, spec.rows)
        m_tiles = math.prod(padded[:-1]) // TILE
        width = spec.n if out_width is None else out_width
        assert width >= spec.n and width % TILE == 0, f"out_width {out_width} < N {spec.n} or not whole tiles"
        out_tiles = width // TILE
        x_tile, w_tile, o_tile = TILE_BYTES[x.dtype], TILE_BYTES[stream.dtype], TILE_BYTES[out_dtype]
        out = ttnn.empty(
            list(x.shape)[:-1] + [width],
            dtype=out_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
        )
        ones = _ones_tensor(mesh, x.dtype)
        grid = mesh.compute_with_storage_grid_size()
        budget = _l1_budget(mesh)  # (after the output: an L1 output lowers it)
        key = (m_tiles, str(x.dtype), str(stream.dtype), str(out_dtype), budget, out_tiles)
        key += (str(x.memory_config()), str(out.memory_config()))  # (the tensor accessors' compile-time args)
        cached = self._programs.get(key)
        if cached is None:
            plan = plan_work(spec, m_tiles, grid.x * grid.y, x_tile, w_tile, o_tile, budget, out_tiles)
            cached = self._programs[key] = (plan, self._descriptors(plan, x, stream, ones, out, out_dtype, grid))
        plan, (kernels, cbs) = cached
        # Only the buffer addresses change between calls (common runtime args; the program hash ignores their values).
        kernels[0].common_runtime_args = [stream.buffer_address(), x.buffer_address(), ones.buffer_address()]
        kernels[1].common_runtime_args = [out.buffer_address()]
        ttnn.generic_op([x, stream, ones, out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        return out

    def _descriptors(self, plan, x, stream, ones, out, out_dtype, grid):
        """The kernels (addresses left to the caller: common runtime args) and circular buffers of one plan."""
        import ttnn

        from .fused_decode.stream import TILE_BYTES

        spec = self.spec
        x_tile, w_tile, o_tile = TILE_BYTES[x.dtype], TILE_BYTES[stream.dtype], TILE_BYTES[out_dtype]
        cores = _core_ranges(ttnn, len(plan.items), grid.x)
        cb_x, cb_w, cb_out, cb_zero = 0, 1, 16, 17

        def cb(index, dtype, page, pages):
            return ttnn.CBDescriptor(
                total_size=page * pages,
                core_ranges=cores,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)],
            )

        cbs = [
            cb(cb_x, x.dtype, x_tile, plan.x_bufs * plan.MB * spec.kt),
            cb(cb_w, stream.dtype, w_tile, plan.w_pages * spec.kt),
            cb(cb_out, out_dtype, o_tile, 2 * MAX_RT),
        ]
        out_tiles = list(out.padded_shape)[-1] // TILE
        if out_tiles > spec.n_tiles:
            cbs.append(cb(cb_zero, out_dtype, o_tile, 1))
        reader_rt, compute_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for i, it in enumerate(plan.items):
            cx, cy = i % grid.x, i // grid.x
            reader_rt[cx][cy] = [it.bank, it.j0 * spec.kt * w_tile, it.cols, it.m0, it.m_cnt]
            compute_rt[cx][cy] = [it.cols, it.m_cnt]
            writer_rt[cx][cy] = [it.m0, it.m_cnt, it.cols, it.z0, it.z1] + spec.columns(it.bank, it.j0, it.cols)
        resident = int(plan.resident)
        reader_ct = [cb_x, cb_w, spec.kx, w_tile, plan.MB, plan.G, resident]
        reader_ct += ttnn.TensorAccessorArgs(x).get_compile_time_args()
        reader_ct += ttnn.TensorAccessorArgs(ones).get_compile_time_args()
        compute_ct = [cb_x, cb_w, cb_out, spec.kt, plan.MB, plan.G, resident]
        writer_ct = [cb_out, cb_zero, out_tiles, plan.MB, plan.G] + ttnn.TensorAccessorArgs(out).get_compile_time_args()
        dm = ttnn.DataMovementConfigDescriptor
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "stream_prefill_reader.cpp"),
                core_ranges=cores,
                compile_time_args=reader_ct,
                runtime_args=reader_rt,
                common_runtime_args=[0, 0, 0],
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "stream_prefill_writer.cpp"),
                core_ranges=cores,
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                common_runtime_args=[0],
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "stream_prefill_compute.cpp"),
                core_ranges=cores,
                compile_time_args=compute_ct,
                runtime_args=compute_rt,
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=getattr(ttnn.MathFidelity, self.math_fidelity),
                    fp32_dest_acc_en=True,
                    math_approx_mode=False,
                ),
            ),
        ]
        return kernels, cbs


def resplit_vocab(logits, tp, vd, per_device, cluster_axis):
    """Device: LM-head logits in the decode vocab split (device d: ids [d * vd, (d + 1) * vd) in its first vd columns,
    zeros up to per_device) -> the prefill split (device d: ids [d * per_device, (d + 1) * per_device), zeros past the
    tp * vd ids), [.., per_device] per device. Tile-aligned moves only (all_gather, slices, concat, mesh_partition):
    the values are unchanged. Frees `logits`."""
    import ttnn

    dram = ttnn.DRAM_MEMORY_CONFIG
    rank = len(logits.shape)
    lead = [logits.shape[i] for i in range(rank - 1)]
    assert logits.shape[-1] == per_device, f"logits width {logits.shape[-1]} != {per_device}"
    gathered = ttnn.all_gather(logits, rank - 1, cluster_axis=cluster_axis, memory_config=dram)
    ttnn.deallocate(logits)
    pieces = [
        ttnn.slice(gathered, [0] * (rank - 1) + [a], lead + [b], memory_config=dram)
        for a, b in vocab_resplit_pieces(tp, vd, per_device)
    ]
    ttnn.deallocate(gathered)
    full = ttnn.concat(pieces, dim=rank - 1, memory_config=dram)
    for p in pieces:
        ttnn.deallocate(p)
    out = ttnn.mesh_partition(full, rank - 1, cluster_axis=cluster_axis, memory_config=dram)
    ttnn.deallocate(full)
    return out
