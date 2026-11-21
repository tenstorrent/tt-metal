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

Work split (plan_work): every core owns `cols` consecutive real slots of the banks' real slots in bank order (one or
more segments, each a contiguous byte range of one bank: a chunk may cross a bank boundary, so the column chunks need
not divide a bank's slots) and a range of output row tiles. Per M block (MB row tiles) the core reads its x rows (K / 32
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
MAX_RT = 4  # output tiles per DST block (FP32 DST, half sync: 4 tiles)


def block_shape(rows):
    """(rt, ct) of the compute kernel's DST blocks in an M block of `rows` row tiles: rt rows (the last block of the M
    block may have fewer) by ct consecutive weight columns (the last column block may have fewer), rt * ct <= MAX_RT.
    The x rows of K = 2880 fill L1 fast, so an M block is often 1 or 2 rows: the columns fill the DST block then
    (stream_prefill_compute.cpp)."""
    rt = min(rows, MAX_RT)
    return rt, MAX_RT // rt


def dst_blocks(rows, cols):
    """The DST blocks of one M block and weight group, in compute order: (r0, rt, c0, ct) for column blocks of ct
    columns, then row blocks of rt rows; each block's tiles in row-major order (row r, then column c)."""
    rt_m, ct_m = block_shape(rows)
    out = []
    for c0 in range(0, cols, ct_m):
        ct = min(ct_m, cols - c0)
        for r0 in range(0, rows, rt_m):
            out.append((r0, min(rt_m, rows - r0), c0, ct))
    return out


MAX_MB = 8  # row tiles per M block (L1 permitting)
L1_CAP = 900 * 1024  # circular-buffer bytes per core at most
L1_MARGIN = 16 * 1024  # kept free below the lowest L1 buffer
# The free L1 below the lowest L1 buffer is rounded down to this step before it sizes a plan: the budget is part of the
# program cache key, so small changes in L1 occupancy between calls reuse the plan and kernels instead of building
# new ones in the middle of a request.
L1_BUDGET_STEP = 64 * 1024
MAX_WRITER_COLS = 300  # the writer's runtime args hold the core's column list
# plan_work cost model (rough, Blackhole per device): DRAM read bandwidth, one core's read bandwidth, one 32x32x32
# HiFi2 tile multiply
DRAM_BW = 400e9
CORE_BW = 30e9
# one bank's read bandwidth for the resident weight load (cost model of resident plans only): without weight multicast,
# QKV at 8K loaded 5.9 MB from its busiest bank (12 row ranges reading the same slots) in 290 us
BANK_BW = 20e9
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
    """One core's share: slots [j0, j0 + cols) of bank `bank` (or, with segs, the slot segments [(bank, j0, cnt), ...]
    in order), output row tiles [m0, m0 + m_cnt), and the zero output columns [z0, z1) it writes."""

    def __init__(self, bank, j0, cols, m0, m_cnt, z0=0, z1=0, segs=None):
        self.segs = list(segs) if segs is not None else [(bank, j0, cols)]
        assert self.segs and all(cnt >= 0 for _, _, cnt in self.segs), self.segs
        self.cols = sum(cnt for _, _, cnt in self.segs)
        self.m0, self.m_cnt, self.z0, self.z1 = m0, m_cnt, z0, z1

    def columns(self, spec):
        """Output column tiles of the core's slots, in slot order."""
        return [c for b, j0, cnt in self.segs for c in spec.columns(b, j0, cnt)]

    def slot(self, c):
        """(bank, slot) of the core's column c."""
        for b, j0, cnt in self.segs:
            if c < cnt:
                return b, j0 + c
            c -= cnt
        raise IndexError(c)

    def __repr__(self):
        segs = " ".join(f"b{b}j{j0}+{cnt}" for b, j0, cnt in self.segs)
        return f"WorkItem({segs} m{self.m0}+{self.m_cnt} z[{self.z0},{self.z1}))"


class WorkPlan:
    """plan_work's result: the items (one per core), the block geometry and the circular-buffer sizes."""

    def __init__(self, items, MB, G, resident, x_bufs, cost, group=0, column=False):
        self.items, self.MB, self.G, self.resident, self.x_bufs, self.cost = items, MB, G, resident, x_bufs, cost
        # x multicast (group > 0): the items of one row range are one grid row of `group` cores (item i on core
        # (i % group, i // group)), or with column one grid column (core (i // group, i % group)); its first core
        # reads the x blocks and multicasts them to the others.
        self.group = group
        self.column = column

    @property
    def ranges(self):
        """Row ranges (items per column chunk)."""
        return len(self.items) // self.group if self.group else 0

    @property
    def w_mcast(self):
        """Weight multicast: resident weights in a multicast plan with >= 2 row ranges. The cores of one column chunk
        are then one grid column (row multicast mode) or row (column mode); the first (row range 0) reads the
        chunk's weights and multicasts them to the others (stream_prefill_reader.cpp W_MCAST)."""
        return bool(self.group and self.resident and self.ranges >= 2)

    def core(self, i, gx):
        """Grid core (x, y) of item i."""
        if self.group and self.column:
            return i // self.group, i % self.group
        g = self.group or gx
        return i % g, i // g

    @property
    def w_pages(self):
        """Weight circular-buffer pages: one group (resident) or a two-group ring, kt tiles per column."""
        return self.G * (1 if self.resident else 2)

    def __repr__(self):
        return (
            f"WorkPlan({len(self.items)} cores, MB {self.MB}, G {self.G}, resident {self.resident}, "
            f"x_bufs {self.x_bufs}, cost {self.cost * 1e6:.1f} us"
            + (f", x multicast over {self.group} cores" + (" (grid columns)" if self.column else "") if self.group else "")
            + (f", weights multicast over {self.ranges} cores" if self.w_mcast else "")
            + ")"
        )


def _geometry(chunk, rows, kt, x_tile, w_tile, out_tile, budget, streamed=True):
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
    if not streamed:
        return None
    for mb, bufs in xs:
        xb = bufs * mb * kt * x_tile
        for g in range(min(chunk, 4), 0, -1):
            wb = 2 * g * kt * w_tile
            if wb + xb + out_bytes <= budget:
                return g, mb, False, bufs, wb + xb + out_bytes
    return None


def plan_work(spec, m_tiles, n_cores, x_tile, w_tile, out_tile, budget, out_tiles=None, grid=None):
    """The work split of one call: the banks' real slots (in bank order) in `pn` chunks of <= chunk consecutive slots
    (one core each; a chunk may span two banks) times `pm` ranges of output row tiles, minimizing a rough time estimate
    max(DRAM bytes / bandwidth, the busiest bank's weight bytes / its share, the busiest core's reads / a core's
    bandwidth, its tile multiplies): x is read once per column chunk (pn times), the weights once per row range (pm
    times; per M block when not resident). out_tiles (> spec.n_tiles): the extra zero output columns, split over the
    chunks of each row range.

    grid (gx, gy): also consider multicasting x. The column chunks of one row range are then one grid row (at most gx
    chunks, at most gy row ranges) or one grid column (at most gy chunks, at most gx row ranges); its first core reads
    each x block from DRAM and multicasts it to the others, so x is read once per row range instead of once per chunk
    (8x for QKV at 8K: x bandwidth bound). Resident weights are then multicast the other way (WorkPlan.w_mcast):
    read once per chunk instead of once per row range (the row ranges' reads of one bank bound the op before). The
    cheapest mode is taken."""
    assert m_tiles >= 1
    out_tiles = spec.n_tiles if out_tiles is None else out_tiles
    assert out_tiles >= spec.n_tiles
    slots = [(b, j) for b in range(spec.banks) for j in range(spec.real[b])]
    R = max(spec.real)
    kt = spec.kt
    w_total = len(slots) * kt * w_tile
    x_total = m_tiles * spec.kx * x_tile
    best = None
    modes = [(None, n_cores)]
    if grid is not None:
        modes += [((grid[0], grid[1], False), grid[0] * grid[1]), ((grid[1], grid[0], True), grid[0] * grid[1])]
    # resident weights whenever some split fits them (read once, then multicast), else streamed weight groups
    for streamed in (False, True):
        for mc, cores_avail in modes:
            best = _plan_search(
                spec, m_tiles, cores_avail, x_tile, w_tile, out_tile, budget, len(slots), R, kt, w_total, x_total, mc,
                best, streamed,
            )
        if best is not None:
            break
    assert best is not None, f"no work split fits {budget} bytes of L1 (K {spec.k}, N {spec.n}, {m_tiles} row tiles)"
    (t, _), pn, rows, (G, MB, resident, x_bufs, _), mc = best
    chunks = []  # balanced: sizes differ by at most one
    for i in range(pn):
        a, b = len(slots) * i // pn, len(slots) * (i + 1) // pn
        segs = []
        for bank, j in slots[a:b]:
            if segs and segs[-1][0] == bank:
                segs[-1][2] += 1
            else:
                segs.append([bank, j, 1])
        chunks.append([tuple(sg) for sg in segs])
    items = []
    for m0 in range(0, m_tiles, rows):
        m_cnt = min(rows, m_tiles - m0)
        row_items = [WorkItem(0, 0, 0, m0, m_cnt, segs=segs) for segs in chunks]
        extra = out_tiles - spec.n_tiles  # zero columns, split evenly over this row range's cores
        for i, it in enumerate(row_items):
            it.z0 = spec.n_tiles + extra * i // len(row_items)
            it.z1 = spec.n_tiles + extra * (i + 1) // len(row_items)
        items += row_items
    assert len(items) <= (grid[0] * grid[1] if mc else n_cores)
    assert all(it.cols <= MAX_WRITER_COLS for it in items)
    return WorkPlan(items, MB, G, resident, x_bufs, t, pn if mc else 0, bool(mc and mc[2]))


def _plan_search(
    spec, m_tiles, n_cores, x_tile, w_tile, out_tile, budget, S, R, kt, w_total, x_total, mc, best, streamed=True
):
    """plan_work's search over (column chunks, row ranges) in one mode (mc: None, or (most chunks per group, most
    groups, along grid columns)); returns the best (key, chunks, rows per range, geometry, mc) so far."""
    for pn in range(1, S + 1):
        chunk = -(-S // pn)
        if pn > n_cores:
            break
        if chunk > MAX_WRITER_COLS:
            continue  # the writer's runtime args hold the core's column list
        if mc and (pn < 2 or pn > mc[0]):
            continue  # multicast needs >= 2 chunks in one grid row / column
        seen = set()
        pm_max = min(m_tiles, mc[1] if mc else n_cores // pn)
        for pm in range(1, pm_max + 1):
            rows = -(-m_tiles // pm)
            pm_eff = -(-m_tiles // rows)
            if pm_eff in seen:
                continue
            seen.add(pm_eff)
            geo = _geometry(chunk, rows, kt, x_tile, w_tile, out_tile, budget, streamed)
            if geo is None:
                continue
            G, MB, resident, x_bufs, _ = geo
            passes = 1 if resident else -(-rows // MB)
            w_copies = 1 if (mc and resident and pm_eff >= 2) else pm_eff * passes  # WorkPlan.w_mcast
            w_reads = w_total * w_copies
            bank_reads = R * kt * w_tile * w_copies
            x_reads = x_total if mc else pn * x_total
            if resident:
                # the weights are loaded before the first M block's compute: their time adds to the rest
                load = max(w_reads / DRAM_BW, bank_reads / BANK_BW, chunk * kt * w_tile / CORE_BW)
                t = load + max(x_reads / DRAM_BW, rows * spec.kx * x_tile / CORE_BW, rows * chunk * kt * TILE_MM_S)
            else:
                t = max(
                    (w_reads + x_reads) / DRAM_BW,
                    bank_reads / (DRAM_BW / spec.banks),
                    (rows * spec.kx * x_tile + passes * chunk * kt * w_tile) / CORE_BW,
                    rows * chunk * kt * TILE_MM_S,
                )
            key = (t, pn * pm_eff)
            if best is None or key < best[0]:
                best = (key, pn, rows, geo, mc)
    return best


# ---- the kernels' loops in torch (tests/unit/test_stream_prefill_linear.py) ----


def compute_order(item, plan):
    """(local row, local column) of each output tile in the order the compute kernel packs them."""
    out = []
    for mb in range(0, item.m_cnt, plan.MB):
        rows = min(plan.MB, item.m_cnt - mb)
        for g in range(0, item.cols, plan.G):
            for r0, rt, c0, ct in dst_blocks(rows, min(plan.G, item.cols - g)):
                out += [(mb + r, g + c) for r in range(r0, r0 + rt) for c in range(c0, c0 + ct)]
    return out


def writer_pages(item, plan, columns, out_tiles):
    """Output tile pages in the order the writer writes the compute kernel's tiles (stream_prefill_writer.cpp), after
    its zero columns."""
    pages = [(item.m0 + r) * out_tiles + z for r in range(item.m_cnt) for z in range(item.z0, item.z1)]
    for mb in range(0, item.m_cnt, plan.MB):
        rows = min(plan.MB, item.m_cnt - mb)
        for g in range(0, item.cols, plan.G):
            for r0, rt, c0, ct in dst_blocks(rows, min(plan.G, item.cols - g)):
                pages += [
                    (item.m0 + mb + r) * out_tiles + columns[g + c]
                    for r in range(r0, r0 + rt)
                    for c in range(c0, c0 + ct)
                ]
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
        columns = it.columns(spec)
        for r in range(it.m_cnt):
            for z in range(it.z0, it.z1):
                row = it.m0 + r
                out[row * TILE : (row + 1) * TILE, z * TILE : (z + 1) * TILE] = 0

        # the weights as stored (the reader lays each group out in cb_w as tile (k, c) at k * G + c)
        def w_tile(c, k):
            bank, j = it.slot(c)
            t = j * spec.kt + k
            return layout[t * TILE : (t + 1) * TILE, bank * TILE : (bank + 1) * TILE].double()

        for mb in range(0, it.m_cnt, plan.MB):
            rows = min(plan.MB, it.m_cnt - mb)
            # cb_x: row r's kx x tiles, then the ones tile
            block = []
            for r in range(rows):
                row = it.m0 + mb + r
                xr = xd[row * TILE : (row + 1) * TILE]
                block.append([xr[:, k * TILE : (k + 1) * TILE] for k in range(spec.kx)] + [ones])
            for g in range(0, it.cols, plan.G):
                for r0, rt, c0, ct in dst_blocks(rows, min(plan.G, it.cols - g)):
                    for r in range(r0, r0 + rt):
                        for c in range(g + c0, g + c0 + ct):
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


def _l1_base(mesh_device):
    """The L1 allocator base on the compute cores (where a program's circular buffers can start)."""
    import ttnn

    return ttnn.get_allocator_base_address(mesh_device, ttnn.BufferType.L1)


def _l1_budget(mesh_device):
    """Circular-buffer bytes per core: below the lowest L1 buffer now allocated (a program's circular buffers must end
    below it), rounded down to L1_BUDGET_STEP, at most L1_CAP."""
    base = _l1_base(mesh_device)
    low = mesh_device.lowest_occupied_compute_l1_address()
    if not low:
        return L1_CAP
    free = max(0, low - base - L1_MARGIN)
    return min(L1_CAP, free // L1_BUDGET_STEP * L1_BUDGET_STEP)


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
            plan = plan_work(
                spec, m_tiles, grid.x * grid.y, x_tile, w_tile, o_tile, budget, out_tiles, (grid.x, grid.y)
            )
            kernels, cbs, sems, tables = self._descriptors(plan, x, stream, ones, out, out_dtype, grid)
            # The program is fixed by the key (the addresses are common runtime args, which the cache hit patches): a
            # custom hash skips hashing every core's runtime args on each call (the host time of a call at 1K).
            phash = hash(("StreamPrefillLinear", spec.k, spec.n, tuple(spec.order), self.math_fidelity) + key)
            cached = self._programs[key] = (plan, kernels, cbs, sems, phash & ((1 << 64) - 1), tables)
        plan, kernels, cbs, sems, phash, tables = cached
        # Only the buffer addresses change between calls (common runtime args; the program hash ignores their values).
        # The kernels have no per-core runtime args (stream_prefill_args.hpp): a cache hit patches these 6 words.
        kernels[0].common_runtime_args = [
            stream.buffer_address(),
            x.buffer_address(),
            ones.buffer_address(),
            tables[0].buffer_address(),
        ]
        kernels[1].common_runtime_args = [out.buffer_address(), tables[1].buffer_address()]
        desc = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
        desc.custom_program_hash = phash
        ttnn.generic_op([x, stream, ones, out], desc)
        return out

    def _descriptors(self, plan, x, stream, ones, out, out_dtype, grid):
        """The kernels (addresses left to the caller: common runtime args), circular buffers, semaphores and the
        reader's / writer's per-core argument tables of one plan."""
        import torch
        import ttnn

        from .fused_decode.stream import TILE_BYTES

        spec = self.spec
        x_tile, w_tile, o_tile = TILE_BYTES[x.dtype], TILE_BYTES[stream.dtype], TILE_BYTES[out_dtype]
        if plan.group:
            last = plan.core(len(plan.items) - 1, grid.x)
            cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(last[0], last[1]))])
        else:
            cores = _core_ranges(ttnn, len(plan.items), grid.x)
        cb_x, cb_w, cb_out, cb_zero, cb_rargs, cb_wargs = 0, 1, 16, 17, 18, 19
        ready_sem, valid_sem = 0, 1  # x multicast handshake (plan.group)
        w_ready_sem, w_valid_sem = 2, 3  # weight multicast handshake (plan.w_mcast)

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
        per_core = {"reader": {}, "compute": {}, "writer": {}}
        noc = self.mesh_device.worker_core_from_logical_core
        for i, it in enumerate(plan.items):
            cx, cy = plan.core(i, grid.x)
            args = [it.cols, it.m0, it.m_cnt]
            if plan.group:
                # [is_sender, multicast rectangle (receivers: members 1..group-1 of the row / column), receivers,
                # sender]
                if plan.column:
                    member, ends, sender = cy, ((cx, 1), (cx, plan.group - 1)), (cx, 0)
                else:
                    member, ends, sender = cx, ((1, cy), (plan.group - 1, cy)), (0, cy)
                first, last = (noc(ttnn.CoreCoord(*e)) for e in ends)
                snd = noc(ttnn.CoreCoord(*sender))
                args += [int(member == 0), first.x, first.y, last.x, last.y, plan.group - 1, snd.x, snd.y]
            else:
                args += [1, 0, 0, 0, 0, 0, 0, 0]
            if plan.w_mcast:
                # [w_is_sender, multicast rectangle (row ranges 1.. of the chunk), receivers, sender]: the chunk's
                # cores are the grid column (row mode) / row (column mode) across the row ranges
                rng, nr = i // plan.group, plan.ranges
                if plan.column:
                    ends, sender = ((1, cy), (nr - 1, cy)), (0, cy)
                else:
                    ends, sender = ((cx, 1), (cx, nr - 1)), (cx, 0)
                first, last = (noc(ttnn.CoreCoord(*e)) for e in ends)
                snd = noc(ttnn.CoreCoord(*sender))
                args += [int(rng == 0), first.x, first.y, last.x, last.y, nr - 1, snd.x, snd.y]
            else:
                args += [1, 0, 0, 0, 0, 0, 0, 0]
            args.append(len(it.segs))
            for bank, j0, cnt in it.segs:
                args += [bank, j0 * spec.kt * w_tile, cnt]
            per_core["reader"][cx, cy] = args
            per_core["compute"][cx, cy] = (it.cols, it.m_cnt)
            per_core["writer"][cx, cy] = [it.m0, it.m_cnt, it.cols, it.z0, it.z1] + it.columns(spec)
        # The reader's / writer's per-core args: rows of two DRAM tables (row = y * grid.x + x), which the kernels read
        # at start (stream_prefill_args.hpp) instead of per-core runtime args, the host cost of a call (~0.5 us per
        # core and kernel on every call: 0.2-0.5 ms on 120 cores; with none 0.014 ms).
        tables, table_ct = [], []
        for name, cb_args in (("reader", cb_rargs), ("writer", cb_wargs)):
            rows = per_core[name]
            width = -(-max(len(a) for a in rows.values()) // 16) * 16  # 64-byte rows
            data = torch.zeros(grid.x * grid.y, width, dtype=torch.int32)
            for (cx, cy), a in rows.items():
                data[cy * grid.x + cx, : len(a)] = torch.tensor(a, dtype=torch.int64).to(torch.int32)
            t = ttnn.from_torch(
                data,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            tables.append(t)
            table_ct.append([cb_args, grid.x, width * 4] + ttnn.TensorAccessorArgs(t).get_compile_time_args())
            cbs.append(cb(cb_args, ttnn.uint32, width * 4, 1))
        resident = int(plan.resident)
        reader_ct = [
            cb_x,
            cb_w,
            spec.kx,
            w_tile,
            plan.MB,
            plan.G,
            resident,
            int(bool(plan.group)),
            ready_sem,
            valid_sem,
            int(plan.w_mcast),
            w_ready_sem,
            w_valid_sem,
        ]
        reader_ct += ttnn.TensorAccessorArgs(x).get_compile_time_args()
        reader_ct += ttnn.TensorAccessorArgs(ones).get_compile_time_args()
        reader_ct += table_ct[0]
        compute_ct = [cb_x, cb_w, cb_out, spec.kt, plan.MB, plan.G, resident]
        writer_ct = [cb_out, cb_zero, out_tiles, plan.MB, plan.G] + ttnn.TensorAccessorArgs(out).get_compile_time_args()
        writer_ct += table_ct[1]
        dm = ttnn.DataMovementConfigDescriptor
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "stream_prefill_reader.cpp"),
                core_ranges=cores,
                compile_time_args=reader_ct,
                common_runtime_args=[0, 0, 0, 0],
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "stream_prefill_writer.cpp"),
                core_ranges=cores,
                compile_time_args=writer_ct,
                common_runtime_args=[0, 0],
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1),
            ),
        ]
        # The compute kernel's per-core values (columns, row tiles) as compile-time args: one kernel per distinct
        # pair (a balanced plan has at most two of each).
        by_shape = {}
        for (cx, cy), shape in per_core["compute"].items():
            by_shape.setdefault(shape, []).append(ttnn.CoreRange(ttnn.CoreCoord(cx, cy), ttnn.CoreCoord(cx, cy)))
        for (cols, m_cnt), ranges in sorted(by_shape.items()):
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "stream_prefill_compute.cpp"),
                    core_ranges=ttnn.CoreRangeSet(ranges),
                    compile_time_args=compute_ct + [cols, m_cnt],
                    config=ttnn.ComputeConfigDescriptor(
                        math_fidelity=getattr(ttnn.MathFidelity, self.math_fidelity),
                        fp32_dest_acc_en=True,
                        math_approx_mode=False,
                    ),
                )
            )
        sems = []
        if plan.group:
            sems = [
                ttnn.SemaphoreDescriptor(ready_sem, ttnn.CoreType.WORKER, cores, 0),
                ttnn.SemaphoreDescriptor(valid_sem, ttnn.CoreType.WORKER, cores, 0),
            ]
        if plan.w_mcast:
            sems += [
                ttnn.SemaphoreDescriptor(w_ready_sem, ttnn.CoreType.WORKER, cores, 0),
                ttnn.SemaphoreDescriptor(w_valid_sem, ttnn.CoreType.WORKER, cores, 0),
            ]
        return kernels, cbs, sems, tables


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
