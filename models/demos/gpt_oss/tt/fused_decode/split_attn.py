# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Split decode attention: the one-user decode's attention (sliding-window and full layers) fused into the decode QKV
program (fused_decode/kernels/sda_*.cpp), replacing the paged SDPA decode op.

One query row (one token) per device over the paged K / V cache; kv head g's 8 query heads run on S slice cores:
  - the attended positions [lo, p] (lo = p + 1 - window with a sliding window, else 0) are cut into A <= S contiguous
    slices of at least min_chunks 32-position chunks (A = min(S, ceil(N / min_chunks))); each slice core runs the 8
    heads (rows 0..7 of one Q tile) over its chunks with an online softmax over groups of G chunks; its reader fetches
    the chunks that are already in the cache while QKV runs, Q and the position's chunk once the QKV writers signal;
  - each slice's state is (m, O, L) (FP32 rows; L = rowsum P comes out of the P . V matmul as P . ones); the active
    slices of a kv head all-gather their row maxima m (one NOC write of 32 bytes per pair), so every slice knows
    M = max m and writes its state scaled by exp(m - M) into its slot of the kv head's leader (slice 0); the leader
    sums the slots (a matmul with a 0/1 fold matrix: pack tile k holds slice 4 k + i in rows 8 i .. 8 i + 7) and
    writes out = sum O / sum L;
  - slice 0 also takes the heads' attention sinks (an extra score column in the max and in L, no V).
The output is the SDPA output layout [1, 1, heads, 64] (BF16 tiles, L1): the o_proj stream reads its rows directly.

Every argument is common (no per-core runtime args: dispatch writes those core by core); each core finds its index
from its logical coordinates in a byte table over the grid.
"""

import struct

import torch

import ttnn

from .boundary import KERNEL_DIR

TILE = ttnn.TILE_SIZE
G = 8  # chunks per group (kernels/sda_common.hpp)
P_SEM = 9  # the slices' scaled states landed in the leader (count)
M_SEM = 10  # the kv head's slices' row maxima landed (count)
# circular buffers (kernels/sda_common.hpp)
(
    CB_Q,
    CB_K,
    CB_V,
    CB_AUX,
    CB_SCR,
    CB_QS,
    CB_S,
    CB_P,
    CB_M,
    CB_O,
    CB_AB,
    CB_ONES,
    CB_MOUT,
    CB_MX,
    CB_W,
    CB_PART,
    CB_POS,
    CB_MG,
    CB_OPACK,
    CB_FOLD,
    CB_ACC,
    CB_OUT,
) = range(22)
AUX_TILES = 7


def _f32_bits(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def rects(cores):
    """The cores as few rectangles as possible (row runs, stacked when consecutive rows share the run): dispatch writes
    kernel binaries and arguments per core range."""
    rows = {}
    for c in cores:
        rows.setdefault(c.y, []).append(c.x)
    runs = []
    for y in sorted(rows):
        xs = sorted(rows[y])
        x0 = prev = xs[0]
        for x in xs[1:] + [None]:
            if x is None or x != prev + 1:
                runs.append((x0, prev, y))
                x0 = x
            prev = x if x is not None else prev
    out = []
    for x0, x1, y in runs:
        r = next((r for r in out if r[0] == x0 and r[1] == x1 and r[3] == y - 1), None)
        if r is not None:
            r[3] = y
        else:
            out.append([x0, x1, y, y])
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(a, c), ttnn.CoreCoord(b, d)) for a, b, c, d in out])


class SplitDecodeAttention:
    def __init__(
        self,
        mesh_device,
        n_heads,
        n_kv,
        head_dim,
        window,
        cores,
        slices,
        min_chunks=1,
        cap=3,
        out=None,
        consts=None,
    ):
        """cores: n_kv * slices logical cores, kv head g's slices at [g * slices, (g + 1) * slices) (its leader first).
        window: the sliding window (0: full attention). cap: K / V groups in flight per core. out / consts: the output
        tensor (output_tensor()) and the constants (consts_tensor(), in the K / V cache's dtype), allocated here when
        None."""
        assert head_dim == 64 and n_heads // n_kv == 8, "8 query heads per kv head, head_dim 64"
        assert len(cores) == n_kv * slices and 1 <= slices <= 63
        self.mesh, self.nq, self.nkv, self.window = mesh_device, n_heads, n_kv, window
        self.S, self.min_chunks, self.cap = slices, min_chunks, cap
        self.core_list = list(cores)
        self.cores = rects(self.core_list)
        self.grid = mesh_device.compute_with_storage_grid_size()
        assert len(self.core_list) < 0xFF
        self.out = out if out is not None else output_tensor(mesh_device, n_heads, head_dim)
        self.consts = consts if consts is not None else consts_tensor(mesh_device)

    def noc_cores(self):
        return [self.mesh.worker_core_from_logical_core(c) for c in self.core_list]

    def semaphores(self):
        """The op's own program semaphores (besides the QKV wait semaphore)."""
        return [ttnn.SemaphoreDescriptor(i, ttnn.CoreType.WORKER, self.cores, 0) for i in (P_SEM, M_SEM)]

    def parts(self, q, k_cache, v_cache, pos, page_table, sinks, scale, wait):
        """(kernels, cbs, io). q: the [1, 1, heads, 64] BF16 Q head tensor (head h in row h % 32 of tile row h / 32);
        k_cache / v_cache: paged [blocks, kv heads, block_size, 64]; pos: INT32 [1]; page_table: INT32 [1, blocks];
        sinks: the heads' [heads, 32] BF16 tile (column 0 = sink / scale); wait = (semaphore id, count): the QKV
        writers' signal that Q and the position's K / V rows are written."""
        S, cap = self.S, self.cap
        K = -(-S // 4)
        kv_dtype = k_cache.dtype
        kv_tile = {ttnn.bfloat8_b: 1088, ttnn.bfloat16: 2048}[kv_dtype]
        block = k_cache.shape[2]
        assert k_cache.shape[1] == self.nkv and block % TILE == 0
        pt_page = page_table.shape[-1] * 4
        f32, bf16 = ttnn.float32, ttnn.bfloat16
        # the most chunks a slice can hold (sda_common.hpp split): one online-softmax group -> no rescale code
        n_max = (self.window // TILE + 1) if self.window else -(-k_cache.shape[0] * block // TILE)
        per_slice = max(self.min_chunks, -(-n_max // S)) + 1
        one_group = per_slice <= G

        def cb(i, dtype, page, pages):
            return ttnn.CBDescriptor(
                total_size=page * pages,
                core_ranges=self.cores,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=dtype, page_size=page)],
            )

        assert self.consts.dtype == kv_dtype, "the ones tile is unpacked in the V format"
        cbs = [
            cb(CB_Q, bf16, 2048, 2),
            cb(CB_K, kv_dtype, kv_tile, 2 * G * cap),
            cb(CB_V, kv_dtype, kv_tile, 2 * G * cap),
            cb(CB_AUX, bf16, 2048, AUX_TILES),
            cb(CB_SCR, bf16, 64 + max(64, -(-pt_page // 64) * 64), 1),
            cb(CB_S, f32, 4096, G + 1),
            cb(CB_P, bf16, 2048, G + 1),
            cb(CB_M, f32, 4096, 2),
            cb(CB_O, f32, 4096, 6),
            cb(CB_AB, f32, 4096, 2),
            cb(CB_ONES, kv_dtype, kv_tile, 1),
            cb(CB_MOUT, f32, 4096, 1),
            cb(CB_MX, f32, 4096, 1),
            cb(CB_W, f32, 4096, 1),
            cb(CB_PART, f32, 4096, 3),
            cb(CB_POS, bf16, 64, 1),
            cb(CB_MG, f32, 32 * S, 1),
            cb(CB_OPACK, f32, 4096, 3 * K),
            cb(CB_FOLD, f32, 4096, 1),
            cb(CB_ACC, f32, 4096, 3),
            cb(CB_OUT, bf16, 2048, 2),
        ]
        reader_ct = [self.nkv, block, self.window, kv_tile, pt_page, S, self.min_chunks, cap]
        for t in (q, k_cache, v_cache, pos, page_table, sinks, self.consts):
            reader_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        writer_ct = [self.window, S, self.min_chunks]
        writer_ct += ttnn.TensorAccessorArgs(self.out).get_compile_time_args()
        compute_ct = [_f32_bits(scale)]
        gx, gy = self.grid.x, self.grid.y
        idx = [0xFF] * (gx * gy)
        for i, c in enumerate(self.core_list):
            idx[c.y * gx + c.x] = i
        idx += [0xFF] * (-len(idx) % 4)
        idx_words = [sum(idx[w + k] << (8 * k) for k in range(4)) for w in range(0, len(idx), 4)]
        nocs = [(p.x << 8) | p.y for p in self.noc_cores()]
        nocs += [0] * (len(nocs) % 2)
        noc_words = [nocs[w] | (nocs[w + 1] << 16) for w in range(0, len(nocs), 2)]
        reader_common = [
            q.buffer_address(),
            k_cache.buffer_address(),
            v_cache.buffer_address(),
            pos.buffer_address(),
            page_table.buffer_address(),
            sinks.buffer_address(),
            self.consts.buffer_address(),
        ] + idx_words
        writer_common = [self.out.buffer_address()] + idx_words + noc_words
        defines = [
            ("SDA_WAIT_SEM", str(wait[0])),
            ("SDA_WAIT_N", str(wait[1])),
            ("SDA_P_SEM", str(P_SEM)),
            ("SDA_M_SEM", str(M_SEM)),
            ("SDA_GX", str(gx)),
            ("SDA_NIDX", str(len(idx_words))),
        ]
        compute_defines = [("SDA_ONE_GROUP", "1")] if one_group else []
        dm = ttnn.DataMovementConfigDescriptor
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "sda_reader.cpp"),
                core_ranges=self.cores,
                compile_time_args=reader_ct,
                runtime_args=[],
                common_runtime_args=reader_common,
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0),
                defines=defines,
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "sda_writer.cpp"),
                core_ranges=self.cores,
                compile_time_args=writer_ct,
                runtime_args=[],
                common_runtime_args=writer_common,
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1),
                defines=defines,
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "sda_compute.cpp"),
                core_ranges=self.cores,
                compile_time_args=compute_ct,
                runtime_args=[],
                defines=compute_defines,
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi2,
                    fp32_dest_acc_en=True,
                    math_approx_mode=False,
                ),
            ),
        ]
        io = [q, k_cache, v_cache, pos, page_table, sinks, self.consts, self.out]
        return kernels, cbs, [t for i, t in enumerate(io) if all(t is not u for u in io[:i])]


def output_tensor(mesh_device, n_heads, head_dim):
    """The op's output: [1, 1, heads, head_dim] BF16 tiles in L1 (the o_proj stream reads rows straight from it), the
    padded heads zero."""
    return ttnn.from_torch(
        torch.zeros(1, 1, n_heads, head_dim),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def consts_tensor(mesh_device, dtype=ttnn.bfloat8_b):
    """The op's constants: one all-ones tile in the K / V cache's dtype (L = P . ones), DRAM."""
    return ttnn.from_torch(
        torch.ones(1, 1, TILE, TILE),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def pick_cores(mesh_device, taken, n_kv, slices):
    """n_kv * slices logical worker cores clear of `taken` ((x, y) set) as few rectangles as possible (dispatch writes
    the op's kernel binaries once per core range: a fragmented set made the fused QKV program launch ~15 us late),
    rectangle by rectangle in row-major order (kv head g's slices consecutive)."""
    grid = mesh_device.compute_with_storage_grid_size()
    n = n_kv * slices
    free = {(x, y) for y in range(grid.y) for x in range(grid.x) if (x, y) not in taken}
    assert len(free) >= n, f"{len(free)} free cores, {n} needed"
    out = []
    while len(out) < n:
        need = n - len(out)
        best = None  # (area, -y0, x0, y0, x1, y1): the largest free rectangle, the bottom-most first
        for y0 in range(grid.y):
            for x0 in range(grid.x):
                if (x0, y0) not in free:
                    continue
                x1 = x0
                while x1 + 1 < grid.x and (x1 + 1, y0) in free:
                    x1 += 1
                for xe in range(x0, x1 + 1):
                    ye = y0
                    while ye + 1 < grid.y and all((x, ye + 1) in free for x in range(x0, xe + 1)):
                        ye += 1
                    key = ((xe - x0 + 1) * (ye - y0 + 1), y0, x0, y0, xe, ye)
                    if best is None or key > best:
                        best = key
        _, _, x0, y0, x1, y1 = best
        w = x1 - x0 + 1
        if (y1 - y0 + 1) * w > need:  # trim to whole rows (or one partial row)
            if need >= w:
                y0 = y1 - need // w + 1
            else:
                y0, x1 = y1, x0 + need - 1
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                out.append(ttnn.CoreCoord(x, y))
                free.discard((x, y))
    return out


def max_slices(mesh_device, taken, n_kv, cap=63):
    grid = mesh_device.compute_with_storage_grid_size()
    free = sum(1 for y in range(grid.y) for x in range(grid.x) if (x, y) not in taken)
    return max(1, min(cap, free // n_kv))
