# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Qwen context-parallel causal SDPA as ``ttnn.generic_op`` programs (streaming compute, 16-bit DEST).

``chunked_sdpa_port``: a Python port of the stock SDPA program factory
(ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp, ``create_descriptor``) for exactly the
case the context-parallel prefill runs: ``ttnn.transformer.chunked_scaled_dot_product_attention`` with a paged K / V,
a page table and a per-device ``chunk_start_idx_tensor`` (flexible chunked mode), causal, no mask / sink / MLA /
sliding window / windowed mode, streaming compute (fp32_dest_acc_en False). Same CBs, compile-time args, defines and
runtime args as the stock factory, so the output is bit-identical. ``kernels="stock"`` runs the stock kernel files,
``kernels="local"`` the verbatim copies in kernels/ (``stock_*.cpp``).
"""

from __future__ import annotations

import functools
import math
import os
import random
import struct
from dataclasses import dataclass

import ttnn
from models.demos.llama3_70b_galaxy.tt.custom_ops.cache_key import tensor_key

KERNEL_ROOT = "models/demos/llama3_70b_galaxy/tt/custom_ops/qwen_cp_sdpa/kernels"
STOCK_ROOT = "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels"
STOCK_KERNELS = {
    "reader": f"{STOCK_ROOT}/dataflow/reader_interleaved.cpp",
    "writer": f"{STOCK_ROOT}/dataflow/writer_interleaved.cpp",
    "compute": f"{STOCK_ROOT}/compute/sdpa.cpp",
}
LOCAL_STOCK_KERNELS = {
    "reader": f"{KERNEL_ROOT}/stock_reader.cpp",
    "writer": f"{KERNEL_ROOT}/stock_writer.cpp",
    "compute": f"{KERNEL_ROOT}/stock_compute.cpp",
}
# tt-metal resolves kernel_source paths against its root dir, but compiler_include_paths only against the current
# working directory (tt_metal/impl/kernels/kernel.cpp, resolve_compiler_include_dir), and a vLLM server does not run
# from the repo root. Anchor the include dirs at the same root as the kernel sources: TT_METAL_HOME, else this checkout.
_TT_METAL_ROOT = os.environ.get("TT_METAL_HOME") or os.path.abspath(
    os.path.join(os.path.dirname(__file__), *[".."] * 6)
)
# The local copies include the unchanged stock headers (dataflow_common.hpp, compute_common.hpp, ...) from here.
DATAFLOW_INCLUDES = [os.path.join(_TT_METAL_ROOT, STOCK_ROOT, "dataflow")]
COMPUTE_INCLUDES = [os.path.join(_TT_METAL_ROOT, STOCK_ROOT, "compute")]

TILE = 32
INACTIVE = 0xFFFFFFFF
TILE_BYTES = {ttnn.bfloat16: 2048, ttnn.bfloat8_b: 1088, ttnn.float32: 4096}

_DESCRIPTOR_CACHE: dict = {}
_DESCRIPTOR_CACHE_MAX = 512


def _f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _bf16_bits(x: float) -> int:
    """bfloat16(x) bits as the host's class bfloat16 rounds it (round to nearest even)."""
    u = _f32_bits(x)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) & 0xFFFF


def _accessor(t) -> list[int]:
    return list(ttnn.TensorAccessorArgs(t).get_compile_time_args()) if t is not None else [0, 0]


def _addr(t) -> int:
    return t.buffer_address() if t is not None else 0


def largest_subblock(block_h: int, block_w: int, dst_size: int, max_h: int = 1 << 30) -> tuple[int, int]:
    """detail::determine_largest_subblock_size (sdpa_subblock_utils.hpp)."""
    order = [(2, 4), (4, 2), (1, 8), (8, 1), (1, 7), (7, 1), (2, 3), (3, 2), (1, 6), (6, 1)]
    order += [(1, 5), (5, 1), (2, 2), (1, 4), (4, 1), (1, 3), (3, 1), (1, 2), (2, 1), (1, 1)]
    for h, w in order:
        if h * w > dst_size or h > max_h:
            continue
        if block_h % h or block_w % w:
            continue
        return h, w
    return 1, 1


def valid_granularity(count: int, max_g: int) -> int:
    """detail::find_valid_granularity."""
    g = min(count, max_g)
    while g > 1 and count % g:
        g -= 1
    return g


def streaming_qktv_h(sub_h: int, sub_w: int, dst_size: int, sq_chunk_t: int) -> int:
    """ttnn::transformer::sdpa::streaming_qktv_h (sdpa_streaming_qktv.hpp)."""
    return 2 if (sub_h == 1 and 2 * sub_w <= dst_size and sq_chunk_t >= 2) else sub_h


@dataclass(frozen=True)
class SdpaGeometry:
    """The derived sizes of create_descriptor for one causal, paged, flexible-chunked, streaming call."""

    B: int
    NQH: int
    NKH: int
    NVH: int
    Sq: int
    DH: int
    q_chunk: int
    k_chunk: int
    block_size: int
    max_blocks: int
    num_cores: int

    @property
    def Sk(self) -> int:  # flexible chunked: Sq + max prefix from the page table
        return self.Sq + self.max_blocks * self.block_size

    @property
    def padded_Sq(self) -> int:
        return math.ceil(self.Sq / self.q_chunk) * self.q_chunk

    @property
    def padded_Sk(self) -> int:
        return math.ceil(self.Sk / self.k_chunk) * self.k_chunk

    @property
    def Sqt(self) -> int:
        return self.padded_Sq // TILE

    @property
    def Skt(self) -> int:
        return self.padded_Sk // TILE

    @property
    def DHt(self) -> int:
        return self.DH // TILE

    @property
    def valid_Sqt(self) -> int:
        return math.ceil(self.Sq / TILE)

    @property
    def valid_Skt(self) -> int:
        return math.ceil(self.Sk / TILE)

    @property
    def Sq_chunk_t(self) -> int:
        return self.q_chunk // TILE

    @property
    def Sk_chunk_t(self) -> int:
        return self.k_chunk // TILE

    @property
    def q_num_chunks(self) -> int:
        return self.padded_Sq // self.q_chunk

    @property
    def k_num_chunks(self) -> int:
        return self.padded_Sk // self.k_chunk

    @property
    def block_size_t(self) -> int:
        return self.block_size // TILE

    @property
    def page_table_stick_size(self) -> int:
        return self.max_blocks * 4


def _geometry(q, k, page_table, q_chunk, k_chunk, num_cores) -> SdpaGeometry:
    B, NQH, Sq, DH = (int(x) for x in q.shape)
    return SdpaGeometry(
        B=B,
        NQH=NQH,
        NKH=int(k.shape[1]),
        NVH=int(k.shape[1]),
        Sq=Sq,
        DH=DH,
        q_chunk=q_chunk,
        k_chunk=k_chunk,
        block_size=int(k.shape[2]),
        max_blocks=int(page_table.padded_shape[1]),
        num_cores=num_cores,
    )


def _global_q_ranges(g: SdpaGeometry) -> list[tuple[int, int]]:
    """Per core (global_q_start, global_q_count): causal pair distribution of create_descriptor."""
    total = g.B * g.NQH * g.q_num_chunks
    if g.q_num_chunks % 2 == 0:
        pairs = total // 2
        base, extra_cores, extra = (pairs // g.num_cores) * 2, pairs % g.num_cores, 2
    else:
        base, extra_cores, extra = total // g.num_cores, total % g.num_cores, 1
    out = []
    for i in range(g.num_cores):
        start = i * base + min(i, extra_cores) * extra
        count = base + (extra if i < extra_cores else 0)
        if start >= total:
            start, count = total, 0
        elif start + count > total:
            count = total - start
        out.append((start, count))
    return out


def _cb(idx, tiles_or_pages, page_size, dtype, cores) -> ttnn.CBDescriptor:
    return ttnn.CBDescriptor(
        total_size=tiles_or_pages * page_size,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dtype, page_size=page_size)],
    )


def _build_stock_port(q, k, v, page_table, cs_tensor, out, *, scale, ckc, exp_approx, grid, g, kernels):
    """create_descriptor for the chunked (flexible), causal, streaming case. Returns a ProgramDescriptor."""
    gx, gy = grid
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})
    fp32_dest = bool(ckc.fp32_dest_acc_en)
    assert not fp32_dest, "streaming compute only (fp32_dest_acc_en=False)"
    dst_size = 4 if fp32_dest else 8
    Sq_chunk_t, Sk_chunk_t, DHt = g.Sq_chunk_t, g.Sk_chunk_t, g.DHt
    vDHt = DHt
    ranges = _global_q_ranges(g)
    max_per_core = max(c for _, c in ranges)
    q_buffer_factor = 2 if max_per_core > 1 else 1

    qk_h, qk_w = largest_subblock(Sq_chunk_t, Sk_chunk_t, dst_size)
    out_h, out_w = largest_subblock(Sq_chunk_t, vDHt, dst_size, max_h=2)
    assert Sq_chunk_t % out_h == 0
    out0_t = 2 * streaming_qktv_h(out_h, out_w, dst_size, Sq_chunk_t) * vDHt

    stats_g = valid_granularity(Sq_chunk_t, dst_size)
    sub_exp_g = valid_granularity(Sk_chunk_t, dst_size)
    dht_g = min(DHt, vDHt, dst_size)  # compute_dht_granularity
    while dht_g > 1 and (DHt % dht_g or vDHt % dht_g):
        dht_g -= 1
    reduce_g = valid_granularity(Sq_chunk_t, dst_size // 2)
    identity_packed = (_bf16_bits(1.0) << 16) | _bf16_bits(1.0)
    scale_packed = _f32_bits(scale)

    # ---- CBs, allocated in create_descriptor's order ----
    bf8, bf16, i32 = ttnn.bfloat8_b, ttnn.bfloat16, ttnn.int32
    q_df, k_df, v_df, out_df = q.dtype, k.dtype, v.dtype, out.dtype
    cbs = []
    ids = {}

    def alloc(name, n, page, dtype):
        ids[name] = len(cbs)
        cbs.append(_cb(ids[name], n, page, dtype, cores))

    alloc("q_in", Sq_chunk_t * DHt * q_buffer_factor, TILE_BYTES[q_df], q_df)
    alloc("k_in", Sk_chunk_t * DHt * 2, TILE_BYTES[k_df], k_df)
    alloc("v_in", Sk_chunk_t * vDHt * 2, TILE_BYTES[v_df], v_df)
    alloc("mask_in", 2, TILE_BYTES[bf16], bf16)  # lightweight causal: neginf + diagonal
    alloc("identity_scale_in", 1, TILE_BYTES[bf16], bf16)
    alloc("col_identity", 1, TILE_BYTES[bf16], bf16)
    alloc("page_table", 1, g.page_table_stick_size, i32)
    alloc("chunk_start_idx_compute", 1, 32, i32)
    alloc("chunk_start_idx_writer", 1, 32, i32)
    alloc("recip_scratch", 1, TILE_BYTES[bf16], bf16)
    alloc("qk_im", Sq_chunk_t * Sk_chunk_t, TILE_BYTES[bf16], bf16)
    for name in ("out_im_A", "out_im_B"):
        alloc(name, Sq_chunk_t * vDHt, TILE_BYTES[bf16], bf16)
    for name in ("max_A", "max_B", "sum_A", "sum_B", "exp_max_diff"):
        alloc(name, Sq_chunk_t, TILE_BYTES[bf16], bf16)
    alloc("out", out0_t, TILE_BYTES[out_df], out_df)
    # windowed fallbacks point at q_in; the sink is inactive
    win = ids["q_in"]
    reader_cb = [ids["q_in"], ids["k_in"], ids["v_in"], ids["mask_in"], INACTIVE, ids["page_table"]]
    reader_cb += [ids["chunk_start_idx_compute"], ids["chunk_start_idx_writer"], win, win]
    writer_cb = [ids["mask_in"], ids["identity_scale_in"], ids["col_identity"], ids["chunk_start_idx_writer"]]
    writer_cb += [ids["out"], win, win]
    compute_cb = [ids["q_in"], ids["k_in"], ids["v_in"], ids["mask_in"], INACTIVE, ids["identity_scale_in"]]
    compute_cb += [ids["col_identity"], ids["chunk_start_idx_compute"], ids["recip_scratch"], ids["out"]]
    compute_cb += [ids["qk_im"], ids["out_im_A"], ids["out_im_B"], ids["max_A"], ids["max_B"], ids["sum_A"]]
    compute_cb += [ids["sum_B"], ids["exp_max_diff"], win]

    # ---- compile-time args ----
    reader_ct = [g.B, g.NQH, g.NKH, g.NVH, g.Skt, g.valid_Sqt, g.valid_Skt, DHt, vDHt, Sq_chunk_t, g.q_num_chunks]
    reader_ct += [Sk_chunk_t, g.k_num_chunks, g.num_cores, 1, 0, 0, 0, 0, 1, g.block_size_t, g.page_table_stick_size]
    reader_ct += [0, 0, 0, qk_h, 0, 1]  # sink, mla, mla overlap, qk_out_subblock_h, sliding, streaming
    reader_ct += [0, 0, 0, 0, 1, 0]  # chain semaphores x3, mcast, zigzag, windowed
    for t in (q, k, v, None, page_table, None, cs_tensor, None, None):
        reader_ct += _accessor(t)
    reader_ct += reader_cb

    writer_ct = [g.B, g.NQH, g.valid_Sqt, g.Sk, vDHt, Sq_chunk_t, g.q_num_chunks, Sk_chunk_t, g.k_num_chunks]
    writer_ct += [identity_packed, g.num_cores, 1, 0, 0, 1, 0, 1, 1, out_h, 0, 1, 0, 0]
    for t in (out, None, None):
        writer_ct += _accessor(t)
    writer_ct += writer_cb

    compute_ct = [g.Skt, DHt, vDHt, Sq_chunk_t, g.q_num_chunks, Sk_chunk_t, g.k_num_chunks, DHt, qk_w, qk_h]
    compute_ct += [Sq_chunk_t // qk_h, Sk_chunk_t // qk_w, Sk_chunk_t, out_w, out_h, Sq_chunk_t // out_h]
    compute_ct += [vDHt // out_w, 1, 0, 0, 1, scale_packed, 0, 0, 1, g.valid_Skt, 0, 1, 0]
    compute_ct += compute_cb
    compute_ct += _accessor(out)

    defines = [
        ("STATS_GRANULARITY", str(stats_g)),
        ("SUB_EXP_GRANULARITY", str(sub_exp_g)),
        ("DHT_GRANULARITY", str(dht_g)),
        ("REDUCE_GRANULARITY", str(reduce_g)),
        ("EXP_APPROX_MODE", str(int(bool(exp_approx)))),
    ]

    # ---- runtime args ----
    reader_rt, writer_rt, compute_rt = [], [], []
    for i, (gs, gc) in enumerate(ranges):
        xy = (i % gx, i // gx)
        reader_rt.append(
            (xy, [_addr(q), _addr(k), _addr(v), 0, _addr(page_table), 0, _addr(cs_tensor), 1, 0, 0, gs, gc, 0, 0, 0, 0])
        )
        writer_rt.append((xy, [_addr(out), 1, 1, 0, 0, 0, 0, gs, gc, 0, 0, 0, 0]))
        compute_rt.append((xy, [1, 1, 0, 0, gs, gc]))

    paths = STOCK_KERNELS if kernels == "stock" else LOCAL_STOCK_KERNELS
    local = kernels != "stock"
    kw_df = dict(compiler_include_paths=DATAFLOW_INCLUDES) if local else {}
    kw_cp = dict(compiler_include_paths=COMPUTE_INCLUDES) if local else {}
    kdescs = [
        ttnn.KernelDescriptor(
            kernel_source=paths["reader"],
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=reader_ct,
            runtime_args=reader_rt,
            defines=defines,
            config=ttnn.ReaderConfigDescriptor(),
            **kw_df,
        ),
        ttnn.KernelDescriptor(
            kernel_source=paths["writer"],
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=writer_ct,
            runtime_args=writer_rt,
            defines=defines,
            config=ttnn.WriterConfigDescriptor(),
            **kw_df,
        ),
        ttnn.KernelDescriptor(
            kernel_source=paths["compute"],
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=compute_ct,
            runtime_args=compute_rt,
            defines=defines,
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ckc.math_fidelity,
                fp32_dest_acc_en=fp32_dest,
                dst_full_sync_en=bool(ckc.dst_full_sync_en),
                math_approx_mode=bool(ckc.math_approx_mode),
            ),
            **kw_cp,
        ),
    ]
    return ttnn.ProgramDescriptor(kernels=kdescs, semaphores=[], cbs=cbs)


def chunked_sdpa_port(
    q: ttnn.Tensor,
    k: ttnn.Tensor,
    v: ttnn.Tensor,
    page_table: ttnn.Tensor,
    chunk_start_idx_tensor: ttnn.Tensor,
    *,
    scale: float,
    compute_kernel_config,
    program_config: ttnn.SDPAProgramConfig,
    kernels: str = "stock",
) -> ttnn.Tensor:
    """Same result as ttnn.transformer.chunked_scaled_dot_product_attention(q, k, v, page_table,
    chunk_start_idx_tensor=..., scale=scale, compute_kernel_config=..., program_config=...), bit for bit.

    q [B, NQH, Sq, DH] bf8 DRAM interleaved; k / v paged [blocks, NKH, block, DH]; page_table [B, max_blocks] int32
    row-major; chunk_start_idx_tensor one int32 per device. One program for the whole mesh (the start is read at run
    time). Returns a fresh [B, NQH, Sq, DH] tensor of q's dtype in DRAM.
    """
    if kernels not in ("stock", "local"):
        raise ValueError(f"kernels must be 'stock' or 'local', got {kernels}")
    if compute_kernel_config.fp32_dest_acc_en:
        raise ValueError("only the streaming compute (fp32_dest_acc_en=False) is ported")
    device = q.device()
    gsz = program_config.compute_with_storage_grid_size
    grid = (int(gsz.x), int(gsz.y))
    g = _geometry(
        q, k, page_table, int(program_config.q_chunk_size), int(program_config.k_chunk_size), grid[0] * grid[1]
    )
    exp_approx = program_config.exp_approx_mode
    exp_approx = True if exp_approx is None else bool(exp_approx)
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([g.B, g.NQH, g.Sq, g.DH]), q.dtype, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    key = (
        "port",
        kernels,
        tuple(device.shape),
        grid,
        g,
        _f32_bits(scale),
        exp_approx,
        str(compute_kernel_config.math_fidelity),
        bool(compute_kernel_config.math_approx_mode),
        bool(compute_kernel_config.dst_full_sync_en),
        tuple(tensor_key(t) for t in (q, k, v, page_table, chunk_start_idx_tensor, out)),
    )
    mesh_program = _DESCRIPTOR_CACHE.get(key)
    if mesh_program is None:
        program = _build_stock_port(
            q,
            k,
            v,
            page_table,
            chunk_start_idx_tensor,
            out,
            scale=scale,
            ckc=compute_kernel_config,
            exp_approx=exp_approx,
            grid=grid,
            g=g,
            kernels=kernels,
        )
        rows, cols = list(device.shape)
        mesh_program = ttnn.MeshProgramDescriptor()
        mesh_program[
            ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(rows - 1, cols - 1))
        ] = program
        if len(_DESCRIPTOR_CACHE) >= _DESCRIPTOR_CACHE_MAX:
            _DESCRIPTOR_CACHE.clear()
        _DESCRIPTOR_CACHE[key] = mesh_program
    ttnn.generic_op([q, k, v, page_table, chunk_start_idx_tensor, out], mesh_program)
    return out


# ---------------------------------------------------------------------------------------------------------------
# Both owned slices of a mesh column in one program (kernels/cp_*.cpp).
# ---------------------------------------------------------------------------------------------------------------

CP_KERNELS = {
    "reader": f"{KERNEL_ROOT}/cp_reader.cpp",
    "writer": f"{KERNEL_ROOT}/cp_writer.cpp",
    "compute": f"{KERNEL_ROOT}/cp_compute.cpp",
}
ITEM_OVERHEAD = 0.3  # per-Q-chunk fixed cost in K-chunk units (Q read, normalize, drain), for the schedule only


def column_slices(col: int, cols: int) -> tuple[int, int]:
    return col, 2 * cols - 1 - col


def item_cost(g: int, sq_t: int, sk_t: int) -> float:
    """K-chunk iterations of global causal Q chunk g, the diagonal chunk counted by its active width."""
    end_t = (g + 1) * sq_t
    n = -(-end_t // sk_t)
    return (n - 1) + (end_t - (n - 1) * sk_t) / sk_t + ITEM_OVERHEAD


@functools.lru_cache(maxsize=64)
def schedule(
    offsets: tuple[int, int], n_heads: int, q_num_chunks: int, sq_t: int, sk_t: int, num_cores: int, iters: int = 100000
) -> tuple[tuple[int, ...], ...]:
    """Per core, the encoded items ((p << 16) | (head << 8) | q_chunk) of both slices, balanced by item_cost:
    longest-processing-time first, then a seeded simulated-annealing pass of moves / swaps minimising the busiest
    core. Deterministic (fixed seed); cached per geometry."""
    codes, cost = [], []
    for p, off in enumerate(offsets):
        for h in range(n_heads):
            for j in range(q_num_chunks):
                codes.append((p << 16) | (h << 8) | j)
                cost.append(item_cost(off + j, sq_t, sk_t))
    n = len(codes)
    assign = [0] * n
    load = [0.0] * num_cores
    for i in sorted(range(n), key=lambda i: (-cost[i], codes[i])):
        b = min(range(num_cores), key=lambda x: (load[x], x))
        assign[i] = b
        load[b] += cost[i]

    def objective():
        m = max(load)
        return (round(m, 9), sum(1 for x in load if x > m - 1e-9))

    best, best_assign = objective(), list(assign)
    rnd = random.Random(0)
    for t in range(iters):
        temp = 2.0 * (1 - t / iters) + 1e-3
        i = rnd.randrange(n)
        a = assign[i]
        if rnd.random() < 0.5:
            b, d, j = rnd.randrange(num_cores), cost[i], None
        else:
            j = rnd.randrange(n)
            b, d = assign[j], cost[i] - cost[j]
        if b == a:
            continue
        old, new = max(load[a], load[b]), max(load[a] - d, load[b] + d)
        if new <= old or rnd.random() < math.exp(-(new - old) / temp):
            load[a] -= d
            load[b] += d
            assign[i] = b
            if j is not None:
                assign[j] = a
            o = objective()
            if o < best:
                best, best_assign = o, list(assign)
    bins = [[] for _ in range(num_cores)]
    for i in sorted(range(n), key=lambda i: (-cost[i], codes[i])):
        bins[best_assign[i]].append(codes[i])
    return tuple(tuple(b) for b in bins)


def extra_defines() -> tuple[tuple[str, str], ...]:
    """QWEN_CP_SDPA_DEFINES="NAME[=VALUE],...": extra compile-time defines for the local kernels (cp_step.hpp
    measurement knobs; unset in the model)."""
    out = []
    for item in os.environ.get("QWEN_CP_SDPA_DEFINES", "").split(","):
        item = item.strip()
        if item:
            name, _, value = item.partition("=")
            out.append((name, value or "1"))
    return tuple(out)


def _build_cp(q0, q1, k, v, page_table, out, *, col, cols, scale, ckc, exp_approx, grid, g, concat_heads, extra=()):
    gx, gy = grid
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})
    dst_size = 8
    Sq_chunk_t, Sk_chunk_t, DHt = g.Sq_chunk_t, g.Sk_chunk_t, g.DHt
    vDHt = DHt
    offsets = [s * g.Sq // g.q_chunk for s in column_slices(col, cols)]
    bins = schedule(tuple(offsets), g.NQH, g.q_num_chunks, Sq_chunk_t, Sk_chunk_t, g.num_cores)
    q_buffer_factor = 2 if max(len(b) for b in bins) > 1 else 1

    qk_h, qk_w = largest_subblock(Sq_chunk_t, Sk_chunk_t, dst_size)
    out_h, out_w = largest_subblock(Sq_chunk_t, vDHt, dst_size, max_h=2)
    assert Sq_chunk_t % out_h == 0
    out0_t = 2 * streaming_qktv_h(out_h, out_w, dst_size, Sq_chunk_t) * vDHt
    dht_g = min(DHt, vDHt, dst_size)
    while dht_g > 1 and (DHt % dht_g or vDHt % dht_g):
        dht_g -= 1
    identity_packed = (_bf16_bits(1.0) << 16) | _bf16_bits(1.0)

    bf16, i32 = ttnn.bfloat16, ttnn.int32
    cbs, ids = [], {}

    def alloc(name, n, page, dtype):
        ids[name] = len(cbs)
        cbs.append(_cb(ids[name], n, page, dtype, cores))

    alloc("q_in", Sq_chunk_t * DHt * q_buffer_factor, TILE_BYTES[q0.dtype], q0.dtype)
    alloc("k_in", Sk_chunk_t * DHt * 2, TILE_BYTES[k.dtype], k.dtype)
    alloc("v_in", Sk_chunk_t * vDHt * 2, TILE_BYTES[v.dtype], v.dtype)
    alloc("mask_in", 2, TILE_BYTES[bf16], bf16)
    alloc("identity_scale_in", 1, TILE_BYTES[bf16], bf16)
    alloc("col_identity", 1, TILE_BYTES[bf16], bf16)
    alloc("page_table", 1, g.page_table_stick_size, i32)
    alloc("recip_scratch", 1, TILE_BYTES[bf16], bf16)
    alloc("qk_im", Sq_chunk_t * Sk_chunk_t, TILE_BYTES[bf16], bf16)
    for name in ("out_im_A", "out_im_B"):
        alloc(name, Sq_chunk_t * vDHt, TILE_BYTES[bf16], bf16)
    for name in ("max_A", "max_B", "sum_A", "sum_B", "exp_max_diff"):
        alloc(name, Sq_chunk_t, TILE_BYTES[bf16], bf16)
    alloc("out", out0_t, TILE_BYTES[out.dtype], out.dtype)

    reader_ct = [g.B, g.NQH, g.NKH, g.NVH, g.Skt, g.valid_Sqt, g.valid_Skt, DHt, vDHt, Sq_chunk_t, Sk_chunk_t]
    reader_ct += [g.num_cores, g.block_size_t, g.page_table_stick_size, qk_h]
    for t in (q0, k, v, page_table):
        reader_ct += _accessor(t)
    reader_ct += [ids["q_in"], ids["k_in"], ids["v_in"], ids["page_table"]]

    writer_ct = [g.B, g.NQH, g.valid_Sqt, 2 * g.valid_Sqt, vDHt, Sq_chunk_t, identity_packed, g.num_cores, out_h]
    writer_ct += [int(concat_heads)] + _accessor(out)
    writer_ct += [ids["mask_in"], ids["identity_scale_in"], ids["col_identity"], ids["out"]]

    compute_ct = [DHt, vDHt, Sq_chunk_t, Sk_chunk_t, g.k_num_chunks, qk_w, qk_h, out_w, out_h, _f32_bits(scale)]
    compute_ct += [g.valid_Skt]
    compute_ct += [ids[n] for n in ("q_in", "k_in", "v_in", "mask_in", "identity_scale_in", "col_identity")]
    compute_ct += [ids[n] for n in ("recip_scratch", "out", "qk_im", "out_im_A", "out_im_B", "max_A", "max_B")]
    compute_ct += [ids[n] for n in ("sum_A", "sum_B", "exp_max_diff")]

    defines = [
        ("STATS_GRANULARITY", str(valid_granularity(Sq_chunk_t, dst_size))),
        ("SUB_EXP_GRANULARITY", str(valid_granularity(Sk_chunk_t, dst_size))),
        ("DHT_GRANULARITY", str(dht_g)),
        ("REDUCE_GRANULARITY", str(valid_granularity(Sq_chunk_t, dst_size // 2))),
        ("EXP_APPROX_MODE", str(int(bool(exp_approx)))),
    ] + list(extra)

    reader_rt, writer_rt, compute_rt = [], [], []
    head = [_addr(q0), _addr(q1), _addr(k), _addr(v), _addr(page_table)] + offsets
    for i, items in enumerate(bins):
        xy = (i % gx, i // gx)
        reader_rt.append((xy, head + [len(items)] + list(items)))
        writer_rt.append((xy, [_addr(out), len(items)] + list(items)))
        compute_rt.append((xy, offsets + [len(items)] + list(items)))

    def kd(kind, ct, rt, config, inc):
        return ttnn.KernelDescriptor(
            kernel_source=CP_KERNELS[kind],
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores,
            compile_time_args=ct,
            runtime_args=rt,
            defines=defines,
            config=config,
            compiler_include_paths=inc,
        )

    kdescs = [
        kd("reader", reader_ct, reader_rt, ttnn.ReaderConfigDescriptor(), DATAFLOW_INCLUDES),
        kd("writer", writer_ct, writer_rt, ttnn.WriterConfigDescriptor(), DATAFLOW_INCLUDES),
        kd(
            "compute",
            compute_ct,
            compute_rt,
            ttnn.ComputeConfigDescriptor(
                math_fidelity=ckc.math_fidelity,
                fp32_dest_acc_en=False,
                dst_full_sync_en=bool(ckc.dst_full_sync_en),
                math_approx_mode=bool(ckc.math_approx_mode),
            ),
            COMPUTE_INCLUDES,
        ),
    ]
    return ttnn.ProgramDescriptor(kernels=kdescs, semaphores=[], cbs=cbs)


def qwen_cp_sdpa(
    q0: ttnn.Tensor,
    q1: ttnn.Tensor,
    k: ttnn.Tensor,
    v: ttnn.Tensor,
    page_table: ttnn.Tensor,
    *,
    scale: float,
    compute_kernel_config,
    program_config: ttnn.SDPAProgramConfig,
    concat_heads: bool = True,
) -> ttnn.Tensor:
    """Causal attention of mesh column c's two ring slices (c and 2 * cols - 1 - c) in one program per device.

    q0 / q1: [1, NQH, sl, DH] bf8, the column's local slices 0 / 1 (qwen_q_prep q_out1). k / v: paged
    [S/block, 1, block, DH] read through page_table [1, S/block] int32 (the identity table of the scratch K / V).
    Every Q chunk is computed as the stock chunked_scaled_dot_product_attention call of its slice computes it (same
    per-chunk math, same K-chunk order); the chunks of both slices are spread over all grid cores by expected cost.
    Returns, in the column's local order [slice c ; slice 2 * cols - 1 - c]: [1, 1, 2 * sl, NQH * DH] with
    concat_heads (what concat + nlp_concat_heads of the stock pair give), else [1, NQH, 2 * sl, DH].
    """
    if compute_kernel_config.fp32_dest_acc_en:
        raise ValueError("only the streaming compute (fp32_dest_acc_en=False) is supported")
    if tuple(q0.shape) != tuple(q1.shape) or q0.dtype != q1.dtype:
        raise ValueError(f"q0 / q1 must match: {tuple(q0.shape)} {q0.dtype} vs {tuple(q1.shape)} {q1.dtype}")
    # The reader takes q1's address with q0's tensor accessor, so q1 must share q0's layout and memory config.
    for name, t in (("q0", q0), ("q1", q1)):
        if t.layout != ttnn.TILE_LAYOUT or t.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
            raise ValueError(f"{name} must be TILE layout DRAM interleaved, got {t.layout} {t.memory_config()}")
    B, NQH, sl, DH = (int(x) for x in q0.shape)
    if B != 1 or NQH > 255:
        raise ValueError(f"q must be [1, NQH <= 255, sl, DH], got {tuple(q0.shape)}")
    device = q0.device()
    rows, cols = list(device.shape)
    gsz = program_config.compute_with_storage_grid_size
    grid = (int(gsz.x), int(gsz.y))
    g = _geometry(
        q0, k, page_table, int(program_config.q_chunk_size), int(program_config.k_chunk_size), grid[0] * grid[1]
    )
    if g.q_num_chunks > 255 or sl % g.q_chunk:
        raise ValueError(f"slice length {sl} must be a multiple of q_chunk {g.q_chunk} with <= 255 chunks")
    if g.NQH % g.NKH:
        raise ValueError(f"query heads {g.NQH} must be a multiple of the KV heads {g.NKH}")
    if sl % g.k_chunk:
        raise ValueError(f"slice length {sl} must be a multiple of k_chunk {g.k_chunk}")
    if 2 * cols * sl > g.max_blocks * g.block_size:
        raise ValueError("page table shorter than the 2 * cols slices")
    exp_approx = program_config.exp_approx_mode
    exp_approx = True if exp_approx is None else bool(exp_approx)
    shape = [1, 1, 2 * sl, NQH * DH] if concat_heads else [1, NQH, 2 * sl, DH]
    out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), q0.dtype, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    extra = extra_defines()
    key = (
        "cp",
        extra,
        concat_heads,
        tuple(device.shape),
        grid,
        g,
        _f32_bits(scale),
        exp_approx,
        str(compute_kernel_config.math_fidelity),
        bool(compute_kernel_config.math_approx_mode),
        bool(compute_kernel_config.dst_full_sync_en),
        tuple(tensor_key(t) for t in (q0, q1, k, v, page_table, out)),
    )
    mesh_program = _DESCRIPTOR_CACHE.get(key)
    if mesh_program is None:
        mesh_program = ttnn.MeshProgramDescriptor()
        for col in range(cols):
            program = _build_cp(
                q0,
                q1,
                k,
                v,
                page_table,
                out,
                col=col,
                cols=cols,
                scale=scale,
                ckc=compute_kernel_config,
                exp_approx=exp_approx,
                grid=grid,
                g=g,
                concat_heads=concat_heads,
                extra=extra,
            )
            mesh_program[
                ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, col), ttnn.MeshCoordinate(rows - 1, col))
            ] = program
        if len(_DESCRIPTOR_CACHE) >= _DESCRIPTOR_CACHE_MAX:
            _DESCRIPTOR_CACHE.clear()
        _DESCRIPTOR_CACHE[key] = mesh_program
    ttnn.generic_op([q0, q1, k, v, page_table, out], mesh_program)
    return out
