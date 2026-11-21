# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PB.AR (OWNER 2026-11-21): an ANALYSIS, probes only (no model or fork change). The roofline of the compressed layers'
sparse attention over the top-k rows at the shipped settings, on the whole 8x4 mesh (mesh only, no model load; every
chip runs the same replicated work, so the profiler's slowest chip and the per-chip spread both show).

The model's call (tt/packed_onecall.sdpa_lanes, PACKED_GLUED + chip-aligned lanes; single requests: TtSparseAttention
._sdpa ROW_BFP4 + kv_window): q [1, 64 heads, S, 512] bf16 ROW_MAJOR (S = chunk / 32 = 256 at 8192, 160 at 5120), the
history as 320 B bfloat4_b row records (HISTORY_FORMAT=bfp4), the window as the chip's bf16 region [carry 128 | S]
(WINDOW_KV_FORMAT=bf16_tiles), index [1, 1, S, 128 + 512] uint32, k_chunk 128, the per-head sink, scale 1 (512^-0.5 is
folded into q), attn_compute_config (HiFi2, fp32 DEST, approx off), max_grid 11 x 10.

    (a) plain matmuls of the same FLOPs at the same fidelity and formats: the math-utilization baseline. bmm = one
        [64 x 512] x [512 x 640] and one [64 x 640] x [640 x 512] a query (in1 a batch of its own: the sdpa's shapes),
        reuse = the same in0 as one [B x 64, K] matrix against ONE in1 with the same per-core blocks (per_core_M 2),
        mcast = that matmul with a 2D multicast config (the large-M peak); in1 bfloat4_b (the top-k records' format)
        and bf16 (the window's).
    (b) ttnn.bringup.sparse_sdpa alone at the model's shapes, synthetic top-k ids: random over a 1M slot's ratio-2
        history (524288 rows), clustered (16 runs of 32 consecutive rows), shared (every query the same 512 rows),
        a 4-lane chip-aligned call (the 4 x 2k path: kv_lanes + kv_lane_map_runtime, 4 distinct histories), a small
        history (32768 rows), the window-only layers' call (BF16 region, 128 keys), S 110 / 160 / 220 / 256 and top-k
        128 / 256 / 512 (the per-query and per-k-chunk cost).
    (c) a probe kernel (ttnn.generic_op) that only gathers what one sdpa call reads (per query its index row, the 128
        bf16 window rows of 1024 B, the 512 records of 320 B, k_chunk 128 at a time, a barrier a chunk) on the same 110
        cores, both data-movement RISCs (half the rows each, as the sdpa's dual-NoC split) or one: the DRAM read rate of
        scattered 320 B rows.

Device time per call (real-time profiler, slowest chip; per-chip spread from kernel_ns_dev), ATTN_ITERS calls a case.
Metrics ar_* (all > 0) and a repeat-determinism check of the main sdpa case (bit-identical outputs). Summary JSON:
ATTN_OUT (default /tmp/pb_ar_attention_roofline.json).

    python -m models.demos.common.bringup.resident submit --spec $BRINGUP_SPEC --cmd "<the gate's env>
        scripts/run_safe_pytest.sh --run-all --no-precompile
        models/demos/deepseek_v41_flash_d_p/tests/perf/test_attention_roofline_probe.py"
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest
import torch

from models.demos.common.bringup.testing.harness import mesh_parametrize

BATCH_SHARED = True
BATCH_LAYERS = "executed"
BATCH_ITEM_VARS = ("ATTN_ITERS", "ATTN_OUT", "ATTN_CASES")

ITERS = int(os.environ.get("ATTN_ITERS", "10") or 10)
OUT = os.environ.get("ATTN_OUT", "/tmp/pb_ar_attention_roofline.json")
CASES = [c for c in os.environ.get("ATTN_CASES", "").split(",") if c]  # (empty: every case)
H, D, WIN, KC, TOPK = 64, 512, 128, 128, 512
REC = 320  # (a bfloat4_b row record: 32 exponents twice + 256 nibble bytes)
T_1M = 524288  # (a 1M slot's ratio-2 history: the KV sources 2 / 8 / 14 that layers 2-19 read)
T_SMALL = 32768
BASE = 32768  # (rows encoded on the host; the long tables are device concats of it)
GRID = (11, 10)  # (MAX_WORKER_CORES 110 on the Galaxy: max_grid 11 x 10, rule 18)

_spec = importlib.util.spec_from_file_location(
    "ckv_bfp4_reference",
    Path(__file__).resolve().parents[5] / "ttnn" / "ttnn" / "bringup" / "ckv_bfp4_ttnn" / "tests" / "reference.py",
)
ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ref)


def _want(tag: str) -> bool:
    return not CASES or any(tag.startswith(c) for c in CASES)


# ------------------------------------------------------------------------------------------------ the gather probe
_GATHER = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
constexpr auto kv_args = TensorAccessorArgs<0>();
constexpr auto ix_args = TensorAccessorArgs<kv_args.next_compile_time_args_offset()>();
constexpr auto w_args = TensorAccessorArgs<ix_args.next_compile_time_args_offset()>();
void kernel_main() {
    const uint32_t kv = get_arg_val<uint32_t>(0);
    const uint32_t ix = get_arg_val<uint32_t>(1);
    const uint32_t wn = get_arg_val<uint32_t>(2);
    const uint32_t q0 = get_arg_val<uint32_t>(3);
    const uint32_t q1 = get_arg_val<uint32_t>(4);
    const auto KV = TensorAccessor(kv_args, kv, REC);
    const auto IX = TensorAccessor(ix_args, ix, IDXP);
    const auto W = TensorAccessor(w_args, wn, WROW);
    const uint32_t idx_l1 = (get_write_ptr(PART) + 63) & ~63u;
    const uint32_t ring = idx_l1 + IDXP_AL;
    volatile tt_l1_ptr uint32_t* idx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_l1);
    constexpr uint32_t lo = KC * PART / PARTS, hi = KC * (PART + 1) / PARTS;
    for (uint32_t q = q0; q < q1; ++q) {
        noc_async_read(IX.get_noc_addr(q), idx_l1, IDXP);
        noc_async_read_barrier();
#if DO_WIN
        for (uint32_t j = lo; j < hi; ++j) {
            noc_async_read(W.get_noc_addr(idx[j]), ring + j * WROW, WROW);
        }
        noc_async_read_barrier();
#endif
#if DO_TOPK
        for (uint32_t c = 0; c < NTOP / KC; ++c) {
            for (uint32_t j = lo; j < hi; ++j) {
                noc_async_read(KV.get_noc_addr(idx[WIN + c * KC + j]), ring + j * REC, REC);
            }
            noc_async_read_barrier();
        }
#endif
    }
}
"""


def _gather_op(ttnn, mesh, kv, index, window, S: int, parts: int, do_win: bool, do_topk: bool):
    """The gather-only program on GRID's cores, queries split as the sdpa splits them (S / cores, the first S % cores
    one more)."""
    gx, gy = GRID
    cores = [ttnn.CoreCoord(x, y) for y in range(gy) for x in range(gx)]
    n = len(cores)
    base, extra = S // n, S % n
    spans, q = [], 0
    for i in range(n):
        c = base + (1 if i < extra else 0)
        spans.append((q, q + c))
        q += c
    idxp = int(index.shape[-1]) * 4
    idxp_al = -(-idxp // 64) * 64
    ct = list(ttnn.TensorAccessorArgs(kv).get_compile_time_args())
    ct += list(ttnn.TensorAccessorArgs(index).get_compile_time_args())
    ct += list(ttnn.TensorAccessorArgs(window).get_compile_time_args())
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])
    kernels, cbs = [], []
    for part in range(parts):
        src = (
            f"#define REC {REC}u\n#define IDXP {idxp}u\n#define IDXP_AL {idxp_al}u\n#define WROW {D * 2}u\n"
            f"#define WIN {WIN}u\n#define KC {KC}u\n#define NTOP {int(index.shape[-1]) - WIN}u\n"
            f"#define PART {part}u\n#define PARTS {parts}u\n#define DO_WIN {int(do_win)}\n#define DO_TOPK {int(do_topk)}\n"
        ) + _GATHER
        rt = ttnn.RuntimeArgs()
        for cc, (a, b) in zip(cores, spans):
            rt[cc.x][cc.y] = [int(kv.buffer_address()), int(index.buffer_address()), int(window.buffer_address()), a, b]
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=src,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=crs,
                compile_time_args=ct,
                runtime_args=rt,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_0 if part == 0 else ttnn.DataMovementProcessor.RISCV_1,
                    noc=ttnn.NOC.RISCV_0_default if part == 0 else ttnn.NOC.RISCV_1_default,
                ),
            )
        )
        size = 64 + idxp_al + KC * D * 2
        cbs.append(
            ttnn.CBDescriptor(
                total_size=size, core_ranges=crs, format_descriptors=[ttnn.CBFormatDescriptor(part, ttnn.uint8, size)]
            )
        )
    prog = ttnn.ProgramDescriptor()
    prog.kernels = kernels
    prog.cbs = cbs
    md = ttnn.MeshProgramDescriptor()
    R, C = tuple(mesh.shape)
    md[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(R - 1, C - 1))] = prog

    def run():
        ttnn.generic_op([kv, index, window], md)

    return run


# ------------------------------------------------------------------------------------------------ the test
@mesh_parametrize(timeout_s=1800)
def test_attention_roofline_probe(mesh_device):
    if mesh_device is None:
        pytest.skip("device test")
    import ttnn
    from models.demos.common.bringup.testing import profiler
    from models.demos.deepseek_v41_flash_d_p.tests.packed import ladder_common as L

    mesh = mesh_device
    sp, tp = tuple(mesh.shape)
    nchips = sp * tp
    g = torch.Generator().manual_seed(17)
    rep = ttnn.ReplicateTensorToMesh(mesh)
    DRAM = ttnn.DRAM_MEMORY_CONFIG

    def up(t, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
        return ttnn.from_torch(
            t.contiguous(), dtype=dtype, layout=layout, device=mesh, memory_config=DRAM, mesh_mapper=rep
        )

    ckc = ttnn.init_device_compute_kernel_config(  # (tt/common.attn_compute_config at the shipped settings)
        mesh.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    grid = ttnn.CoreCoord(*GRID)

    # ---------------------------------------------------------------- tables (device concats of one encoded block)
    base = up(ref.encode_rows(torch.randn(BASE, D, generator=g).bfloat16()).view(1, 1, BASE, REC), ttnn.uint8)

    def table(rows: int):
        k = rows // BASE
        return base if k == 1 else ttnn.concat([base] * k, dim=2, memory_config=DRAM)

    tables = {}

    def tab(name: str, rows: int):
        if name not in tables:
            tables[name] = table(rows)
        return tables[name]

    # ---------------------------------------------------------------- index lists
    def index(S: int, top: int, T: int, kind: str):
        idx = torch.empty(S, WIN + top, dtype=torch.int64)
        idx[:, :WIN] = torch.arange(S).view(-1, 1) + 1 + torch.arange(WIN).view(1, -1)  # (region rows s+1 .. s+128)
        if top:
            if kind == "random":
                for s in range(S):
                    idx[s, WIN:] = torch.randperm(T, generator=g)[:top]
            elif kind == "clustered":  # (16 runs of 32 consecutive rows a query: candidate blocks, local attention)
                runs = top // 32
                for s in range(S):
                    starts = torch.randperm(T // 32, generator=g)[:runs] * 32
                    idx[s, WIN:] = (starts.view(-1, 1) + torch.arange(32).view(1, -1)).reshape(-1)
            elif kind == "shared":  # (every query the same rows)
                idx[:, WIN:] = torch.randperm(T, generator=g)[:top].view(1, -1)
            else:
                raise ValueError(kind)
        return up(idx.to(torch.int32).view(1, 1, S, WIN + top), ttnn.uint32)

    qs, regions = {}, {}

    def qt(S):
        if S not in qs:
            qs[S] = up((torch.randn(1, H, S, D, generator=g) * D**-0.5).bfloat16(), ttnn.bfloat16)
        return qs[S]

    def region(S):
        if S not in regions:
            regions[S] = up(torch.randn(1, 1, WIN + S, D, generator=g).bfloat16(), ttnn.bfloat16)
        return regions[S]

    sink = up((torch.randn(1, 1, 1, H, generator=g) * 0.5).bfloat16(), ttnn.bfloat16)
    lane_of_chip = [q * 4 // nchips for q in range(nchips)]  # (4 chip-aligned segments: the 4 x 2k layout)

    cases = []  # (tag, run, flops, bytes, info)

    def add(tag, run, flops, nbytes, **info):
        if not _want(tag):
            return
        try:
            out = run()
            if out is not None:
                ttnn.deallocate(out)
        except Exception as ex:  # (a variant the build refuses: reported, the rest still runs)
            print(f"[PB.AR] {tag}: FAILED {str(ex).splitlines()[0][:200]}", flush=True)
            return
        cases.append((tag, run, flops, nbytes, info))

    def sdpa_flops(S, keys):
        return 4.0 * H * S * keys * D  # (QK^T and PV, 2 FLOP a MAC)

    def sdpa_bytes(S, top):
        return S * (WIN * D * 2 + top * REC + (WIN + top) * 4 + H * D * 2 * 2)  # (window, records, ids, q in + out)

    # ---------------------------------------------------------------- (b) sparse_sdpa at the model's shapes
    def sdpa_case(tag, S, top, T, kind, lanes: bool = False, window_only: bool = False):
        q = qt(S)
        reg = region(S)
        ix = index(S, top, T, kind)
        if window_only:

            def run(q=q, reg=reg, ix=ix):
                return ttnn.bringup.sparse_sdpa(
                    q, reg, ix, D, kv_format=ttnn.bringup.SparseKVFormat.BF16, scale=1.0, k_chunk_size=KC,
                    compute_kernel_config=ckc, attention_sink=sink, max_grid=grid,
                )  # fmt: skip

        elif lanes:
            bufs = [tab(f"L{k}_{T}", T) for k in range(4)]

            def run(q=q, reg=reg, ix=ix, bufs=bufs):
                return ttnn.bringup.sparse_sdpa(
                    q, bufs[0], ix, D, kv_format=ttnn.bringup.SparseKVFormat.ROW_BFP4, scale=1.0, k_chunk_size=KC,
                    compute_kernel_config=ckc, attention_sink=sink, kv_window=reg, max_grid=grid,
                    kv_lanes=list(bufs), kv_lane_of_chip=lane_of_chip, kv_lane_map_runtime=True,
                )  # fmt: skip

        else:
            kv = tab(f"T{T}", T)

            def run(q=q, reg=reg, ix=ix, kv=kv):
                return ttnn.bringup.sparse_sdpa(
                    q, kv, ix, D, kv_format=ttnn.bringup.SparseKVFormat.ROW_BFP4, scale=1.0, k_chunk_size=KC,
                    compute_kernel_config=ckc, attention_sink=sink, kv_window=reg, max_grid=grid,
                )  # fmt: skip

        add(tag, run, sdpa_flops(S, WIN + top), sdpa_bytes(S, top), S=S, top=top, T=T, kind=kind, ix=ix)
        return ix

    # smallest first
    sdpa_case("sdpa_S110_top512_random", 110, 512, T_1M, "random")
    ix_main = sdpa_case("sdpa_S256_top512_random", 256, 512, T_1M, "random")
    sdpa_case("sdpa_S256_top512_random_T32k", 256, 512, T_SMALL, "random")
    sdpa_case("sdpa_S256_top512_clustered", 256, 512, T_1M, "clustered")
    sdpa_case("sdpa_S256_top512_shared", 256, 512, T_1M, "shared")
    sdpa_case("sdpa_S256_top512_lanes4", 256, 512, T_1M, "random", lanes=True)
    sdpa_case("sdpa_S160_top512_random", 160, 512, T_1M, "random")
    sdpa_case("sdpa_S220_top512_random", 220, 512, T_1M, "random")
    sdpa_case("sdpa_S256_top256_random", 256, 256, T_1M, "random")
    sdpa_case("sdpa_S256_top128_random", 256, 128, T_1M, "random")
    sdpa_case("sdpa_S256_window_only", 256, 0, T_1M, "random", window_only=True)

    # ---------------------------------------------------------------- (c) the gather alone
    if ix_main is not None and (not CASES or any(c.startswith("gather") for c in CASES)):
        kv, reg = tab(f"T{T_1M}", T_1M), region(256)
        gbytes = 256 * (WIN * D * 2 + TOPK * REC)
        for parts in (2, 1):
            add(f"gather_S256_rec+win_{parts}risc", _gather_op(ttnn, mesh, kv, ix_main, reg, 256, parts, True, True),
                0.0, gbytes)  # fmt: skip
        add(
            "gather_S256_rec_2risc",
            _gather_op(ttnn, mesh, kv, ix_main, reg, 256, 2, False, True),
            0.0,
            256 * TOPK * REC,
        )
        add(
            "gather_S256_win_2risc",
            _gather_op(ttnn, mesh, kv, ix_main, reg, 256, 2, True, False),
            0.0,
            256 * WIN * D * 2,
        )

    # ---------------------------------------------------------------- (a) plain matmuls of the sdpa's FLOPs
    def mm_case(tag, B, kind: str, in1_dtype, cfg=None):
        """kind bmm: in1 a batch of its own a query (the sdpa's shapes, no reuse); reuse: in0 as one [B x 64, K] matrix
        against ONE in1, the same per-core blocks (per_core_M 2, the whole N); mcast: the same matmul with a 2D
        multicast config (large per-core blocks, in0 / in1 reuse: the large-M peak)."""
        Mt = 2  # (64 heads)
        N1, N2 = WIN + TOPK, D
        try:
            shp = (1, B, H) if kind == "bmm" else (1, 1, B * H)
            a_qk = up(torch.randn(*shp, D, generator=g).bfloat16(), ttnn.bfloat16, ttnn.TILE_LAYOUT)
            a_pv = up(torch.randn(*shp, N1, generator=g).bfloat16(), ttnn.bfloat16, ttnn.TILE_LAYOUT)
            nb = B if kind == "bmm" else 1
            w_qk = up(torch.randn(1, nb, D, N1, generator=g).bfloat16(), in1_dtype, ttnn.TILE_LAYOUT)
            w_pv = up(torch.randn(1, nb, N1, N2, generator=g).bfloat16(), in1_dtype, ttnn.TILE_LAYOUT)
        except Exception as ex:
            print(f"[PB.AR] {tag}: FAILED upload {str(ex).splitlines()[0][:200]}", flush=True)
            return
        if kind == "mcast":  # (grid x along N: QK 10 x 10 = 100 cores, PV 8 x 10 = 80; per_core_M 52 of M 512 tiles)
            Mtot = B * H // 32
            pcm = -(-Mtot // 10)

            def mc(nt, kt):  # (K-51: the whole K a block, out_block_h 4 < per_core_M: no fp32 partials spilled)
                gx = nt // 2
                return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=ttnn.CoreCoord(gx, 10), in0_block_w=kt, out_subblock_h=2,
                    out_subblock_w=2, out_block_h=4, out_block_w=2, per_core_M=pcm, per_core_N=2,
                    transpose_mcast=False, fused_activation=None, fuse_batch=True,
                )  # fmt: skip

            cfg_qk, cfg_pv = mc(N1 // 32, D // 32), mc(N2 // 32, N1 // 32)
        else:
            cfg_qk = ttnn.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(*GRID), in0_block_w=D // 32, out_subblock_h=1,
                out_subblock_w=4, per_core_M=Mt, per_core_N=N1 // 32,
            )  # fmt: skip
            cfg_pv = ttnn.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(*GRID), in0_block_w=N1 // 32 // 2, out_subblock_h=1,
                out_subblock_w=4, per_core_M=Mt, per_core_N=N2 // 32,
            )  # fmt: skip

        def run():
            o1 = ttnn.matmul(a_qk, w_qk, program_config=cfg_qk, compute_kernel_config=cfg or ckc, memory_config=DRAM,
                             dtype=ttnn.bfloat16)  # fmt: skip
            o2 = ttnn.matmul(a_pv, w_pv, program_config=cfg_pv, compute_kernel_config=cfg or ckc, memory_config=DRAM,
                             dtype=ttnn.bfloat16)  # fmt: skip
            ttnn.deallocate(o1)
            return o2

        in1_bytes = nb * (D * N1 + N1 * N2) * (0.5625 if in1_dtype == ttnn.bfloat4_b else 2.0)
        add(tag, run, sdpa_flops(B, N1), B * H * (D + N1 + N1 + N2) * 2 + in1_bytes, B=B, kind=kind)

    for B in (256,):
        mm_case(f"mm_bmm_B{B}_bfp4", B, "bmm", ttnn.bfloat4_b)
        mm_case(f"mm_reuse_B{B}_bfp4", B, "reuse", ttnn.bfloat4_b)
        mm_case(f"mm_mcast_B{B}_bfp4", B, "mcast", ttnn.bfloat4_b)
        mm_case(f"mm_mcast_B{B}_bf16", B, "mcast", ttnn.bfloat16)
        dest16 = ttnn.init_device_compute_kernel_config(  # (reference only: the sdpa runs fp32 DEST)
            mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )  # fmt: skip
        mm_case(f"mm_mcast_B{B}_bfp4_dest16", B, "mcast", ttnn.bfloat4_b, cfg=dest16)
        mm_case(f"mm_bmm_B{B}_bfp4_dest16", B, "bmm", ttnn.bfloat4_b, cfg=dest16)
    mm_case("mm_bmm_B220_bfp4", 220, "bmm", ttnn.bfloat4_b)

    def mm_peak(tag, in1_dtype, cfg=None):
        """A compute-bound matmul at the same fidelity / formats / fp32 DEST: [4480 x 4096] bf16 x [4096 x 2560] on
        10 x 10 cores (per core 14 x 8 output tiles, the whole K in 16 blocks of 8): the per-chip HiFi2 peak."""
        M, K, N = 4480, 4096, 2560
        try:
            a = up(torch.randn(1, 1, M, K, generator=g).bfloat16(), ttnn.bfloat16, ttnn.TILE_LAYOUT)
            w = up(torch.randn(1, 1, K, N, generator=g).bfloat16(), in1_dtype, ttnn.TILE_LAYOUT)
        except Exception as ex:
            print(f"[PB.AR] {tag}: FAILED upload {str(ex).splitlines()[0][:200]}", flush=True)
            return
        pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(10, 10), in0_block_w=8, out_subblock_h=2, out_subblock_w=2,
            per_core_M=M // 32 // 10, per_core_N=N // 32 // 10, transpose_mcast=False, fused_activation=None,
            fuse_batch=True,
        )  # fmt: skip

        def run():
            return ttnn.matmul(a, w, program_config=pc, compute_kernel_config=cfg or ckc, memory_config=DRAM,
                               dtype=ttnn.bfloat16)  # fmt: skip

        add(tag, run, 2.0 * M * K * N, (M * K + M * N) * 2 + K * N * 0.5625, cores=100)

    mm_peak("mm_peak_bfp4", ttnn.bfloat4_b)
    mm_peak("mm_peak_bf16", ttnn.bfloat16)

    # ---------------------------------------------------------------- timing (real-time profiler, every chip)
    profiler.enable(mesh, rt=True)
    try:
        for tag, run, *_ in cases:
            profiler.signpost(tag)
            for _ in range(ITERS):
                o = run()
                if o is not None:
                    ttnn.deallocate(o)
        profiler.signpost("end")
        profiler.finish_rt()
        prof = profiler.result()
    finally:
        profiler.disable()

    # ---------------------------------------------------------------- repeat determinism (the main sdpa case)
    det = None
    main = [c for c in cases if c[0] == "sdpa_S256_top512_random"]
    if main:
        o1, o2 = main[0][1](), main[0][1]()
        t1 = ttnn.to_torch(ttnn.get_device_tensors(o1)[0]).float()
        t2 = ttnn.to_torch(ttnn.get_device_tensors(o2)[0]).float()
        det = float((t1 - t2).abs().max())
        finite = bool(torch.isfinite(t1).all())
        ttnn.deallocate(o1)
        ttnn.deallocate(o2)
        print(f"[PB.AR] sdpa repeat max |diff| {det} (finite {finite})", flush=True)
        assert finite, "sparse_sdpa probe output not finite"
        assert det == 0.0, f"sparse_sdpa probe not deterministic: {det}"

    # ---------------------------------------------------------------- report
    res = {"iters": ITERS, "grid": list(GRID), "mesh": [sp, tp], "cases": {}, "sdpa_repeat_max_abs_diff": det}
    for tag, _, flops, nbytes, info in cases:
        ns = prof["kernel_ns"].get(tag, float("nan")) / ITERS
        per = prof.get("kernel_ns_dev", {}).get(tag, {}) or {}
        chip_ms = sorted(v / ITERS / 1e6 for v in per.values())
        ms = ns / 1e6
        row = {
            "ms": round(ms, 4),
            "chip_ms_min": round(chip_ms[0], 4) if chip_ms else None,
            "chip_ms_median": round(chip_ms[len(chip_ms) // 2], 4) if chip_ms else None,
            "chip_ms_max": round(chip_ms[-1], 4) if chip_ms else None,
            "gflop": round(flops / 1e9, 3),
            "mbytes": round(nbytes / 1e6, 3),
            "tflops": round(flops / ns / 1e3, 2) if flops and ns > 0 else None,
            "gbs": round(nbytes / ns, 1) if ns > 0 else None,
        }
        row.update({k: v for k, v in info.items() if k != "ix"})
        res["cases"][tag] = row
        print(
            f"[PB.AR] {tag:34s} {ms:8.4f} ms (chips {row['chip_ms_min']}..{row['chip_ms_max']}) "
            f"{row['tflops']} TFLOP/s {row['gbs']} GB/s",
            flush=True,
        )
        if ms > 0:
            L.record(f"ar_{tag}_ms", round(ms, 4))
            if row["tflops"]:
                L.record(f"ar_{tag}_tflops", row["tflops"])
            if row["gbs"]:
                L.record(f"ar_{tag}_gbs", row["gbs"])
    Path(OUT).write_text(json.dumps(res, indent=1))
    print(f"[PB.AR] summary -> {OUT}", flush=True)
    assert any(t.startswith("sdpa_") for t in res["cases"]), "no sparse_sdpa case ran"
