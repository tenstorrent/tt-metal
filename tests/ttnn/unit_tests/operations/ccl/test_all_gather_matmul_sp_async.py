# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.experimental.all_gather_matmul_sp_async: fused sequence-parallel gather-then-multiply on dim 2.

Runs on a 1x4 mesh (cluster_axis=1). Which topology is exercised is selected by the mesh graph descriptor the
runner points TT_MESH_GRAPH_DESC_PATH at:
  *ring_ring*.textproto  ->  -k ring   (FABRIC_2D_TORUS_XY, ttnn.Topology.Ring)
  *line_line*.textproto  ->  -k line   (FABRIC_2D,          ttnn.Topology.Linear)
Cases whose MGD does not match are skipped before any device is opened.

Reference = the unfused path: all_gather_async(dim=2, cluster_axis=1) then ttnn.linear(gathered, w, transpose_b).
The gathered output must be bitwise equal; the matmul output is checked against the fp32 torch reference and the
unfused ttnn path (PCC > 0.9999, max |diff| reported). `-k perf` prints traced us/op for fused vs unfused on the
Llama-8B TP4 shapes (2 warmups, trace of 10, 20 replays).
"""
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

MESH_SHAPE = (1, 4)
CLUSTER_AXIS = 1
TRACE_REGION_SIZE = 90_000_000
PERF_REPS = 10
PERF_ITERS = 20

TOPOLOGY_PARAMS = {
    "ring": (
        {"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY, "trace_region_size": TRACE_REGION_SIZE},
        ttnn.Topology.Ring,
        "ring_ring",
    ),
    "line": (
        {"fabric_config": ttnn.FabricConfig.FABRIC_2D, "trace_region_size": TRACE_REGION_SIZE},
        ttnn.Topology.Linear,
        "line_line",
    ),
}

# (name, B, S_local, K, N_local, transpose_b). S_local is this rank's sequence shard (the activation is
# [B,1,S_local,K]); N_local is this rank's output-feature shard of the weight.
CASES = [
    ("small_b1", 1, 256, 128, 128, False),
    ("small_b2_tb", 2, 128, 256, 192, True),
    # Llama-8B TP4 column-parallel forward: x [1,1,512,4096] -> [1,1,2048,4096] @ W[1536,4096]^T  (fused qkv)
    ("llama8b_qkv", 1, 512, 4096, 1536, True),
    # Llama-8B TP4 column-parallel forward: gate_up W[7168,4096]^T
    ("llama8b_gate_up", 1, 512, 4096, 7168, True),
    # Llama-8B TP4 row-parallel dgrad: grad [1,1,512,4096] -> [1,1,2048,4096] @ W[4096,1024]  (no transpose)
    ("llama8b_dgrad_row", 1, 512, 4096, 1024, False),
    # The same three shapes with 5 samples per device ([5,1,512,4096] -> gathered [5,1,2048,4096]): 20 sub-batches.
    ("llama8b_qkv_b5", 5, 512, 4096, 1536, True),
    ("llama8b_gate_up_b5", 5, 512, 4096, 7168, True),
    ("llama8b_dgrad_row_b5", 5, 512, 4096, 1024, False),
    # Llama-8B TP4 row-parallel (w2) dgrad: grad [B,1,512,4096] -> [B,1,2048,4096] @ W2[4096,3584] (no transpose):
    # the largest all-gather+matmul of the backward pass (weight slab 2.5 MB/core: not L1-resident).
    ("llama8b_dgrad_row_w2", 1, 512, 4096, 3584, False),
    ("llama8b_dgrad_row_w2_b5", 5, 512, 4096, 3584, False),
]
PERF_CASES = {
    "llama8b_qkv",
    "llama8b_gate_up",
    "llama8b_dgrad_row",
    "llama8b_qkv_b5",
    "llama8b_gate_up_b5",
    "llama8b_dgrad_row_b5",
    "llama8b_dgrad_row_w2",
    "llama8b_dgrad_row_w2_b5",
}
# B=5 correctness is checked on the shapes whose fp32 host reference is affordable (gate_up_b5 / dgrad_row_w2_b5 would
# need > 1 TFLOP CPU matmuls and > 1 GB reference tensors); perf/decomp cover all of them.
CHECK_SKIP_CASES = {"llama8b_gate_up_b5", "llama8b_dgrad_row_w2_b5"}


@pytest.fixture
def sp_topology(request):
    """Topology under test; skips (before the mesh is opened) unless the MGD env var selects it."""
    _device_params, topology, mgd_tag = TOPOLOGY_PARAMS[request.param]
    mgd = os.environ.get("TT_MESH_GRAPH_DESC_PATH", "")
    if mgd_tag not in os.path.basename(mgd):
        pytest.skip(f"TT_MESH_GRAPH_DESC_PATH={mgd!r} does not select a 1x4 *{mgd_tag}* mesh graph descriptor")
    return topology


def _sems(mesh_device, cores, n):
    return [ttnn.create_global_semaphore(mesh_device, cores, 0) for _ in range(n)]


def _compute_kernel_config(name):
    if name == "bf16acc":
        # ttnn.matmul's own defaults for bf16 (what the fused op uses when none is given); passed explicitly to both
        # paths so the comparison is between the same numerics.
        return ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
    # tt-train's ComputeKernelConfig::matmul(): fp32 accumulation, HiFi4.
    assert name == "fp32acc"
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


# fp32-accumulation numerics are checked on the Llama column-parallel shape and one B=2 case.
FP32ACC_CASES = {"llama8b_qkv", "small_b2_tb"}


def _timed(mesh_device, fn):
    for _ in range(2):
        fn()
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for _ in range(PERF_REPS):
        fn()
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    for _ in range(PERF_ITERS):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    dt = (time.perf_counter() - t0) / (PERF_ITERS * PERF_REPS) * 1e6
    ttnn.release_trace(mesh_device, tid)
    return dt


def _bf16_ulp_stats(a, ref):
    """max and p99.9 of |a - ref| in units of the bf16 ulp AT THE RMS MAGNITUDE of ref (ulp = 2^(floor(log2 rms) - 7)):
    an "ulp at scale" error, so near-zero outputs do not dominate as they would with a per-element ulp."""
    rms = ref.float().pow(2).mean().sqrt().clamp_min(1e-30)
    ulp = torch.exp2(torch.floor(torch.log2(rms)) - 7)
    err = ((a - ref).abs() / ulp).flatten()
    return err.max().item(), torch.quantile(err[:: max(1, err.numel() // 1_000_000)], 0.999).item()


def _identity_schedule_words(n):
    # sp_matmul_schedule_test (rs mode) packing: in0_idx | out_idx << 8, no wait/local bits.
    return [(i & 0xFF) | ((i & 0xFF) << 8) for i in range(n)]


def _fits(grid, rows, links, workers):
    return links * 2 * (workers + (1 if workers > 1 else 0)) <= rows * grid.x


def _decomposition(mesh_device, topology, label, x, w, T, B, S, K, N, transpose_b, num_links, ckc, ag_sems, barrier):
    """Per-component traced timings (us/op) to explain fused vs unfused:
    linear_alone        ttnn.linear on the gathered tensor (auto program config, full grid)
    mm_sp_alone[rows]   the sub-batched 2D-mcast matmul alone with the derived SP config on grid.x x (grid.y-rows)
                        (sp_matmul_schedule_test, rs mode, identity schedule: same kernels as the fused matmul)
    ag_default          all_gather_async alone, default workers on the full grid
    ag_region[rows,w]   all_gather_async alone restricted to the bottom `rows` rows with w workers/link
    fused_serial[rows,w] the fused op with the matmul waiting for the whole all-gather (no overlap); skipped when
                        SP_DECOMP_NO_SERIAL=1 (debug variant only)
    fused[rows,w]       the fused op
    """
    no_serial = os.environ.get("SP_DECOMP_NO_SERIAL", "") not in ("", "0")
    grid = mesh_device.compute_with_storage_grid_size()
    ROWS = (1, 2, 3)
    WORKERS = (2, 4, 5, 8)
    rows_out = []

    def step(text):
        # one line per timing so a hang is localized to the step that never prints
        rows_out.append(text)
        print(f"DECOMP-STEP {label}: {text}", flush=True)

    print(f"DECOMP-STEP {label}: start", flush=True)
    g_full = ttnn.experimental.all_gather_async(
        x, 2, CLUSTER_AXIS, mesh_device, topology, ag_sems, barrier_semaphore=barrier, num_links=num_links
    )
    ttnn.synchronize_device(mesh_device)

    t_linear = _timed(mesh_device, lambda: ttnn.linear(g_full, w, transpose_b=transpose_b, compute_kernel_config=ckc))
    step(f"linear_alone(full grid, auto config)        {t_linear:8.1f}")
    t_ag_default = _timed(
        mesh_device,
        lambda: ttnn.experimental.all_gather_async(
            x, 2, CLUSTER_AXIS, mesh_device, topology, ag_sems, barrier_semaphore=barrier, num_links=num_links
        ),
    )
    step(f"ag_default(full grid, default workers)     {t_ag_default:8.1f}")

    gv = ttnn.experimental.sp_sub_batched_view(g_full, T)
    identity = _identity_schedule_words(B * T)
    mm_alone = {}
    for rows in ROWS:
        mm_grid = ttnn.CoreCoord(grid.x, grid.y - rows)
        pc = ttnn.experimental.sp_matmul_program_config(gv, w, mm_grid, transpose_b, ckc)
        mm_alone[rows] = _timed(
            mesh_device,
            lambda pc=pc: ttnn.experimental.sp_matmul_schedule_test(
                gv, w, identity, transpose_b, pc, compute_kernel_config=ckc
            ),
        )
        step(
            f"mm_sp_alone[rows={rows}] grid {grid.x}x{grid.y - rows} per_core_M={pc.per_core_M} per_core_N={pc.per_core_N} "
            f"in0_block_w={pc.in0_block_w} out_block={pc.out_block_h}x{pc.out_block_w} subblock={pc.out_subblock_h}x{pc.out_subblock_w}"
            f"  {mm_alone[rows]:8.1f}"
        )

    for rows in ROWS:
        for workers in WORKERS:
            if not _fits(grid, rows, num_links, workers):
                continue
            region = ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, grid.y - rows), ttnn.CoreCoord(grid.x - 1, grid.y - 1))}
            )
            print(f"DECOMP-STEP {label}: rows={rows} w={workers}: ag_region ...", flush=True)
            t_ag_region = _timed(
                mesh_device,
                lambda: ttnn.experimental.all_gather_async(
                    x,
                    2,
                    CLUSTER_AXIS,
                    mesh_device,
                    topology,
                    ag_sems,
                    barrier_semaphore=barrier,
                    num_links=num_links,
                    num_workers_per_link=workers,
                    sub_core_grids=region,
                ),
            )

            def fused_cfg():
                return ttnn.experimental.all_gather_matmul_sp_async(
                    x,
                    w,
                    CLUSTER_AXIS,
                    ag_sems,
                    barrier_semaphore=barrier,
                    transpose_b=transpose_b,
                    num_links=num_links,
                    topology=topology,
                    compute_kernel_config=ckc,
                    ccl_core_rows=rows,
                    num_workers_per_link=workers,
                )

            t_serial = float("nan")
            if not no_serial:
                print(
                    f"DECOMP-STEP {label}: rows={rows} w={workers}: ag_region {t_ag_region:.1f}; fused_serial ...",
                    flush=True,
                )
                os.environ["TT_SP_AG_MM_SERIALIZE"] = "1"
                try:
                    t_serial = _timed(mesh_device, fused_cfg)
                finally:
                    os.environ.pop("TT_SP_AG_MM_SERIALIZE", None)
            print(f"DECOMP-STEP {label}: rows={rows} w={workers}: fused_serial {t_serial:.1f}; fused ...", flush=True)
            t_fused = _timed(mesh_device, fused_cfg)
            step(
                f"rows={rows} w={workers}: ag_region {t_ag_region:8.1f} | fused_serial {t_serial:8.1f} "
                f"(=> ag in fused {t_serial - mm_alone[rows]:7.1f}) | fused {t_fused:8.1f} "
                f"(overlap gain {t_serial - t_fused:6.1f}, vs unfused {t_ag_default + t_linear:7.1f}: {(t_ag_default + t_linear) / t_fused:.2f}x)"
            )

    msg = f"DECOMP {label} (us/op):\n  " + "\n  ".join(rows_out)
    logger.info(msg)
    print(msg, flush=True)


def test_ag_schedule_ring_and_line():
    """Host-side check of the matmul schedule (derived from the all_gather_async kernels, see sp_ag_schedule)."""
    sched = ttnn.experimental.all_gather_matmul_sp_ag_schedule

    def rows(topo, T, r, B=1):
        return [tuple(x) for x in sched(topo, T, r, B)]

    # (in0_idx, out_idx, wait_dir, wait_count, is_local)
    # Ring T=4 (static_alternate=False): dir0 <- 2 slices from r-1, r-2; dir1 <- 1 slice from r+1.
    assert rows(ttnn.Topology.Ring, 4, 1) == [(0, 1, 0, 0, 1), (0, 0, 0, 1, 0), (2, 2, 1, 2, 0), (3, 3, 0, 2, 0)]
    assert rows(ttnn.Topology.Ring, 4, 0) == [(0, 0, 0, 0, 1), (3, 3, 0, 1, 0), (1, 1, 1, 2, 0), (2, 2, 0, 2, 0)]
    # Linear T=4: dir1 <- r+1..T-1 (counts start at 2: the writer's local signal comes first), dir0 <- r-1..0.
    assert rows(ttnn.Topology.Linear, 4, 0) == [(0, 0, 0, 0, 1), (1, 1, 1, 2, 0), (2, 2, 1, 3, 0), (3, 3, 1, 4, 0)]
    assert rows(ttnn.Topology.Linear, 4, 3) == [(0, 3, 0, 0, 1), (2, 2, 0, 1, 0), (1, 1, 0, 2, 0), (0, 0, 0, 3, 0)]
    assert rows(ttnn.Topology.Linear, 4, 1) == [(0, 1, 0, 0, 1), (0, 0, 0, 1, 0), (2, 2, 1, 2, 0), (3, 3, 1, 3, 0)]
    # B=2: batch-minor within each slice; in0 for local slices indexes the [B,1,S/T,K] input (b), out is b*T+r.
    assert rows(ttnn.Topology.Linear, 2, 0, B=2) == [(0, 0, 0, 0, 1), (1, 2, 0, 0, 1), (1, 1, 1, 2, 0), (3, 3, 1, 2, 0)]
    for topo in (ttnn.Topology.Ring, ttnn.Topology.Linear):
        for T in (2, 4, 8):
            for r in range(T):
                for B in (1, 2):
                    o = rows(topo, T, r, B)
                    assert sorted(x[1] for x in o) == list(range(B * T))
                    assert [x for x in o if x[4]] == [(b, b * T + r, 0, 0, 1) for b in range(B)]
                    remote = [x for x in o if not x[4]]
                    assert all(x[0] == x[1] for x in remote)
                    for d in (0, 1):
                        counts = [x[3] for x in remote if x[2] == d]
                        # per direction: strictly increasing, contiguous, starting at 1 (dir0) / 2 (dir1)
                        uniq = sorted(set(counts))
                        assert uniq == list(range(1 + d, 1 + d + len(uniq))), (topo, T, r, d, counts)


@pytest.mark.parametrize(
    "sp_topology, device_params",
    [("ring", TOPOLOGY_PARAMS["ring"][0]), ("line", TOPOLOGY_PARAMS["line"][0])],
    ids=["ring", "line"],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("num_links", [1, 2], ids=["links1", "links2"])
@pytest.mark.parametrize("mode", ["check", "perf", "decomp", "perf2"])
@pytest.mark.parametrize("ckc_name", ["bf16acc", "fp32acc"])
@pytest.mark.parametrize("payload", ["bf16", "bfp8"])
def test_all_gather_matmul_sp_async(
    sp_topology, mesh_device, device_params, case, num_links, mode, ckc_name, payload
):
    """perf2: one traced timing set per (shape, compute config, payload dtype) at the given link count:
    fused | unfused (all_gather + linear) | linear alone on the gathered tensor | sub-batched matmul alone (same
    kernels as the fused op, identity schedule) | all_gather alone -- plus, for the B=1 Llama shapes, the numerics of
    the fused output against the fp32 reference and (payload=bfp8) against the bf16-payload fused output.
    payload=bfp8 typecasts the activation to bfloat8_b before the gather (timed as part of the op): the gathered copy
    kept for wgrad is then bfp8; the matmul output stays bf16."""
    topology = sp_topology
    name, B, S, K, N, transpose_b = case
    T = MESH_SHAPE[1]
    if mode in ("perf", "decomp", "perf2") and name not in PERF_CASES:
        pytest.skip("perf/decomp modes only for the Llama shapes")
    if mode == "check" and name in CHECK_SKIP_CASES:
        pytest.skip("check mode skipped for this B=5 shape (host fp32 reference too large); see CHECK_SKIP_CASES")
    if mode == "decomp" and num_links != 2:
        pytest.skip("decomposition at 2 links only")
    if payload == "bfp8" and mode != "perf2":
        pytest.skip("bfp8 payload is evaluated in perf2 mode only")
    if ckc_name == "fp32acc" and mode not in ("perf2",) and (mode != "check" or name not in FP32ACC_CASES):
        pytest.skip("fp32 accumulation is checked on the Llama column-parallel shape and one B=2 case")
    torch.manual_seed(0)

    grid = mesh_device.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    ag_sems = _sems(mesh_device, crs, 2)
    barrier = ttnn.create_global_semaphore(mesh_device, crs, 0)
    ckc = _compute_kernel_config(ckc_name)

    # Full activations [B,1,S*T,K] sharded on the sequence across the 4 devices; the weight sharded on its N dim.
    x_full = torch.randn([B, 1, S * T, K], dtype=torch.bfloat16)
    w_full = torch.randn([1, 1, N * T, K] if transpose_b else [1, 1, K, N * T], dtype=torch.bfloat16) * 0.05
    x = ttnn.from_torch(
        x_full,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(None, 2)),
    )
    w = ttnn.from_torch(
        w_full,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(None, 2 if transpose_b else 3)),
    )
    w_ref = w_full.float().transpose(-1, -2) if transpose_b else w_full.float()
    ref_mm = x_full.float() @ w_ref  # [B,1,S*T,N*T]

    def fused():
        return ttnn.experimental.all_gather_matmul_sp_async(
            x,
            w,
            CLUSTER_AXIS,
            ag_sems,
            barrier_semaphore=barrier,
            transpose_b=transpose_b,
            num_links=num_links,
            topology=topology,
            compute_kernel_config=ckc,
        )

    def unfused():
        g = ttnn.experimental.all_gather_async(
            x,
            2,
            CLUSTER_AXIS,
            mesh_device,
            topology,
            ag_sems,
            barrier_semaphore=barrier,
            num_links=num_links,
        )
        return g, ttnn.linear(g, w, transpose_b=transpose_b, compute_kernel_config=ckc)

    def unfused_same_config(g):
        # The plain 2D-mcast matmul on the [B*T,1,S/T,K] view with the program config the fused op derives
        # (ccl_core_rows=1 -> matmul region grid.x x (grid.y-1)). Same kernels, same K-block order -> bitwise.
        gv = ttnn.experimental.sp_sub_batched_view(g, T)
        pc = ttnn.experimental.sp_matmul_program_config(gv, w, ttnn.CoreCoord(grid.x, grid.y - 1), transpose_b, ckc)
        return ttnn.matmul(gv, w, transpose_b=transpose_b, program_config=pc, compute_kernel_config=ckc)

    label = f"[{'ring' if topology == ttnn.Topology.Ring else 'line'}] {name} B={B} S_local={S} K={K} N_local={N} links={num_links} transpose_b={transpose_b} {ckc_name}"

    def gathered_to_torch(t):
        # replicated [B,1,S*T,K] per device -> [T,B,1,S*T,K]
        return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).view(T, B, 1, S * T, K)

    def mm_to_torch(t):
        # per-device [B,1,S*T,N_local] -> [B,1,S*T,N*T] in ring-index order
        return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))

    if mode == "perf2":
        payload_dtype = ttnn.bfloat8_b if payload == "bfp8" else ttnn.bfloat16

        def x_payload():
            return ttnn.typecast(x, payload_dtype) if payload == "bfp8" else x

        def fused_p():
            return ttnn.experimental.all_gather_matmul_sp_async(
                x_payload(),
                w,
                CLUSTER_AXIS,
                ag_sems,
                barrier_semaphore=barrier,
                transpose_b=transpose_b,
                num_links=num_links,
                topology=topology,
                dtype=ttnn.bfloat16,
                compute_kernel_config=ckc,
            )

        def ag_p():
            return ttnn.experimental.all_gather_async(
                x_payload(), 2, CLUSTER_AXIS, mesh_device, topology, ag_sems, barrier_semaphore=barrier, num_links=num_links
            )

        def unfused_p():
            g = ag_p()
            return g, ttnn.linear(g, w, transpose_b=transpose_b, dtype=ttnn.bfloat16, compute_kernel_config=ckc)

        g_full = ag_p()
        ttnn.synchronize_device(mesh_device)
        gv = ttnn.experimental.sp_sub_batched_view(g_full, T)
        pc = ttnn.experimental.sp_matmul_program_config(gv, w, ttnn.CoreCoord(grid.x, grid.y - 2), transpose_b, ckc)
        identity = _identity_schedule_words(B * T)
        t = {
            "fused": _timed(mesh_device, fused_p),
            "unfused": _timed(mesh_device, unfused_p),
            "linear": _timed(
                mesh_device,
                lambda: ttnn.linear(g_full, w, transpose_b=transpose_b, dtype=ttnn.bfloat16, compute_kernel_config=ckc),
            ),
            "mm_sp": _timed(
                mesh_device,
                lambda: ttnn.experimental.sp_matmul_schedule_test(
                    gv, w, identity, transpose_b, pc, compute_kernel_config=ckc, ag_mode=True
                ),
            ),
            "ag": _timed(mesh_device, ag_p),
        }
        num = ""
        if B == 1:
            g_f, mm_f = fused_p()
            ttnn.synchronize_device(mesh_device)
            f = mm_to_torch(mm_f).float()
            _, pcc_ref = comp_pcc(ref_mm, f, 0.9999)
            ulp_max, ulp_p999 = _bf16_ulp_stats(f, ref_mm)
            num = f" | NUMERICS vs fp32 ref: {pcc_ref} max|d|={(f - ref_mm).abs().max().item():.4g} ulp(max/p99.9)={ulp_max:.1f}/{ulp_p999:.2f}"
            if payload == "bfp8":
                _, mm_b = fused()  # bf16 payload, same config
                ttnn.synchronize_device(mesh_device)
                fb = mm_to_torch(mm_b).float()
                _, pcc_b = comp_pcc(fb, f, 0.9999)
                ulp_max_b, ulp_p999_b = _bf16_ulp_stats(f, fb)
                gx = gathered_to_torch(g_f)[0].float()
                ulp_max_g, ulp_p999_g = _bf16_ulp_stats(gx, x_full.float())
                num += (
                    f" | bfp8-vs-bf16 fused: {pcc_b} max|d|={(f - fb).abs().max().item():.4g} "
                    f"ulp(max/p99.9)={ulp_max_b:.1f}/{ulp_p999_b:.2f} | gathered bfp8 vs bf16 input ulp(max/p99.9)={ulp_max_g:.1f}/{ulp_p999_g:.2f}"
                )
        msg = (
            f"PERF2 {label} payload={payload} pc(per_core_N={pc.per_core_N} sub={pc.out_subblock_h}x{pc.out_subblock_w}): "
            f"fused {t['fused']:.1f} | unfused {t['unfused']:.1f} ({t['unfused'] / t['fused']:.2f}x) | linear {t['linear']:.1f} "
            f"| mm_sp_alone {t['mm_sp']:.1f} | ag_alone {t['ag']:.1f} | exposed_ag {t['fused'] - t['mm_sp']:.1f}{num}"
        )
        logger.info(msg)
        print(msg, flush=True)
        return

    if mode == "perf":
        t_fused = _timed(mesh_device, fused)
        t_unfused = _timed(mesh_device, unfused)
        msg = (
            f"PERF {label}: fused {t_fused:.1f} us/op | unfused all_gather+linear {t_unfused:.1f} us/op | "
            f"speedup {t_unfused / t_fused:.2f}x"
        )
        logger.info(msg)
        print(msg, flush=True)
        return

    if mode == "decomp":
        _decomposition(mesh_device, topology, label, x, w, T, B, S, K, N, transpose_b, num_links, ckc, ag_sems, barrier)
        return

    # --- correctness, program cache, trace ---------------------------------------------------------------------
    n0 = mesh_device.num_program_cache_entries()
    g_f, mm_f = fused()
    ttnn.synchronize_device(mesh_device)
    n1 = mesh_device.num_program_cache_entries()
    g_f2, mm_f2 = fused()
    ttnn.synchronize_device(mesh_device)
    n2 = mesh_device.num_program_cache_entries()
    g_u, mm_u = unfused()
    ttnn.synchronize_device(mesh_device)
    mm_sc = unfused_same_config(g_u)
    ttnn.synchronize_device(mesh_device)

    gf = gathered_to_torch(g_f)
    gf2 = gathered_to_torch(g_f2)
    gu = gathered_to_torch(g_u)
    f = mm_to_torch(mm_f).float()
    f2 = mm_to_torch(mm_f2).float()
    u = mm_to_torch(mm_u).float()
    # [B*T,1,S,N*T] sub-batches -> the [B,1,S*T,N*T] layout of f (sub-batch b*T+t = rows t*S..(t+1)*S of batch b)
    u_sc = mm_to_torch(mm_sc).float().reshape(B, T * S, N * T).unsqueeze(1)
    assert f.shape == ref_mm.shape == u.shape == u_sc.shape, (f.shape, ref_mm.shape, u.shape, u_sc.shape)

    # Gathered output: every device holds the full sequence, bitwise.
    for dev in range(T):
        assert torch.equal(gf[dev], x_full), f"fused gathered output on device {dev} differs from the input"
    assert torch.equal(gf, gu), "fused gathered output differs from all_gather_async"
    assert torch.equal(gf, gf2), "two consecutive fused runs differ (gathered)"

    ok_ref, pcc_ref = comp_pcc(ref_mm, f, 0.9999)
    # Fused vs ttnn.linear (auto-derived program config) is only a loose gate with bf16 accumulation: when
    # ttnn.linear's in0_block_w / out_block differ from the SP-derived config, the per-K-block bf16 rounding of the
    # partial sums differs (K=4096 -> 32 spills at in0_block_w=4), which at PCC level is ~1e-4; ttnn.linear itself is
    # only ~0.9999 vs the fp32 reference there. The proof that the fused op computes the right thing is the
    # same-config comparison below, which is bitwise. With fp32 accumulation (tt-train's matmul config) the block
    # order only moves the fp32 partials, so fused vs ttnn.linear is near-bitwise (<= 1 bf16 ulp on the output).
    ok_unf, pcc_unf = comp_pcc(u, f, 0.9998 if ckc_name == "bf16acc" else 0.99999)
    _, pcc_unfused_vs_ref = comp_pcc(ref_mm, u, 0.9999)
    max_abs_fused_vs_unfused = (f - u).abs().max().item()
    max_abs_fused_vs_ref = (f - ref_mm).abs().max().item()
    max_abs_unfused_vs_ref = (u - ref_mm).abs().max().item()
    bitwise_same_config = torch.equal(f, u_sc)
    max_abs_same_config = (f - u_sc).abs().max().item()
    msg = (
        f"CHECK {label}: fused-vs-ref {pcc_ref} (max|d|={max_abs_fused_vs_ref:.4g}) | linear-vs-ref {pcc_unfused_vs_ref} "
        f"(max|d|={max_abs_unfused_vs_ref:.4g}) | fused-vs-linear {pcc_unf} (max|d|={max_abs_fused_vs_unfused:.4g}) | "
        f"fused-vs-matmul(same config) bitwise_equal={bitwise_same_config} (max|d|={max_abs_same_config:.4g}) | "
        f"program_cache: +{n1 - n0} on first call, +{n2 - n1} on second"
    )
    logger.info(msg)
    print(msg, flush=True)
    assert ok_ref, f"fused mm vs torch reference: {pcc_ref}"
    assert ok_unf, f"fused mm vs unfused ttnn.linear path: {pcc_unf}"
    assert (
        bitwise_same_config
    ), f"fused mm differs from the plain matmul with the same program config (max|d|={max_abs_same_config})"
    assert torch.equal(f, f2), "two consecutive fused runs differ (mm)"

    # One program-cache entry for the fused op, and the second call must be a cache hit.
    assert n1 - n0 == 1, f"expected 1 new program cache entry, got {n1 - n0}"
    assert n2 == n1, f"second fused call was not a program cache hit (+{n2 - n1} entries)"

    # Trace capture / replay.
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    g_t, mm_t = fused()
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    traced_g = gathered_to_torch(g_t)
    traced_mm = mm_to_torch(mm_t).float()
    ttnn.release_trace(mesh_device, tid)
    assert torch.equal(traced_g, gf), "traced fused run differs from the eager fused run (gathered)"
    assert torch.equal(traced_mm, f), "traced fused run differs from the eager fused run (mm)"
