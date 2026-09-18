# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.experimental.matmul_reduce_scatter_sp_async: fused sequence-parallel multiply-then-scatter on dim 2.

Runs on a 1x4 mesh (cluster_axis=1). Which topology is exercised is selected by the mesh graph descriptor the
runner points TT_MESH_GRAPH_DESC_PATH at:
  *ring_ring*.textproto  ->  -k ring   (FABRIC_2D_TORUS_XY, ttnn.Topology.Ring)
  *line_line*.textproto  ->  -k line   (FABRIC_2D,          ttnn.Topology.Linear)
Cases whose MGD does not match are skipped before any device is opened.

Reference = the unfused path: ttnn.linear(x, w, transpose_b) then reduce_scatter_minimal_async(dim=2).
`-k perf` prints traced us/op for fused vs unfused on the Llama-8B TP4 shapes (2 warmups, trace of 10, 20 replays).
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

# (name, B, S, K_local, N, transpose_b). K_local is this rank's K shard (the activation is [B,1,S,K_local]).
CASES = [
    ("small_b1", 1, 256, 128, 128, False),
    ("small_b2_tb", 2, 256, 128, 128, True),
    # Llama-8B TP4 row-parallel forward: x [1,1,2048,1024] @ W[4096,1024]^T  (out_proj)
    ("llama8b_out_proj", 1, 2048, 1024, 4096, True),
    # Llama-8B TP4 row-parallel forward: x [1,1,2048,3584] @ W[4096,3584]^T  (w2)
    ("llama8b_w2", 1, 2048, 3584, 4096, True),
    # Llama-8B TP4 column-parallel dgrad: grad [1,1,2048,1536] @ W[1536,4096]  (no transpose)
    ("llama8b_dgrad_col", 1, 2048, 1536, 4096, False),
    # The same three shapes with 5 samples per device ([5,1,2048,K_local]): 20 sub-batches.
    ("llama8b_out_proj_b5", 5, 2048, 1024, 4096, True),
    ("llama8b_w2_b5", 5, 2048, 3584, 4096, True),
    ("llama8b_dgrad_col_b5", 5, 2048, 1536, 4096, False),
    # Llama-8B TP4 column-parallel (gate_up) dgrad: grad [B,1,2048,7168] @ W[7168,4096] (no transpose): the largest
    # matmul+reduce-scatter of the backward pass (weight 56 MB, slab 5.4 MB/core: not L1-resident).
    ("llama8b_dgrad_col_gate_up", 1, 2048, 7168, 4096, False),
    ("llama8b_dgrad_col_gate_up_b5", 5, 2048, 7168, 4096, False),
]
PERF_CASES = {
    "llama8b_out_proj",
    "llama8b_w2",
    "llama8b_dgrad_col",
    "llama8b_out_proj_b5",
    "llama8b_w2_b5",
    "llama8b_dgrad_col_b5",
    "llama8b_dgrad_col_gate_up",
    "llama8b_dgrad_col_gate_up_b5",
}
# B=5 correctness is checked on out_proj only (the others need a > 0.5 TFLOP fp32 host reference); perf covers all.
CHECK_SKIP_CASES = {"llama8b_w2_b5", "llama8b_dgrad_col_b5", "llama8b_dgrad_col_gate_up_b5"}


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


def _to_torch_concat_seq(mesh_device, t):
    # Per-device [B,1,S/T,N] shards -> [B,1,S,N] in ring-index order along cluster_axis=1 of the (1,4) mesh.
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=2))


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


def _compute_kernel_config(name):
    if name == "bf16acc":
        return ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
        )
    assert name == "fp32acc"  # tt-train's ComputeKernelConfig::matmul()
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )


def _bf16_ulp_stats(a, ref):
    """max and p99.9 of |a - ref| in units of the bf16 ulp AT THE RMS MAGNITUDE of ref (ulp = 2^(floor(log2 rms) - 7)):
    an "ulp at scale" error, so near-zero outputs do not dominate as they would with a per-element ulp."""
    rms = ref.float().pow(2).mean().sqrt().clamp_min(1e-30)
    ulp = torch.exp2(torch.floor(torch.log2(rms)) - 7)
    err = ((a - ref).abs() / ulp).flatten()
    return err.max().item(), torch.quantile(err[:: max(1, err.numel() // 1_000_000)], 0.999).item()


def test_rs_first_touch_order_ring_and_line():
    """Host-side check of the slice order the SP schedule will use (derived from the RS reader kernels)."""
    order = ttnn.experimental.matmul_reduce_scatter_sp_rs_first_touch_order
    # Ring: both direction cores start at r+T/2, forward decreasing / backward increasing, local slice last.
    assert order(ttnn.Topology.Ring, 4, 0) == [2, 1, 3, 0]
    assert order(ttnn.Topology.Ring, 4, 1) == [3, 2, 0, 1]
    assert order(ttnn.Topology.Ring, 4, 3) == [1, 0, 2, 3]
    assert order(ttnn.Topology.Ring, 2, 0) == [1, 0]
    assert order(ttnn.Topology.Ring, 8, 0) == [4, 3, 5, 2, 6, 1, 7, 0]
    # Linear: FWD core T-1..r+1, BWD core 0..r-1, interleaved, local slice last.
    assert order(ttnn.Topology.Linear, 4, 0) == [3, 2, 1, 0]
    assert order(ttnn.Topology.Linear, 4, 1) == [3, 0, 2, 1]
    assert order(ttnn.Topology.Linear, 4, 2) == [3, 0, 1, 2]
    assert order(ttnn.Topology.Linear, 4, 3) == [0, 1, 2, 3]
    for topo in (ttnn.Topology.Ring, ttnn.Topology.Linear):
        for T in (2, 4, 8):
            for r in range(T):
                o = order(topo, T, r)
                assert sorted(o) == list(range(T)) and o[-1] == r


@pytest.mark.parametrize(
    "sp_topology, device_params",
    [("ring", TOPOLOGY_PARAMS["ring"][0]), ("line", TOPOLOGY_PARAMS["line"][0])],
    ids=["ring", "line"],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("num_links", [1, 2], ids=["links1", "links2"])
@pytest.mark.parametrize("mode", ["check", "check_fp32acc", "perf", "perf2"])
@pytest.mark.parametrize("ckc_name", ["bf16acc", "fp32acc"])
@pytest.mark.parametrize("payload", ["bf16", "bfp8"])
def test_matmul_reduce_scatter_sp_async(
    sp_topology, mesh_device, device_params, case, num_links, mode, ckc_name, payload
):
    """perf2: one traced timing set per (shape, compute config, payload dtype): fused | unfused (linear + RS) | linear
    alone | sub-batched matmul alone (same kernels, identity schedule; bf16 output) | RS alone -- plus, for the B=1
    Llama shapes, the numerics of the fused output against the fp32 reference and (payload=bfp8) against the
    bf16-payload fused output. payload=bfp8 makes the matmul pack its partial as bfloat8_b (dtype=bfp8), so the
    reduce-scatter moves and re-quantises half the bytes and the op output is bfp8. ckc_name selects the matmul
    compute config for both paths (bf16acc = HiFi2 + bf16 dest acc, fp32acc = tt-train's HiFi4 + fp32 dest acc)."""
    topology = sp_topology
    name, B, S, K, N, transpose_b = case
    T = MESH_SHAPE[1]
    if mode != "check" and name not in PERF_CASES:
        pytest.skip("perf / fp32-acc modes only for the Llama shapes")
    if mode not in ("perf", "perf2") and name in CHECK_SKIP_CASES:
        pytest.skip("check modes skipped for this B=5 shape (host fp32 reference too large); see CHECK_SKIP_CASES")
    if mode != "perf2" and (ckc_name != "bf16acc" or payload != "bf16"):
        pytest.skip("ckc_name / payload only vary in perf2 mode")
    # Compute kernel config handed to BOTH the fused op and ttnn.linear. None = the op/ttnn.matmul defaults
    # (HiFi2, bf16 dest accumulation, packer L1 acc); fp32acc = tt-train ComputeKernelConfig::matmul()
    # (HiFi4 + fp32 dest accumulation + packer L1 acc).
    user_ckc = (
        ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        if mode == "check_fp32acc"
        else None
    )
    # What the op resolves internally when user_ckc is None (matches ttnn.matmul's defaults for bf16).
    resolved_ckc = user_ckc or ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    torch.manual_seed(0)

    grid = mesh_device.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    rs_sems = _sems(mesh_device, crs, 3)
    barrier = ttnn.create_global_semaphore(mesh_device, crs, 0)

    # Full activations [B,1,S,K*T] sharded on K across the 4 devices; weight sharded on its K dim too.
    x_full = torch.randn([B, 1, S, K * T], dtype=torch.bfloat16)
    w_full = torch.randn([1, 1, N, K * T] if transpose_b else [1, 1, K * T, N], dtype=torch.bfloat16) * 0.05
    x = ttnn.from_torch(
        x_full,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(None, 3)),
    )
    w = ttnn.from_torch(
        w_full,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(None, 3 if transpose_b else 2)),
    )
    w_ref = w_full.float().transpose(-1, -2) if transpose_b else w_full.float()
    # [B,1,S,N]: the sum over the T partial products (only where a mode compares against it: up to 2.4 TFLOP on the
    # host for the B=5 shapes)
    ref = x_full.float() @ w_ref if mode not in ("perf", "perf2") or B == 1 else None

    def fused(**rs_grid_kwargs):
        return ttnn.experimental.matmul_reduce_scatter_sp_async(
            x,
            w,
            CLUSTER_AXIS,
            rs_sems,
            barrier_semaphore=barrier,
            transpose_b=transpose_b,
            num_links=num_links,
            topology=topology,
            compute_kernel_config=user_ckc,
            **rs_grid_kwargs,
        )

    def rs(t):
        return ttnn.experimental.reduce_scatter_minimal_async(
            t,
            dim=2,
            multi_device_global_semaphore=rs_sems,
            barrier_semaphore=barrier,
            num_links=num_links,
            topology=topology,
            cluster_axis=CLUSTER_AXIS,
        )

    def unfused():
        return rs(ttnn.linear(x, w, transpose_b=transpose_b, compute_kernel_config=user_ckc))

    def unfused_same_config():
        # ttnn.linear on the fused op's own sub-batched view with the fused op's own derived program config and
        # compute config (fuse_batch=False, the matmul rectangle above the 2 default CCL rows), then the standalone
        # reduce-scatter: the same arithmetic as the fused op, so it must match BITWISE.
        x_view = ttnn.experimental.sp_sub_batched_view(x, T)
        cfg = ttnn.experimental.sp_matmul_program_config(
            x_view, w, ttnn.CoreCoord(grid.x, grid.y - 2), transpose_b, resolved_ckc
        )
        mm = ttnn.linear(x_view, w, transpose_b=transpose_b, program_config=cfg, compute_kernel_config=resolved_ckc)
        return rs(ttnn.reshape(mm, [B, 1, S, N]))

    topo_name = "ring" if topology == ttnn.Topology.Ring else "line"
    label = f"[{topo_name}] {name} B={B} S={S} K={K} N={N} links={num_links} transpose_b={transpose_b}"

    if mode == "perf2":
        ckc = _compute_kernel_config(ckc_name)
        out_dtype = ttnn.bfloat8_b if payload == "bfp8" else ttnn.bfloat16

        def fused_p(dtype=out_dtype):
            return ttnn.experimental.matmul_reduce_scatter_sp_async(
                x,
                w,
                CLUSTER_AXIS,
                rs_sems,
                barrier_semaphore=barrier,
                transpose_b=transpose_b,
                num_links=num_links,
                topology=topology,
                dtype=dtype,
                compute_kernel_config=ckc,
            )

        def linear_p():
            return ttnn.linear(x, w, transpose_b=transpose_b, dtype=out_dtype, compute_kernel_config=ckc)

        mm_partial = linear_p()
        ttnn.synchronize_device(mesh_device)
        x_view = ttnn.experimental.sp_sub_batched_view(x, T)
        cfg = ttnn.experimental.sp_matmul_program_config(x_view, w, ttnn.CoreCoord(grid.x, grid.y - 2), transpose_b, ckc)
        words = [j | (j << 8) for j in range(B * T)]
        t = {
            "fused": _timed(mesh_device, fused_p),
            "unfused": _timed(mesh_device, lambda: rs(linear_p())),
            "linear": _timed(mesh_device, linear_p),
            "mm_sp": _timed(
                mesh_device,
                lambda: ttnn.experimental.sp_matmul_schedule_test(
                    x_view, w, words, transpose_b, cfg, compute_kernel_config=ckc
                ),
            ),
            "rs": _timed(mesh_device, lambda: rs(mm_partial)),
        }
        num = ""
        if B == 1:
            f = _to_torch_concat_seq(mesh_device, fused_p()).float()
            _, pcc_ref = comp_pcc(ref, f, 0.9999)
            ulp_max, ulp_p999 = _bf16_ulp_stats(f, ref)
            num = f" | NUMERICS vs fp32 ref: {pcc_ref} max|d|={(f - ref).abs().max().item():.4g} ulp(max/p99.9)={ulp_max:.1f}/{ulp_p999:.2f}"
            if payload == "bfp8":
                fb = _to_torch_concat_seq(mesh_device, fused_p(ttnn.bfloat16)).float()
                _, pcc_b = comp_pcc(fb, f, 0.9999)
                ulp_max_b, ulp_p999_b = _bf16_ulp_stats(f, fb)
                _, pcc_b_ref = comp_pcc(ref, fb, 0.9999)
                num += (
                    f" | bfp8-vs-bf16 fused: {pcc_b} max|d|={(f - fb).abs().max().item():.4g} "
                    f"ulp(max/p99.9)={ulp_max_b:.1f}/{ulp_p999_b:.2f} | bf16 fused vs ref: {pcc_b_ref}"
                )
        msg = (
            f"PERF2 {label} {ckc_name} payload={payload} pc(per_core_N={cfg.per_core_N} sub={cfg.out_subblock_h}x{cfg.out_subblock_w}): "
            f"fused {t['fused']:.1f} | unfused {t['unfused']:.1f} ({t['unfused'] / t['fused']:.2f}x) | linear {t['linear']:.1f} "
            f"| mm_sp_alone {t['mm_sp']:.1f} | rs_alone {t['rs']:.1f} | exposed_rs {t['fused'] - t['mm_sp']:.1f}{num}"
        )
        logger.info(msg)
        print(msg, flush=True)
        return

    if mode == "perf":
        mm_ckc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
        )
        mm_partial = ttnn.linear(x, w, transpose_b=transpose_b)  # [B,1,S,N] input for the standalone RS

        def rs_alone(**kw):
            return ttnn.experimental.reduce_scatter_minimal_async(
                mm_partial,
                dim=2,
                multi_device_global_semaphore=rs_sems,
                barrier_semaphore=barrier,
                num_links=num_links,
                topology=topology,
                cluster_axis=CLUSTER_AXIS,
                **kw,
            )

        def mm_sub_batched(rows):
            # The fused op's matmul alone: same derived config, same grid (rows above the CCL rows), the SP slice
            # schedule mechanism with an identity order, no CCL.
            mm_grid = ttnn.CoreCoord(grid.x, grid.y - rows)
            x_view = ttnn.experimental.sp_sub_batched_view(x, T)
            cfg = ttnn.experimental.sp_matmul_program_config(x_view, w, mm_grid, transpose_b, mm_ckc)
            words = [j | (j << 8) for j in range(B * T)]
            return lambda: ttnn.experimental.sp_matmul_schedule_test(
                x_view, w, words, transpose_b, cfg, compute_kernel_config=mm_ckc
            )

        def rs_workers_fit(rows, workers):
            mux = 0 if (topology == ttnn.Topology.Ring and workers == 1) else 1
            return 2 * num_links * (workers + mux) <= rows * grid.x

        lines = [f"PERF {label}"]
        t_unfused = _timed(mesh_device, unfused)
        t_linear = _timed(mesh_device, lambda: ttnn.linear(x, w, transpose_b=transpose_b))
        lines.append(
            f"    unfused: ttnn.linear {t_linear:.1f} + reduce_scatter (default workers) = {t_unfused:.1f} us/op"
        )
        for rows in (1, 2, 3):
            lines.append(
                f"    matmul alone, sub-batched, {grid.y - rows} rows: {_timed(mesh_device, mm_sub_batched(rows)):.1f} us/op"
            )
        for workers in (1, 2, 4, 5, 8):
            lines.append(
                f"    reduce_scatter alone, {workers} workers/dir/link: {_timed(mesh_device, lambda w=workers: rs_alone(num_workers_per_link=w)):.1f} us/op"
            )
        for rows in (1, 2, 3):
            for workers in (2, 4, 5, 8):
                if not rs_workers_fit(rows, workers):
                    continue
                kw = dict(ccl_core_rows=rows, num_workers_per_link=workers)
                t_serial = _timed(mesh_device, lambda: fused(debug_serialize_reduce_scatter=True, **kw))
                t_fused = _timed(mesh_device, lambda: fused(**kw))
                lines.append(
                    f"    fused rows={rows} rs_workers={workers}: serialized (RS after whole matmul) {t_serial:.1f} | "
                    f"overlapped {t_fused:.1f} us/op | vs unfused {t_unfused / t_fused:.2f}x"
                )
        t_default = _timed(mesh_device, fused)
        lines.append(
            f"    fused DEFAULTS (rows=2, measured workers): {t_default:.1f} us/op | vs unfused {t_unfused / t_default:.2f}x"
        )
        msg = "\n".join(lines)
        logger.info(msg)
        print(msg, flush=True)
        return

    # --- correctness, program cache, trace ---------------------------------------------------------------------
    n0 = mesh_device.num_program_cache_entries()
    out_fused = fused()
    ttnn.synchronize_device(mesh_device)
    n1 = mesh_device.num_program_cache_entries()
    out_fused_2 = fused()
    ttnn.synchronize_device(mesh_device)
    n2 = mesh_device.num_program_cache_entries()
    out_unfused = unfused()
    ttnn.synchronize_device(mesh_device)

    f = _to_torch_concat_seq(mesh_device, out_fused).float()
    f2 = _to_torch_concat_seq(mesh_device, out_fused_2).float()
    u = _to_torch_concat_seq(mesh_device, out_unfused).float()
    assert f.shape == ref.shape == u.shape, (f.shape, ref.shape, u.shape)

    out_same = unfused_same_config()
    ttnn.synchronize_device(mesh_device)
    same = _to_torch_concat_seq(mesh_device, out_same).float()

    # Three-way check.
    # (1) fused == ttnn.linear(same view, same program config, same compute config) + standalone reduce-scatter,
    #     BITWISE: proves the fusion (sub-batched views, slice schedule, per-slice waits, fused RS) is exact.
    # (2) fused vs fp32 torch reference: PCC > 0.9999.
    # (3) fused vs ttnn.linear with ITS auto-derived block config + reduce-scatter: PCC > 0.9998 only. With
    #     fp32_dest_acc_en=False the matmul rounds each in0_block_w K-block partial sum to bf16 before the packer
    #     accumulates it in L1, so a different in0_block_w / out_block (which ttnn.linear picks for a transposed
    #     weight and for the small shapes) gives a different rounding order: a few bf16 ulps over K=14336 (PCC
    #     0.99987 on w2), and the fused result is not the worse of the two (fused-vs-ref 0.99994 vs unfused-vs-ref
    #     0.99991). Where the auto config coincides with ours (dgrad_col) the two are bitwise equal.
    #     With fp32 accumulation (check_fp32acc) the block structure no longer matters and the two agree to within
    #     the final bf16 rounding (max |d| reported below).
    bitwise_same_config = torch.equal(f, same)
    ok_ref, pcc_ref = comp_pcc(ref, f, 0.9999)
    ok_unf, pcc_unf = comp_pcc(u, f, 0.9998)
    _, pcc_unfused_vs_ref = comp_pcc(ref, u, 0.9999)
    msg = (
        f"CHECK {label} ckc={'fp32acc' if user_ckc else 'default'}: fused == same-config linear+RS bitwise: "
        f"{bitwise_same_config} (max|d|={(f - same).abs().max().item():.4g}) | fused-vs-ref {pcc_ref} "
        f"(max|d|={(f - ref).abs().max().item():.4g}) | unfused(auto config)-vs-ref {pcc_unfused_vs_ref} "
        f"(max|d|={(u - ref).abs().max().item():.4g}) | fused-vs-unfused(auto config) {pcc_unf} "
        f"(max|d|={(f - u).abs().max().item():.4g}, bitwise={torch.equal(f, u)}) | "
        f"program_cache: +{n1 - n0} on first call, +{n2 - n1} on second"
    )
    logger.info(msg)
    print(msg, flush=True)
    assert (
        bitwise_same_config
    ), "fused op differs from ttnn.linear(same config) + reduce_scatter: the fusion is not exact"
    assert ok_ref, f"fused vs torch reference: {pcc_ref}"
    assert ok_unf, f"fused vs unfused ttnn path (auto block config): {pcc_unf}"
    assert torch.equal(f, f2), "two consecutive fused runs differ"

    # Exactly one program-cache entry for the fused op, and the second call must be a cache hit.
    assert n1 - n0 == 1, f"expected 1 new program cache entry, got {n1 - n0}"
    assert n2 == n1, f"second fused call was not a program cache hit (+{n2 - n1} entries)"

    # Trace capture / replay.
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out_traced = fused()
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    traced = _to_torch_concat_seq(mesh_device, out_traced).float()
    ttnn.release_trace(mesh_device, tid)
    assert torch.equal(traced, f), "traced fused run differs from the eager fused run"


@pytest.mark.parametrize(
    "sp_topology, device_params",
    [("ring", TOPOLOGY_PARAMS["ring"][0]), ("line", TOPOLOGY_PARAMS["line"][0])],
    ids=["ring", "line"],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("case", [CASES[2], CASES[3]], ids=[CASES[2][0], CASES[3][0]])
@pytest.mark.parametrize("num_links", [2], ids=["links2"])
def test_localize_transpose_b_mismatch(sp_topology, mesh_device, device_params, case, num_links):
    """Bitwise localisation of the fused-vs-unfused difference seen with transpose_b (w2 / out_proj).

    Compares, all on device and bitwise:
      A   fused op                                 vs  A_ser fused op with the RS waiting for the whole matmul
      P   fused op's matmul alone (sp_matmul_schedule_test, identity words, same config/grid/transposes)
      P_cfg plain ttnn.linear on the same [B*T,1,S/T,K] view with the SAME program config + compute config
      P_ref ttnn.linear(x, w, transpose_b=True) as the unfused path runs it (default config)
      P_man ttnn.linear(x, transpose(w), transpose_b=False)
      RS(P) standalone reduce_scatter of P                vs  A
    """
    topology = sp_topology
    name, B, S, K, N, transpose_b = case
    T = MESH_SHAPE[1]
    torch.manual_seed(0)
    grid = mesh_device.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    rs_sems = _sems(mesh_device, crs, 3)
    barrier = ttnn.create_global_semaphore(mesh_device, crs, 0)
    x_full = torch.randn([B, 1, S, K * T], dtype=torch.bfloat16)
    w_full = torch.randn([1, 1, N, K * T] if transpose_b else [1, 1, K * T, N], dtype=torch.bfloat16) * 0.05
    x = ttnn.from_torch(
        x_full,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(None, 3)),
    )
    w = ttnn.from_torch(
        w_full,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(None, 3 if transpose_b else 2)),
    )
    ref = x_full.float() @ (w_full.float().transpose(-1, -2) if transpose_b else w_full.float())
    mm_ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
    )

    def rs(t):
        return ttnn.experimental.reduce_scatter_minimal_async(
            t,
            dim=2,
            multi_device_global_semaphore=rs_sems,
            barrier_semaphore=barrier,
            num_links=num_links,
            topology=topology,
            cluster_axis=CLUSTER_AXIS,
        )

    def fused(**kw):
        return ttnn.experimental.matmul_reduce_scatter_sp_async(
            x,
            w,
            CLUSTER_AXIS,
            rs_sems,
            barrier_semaphore=barrier,
            transpose_b=transpose_b,
            num_links=num_links,
            topology=topology,
            **kw,
        )

    def seq(t):  # [B,1,S/T,N] per device -> [B,1,S,N]
        return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=2)).float()

    def full(t):  # [B,1,S,N] per device, one partial per device -> [T,B,1,S,N]
        return (
            ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).float().reshape(T, B, 1, S, N)
        )

    mm_grid = ttnn.CoreCoord(grid.x, grid.y - 2)  # the op's default: ccl_core_rows=2
    x_view = ttnn.experimental.sp_sub_batched_view(x, T)
    cfg = ttnn.experimental.sp_matmul_program_config(x_view, w, mm_grid, transpose_b, mm_ckc)
    words = [j | (j << 8) for j in range(B * T)]

    A = seq(fused())
    A_ser = seq(fused(debug_serialize_reduce_scatter=True))
    P = ttnn.experimental.sp_matmul_schedule_test(x_view, w, words, transpose_b, cfg, compute_kernel_config=mm_ckc)
    P_cfg = ttnn.linear(x_view, w, transpose_b=transpose_b, program_config=cfg, compute_kernel_config=mm_ckc)
    P_ref = ttnn.linear(x, w, transpose_b=transpose_b)
    P_man = ttnn.linear(x, ttnn.transpose(w, -2, -1), transpose_b=False) if transpose_b else P_ref
    RS_P = seq(rs(ttnn.reshape(P, [B, 1, S, N])))
    U = seq(rs(P_ref))
    P4, P_cfg4 = full(ttnn.reshape(P, [B, 1, S, N])), full(ttnn.reshape(P_cfg, [B, 1, S, N]))
    P_ref4, P_man4 = full(P_ref), full(P_man)

    def cmp(tag, a, b):
        eq = torch.equal(a, b)
        _, pcc = comp_pcc(a, b, 0.9999)
        line = f"LOCALIZE [{'ring' if topology == ttnn.Topology.Ring else 'line'}] {name}: {tag}: bitwise_equal={eq} max|d|={(a - b).abs().max().item():.4g} {pcc}"
        logger.info(line)
        print(line, flush=True)
        return eq

    r = {}
    r["fused == fused(serialized RS)"] = cmp("fused vs fused(serialized RS)", A, A_ser)
    r["fused == RS(fused matmul alone)"] = cmp("fused vs standalone RS(fused matmul alone)", A, RS_P)
    r["fused matmul alone == linear(same view, same config)"] = cmp(
        "fused matmul alone vs ttnn.linear(same view+config)", P4, P_cfg4
    )
    r["linear(same view, same config) == linear default"] = cmp(
        "ttnn.linear(same view+config) vs ttnn.linear(default, transpose_b)", P_cfg4, P_ref4
    )
    r["linear default == linear(pre-transposed w)"] = cmp(
        "ttnn.linear(default, transpose_b) vs ttnn.linear(transpose(w), no transpose_b)", P_ref4, P_man4
    )
    r["fused matmul alone == linear(pre-transposed w)"] = cmp(
        "fused matmul alone vs ttnn.linear(transpose(w), no transpose_b)", P4, P_man4
    )
    cmp("fused vs unfused", A, U)
    cmp("fused vs fp32 ref", A, ref)
    cmp("unfused vs fp32 ref", U, ref)
    cmp("fused matmul partial (sum over devices) vs fp32 ref", P4.sum(0), ref)
    cmp("ttnn.linear default partial (sum over devices) vs fp32 ref", P_ref4.sum(0), ref)
    # The fused pipeline must be self-consistent: schedule-independent, and exactly RS(matmul).
    assert r["fused == fused(serialized RS)"], "schedule/ordinals change the result"
    assert r["fused == RS(fused matmul alone)"], "fused RS differs from the standalone RS of the same partial"
