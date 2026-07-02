# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# PCC test for the standalone halo-only op ttnn.experimental.neighbor_pad_halo.
#
# The op emits ONLY the compact halo buffer [H-top | H-bot | W-left | W-right] (no interior copy,
# no conv). We reuse the standalone neighbor_pad_async 2D golden (a full per-device padded tensor),
# then slice out exactly the halo bands in the compact-buffer stick order and compare byte-for-byte
# (bf16 copy, no arithmetic).

import time
import torch
import pytest
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc

from ttnn import ShardTensor2dMesh
from tests.nightly.t3000.ccl.test_neighbor_pad_async import compute_2d_pad_golden


def _trace_and_time(mesh_device, run_op, num_iters=30):
    """Trace-replay wall/iter = device latency (untraced wall is host-dispatch-bound)."""
    run_op()  # warmup + cold compile
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    run_op()
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    for _ in range(num_iters):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    us = (time.perf_counter() - t0) * 1e6 / num_iters
    ttnn.release_trace(mesh_device, tid)
    return us


def compact_halo_reference(golden, outer, H_dev, W_dev, pH, pW):
    """Build the expected compact halo buffer [total_sticks, C] from a per-device padded golden.

    golden: per-device tensor [B, T, H_dev+2pH, W_dev+2pW, C]. Sections + stick order match the
    program factory: H sections are W_dev wide (interior W only); W sections are h_total tall
    (include the corner/H-pad rows). Order within each section is (t, row, col), t-major.
    """
    C = golden.shape[-1]
    h_total = H_dev + 2 * pH
    g = golden.reshape(outer, H_dev + 2 * pH, W_dev + 2 * pW, C)
    sticks = []
    # H-top: pH rows above the chunk, interior W columns
    for t in range(outer):
        for pr in range(pH):
            for w in range(W_dev):
                sticks.append(g[t, pr, pW + w, :])
    # H-bot: pH rows below the chunk, interior W columns
    for t in range(outer):
        for pr in range(pH):
            for w in range(W_dev):
                sticks.append(g[t, pH + H_dev + pr, pW + w, :])
    # W-left: pW columns left of the chunk, full h_total rows (incl. corners)
    for t in range(outer):
        for hp in range(h_total):
            for wc in range(pW):
                sticks.append(g[t, hp, wc, :])
    # W-right: pW columns right of the chunk, full h_total rows (incl. corners)
    for t in range(outer):
        for hp in range(h_total):
            for wc in range(pW):
                sticks.append(g[t, hp, pW + W_dev + wc, :])
    return torch.stack(sticks, dim=0)  # [total_sticks, C]


def run_neighbor_pad_halo_2d(mesh_device, input_shape, h_dim, w_dim, h_axis, w_axis, pH, pW, padding_mode, num_links):
    mesh_shape = tuple(mesh_device.shape)
    h_factor = mesh_shape[h_axis]
    w_factor = mesh_shape[w_axis]
    assert input_shape[h_dim] % h_factor == 0
    assert input_shape[w_dim] % w_factor == 0

    torch.manual_seed(42)
    input_tensor = torch.rand(input_shape).bfloat16()
    goldens = compute_2d_pad_golden(input_tensor, mesh_shape, h_dim, w_dim, h_axis, w_axis, pH, pW, padding_mode)

    outer = 1
    for d in range(h_dim):
        outer *= input_shape[d]
    H_dev = input_shape[h_dim] // h_factor
    W_dev = input_shape[w_dim] // w_factor
    C = input_shape[-1]
    h_total = H_dev + 2 * pH
    total_sticks = outer * 2 * pH * W_dev + outer * 2 * pW * h_total

    # Sub-device + semaphores
    grid = mesh_device.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    sub = ttnn.SubDevice([crs])
    sub_id = ttnn.SubDeviceId(0)
    mgr = mesh_device.create_sub_device_manager([sub], 0)
    mesh_device.load_sub_device_manager(mgr)
    mesh_device.set_sub_device_stall_group([sub_id])

    h_sem = ttnn.create_global_semaphore(mesh_device, crs, 0)
    w_sem = ttnn.create_global_semaphore(mesh_device, crs, 0)
    barrier_sem = ttnn.create_global_semaphore(mesh_device, crs, 0)

    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    dims = [None, None]
    dims[h_axis] = h_dim
    dims[w_axis] = w_dim
    input_tensor_mesh = ttnn.from_torch(
        input_tensor,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=mem_config,
        mesh_mapper=ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=dims),
    )
    # Compact halo buffer: per-device [total_sticks, C], replicated (each device owns one).
    halo_buf = ttnn.from_torch(
        torch.zeros([total_sticks, C]).bfloat16(),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=mem_config,
    )

    out = ttnn.experimental.neighbor_pad_halo(
        input_tensor_mesh,
        halo_buf,
        np_padding_h=pH,
        np_padding_w=pW,
        np_cluster_axis=h_axis,
        np_num_links=num_links,
        np_topology=ttnn.Topology.Linear,
        h_neighbor_semaphore=h_sem,
        barrier_semaphore=barrier_sem,
        w_neighbor_semaphore=w_sem,
        np_pad_dim2=w_dim,
        np_pad2_left=pW,
        np_pad2_right=pW,
        np_pad2_cluster_axis=w_axis,
        np_pad2_num_links=num_links,
        padding_mode=padding_mode,
    )
    ttnn.synchronize_device(mesh_device, sub_device_ids=[sub_id])

    out_host = ttnn.from_device(out)
    dev_tensors = ttnn.get_device_tensors(out_host)
    all_pass = True
    for row in range(mesh_shape[0]):
        for col in range(mesh_shape[1]):
            device_idx = row * mesh_shape[1] + col
            dev = ttnn.to_torch(dev_tensors[device_idx])
            ref = compact_halo_reference(goldens[(row, col)], outer, H_dev, W_dev, pH, pW)
            assert dev.shape == ref.shape, f"dev({row},{col}) shape {dev.shape} != ref {ref.shape}"
            eq, msg = comp_equal(dev, ref)
            if not eq:
                _, pcc = comp_pcc(dev, ref, 0.0)
                all_pass = False
                # Per-section diagnostic: which of [H-top | H-bot | W-left | W-right] mismatches.
                h_sec = outer * pH * W_dev
                w_sec = outer * (H_dev + 2 * pH) * pW
                bounds = [("Htop", 0, h_sec), ("Hbot", h_sec, 2 * h_sec),
                          ("Wleft", 2 * h_sec, 2 * h_sec + w_sec),
                          ("Wright", 2 * h_sec + w_sec, 2 * h_sec + 2 * w_sec)]
                secmsg = []
                for name, a, b in bounds:
                    if b <= dev.shape[0]:
                        seq, _ = comp_equal(dev[a:b], ref[a:b])
                        secmsg.append(f"{name}={'OK' if seq else 'BAD'}")
                print(f"FAIL dev({row},{col}): {pcc} | sections: {' '.join(secmsg)}")
            else:
                print(f"PASS dev({row},{col})")

    mesh_device.reset_sub_device_stall_group()
    mesh_device.clear_loaded_sub_device_manager()
    assert all_pass, "compact halo mismatch"


@pytest.mark.timeout(180)
@pytest.mark.parametrize("mesh_device", [(2, 4)], ids=["2x4"], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("padding_mode", ["zeros", "replicate"])
@pytest.mark.parametrize(
    "input_shape",
    [
        [1, 2, 8, 16, 16],  # per-stick W path (wright_base not 8-aligned -> W_COALESCE off)
        [1, 4, 16, 32, 16],  # coalesce-eligible, 8-aligned rows + W_dev
        [1, 3, 16, 24, 16],  # coalesce, NON-8-aligned rows (w_outer=30) + W_dev (6): exercises the relaxed gate
        [1, 7, 32, 64, 32],  # mid-size, NON-8-aligned rows (63/link), W_dev=16 (s4-like ratio)
        [1, 7, 32, 48, 32],  # mid-size, NON-8-aligned rows + W_dev=12 (s3-like non-aligned W)
    ],
    ids=["perstick", "coalesce", "coalesce_nonalign", "mid_wdev16", "mid_wdev12"],
)
def test_neighbor_pad_halo_2d(mesh_device, device_params, padding_mode, input_shape):
    # [B, T, H, W, C]; H sharded over axis 0 (2 dev), W over axis 1 (4 dev). k333 halo (pH=pW=1).
    run_neighbor_pad_halo_2d(
        mesh_device,
        input_shape=input_shape,
        h_dim=2,
        w_dim=3,
        h_axis=0,
        w_axis=1,
        pH=1,
        pW=1,
        padding_mode=padding_mode,
        num_links=1,
    )


def run_halo_vs_async_perf(mesh_device, input_shape, h_dim, w_dim, h_axis, w_axis, pH, pW, num_links):
    """Trace-timed device latency: halo-only op vs full-pad neighbor_pad_async on the same input."""
    mesh_shape = tuple(mesh_device.shape)
    h_factor, w_factor = mesh_shape[h_axis], mesh_shape[w_axis]
    torch.manual_seed(0)
    inp = torch.rand(input_shape).bfloat16()

    outer = 1
    for d in range(h_dim):
        outer *= input_shape[d]
    H_dev, W_dev, C = input_shape[h_dim] // h_factor, input_shape[w_dim] // w_factor, input_shape[-1]
    h_total = H_dev + 2 * pH
    total_sticks = outer * 2 * pH * W_dev + outer * 2 * pW * h_total

    grid = mesh_device.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    mgr = mesh_device.create_sub_device_manager([ttnn.SubDevice([crs])], 0)
    mesh_device.load_sub_device_manager(mgr)
    sub_id = ttnn.SubDeviceId(0)
    mesh_device.set_sub_device_stall_group([sub_id])

    h_sem = ttnn.create_global_semaphore(mesh_device, crs, 0)
    w_sem = ttnn.create_global_semaphore(mesh_device, crs, 0)
    barrier_sem = ttnn.create_global_semaphore(mesh_device, crs, 0)

    mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    dims = [None, None]
    dims[h_axis], dims[w_axis] = h_dim, w_dim
    inp_mesh = ttnn.from_torch(
        inp,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=mem,
        mesh_mapper=ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=dims),
    )
    halo_buf = ttnn.from_torch(
        torch.zeros([total_sticks, C]).bfloat16(),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=mem,
    )
    # persistent full-pad output for the async op (needed for tracing)
    out_shape = list(input_shape)
    out_shape[h_dim] += h_factor * 2 * pH
    out_shape[w_dim] += w_factor * 2 * pW
    persist = ttnn.from_torch(
        torch.zeros(out_shape).bfloat16(),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=mem,
        mesh_mapper=ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=dims),
    )

    def run_halo():
        return ttnn.experimental.neighbor_pad_halo(
            inp_mesh,
            halo_buf,
            np_padding_h=pH,
            np_padding_w=pW,
            np_cluster_axis=h_axis,
            np_num_links=num_links,
            np_topology=ttnn.Topology.Linear,
            h_neighbor_semaphore=h_sem,
            barrier_semaphore=barrier_sem,
            w_neighbor_semaphore=w_sem,
            np_pad_dim2=w_dim,
            np_pad2_left=pW,
            np_pad2_right=pW,
            np_pad2_cluster_axis=w_axis,
            np_pad2_num_links=num_links,
            padding_mode="zeros",
        )

    def run_async():
        return ttnn.experimental.neighbor_pad_async(
            inp_mesh,
            [h_dim, w_dim],
            [pH, pW],
            [pH, pW],
            "zeros",
            [h_axis, w_axis],
            [h_sem, w_sem],
            [barrier_sem],
            num_links=[num_links, num_links],
            memory_config=mem,
            topology=ttnn.Topology.Linear,
            persistent_output_buffer=persist,
        )

    halo_us = _trace_and_time(mesh_device, run_halo)
    async_us = _trace_and_time(mesh_device, run_async)
    # Per-device halo transport: the op sends its halo out + receives the neighbor's halo in (~2x the
    # compact buffer). GB/s = bytes/time gauges whether the op is bandwidth-bound (near the 2-link
    # fabric ceiling ~30 GB/s) vs overhead-bound (far below).
    halo_bytes = total_sticks * C * 2  # compact buffer per device (bf16)
    gbps = (halo_bytes * 2) / (halo_us * 1e-6) / 1e9
    print(f"\n=== PERF (trace wall/iter, device latency) shape={input_shape} outer={outer} 2x4 ===")
    print(f"  neighbor_pad_async (full-pad): {async_us:8.1f} us")
    print(f"  neighbor_pad_halo  (compact):  {halo_us:8.1f} us")
    print(f"  speedup: {async_us / halo_us:.2f}x")
    print(f"  halo per device: {halo_bytes/1e6:.2f} MB;  achieved: {gbps:.1f} GB/s (send+recv)")

    mesh_device.reset_sub_device_stall_group()
    mesh_device.clear_loaded_sub_device_manager()


@pytest.mark.timeout(400)
@pytest.mark.parametrize("mesh_device", [(2, 4)], ids=["2x4"], indirect=True)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112 * 64}], indirect=True
)
@pytest.mark.parametrize("T", [8, 32, 96], ids=["T8", "T32", "T96"])
def test_neighbor_pad_halo_perf(mesh_device, device_params, T):
    # s4_out-like at varying T: outer=T, per-device H=136, W=120, C=128, k333 halo. Small T is
    # barrier-dominated; large T shifts the op toward bandwidth-bound (fixed barriers, scaling data).
    # num_links is HARDWARE-CAPPED at 2 on BH-LB: each inter-chip hop has exactly 2 ethernet channels
    # (num_links=4 => TT_FATAL "Requested link index 2 is out of bounds. 2 ethernet channels available").
    # The op already uses both, so "more fabric links" is not a lever here.
    run_halo_vs_async_perf(
        mesh_device, input_shape=[1, T, 272, 480, 128], h_dim=2, w_dim=3, h_axis=0, w_axis=1, pH=1, pW=1, num_links=2
    )


# Production LTX 1080p 2x4 decoder NP-bound layers, as full [B,T,H,W,C] (per-device = H/2, W/4). All k333
# (pH=pW=1). Verifies the mux speedup holds on the real deployed shapes, not just the synthetic sweep.
_LTX_PROD_2x4 = [
    ([1, 147, 272, 480, 128], "s4_res_out_C128"),  # per-dev 136x120, C_in=128 (s4_res / s4_out)
    ([1, 147, 136, 240, 256], "s3_res_chg_C256"),  # per-dev  68x60,  C_in=256 (s3_res / s3_chg)
]


@pytest.mark.timeout(600)
@pytest.mark.parametrize("mesh_device", [(2, 4)], ids=["2x4"], indirect=True)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112 * 64}], indirect=True
)
@pytest.mark.parametrize("input_shape, shape_id", _LTX_PROD_2x4, ids=[s[1] for s in _LTX_PROD_2x4])
def test_neighbor_pad_halo_prod_perf(mesh_device, device_params, input_shape, shape_id):
    run_halo_vs_async_perf(
        mesh_device, input_shape=input_shape, h_dim=2, w_dim=3, h_axis=0, w_axis=1, pH=1, pW=1, num_links=2
    )


@pytest.mark.timeout(600)
@pytest.mark.parametrize("mesh_device", [(2, 4)], ids=["2x4"], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("padding_mode", ["zeros", "replicate"])
@pytest.mark.parametrize("input_shape, shape_id", _LTX_PROD_2x4, ids=[s[1] for s in _LTX_PROD_2x4])
def test_neighbor_pad_halo_prod_pcc(mesh_device, device_params, padding_mode, input_shape, shape_id):
    # Byte-exact PCC of the compact halo buffer on the production LTX 2x4 shapes (the mux path for zeros).
    run_neighbor_pad_halo_2d(
        mesh_device, input_shape=input_shape, h_dim=2, w_dim=3, h_axis=0, w_axis=1, pH=1, pW=1,
        padding_mode=padding_mode, num_links=2
    )


@pytest.mark.timeout(300)
@pytest.mark.parametrize("mesh_device", [(2, 4)], ids=["2x4"], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_neighbor_pad_halo_devfw(mesh_device, device_params):
    # Non-traced dispatch (device-FW profiled via tracy) to find the bound (H cores vs W cores).
    input_shape, h_dim, w_dim, h_axis, w_axis, pH, pW, num_links = [1, 8, 272, 480, 128], 2, 3, 0, 1, 1, 1, 2
    mesh_shape = tuple(mesh_device.shape)
    h_factor, w_factor = mesh_shape[h_axis], mesh_shape[w_axis]
    torch.manual_seed(0)
    inp = torch.rand(input_shape).bfloat16()
    outer = input_shape[0] * input_shape[1]
    H_dev, W_dev, C = input_shape[h_dim] // h_factor, input_shape[w_dim] // w_factor, input_shape[-1]
    total_sticks = outer * 2 * pH * W_dev + outer * 2 * pW * (H_dev + 2 * pH)
    grid = mesh_device.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    mgr = mesh_device.create_sub_device_manager([ttnn.SubDevice([crs])], 0)
    mesh_device.load_sub_device_manager(mgr)
    sub_id = ttnn.SubDeviceId(0)
    mesh_device.set_sub_device_stall_group([sub_id])
    h_sem = ttnn.create_global_semaphore(mesh_device, crs, 0)
    w_sem = ttnn.create_global_semaphore(mesh_device, crs, 0)
    barrier_sem = ttnn.create_global_semaphore(mesh_device, crs, 0)
    mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    dims = [None, None]
    dims[h_axis], dims[w_axis] = h_dim, w_dim
    inp_mesh = ttnn.from_torch(
        inp,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=mem,
        mesh_mapper=ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=dims),
    )
    halo_buf = ttnn.from_torch(
        torch.zeros([total_sticks, C]).bfloat16(),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=mem,
    )
    for _ in range(6):
        ttnn.experimental.neighbor_pad_halo(
            inp_mesh,
            halo_buf,
            np_padding_h=pH,
            np_padding_w=pW,
            np_cluster_axis=h_axis,
            np_num_links=num_links,
            np_topology=ttnn.Topology.Linear,
            h_neighbor_semaphore=h_sem,
            barrier_semaphore=barrier_sem,
            w_neighbor_semaphore=w_sem,
            np_pad_dim2=w_dim,
            np_pad2_left=pW,
            np_pad2_right=pW,
            np_pad2_cluster_axis=w_axis,
            np_pad2_num_links=num_links,
            padding_mode="zeros",
        )
    ttnn.synchronize_device(mesh_device, sub_device_ids=[sub_id])
    mesh_device.reset_sub_device_stall_group()
    mesh_device.clear_loaded_sub_device_manager()
