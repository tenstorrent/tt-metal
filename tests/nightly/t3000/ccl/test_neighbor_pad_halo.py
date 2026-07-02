# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# PCC test for the standalone halo-only op ttnn.experimental.neighbor_pad_halo.
#
# The op emits ONLY the compact halo buffer [H-top | H-bot | W-left | W-right] (no interior copy,
# no conv). We reuse the standalone neighbor_pad_async 2D golden (a full per-device padded tensor),
# then slice out exactly the halo bands in the compact-buffer stick order and compare byte-for-byte
# (bf16 copy, no arithmetic).

import torch
import pytest
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc

from ttnn import ShardTensor2dMesh
from tests.nightly.t3000.ccl.test_neighbor_pad_async import compute_2d_pad_golden


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
                print(f"FAIL dev({row},{col}): {msg} | {pcc}")
            else:
                print(f"PASS dev({row},{col})")

    mesh_device.reset_sub_device_stall_group()
    mesh_device.clear_loaded_sub_device_manager()
    assert all_pass, "compact halo mismatch"


@pytest.mark.timeout(180)
@pytest.mark.parametrize("mesh_device", [(2, 4)], ids=["2x4"], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("padding_mode", ["zeros", "replicate"])
def test_neighbor_pad_halo_2d(mesh_device, device_params, padding_mode):
    # [B, T, H, W, C]; H sharded over axis 0 (2 dev), W over axis 1 (4 dev). k333 halo (pH=pW=1).
    run_neighbor_pad_halo_2d(
        mesh_device,
        input_shape=[1, 2, 8, 16, 16],
        h_dim=2,
        w_dim=3,
        h_axis=0,
        w_axis=1,
        pH=1,
        pW=1,
        padding_mode=padding_mode,
        num_links=1,
    )
