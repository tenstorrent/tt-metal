# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Standalone test for fused 2D neighbor_pad_async.
Uses deterministic input patterns to precisely identify where wrong values come from.
"""

import torch
import pytest
import os
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from ttnn import ShardTensor2dMesh, ConcatMesh2dToTensor


def compute_2d_pad_golden(input_tensor, mesh_shape, h_dim, w_dim, h_axis, w_axis, pH, pW, padding_mode="zeros"):
    """
    Compute golden output for 2D neighbor pad.
    Pads h_dim along h_axis of mesh, then w_dim along w_axis of mesh.

    input_tensor: full (unsharded) tensor
    mesh_shape: (rows, cols) of the mesh
    h_dim: tensor dimension for H padding
    w_dim: tensor dimension for W padding
    h_axis: mesh axis for H distribution (0=rows, 1=cols)
    w_axis: mesh axis for W distribution
    pH: padding amount for H
    pW: padding amount for W
    """
    h_factor = mesh_shape[h_axis]
    w_factor = mesh_shape[w_axis]

    # Step 1: Chunk along H, pad H boundaries
    h_chunks = list(torch.chunk(input_tensor, h_factor, dim=h_dim))
    h_padded = []
    for i in range(h_factor):
        chunk = h_chunks[i]
        # Left H pad
        if padding_mode == "zeros":
            left_parts = []
            for p in range(pH):
                if i == 0:
                    left_parts.append(torch.zeros_like(torch.narrow(chunk, h_dim, 0, 1)))
                else:
                    left_parts.append(torch.narrow(h_chunks[i - 1], h_dim, h_chunks[i - 1].shape[h_dim] - pH + p, 1))
        else:
            raise NotImplementedError("Only zeros padding_mode supported")
        # Right H pad
        right_parts = []
        for p in range(pH):
            if i == h_factor - 1:
                right_parts.append(torch.zeros_like(torch.narrow(chunk, h_dim, 0, 1)))
            else:
                right_parts.append(torch.narrow(h_chunks[i + 1], h_dim, p, 1))
        h_padded.append(torch.cat(left_parts + [chunk] + right_parts, dim=h_dim))

    # Step 2: For each H chunk, chunk along W and pad W boundaries
    # The W padding operates on the H-padded chunks
    goldens = {}
    for h_idx in range(h_factor):
        w_chunks = list(torch.chunk(h_padded[h_idx], w_factor, dim=w_dim))
        for w_idx in range(w_factor):
            chunk = w_chunks[w_idx]
            # Left W pad
            left_parts = []
            for p in range(pW):
                if w_idx == 0:
                    left_parts.append(torch.zeros_like(torch.narrow(chunk, w_dim, 0, 1)))
                else:
                    left_parts.append(
                        torch.narrow(w_chunks[w_idx - 1], w_dim, w_chunks[w_idx - 1].shape[w_dim] - pW + p, 1)
                    )
            # Right W pad
            right_parts = []
            for p in range(pW):
                if w_idx == w_factor - 1:
                    right_parts.append(torch.zeros_like(torch.narrow(chunk, w_dim, 0, 1)))
                else:
                    right_parts.append(torch.narrow(w_chunks[w_idx + 1], w_dim, p, 1))
            if h_axis == 0:
                goldens[(h_idx, w_idx)] = torch.cat(left_parts + [chunk] + right_parts, dim=w_dim)
            else:
                goldens[(w_idx, h_idx)] = torch.cat(left_parts + [chunk] + right_parts, dim=w_dim)

    return goldens


def analyze_mismatches(golden, actual, shape_names, h_dim, w_dim, pH, pW, device_label):
    """Detailed analysis of mismatches with value provenance."""
    diff_mask = ~torch.isclose(golden, actual, atol=0, rtol=0)
    num_mismatches = diff_mask.sum().item()

    if num_mismatches == 0:
        logger.info(f"  {device_label}: EXACT MATCH")
        return True

    # Classify errors by region
    ndim = golden.ndim
    h_size = golden.shape[h_dim]
    w_size = golden.shape[w_dim]

    interior_count = 0
    h_pad_count = 0
    w_pad_count = 0
    corner_count = 0

    mismatch_indices = torch.nonzero(diff_mask, as_tuple=False)
    print("DIFF_MASK SHAPE:", diff_mask.shape)

    # for idx_t in mismatch_indices[:20]:  # Analyze first 20 mismatches
    for idx_t in mismatch_indices:
        idx = tuple(idx_t.tolist())
        h_pos = idx[h_dim]
        w_pos = idx[w_dim]
        is_h_pad = h_pos < pH or h_pos >= h_size - pH
        is_w_pad = w_pos < pW or w_pos >= w_size - pW

        if is_h_pad and is_w_pad:
            region = "corner"
        elif is_h_pad:
            region = "h_pad"
        elif is_w_pad:
            region = "w_pad"
        else:
            region = "interior"

        got = actual[idx].item()
        expected = golden[idx].item()
        logger.error(f"    {device_label} MISMATCH at {idx}: got={got:.4f}, expected={expected:.4f} [{region}]")

    # Count all errors by region
    for idx_t in mismatch_indices:
        idx = tuple(idx_t.tolist())
        h_pos = idx[h_dim]
        w_pos = idx[w_dim]
        is_h_pad = h_pos < pH or h_pos >= h_size - pH
        is_w_pad = w_pos < pW or w_pos >= w_size - pW
        if is_h_pad and is_w_pad:
            corner_count += 1
        elif is_h_pad:
            h_pad_count += 1
        elif is_w_pad:
            w_pad_count += 1
        else:
            interior_count += 1

    logger.error(
        f"  {device_label}: {num_mismatches} mismatches "
        f"(interior={interior_count}, h_pad={h_pad_count}, w_pad={w_pad_count}, corner={corner_count})"
    )

    # Check if wrong values are zeros (uninitialized / not written)
    wrong_zeros = (actual[diff_mask] == 0).sum().item()
    if wrong_zeros > 0:
        logger.error(
            f"    {device_label}: {wrong_zeros}/{num_mismatches} wrong values are ZEROS (not written by Phase 2?)"
        )

    return False


def run_2d_neighbor_pad_test(
    mesh_device,
    input_shape,
    h_dim,
    w_dim,
    h_axis,
    w_axis,
    pH,
    pW,
    padding_mode,
    input_dtype,
    topology,
    use_pattern="arange",
):
    """
    Run fused 2D neighbor pad and compare against golden.

    input_shape: full tensor shape (before sharding)
    h_dim: tensor dimension for H (e.g., 2 for [B,T,H,W,C])
    w_dim: tensor dimension for W (e.g., 3 for [B,T,H,W,C])
    h_axis: mesh axis for H distribution
    w_axis: mesh axis for W distribution
    pH, pW: padding amounts
    """
    mesh_shape = tuple(mesh_device.shape)
    h_factor = mesh_shape[h_axis]
    w_factor = mesh_shape[w_axis]

    # Validate shapes are divisible
    assert input_shape[h_dim] % h_factor == 0, f"H dim {input_shape[h_dim]} not divisible by h_factor {h_factor}"
    assert input_shape[w_dim] % w_factor == 0, f"W dim {input_shape[w_dim]} not divisible by w_factor {w_factor}"

    logger.info(f"Input shape: {input_shape}, mesh: {mesh_shape}")
    logger.info(f"H: dim={h_dim}, axis={h_axis}, factor={h_factor}, pad={pH}")
    logger.info(f"W: dim={w_dim}, axis={w_axis}, factor={w_factor}, pad={pW}")

    # Create input with known pattern
    total_elements = 1
    for s in input_shape:
        total_elements *= s

    if use_pattern == "arange":
        input_tensor = torch.arange(total_elements, dtype=torch.float32).reshape(input_shape).bfloat16()
    elif use_pattern == "random":
        torch.manual_seed(42)
        input_tensor = torch.rand(input_shape).bfloat16()
    else:
        raise ValueError(f"Unknown pattern: {use_pattern}")

    # Compute golden per-device
    goldens = compute_2d_pad_golden(input_tensor, mesh_shape, h_dim, w_dim, h_axis, w_axis, pH, pW, padding_mode)

    # Set up sub-device manager
    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    worker_sub_device = ttnn.SubDevice([ccl_sub_device_crs])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_stall_group = [worker_sub_device_id]
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group(sub_device_stall_group)

    # Create semaphores
    h_neighbor_sem = ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0)
    w_neighbor_sem = ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0)
    barrier_sem = ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0)

    # Shard input to mesh
    dims = [None, None]
    dims[h_axis] = h_dim
    dims[w_axis] = w_dim
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    input_tensor_mesh = ttnn.from_torch(
        input_tensor,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=input_dtype,
        memory_config=mem_config,
        mesh_mapper=ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=dims),
    )

    # Run fused 2D neighbor pad
    logger.info("Running fused 2D neighbor_pad_async...")
    output_tensor = ttnn.experimental.neighbor_pad_async(
        input_tensor_mesh,
        [h_dim, w_dim],
        [pH, pW],
        [pH, pW],
        padding_mode,
        [h_axis, w_axis],
        [h_neighbor_sem, w_neighbor_sem],
        [barrier_sem],
        num_links=[1, 1],
        memory_config=mem_config,
        topology=topology,
    )
    ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)
    logger.info("Done.")

    # Compare per-device
    output_host = ttnn.from_device(output_tensor)
    all_passed = True

    for row in range(mesh_shape[0]):
        for col in range(mesh_shape[1]):
            device_idx = row * mesh_shape[1] + col
            device_tensor = ttnn.to_torch(
                output_host,
                mesh_composer=ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_shape, dims=dims),
            )
            # Get per-device tensor directly
            device_tensors = ttnn.get_device_tensors(output_host)
            dev_tensor = ttnn.to_torch(device_tensors[device_idx])

            golden = goldens[(row, col)]
            label = f"Device {device_idx} ({row},{col})"

            print("OUTPUT SHAPE:", dev_tensor.shape, "GOLDEN SHAPE:", golden.shape)
            if dev_tensor.shape != golden.shape:
                logger.error(f"  {label}: shape mismatch: got {dev_tensor.shape}, expected {golden.shape}")
                all_passed = False
                continue

            passed = analyze_mismatches(golden, dev_tensor, input_shape, h_dim, w_dim, pH, pW, label)
            if not passed:
                all_passed = False

    mesh_device.reset_sub_device_stall_group()
    mesh_device.clear_loaded_sub_device_manager()

    return all_passed


@pytest.mark.timeout(120)
@pytest.mark.parametrize("mesh_device", [(2, 4), (4, 8)], ids=["2x4", "4x8"], indirect=True)
@pytest.mark.parametrize(
    "input_shape, h_dim, w_dim, h_axis, w_axis, pH, pW",
    [
        # Small 5D: [B, T, H, W, C] — H along axis 0, W along axis 1
        # ([1, 2, 8, 16, 4], 2, 3, 0, 1, 1, 1),
        # ([1, 8, 32, 32, 32], 2, 3, 0, 1, 1, 1),
        ([1, 2, 8, 16, 32], 2, 3, 0, 1, 1, 1),
        # ([1, 3, 10, 16, 32], 2, 3, 0, 1, 1, 1),
        # Slightly larger
        ([1, 3, 10, 16, 32], 2, 3, 0, 1, 1, 1),
        # Matching conv_0 VAE shape (per-device [1,3,45,40,32] → full H=90, W=160)
        ([1, 3, 90, 160, 32], 2, 3, 0, 1, 1, 1),
        # Flipped axes: H along axis 1, W along axis 0
        ([1, 2, 16, 8, 32], 2, 3, 1, 0, 1, 1),
        # 4D tensor [B, H, W, C]
        ([2, 8, 16, 32], 1, 2, 0, 1, 1, 1),
    ],
    ids=[
        "small_5d_h0w1",
        "medium_5d_h0w1",
        "vae_conv0_h0w1",
        "small_5d_h1w0",
        "small_4d_h0w1",
    ],
)
@pytest.mark.parametrize("use_pattern", ["arange", "random"], ids=["arange", "random"])
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
def test_fused_2d_neighbor_pad(
    mesh_device,
    input_shape,
    h_dim,
    w_dim,
    h_axis,
    w_axis,
    pH,
    pW,
    use_pattern,
    device_params,
):
    passed = run_2d_neighbor_pad_test(
        mesh_device,
        input_shape=list(input_shape),
        h_dim=h_dim,
        w_dim=w_dim,
        h_axis=h_axis,
        w_axis=w_axis,
        pH=pH,
        pW=pW,
        padding_mode="zeros",
        input_dtype=ttnn.bfloat16,
        topology=ttnn.Topology.Linear,
        use_pattern=use_pattern,
    )
    assert passed, "Fused 2D neighbor pad produced incorrect results"


# Also test 1D to make sure we didn't break it
@pytest.mark.timeout(120)
@pytest.mark.parametrize("mesh_device", [(2, 4), (4, 8)], ids=["2x4", "4x8"], indirect=True)
@pytest.mark.parametrize(
    "input_shape, pad_dim, other_dim, pad_axis, pH",
    [
        # 1D H pad only (W factor=1 equivalent — pad along axis 0 with 2 devices)
        ([1, 2, 8, 16, 32], 2, 3, 0, 1),
        # 1D W pad only (H factor=1 equivalent — pad along axis 1 with 4 devices)
        ([1, 2, 8, 16, 32], 3, 2, 1, 1),
    ],
    ids=["1d_h_pad", "1d_w_pad"],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
def test_1d_neighbor_pad_sanity(
    mesh_device,
    input_shape,
    pad_dim,
    other_dim,
    pad_axis,
    pH,
    device_params,
):
    """Sanity check that 1D neighbor pad still works with the new vector API."""
    mesh_shape = tuple(mesh_device.shape)
    factor = mesh_shape[pad_axis]

    assert input_shape[pad_dim] % factor == 0

    total_elements = 1
    for s in input_shape:
        total_elements *= s
    torch.manual_seed(42)
    input_tensor = torch.rand(input_shape).bfloat16()

    # Compute 1D golden
    chunks = list(torch.chunk(input_tensor, factor, dim=pad_dim))
    padded = []
    for i in range(factor):
        c = chunks[i]
        if i == 0:
            left = torch.zeros_like(torch.narrow(c, pad_dim, 0, 1))
        else:
            left = torch.narrow(chunks[i - 1], pad_dim, chunks[i - 1].shape[pad_dim] - pH, pH)
        if i == factor - 1:
            right = torch.zeros_like(torch.narrow(c, pad_dim, 0, 1))
        else:
            right = torch.narrow(chunks[i + 1], pad_dim, 0, pH)
        padded.append(torch.cat([left, c, right], dim=pad_dim))

    # Set up mesh
    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    worker_sub_device = ttnn.SubDevice([ccl_sub_device_crs])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_stall_group = [worker_sub_device_id]
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group(sub_device_stall_group)

    neighbor_sem = ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0)
    barrier_sem_1d = ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0)

    dims = [None, None]
    dims[pad_axis] = pad_dim
    dims[1 - pad_axis] = other_dim
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    input_tensor_mesh = ttnn.from_torch(
        input_tensor,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=mem_config,
        mesh_mapper=ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=dims),
    )

    # 1D call with list API
    output_tensor = ttnn.experimental.neighbor_pad_async(
        input_tensor_mesh,
        [pad_dim],
        [pH],
        [pH],
        "zeros",
        [pad_axis],
        [neighbor_sem],
        [barrier_sem_1d],
        num_links=[1],
        memory_config=mem_config,
        topology=ttnn.Topology.Linear,
    )
    ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)

    # Compare per-device
    output_host = ttnn.from_device(output_tensor)
    device_tensors = ttnn.get_device_tensors(output_host)
    all_passed = True

    for dev_idx in range(len(device_tensors)):
        dev_tensor = ttnn.to_torch(device_tensors[dev_idx])
        # Map device index to chunk index along pad_axis
        row = dev_idx // mesh_shape[1]
        col = dev_idx % mesh_shape[1]
        chunk_idx = row if pad_axis == 0 else col
        golden = padded[chunk_idx]

        if dev_tensor.shape != golden.shape:
            logger.error(f"Device {dev_idx}: shape mismatch: got {dev_tensor.shape}, expected {golden.shape}")
            all_passed = False
            continue

        diff = ~torch.isclose(dev_tensor, golden, atol=0, rtol=0)
        n_diff = diff.sum().item()
        if n_diff > 0:
            logger.error(f"Device {dev_idx} ({row},{col}): {n_diff} mismatches")
            all_passed = False
        else:
            logger.info(f"Device {dev_idx} ({row},{col}): EXACT MATCH")

    mesh_device.reset_sub_device_stall_group()
    mesh_device.clear_loaded_sub_device_manager()

    assert all_passed, "1D neighbor pad produced incorrect results"
