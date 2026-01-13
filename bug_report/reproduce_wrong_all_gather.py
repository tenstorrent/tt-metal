"""
Test script to investigate all_gather behavior with ROW_MAJOR tensors and num_links > 1.

Hypothesis:
- ttnn.all_gather with ROW_MAJOR layout may produce incorrect results when:
  (1) num_links > 1, AND
  (2) the gather dimension size is NOT aligned to TILE_SIZE (32)
- This bug was observed on Galaxy (default 4 links) when running all_gather on
  untilized bfp8 tensors that were converted to bfloat16 for the ROW_MAJOR path.

Test Types:
1. test_all_gather_with_layout: Basic ttnn.all_gather with ROW_MAJOR/TILE layouts
2. test_all_gather_with_topology: ttnn.all_gather with topology param (deepseek pattern)
3. test_all_gather_async_linear: ttnn.experimental.all_gather_async with Linear topology
4. test_all_gather_async_ring: ttnn.experimental.all_gather_async with Ring topology (llama pattern)

Device configs:
- T3K: FABRIC_2D, mesh_shape=(2,4), max 2 ethernet channels
- Galaxy: FABRIC_2D_TORUS_XY, mesh_shape=(8,4), max 4 ethernet channels

References:
- models/tt_transformers/tt/ccl.py (tt_all_gather with all_gather_async)
- models/demos/llama3_70b_galaxy/tt/llama_ccl.py (ring_all_gather, line_all_gather)
- models/demos/deepseek_v3/tests/unit/test_all_gather.py (all_gather with topology)
- models/demos/gpt_oss/config.py (allgather with all_gather_async)

Usage:
    pytest bug_report/test_row_major_all_gather_num_links.py -v
    pytest bug_report/test_row_major_all_gather_num_links.py::test_all_gather_with_layout -v
    pytest bug_report/test_row_major_all_gather_num_links.py -v -k "num_links_2"
    pytest bug_report/test_row_major_all_gather_num_links.py -v -k "async"
"""

import pytest
import torch

import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from loguru import logger


# =============================================================================
# Device Configuration
# =============================================================================


def is_galaxy():
    """Check if device is Galaxy or TG cluster"""
    cluster_type = ttnn.cluster.get_cluster_type()
    return cluster_type in (ttnn.cluster.ClusterType.GALAXY, ttnn.cluster.ClusterType.TG)


def get_fabric_config():
    """Get fabric config based on device type"""
    if is_galaxy():
        return {"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY, "trace_region_size": 90112}
    else:  # T3K
        return {"fabric_config": ttnn.FabricConfig.FABRIC_2D, "trace_region_size": 90112}


def get_mesh_shape():
    """Get mesh shape: T3K=(2,4), Galaxy=(8,4)"""
    if is_galaxy():
        return (8, 4)
    else:  # T3K
        return (2, 4)


def get_max_num_links():
    """Get max num_links: T3K=1, Galaxy=4"""
    if is_galaxy():
        return 4
    else:  # T3K
        return 1


# =============================================================================
# Test Fixtures and Config
# =============================================================================

fabric_config = get_fabric_config()
mesh_shape = get_mesh_shape()
max_links = get_max_num_links()

# Test shapes covering various cases
TEST_SHAPES = [
    # Tile-aligned shapes (height % 32 == 0)
    [1, 1, 32, 256],
    [1, 1, 64, 1024],
    [1, 1, 128, 2048],
    # Non-tile-aligned heights - key cases for the hypothesis
    [1, 1, 8, 512],
    [1, 1, 16, 512],
    [1, 1, 24, 512],
    [1, 1, 48, 1024],
]

# Generate num_links values based on device
NUM_LINKS_VALUES = list(range(1, max_links + 1))


# =============================================================================
# Helper Functions
# =============================================================================


def create_test_tensor(shape, mesh_device, dtype, layout, memory_config):
    """Create input tensor with random data, replicated across mesh."""
    torch_input = torch.randn(shape, dtype=torch.float32)

    tt_input = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=memory_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    return tt_input, torch_input


def compute_expected_output(torch_input, dim, num_devices_along_axis):
    """Compute expected output after all_gather."""
    repeat_dims = [1] * len(torch_input.shape)
    repeat_dims[dim] = num_devices_along_axis
    return torch_input.repeat(repeat_dims)


def verify_all_gather_output(tt_output, torch_expected, dtype, test_name):
    """Verify output tensor against expected torch tensor.

    Uses exact match (comp_equal) for bfloat16, PCC for other dtypes.

    Returns:
        tuple: (passed: bool, failure_details: list)
    """
    all_passed = True
    failure_details = []

    for device_idx, device_tensor in enumerate(ttnn.get_device_tensors(tt_output)):
        tt_output_torch = ttnn.to_torch(device_tensor)

        if dtype == ttnn.bfloat16:
            eq, output = comp_equal(tt_output_torch, torch_expected)
        else:
            eq, output = comp_pcc(tt_output_torch, torch_expected)

        if not eq:
            all_passed = False
            failure_details.append(
                {
                    "device_idx": device_idx,
                    "comparison_output": output,
                }
            )
            logger.error(f"[{test_name}] Device {device_idx} FAILED: {output}")
        else:
            logger.info(f"[{test_name}] Device {device_idx} PASSED: {output}")

    return all_passed, failure_details


def create_semaphores(mesh_device):
    """Create semaphores required for all_gather_async.

    Based on models/tt_transformers/tt/ccl.py pattern.
    """
    sub_device_crs = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(
                    mesh_device.compute_with_storage_grid_size().x - 1,
                    mesh_device.compute_with_storage_grid_size().y - 1,
                ),
            )
        }
    )

    # Double-buffered semaphores for all_gather_async
    ag_semaphores = [ttnn.create_global_semaphore(mesh_device, sub_device_crs, 0) for _ in range(2)]
    barrier_semaphore = ttnn.create_global_semaphore(mesh_device, sub_device_crs, 0)

    return ag_semaphores, barrier_semaphore


# =============================================================================
# Test 1: Basic ttnn.all_gather with layout variations
# =============================================================================


@pytest.mark.timeout(60)
@pytest.mark.parametrize("shape", TEST_SHAPES)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("num_links", NUM_LINKS_VALUES, ids=lambda x: f"num_links_{x}")
@pytest.mark.parametrize("cluster_axis", [0, 1], ids=lambda x: f"axis_{x}")
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], ids=["ROW_MAJOR", "TILE"])
@pytest.mark.parametrize("dim", [2])
@pytest.mark.parametrize("device_params", [fabric_config], indirect=True)
@pytest.mark.parametrize("mesh_device", [mesh_shape], indirect=True)
def test_all_gather_with_layout(shape, dtype, num_links, cluster_axis, layout, dim, mesh_device):
    """
    Test basic ttnn.all_gather with varying num_links and layouts (ROW_MAJOR/TILE).

    This test investigates whether ROW_MAJOR tensors produce incorrect results
    when num_links > 1, compared to TILE layout as a baseline.
    """
    # bfloat8_b and bfloat4_b require TILE layout
    if dtype in (ttnn.bfloat8_b, ttnn.bfloat4_b) and layout == ttnn.ROW_MAJOR_LAYOUT:
        pytest.skip("bfloat8_b/bfloat4_b require TILE layout")

    layout_name = "ROW_MAJOR" if layout == ttnn.ROW_MAJOR_LAYOUT else "TILE"
    test_name = f"all_gather dtype={dtype} layout={layout_name} shape={shape} links={num_links} axis={cluster_axis}"
    logger.info(f"Running: {test_name}")

    tt_input, torch_input = create_test_tensor(
        shape=shape,
        mesh_device=mesh_device,
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_output = ttnn.all_gather(
        tt_input,
        dim=dim,
        cluster_axis=cluster_axis,
        num_links=num_links,
    )

    num_devices = mesh_device.shape[cluster_axis]
    expected_shape_dim = shape[dim] * num_devices
    assert (
        tt_output.shape[dim] == expected_shape_dim
    ), f"Shape mismatch: expected dim[{dim}]={expected_shape_dim}, got {tt_output.shape[dim]}"

    torch_expected = compute_expected_output(torch_input, dim, num_devices)
    all_passed, failures = verify_all_gather_output(tt_output, torch_expected, dtype, test_name)

    assert all_passed, f"all_gather failed with layout={layout_name}, num_links={num_links}. " f"Failures: {failures}"


# =============================================================================
# Test 2: ttnn.all_gather with topology parameter (deepseek pattern)
# Reference: models/demos/deepseek_v3/tests/unit/test_all_gather.py
# =============================================================================


@pytest.mark.timeout(60)
@pytest.mark.parametrize("shape", TEST_SHAPES)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("num_links", NUM_LINKS_VALUES, ids=lambda x: f"num_links_{x}")
@pytest.mark.parametrize("cluster_axis", [0, 1], ids=lambda x: f"axis_{x}")
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], ids=["ROW_MAJOR", "TILE"])
@pytest.mark.parametrize("topology", [ttnn.Topology.Linear], ids=["Linear"])
@pytest.mark.parametrize("dim", [2])
@pytest.mark.parametrize("device_params", [fabric_config], indirect=True)
@pytest.mark.parametrize("mesh_device", [mesh_shape], indirect=True)
def test_all_gather_with_topology(shape, dtype, num_links, cluster_axis, layout, topology, dim, mesh_device):
    """
    Test ttnn.all_gather with explicit topology parameter.

    Pattern from: models/demos/deepseek_v3/tests/unit/test_all_gather.py
    Uses topology parameter which may affect how all_gather handles different layouts.
    """
    # bfloat8_b and bfloat4_b require TILE layout
    if dtype in (ttnn.bfloat8_b, ttnn.bfloat4_b) and layout == ttnn.ROW_MAJOR_LAYOUT:
        pytest.skip("bfloat8_b/bfloat4_b require TILE layout")

    layout_name = "ROW_MAJOR" if layout == ttnn.ROW_MAJOR_LAYOUT else "TILE"
    test_name = (
        f"all_gather_topology dtype={dtype} layout={layout_name} shape={shape} links={num_links} axis={cluster_axis}"
    )
    logger.info(f"Running: {test_name}")

    tt_input, torch_input = create_test_tensor(
        shape=shape,
        mesh_device=mesh_device,
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_output = ttnn.all_gather(
        tt_input,
        dim=dim,
        cluster_axis=cluster_axis,
        num_links=num_links,
        topology=topology,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    num_devices = mesh_device.shape[cluster_axis]
    expected_shape_dim = shape[dim] * num_devices
    assert (
        tt_output.shape[dim] == expected_shape_dim
    ), f"Shape mismatch: expected dim[{dim}]={expected_shape_dim}, got {tt_output.shape[dim]}"

    torch_expected = compute_expected_output(torch_input, dim, num_devices)
    all_passed, failures = verify_all_gather_output(tt_output, torch_expected, dtype, test_name)

    assert all_passed, (
        f"all_gather with topology failed with layout={layout_name}, num_links={num_links}. " f"Failures: {failures}"
    )


# =============================================================================
# Test 3: ttnn.experimental.all_gather_async with Linear topology
# Reference: models/tt_transformers/tt/ccl.py (tt_all_gather function)
#            models/demos/gpt_oss/config.py (allgather function)
# =============================================================================


@pytest.mark.timeout(60)
@pytest.mark.parametrize("shape", TEST_SHAPES)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("num_links", NUM_LINKS_VALUES, ids=lambda x: f"num_links_{x}")
@pytest.mark.parametrize("cluster_axis", [0, 1], ids=lambda x: f"axis_{x}")
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], ids=["ROW_MAJOR", "TILE"])
@pytest.mark.parametrize("dim", [2])
@pytest.mark.parametrize("device_params", [fabric_config], indirect=True)
@pytest.mark.parametrize("mesh_device", [mesh_shape], indirect=True)
def test_all_gather_async_linear(shape, dtype, num_links, cluster_axis, layout, dim, mesh_device):
    """
    Test ttnn.experimental.all_gather_async with Linear topology.

    Pattern from:
    - models/tt_transformers/tt/ccl.py (tt_all_gather, line_all_gather)
    - models/demos/gpt_oss/config.py (allgather function)

    This is the most commonly used async all_gather pattern in production models.
    """
    # bfloat8_b and bfloat4_b require TILE layout
    if dtype in (ttnn.bfloat8_b, ttnn.bfloat4_b) and layout == ttnn.ROW_MAJOR_LAYOUT:
        pytest.skip("bfloat8_b/bfloat4_b require TILE layout")

    layout_name = "ROW_MAJOR" if layout == ttnn.ROW_MAJOR_LAYOUT else "TILE"
    test_name = f"all_gather_async_linear dtype={dtype} layout={layout_name} shape={shape} links={num_links} axis={cluster_axis}"
    logger.info(f"Running: {test_name}")

    tt_input, torch_input = create_test_tensor(
        shape=shape,
        mesh_device=mesh_device,
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ag_semaphores, barrier_semaphore = create_semaphores(mesh_device)

    tt_output = ttnn.experimental.all_gather_async(
        tt_input,
        dim=dim,
        cluster_axis=cluster_axis,
        mesh_device=mesh_device,
        topology=ttnn.Topology.Linear,
        multi_device_global_semaphore=ag_semaphores,
        barrier_semaphore=barrier_semaphore,
        num_links=num_links,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    num_devices = mesh_device.shape[cluster_axis]
    expected_shape_dim = shape[dim] * num_devices
    assert (
        tt_output.shape[dim] == expected_shape_dim
    ), f"Shape mismatch: expected dim[{dim}]={expected_shape_dim}, got {tt_output.shape[dim]}"

    torch_expected = compute_expected_output(torch_input, dim, num_devices)
    all_passed, failures = verify_all_gather_output(tt_output, torch_expected, dtype, test_name)

    assert all_passed, (
        f"all_gather_async (Linear) failed with layout={layout_name}, num_links={num_links}. " f"Failures: {failures}"
    )


# =============================================================================
# Test 4: ttnn.experimental.all_gather_async with Ring topology
# Reference: models/demos/llama3_70b_galaxy/tt/llama_ccl.py (ring_all_gather)
# =============================================================================


@pytest.mark.timeout(60)
@pytest.mark.parametrize("shape", TEST_SHAPES)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("num_links", NUM_LINKS_VALUES, ids=lambda x: f"num_links_{x}")
@pytest.mark.parametrize("cluster_axis", [0, 1], ids=lambda x: f"axis_{x}")
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], ids=["ROW_MAJOR", "TILE"])
@pytest.mark.parametrize("dim", [2])
@pytest.mark.parametrize("device_params", [fabric_config], indirect=True)
@pytest.mark.parametrize("mesh_device", [mesh_shape], indirect=True)
def test_all_gather_async_ring(shape, dtype, num_links, cluster_axis, layout, dim, mesh_device):
    """
    Test ttnn.experimental.all_gather_async with Ring topology.

    Pattern from: models/demos/llama3_70b_galaxy/tt/llama_ccl.py (ring_all_gather, line 1087-1127)

    Ring topology is used in Galaxy for better performance with torus interconnect.
    This test checks if Ring topology has different behavior with ROW_MAJOR tensors.
    """
    # bfloat8_b and bfloat4_b require TILE layout
    if dtype in (ttnn.bfloat8_b, ttnn.bfloat4_b) and layout == ttnn.ROW_MAJOR_LAYOUT:
        pytest.skip("bfloat8_b/bfloat4_b require TILE layout")

    layout_name = "ROW_MAJOR" if layout == ttnn.ROW_MAJOR_LAYOUT else "TILE"
    test_name = (
        f"all_gather_async_ring dtype={dtype} layout={layout_name} shape={shape} links={num_links} axis={cluster_axis}"
    )
    logger.info(f"Running: {test_name}")

    tt_input, torch_input = create_test_tensor(
        shape=shape,
        mesh_device=mesh_device,
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ag_semaphores, barrier_semaphore = create_semaphores(mesh_device)

    tt_output = ttnn.experimental.all_gather_async(
        tt_input,
        dim=dim,
        cluster_axis=cluster_axis,
        mesh_device=mesh_device,
        topology=ttnn.Topology.Ring,
        multi_device_global_semaphore=ag_semaphores,
        barrier_semaphore=barrier_semaphore,
        num_links=num_links,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    num_devices = mesh_device.shape[cluster_axis]
    expected_shape_dim = shape[dim] * num_devices
    assert (
        tt_output.shape[dim] == expected_shape_dim
    ), f"Shape mismatch: expected dim[{dim}]={expected_shape_dim}, got {tt_output.shape[dim]}"

    torch_expected = compute_expected_output(torch_input, dim, num_devices)
    all_passed, failures = verify_all_gather_output(tt_output, torch_expected, dtype, test_name)

    assert all_passed, (
        f"all_gather_async (Ring) failed with layout={layout_name}, num_links={num_links}. " f"Failures: {failures}"
    )
