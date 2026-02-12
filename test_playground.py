import ttnn
import torch
import pytest
from loguru import logger
from models.tt_transformers.tt.ccl import get_num_links
from tracy import signpost


def create_sharded_mesh_tensor(
    mesh_device,
    tensor_shape: tuple,
    dims: tuple,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
):
    """
    Helper function to create a sharded tensor across mesh device.

    Args:
        mesh_device: The mesh device to create tensor on
        tensor_shape: Shape of the tensor (e.g., (4384, 7168))
        dims: Sharding dimensions for ShardTensor2dMesh (e.g., (None, -1))
        dtype: Data type for the ttnn tensor (default: ttnn.bfloat16)
        layout: Layout for the ttnn tensor (default: ttnn.TILE_LAYOUT)

    Returns:
        tuple: (torch_tensor, mesh_tensor) - Original PyTorch tensor and sharded ttnn tensor
    """
    # Create PyTorch tensor
    torch_tensor = torch.randn(tensor_shape)
    logger.info(f"Created torch tensor with shape: {torch_tensor.shape}")

    # Create mesh mapper
    mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_device.shape, dims=dims)

    # Convert to ttnn tensor with sharding
    mesh_tensor = ttnn.from_torch(
        torch_tensor,
        mesh_mapper=mesh_mapper,
        layout=layout,
        device=mesh_device,
        dtype=dtype,
    )
    logger.info(f"Created sharded mesh tensor with shape: {mesh_tensor.shape}")

    return torch_tensor, mesh_tensor


def check_ethernet_links(mesh_device, num_links):
    """
    Helper function to query and check available ethernet links.

    Args:
        mesh_device: The mesh device to query
        num_links: Number of links requested for the operation

    Returns:
        tuple: (num_links_available, num_links_axis0, num_links_axis1)

    Raises:
        pytest.skip: If requested num_links exceeds available links
    """
    # Query available ethernet links
    num_links_available = get_num_links(mesh_device)
    num_links_axis0 = get_num_links(mesh_device, cluster_axis=0)
    num_links_axis1 = get_num_links(mesh_device, cluster_axis=1)
    logger.info(f"Available ethernet links (all axes min): {num_links_available}")
    logger.info(f"Available ethernet links (axis 0 - rows): {num_links_axis0}")
    logger.info(f"Available ethernet links (axis 1 - cols): {num_links_axis1}")
    logger.info(f"Requested num_links: {num_links}")

    # Skip test if requested num_links exceeds available links
    if num_links > num_links_available:
        pytest.skip(f"Requested num_links={num_links} exceeds available links={num_links_available}")

    return num_links_available, num_links_axis0, num_links_axis1


def test_mesh_device_configuration(mesh_device):
    """
    Test to query and display mesh device configuration.
    Shows mesh shape, device IDs, worker cores, and DRAM cores.
    """

    # Get and print the mesh shape
    mesh_shape = mesh_device.shape
    logger.info(f"Mesh Device Shape: {mesh_shape}")
    logger.info(f"Number of devices in mesh: {mesh_device.get_num_devices()}")
    logger.info(f"Device IDs: {mesh_device.get_device_ids()}")

    # Get worker core grid size
    worker_grid = mesh_device.compute_with_storage_grid_size()
    logger.info(f"Worker Core Grid Size: {worker_grid}")
    logger.info(f"Total worker cores: {worker_grid.x * worker_grid.y}")

    # Get DRAM core grid size
    dram_grid = mesh_device.dram_grid_size()
    logger.info(f"DRAM Core Grid Size: {dram_grid}")
    logger.info(f"Total DRAM cores: {dram_grid.x * dram_grid.y}")

    # Query available ethernet links
    num_links_all = get_num_links(mesh_device)
    num_links_axis0 = get_num_links(mesh_device, cluster_axis=0)
    num_links_axis1 = get_num_links(mesh_device, cluster_axis=1)
    logger.info(f"Available ethernet links (all axes min): {num_links_all}")
    logger.info(f"Available ethernet links (axis 0 - rows): {num_links_axis0}")
    logger.info(f"Available ethernet links (axis 1 - cols): {num_links_axis1}")


@pytest.mark.parametrize("tensor_shape,dims", [((4 * 1096, 7 * 1024), (None, -1))])
def test_create_sharded_tensor(mesh_device, tensor_shape, dims):
    """
    Test to create a sharded tensor across mesh device.
    Shards according to the specified dims parameter.
    """
    logger.info(f"Tensor shape: {tensor_shape}")
    logger.info(f"Mesh shape: {mesh_device.shape}")
    logger.info(f"Sharding dims: {dims}")

    # Create sharded tensor across mesh
    torch_tensor, mesh_tensor = create_sharded_mesh_tensor(mesh_device, tensor_shape, dims)

    # Visualize tensor distribution across mesh
    logger.info(f"Visualizing tensor distribution across mesh devices:")
    ttnn.visualize_tensor(mesh_tensor)

    # Print detailed tensor information
    logger.info(f"Tensor dtype: {mesh_tensor.dtype}")
    logger.info(f"Tensor layout: {mesh_tensor.layout}")
    logger.info(f"Tensor storage type: {mesh_tensor.storage_type()}")
    logger.info(f"Is sharded: {mesh_tensor.is_sharded()}")

    # Verify tensor was created successfully
    assert mesh_tensor is not None
    logger.info(f"✓ Successfully created sharded tensor across mesh!")


@pytest.mark.parametrize("tensor_shape,dims", [((4 * 1096, 7 * 1024), (None, -1)), ((4 * 1096, 256), (None, -1))])
@pytest.mark.parametrize("num_links", [1, 2])
@pytest.mark.parametrize(
    "device_params,topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}, ttnn.Topology.Ring),
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D}, ttnn.Topology.Linear),
    ],
    indirect=["device_params"],
    ids=["fabric_ring", "fabric_linear"],
)
def test_all_gather_fabric(mesh_device, tensor_shape, dims, device_params, topology, num_links):
    """
    Test all_gather operation with different fabric configurations.
    Tests both 1D Ring and 1D Line topologies.
    """
    # Format parameters for Tracy signpost
    topology_name = "Ring" if topology == ttnn.Topology.Ring else "Linear"
    sharding_dim = "cols" if dims == (None, -1) else "rows" if dims == (-1, None) else str(dims)
    tracy_msg = f"AllGather shape={tensor_shape}, {topology_name}, {num_links}link(s), shard={sharding_dim}"
    signpost(tracy_msg)

    fabric_config = device_params.get("fabric_config", "DISABLED")
    logger.info(f"Testing all_gather with fabric config: {fabric_config}, topology: {topology}")
    logger.info(f"Tensor shape: {tensor_shape}")
    logger.info(f"Mesh shape: {mesh_device.shape}")
    logger.info(f"Sharding dims: {dims}")

    # Check ethernet links availability
    check_ethernet_links(mesh_device, num_links)

    # Create sharded tensor across mesh
    torch_tensor, mesh_tensor = create_sharded_mesh_tensor(mesh_device, tensor_shape, dims)

    # Visualize the sharded tensor before all_gather
    logger.info(f"Sharded tensor BEFORE all_gather:")
    ttnn.visualize_tensor(mesh_tensor)

    # Perform all_gather based on topology
    # Both topologies support num_links, but Linear also requires cluster_axis
    if topology == ttnn.Topology.Ring:
        gathered_tensor = ttnn.all_gather(
            mesh_tensor, dim=-1, num_links=num_links, topology=ttnn.Topology.Ring  # Gather along last dimension
        )
    else:  # Linear (topology == ttnn.Topology.Linear)
        gathered_tensor = ttnn.all_gather(
            mesh_tensor,
            dim=-1,  # Gather along last dimension
            cluster_axis=1,  # Gather along mesh column dimension
            num_links=num_links,
            topology=ttnn.Topology.Linear,
        )

    logger.info(f"Gathered tensor shape: {gathered_tensor.shape}")

    # Visualize the gathered tensor after all_gather
    logger.info(f"Gathered tensor AFTER all_gather ({topology}):")
    ttnn.visualize_tensor(gathered_tensor)

    # Verify the gathered tensor has expected shape
    # After all_gather on dim=-1 across 4 devices, width should be 4x larger
    expected_width = tensor_shape[-1]
    assert (
        gathered_tensor.shape[-1] == expected_width
    ), f"Expected width {expected_width}, got {gathered_tensor.shape[-1]}"

    logger.info(f"✓ Successfully completed all_gather with {topology} topology!")


@pytest.mark.parametrize("tensor_shape,dims", [((4 * 1096, 7 * 1024), (None, None)), ((4 * 1096, 256), (None, None))])
@pytest.mark.parametrize("num_links", [1, 2])
@pytest.mark.parametrize(
    "device_params,topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}, ttnn.Topology.Ring),
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D}, ttnn.Topology.Linear),
    ],
    indirect=["device_params"],
    ids=["fabric_ring", "fabric_linear"],
)
def test_reduce_scatter_fabric(mesh_device, tensor_shape, dims, device_params, topology, num_links):
    """
    Test reduce_scatter operation with different fabric configurations.
    Tests both 1D Ring and 1D Line topologies.
    Reduce_scatter reduces data across devices then scatters unique portions to each device.
    """
    # Format parameters for Tracy signpost
    topology_name = "Ring" if topology == ttnn.Topology.Ring else "Linear"
    sharding_dim = "cols" if dims == (None, -1) else "rows" if dims == (-1, None) else str(dims)
    tracy_msg = f"ReduceScatter shape={tensor_shape}, {topology_name}, {num_links}link(s), shard={sharding_dim}"
    signpost(tracy_msg)

    fabric_config = device_params.get("fabric_config", "DISABLED")
    logger.info(f"Testing reduce_scatter with fabric config: {fabric_config}, topology: {topology}")
    logger.info(f"Tensor shape: {tensor_shape}")
    logger.info(f"Mesh shape: {mesh_device.shape}")
    logger.info(f"Sharding dims: {dims}")

    # Check ethernet links availability
    check_ethernet_links(mesh_device, num_links)

    # Create sharded tensor across mesh
    torch_tensor, mesh_tensor = create_sharded_mesh_tensor(mesh_device, tensor_shape, dims)

    # Visualize the sharded tensor before reduce_scatter
    logger.info(f"Sharded tensor BEFORE reduce_scatter:")
    ttnn.visualize_tensor(mesh_tensor)

    # Perform reduce_scatter based on topology
    # reduce_scatter reduces data across devices then scatters unique portions
    if topology == ttnn.Topology.Ring:
        scattered_tensor = ttnn.reduce_scatter(
            mesh_tensor,
            dim=-1,  # Reduce and scatter along last dimension
            num_links=num_links,
            topology=ttnn.Topology.Ring,
        )
    else:  # Linear (topology == ttnn.Topology.Linear)
        scattered_tensor = ttnn.reduce_scatter(
            mesh_tensor,
            dim=-1,  # Reduce and scatter along last dimension
            cluster_axis=1,  # Scatter along mesh column dimension
            num_links=num_links,
            topology=ttnn.Topology.Linear,
        )

    logger.info(f"Scattered tensor shape: {scattered_tensor.shape}")

    # Visualize the scattered tensor after reduce_scatter
    logger.info(f"Scattered tensor AFTER reduce_scatter ({topology}):")
    ttnn.visualize_tensor(scattered_tensor)

    # Verify the scattered tensor has expected shape
    # Input is already sharded to tensor_shape[-1] // num_devices per device
    # After reduce_scatter, each device gets a unique 1/num_devices portion of the reduced result
    num_devices = mesh_device.get_num_devices()
    if dims[0] is not None or dims[1] is not None:
        input_width_per_device = tensor_shape[-1] // num_devices
    else:
        input_width_per_device = tensor_shape[-1]
    expected_width = input_width_per_device // num_devices
    assert (
        scattered_tensor.shape[-1] == expected_width
    ), f"Expected width {expected_width}, got {scattered_tensor.shape[-1]}"

    logger.info(f"✓ Successfully completed reduce_scatter with {topology} topology!")


@pytest.mark.parametrize("tensor_shape,dims", [((4 * 1096, 7 * 1024), (None, None)), ((4 * 1096, 256), (None, None))])
@pytest.mark.parametrize("num_links", [1, 2])
@pytest.mark.parametrize("num_cores", [10, None], ids=["10cores", "all_cores"])
@pytest.mark.parametrize(
    "device_params,topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}, ttnn.Topology.Ring),
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D}, ttnn.Topology.Linear),
    ],
    indirect=["device_params"],
    ids=["fabric_ring", "fabric_linear"],
)
def test_reduce_scatter_minimal_async(mesh_device, tensor_shape, dims, device_params, topology, num_links, num_cores):
    """
    Test reduce_scatter_minimal_async operation with different fabric configurations.
    This is the experimental async version that requires global semaphores and sub-device setup.
    """
    # Format parameters for Tracy signpost
    topology_name = "Ring" if topology == ttnn.Topology.Ring else "Linear"
    sharding_dim = "cols" if dims == (None, -1) else "rows" if dims == (-1, None) else str(dims)
    cores_desc = f"{num_cores}cores" if num_cores else "allcores"
    tracy_msg = f"ReduceScatterAsync shape={tensor_shape}, {topology_name}, {num_links}link(s), {cores_desc}, shard={sharding_dim}"
    signpost(tracy_msg)

    fabric_config = device_params.get("fabric_config", "DISABLED")
    logger.info(f"Testing reduce_scatter_minimal_async with fabric config: {fabric_config}, topology: {topology}")
    logger.info(f"Tensor shape: {tensor_shape}")
    logger.info(f"Mesh shape: {mesh_device.shape}")
    logger.info(f"Sharding dims: {dims}")

    # Check ethernet links availability
    check_ethernet_links(mesh_device, num_links)

    # Setup sub-device configuration for CCL
    if num_cores is not None:
        # Limited core count
        ccl_sub_device_crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))})
        logger.info(f"CCL operation limited to {num_cores} cores: (0,0) to ({num_cores-1},0)")
    else:
        # Use all available cores
        compute_grid_size = mesh_device.compute_with_storage_grid_size()
        ccl_sub_device_crs = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
        )
        logger.info(f"CCL operation using all cores: (0,0) to ({compute_grid_size.x-1},{compute_grid_size.y-1})")

    worker_sub_device = ttnn.SubDevice([ccl_sub_device_crs])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_stall_group = [worker_sub_device_id]

    # Create sub-device manager and set stall group
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group(sub_device_stall_group)

    # Create global semaphores required for async operation
    logger.info("Creating global semaphores for async operation")
    ccl_semaphore_handles = [ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0) for _ in range(3)]

    # Create sharded tensor across mesh
    torch_tensor, mesh_tensor = create_sharded_mesh_tensor(mesh_device, tensor_shape, dims)

    # Visualize the sharded tensor before reduce_scatter
    logger.info(f"Sharded tensor BEFORE reduce_scatter_minimal_async:")
    ttnn.visualize_tensor(mesh_tensor)

    # Perform reduce_scatter_minimal_async based on topology
    logger.info(f"Running reduce_scatter_minimal_async with topology: {topology}")
    if topology == ttnn.Topology.Ring:
        scattered_tensor = ttnn.experimental.reduce_scatter_minimal_async(
            mesh_tensor,
            dim=-1,  # Reduce and scatter along last dimension
            multi_device_global_semaphore=ccl_semaphore_handles,
            num_links=num_links,
            topology=ttnn.Topology.Ring,
            subdevice_id=worker_sub_device_id,
        )
    else:  # Linear (topology == ttnn.Topology.Linear)
        scattered_tensor = ttnn.experimental.reduce_scatter_minimal_async(
            mesh_tensor,
            dim=-1,  # Reduce and scatter along last dimension
            multi_device_global_semaphore=ccl_semaphore_handles,
            cluster_axis=1,  # Scatter along mesh column dimension
            num_links=num_links,
            topology=ttnn.Topology.Linear,
            subdevice_id=worker_sub_device_id,
        )

    # Synchronize devices to ensure operation completes
    logger.info("Synchronizing devices after async operation")
    ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)

    logger.info(f"Scattered tensor shape: {scattered_tensor.shape}")

    # Visualize the scattered tensor after reduce_scatter
    logger.info(f"Scattered tensor AFTER reduce_scatter_minimal_async ({topology}):")
    ttnn.visualize_tensor(scattered_tensor)

    # Verify the scattered tensor has expected shape
    num_devices = mesh_device.get_num_devices()
    if dims[0] is not None or dims[1] is not None:
        input_width_per_device = tensor_shape[-1] // num_devices
    else:
        input_width_per_device = tensor_shape[-1]
    expected_width = input_width_per_device // num_devices
    assert (
        scattered_tensor.shape[-1] == expected_width
    ), f"Expected width {expected_width}, got {scattered_tensor.shape[-1]}"

    # Cleanup: reset sub-device configuration
    mesh_device.reset_sub_device_stall_group()
    mesh_device.clear_loaded_sub_device_manager()

    logger.info(f"✓ Successfully completed reduce_scatter_minimal_async with {topology} topology!")
