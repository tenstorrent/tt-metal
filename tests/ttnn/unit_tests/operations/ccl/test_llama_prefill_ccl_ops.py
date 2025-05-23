import ttnn
import pytest
from tests.ttnn.unit_tests.operations.ccl.test_new_all_gather_matmul import run_all_gather_impl
from tests.ttnn.unit_tests.operations.ccl.test_new_reduce_scatter import run_reduce_scatter_impl


@pytest.mark.parametrize(
    "num_devices, num_links, ag_output_shape, dim, layout, matmul_output_dim, max_in0_block_w, matmul_weights_dtype, ag_input_dtype, use_bias",
    [
        (8, 1, [1, 1, 4096, 2048], 3, ttnn.TILE_LAYOUT, 960, 2, ttnn.bfloat8_b, ttnn.bfloat8_b, True),
        #     (8, 1, [1, 1, 4096, 2560], 3, ttnn.TILE_LAYOUT, 960, 2, ttnn.bfloat8_b, ttnn.bfloat8_b, True),
        #     (8, 1, [1, 1, 4096, 7168], 3, ttnn.TILE_LAYOUT, 960, 2, ttnn.bfloat8_b, ttnn.bfloat8_b, True),
        #     (8, 1, [1, 1, 4096, 256], 3, ttnn.TILE_LAYOUT, 960, 2, ttnn.bfloat16, ttnn.bfloat16, True),
    ],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_ag, mem_config_mm",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        )
    ],
)
@pytest.mark.parametrize(
    "enable_trace",
    [
        True,
        # False,
    ],
)
@pytest.mark.parametrize(
    "use_non_fused",
    [
        True,
        # False,
    ],
)
@pytest.mark.parametrize(
    "device_params, use_legacy_allgather, all_gather_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 90112}, False, ttnn.Topology.Ring),
        # ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, False, ttnn.Topology.Linear),
        # (
        #    {"trace_region_size": 90112},
        #    True,
        #    ttnn.Topology.Ring,
        # ),
    ],
    indirect=["device_params"],
)
@pytest.mark.parametrize(
    "num_iters",
    [
        # 1,
        10,
    ],
)
def test_all_gather_async(
    t3k_mesh_device,
    num_devices,
    ag_output_shape,
    dim,
    num_links,
    ag_input_dtype,
    layout,
    matmul_output_dim,
    matmul_weights_dtype,
    max_in0_block_w,
    use_bias,
    mem_config_input,
    mem_config_ag,
    mem_config_mm,
    enable_trace,
    use_non_fused,
    use_legacy_allgather,
    use_program_cache,
    all_gather_topology,
    num_iters,
):
    run_all_gather_impl(
        t3k_mesh_device,
        num_devices,
        ag_output_shape,
        dim,
        num_links,
        ag_input_dtype,
        layout,
        matmul_output_dim,
        matmul_weights_dtype,
        max_in0_block_w,
        use_bias,
        mem_config_input,
        mem_config_ag,
        mem_config_mm,
        use_program_cache=use_program_cache,
        all_gather_topology=all_gather_topology,
        enable_trace=enable_trace,
        use_non_fused=use_non_fused,
        use_legacy_allgather=use_legacy_allgather,
        num_iters=num_iters,
    )


@pytest.mark.parametrize(
    "num_devices, num_links, rs_input_shape, dim, layout, rs_input_dtype",
    [
        (8, 1, [1, 1, 4096, 1280], 3, ttnn.TILE_LAYOUT, ttnn.bfloat8_b),
        (8, 1, [1, 1, 4096, 2048], 3, ttnn.TILE_LAYOUT, ttnn.bfloat8_b),
        (8, 1, [1, 1, 4096, 3584], 3, ttnn.TILE_LAYOUT, ttnn.bfloat8_b),
    ],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_rs",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        )
    ],
)
@pytest.mark.parametrize(
    "enable_trace, num_iters",
    [
        (True, 10),
        # (False, 1),
    ],
)
@pytest.mark.parametrize(
    "device_params, rs_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 90112}, ttnn.Topology.Ring),
        # ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, ttnn.Topology.Linear),
        # (
        #    {"trace_region_size": 90112},
        #    ttnn.Topology.Ring,
        # ),
    ],
    indirect=["device_params"],
)
def test_reduce_scatter_async(
    t3k_mesh_device,
    num_devices,
    num_links,
    rs_input_shape,
    dim,
    layout,
    rs_input_dtype,
    mem_config_input,
    mem_config_rs,
    enable_trace,
    num_iters,
    use_program_cache,
    rs_topology,
):
    run_reduce_scatter_impl(
        t3k_mesh_device,
        num_devices,
        rs_input_shape,
        dim,
        num_links,
        rs_input_dtype,
        layout,
        mem_config_input,
        mem_config_rs,
        use_program_cache=use_program_cache,
        rs_topology=rs_topology,
        enable_trace=enable_trace,
        num_iters=num_iters,
    )
