from loguru import logger
import pytest
import torch
import ttnn
from tests.didt.op_test_base import OpTestBase
from models.utility_functions import skip_for_blackhole


class EltwiseOpTest(OpTestBase):
    def __init__(
        self,
        mesh_device,
        in0_shape,
        in1_shape,
        in0_mem_config,
        in1_mem_config,
        out_mem_config,
        in0_dtype,
        in1_dtype,
        out_dtype,
        in0_layout,
        in1_layout,
        loop_count=1000,
        determinism_check_enabled=False,
        determinism_check_iterations=False,
        eltwise_ttnn_function=ttnn.add,
    ):
        super().__init__(
            mesh_device,
            in0_shape,
            in1_shape,
            in0_mem_config,
            in1_mem_config,
            out_mem_config,
            in0_dtype,
            in1_dtype,
            out_dtype,
            in0_layout,
            in1_layout,
            program_config=None,
            compute_config=None,
            loop_count=loop_count,
            determinism_check_enabled=determinism_check_enabled,
            determinism_check_iterations=determinism_check_iterations,
        )
        self.eltwise_ttn_function = eltwise_ttnn_function

    # Override base class to generate random numbers in uniform distribution
    def generate_torch_activations(self, shape):
        return torch.rand(shape)

    # Override base class to generate random numbers in uniform distribution
    def generate_torch_weights(self, shape):
        return torch.rand(shape)

    # Override base class to execute eltwise ttnn operation instead of default ttnn matmul
    def run_device_operation(self):
        return self.eltwise_ttn_function(
            self.activations,
            self.weights,
            memory_config=self.out_mem_config,
            dtype=self.out_dtype,
        )


@skip_for_blackhole("Not tested for Blackhole")
@pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param(1, id="1chips"),
        pytest.param(2, id="2chips"),
        pytest.param(8, id="8chips"),
        pytest.param((8, 4), id="galaxy"),
    ],
    indirect=["mesh_device"],
)
def test_eltwise_add(
    mesh_device,
    iterations,
    determinism_check_iterations,
):
    # Initialize in1 and in2 shape, data type, memory configuration and layout
    in_shape = [640, 1280]
    in_dtype = ttnn.DataType.BFLOAT16
    in_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
    in_layout = ttnn.TILE_LAYOUT

    # Initialize output data type and memory configuration
    out_dtype = ttnn.DataType.BFLOAT16
    out_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)

    # Instance EltwiseOpTest class object and pass ttnn.add as eltwise ttnn function
    eltwise_test = EltwiseOpTest(
        mesh_device,
        in0_shape=in_shape,
        in1_shape=in_shape,
        in0_mem_config=in_mem_config,
        in1_mem_config=in_mem_config,
        out_mem_config=out_mem_config,
        in0_dtype=in_dtype,
        in1_dtype=in_dtype,
        out_dtype=out_dtype,
        in0_layout=in_layout,
        in1_layout=in_layout,
        loop_count=iterations,
        determinism_check_enabled=True if determinism_check_iterations > 0 else False,
        determinism_check_iterations=determinism_check_iterations,
        eltwise_ttnn_function=ttnn.add,
    )

    # Run eltwise operation test
    eltwise_test.run_op_test()


@skip_for_blackhole("Multi-chip Blackhole has not been tested")
@pytest.mark.parametrize("logical_chip_id", range(36), ids=[f"logical_chip_{i}_" for i in range(36)])
@pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param(1, id="1chips"),
        pytest.param(2, id="2chips"),
        pytest.param(8, id="8chips"),
        pytest.param((8, 4), id="galaxy"),
    ],
    indirect=["mesh_device"],
)
def test_specific_chip_eltwise_add(
    mesh_device,
    logical_chip_id,
    iterations,
    determinism_check_iterations,
):
    # Special case for galaxy:
    #   MeshDevice contains 32 chips, but their ids go from 4 - 35
    if len(mesh_device.get_device_ids()) == 32:
        assert (
            logical_chip_id >= 4 and logical_chip_id <= 35
        ), f"For TG configuration, logical chip id needs to be in range [4, 35] inclusive, but is {logical_chip_id}"
    else:
        assert len(mesh_device.get_device_ids()) > logical_chip_id, "Not enough devices!"

    test_eltwise_add(
        mesh_device.get_device(logical_chip_id),
        iterations,
        determinism_check_iterations,
    )


@skip_for_blackhole("Multi-board Blackhole has not been tested")
@pytest.mark.parametrize(
    "board_mesh_device",
    range(4),
    ids=[f"board_id_{i}" for i in range(4)],
    indirect=["board_mesh_device"],
)
def test_specific_board_eltwise_add(
    board_mesh_device,
    iterations,
    determinism_check_iterations,
):
    test_eltwise_add(
        board_mesh_device,
        iterations,
        determinism_check_iterations,
    )
