from loguru import logger
import pytest
import torch
import ttnn
from tests.didt.op_test_base import OpTestBase


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

    def generate_torch_activations(self, shape):
        return torch.rand(shape)

    def generate_torch_weights(self, shape):
        return torch.rand(shape)

    def run_device_operation(self):
        return self.eltwise_ttn_function(
            self.activations,
            self.weights,
            memory_config=self.out_mem_config,
            dtype=self.out_dtype,
        )


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
    in_shape = [640, 1280]
    in_dtype = ttnn.DataType.BFLOAT16
    in_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
    in_layout = ttnn.TILE_LAYOUT

    out_dtype = ttnn.DataType.BFLOAT16
    out_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)

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

    eltwise_test.run_op_test()
