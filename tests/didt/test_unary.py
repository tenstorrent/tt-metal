from loguru import logger
import pytest
import torch
import ttnn
from tests.didt.op_test_base import OpTestBase


class UnaryOpTest(OpTestBase):
    def __init__(
        self,
        mesh_device,
        in0_shape,
        in0_mem_config,
        in0_dtype,
        in0_layout,
        loop_count=1000,
        determinism_check_enabled=False,
        determinism_check_iterations=False,
        unary_ttnn_function=ttnn.sqrt,
    ):
        super().__init__(
            mesh_device,
            in0_shape,
            in1_shape=None,
            in0_mem_config=in0_mem_config,
            in1_mem_config=None,
            out_mem_config=None,
            in0_dtype=in0_dtype,
            in1_dtype=None,
            out_dtype=None,
            in0_layout=in0_layout,
            in1_layout=None,
            program_config=None,
            compute_config=None,
            loop_count=loop_count,
            determinism_check_enabled=determinism_check_enabled,
            determinism_check_iterations=determinism_check_iterations,
            unary_op=True,
        )
        self.unary_ttnn_function = unary_ttnn_function

    def generate_torch_activations(self, shape):
        return torch.rand(shape)

    def generate_torch_weights(self, shape):
        return torch.rand(shape)

    def run_device_operation(self):
        return self.unary_ttnn_function(
            self.activations,
            memory_config=self.out_mem_config,
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
def test_unary_sqrt(
    mesh_device,
    iterations,
    determinism_check_iterations,
):
    in_shape = [640, 1280]
    in_dtype = ttnn.DataType.BFLOAT16
    in_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
    in_layout = ttnn.TILE_LAYOUT

    unary_test = UnaryOpTest(
        mesh_device,
        in0_shape=in_shape,
        in0_mem_config=in_mem_config,
        in0_dtype=in_dtype,
        in0_layout=in_layout,
        loop_count=iterations,
        determinism_check_enabled=True if determinism_check_iterations > 0 else False,
        determinism_check_iterations=determinism_check_iterations,
        unary_ttnn_function=ttnn.sqrt,
    )

    unary_test.run_op_test()
