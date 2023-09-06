import pytest
import sys
import torch
from pathlib import Path
from functools import partial
import tt_lib as ttl

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../../../..")


from tests.tt_eager.python_api_testing.sweep_tests import comparison_funcs, generation_funcs
from tests.tt_eager.python_api_testing.sweep_tests.run_pytorch_ci_tests import run_single_pytorch_test
from tests.tt_eager.python_api_testing.sweep_tests.common import skip_for_wormhole_b0, is_wormhole_b0

shapes = [
        [[1, 1, 32, 32]],  # Single core
        [[1, 1, 320, 384]],  # Multi core
        [[1, 3, 320, 384]],  # Multi core
]
output_mem_configs = [
    ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM),
    ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1),
]
if is_wormhole_b0():
    del shapes[1:]
    del output_mem_configs[1:]

@pytest.mark.parametrize(
    "input_shapes",
    shapes
)
@pytest.mark.parametrize("pcie_slot", [0])
@pytest.mark.parametrize(
    "output_mem_config",
    output_mem_configs
)
def test_run_move_op(
    input_shapes,
    output_mem_config,
    pcie_slot,
    function_level_defaults,
):
    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_rand, low=-100, high=100), torch.bfloat16
        )
    ]
    test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
    test_args.update(
        {"output_mem_config": output_mem_config}
    )
    comparison_func = comparison_funcs.comp_equal
    run_single_pytorch_test(
        "move",
        input_shapes,
        datagen_func,
        comparison_func,
        pcie_slot,
        test_args,
    )
