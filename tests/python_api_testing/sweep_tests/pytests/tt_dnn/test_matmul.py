import pytest
import sys
import torch
from pathlib import Path
from functools import partial

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")


from python_api_testing.sweep_tests import comparison_funcs, generation_funcs
from python_api_testing.sweep_tests.run_pytorch_ci_tests import run_single_pytorch_test


@pytest.mark.parametrize(
    "input_shapes, pcie_slot",
    (
        # Single core (won't be hit after padding is added for multicast)
        ([[1, 1, 32, 32], [1, 1, 32, 32]], 0),
        # Multi core (2% math util)
        ([[1, 2, 320, 1024], [1, 1, 1024, 384]], 0),
        # Multi core reuse (25% math util)
        ([[1, 2, 512, 1024], [1, 1, 1024, 512]], 0),
        # Multi core reuse multicast in0/in1 (25% math util)
        ([[1, 2, 5120, 1024], [1, 1, 1024, 6144]], 0),
        # Multi core reuse multicast in0 (25% math util)
        ([[1, 2, 512, 1024], [1, 1, 1024, 6144]], 0),
        # Multi core reuse multicast in1 (25% math util)
        ([[1, 2, 5120, 1024], [1, 1, 1024, 512]], 0),
        # Multi core reuse with padding (?% math util)
        ([[1, 2, 480, 1024], [1, 1, 1024, 480]], 0),
        # Multi core reuse multicast in0/in1 with padding (?% math util)
        ([[1, 2, 5088, 1024], [1, 1, 1024, 6112]], 0),
        # Multi core reuse multicast in0 with padding (?% math util)
        ([[1, 2, 480, 1024], [1, 1, 1024, 6112]], 0),
        # Multi core reuse multicast in1 with padding (?% math util)
        ([[1, 2, 5088, 1024], [1, 1, 1024, 480]], 0),
    ),
)
def test_run_matmul_test(input_shapes, pcie_slot, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_rand, low=-100, high=100), torch.float32
        )
    ] * 2
    comparison_func = partial(comparison_funcs.comp_pcc)
    run_single_pytorch_test(
        "matmul",
        input_shapes,
        datagen_func,
        comparison_func,
        pcie_slot,
    )


@pytest.mark.parametrize(
    "input_shapes, pcie_slot",
    (
        # Single core (won't be hit after padding is added for multicast)
        ([[1, 1, 32, 32], [1, 1, 32, 32]], 0),
        # Multi core (2% math util)
        ([[1, 2, 320, 1024], [1, 2, 1024, 384]], 0),
        # Multi core reuse (25% math util)
        ([[1, 2, 512, 1024], [1, 2, 1024, 512]], 0),
        # Multi core reuse multicast in0/in1 (25% math util)
        ([[1, 2, 5120, 1024], [1, 2, 1024, 6144]], 0),
        # Multi core reuse multicast in0 (25% math util)
        ([[1, 2, 512, 1024], [1, 2, 1024, 6144]], 0),
        # Multi core reuse multicast in1 (25% math util)
        ([[1, 2, 5120, 1024], [1, 2, 1024, 512]], 0),
        # Multi core reuse with padding (?% math util)
        ([[1, 2, 480, 1024], [1, 2, 1024, 480]], 0),
        # Multi core reuse multicast in0/in1 with padding (?% math util)
        ([[1, 2, 5088, 1024], [1, 2, 1024, 6112]], 0),
        # Multi core reuse multicast in0 with padding (?% math util)
        ([[1, 2, 480, 1024], [1, 2, 1024, 6112]], 0),
        # Multi core reuse multicast in1 with padding (?% math util)
        ([[1, 2, 5088, 1024], [1, 2, 1024, 480]], 0),
    ),
)
def test_run_bmm_test(input_shapes, pcie_slot, function_level_defaults):
    datagen_func = [
        generation_funcs.gen_func_with_cast(
            partial(generation_funcs.gen_rand, low=-100, high=100), torch.float32
        )
    ] * 2
    comparison_func = partial(comparison_funcs.comp_pcc)
    run_single_pytorch_test(
        "bmm",
        input_shapes,
        datagen_func,
        comparison_func,
        pcie_slot,
    )
