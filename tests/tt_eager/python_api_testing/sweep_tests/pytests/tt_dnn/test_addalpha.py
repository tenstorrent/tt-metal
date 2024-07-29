# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from functools import partial
import tt_lib as ttl
import numpy as np


from tests.tt_eager.python_api_testing.sweep_tests import (
    comparison_funcs,
    generation_funcs,
)
from tests.tt_eager.python_api_testing.sweep_tests.run_pytorch_ci_tests import (
    run_single_pytorch_test,
)
from models.utility_functions import is_wormhole_b0

shapes = [
    [[1, 1, 32, 32], [1, 1, 32, 32]],  # Single core
    [[1, 1, 32, 32], [32, 1, 32, 32]],  # Single core
    [[64, 1, 32, 32], [1, 1, 32, 32]],  # Single core
    [[1, 1, 320, 384], [1, 1, 320, 384]],  # Multi core
    [[1, 3, 320, 384], [1, 3, 320, 384]],  # Multi core
]

input_mem_cfgs = generation_funcs.supported_mem_configs
output_mem_cfgs = generation_funcs.supported_mem_configs

if is_wormhole_b0():
    shapes = [
        shapes[0],
    ]
    input_mem_cfgs = [
        input_mem_cfgs[0],
    ]


@pytest.mark.parametrize(
    "input_shapes",
    shapes,
)
@pytest.mark.parametrize("input_mem_config", input_mem_cfgs)
@pytest.mark.parametrize("output_mem_config", output_mem_cfgs)
@pytest.mark.parametrize("fn_kind", ["addalpha"])
def test_run_addalpha(
    input_shapes,
    fn_kind,
    input_mem_config,
    output_mem_config,
    device,
    function_level_defaults,
):
    datagen_func = [
        generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-100, high=100), torch.float32)
    ] * len(input_shapes)
    test_args = list(generation_funcs.gen_default_dtype_layout_device(input_shapes))[0]
    test_args.update(
        {
            "input_mem_config": [input_mem_config, input_mem_config],
            "output_mem_config": output_mem_config,
            "alpha": np.random.randint(1, 100),
        }
    )
    comparison_func = comparison_funcs.comp_pcc
    run_single_pytorch_test(
        f"eltwise-{fn_kind}",
        input_shapes,
        datagen_func,
        comparison_func,
        device,
        test_args,
<<<<<<< HEAD
    )


shapes_w_output = [
    [[1, 1, 32, 32], [1, 1, 32, 32], [1, 1, 32, 32]],  # Single core
    [[1, 1, 32, 32], [32, 1, 32, 32], [32, 1, 32, 32]],  # Single core
    [[64, 1, 32, 32], [1, 1, 32, 32], [64, 1, 32, 32]],  # Single core
    [[1, 1, 320, 384], [1, 1, 320, 384], [1, 1, 320, 384]],  # Multi core
    [[1, 3, 320, 384], [1, 3, 320, 384], [1, 3, 320, 384]],  # Multi core
]


@pytest.mark.skip(reason="Issue: #10735, will be enabled once optional tensor support is added for ttnn.addalpha")
@pytest.mark.parametrize(
    "input_shapes",
    shapes_w_output,
)
@pytest.mark.parametrize("input_mem_config", input_mem_cfgs)
@pytest.mark.parametrize("fn_kind", ["addalpha"])
@pytest.mark.parametrize("pass_qid", [True, False])
def test_run_addalpha_optional_output(
    input_shapes,
    fn_kind,
    input_mem_config,
    device,
    pass_qid,
    function_level_defaults,
):
    datagen_func = [
        generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-100, high=100), torch.float32)
    ] * len(input_shapes)
    datagen_func.append(
        generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-80, high=80), torch.bfloat16)
    )
    test_args = list(generation_funcs.gen_default_dtype_layout_device(input_shapes))[0]
    test_args.update(
        {
            "input_mem_config": [input_mem_config, input_mem_config, input_mem_config],
            "alpha": np.random.randint(1, 100),
            "queue_id": pass_qid,
        }
    )
    comparison_func = comparison_funcs.comp_pcc
    run_single_pytorch_test(
        f"eltwise-{fn_kind}-optional",
        input_shapes,
        datagen_func,
        comparison_func,
        device,
        test_args,
=======
        ttnn_op=True,
>>>>>>> #10147: Migrated ops asinh to celu from tt-dnn/test_composite.py
    )
