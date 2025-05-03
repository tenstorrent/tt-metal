# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from functools import partial
import ttnn

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
class TestEltwiseBinary:
    @pytest.mark.parametrize("fn_kind", ["add", "sub", "mul", "squared_difference"])
    @pytest.mark.parametrize("in0_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
    @pytest.mark.parametrize("in1_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
    def test_run_eltwise_binary_ops(
        self,
        input_shapes,
        fn_kind,
        in0_dtype,
        in1_dtype,
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
                "dtype": [in0_dtype, in1_dtype],
                "input_mem_config": [input_mem_config, input_mem_config],
                "output_mem_config": output_mem_config,
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
        )

    @pytest.mark.parametrize(
        "fn_kind",
        [
            "bias_gelu",
        ],
    )
    def test_run_eltwise_binary_bias_ops(
        self,
        input_shapes,
        fn_kind,
        input_mem_config,
        output_mem_config,
        device,
        function_level_defaults,
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-100, high=100), torch.bfloat16)
        ] * len(input_shapes)

        test_args = list(generation_funcs.gen_default_dtype_layout_device(input_shapes))[0]
        test_args.update(
            {
                "input_mem_config": [input_mem_config, input_mem_config],
                "output_mem_config": output_mem_config,
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
        )

    @pytest.mark.parametrize("cmp_kind", ["lt", "gt", "lte", "gte", "ne", "eq"])
    def test_run_eltwise_binary_cmp_ops(
        self,
        input_shapes,
        input_mem_config,
        output_mem_config,
        cmp_kind,
        device,
        function_level_defaults,
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-100, high=100), torch.bfloat16)
        ] * len(input_shapes)
        test_args = list(generation_funcs.gen_default_dtype_layout_device(input_shapes))[0]
        test_args.update(
            {
                "input_mem_config": [input_mem_config, input_mem_config],
                "output_mem_config": output_mem_config,
            }
        )
        comparison_func = comparison_funcs.comp_equal
        run_single_pytorch_test(
            f"eltwise-{cmp_kind}",
            input_shapes,
            datagen_func,
            comparison_func,
            device,
            test_args,
        )

    @pytest.mark.parametrize(
        "log_kind, input_range",
        (
            ("logaddexp", {"low": -80, "high": 80}),
            ("ldexp", {"low": -60, "high": 60}),
            ("logaddexp2", {"low": -60, "high": 100}),
        ),
    )
    def test_run_eltwise_binary_log_ops(
        self, input_shapes, input_mem_config, output_mem_config, log_kind, input_range, device, function_level_defaults
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, **input_range), torch.bfloat16)
        ] * len(input_shapes)
        test_args = list(generation_funcs.gen_default_dtype_layout_device(input_shapes))[0]
        test_args.update(
            {"input_mem_config": [input_mem_config, input_mem_config], "output_mem_config": output_mem_config}
        )
        comparison_func = comparison_funcs.comp_pcc
        run_single_pytorch_test(
            f"eltwise-{log_kind}",
            input_shapes,
            datagen_func,
            comparison_func,
            device,
            test_args,
        )

    @pytest.mark.parametrize("logical_kind", ["logical_and", "logical_or"])
    def test_run_eltwise_binary_logical_ops(
        self,
        input_shapes,
        input_mem_config,
        output_mem_config,
        logical_kind,
        device,
        function_level_defaults,
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-100, high=100), torch.int32)
        ] * len(input_shapes)
        test_args = list(generation_funcs.gen_default_dtype_layout_device(input_shapes))[0]
        test_args.update(
            {
                "input_mem_config": [input_mem_config, input_mem_config],
                "output_mem_config": output_mem_config,
            }
        )
        comparison_func = comparison_funcs.comp_equal
        run_single_pytorch_test(
            f"eltwise-{logical_kind}",
            input_shapes,
            datagen_func,
            comparison_func,
            device,
            test_args,
        )
