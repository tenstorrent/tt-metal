# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from functools import partial
import tt_lib as ttl


from tests.tt_eager.python_api_testing.sweep_tests import (
    comparison_funcs,
    generation_funcs,
)
from tests.tt_eager.python_api_testing.sweep_tests.run_pytorch_ci_tests import (
    run_single_pytorch_test,
)

mem_configs = [
    ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
    ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
]


@pytest.mark.parametrize(
    "input_shapes",
    [
        [[1, 1, 32, 32]],  # Single core
        # [[1, 1, 300, 380]],  # multi core
        # [[1, 3, 320, 380]],  # multi core
        # [[1, 1, 32, 32]],  # Single core
        # [[1, 1, 320, 384]],  # Multi core
        # [[1, 3, 320, 384]],  # Multi core
    ],
)
@pytest.mark.parametrize(
    "input_mem_config",
    mem_configs,
)
@pytest.mark.parametrize(
    "dst_mem_config",
    mem_configs,
)
class TestCopy:
    def test_run_copy_op(
        self,
        input_shapes,
        input_mem_config,
        dst_mem_config,
        device,
        function_level_defaults,
    ):
        input_shapes = input_shapes * 2
        datagen_func = [
            generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-100, high=100), torch.bfloat16)
        ] * len(input_shapes)
        test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
        test_args["input_mem_config"] = [input_mem_config, dst_mem_config]
        del test_args["output_mem_config"]
        comparison_func = comparison_funcs.comp_equal
        run_single_pytorch_test(
            "copy",
            input_shapes,
            datagen_func,
            comparison_func,
            device,
            test_args,
        )

    def test_run_assign_unary_op(
        self,
        input_shapes,
        input_mem_config,
        dst_mem_config,
        device,
        function_level_defaults,
    ):
        input_shapes = input_shapes * 2
        datagen_func = [
            generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-100, high=100), torch.bfloat16)
        ] * len(input_shapes)
        test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
        test_args["input_mem_config"] = [input_mem_config, dst_mem_config]
        del test_args["output_mem_config"]
        comparison_func = comparison_funcs.comp_pcc
        run_single_pytorch_test(
            "eltwise-assign_unary",
            input_shapes,
            datagen_func,
            comparison_func,
            device,
            test_args,
        )

    def test_run_assign_binary_op(
        self,
        input_shapes,
        input_mem_config,
        dst_mem_config,
        device,
        function_level_defaults,
    ):
        input_shapes = input_shapes * 2
        datagen_func = [
            generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-100, high=100), torch.bfloat16)
        ] * len(input_shapes)
        test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
        test_args["input_mem_config"] = [input_mem_config, dst_mem_config]
        del test_args["output_mem_config"]
        comparison_func = comparison_funcs.comp_pcc
        run_single_pytorch_test(
            "eltwise-assign_binary",
            input_shapes,
            datagen_func,
            comparison_func,
            device,
            test_args,
        )

    def test_run_clone_op(
        self,
        input_shapes,
        input_mem_config,
        dst_mem_config,
        device,
        function_level_defaults,
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-100, high=100), torch.bfloat16)
        ]
        test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
        test_args["input_mem_config"] = [input_mem_config]
        test_args.update({"output_mem_config": dst_mem_config})
        comparison_func = comparison_funcs.comp_equal
        run_single_pytorch_test(
            "clone",
            input_shapes,
            datagen_func,
            comparison_func,
            device,
            test_args,
        )

    def test_run_typecast_op(
        self,
        input_shapes,
        input_mem_config,
        dst_mem_config,
        device,
        function_level_defaults,
    ):
        datagen_func = [
            generation_funcs.gen_func_with_cast(
                partial(generation_funcs.gen_constant, constant=torch.pi), torch.float32
            )
        ]
        test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
        test_args["input_mem_config"] = [input_mem_config]
        test_args.update({"output_mem_config": dst_mem_config})
        comparison_func = comparison_funcs.comp_pcc
        run_single_pytorch_test(
            "typecast",
            input_shapes,
            datagen_func,
            comparison_func,
            device,
            test_args,
        )
