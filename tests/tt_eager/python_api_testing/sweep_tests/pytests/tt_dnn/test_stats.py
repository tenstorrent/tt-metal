# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from functools import partial


from tests.tt_eager.python_api_testing.sweep_tests import comparison_funcs, generation_funcs
from tests.tt_eager.python_api_testing.sweep_tests.run_pytorch_ci_tests import run_single_pytorch_test
from models.utility_functions import is_wormhole_b0, skip_for_blackhole
import ttnn
import numpy as np

fns = [
    "std_hw",
    "mean_hw",
    "var_hw",
    "normalize_hw",
    "normalize_global",
]  # "std_global","var_global"]

WH = is_wormhole_b0() * -0.01
WH2 = 10 * WH
shapes = [
    [
        [2, 2, 32, 64],
        (
            0.99 + WH2,
            0.85 + WH,
            0.85,
            0.9,
            0.9,
        ),
    ],  # Single core
    [
        [4, 2, 64, 64],
        (
            0.97 + WH2,
            0.82 + WH,
            0.85,
            0.9,
            0.9,
        ),
    ],  # Multi core
    [
        [8, 6, 32, 96],
        (
            0.91 + WH,
            0.82 + WH,
            0.85,
            0.9,
            0.9,
        ),
    ],  # Multi core
]


@skip_for_blackhole("Mismatching on BH, see #12349")
@pytest.mark.parametrize(
    "input_shapes_and_pcc",
    shapes,
)
class TestStats:
    @pytest.mark.parametrize("fn_kind", fns)
    def test_run_stats_ops(self, input_shapes_and_pcc, fn_kind, device, function_level_defaults):
        if fn_kind in ["normalize_hw", "normalize_global"]:
            is_ttnn_op = True
        else:
            is_ttnn_op = False

        input_shapes, accepted_pcc = input_shapes_and_pcc
        input_shapes = [input_shapes]
        datagen_func = [
            generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-100, high=100), torch.float32)
        ]
        test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
        test_args.update({"input_shapes": input_shapes})
        accepted_pcc = accepted_pcc[fns.index(fn_kind)]
        comparison_func = partial(comparison_funcs.comp_pcc, pcc=accepted_pcc)
        run_single_pytorch_test(
            f"stats-{fn_kind}", input_shapes, datagen_func, comparison_func, device, test_args, ttnn_op=is_ttnn_op
        )


class TestEPS:
    def test_basic_gs(self):
        assert ttnn.device.EPS_GS == 0.001953125

    def test_basic_whb0(self):
        assert np.isclose(ttnn.device.EPS_WHB0, 1.19209e-07)

    def test_basic_bh(self):
        assert np.isclose(ttnn.device.EPS_BH, 1.19209e-07)
