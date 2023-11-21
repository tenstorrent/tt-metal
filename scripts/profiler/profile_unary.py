#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import time
import os
import sys
import torch
from pathlib import Path
from functools import partial
from math import pi
import copy
import tt_lib as ttl
import argparse

from tests.tt_eager.python_api_testing.sweep_tests import (
    comparison_funcs,
    generation_funcs,
)
from tests.tt_eager.python_api_testing.sweep_tests.run_pytorch_ci_tests import (
    run_single_pytorch_test,
)
from models.utility_functions import is_wormhole_b0

if is_wormhole_b0():
    shapes = [
        [[4, 20, 32, 32]],  # 80 cores
    ]
    arch = "Wh_B0"
else:
    shapes = [
        [[1, 1, 32, 32]],
    ]
    arch = "GS"

default_ops = [
    "sin",
    "cos",
    "tan",
    "gelu",
    "sqrt",
    "exp",
    "relu",
    "relu6",
    "sigmoid",
    "tanh",
    "square",
    "atan",
    "erfinv",
    "logical_not_unary",
    "rsqrt",
    "exp2",
    "expm1",
    "recip",
    "ltz",
    "gtz",
    "lez",
    "gez",
    "eqz",
    "nez",
    "log",
    "log2",
    "log10",
    "abs",
    "sign",
    "log_sigmoid",
    "asin",
    "acos",
    "erf",
    "erfc",
    "isinf",
    "isposinf",
    "isneginf",
    "isnan",
    "isfinite",
    "heaviside",
    "elu",
    "leaky_relu",
    # "clip",
    # "hardtanh"
]

REFERENCE_FOLDER = os.path.join(os.getcwd(), f"profiler_stats_{int(time.time())}")


def get_args():
    parser = argparse.ArgumentParser(usage="python3 profile_unary.py  --shape 1 1 32 32 --ops recip --memory L1'")
    parser.add_argument(
        "--path",
        help=f"path prefix for the profile dumps: {REFERENCE_FOLDER} (default)",
        default=REFERENCE_FOLDER,
        type=str,
    )
    parser.add_argument("--memory", help="choices of memory : DRAM (default), L1", default=config, required=False)
    parser.add_argument("--ops", dest="unary_ops", nargs="+", help="", default=default_ops, required=False)
    parser.add_argument("--shape", dest="shape", type=int, nargs="+", help="", default=shapes[0][0], required=False)
    return parser.parse_args()


def comp_any(golden, calculated, pcc=0.99):
    return True, "all is well"


def run_eltwise_unary_ops(
    args: argparse.Namespace,
    device,
    input_shapes,
    fn_kind,
    input_mem_config,
    output_mem_config,
    function_level_defaults={},
):
    ttl.profiler.set_profiler_location(f"{args.path}/{fn_kind}-log.csv")

    data_type = data_type_mapping.get(fn_kind, torch.bfloat16)
    datagen_func = [
        generation_funcs.gen_func_with_cast(partial(generation_funcs.gen_rand, low=-100, high=100), data_type)
    ]

    test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
    test_args.update(
        {
            "input_mem_config": [input_mem_config],
            "output_mem_config": output_mem_config,
        }
    )

    if fn_kind in ["gelu", "rsqrt", "erf", "erfc", "exp"]:
        test_args.update(
            {
                "fast_and_appx": True,
                "fast_and_approx": True,
            }
        )

    if fn_kind in op_dict and op_dict[fn_kind] is not None:
        test_args.update(op_dict[fn_kind])
    comparison_func = comp_any
    for repetitions in range(no_iterations):
        run_single_pytorch_test(
            f"eltwise-{fn_kind}",
            input_shapes,
            datagen_func,
            comparison_func,
            device,
            test_args,
        )


# Default data_type bfloat16
data_type_mapping = {
    "relu": torch.float32,
    "sigmoid": torch.float32,
    "square": torch.float32,
    "tanh": torch.float32,
    "relu6": torch.float32,
    "add1": torch.float32,
    "deg2rad": torch.float32,
    "rad2deg": torch.float32,
    "gelu": torch.float32,
    "log": torch.float32,
    "log2": torch.float32,
    "log10": torch.float32,
    "sin": torch.float32,
    "cos": torch.float32,
    "clip": torch.float32,
    "hardtanh": torch.float32,
    "elu": torch.float32,
    "leaky relu": torch.float32,
    "log_sigmoid": torch.float32,
}

# update this dict if ops have any scalar values
op_dict = {
    "elu": {"alpha": 0.5},
    "heaviside": {"scalar": 0.5},
    "leaky_relu": {"negative_slope": -0.5},
    "clip": {"low": -2.0, "high": 2.0},
    "hardtanh": {"low": -2.0, "high": 2.0},
}

# update with required config to run "DRAM" or "L1"
config = "L1"

# number of times you need to run the test case
no_iterations = 1000


def main():
    args = get_args()
    input_mem_cfgs = copy.copy(generation_funcs.supported_mem_configs)[args.memory == "L1"]
    output_mem_cfgs = copy.copy(generation_funcs.supported_mem_configs)[args.memory == "L1"]
    device = ttl.device.CreateDevice(0)

    for input_shapes in [[args.shape]]:
        for fn_kind in args.unary_ops:
            run_eltwise_unary_ops(args, device, input_shapes, fn_kind, input_mem_cfgs, output_mem_cfgs, None)


if __name__ == "__main__":
    main()
