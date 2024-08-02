# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple, List
import torch
import ttnn
import random

from tests.ttnn.utils_for_testing import check_with_pcc
from models.utility_functions import torch_random


def skip(**_) -> Tuple[bool, Optional[str]]:
    return False, None


def xfail(config, **_) -> Tuple[bool, Optional[str]]:
    return False, None


def round_to_nearest(b: int, round_to: int) -> int:
    return (b + round_to // 2) // round_to * round_to


def generate_random_numbers(a: int, round_to=None) -> List[int]:
    numbers = []
    remaining_sum = a
    while remaining_sum > 0:
        num = random.randint(1, remaining_sum)
        if round_to is not None:
            num = round_to_nearest(num, round_to)
        numbers.append(num)
        remaining_sum -= num
    return numbers


def dtype_to_rounding_mode(dtype):
    if dtype == ttnn.bfloat16:
        return 2
    elif dtype == ttnn.bfloat8_b:
        return 4
    return 1


def generate_config(rank, max_random_size, split_dim, layout, dtype):
    base_shape = []
    base_shape = [random.randint(1, max_random_size) for _ in range(rank)]

    round_val = dtype_to_rounding_mode(dtype)

    if layout == ttnn.ROW_MAJOR_LAYOUT and base_shape[-1] % round_val != 0:
        rem = base_shape[-1] % round_val
        base_shape[-1] = (base_shape[-1] + rem) % max_random_size
        if base_shape[-1] == 0:
            base_shape[-1] = round_val

    splits = generate_random_numbers(
        base_shape[split_dim],
        round_to=round_val if (layout == ttnn.ROW_MAJOR_LAYOUT and base_shape[-1] % round_val != 0) else None,
    )
    return base_shape, splits


def generate_configurations(ranks, layouts, dtypes, configs_per_variant=1):
    random.seed(0)

    configs = []

    for _ in range(configs_per_variant):
        for rank in ranks:
            for layout in layouts:
                if rank < 2 and layout == ttnn.TILE_LAYOUT:
                    continue
                for dtype in dtypes:
                    if dtype == ttnn.bfloat8_b and layout == ttnn.ROW_MAJOR_LAYOUT:
                        continue
                    for split_dim in range(rank):
                        base_and_variable = generate_config(rank, 48, split_dim, layout, dtype)
                        config = {
                            "rank": rank,
                            "split_dim": split_dim,
                            "shape": base_and_variable[0],
                            "splits": base_and_variable[1],
                            "layout": layout,
                            "dtype": dtype,
                        }
                        configs.append(config)

    return configs


def known_configs(configs, **_):
    known_working = [
        [1, 2, 32, 64],
        [1, 2, 64, 64],
        [1, 2, 64, 128],
        [1, 2, 1024, 128],
        [1, 2, 256, 2560],
        [1, 2, 1024, 2560],
        [1, 2, 256, 5120],
        [1, 2, 64, 10240],
        [1, 2, 16, 10240],
    ]

    for shape in known_working:
        config2 = {
            "rank": len(shape),
            "split_dim": 2,
            "shape": shape,
            "splits": 2,
            "layout": ttnn.TILE_LAYOUT,
            "dtype": ttnn.bfloat16,
        }

        config3 = {
            "rank": len(shape),
            "split_dim": 3,
            "shape": shape,
            "splits": 2,
            "layout": ttnn.TILE_LAYOUT,
            "dtype": ttnn.bfloat16,
        }

        configs.append(config2)
        configs.append(config3)

    return configs


configs = generate_configurations(
    [1, 2, 3, 4], [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], [ttnn.bfloat16, ttnn.bfloat8_b], 5
)

# RNG splits don't work because we only have split in 2 chunks, so we need to add some known working configs
# this can be commented out and removed once our split implementation supports more than 2 chunks
configs = known_configs(configs)

parameters = {
    "config": configs,
    "memory_config": [
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.L1_MEMORY_CONFIG,
        ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
    ],
}


def run(config, memory_config, *, device) -> Tuple[bool, Optional[str]]:
    shape = config["shape"]
    splits = config["splits"]
    split_dim = config["split_dim"]
    layout = config["layout"]
    dtype = config["dtype"]

    torch_input_tensor = torch_random(shape, -0.1, 0.1, dtype=torch.bfloat16)
    torch_output_tensors = torch.split(torch_input_tensor, splits, dim=split_dim)

    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor, layout=layout, device=device, dtype=dtype, memory_config=memory_config
    )
    ttnn_output_tensors = ttnn.split(ttnn_input_tensor, splits, dim=split_dim)

    output_tensors = [ttnn.to_torch(ttnn_output_tensor) for ttnn_output_tensor in ttnn_output_tensors]

    if len(torch_output_tensors) != len(output_tensors):
        return (
            False,
            f"Number of tensors do not match: ttnn length {len(output_tensors)} vs pytorch length {len(torch_output_tensors)}",
        )

    shape_mismatch_exceptions = ""
    for i in range(len(torch_output_tensors)):
        if torch_output_tensors[i].shape != output_tensors[i].shape:
            shape_mismatch_exceptions += (
                f"tensor {i}: ttnn shape {output_tensors[i].shape} vs pytorch shape {torch_output_tensors[i].shape} "
            )
    if len(shape_mismatch_exceptions) > 0:
        return (
            False,
            f"Shapes do not match: " + shape_mismatch_exceptions,
        )

    pcc_mismatch_exceptions = ""
    for i in range(len(torch_output_tensors)):
        pcc_passed, pcc_message = check_with_pcc(torch_output_tensors[i], output_tensors[i], 0.9999)
        if not pcc_passed:
            pcc_mismatch_exceptions += f"tensor {i}: {pcc_message} "
    if len(pcc_mismatch_exceptions) > 0:
        return (
            False,
            f"PCC mismatch: " + pcc_mismatch_exceptions,
        )
    return True, None
