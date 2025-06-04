# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from typing import Union, Tuple

from tests.ttnn.utils_for_testing import tt_dtype_to_torch_dtype


def get_shard_grid_from_num_cores(ncores: Union[int, Tuple[int, int]], device) -> ttnn.CoreRangeSet:
    max_grid_size = (device.compute_with_storage_grid_size().y, device.compute_with_storage_grid_size().x)
    if isinstance(ncores, int):
        if ncores % max_grid_size[1] == 0:
            core_grid = ttnn.CoreGrid(y=ncores // max_grid_size[1], x=max_grid_size[1])
            grid_coord = ttnn.CoreCoord(core_grid.x - 1, core_grid.y - 1)
            return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
        else:
            if ncores < max_grid_size[1]:
                core_grid = ttnn.CoreGrid(y=1, x=ncores)
                grid_coord = ttnn.CoreCoord(core_grid.x - 1, 0)
                return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
            else:
                core_grid_1 = ttnn.CoreGrid(y=ncores // max_grid_size[1], x=max_grid_size[1])
                core_grid_2 = ttnn.CoreGrid(y=ncores // max_grid_size[1] + 1, x=ncores % max_grid_size[1])
                grid_coord_1 = ttnn.CoreCoord(core_grid_1.x - 1, core_grid_1.y - 1)
                grid_coord_2 = ttnn.CoreCoord(core_grid_2.x - 1, core_grid_2.y - 1)
                return ttnn.CoreRangeSet(
                    {
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord_1),
                        ttnn.CoreRange(ttnn.CoreCoord(0, grid_coord_2.y), grid_coord_2),
                    }
                )
    elif isinstance(ncores, tuple):
        ncores_h, ncores_w = ncores
        assert ncores_h <= max_grid_size[0]
        assert ncores_w <= max_grid_size[1]
        return ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(ncores_w - 1, ncores_h - 1),
                )
            }
        )
    else:
        raise ValueError("Invalid ncores")


def is_ttnn_float_type(tt_dtype) -> bool:
    match tt_dtype:
        case ttnn.bfloat16 | ttnn.float32 | ttnn.bfloat8_b | ttnn.bfloat4_b:
            return True
        case _:
            return False


def create_random_torch_tensors(tensor_shape: tuple, tt_dtype, num_tensors: int):
    torch.manual_seed(0)
    torch_dtype = tt_dtype_to_torch_dtype[tt_dtype]

    results = []
    for _ in range(num_tensors):
        if is_ttnn_float_type(tt_dtype):
            t = torch.rand(tensor_shape, dtype=torch_dtype)
        else:
            t = torch.randint(0, 100, tensor_shape, dtype=torch_dtype)
        results.append(t)

    return tuple(results)


def convert_torch_to_ttnn_tensor(
    torch_tensors: tuple,
    device,
    tt_dtype,
    layout,
    mem_config,
):
    ttnn_results = []
    for t in torch_tensors:
        tt_tensor = ttnn.from_torch(
            t,
            layout=layout,
            dtype=tt_dtype,
            memory_config=mem_config,
            device=device,
        )
        tt_tensor = ttnn.to_device(tt_tensor, device)
        ttnn_results.append(tt_tensor)

    return tuple(ttnn_results)
