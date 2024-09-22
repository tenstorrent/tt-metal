# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import random
from itertools import permutations, product
from functools import lru_cache
import ttnn
import numpy as np
from tt_lib.utils import _nearest_32 as nearest_32, tilize
from loguru import logger


# torch.testing.get_all_dtypes()
supported_dtypes = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "int32": torch.int32,
    "cfloat": torch.cfloat,
    "chalf": torch.chalf,
}

supported_tt_dtypes = [ttnn.bfloat16]

supported_tt_layouts = [
    ttnn.ROW_MAJOR_LAYOUT,
    ttnn.TILE_LAYOUT,
]

supported_tt_buffer_types = [
    ttnn.BufferType.DRAM,
    ttnn.BufferType.L1,
]

supported_mem_configs = [
    ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
    ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
]


# Wrapper around gen functions to include casting
def gen_func_with_cast(gen_func, dtype, tilize_input=False):
    return lambda size: tilize(gen_func(size).to(dtype)) if tilize_input else gen_func(size).to(dtype)


def gen_func_with_cast_tt(gen_func, dtype):
    def tensor_to_dtype(x):
        if dtype == ttnn.bfloat16:
            x = x.to(torch.bfloat16)

        elif dtype == ttnn.bfloat8_b:
            tt_tensor = ttnn.from_torch(
                x, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=None, memory_config=None
            )

            x = ttnn.to_torch(tt_tensor)

        elif dtype == ttnn.uint16:
            x = x.to(torch.int16)

        elif dtype == ttnn.uint32:
            x = x.to(torch.int32)

        elif dtype == ttnn.int32:
            x = x.to(torch.int32)

        else:
            logger.warning(f"Unknown dtype {dtype} passed to gen_func_with_cast_tt")

        return x

    def func(size):
        x = gen_func(size)

        if x.dtype == torch.cfloat:
            x.real = tensor_to_dtype(x.real).to(torch.float)
            x.imag = tensor_to_dtype(x.imag).to(torch.float)
            return x

        return tensor_to_dtype(x)

    return func


def gen_zeros(size):
    return torch.zeros(size)


def gen_ones(size):
    return torch.ones(size)


def gen_constant(size, constant=1.0):
    return torch.full(size, constant)


def gen_rand(size, low=0, high=100):
    return torch.Tensor(size=size).uniform_(low, high)


def gen_rand_infinite(size, low=-100, high=100):
    x = torch.rand(size=size)
    x[x <= 0.33] = float("inf")
    x[(x > 0.33) & (x <= 0.66)] = float("-inf")
    x[x > 0.66] = x[x > 0.66] * (high - low) + low
    return x


def gen_rand_complex(size, low=0, high=100):
    real = torch.Tensor(size=size).uniform_(low, high).to(torch.bfloat16).to(torch.float)
    imag = torch.Tensor(size=size).uniform_(low, high).to(torch.bfloat16).to(torch.float)

    torch_x = torch.complex(real, imag)

    return torch_x


def gen_rand_inf(size, low=-100, high=100):
    x = torch.rand(size=size)
    x[x <= 0.25] = float("inf")
    x[(x > 0.25) & (x <= 0.5)] = float("-inf")
    x[(x > 0.5) & (x <= 0.75)] = float("nan")
    x[x > 0.75] = x[x > 0.75] * (high - low) + low
    return x


def gen_bin(size, probabilityones=0.5):
    element_count = 1
    for i in size:
        element_count = element_count * i
    raw = torch.zeros(element_count)
    raw[: int(probabilityones * element_count)] = 1
    ridx = torch.randperm(element_count)  # a random permutation of the entries
    mask = torch.reshape(raw[ridx], size)
    return mask


def gen_linspace(size, low=0, high=100):
    lsteps = size[0] * size[1] * size[2] * size[3]
    return torch.linspace(low, high, lsteps).reshape(size)


def gen_rand_symmetric(size, low=0, high=100):
    signs = torch.randint(0, 2, size) * 2 - 1
    return torch.Tensor(size=size).uniform_(low, high) * signs


def gen_rand_along_dim(size, low=0, high=100, dim=-1):
    numel = torch.Size(size).numel()

    # Create a flat tensor and generate a subrange of random numbers each the size of specified dimension
    output = torch.zeros(numel)
    num_subel = size[dim]
    for i in range(0, numel, num_subel):
        subrange = torch.Tensor(size=(2,)).uniform_(low, high)
        output[i : i + num_subel] = torch.Tensor(size=(num_subel,)).uniform_(torch.min(subrange), torch.max(subrange))

    # Reshape the flat tensor where the specified dim and last dim are swapped
    swapped_size = size.copy()
    swapped_size[dim], swapped_size[-1] = size[-1], size[dim]
    output = output.reshape(swapped_size)

    # Swap the last and specified dim so that the output size matches the specified size
    output = output.transpose(dim, -1)
    return output


def gen_randint(size, low=0, high=2):
    return torch.randint(low, high, size, dtype=torch.float32)


def gen_scaled_dirichlet_along_dim(size, start=1, step=1, max=1, dim=-1):
    assert start <= max, "start must be <= max"

    numel = torch.Size(size).numel()

    num_subel = 32 * 32

    num_tiles = numel // num_subel

    # Get tile shape with correct number of dimensions
    tile_shape = [1] * len(size)
    tile_shape[-2], tile_shape[-1] = 32, 32

    # RNG that sums to 1
    dirichlet = torch.distributions.dirichlet.Dirichlet(torch.ones(size=tile_shape))

    scale = start
    tiles = []
    for i in range(num_tiles):
        scaled_sum = scale * (i % 2 * 2 - 1)
        tiles.append(dirichlet.sample() * scaled_sum)
        scale += step
        if scale > max:
            scale = start

    for d in reversed(range(len(size))):
        tiles = torch.concat(tiles, dim=d).split(size[d], dim=d)

    output = torch.concat(tiles)

    # Swap the last and specified dim so that the output size matches the specified size
    output = output.transpose(dim, -1)

    return output


def gen_scaled_dirichlet_per_tile(size, start=1, step=1, max=1):
    assert start <= max, "start must be <= max"

    numel = torch.Size(size).numel()

    num_subel = 32 * 32

    num_tiles = numel // num_subel

    # Get tile shape with correct number of dimensions
    tile_shape = [1] * len(size)
    tile_shape[-2], tile_shape[-1] = 32, 32

    # RNG that sums to 1
    dirichlet = torch.distributions.dirichlet.Dirichlet(torch.ones(size=(num_subel,)))

    scale = start
    tiles = []
    for i in range(num_tiles):
        scaled_sum = scale * (i % 2 * 2 - 1)
        tiles.append((dirichlet.sample() * scaled_sum).reshape(tile_shape))
        scale += step
        if scale > max:
            scale = start

    for d in reversed(range(len(size))):
        tiles = torch.concat(tiles, dim=d).split(size[d], dim=d)

    output = torch.concat(tiles)
    return output


def gen_checkerboard(size, low=0, high=100):
    value = torch.Tensor(1).uniform_(low, high).item()

    # Create a checkerboard of alternating signed values
    checkerboard = torch.full([size[-2], size[-1]], value)
    checkerboard[1::2, ::2] = value * -1
    checkerboard[::2, 1::2] = value * -1

    # Duplicate across batch dims
    output = torch.tile(checkerboard, (*size[:-2], 1, 1))

    return output


def gen_arange(size):
    numel = torch.Size(size).numel()
    return torch.arange(numel, dtype=torch.float32).reshape(size)


def gen_identity(size):
    return torch.eye(size[0], size[1])


###################################################
#################### test_args ####################
###################################################


def gen_tensor_pad_args(
    input_shapes, supported_dtypes=None, supported_layouts=None, mem_configs=None, do_sanitize_args=False, coregrid=[]
):
    assert len(input_shapes) == 1
    assert len(input_shapes[0]) == 4
    test_args = {}

    pad_sizes = (64, 64, 64, 64)
    output_tensor_shape = [random.randint(input_shapes[0][i], input_shapes[0][i] + pad_sizes[i]) for i in range(4)]
    input_tensor_start = [random.randint(0, output_tensor_shape[i] - input_shapes[0][i]) for i in range(4)]
    pad_value = random.uniform(-100, 100)
    # Cast to bfloat16 then back to float for exact match
    pad_value = torch.Tensor([pad_value]).to(torch.bfloat16).to(torch.float).item()

    test_args.update(
        {
            "output_tensor_shape": output_tensor_shape,
            "input_tensor_start": input_tensor_start,
            "pad_value": pad_value,
            "dtype": [ttnn.bfloat16],
            "layout": [ttnn.ROW_MAJOR_LAYOUT],
            "input_mem_config": [None],
            "output_mem_config": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        }
    )

    return [test_args]


def gen_tensor_unpad_args(
    input_shapes, supported_dtypes=None, supported_layouts=None, mem_configs=None, do_sanitize_args=True, coregrid=[]
):
    assert len(input_shapes) == 1
    assert len(input_shapes[0]) == 4
    test_args = {}
    output_tensor_start = [random.randint(0, input_shapes[0][i] - 1) for i in range(4)]
    output_tensor_end = [random.randint(output_tensor_start[i] + 1, input_shapes[0][i]) for i in range(4)]

    test_args.update(
        {
            "output_tensor_start": output_tensor_start,
            "output_tensor_end": output_tensor_end,
            "dtype": [ttnn.bfloat16],
            "layout": [ttnn.ROW_MAJOR_LAYOUT],
            "input_mem_config": [None],
            "output_mem_config": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        }
    )

    return [test_args]


def gen_pad_to_tile_args(
    input_shapes, supported_dtypes=None, supported_layouts=None, mem_configs=None, do_sanitize_args=True, coregrid=[]
):
    assert len(input_shapes) == 1
    assert len(input_shapes[0]) == 4

    pad_value = random.uniform(-100, 100)
    # Cast to bfloat16 then back to float for exact match
    pad_value = torch.Tensor([pad_value]).to(torch.bfloat16).to(torch.float).item()

    test_args = {
        "pad_value": pad_value,
        "dtype": [ttnn.bfloat16],
        "layout": [ttnn.ROW_MAJOR_LAYOUT],
        "input_mem_config": [None],
        "output_mem_config": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
    }

    return [test_args]


def gen_unpad_from_tile_args(
    input_shapes, supported_dtypes=None, supported_layouts=None, mem_configs=None, do_sanitize_args=True, coregrid=[]
):
    assert len(input_shapes) == 1
    assert len(input_shapes[0]) == 4
    assert input_shapes[0][-2] % 32 == 0
    assert input_shapes[0][-1] % 32 == 0

    output_tensor_shape = [
        *input_shapes[0][:-2],
        random.randint(input_shapes[0][-2] - 32 + 1, input_shapes[0][-2]),
        random.randint(input_shapes[0][-1] - 32 + 1, input_shapes[0][-1]),
    ]

    test_args = {
        "output_tensor_shape": output_tensor_shape,
        "dtype": [ttnn.bfloat16],
        "layout": [ttnn.ROW_MAJOR_LAYOUT],
        "input_mem_config": [None],
        "output_mem_config": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
    }

    return [test_args]


def gen_default_dtype_layout_device(
    input_shapes, dtypes=None, layouts=None, mem_configs=None, do_sanitize_args=False, coregrid=[]
):
    dtype = []
    layout = []
    input_mem_config = []

    for input_shape in input_shapes:
        dtype.append(ttnn.bfloat16)
        input_mem_config.append(ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM))

        if input_shape[-2] % 32 == 0 and input_shape[-1] % 32 == 0:
            layout.append(ttnn.TILE_LAYOUT)
        else:
            layout.append(ttnn.ROW_MAJOR_LAYOUT)

    return [
        {
            "dtype": dtype,
            "layout": layout,
            "input_mem_config": input_mem_config,
            "output_mem_config": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        }
    ]


def gen_default_dtype_layout_rm_device(
    input_shapes, dtypes=None, layouts=None, mem_configs=None, do_sanitize_args=False, coregrid=[]
):
    return [
        {
            "dtype": [ttnn.bfloat16] * len(input_shapes),
            "layout": [ttnn.ROW_MAJOR_LAYOUT] * len(input_shapes),
            "input_mem_config": [ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)]
            * len(input_shapes),
            "output_mem_config": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            "use_multicore": True,
        }
    ]


def sanitize_args(input_shapes, input_setup):
    for i in range(len(input_shapes)):
        shape = input_shapes[i]

        if (
            (
                input_setup[i]["layout"] == ttnn.TILE_LAYOUT and (shape[-2] % 32 != 0 or shape[-1] % 32 != 0)
            )  # Shape cannot be tilized
            or (
                input_setup[i]["layout"] == ttnn.ROW_MAJOR_LAYOUT
                and input_setup[i]["input_mem_config"] != None
                and shape[-1] % 2 != 0
            )  # Shape cannot be placed as row major on device
            or (
                input_setup[i]["dtype"] == ttnn.bfloat8_b and input_setup[i]["layout"] != ttnn.TILE_LAYOUT
            )  # BFLOAT8_B must be tile layout
        ):
            return None
    return input_setup


def gen_dtype_layout_device(
    input_shapes,
    dtypes=[supported_tt_dtypes],
    layouts=[supported_tt_layouts],
    mem_configs=[supported_mem_configs],  # mem_configs[-1] is output_mem_config
    do_sanitize_args=True,
    coregrid=[],
):
    # last buffer_types option is for output buffer
    dtype_mem_config_layouts = []

    for i in range(len(input_shapes)):
        dtype_mem_config_layout = []

        for dtype, layout, input_mem_config in product(
            dtypes[i],
            layouts[i],
            mem_configs[i],
        ):
            dtype_mem_config_layout.append({"dtype": dtype, "layout": layout, "input_mem_config": input_mem_config})

        dtype_mem_config_layouts.append(dtype_mem_config_layout)

    result = []

    for out_mem_config in mem_configs[-1]:
        for dtype_mem_config_layout_combination in product(*dtype_mem_config_layouts):
            if do_sanitize_args:
                out = sanitize_args(input_shapes, dtype_mem_config_layout_combination)
            else:
                out = 1

            if out is not None:
                dtype = []
                layout = []
                input_mem_config = []

                for x in dtype_mem_config_layout_combination:
                    dtype.append(x["dtype"])
                    layout.append(x["layout"])
                    input_mem_config.append(x["input_mem_config"])

                result.append(
                    {
                        "dtype": dtype,
                        "layout": layout,
                        "input_mem_config": input_mem_config,
                        "output_mem_config": out_mem_config,
                    }
                )

    return result


def sanitize_args_layernorm(
    input_shapes, input_setup, runtime_tile_padding_layernorm=False, runtime_tile_padding_add_layernorm=False
):
    for i in range(len(input_shapes)):
        shape = input_shapes[i]
        if (
            (
                input_setup[i]["layout"] == ttnn.TILE_LAYOUT
                and (
                    (
                        shape[2] % 32 != 0
                        and not runtime_tile_padding_layernorm
                        and not runtime_tile_padding_add_layernorm
                    )
                    or (runtime_tile_padding_layernorm and i > 0 and shape[2] != 1)
                    or (runtime_tile_padding_add_layernorm and i > 1 and shape[2] != 1)
                    or (shape[3] % 32 != 0)
                )
            )  # Shape cannot be tilized
            or (
                input_setup[i]["layout"] == ttnn.ROW_MAJOR_LAYOUT
                and input_setup[i]["input_mem_config"] != None
                and shape[3] % 2 != 0
            )  # Shape cannot be placed as row major on device
            or (
                input_setup[i]["dtype"] == ttnn.bfloat8_b and input_setup[i]["layout"] != ttnn.TILE_LAYOUT
            )  # BFLOAT8_B must be tile layout
        ):
            return None
    return input_setup


def gen_dtype_layout_device_layernorm(
    input_shapes,
    dtypes=[supported_tt_dtypes],
    layouts=[supported_tt_layouts],
    mem_configs=[supported_mem_configs],
    runtime_tile_padding_layernorm=True,
    runtime_tile_padding_add_layernorm=False,
):
    # last buffer_types option is for output buffer
    input_setups = []

    for i in range(len(input_shapes)):
        input_setup = []

        for dtype, layout, input_mem_config in product(
            dtypes[i],
            layouts[i],
            mem_configs[i],
        ):
            input_setup.append({"dtype": dtype, "layout": layout, "input_mem_config": input_mem_config})

        input_setups.append(input_setup)

    result = []

    for out_mem_config in mem_configs[-1]:
        for input_setup_combination in product(*input_setups):
            out = sanitize_args_layernorm(
                input_shapes,
                input_setup_combination,
                runtime_tile_padding_layernorm,
                runtime_tile_padding_add_layernorm,
            )

            if out is not None:
                dtype = []
                layout = []
                input_mem_config = []

                for x in input_setup_combination:
                    dtype.append(x["dtype"])
                    layout.append(x["layout"])
                    input_mem_config.append(x["input_mem_config"])

                result.append(
                    {
                        "dtype": dtype,
                        "layout": layout,
                        "input_mem_config": input_mem_config,
                        "output_mem_config": out_mem_config,
                    }
                )

    return result


def gen_layernorm_args(
    input_shapes,
    dtypes=[supported_tt_dtypes],
    layouts=[supported_tt_layouts],
    mem_configs=[supported_mem_configs],
    do_sanitize_args=False,
    coregrid=[],
):
    return gen_dtype_layout_device_layernorm(
        input_shapes,
        dtypes,
        layouts,
        mem_configs,
        runtime_tile_padding_layernorm=True,
        runtime_tile_padding_add_layernorm=False,
    )


def gen_add_layernorm_args(
    input_shapes,
    dtypes=[supported_tt_dtypes],
    layouts=[supported_tt_layouts],
    mem_configs=[supported_mem_configs],
    do_sanitize_args=False,
    coregrid=[],
):
    return gen_dtype_layout_device_layernorm(
        input_shapes,
        dtypes,
        layouts,
        mem_configs,
        runtime_tile_padding_layernorm=False,
        runtime_tile_padding_add_layernorm=True,
    )


def gen_permute_args(
    input_shapes,
    dtypes=[supported_tt_dtypes],
    layouts=[supported_tt_layouts],
    mem_configs=[supported_mem_configs],
    do_sanitize_args=True,
    coregrid=[],
):
    dim_to_permute = []

    for i in range(len(input_shapes[0])):
        dim_to_permute.append(i)

    for permute_dims in permutations(dim_to_permute):
        for input_info in gen_dtype_layout_device(
            input_shapes,
            dtypes,
            layouts,
            mem_configs,
            do_sanitize_args=do_sanitize_args,
        ):
            if input_info["layout"][0] == ttnn.ROW_MAJOR_LAYOUT:
                # Last dim of output must be divisible by 2 for row_major
                last_dim = permute_dims[-1]
                last_dim_shape = input_shapes[0][last_dim]

                if last_dim_shape % 2 == 1:
                    continue

            if input_info is not None:
                input_info.update({"permute_dims": permute_dims})
                yield input_info


def gen_fill_rm_args(input_shapes, dtypes, layouts, mem_configs, do_sanitize_args=True, coregrid=[]):
    H = input_shapes[0][-2]
    W = input_shapes[0][-1]

    for input_info in gen_dtype_layout_device(
        input_shapes, dtypes, layouts, mem_configs, do_sanitize_args=do_sanitize_args
    ):
        if input_info is not None:
            input_info["hOnes"] = random.randint(1, H)
            input_info["wOnes"] = random.randint(1, W)

            input_info["val_hi"] = random.uniform(-100, 100)
            input_info["val_lo"] = random.uniform(-100, 100)

            yield input_info


def gen_fill_ones_rm_args(
    input_shapes,
    dtypes,
    layouts,
    mem_configs,
    do_sanitize_args=True,
    coregrid=[],
):
    H = input_shapes[0][-2]
    W = input_shapes[0][-1]

    for input_info in gen_dtype_layout_device(
        input_shapes, dtypes, layouts, mem_configs, do_sanitize_args=do_sanitize_args
    ):
        if input_info is not None:
            input_info["hOnes"] = random.randint(1, H)
            input_info["wOnes"] = random.randint(1, W)
            yield input_info


@lru_cache(maxsize=5000)
def _get_factors(i, s):
    factors = []
    for j in range(s, i + 1, s):
        if i % j == 0:
            factors.append(j)
    return factors


@lru_cache(maxsize=5000)
def _gen_reshape_args_from_volume(volume, step, out_dims=4):
    shapes = []

    if out_dims == 4:
        for w in _get_factors(volume, step):
            v = volume // w
            for h in _get_factors(v, step):
                v2 = v // h
                for c in _get_factors(v2, 1):
                    b = v2 // c
                    shapes.append({"reshape_dims": (b, c, h, w)})
    elif out_dims == 3:
        for h in _get_factors(volume, step):
            v2 = volume // h
            for c in _get_factors(v2, 1):
                b = v2 // c
                shapes.append({"reshape_dims": (b, c, h)})
    elif out_dims == 2:
        for c in _get_factors(volume, 1):
            b = volume // c
            shapes.append({"reshape_dims": (b, c)})

    return shapes


def gen_reshape_args(
    input_shapes,
    dtypes=[[ttnn.bfloat16]],
    layouts=[[ttnn.TILE_LAYOUT]],
    mem_configs=[[ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)]],
    max_out_shapes=2,
    do_sanitize_args=True,
    coregrid=[],
):
    vol = 1

    for x in input_shapes[0]:
        vol *= x

    out_shapes = _gen_reshape_args_from_volume(vol, step=1, out_dims=len(input_shapes[0]))
    random.shuffle(out_shapes)
    n_out_shapes_used = 0

    for reshape_dims in out_shapes:
        if reshape_dims["reshape_dims"][-2] % 32 != 0:
            continue

        if reshape_dims["reshape_dims"][-1] % 32 != 0:
            continue

        for input_info in gen_dtype_layout_device(
            input_shapes,
            dtypes,
            layouts,
            mem_configs,
            do_sanitize_args=do_sanitize_args,
        ):
            if input_info is not None:
                input_info.update(reshape_dims)
                yield input_info

        n_out_shapes_used += 1

        # Reached max out shapes
        if n_out_shapes_used > max_out_shapes:
            break


def gen_split_args(input_shapes, dtypes, layouts, mem_configs, do_sanitize_args=True):
    for input_info in gen_dtype_layout_device(
        input_shapes, dtypes, layouts, mem_configs, do_sanitize_args=do_sanitize_args
    ):
        if input_info is not None:
            maxdim = len(input_shapes[0]) - 1
            dim = random.randint(-maxdim - 1, maxdim)

            max_split = input_shapes[0][dim] // 2
            max_split = max(max_split, 1)
            split_size = random.randint(1, max_split)

            input_info["dim"] = dim
            input_info["split_size"] = split_size
            yield input_info


def gen_tilize_with_val_padding_args(
    input_shapes,
    dtypes=[supported_tt_dtypes],
    layouts=[[ttnn.ROW_MAJOR_LAYOUT]],
    mem_configs=[supported_mem_configs],
    do_sanitize_args=True,
    coregrid=[],
):
    assert len(input_shapes) == 1
    assert len(input_shapes[0]) == 4

    for input_info in gen_dtype_layout_device(
        input_shapes, dtypes, layouts, mem_configs, do_sanitize_args=do_sanitize_args
    ):
        if input_info is not None:
            pad_sizes = (10, 10, 64, 64)
            output_tensor_shape = [
                random.randrange(
                    input_shapes[0][i],
                    input_shapes[0][i] + pad_sizes[i],
                    1,
                )
                for i in range(4)
            ]
            output_tensor_shape[-2] = nearest_32(output_tensor_shape[-2])
            output_tensor_shape[-1] = nearest_32(output_tensor_shape[-1])
            pad_value = random.uniform(-100, 100)
            # Cast to bfloat16 then back to float for exact match
            pad_value = torch.Tensor([pad_value]).to(torch.bfloat16).to(torch.float).item()

            input_info.update(
                {
                    "output_tensor_shape": output_tensor_shape,
                    "pad_value": pad_value,
                }
            )
            yield input_info


def gen_untilize_with_unpadding_args(
    input_shapes,
    dtypes=[supported_tt_dtypes],
    layouts=[supported_tt_layouts],
    mem_configs=[supported_mem_configs],
    do_sanitize_args=True,
    coregrid=[],
):
    assert len(input_shapes) == 1
    assert len(input_shapes[0]) == 4
    assert input_shapes[0][-2] % 32 == 0
    assert input_shapes[0][-1] % 32 == 0

    for input_info in gen_dtype_layout_device(
        input_shapes, dtypes, layouts, mem_configs, do_sanitize_args=do_sanitize_args
    ):
        if input_info is not None:
            output_tensor_end = [random.randrange(0, input_shapes[0][i], 1) for i in range(4)]
            if output_tensor_end[-1] % 2 == 0:
                output_tensor_end[-1] += 1
            input_info.update(
                {
                    "output_tensor_end": output_tensor_end,
                    "use_multicore": True,
                }
            )
            yield input_info


def gen_pad_args(
    input_shapes,
    dtypes=[supported_tt_dtypes],
    layouts=[supported_tt_layouts],
    mem_configs=[supported_mem_configs],
    do_sanitize_args=True,
    coregrid=[],
):
    assert len(input_shapes) == 1
    assert len(input_shapes[0]) == 4

    for input_info in gen_dtype_layout_device(
        input_shapes, dtypes, layouts, mem_configs, do_sanitize_args=do_sanitize_args
    ):
        if input_info is not None:
            if input_info["layout"][0] == ttnn.ROW_MAJOR_LAYOUT:
                pad_sizes = (10, 10, 64, 64)
                output_tensor_shape = [
                    random.randint(input_shapes[0][i], input_shapes[0][i] + pad_sizes[i]) for i in range(4)
                ]
                if output_tensor_shape[-1] % 2 != 0:
                    output_tensor_shape[-1] += 1
                input_tensor_start = [0, 0, 0, 0]
                pad_value = random.uniform(-100, 100)
                # Cast to bfloat16 then back to float for exact match
                pad_value = torch.Tensor([pad_value]).to(torch.bfloat16).to(torch.float).item()

                input_info.update(
                    {
                        "output_tensor_shape": output_tensor_shape,
                        "input_tensor_start": input_tensor_start,
                        "pad_value": pad_value,
                    }
                )
            elif input_info["layout"][0] == ttnn.TILE_LAYOUT:
                pad_sizes = (10, 10, 64, 64)
                output_tensor_shape = [
                    random.randrange(
                        input_shapes[0][i],
                        input_shapes[0][i] + pad_sizes[i],
                        1,
                    )
                    for i in range(4)
                ]
                output_tensor_shape[-2] = nearest_32(output_tensor_shape[-2])
                output_tensor_shape[-1] = nearest_32(output_tensor_shape[-1])
                input_tensor_start = [0, 0, 0, 0]
                pad_value = random.uniform(-100, 100)
                # Cast to bfloat16 then back to float for exact match
                pad_value = torch.Tensor([pad_value]).to(torch.bfloat16).to(torch.float).item()

                input_info.update(
                    {
                        "output_tensor_shape": output_tensor_shape,
                        "input_tensor_start": input_tensor_start,
                        "pad_value": pad_value,
                    }
                )

            yield input_info


def gen_unpad_args(
    input_shapes,
    dtypes=[supported_tt_dtypes],
    layouts=[supported_tt_layouts],
    mem_configs=[supported_mem_configs],
    do_sanitize_args=True,
    coregrid=[],
):
    assert len(input_shapes) == 1
    assert len(input_shapes[0]) == 4

    for input_info in gen_dtype_layout_device(
        input_shapes, dtypes, layouts, mem_configs, do_sanitize_args=do_sanitize_args
    ):
        if input_info is not None:
            if input_info["layout"][0] == ttnn.ROW_MAJOR_LAYOUT:
                output_tensor_start = [0, 0, 0, 0]
                output_tensor_end = [
                    random.randrange(output_tensor_start[i] + 1, input_shapes[0][i], 1) for i in range(4)
                ]
                if output_tensor_end[-1] % 2 != 0:
                    output_tensor_end[-1] += 1
                input_info.update(
                    {
                        "output_tensor_start": output_tensor_start,
                        "output_tensor_end": output_tensor_end,
                    }
                )
            elif input_info["layout"][0] == ttnn.TILE_LAYOUT:
                output_tensor_start = [0, 0, 0, 0]
                output_tensor_end = [
                    random.randrange(output_tensor_start[i] + 1, input_shapes[0][i], 1) for i in range(4)
                ]
                output_tensor_end[-2] = max(nearest_32(output_tensor_end[-2]), 32)
                output_tensor_end[-1] = max(nearest_32(output_tensor_end[-1]), 32)
                input_info.update(
                    {
                        "output_tensor_start": output_tensor_start,
                        "output_tensor_end": output_tensor_end,
                    }
                )
            yield input_info


def gen_scalar_args(
    input_shapes,
    dtypes,
    layouts,
    mem_configs,
    arg_name="scalar",
    low=-100,
    high=100,
    dtype=torch.bfloat16,
    do_sanitize_args=True,
    coregrid=[],
):
    for input_info in gen_dtype_layout_device(
        input_shapes, dtypes, layouts, mem_configs, do_sanitize_args=do_sanitize_args
    ):
        if input_info is not None:
            if dtype.is_floating_point:
                scalar = torch.tensor(1, dtype=dtype).uniform_(low, high).item()
            else:
                scalar = torch.tensor(1, dtype=dtype).random_(low, high).item()
            input_info.update({arg_name: scalar})
            yield input_info


def gen_conv2d_args(
    input_shapes,
    dtypes,
    layouts,
    mem_configs,
    do_sanitize_args=True,
    coregrid=[],
):
    for input_info in gen_conv_scalar_args(
        input_shapes, dtypes, layouts, mem_configs, "conv_params", torch.int, do_sanitize_args=do_sanitize_args
    ):
        yield input_info


def gen_conv_scalar_args(
    input_shapes,
    supported_dtypes,
    supported_layouts,
    mem_configs,
    arg0_name="conv_params",
    dtype=torch.bfloat16,
    do_sanitize_args=True,
    coregrid=[],
):
    for input_info in gen_dtype_layout_device(
        input_shapes, supported_dtypes, supported_layouts, mem_configs, do_sanitize_args=do_sanitize_args
    ):
        lowStride = 1
        highStride = 4
        padH = 0
        padW = 0

        w = input_shapes[0][3]
        h = input_shapes[0][2]

        # assert(lowKernel>0 and highKernel<w and highKernel<w)
        # assert(lowStride>0 and highStride<w and highStride<h)

        kernelH = input_shapes[1][2]
        kernelW = input_shapes[1][3]

        strideH = random.randint(lowStride, highStride)
        strideW = random.randint(lowStride, highStride)
        conv_params = [kernelH, kernelW, strideH, strideW, padH, padW]

        input_info.update({arg0_name: conv_params})
        yield input_info


def gen_scalar_symmetric_args(
    input_shapes,
    dtypes,
    layouts,
    mem_configs,
    arg_name="scalar",
    low=0.01,
    high=100,
    dtype=torch.bfloat16,
    do_sanitize_args=True,
    coregrid=[],
):
    for input_info in gen_scalar_args(
        input_shapes, dtypes, layouts, mem_configs, arg_name, low, high, dtype, do_sanitize_args=do_sanitize_args
    ):
        if input_info is not None:
            sign = (torch.tensor(1, dtype=torch.int).random_(0, 2) * 2 - 1).item()
            input_info[arg_name] *= sign
            yield input_info


def gen_power_args(
    input_shapes,
    dtypes,
    layouts,
    mem_configs,
    low=0,
    high=10,
    dtype=torch.int,
    do_sanitize_args=True,
    coregrid=[],
):
    for input_info in gen_scalar_args(
        input_shapes, dtypes, layouts, mem_configs, "exponent", low, high, dtype, do_sanitize_args=do_sanitize_args
    ):
        yield input_info


def gen_relu_min_args(
    input_shapes,
    dtypes,
    layouts,
    mem_configs,
    low=0,
    high=100,
    dtype=torch.bfloat16,
    do_sanitize_args=True,
    coregrid=[],
):
    for input_info in gen_scalar_args(
        input_shapes, dtypes, layouts, mem_configs, "lower_limit", low, high, dtype, do_sanitize_args=do_sanitize_args
    ):
        yield input_info


def gen_relu_max_args(
    input_shapes,
    dtypes,
    layouts,
    mem_configs,
    low=0,
    high=100,
    dtype=torch.bfloat16,
    do_sanitize_args=True,
    coregrid=[],
):
    for input_info in gen_scalar_args(
        input_shapes, dtypes, layouts, mem_configs, "upper_limit", low, high, dtype, do_sanitize_args=do_sanitize_args
    ):
        yield input_info


def gen_scale_mask_softmax_in_place_args(
    input_shapes,
    dtypes,
    layouts,
    mem_configs,
    low=1,
    high=100,
    dtype=torch.bfloat16,
    do_sanitize_args=True,
    coregrid=[],
):
    for input_info in gen_scalar_args(
        input_shapes, dtypes, layouts, mem_configs, "scale", low, high, dtype, do_sanitize_args=do_sanitize_args
    ):
        yield input_info


def gen_lerp_binary_args(
    input_shapes,
    dtypes,
    layouts,
    mem_configs,
    low=-100,
    high=100,
    dtype=torch.bfloat16,
    do_sanitize_args=True,
    coregrid=[],
):
    for input_info in gen_scalar_args(
        input_shapes, dtypes, layouts, mem_configs, "weight", low, high, dtype, do_sanitize_args=do_sanitize_args
    ):
        yield input_info


def gen_subalpha_args(
    input_shapes,
    supported_dtypes,
    supported_layouts,
    mem_configs,
    low=-100,
    high=100,
    dtype=torch.bfloat16,
    do_sanitize_args=True,
    coregrid=[],
):
    for input_info in gen_scalar_args(
        input_shapes,
        supported_dtypes,
        supported_layouts,
        mem_configs,
        "alpha",
        low,
        high,
        dtype,
        do_sanitize_args=do_sanitize_args,
    ):
        yield input_info


def gen_addalpha_args(
    input_shapes,
    supported_dtypes,
    supported_layouts,
    mem_configs,
    low=-100,
    high=100,
    dtype=torch.bfloat16,
    do_sanitize_args=True,
    coregrid=[],
):
    for input_info in gen_scalar_args(
        input_shapes,
        supported_dtypes,
        supported_layouts,
        mem_configs,
        "alpha",
        low,
        high,
        dtype,
        do_sanitize_args=do_sanitize_args,
    ):
        yield input_info


def gen_logit_args(
    input_shapes,
    supported_dtypes,
    supported_layouts,
    mem_configs,
    low=-10,
    high=0.99,
    dtype=torch.bfloat16,
    do_sanitize_args=True,
    coregrid=[],
):
    for input_info in gen_scalar_args(
        input_shapes,
        supported_dtypes,
        supported_layouts,
        mem_configs,
        "eps",
        low,
        high,
        dtype,
        do_sanitize_args=do_sanitize_args,
        coregrid=[],
    ):
        yield input_info


def gen_shrink_args(
    input_shapes,
    supported_dtypes,
    supported_layouts,
    mem_configs,
    low=0,
    high=100,
    dtype=torch.bfloat16,
    do_sanitize_args=True,
    coregrid=[],
):
    for input_info in gen_scalar_args(
        input_shapes,
        supported_dtypes,
        supported_layouts,
        mem_configs,
        "lambd",
        low,
        high,
        dtype,
        do_sanitize_args=do_sanitize_args,
        coregrid=[],
    ):
        yield input_info


def gen_bias_gelu_unary_args(
    input_shapes,
    supported_dtypes,
    supported_layouts,
    mem_configs,
    low=0,
    high=100,
    dtype=torch.bfloat16,
    do_sanitize_args=True,
    coregrid=[],
):
    for input_info in gen_scalar_args(
        input_shapes,
        supported_dtypes,
        supported_layouts,
        mem_configs,
        "bias",
        low,
        high,
        dtype,
        do_sanitize_args=do_sanitize_args,
    ):
        yield input_info


def gen_logical_immediate_args(
    input_shapes,
    dtypes,
    layouts,
    mem_configs,
    low=0,
    high=100,
    dtype=torch.int32,
    do_sanitize_args=True,
    coregrid=[],
):
    for input_info in gen_scalar_args(
        input_shapes, dtypes, layouts, mem_configs, "immediate", low, high, dtype, do_sanitize_args=do_sanitize_args
    ):
        yield input_info


def gen_shrink_args(
    input_shapes,
    dtypes,
    layouts,
    mem_configs,
    low=0,
    high=100,
    dtype=torch.bfloat16,
    do_sanitize_args=True,
    coregrid=[],
):
    for input_info in gen_scalar_args(
        input_shapes, dtypes, layouts, mem_configs, "_lambda", low, high, dtype, do_sanitize_args=do_sanitize_args
    ):
        yield input_info


def gen_leaky_relu_args(
    input_shapes,
    dtypes,
    layouts,
    mem_configs,
    low=0,
    high=100,
    dtype=torch.bfloat16,
    do_sanitize_args=True,
    coregrid=[],
):
    for input_info in gen_scalar_args(
        input_shapes,
        dtypes,
        layouts,
        mem_configs,
        "negative_slope",
        low,
        high,
        dtype,
        do_sanitize_args=do_sanitize_args,
    ):
        yield input_info


def gen_elu_args(
    input_shapes,
    dtypes,
    layouts,
    mem_configs,
    low=-10,
    high=10,
    dtype=torch.bfloat16,
    do_sanitize_args=True,
    coregrid=[],
):
    for input_info in gen_scalar_args(
        input_shapes, dtypes, layouts, mem_configs, "alpha", low, high, dtype, do_sanitize_args=do_sanitize_args
    ):
        yield input_info


def gen_celu_args(
    input_shapes,
    dtypes,
    layouts,
    mem_configs,
    low=0.01,
    high=10,
    dtype=torch.bfloat16,
    do_sanitize_args=True,
    coregrid=[],
):
    for input_info in gen_scalar_symmetric_args(
        input_shapes, dtypes, layouts, mem_configs, "alpha", low, high, dtype, do_sanitize_args=do_sanitize_args
    ):
        yield input_info


def gen_fast_and_approx_args(input_shapes, dtypes, layouts, mem_configs, do_sanitize_args=True, coregrid=[]):
    input_info = gen_dtype_layout_device(input_shapes, dtypes, layouts, mem_configs, do_sanitize_args=do_sanitize_args)

    test_args_combinations = []

    for input_args in input_info:
        if input_args is not None:
            input_args_1 = input_args.copy()
            input_args_2 = input_args.copy()

            input_args_1.update({"fast_and_approx": True})
            test_args_combinations.append(input_args_1)

            input_args_2.update({"fast_and_approx": False})
            test_args_combinations.append(input_args_2)

    return test_args_combinations


def gen_activation_args(
    input_shapes,
    dtypes,
    layouts,
    mem_configs,
    do_sanitize_args=True,
    coregrid=[],
):
    input_info = gen_dtype_layout_device(input_shapes, dtypes, layouts, mem_configs, do_sanitize_args=do_sanitize_args)

    test_args_combinations = []

    for input_args in input_info:
        if input_args is not None:
            input_args = input_args.copy()
            input_args.update({"activation": None})
            test_args_combinations.append(input_args)

            input_args = input_args.copy()
            input_args.update({"activation": "relu"})
            test_args_combinations.append(input_args)

    return test_args_combinations


def gen_two_scalar_args(
    input_shapes,
    dtypes,
    layouts,
    mem_configs,
    arg0_name="scalar0",
    arg1_name="scalar1",
    low=-100,
    high=100,
    dtype=torch.bfloat16,
    do_sanitize_args=True,
    coregrid=[],
):
    for input_info in gen_dtype_layout_device(
        input_shapes, dtypes, layouts, mem_configs, do_sanitize_args=do_sanitize_args
    ):
        if input_info is not None:
            if dtype.is_floating_point:
                scalar0 = torch.tensor(1, dtype=dtype).uniform_(low, high).item()
                scalar1 = torch.tensor(1, dtype=dtype).uniform_(low, high).item()
            else:
                scalar0 = torch.tensor(1, dtype=dtype).random_(low, high).item()
                scalar1 = torch.tensor(1, dtype=dtype).random_(low, high).item()
            input_info.update({arg0_name: scalar0, arg1_name: scalar1})
            yield input_info


def gen_clip_args(
    input_shapes,
    dtypes,
    layouts,
    mem_configs,
    low=-100,
    high=100,
    dtype=torch.bfloat16,
    do_sanitize_args=True,
    coregrid=[],
):
    for input_info in gen_two_scalar_args(
        input_shapes, dtypes, layouts, mem_configs, "low", "high", low, high, dtype, do_sanitize_args=do_sanitize_args
    ):
        if input_info["low"] > input_info["high"]:
            input_info["low"], input_info["high"] = (
                input_info["high"],
                input_info["low"],
            )
        yield input_info


def gen_threshold_args(
    input_shapes,
    dtypes,
    layouts,
    mem_configs,
    low=-100,
    high=100,
    dtype=torch.bfloat16,
    do_sanitize_args=True,
    coregrid=[],
):
    for input_info in gen_two_scalar_args(
        input_shapes,
        dtypes,
        layouts,
        mem_configs,
        "threshold",
        "value",
        low,
        high,
        dtype,
        do_sanitize_args=do_sanitize_args,
    ):
        yield input_info


def gen_hardtanh_args(
    input_shapes,
    dtypes,
    layouts,
    mem_configs,
    low=-10,
    high=10,
    dtype=torch.bfloat16,
    do_sanitize_args=True,
    coregrid=[],
):
    for input_info in gen_two_scalar_args(
        input_shapes, dtypes, layouts, mem_configs, "low", "high", low, high, dtype, do_sanitize_args=do_sanitize_args
    ):
        if input_info["low"] > input_info["high"]:
            input_info["low"], input_info["high"] = (
                input_info["high"],
                input_info["low"],
            )

        yield input_info


def gen_polyval_args(
    input_shapes,
    dtypes,
    layouts,
    mem_configs,
    max_num_coeffs=10,
    low=-100,
    high=100,
    do_sanitize_args=True,
    coregrid=[],
):
    for input_info in gen_dtype_layout_device(
        input_shapes, dtypes, layouts, mem_configs, do_sanitize_args=do_sanitize_args
    ):
        if input_info is not None:
            num_coeffs = torch.tensor(1, dtype=torch.int).random_(1, max_num_coeffs + 1).item()
            coeffs = torch.Tensor(num_coeffs).uniform_(low, high).to(torch.bfloat16).tolist()
            input_info.update({"coeffs": coeffs})
            yield input_info


def gen_arange_args(input_shapes, dtypes, layouts, mem_configs, low=-100, high=100, do_sanitize_args=True):
    for input_info in gen_two_scalar_args(
        input_shapes,
        dtypes,
        layouts,
        mem_configs,
        "start",
        "end",
        low,
        high,
        torch.int,
        do_sanitize_args=do_sanitize_args,
        coregrid=[],
    ):
        if input_info["start"] > input_info["end"]:
            input_info["start"], input_info["end"] = (
                input_info["end"],
                input_info["start"],
            )
        elif input_info["start"] == input_info["end"]:
            input_info["end"] += 1
        input_info["step"] = (
            torch.tensor(1, dtype=torch.int).random_(1, input_info["end"] - input_info["start"] + 1).item()
        )

        yield input_info


def gen_logical_immediate_args(
    input_shapes,
    dtypes,
    layouts,
    buffer_types,
    low=0,
    high=100,
    dtype=torch.int32,
    do_sanitize_args=True,
    coregrid=[],
):
    for input_info in gen_scalar_args(
        input_shapes, dtypes, layouts, buffer_types, "immediate", low, high, dtype, do_sanitize_args=do_sanitize_args
    ):
        yield input_info


def gen_concat_args(input_shapes, dtypes, layouts, mem_configs, do_sanitize_args=True, coregrid=[]):
    for input_info in gen_dtype_layout_device(
        input_shapes, dtypes, layouts, mem_configs, do_sanitize_args=do_sanitize_args
    ):
        if input_info is not None:
            dim = -1

            for i in range(len(input_shapes[0])):
                if input_shapes[0][i] != input_shapes[1][i]:
                    dim = i

            if dim == -1:
                num_dims = len(input_shapes[0])
                dim = random.randint(0, num_dims - 1)

            input_info.update({"dim": dim})
            yield input_info


def gen_geglu_args(input_shapes, dtypes, layouts, mem_configs, do_sanitize_args=True, coregrid=[]):
    for input_info in gen_dtype_layout_device(
        input_shapes, dtypes, layouts, mem_configs, do_sanitize_args=do_sanitize_args
    ):
        if input_info is not None:
            input_info.update({"dim": -1})
            yield input_info


def gen_tilize_args(input_shapes, dtypes, layouts, mem_configs, do_sanitize_args=True, coregrid=[]):
    input_info = gen_dtype_layout_device(input_shapes, dtypes, layouts, mem_configs, do_sanitize_args=do_sanitize_args)

    new_input_info = []

    for input_args in input_info:
        input_args_1 = input_args.copy()
        input_args_2 = input_args.copy()
        input_args_1.update({"use_multicore": True})
        new_input_info.append(input_args_1)
        input_args_2.update({"use_multicore": False})
        new_input_info.append(input_args_2)

    for info in new_input_info:
        if input_info is not None:
            yield info


def gen_polygamma_args(
    input_shapes,
    supported_dtypes,
    supported_layouts,
    on_device,
    low=-1,
    high=10,
    dtype=torch.bfloat16,
    do_sanitize_args=True,
    coregrid=[],
):
    for input_info in gen_scalar_args(
        input_shapes,
        supported_dtypes,
        supported_layouts,
        on_device,
        "k",
        low,
        high,
        dtype,
        do_sanitize_args=do_sanitize_args,
    ):
        # the n(int) order of the polygamma function is b/w 1 to 10
        k_order = np.random.randint(1, 10)
        input_info.update({"k": k_order})
        yield input_info


def gen_rop_args(
    input_shapes,
    supported_dtypes,
    supported_layouts,
    on_device,
    low=-1,
    high=10,
    dtype=torch.bfloat16,
    do_sanitize_args=True,
    coregrid=[],
):
    for input_info in gen_scalar_args(
        input_shapes,
        supported_dtypes,
        supported_layouts,
        on_device,
        "factor",
        low,
        high,
        dtype,
        do_sanitize_args=do_sanitize_args,
    ):
        # the n(int) order of the polygamma function is b/w 1 to 10
        factor = random.uniform(0.1, 10.0)
        input_info.update({"factor": factor})
        yield input_info


def gen_repeat_interleave_args(
    input_shapes,
    dtypes,
    layouts,
    mem_configs,
    low=-100,
    high=100,
    dtype=torch.bfloat16,
    do_sanitize_args=True,
    coregrid=[],
):
    for input_info in gen_two_scalar_args(
        input_shapes, dtypes, layouts, mem_configs, "repeat", "dim", low, high, dtype, do_sanitize_args=do_sanitize_args
    ):
        repeats = np.random.randint(1, 5)
        dims = np.random.choice([0, 2])
        input_info.update({"repeat": repeats, "dim": dims})
        yield input_info


def gen_glu_args(input_shapes, dtypes, layouts, mem_configs, do_sanitize_args=True):
    for input_info in gen_dtype_layout_device(
        input_shapes, dtypes, layouts, mem_configs, do_sanitize_args=do_sanitize_args
    ):
        if input_info is not None:
            # max_dim = len(input_shapes[0]) - 1
            # dim = random.randint(-max_dim-1, max_dim)
            # For now onlu last dim is supported
            input_info.update({"dim": -1})
            yield input_info


def gen_dim_args(input_shapes, dtypes, layouts, mem_configs, do_sanitize_args=True):
    for input_info in gen_dtype_layout_device(
        input_shapes, dtypes, layouts, mem_configs, do_sanitize_args=do_sanitize_args
    ):
        if input_info is not None:
            max_dim = len(input_shapes[0]) - 1
            dim = random.randint(-max_dim - 1, max_dim)
            input_info.update({"dim": dim})
            yield input_info


def gen_rmsnorm_args(
    input_shapes,
    dtypes=[supported_tt_dtypes],
    layouts=[supported_tt_layouts],
    mem_configs=[supported_mem_configs],
    do_sanitize_args=False,
    coregrid=[],
):
    return gen_dtype_layout_device_layernorm(
        input_shapes,
        dtypes,
        layouts,
        mem_configs,
        runtime_tile_padding_layernorm=True,
        runtime_tile_padding_add_layernorm=False,
    )


def gen_power_fp_args(
    input_shapes,
    dtypes,
    layouts,
    mem_configs,
    low=0,
    high=10,
    dtype=torch.bfloat16,
    do_sanitize_args=True,
    coregrid=[],
):
    for input_info in gen_scalar_args(
        input_shapes, dtypes, layouts, mem_configs, "exponent", low, high, dtype, do_sanitize_args=do_sanitize_args
    ):
        yield input_info


def gen_isclose_args(
    input_shapes,
    dtypes,
    layouts,
    mem_configs,
    rtol_low=1e-7,
    rtol_high=1e-5,
    atol_low=1e-9,
    atol_high=1e-7,
    dtype=torch.bfloat16,
    do_sanitize_args=True,
    coregrid=[],
):
    for input_info in gen_dtype_layout_device(
        input_shapes, dtypes, layouts, mem_configs, do_sanitize_args=do_sanitize_args
    ):
        if input_info is not None:
            if dtype.is_floating_point:
                rtol = torch.tensor(1, dtype=dtype).uniform_(rtol_low, rtol_low).item()
            else:
                rtol = torch.tensor(1, dtype=dtype).random_(rtol_low, rtol_low).item()

            if dtype.is_floating_point:
                atol = torch.tensor(1, dtype=dtype).uniform_(atol_low, atol_low).item()
            else:
                atol = torch.tensor(1, dtype=dtype).random_(atol_low, atol_low).item()

            input_info.update({"rtol": rtol, "atol": atol, "equal_nan": False})
            yield input_info


def gen_rand_exclude_range(size, excluderange=None, low=0, high=100):
    res = torch.Tensor(size=size).uniform_(low, high)
    if excluderange is None:
        return res

    exclude_upper = excluderange[1]
    exclude_lower = excluderange[0]
    assert exclude_upper < high
    assert exclude_lower > low

    list_tensor = torch.flatten(res)

    i = 0
    for el in list_tensor:
        while el >= exclude_lower and el <= exclude_upper:
            list_tensor[i] = random.uniform(low, high)
        i = i + 1
    res = torch.reshape(list_tensor, size)

    return res


def gen_ttnn_repeat_interleave_args(
    input_shapes,
    dtypes,
    layouts,
    mem_configs,
    low=-100,
    high=100,
    dtype=torch.bfloat16,
    do_sanitize_args=True,
    coregrid=[],
):
    for input_info in gen_two_scalar_args(
        input_shapes, dtypes, layouts, mem_configs, "repeat", "dim", low, high, dtype, do_sanitize_args=do_sanitize_args
    ):
        shapes_size = len(input_shapes)
        repeats = np.random.randint(1, 5)
        dims = np.random.choice([0, shapes_size - 1])
        input_info.update({"repeat": repeats, "dim": dims})
        yield input_info


def gen_upsample_args(
    input_shapes,
    dtypes,
    layouts,
    buffer_types,
    low=2,
    high=10,
    dtype=torch.int32,
    do_sanitize_args=True,
    coregrid=[],
):
    for input_info in gen_scalar_args(
        input_shapes, dtypes, layouts, buffer_types, "scale_factor", low, high, dtype, do_sanitize_args=do_sanitize_args
    ):
        yield input_info


def gen_softplus_args(
    input_shapes,
    dtypes,
    layouts,
    mem_configs,
    low=-100,
    high=100,
    dtype=torch.bfloat16,
    do_sanitize_args=True,
    coregrid=[],
):
    for input_info in gen_two_scalar_args(
        input_shapes,
        dtypes,
        layouts,
        mem_configs,
        "beta",
        "threshold",
        low,
        high,
        dtype,
        do_sanitize_args=do_sanitize_args,
    ):
        if input_info["beta"] == 0.0 and input_info["threshold"] > 0.0:
            continue
        yield input_info


def gen_min_max_dim_args(
    input_shapes, dtypes, layouts, mem_configs, low=0, high=4, dtype=torch.int, do_sanitize_args=True, coregrid=[]
):
    for input_info in gen_scalar_args(
        input_shapes, dtypes, layouts, mem_configs, "dim", low, high, dtype, do_sanitize_args=do_sanitize_args
    ):
        rank = len(input_shapes[0])
        choices = [(rank - 1,), (rank - 2,)]
        idx = np.random.choice(len(choices), 1)
        dims = choices[idx.item()]

        input_info.update({"dim": dims})
        yield input_info


def gen_dim_args(
    input_shapes,
    dtypes,
    layouts,
    mem_configs,
    low=-100,
    high=100,
    dtype=torch.bfloat16,
    do_sanitize_args=True,
    coregrid=[],
):
    for input_info in gen_scalar_args(
        input_shapes, dtypes, layouts, mem_configs, "dim", low, high, dtype, do_sanitize_args=do_sanitize_args
    ):
        rank = len(input_shapes[0])

        # select one of the possible combnations
        if rank == 4 or rank == 3:
            choices = [(rank - 1,), (rank - 2,), (rank - 1, rank - 2)]
            idx = np.random.choice(len(choices), 1)
            dims = choices[idx.item()]
        else:
            choices = [(rank - 1,), (rank - 2,)]
            idx = np.random.choice(len(choices), 1)
            dims = choices[idx.item()]

        input_info.update({"dim": dims})
        yield input_info


def gen_repeat_args(
    input_shapes, dtypes, layouts, mem_configs, low=-100, high=100, dtype=torch.bfloat16, do_sanitize_args=False
):
    for input_info in gen_scalar_args(
        input_shapes,
        dtypes,
        layouts,
        mem_configs,
        "shape",
        low,
        high,
        dtype,
    ):
        shapes_size = len(input_shapes[0])
        shapes = []

        for i in range(0, shapes_size):
            rand_shape = np.random.randint(1, 3)
            shapes.append(rand_shape)

        input_info.update({"shape": shapes})
        yield input_info


def gen_sharded_args(
    input_shapes,
    dtypes,
    layouts,
    mem_configs,
    low=0,
    high=10,
    dtype=torch.int,
    do_sanitize_args=False,
):
    for input_info in gen_scalar_args(
        input_shapes, dtypes, layouts, mem_configs, "num_slices", 2, 10, dtype, do_sanitize_args=do_sanitize_args
    ):
        yield input_info


def gen_three_scalar_args(
    input_shapes,
    dtypes,
    layouts,
    mem_configs,
    arg0_name="scalar0",
    arg1_name="scalar1",
    arg2_name="scalar2",
    low0=-100,
    high0=100,
    low1=-100,
    high1=100,
    low2=-100,
    high2=100,
    dtype=torch.bfloat16,
    do_sanitize_args=True,
):
    for input_info in gen_dtype_layout_device(
        input_shapes, dtypes, layouts, mem_configs, do_sanitize_args=do_sanitize_args
    ):
        if input_info is not None:
            if dtype.is_floating_point:
                scalar0 = torch.tensor(1, dtype=dtype).uniform_(low0, high0).item()
                scalar1 = torch.tensor(1, dtype=dtype).uniform_(low1, high1).item()
                scalar2 = torch.tensor(1, dtype=dtype).uniform_(low2, high2).item()
            else:
                scalar0 = torch.tensor(1, dtype=dtype).random_(low0, high0).item()
                scalar1 = torch.tensor(1, dtype=dtype).random_(low1, high1).item()
                scalar2 = torch.tensor(1, dtype=dtype).random_(low2, high2).item()
            input_info.update({arg0_name: scalar0, arg1_name: scalar1, arg2_name: scalar2})
            yield input_info


def gen_sharded_args_coregrid(
    input_shapes,
    dtypes,
    layouts,
    mem_configs,
    low=-100,
    high=100,
    dtype=torch.int,
    do_sanitize_args=True,
):
    for input_info in gen_three_scalar_args(
        input_shapes,
        dtypes,
        layouts,
        mem_configs,
        "num_slices",
        "x_core",
        "y_core",
        2,
        10,
        1,
        6,
        1,
        6,
        dtype,
        do_sanitize_args=do_sanitize_args,
    ):
        yield input_info


def gen_fmod_args(
    input_shapes,
    supported_dtypes,
    supported_layouts,
    on_device,
    low=-1,
    high=10,
    dtype=torch.bfloat16,
    do_sanitize_args=True,
    coregrid=[],
):
    for input_info in gen_scalar_args(
        input_shapes,
        supported_dtypes,
        supported_layouts,
        on_device,
        "value",
        low,
        high,
        dtype,
        do_sanitize_args=do_sanitize_args,
        coregrid=coregrid,
    ):
        input_info.update({"value": random.randint(-100, 100) + 0.5})

        yield input_info


def gen_dtype_layout_device_coregrid(
    input_shapes,
    dtypes=[supported_tt_dtypes],
    layouts=[supported_tt_layouts],
    mem_configs=[supported_mem_configs],  # mem_configs[-1] is output_mem_config
    xcoregrid=None,
    ycoregrid=None,
    do_sanitize_args=True,
):
    # last buffer_types option is for output buffer
    dtype_mem_config_layouts = []

    for i in range(len(input_shapes)):
        dtype_mem_config_layout = []

        for dtype, layout, input_mem_config in product(
            dtypes[i],
            layouts[i],
            mem_configs[i],
        ):
            dtype_mem_config_layout.append({"dtype": dtype, "layout": layout, "input_mem_config": input_mem_config})

        dtype_mem_config_layouts.append(dtype_mem_config_layout)

    result = []

    for out_mem_config in mem_configs[-1]:
        for dtype_mem_config_layout_combination in product(*dtype_mem_config_layouts):
            if do_sanitize_args:
                out = sanitize_args(input_shapes, dtype_mem_config_layout_combination)
            else:
                out = 1

            if out is not None:
                dtype = []
                layout = []
                input_mem_config = []

                for x in dtype_mem_config_layout_combination:
                    dtype.append(x["dtype"])
                    layout.append(x["layout"])
                    input_mem_config.append(x["input_mem_config"])

                result.append(
                    {
                        "dtype": dtype,
                        "layout": layout,
                        "input_mem_config": input_mem_config,
                        "output_mem_config": out_mem_config,
                        "xcoregrid": xcoregrid,
                        "ycoregrid": ycoregrid,
                    }
                )

    return result


def gen_dtype_layout_device_matmul(
    input_shapes,
    dtypes=[supported_tt_dtypes],
    layouts=[supported_tt_layouts],
    mem_configs=[supported_mem_configs],  # mem_configs[-1] is output_mem_config
    xcoregrid=-1,
    ycoregrid=-1,
    do_sanitize_args=True,
):
    # last buffer_types option is for output buffer
    dtype_mem_config_layouts = []

    for i in range(len(input_shapes)):
        dtype_mem_config_layout = []

        for dtype, layout, input_mem_config in product(
            dtypes[i],
            layouts[i],
            mem_configs[i],
        ):
            dtype_mem_config_layout.append({"dtype": dtype, "layout": layout, "input_mem_config": input_mem_config})

        dtype_mem_config_layouts.append(dtype_mem_config_layout)

    result = []

    for out_mem_config in mem_configs[-1]:
        for dtype_mem_config_layout_combination in product(*dtype_mem_config_layouts):
            if do_sanitize_args:
                out = sanitize_args(input_shapes, dtype_mem_config_layout_combination)
            else:
                out = 1

            if out is not None:
                dtype = []
                layout = []
                input_mem_config = []

                for x in dtype_mem_config_layout_combination:
                    dtype.append(x["dtype"])
                    layout.append(x["layout"])
                    input_mem_config.append(x["input_mem_config"])

                result.append(
                    {
                        "dtype": dtype,
                        "layout": layout,
                        "input_mem_config": input_mem_config,
                        "output_mem_config": out_mem_config,
                        "xcoregrid": xcoregrid,
                        "ycoregrid": ycoregrid,
                    }
                )

    return result


def gen_matmul_coregrid_args(
    input_shapes,
    dtypes=[supported_tt_dtypes],
    layouts=[supported_tt_layouts],
    mem_configs=[supported_mem_configs],
    do_sanitize_args=False,
    coregrid=[],
):
    xmin = coregrid[0]
    xmax = coregrid[1]
    ymin = coregrid[2]
    ymax = coregrid[3]

    assert xmin < xmax
    assert ymin < ymax

    xcoregrid = random.randint(xmin, xmax)
    ycoregrid = random.randint(ymin, ymax)

    return gen_dtype_layout_device_matmul(
        input_shapes,
        dtypes,
        layouts,
        mem_configs,
        xcoregrid,
        ycoregrid,
        do_sanitize_args,
    )


def gen_bitwise_args(
    input_shapes,
    supported_dtypes,
    supported_layouts,
    on_device,
    low=-1,
    high=10,
    dtype=torch.int32,
    do_sanitize_args=True,
    coregrid=[],
):
    for input_info in gen_scalar_args(
        input_shapes,
        supported_dtypes,
        supported_layouts,
        on_device,
        "value",
        low,
        high,
        dtype,
        do_sanitize_args=do_sanitize_args,
        coregrid=coregrid,
    ):
        input_info.update({"value": random.randint(low, high) for _ in range(5)})

        yield input_info


def gen_floor_args(
    input_shapes,
    supported_dtypes,
    supported_layouts,
    on_device,
    low=-1,
    high=10,
    dtype=torch.bfloat16,
    do_sanitize_args=True,
    coregrid=[],
):
    for input_info in gen_scalar_args(
        input_shapes,
        supported_dtypes,
        supported_layouts,
        on_device,
        "value",
        low,
        high,
        dtype,
        do_sanitize_args=do_sanitize_args,
        coregrid=coregrid,
    ):
        input_info.update({"value": random.uniform(low, high) + 0.5})

        yield input_info


def gen_div_no_nan_args(
    input_shapes,
    supported_dtypes,
    supported_layouts,
    on_device,
    low=-1,
    high=10,
    dtype=torch.bfloat16,
    do_sanitize_args=True,
    coregrid=[],
):
    for input_info in gen_scalar_args(
        input_shapes,
        supported_dtypes,
        supported_layouts,
        on_device,
        "value",
        low,
        high,
        dtype,
        do_sanitize_args=do_sanitize_args,
        coregrid=coregrid,
    ):
        input_info.update({"value": random.uniform(low, high)})

        yield input_info


def gen_topk_args(input_shapes, dtypes, layouts, mem_configs, do_sanitize_args=True, coregrid=[]):
    for input_info in gen_dtype_layout_device(
        input_shapes, dtypes, layouts, mem_configs, do_sanitize_args=do_sanitize_args
    ):
        if input_info is not None:
            # max_dim = len(input_shapes[0]) - 1
            # dim = random.randint(0, max_dim)
            # max_k = input_shapes[0][dim]
            # k = random.randint(1, max_k-1)
            # largest = random.choice([True, False])

            # input_info.update({"dim": dim})
            # input_info.update({"k": k})
            # input_info.update({"largest": largest})

            input_info.update({"dim": -1})
            input_info.update({"k": 32})
            input_info.update({"largest": True})
            yield input_info


def gen_argmax_args(input_shapes, dtypes, layouts, mem_configs, do_sanitize_args=True, coregrid=[]):
    for input_info in gen_dtype_layout_device(
        input_shapes, dtypes, layouts, mem_configs, do_sanitize_args=do_sanitize_args
    ):
        if input_info is not None:
            max_dim = len(input_shapes[0]) - 1
            dim = random.choice([max_dim, None])
            input_info.update({"dim": dim})
            yield input_info
