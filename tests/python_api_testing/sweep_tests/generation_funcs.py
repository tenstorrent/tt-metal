import torch
import random
from itertools import permutations, product
from functools import lru_cache
import tt_lib as ttl
from tt_lib.utils import _nearest_32 as nearest_32, tilize

# torch.testing.get_all_dtypes()
supported_dtypes = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

supported_tt_dtypes = {ttl.tensor.DataType.BFLOAT16}

supported_tt_layouts = [
    ttl.tensor.Layout.ROW_MAJOR,
    ttl.tensor.Layout.TILE,
]

supported_mem_configs = [
    ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM),
    ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1),
]

on_device_options = [
    True,
    False,
]


# Wrapper around gen functions to include casting
def gen_func_with_cast(gen_func, dtype, tilize_input=False):
    return (
        lambda size: tilize(gen_func(size).to(dtype))
        if tilize_input
        else gen_func(size).to(dtype)
    )


def gen_zeros(size):
    return torch.zeros(size)


def gen_ones(size):
    return torch.ones(size)


def gen_constant(size, constant=1.0):
    return torch.full(size, constant)


def gen_rand(size, low=0, high=100):
    return torch.Tensor(size=size).uniform_(low, high)


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
        output[i : i + num_subel] = torch.Tensor(size=(num_subel,)).uniform_(
            torch.min(subrange), torch.max(subrange)
        )

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


def gen_default_args(input_shapes):
    return [{}]


def gen_tensor_pad_args(input_shapes):
    assert len(input_shapes) == 1
    assert len(input_shapes[0]) == 4
    test_args = {}

    pad_sizes = (64, 64, 64, 64)
    output_tensor_shape = [
        random.randint(input_shapes[0][i], input_shapes[0][i] + pad_sizes[i])
        for i in range(4)
    ]
    input_tensor_start = [
        random.randint(0, output_tensor_shape[i] - input_shapes[0][i]) for i in range(4)
    ]
    pad_value = random.uniform(-100, 100)
    # Cast to bfloat16 then back to float for exact match
    pad_value = torch.Tensor([pad_value]).to(torch.bfloat16).to(torch.float).item()

    test_args.update(
        {
            "output_tensor_shape": output_tensor_shape,
            "input_tensor_start": input_tensor_start,
            "pad_value": pad_value,
        }
    )
    return [test_args]


def gen_tensor_unpad_args(input_shapes):
    assert len(input_shapes) == 1
    assert len(input_shapes[0]) == 4
    test_args = {}
    output_tensor_start = [random.randint(0, input_shapes[0][i] - 1) for i in range(4)]
    output_tensor_end = [
        random.randint(output_tensor_start[i], input_shapes[0][i] - 1) for i in range(4)
    ]

    test_args.update(
        {
            "output_tensor_start": output_tensor_start,
            "output_tensor_end": output_tensor_end,
        }
    )
    return [test_args]


def gen_pad_to_tile_args(input_shapes):
    assert len(input_shapes) == 1
    assert len(input_shapes[0]) == 4

    pad_value = random.uniform(-100, 100)
    # Cast to bfloat16 then back to float for exact match
    pad_value = torch.Tensor([pad_value]).to(torch.bfloat16).to(torch.float).item()

    test_args = {
        "pad_value": pad_value,
    }

    return [test_args]


def gen_unpad_from_tile_args(input_shapes):
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
    }
    return [test_args]


def gen_default_dtype_layout_device(input_shapes):
    if input_shapes[0][-2] % 32 == 0 and input_shapes[0][-1] % 32 == 0:
        return [
            {
                "dtype": ttl.tensor.DataType.BFLOAT16,
                "layout": ttl.tensor.Layout.TILE,
                "on_device": True,
            }
        ]
    else:
        return [
            {
                "dtype": ttl.tensor.DataType.BFLOAT16,
                "layout": ttl.tensor.Layout.ROW_MAJOR,
                "on_device": True,
            }
        ]


def gen_default_dtype_layout_rm_device(input_shapes):
    return [
        {
            "dtype": ttl.tensor.DataType.BFLOAT16,
            "layout": ttl.tensor.Layout.ROW_MAJOR,
            "on_device": True,
        }
    ]


def sanitize_args(input_shapes, dtype_device_layout):
    for shape in input_shapes:
        if (
            (
                dtype_device_layout["layout"] == ttl.tensor.Layout.TILE
                and (shape[2] % 32 != 0 or shape[3] % 32 != 0)
            )  # Shape cannot be tilized
            or (
                dtype_device_layout["layout"] == ttl.tensor.Layout.ROW_MAJOR
                and dtype_device_layout["on_device"]
                and shape[3] % 2 != 0
            )  # Shape cannot be placed as row major on device
            or (
                dtype_device_layout["dtype"] == ttl.tensor.DataType.BFLOAT8_B
                and dtype_device_layout["layout"] != ttl.tensor.Layout.TILE
            )  # BFLOAT8_B must be tile layout
        ):
            return None
    return dtype_device_layout


def gen_dtype_layout_device(
    input_shapes,
    supported_dtypes=supported_tt_dtypes,
    supported_layouts=supported_tt_layouts,
    on_device=on_device_options,
):
    for dtype, layout, on_dev in product(
        supported_dtypes, supported_layouts, on_device
    ):
        out = sanitize_args(
            input_shapes, {"dtype": dtype, "layout": layout, "on_device": on_dev}
        )
        if out is not None:
            yield out


def gen_permute_args(input_shapes):
    for permute_dims in permutations([0, 1, 2, 3]):
        permuted_shape = [input_shapes[0][i] for i in permute_dims]
        for input_info in gen_dtype_layout_device((input_shapes[0], permuted_shape)):
            if input_info is not None:
                input_info.update({"permute_dims": permute_dims})
                yield input_info


@lru_cache(maxsize=5000)
def _get_factors(i, s):
    factors = []
    for j in range(s, i + 1, s):
        if i % j == 0:
            factors.append(j)
    return factors


@lru_cache(maxsize=5000)
def _gen_reshape_args_from_volume(volume, step):
    shapes = []
    for w in _get_factors(volume, step):
        v = volume // w
        for h in _get_factors(v, step):
            v2 = v // h
            for c in _get_factors(v2, 1):
                b = v2 // c
                shapes.append({"reshape_dims": [b, c, h, w]})
    return shapes


def gen_reshape_args(input_shapes):
    vol = (
        input_shapes[0][0]
        * input_shapes[0][1]
        * input_shapes[0][2]
        * input_shapes[0][3]
    )
    step = 1
    for reshape_dims in _gen_reshape_args_from_volume(vol, step):
        for input_info in gen_dtype_layout_device(
            (input_shapes[0], reshape_dims["reshape_dims"]),
            supported_layouts=[ttl.tensor.Layout.TILE],
        ):
            if input_info is not None:
                input_info.update(reshape_dims)
                yield input_info


def gen_tilize_with_val_padding_args(input_shapes):
    assert len(input_shapes) == 1
    assert len(input_shapes[0]) == 4
    for input_info in gen_dtype_layout_device(
        input_shapes, supported_layouts=[ttl.tensor.Layout.ROW_MAJOR], on_device=[True]
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
            input_tensor_start = [0, 0, 0, 0]
            pad_value = random.uniform(-100, 100)
            # Cast to bfloat16 then back to float for exact match
            pad_value = (
                torch.Tensor([pad_value]).to(torch.bfloat16).to(torch.float).item()
            )

            input_info.update(
                {
                    "output_tensor_shape": output_tensor_shape,
                    "input_tensor_start": input_tensor_start,
                    "pad_value": pad_value,
                }
            )
            yield input_info


def gen_untilize_with_unpadding_args(input_shapes):
    assert len(input_shapes) == 1
    assert len(input_shapes[0]) == 4
    assert input_shapes[0][-2] % 32 == 0
    assert input_shapes[0][-1] % 32 == 0
    for input_info in gen_dtype_layout_device(
        input_shapes, supported_layouts=[ttl.tensor.Layout.TILE], on_device=[True]
    ):
        if input_info is not None:
            output_tensor_start = [0, 0, 0, 0]
            output_tensor_end = [
                random.randrange(output_tensor_start[i], input_shapes[0][i], 1)
                for i in range(4)
            ]
            if output_tensor_end[-1] % 2 == 0:
                output_tensor_end[-1] += 1
            input_info.update(
                {
                    "output_tensor_start": output_tensor_start,
                    "output_tensor_end": output_tensor_end,
                }
            )
            yield input_info


def gen_pad_args(input_shapes):
    assert len(input_shapes) == 1
    assert len(input_shapes[0]) == 4
    for input_info in gen_dtype_layout_device(
        input_shapes,
        supported_layouts=[ttl.tensor.Layout.ROW_MAJOR, ttl.tensor.Layout.TILE],
        on_device=[True],
    ):
        if input_info is not None:
            if input_info["layout"] == ttl.tensor.Layout.ROW_MAJOR:
                pad_sizes = (10, 10, 64, 64)
                output_tensor_shape = [
                    random.randint(
                        input_shapes[0][i], input_shapes[0][i] + pad_sizes[i]
                    )
                    for i in range(4)
                ]
                if output_tensor_shape[-1] % 2 != 0:
                    output_tensor_shape[-1] += 1
                input_tensor_start = [0, 0, 0, 0]
                pad_value = random.uniform(-100, 100)
                # Cast to bfloat16 then back to float for exact match
                pad_value = (
                    torch.Tensor([pad_value]).to(torch.bfloat16).to(torch.float).item()
                )

                input_info.update(
                    {
                        "output_tensor_shape": output_tensor_shape,
                        "input_tensor_start": input_tensor_start,
                        "pad_value": pad_value,
                    }
                )
            elif input_info["layout"] == ttl.tensor.Layout.TILE:
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
                pad_value = (
                    torch.Tensor([pad_value]).to(torch.bfloat16).to(torch.float).item()
                )

                input_info.update(
                    {
                        "output_tensor_shape": output_tensor_shape,
                        "input_tensor_start": input_tensor_start,
                        "pad_value": pad_value,
                    }
                )
            yield input_info


def gen_unpad_args(input_shapes):
    assert len(input_shapes) == 1
    assert len(input_shapes[0]) == 4

    for input_info in gen_dtype_layout_device(
        input_shapes,
        supported_layouts=[ttl.tensor.Layout.ROW_MAJOR, ttl.tensor.Layout.TILE],
        on_device=[True],
    ):
        if input_info is not None:
            if input_info["layout"] == ttl.tensor.Layout.ROW_MAJOR:
                output_tensor_start = [0, 0, 0, 0]
                output_tensor_end = [
                    random.randrange(output_tensor_start[i], input_shapes[0][i], 1)
                    for i in range(4)
                ]
                if output_tensor_end[-1] % 2 == 0:
                    output_tensor_end[-1] += 1
                input_info.update(
                    {
                        "output_tensor_start": output_tensor_start,
                        "output_tensor_end": output_tensor_end,
                    }
                )
            elif input_info["layout"] == ttl.tensor.Layout.TILE:
                output_tensor_start = [0, 0, 0, 0]
                output_tensor_end = [
                    random.randrange(output_tensor_start[i], input_shapes[0][i], 1)
                    for i in range(4)
                ]
                output_tensor_end[-2] = max(nearest_32(output_tensor_end[-2]), 32) - 1
                output_tensor_end[-1] = max(nearest_32(output_tensor_end[-1]), 32) - 1
                input_info.update(
                    {
                        "output_tensor_start": output_tensor_start,
                        "output_tensor_end": output_tensor_end,
                    }
                )
            yield input_info


def gen_scalar_args(
    input_shapes, arg_name="scalar", low=-100, high=100, dtype=torch.bfloat16
):
    for input_info in gen_dtype_layout_device(input_shapes):
        if input_info is not None:
            if dtype.is_floating_point:
                scalar = torch.tensor(1, dtype=dtype).uniform_(low, high).item()
            else:
                scalar = torch.tensor(1, dtype=dtype).random_(low, high).item()
            input_info.update({arg_name: scalar})
            yield input_info


def gen_scalar_symmetric_args(
    input_shapes, arg_name="scalar", low=0.01, high=100, dtype=torch.bfloat16
):
    for input_info in gen_scalar_args(input_shapes, arg_name, low, high, dtype):
        if input_info is not None:
            sign = (torch.tensor(1, dtype=torch.int).random_(0, 2) * 2 - 1).item()
            input_info[arg_name] *= sign
            yield input_info


def gen_power_args(input_shapes, low=0, high=10, dtype=torch.int):
    for input_info in gen_scalar_args(input_shapes, "exponent", low, high, dtype):
        yield input_info


def gen_relu_min_args(input_shapes, low=-100, high=100, dtype=torch.bfloat16):
    for input_info in gen_scalar_args(input_shapes, "lower_limit", low, high, dtype):
        yield input_info


def gen_relu_max_args(input_shapes, low=-100, high=100, dtype=torch.bfloat16):
    for input_info in gen_scalar_args(input_shapes, "upper_limit", low, high, dtype):
        yield input_info


def gen_heaviside_args(input_shapes, low=-100, high=100, dtype=torch.bfloat16):
    for input_info in gen_scalar_args(input_shapes, "value", low, high, dtype):
        yield input_info


def gen_subalpha_args(input_shapes, low=-100, high=100, dtype=torch.bfloat16):
    for input_info in gen_scalar_args(input_shapes, "alpha", low, high, dtype):
        yield input_info


def gen_shrink_args(input_shapes, low=0, high=100, dtype=torch.bfloat16):
    for input_info in gen_scalar_args(input_shapes, "lambd", low, high, dtype):
        yield input_info


def gen_leaky_relu_args(input_shapes, low=0, high=100, dtype=torch.bfloat16):
    for input_info in gen_scalar_args(input_shapes, "negative_slope", low, high, dtype):
        yield input_info


def gen_elu_args(input_shapes, low=-10, high=10, dtype=torch.bfloat16):
    for input_info in gen_scalar_args(input_shapes, "alpha", low, high, dtype):
        yield input_info


def gen_two_scalar_args(
    input_shapes,
    arg0_name="scalar0",
    arg1_name="scalar1",
    low=-100,
    high=100,
    dtype=torch.bfloat16,
):
    for input_info in gen_dtype_layout_device(input_shapes):
        if input_info is not None:
            if dtype.is_floating_point:
                scalar0 = torch.tensor(1, dtype=dtype).uniform_(low, high).item()
                scalar1 = torch.tensor(1, dtype=dtype).uniform_(low, high).item()
            else:
                scalar0 = torch.tensor(1, dtype=dtype).random_(low, high).item()
                scalar1 = torch.tensor(1, dtype=dtype).random_(low, high).item()
            input_info.update({arg0_name: scalar0, arg1_name: scalar1})
            yield input_info


def gen_clip_args(input_shapes, low=-100, high=100, dtype=torch.bfloat16):
    for input_info in gen_two_scalar_args(
        input_shapes, "low", "high", low, high, dtype
    ):
        if input_info["low"] > input_info["high"]:
            input_info["low"], input_info["high"] = (
                input_info["high"],
                input_info["low"],
            )
        yield input_info


def gen_threshold_args(input_shapes, low=-100, high=100, dtype=torch.bfloat16):
    for input_info in gen_two_scalar_args(
        input_shapes, "threshold", "value", low, high, dtype
    ):
        yield input_info


def gen_polyval_args(input_shapes, max_num_coeffs=10, low=-100, high=100):
    for input_info in gen_dtype_layout_device(input_shapes):
        if input_info is not None:
            num_coeffs = (
                torch.tensor(1, dtype=torch.int).random_(1, max_num_coeffs + 1).item()
            )
            coeffs = torch.Tensor(num_coeffs).uniform_(low, high).tolist()
            input_info.update({"coeffs": coeffs})
            yield input_info


def gen_arange_args(input_shapes, low=-100, high=100):
    for input_info in gen_two_scalar_args(
        input_shapes, "start", "end", low, high, torch.int
    ):
        if input_info["start"] > input_info["end"]:
            input_info["start"], input_info["end"] = (
                input_info["end"],
                input_info["start"],
            )
        elif input_info["start"] == input_info["end"]:
            input_info["end"] += 1
        input_info["step"] = (
            torch.tensor(1, dtype=torch.int)
            .random_(1, input_info["end"] - input_info["start"] + 1)
            .item()
        )

        yield input_info
