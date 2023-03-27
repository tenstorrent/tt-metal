import torch

# torch.testing.get_all_dtypes()
supported_dtypes = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


# Wrapper around gen functions to include casting
def gen_func_with_cast(gen_func, dtype):
    return lambda size: gen_func(size).to(dtype)


def gen_zeros(size):
    return torch.zeros(size)


def gen_ones(size):
    return torch.ones(size)


def gen_constant(size, constant=1.0):
    return torch.full(size, constant)


def gen_rand(size, low=0, high=100):
    return torch.Tensor(size=size).uniform_(low, high)


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
