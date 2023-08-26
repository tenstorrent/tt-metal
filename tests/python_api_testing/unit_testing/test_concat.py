import math
from pathlib import Path
import sys

import torch

import tt_lib as ttl
from tt_models.utility_functions import print_diff_argmax
import pytest

from tests.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


@pytest.mark.parametrize(
    "memcfg",
    (
        ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.L1),
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize(
    "dtype", ((ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B))
)
@pytest.mark.parametrize("nChannels", ((2, 3, 4)))
def test_tile_simple_concat(memcfg, dtype, nChannels, device):
    torch.manual_seed(0)
    N = nChannels
    C = nChannels
    H = 32
    W = 32
    x = torch.arange(0, N * C * H * W).reshape((N, C, H, W))

    N = nChannels
    C = nChannels
    H = 32
    W = 32
    y = 1 + torch.arange(0, N * C * H * W).reshape((N, C, H, W))

    xtt = (
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            dtype,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device, memcfg),
        ttl.tensor.Tensor(
            y.reshape(-1).tolist(), y.shape, dtype, ttl.tensor.Layout.ROW_MAJOR
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device, memcfg),
    )

    dim = 3
    output_shape = list(x.shape)
    output_shape[3] = y.shape[3] + x.shape[3]
    tt_cpu = torch.concat([x, y], dim)
    assert tt_cpu.shape == torch.Size(output_shape)

    tt = ttl.tensor.concat([xtt[0], xtt[1]], dim)
    assert tt.shape() == output_shape
    xtt_data = tt.cpu().to(ttl.tensor.Layout.ROW_MAJOR)
    tt_dev = xtt_data.to_torch().to(torch.bfloat16)
    # debug_show(output_shape[2],tt_dev,tt_cpu)
    # print_diff_argmax(tt_dev, tt_cpu)

    assert comp_pcc(tt_cpu, tt_dev)[0]


# @pytest.mark.skip(reason="For Stable Diffusion Sizes only")
@pytest.mark.parametrize(
    "shape_a_b_dim",
    (
        ((1, 1, 32, 32), (1, 1, 32, 32), 3),
        ((1, 1, 32, 64), (1, 1, 32, 128), 3),
        ((1, 1, 32, 128), (1, 1, 32, 64), 3),
        ((1, 1, 64, 128), (1, 1, 64, 256), 3),
        ((1, 32, 32, 32), (1, 32, 32, 32), 2),
        ((1, 1, 32, 32), (1, 1, 32, 32), 3),
        ((2, 4, 32, 1280), (2, 4, 32, 1280), 3),
        # SD Shapes
        ((2, 1280, 4, 4), (2, 1280, 4, 4), 1),
        ((2, 640, 32, 32), (2, 320, 32, 32), 1),
        ((2, 1280, 8, 8), (2, 1280, 8, 8), 1),
        ((2, 640, 16, 16), (2, 640, 16, 16), 1),
        ((2, 320, 32, 32), (2, 320, 32, 32), 1),
    ),
)
def test_tile_simple_dim3_concat(shape_a_b_dim, device):
    shape_a, shape_b, dim = shape_a_b_dim
    torch.manual_seed(0)

    N = shape_a[0]  # 1x1x64x32
    C = shape_a[1]
    H = shape_a[2]
    Wx = shape_a[3]
    Wy = shape_b[3]

    x = torch.arange(0, N * C * H * Wx).reshape((N, C, H, Wx)).to(torch.bfloat16)

    N = shape_b[0]  # 1x1x64x32
    C = shape_b[1]
    H = shape_b[2]
    Wy = shape_b[3]
    y = torch.arange(0, N * C * H * Wy).reshape((N, C, H, Wy)).to(torch.bfloat16)

    xtt = (
        ttl.tensor.Tensor(
            x.reshape(-1).tolist(),
            x.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        ).to(device),
        ttl.tensor.Tensor(
            y.reshape(-1).tolist(),
            y.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        ).to(device),
    )

    output_shape = list(x.shape)
    output_shape[dim] = y.shape[dim] + x.shape[dim]
    tt_cpu = torch.concat([x, y], dim)
    assert tt_cpu.shape == torch.Size(output_shape)

    tt = ttl.tensor.concat([xtt[0], xtt[1]], dim)
    assert tt.shape() == output_shape
    tt_dev = tt.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch().to(torch.bfloat16)

    assert torch.equal(tt_cpu, tt_dev)

    assert comp_pcc(tt_cpu, tt_dev)[0]


def debug_show(H_cols, tt_dev, tt_cpu):
    for row in range(H_cols):
        print(f"\t [row {row}]\n\t------- cpu -----")
        print(tt_cpu[:, :, row, :])
        print(f"\t ------- device -----")
        print(tt_dev[:, :, row, :])
        print("--" * 40)
