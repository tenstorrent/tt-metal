# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest


def torch_equal_nan(a, b):
    return torch.all((a == b) | (torch.isnan(a) & torch.isnan(b))).item()


@pytest.mark.parametrize(
    "c_shape, t_shape, f_shape",
    [
        ((8, 4, 1), (8, 4, 768), (1, 1, 1)),  # Ccol, Tfull, Fscalar
        ((8, 1, 768), (8, 4, 768), (1, 1, 1)),  # Crow, Tfull, Fscalar
        ((8, 4, 768), (8, 4, 1), (8, 1, 768)),  # Cfull, Tcol, Frow
        ((8, 4, 768), (8, 1, 768), (8, 4, 1)),  # Cfull, Trow, Fcol
        ((8, 4, 1), (8, 1, 768), (8, 4, 768)),  # Ccol, Trow, Ffull
        ((8, 1, 768), (8, 4, 1), (8, 4, 768)),  # Crow, Tcol, Ffull
        ((8, 1, 768), (8, 4, 768), (8, 4, 1)),  # Crow, Tfull, Fcol
        ((8, 4, 1), (8, 4, 768), (8, 1, 768)),  # Ccol, Tfull, Frow
        ((1, 1, 1), (8, 4, 1), (8, 1, 768)),  # Cscalar, Tcol, Frow
        ((1, 1, 1), (8, 1, 768), (8, 4, 1)),  # Cscalar, Trow, Fcol
        ((8, 1, 768), (1, 1, 1), (8, 4, 1)),  # Crow, Tscalar, Fcol
        ((8, 4, 1), (1, 1, 1), (8, 1, 768)),  # Ccol, Tscalar, Frow
    ],
)
def test_ttnn_where_row_col_mixed_bcast(c_shape, t_shape, f_shape, device):
    torch.manual_seed(0)
    C = torch.randint(0, 2, c_shape, dtype=torch.float32)
    T = torch.randn(t_shape, dtype=torch.float32)
    F = torch.ones(f_shape, dtype=torch.float32) * 10
    golden = torch.where(C.bool(), T, F)

    ttnn_C = ttnn.from_torch(C, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_T = ttnn.from_torch(T, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_F = ttnn.from_torch(F, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_result = ttnn.where(ttnn_C, ttnn_T, ttnn_F)
    result = ttnn.to_torch(ttnn_result)

    assert torch_equal_nan(result, golden)


@pytest.mark.parametrize(
    "c_shape, t_shape",
    [
        ((8, 4, 1), (8, 1, 768)),  # Ccol, Trow
        ((8, 1, 768), (8, 4, 1)),  # Crow, Tcol
    ],
)
def test_ttnn_where_row_col_mixed_bcast_tts(c_shape, t_shape, device):
    torch.manual_seed(0)
    C = torch.randint(0, 2, c_shape, dtype=torch.float32)
    T = torch.randn(t_shape, dtype=torch.float32)
    F = 10.0
    golden = torch.where(C.bool(), T, F)

    ttnn_C = ttnn.from_torch(C, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_T = ttnn.from_torch(T, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_result = ttnn.where(ttnn_C, ttnn_T, F)
    result = ttnn.to_torch(ttnn_result)

    assert torch_equal_nan(result, golden)


@pytest.mark.parametrize(
    "c_shape, f_shape",
    [
        ((8, 4, 1), (8, 1, 768)),  # Ccol, Frow
        ((8, 1, 768), (8, 4, 1)),  # Crow, Fcol
    ],
)
def test_ttnn_where_row_col_mixed_bcast_tst(c_shape, f_shape, device):
    torch.manual_seed(0)
    C = torch.randint(0, 2, c_shape, dtype=torch.float32)
    T = 10.0
    F = torch.randn(f_shape, dtype=torch.float32)
    golden = torch.where(C.bool(), T, F)

    ttnn_C = ttnn.from_torch(C, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_F = ttnn.from_torch(F, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_result = ttnn.where(ttnn_C, T, ttnn_F)
    result = ttnn.to_torch(ttnn_result)

    assert torch_equal_nan(result, golden)
