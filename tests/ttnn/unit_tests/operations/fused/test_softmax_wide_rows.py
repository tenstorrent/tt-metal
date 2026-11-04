# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from tt_py_test_utils_common.utility_functions import is_wormhole_b0

pytestmark = [
    pytest.mark.use_module_device,
    pytest.mark.skipif(not is_wormhole_b0(), reason="Wormhole-only programs"),
]

DTYPES = [(ttnn.bfloat16, torch.bfloat16), (ttnn.float32, torch.float32)]
DTYPE_IDS = ["bfloat16", "float32"]


def _softmax_check(device, shape, dtypes, numeric_stable, scale=None, mask_shape=None):
    ttnn_dtype, torch_dtype = dtypes
    torch.manual_seed(0)
    x = (torch.randn(shape) * 4).to(torch_dtype)
    x_tt = ttnn.from_torch(x, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    if mask_shape is None:
        out_tt = ttnn.softmax(x_tt, dim=-1, numeric_stable=numeric_stable)
        golden = torch.softmax(x.float(), dim=-1)
    else:
        mask = torch.zeros(mask_shape)
        mask[..., (shape[-1] * 2) // 3 :] = -10000.0
        mask_tt = ttnn.from_torch(mask.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        out_tt = ttnn.scale_mask_softmax(x_tt, scale, mask_tt, numeric_stable=numeric_stable)
        golden = torch.softmax(x.float() * scale + mask[..., :1, :].float(), dim=-1)
    out = ttnn.to_torch(out_tt).float()
    assert out.shape == golden.shape
    assert_with_pcc(golden, out, 0.999)
    assert torch.allclose(out.sum(dim=-1), torch.ones(out.shape[:-1]), atol=0.05)


@pytest.mark.parametrize("dtypes", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize(
    "shape",
    [(1, 1, 32, 151936), (1, 1, 32, 128000), (1, 1, 64, 128000), (1, 1, 32, 32000), (1, 1, 128, 32000)],
)
def test_softmax_long_rows(device, shape, dtypes):
    _softmax_check(device, shape, dtypes, numeric_stable=True)


@pytest.mark.parametrize("dtypes", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize(
    "shape",
    [(1, 1, 32, 8192), (1, 1, 64, 8192), (1, 8, 32, 4096), (1, 32, 32, 4096), (1, 1, 1024, 1024), (1, 1, 256, 4096)],
)
def test_softmax_wide_rows_multi_core(device, shape, dtypes):
    _softmax_check(device, shape, dtypes, numeric_stable=True)


@pytest.mark.parametrize("numeric_stable", [True, False])
@pytest.mark.parametrize("dtypes", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("width", [224, 256, 288, 512, 4864, 4992, 10240])
def test_softmax_row_width_thresholds(device, width, dtypes, numeric_stable):
    _softmax_check(device, (1, 1, 2048, width) if width >= 4864 else (1, 8, 256, width), dtypes, numeric_stable)


@pytest.mark.parametrize("numeric_stable", [True, False])
@pytest.mark.parametrize("dtypes", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("shape", [(1, 12, 384, 384), (8, 12, 128, 128), (1, 1, 32, 8192)])
def test_scale_mask_softmax_wide_rows(device, shape, dtypes, numeric_stable):
    _softmax_check(device, shape, dtypes, numeric_stable, scale=0.125, mask_shape=(shape[0], 1, 32, shape[-1]))
