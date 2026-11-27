# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""ttnn.transpose / ttnn.permute on MX (OCP microscaling) tensors on Quasar.

Only the 4D CN transpose / permute moves whole MX tiles natively. Every other call that moves data runs in bfloat16 and
is packed back to MX (interleaved input and output only); no-op calls keep their native paths, including sharded ones.
"""

import pytest
import torch

import ttnn
from tests.ttnn.nightly.unit_tests.operations.experimental.quasar.mx_test_utils import (
    check,
    mx_round_trip,
    pcc_floor,
    skip_unless_quasar,
    to_device,
)

DTYPES = [ttnn.mxint8, ttnn.mxfp8_e4m3]
SHAPE = (1, 2, 64, 96)


@pytest.fixture(autouse=True)
def _quasar_only(device):
    skip_unless_quasar(device)


def _sharded_mx(device, dtype, shape=(1, 1, 64, 32)):
    x = torch.randn(shape)
    cfg = ttnn.create_sharded_memory_config(
        shape=(32, 32),
        core_grid=ttnn.CoreGrid(y=1, x=2),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    return x, to_device(device, x, dtype, memory_config=cfg)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "dim0, dim1", [(0, 1), (2, 3), (1, 2), (0, 3), (-1, -2)], ids=["CN", "HW", "CH", "NW", "WH_neg"]
)
def test_transpose_mx(device, dtype, dim0, dim1):
    torch.manual_seed(0)
    x = torch.randn(SHAPE)
    out = ttnn.transpose(to_device(device, x, dtype), dim0, dim1)
    assert out.dtype == dtype
    golden = mx_round_trip(mx_round_trip(x, dtype).transpose(dim0, dim1), dtype)
    check(golden, ttnn.to_torch(out), pcc_floor(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
def test_transpose_mx_identity_same_dim(device, dtype):
    torch.manual_seed(0)
    x = torch.randn(SHAPE)
    out = ttnn.transpose(to_device(device, x, dtype), 2, 2)
    assert out.dtype == dtype
    check(mx_round_trip(x, dtype), ttnn.to_torch(out), pcc_floor(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("dims", [(0, 1, 3, 2), (0, 2, 1, 3), (1, 0, 2, 3), (3, 2, 1, 0)])
def test_permute_mx(device, dtype, dims):
    torch.manual_seed(0)
    x = torch.randn(SHAPE)
    out = ttnn.permute(to_device(device, x, dtype), dims)
    assert out.dtype == dtype
    golden = mx_round_trip(mx_round_trip(x, dtype).permute(dims), dtype)
    check(golden, ttnn.to_torch(out), pcc_floor(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
def test_permute_mx_rank5(device, dtype):
    torch.manual_seed(0)
    x = torch.randn(1, 2, 2, 32, 64)
    dims = (0, 1, 2, 4, 3)
    out = ttnn.permute(to_device(device, x, dtype), dims)
    golden = mx_round_trip(mx_round_trip(x, dtype).permute(dims), dtype)
    check(golden, ttnn.to_torch(out), pcc_floor(dtype))


# Sharded MX: identity and CN keep their native paths; transposes / permutes that move data are rejected.
@pytest.mark.parametrize("dtype", DTYPES)
def test_transpose_mx_sharded_identity(device, dtype):
    x, tt = _sharded_mx(device, dtype)
    out = ttnn.transpose(tt, 3, 3)
    assert out.dtype == dtype
    check(mx_round_trip(x, dtype), ttnn.to_torch(out), pcc_floor(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
def test_permute_mx_sharded_identity(device, dtype):
    x, tt = _sharded_mx(device, dtype)
    out = ttnn.permute(tt, (0, 1, 2, 3))
    assert out.dtype == dtype
    check(mx_round_trip(x, dtype), ttnn.to_torch(out), pcc_floor(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
def test_transpose_mx_sharded_rejected(device, dtype, expect_error):
    _, tt = _sharded_mx(device, dtype)
    with expect_error(RuntimeError, "interleaved input and output only"):
        ttnn.transpose(tt, 2, 3)


@pytest.mark.parametrize("dtype", DTYPES)
def test_permute_mx_sharded_rejected(device, dtype, expect_error):
    _, tt = _sharded_mx(device, dtype)
    with expect_error(RuntimeError, "interleaved input and output only"):
        ttnn.permute(tt, (0, 1, 3, 2))
