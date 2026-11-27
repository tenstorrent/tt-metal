# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""typecast, untilize, to_layout and paged_update_cache with MX (OCP microscaling) tensors on Quasar."""

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

DTYPES = [ttnn.mxint8, ttnn.mxfp8_e4m3, ttnn.mxint4]


@pytest.fixture(autouse=True)
def _quasar_only(device):
    skip_unless_quasar(device)


@pytest.mark.parametrize("dtype", [ttnn.mxint4, ttnn.mxint8])
def test_typecast_bf16_mx_round_trip(device, dtype):
    torch.manual_seed(0)
    x = torch.randn(1, 1, 64, 96)
    bf16 = to_device(device, x, ttnn.bfloat16)
    mx = ttnn.typecast(bf16, dtype)
    assert mx.dtype == dtype
    check(mx_round_trip(x, dtype), ttnn.to_torch(mx), pcc_floor(dtype))
    back = ttnn.typecast(mx, ttnn.bfloat16)
    assert back.dtype == ttnn.bfloat16
    # bf16 -> MX -> bf16 holds exactly the MX values
    check(mx_round_trip(x, dtype), ttnn.to_torch(back), 0.9999)


@pytest.mark.parametrize("dtype", DTYPES)
def test_untilize_mx(device, dtype):
    torch.manual_seed(0)
    x = torch.randn(1, 1, 64, 96)
    out = ttnn.untilize(to_device(device, x, dtype))
    assert out.layout == ttnn.ROW_MAJOR_LAYOUT
    assert out.dtype == ttnn.bfloat16
    check(mx_round_trip(x, dtype), ttnn.to_torch(out), pcc_floor(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
def test_to_layout_mx_to_row_major(device, dtype):
    torch.manual_seed(0)
    x = torch.randn(1, 1, 64, 96)
    out = ttnn.to_layout(to_device(device, x, dtype), ttnn.ROW_MAJOR_LAYOUT)
    assert out.layout == ttnn.ROW_MAJOR_LAYOUT
    assert out.dtype == ttnn.bfloat16
    check(mx_round_trip(x, dtype), ttnn.to_torch(out), pcc_floor(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
def test_to_layout_row_major_bf16_to_mx_tile(device, dtype):
    torch.manual_seed(0)
    x = torch.randn(1, 1, 64, 96).to(torch.bfloat16)
    rm = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    out = ttnn.to_layout(rm, ttnn.TILE_LAYOUT, dtype=dtype)
    assert out.layout == ttnn.TILE_LAYOUT
    assert out.dtype == dtype
    check(mx_round_trip(x.float(), dtype), ttnn.to_torch(out), pcc_floor(dtype))


@pytest.mark.parametrize("dtype", [ttnn.mxint8, ttnn.mxfp8_e4m3])
def test_paged_update_cache_mx_cache(device, dtype):
    torch.manual_seed(0)
    num_users, num_heads, head_dim, max_seq_len = 2, 1, 64, 128
    cache = torch.randn(num_users, num_heads, max_seq_len, head_dim)
    cachett = ttnn.from_torch(cache, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    x = torch.randn(1, num_users, num_heads, head_dim).bfloat16().float()
    x_pad = torch.nn.functional.pad(x, (0, 0, 0, 32 - num_heads), "constant", 0)
    xt = ttnn.Tensor(x_pad, ttnn.bfloat16).to(ttnn.TILE_LAYOUT)
    xt = ttnn.reshape(xt, ttnn.Shape([1, num_users, num_heads, head_dim]))
    num_cores = num_users
    shard_grid = ttnn.num_cores_to_corerangeset(num_cores, device.compute_with_storage_grid_size(), True)
    spec = ttnn.ShardSpec(
        shard_grid,
        [xt.volume() // xt.padded_shape[-1] // num_cores, xt.padded_shape[-1]],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    xt = xt.to(device, ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec))

    idxs = [3, 40]
    cachett = ttnn.experimental.paged_update_cache(cachett, xt, update_idxs=idxs, share_cache=False)
    assert cachett.dtype == dtype

    got = ttnn.to_torch(cachett)
    # compare the updated rows
    for i, idx in enumerate(idxs):
        check(
            mx_round_trip(x.permute(1, 2, 0, 3)[i : i + 1], dtype).reshape(1, -1),
            got[i, :, idx : idx + 1, :].reshape(1, -1),
            pcc_floor(dtype),
        )
