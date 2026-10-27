# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""ttnn.layer_norm / rms_norm over MX (OCP microscaling) inputs on Quasar (issue #62742).

The interleaved layernorm unpacks the MX input (and residual) to Float16_b, keeps every intermediate ring in
Float16_b / Float32, and packs the output in the input's format; gamma / beta stay bf16.
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

SHAPE = (1, 1, 64, 96)
EPS = 1e-5


@pytest.fixture(autouse=True)
def _quasar_only(device):
    skip_unless_quasar(device)


def _rms_norm(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


@pytest.mark.parametrize("dtype", ttnn.MX_DTYPES)
@pytest.mark.parametrize("residual", [False, True], ids=["plain", "residual"])
def test_layer_norm_mx(device, dtype, residual):
    torch.manual_seed(0)
    x = torch.randn(SHAPE)
    r = torch.randn(SHAPE)
    gamma = torch.randn(SHAPE[-1]).bfloat16()
    beta = torch.randn(SHAPE[-1]).bfloat16()
    kwargs = {"residual_input_tensor": to_device(device, r, dtype)} if residual else {}
    out = ttnn.layer_norm(
        to_device(device, x, dtype),
        epsilon=EPS,
        weight=to_device(device, gamma.reshape(1, 1, 1, -1), ttnn.bfloat16),
        bias=to_device(device, beta.reshape(1, 1, 1, -1), ttnn.bfloat16),
        **kwargs,
    )
    assert out.dtype == dtype
    xin = mx_round_trip(x, dtype) + (mx_round_trip(r, dtype) if residual else 0)
    golden = torch.nn.functional.layer_norm(xin, (SHAPE[-1],), gamma.float(), beta.float(), EPS)
    check(mx_round_trip(golden, dtype), ttnn.to_torch(out), pcc_floor(dtype))


@pytest.mark.parametrize("dtype", ttnn.MX_DTYPES)
def test_rms_norm_mx(device, dtype):
    torch.manual_seed(0)
    x = torch.randn(SHAPE)
    gamma = torch.randn(SHAPE[-1]).bfloat16()
    out = ttnn.rms_norm(
        to_device(device, x, dtype), epsilon=EPS, weight=to_device(device, gamma.reshape(1, 1, 1, -1), ttnn.bfloat16)
    )
    assert out.dtype == dtype
    golden = _rms_norm(mx_round_trip(x, dtype), gamma.float(), EPS)
    check(mx_round_trip(golden, dtype), ttnn.to_torch(out), pcc_floor(dtype))


@pytest.mark.parametrize("dtype", [ttnn.mxfp8_e4m3, ttnn.mxint8, ttnn.mxfp4])
def test_rms_norm_mx_residual(device, dtype):
    torch.manual_seed(0)
    x = torch.randn(SHAPE)
    r = torch.randn(SHAPE)
    gamma = torch.randn(SHAPE[-1]).bfloat16()
    out = ttnn.rms_norm(
        to_device(device, x, dtype),
        epsilon=EPS,
        weight=to_device(device, gamma.reshape(1, 1, 1, -1), ttnn.bfloat16),
        residual_input_tensor=to_device(device, r, dtype),
    )
    assert out.dtype == dtype
    golden = _rms_norm(mx_round_trip(x, dtype) + mx_round_trip(r, dtype), gamma.float(), EPS)
    check(mx_round_trip(golden, dtype), ttnn.to_torch(out), pcc_floor(dtype))


@pytest.mark.parametrize("dtype", [ttnn.mxfp8_e4m3, ttnn.mxint8, ttnn.mxfp4])
def test_layer_norm_mx_fp32_dest_acc(device, dtype):
    # Float32 intermediates and a 32-bit Dest; the MX input is still expanded to Float16_b by the unpacker.
    torch.manual_seed(0)
    x = torch.randn(SHAPE)
    gamma = torch.randn(SHAPE[-1]).bfloat16()
    beta = torch.randn(SHAPE[-1]).bfloat16()
    cfg = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
    )
    out = ttnn.layer_norm(
        to_device(device, x, dtype),
        epsilon=EPS,
        weight=to_device(device, gamma.reshape(1, 1, 1, -1), ttnn.bfloat16),
        bias=to_device(device, beta.reshape(1, 1, 1, -1), ttnn.bfloat16),
        compute_kernel_config=cfg,
    )
    assert out.dtype == dtype
    golden = torch.nn.functional.layer_norm(mx_round_trip(x, dtype), (SHAPE[-1],), gamma.float(), beta.float(), EPS)
    check(mx_round_trip(golden, dtype), ttnn.to_torch(out), pcc_floor(dtype))


@pytest.mark.parametrize("dtype", [ttnn.mxfp8_e4m3, ttnn.mxint8, ttnn.mxfp4])
@pytest.mark.parametrize("is_rms", [False, True], ids=["layer_norm", "rms_norm"])
def test_distributed_norm_mx(device, dtype, is_rms):
    # Single-device pipeline of the distributed norm: pre_all_gather computes the bf16 stats from the MX input,
    # post_all_gather normalizes the same MX input with them. The host side accepts MX; the op itself is not on
    # Quasar yet (layernorm_distributed_metal2_helpers.hpp builds Gen1 compute configs only and FATALs on Quasar).
    pytest.skip("layernorm_distributed is Wormhole / Blackhole only (Gen1 compute config FATAL on Quasar)")
    torch.manual_seed(0)
    x = torch.randn(SHAPE)
    gamma = torch.randn(SHAPE[-1]).bfloat16()
    beta = torch.randn(SHAPE[-1]).bfloat16()
    xt = to_device(device, x, dtype)
    gt = to_device(device, gamma.reshape(1, 1, 1, -1), ttnn.bfloat16)
    if is_rms:
        stats = ttnn.rms_norm_pre_all_gather(xt, dtype=ttnn.bfloat16)
        assert stats.dtype == ttnn.bfloat16
        out = ttnn.rms_norm_post_all_gather(xt, stats, epsilon=EPS, weight=gt, dtype=dtype)
        golden = _rms_norm(mx_round_trip(x, dtype), gamma.float(), EPS)
    else:
        stats = ttnn.layer_norm_pre_all_gather(xt, dtype=ttnn.bfloat16)
        assert stats.dtype == ttnn.bfloat16
        bt = to_device(device, beta.reshape(1, 1, 1, -1), ttnn.bfloat16)
        out = ttnn.layer_norm_post_all_gather(xt, stats, epsilon=EPS, weight=gt, bias=bt, dtype=dtype)
        golden = torch.nn.functional.layer_norm(mx_round_trip(x, dtype), (SHAPE[-1],), gamma.float(), beta.float(), EPS)
    assert out.dtype == dtype
    check(mx_round_trip(golden, dtype), ttnn.to_torch(out), pcc_floor(dtype))


@pytest.mark.parametrize("dtype", [ttnn.mxfp8_e4m3, ttnn.mxint8])
def test_layer_norm_mx_sharded(device, dtype):
    # Block-sharded input over a 2x2 grid (one 32x128 shard per core), sharded output in place.
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))])
    gx, gy = device.compute_with_storage_grid_size().x, device.compute_with_storage_grid_size().y
    if gx < 2 or gy < 2:
        pytest.skip("needs a 2x2 grid")
    shape = (1, 1, 64, 256)
    shard_shape = (32, 128)
    mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, shard_shape, ttnn.ShardOrientation.COL_MAJOR),
    )
    torch.manual_seed(0)
    x = torch.randn(shape)
    gamma = torch.randn(shape[-1]).bfloat16()
    beta = torch.randn(shape[-1]).bfloat16()
    cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(2, 2), subblock_w=4, block_h=1, block_w=4, inplace=True
    )
    out = ttnn.layer_norm(
        to_device(device, x, dtype, memory_config=mem),
        epsilon=EPS,
        weight=to_device(device, gamma.reshape(1, 1, 1, -1), ttnn.bfloat16),
        bias=to_device(device, beta.reshape(1, 1, 1, -1), ttnn.bfloat16),
        memory_config=mem,
        program_config=cfg,
    )
    assert out.dtype == dtype
    golden = torch.nn.functional.layer_norm(mx_round_trip(x, dtype), (shape[-1],), gamma.float(), beta.float(), EPS)
    check(mx_round_trip(golden, dtype), ttnn.to_torch(out), pcc_floor(dtype))
