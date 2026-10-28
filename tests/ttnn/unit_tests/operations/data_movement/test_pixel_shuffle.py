# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.pixel_shuffle: the inverse of ttnn.pixel_unshuffle.

NCHW: [N, C*r^2, H, W] -> [N, C, H*r, W*r]; CHANNEL_MAJOR == torch.nn.functional.pixel_shuffle, SPATIAL_MAJOR == ONNX
DepthToSpace (DCR). channels_last: NHWC [N, H, W, C*r^2] height-sharded L1 -> NHWC [N, H*r, W*r, padded_channels]
height-sharded L1 (the layout a height-sharded conv2d consumes), and the 1x1 matmul + pixel_shuffle formulation of a
stride-r transposed conv against torch.conv_transpose2d."""
import pytest
import torch
import torch.nn.functional as F
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

CHANNEL_MAJOR = ttnn.PixelUnshuffleChannelOrder.CHANNEL_MAJOR
SPATIAL_MAJOR = ttnn.PixelUnshuffleChannelOrder.SPATIAL_MAJOR
ORDERS = [pytest.param(CHANNEL_MAJOR, "channel_major", id="channel_major"), pytest.param(SPATIAL_MAJOR, "spatial_major", id="spatial_major")]
HS = ttnn.TensorMemoryLayout.HEIGHT_SHARDED


def golden_nchw(x, r, order):
    """[N, C*r^2, H, W] -> [N, C, H*r, W*r]."""
    N, Cin, H, W = x.shape
    C = Cin // (r * r)
    if order == "channel_major":
        t = x.reshape(N, C, r, r, H, W).permute(0, 1, 4, 2, 5, 3)  # [N, C, H, rh, W, rw]
    else:
        t = x.reshape(N, r, r, C, H, W).permute(0, 3, 4, 1, 5, 2)  # [N, C, H, rh, W, rw]
    return t.reshape(N, C, H * r, W * r)


def golden_nhwc(x, r, order):
    """[N, H, W, C*r^2] -> [N, H*r, W*r, C]."""
    N, H, W, Cin = x.shape
    C = Cin // (r * r)
    if order == "channel_major":
        t = x.reshape(N, H, W, C, r, r).permute(0, 1, 4, 2, 5, 3)  # [N, H, rh, W, rw, C]
    else:
        t = x.reshape(N, H, W, r, r, C).permute(0, 1, 3, 2, 4, 5)
    return t.reshape(N, H * r, W * r, C)


def _grid(n):
    full, rem = divmod(n, 8)
    rs = set()
    if full:
        rs.add(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, full - 1)))
    if rem:
        rs.add(ttnn.CoreRange(ttnn.CoreCoord(0, full), ttnn.CoreCoord(rem - 1, full)))
    return ttnn.CoreRangeSet(rs)


def _hs(cores, rows, cols):
    return ttnn.MemoryConfig(HS, ttnn.BufferType.L1, ttnn.ShardSpec(_grid(cores), (rows, cols), ttnn.ShardOrientation.ROW_MAJOR))


def test_golden_matches_torch():
    x = torch.randn(2, 3 * 4, 6, 8)
    assert torch.equal(golden_nchw(x, 2, "channel_major"), F.pixel_shuffle(x, 2))
    assert torch.equal(golden_nchw(F.pixel_unshuffle(x, 2), 2, "channel_major"), x)


@pytest.mark.parametrize("N,C,H,W", [(1, 1, 16, 16), (1, 3, 8, 12), (2, 4, 6, 10), (1, 2, 32, 48)])
@pytest.mark.parametrize("r", [1, 2, 3, 4])
@pytest.mark.parametrize("input_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("order_enum,order_name", ORDERS)
def test_pixel_shuffle_nchw(device, N, C, H, W, r, input_layout, order_enum, order_name):
    torch.manual_seed(0)
    x = torch.randn(N, C * r * r, H, W).to(torch.bfloat16)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=input_layout, device=device)
    out = ttnn.pixel_shuffle(tx, r, channel_order=order_enum)
    assert list(out.shape) == [N, C, H * r, W * r]
    assert out.layout == ttnn.ROW_MAJOR_LAYOUT
    assert torch.equal(ttnn.to_torch(out).float(), golden_nchw(x.float(), r, order_name))


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.uint16, ttnn.int32])
def test_pixel_shuffle_dtypes(device, dtype):
    torch.manual_seed(0)
    r = 2
    if dtype in (ttnn.uint16, ttnn.int32):
        x = torch.randint(0, 1000, (1, 8, 6, 10), dtype=torch.int32)
    else:
        x = torch.randn(1, 8, 6, 10)
    tx = ttnn.from_torch(x, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    out = ttnn.pixel_shuffle(tx, r)
    assert out.dtype == dtype
    assert torch.equal(ttnn.to_torch(out).to(torch.int64 if dtype in (ttnn.uint16, ttnn.int32) else torch.float32), golden_nchw(ttnn.to_torch(tx), r, "channel_major").to(torch.int64 if dtype in (ttnn.uint16, ttnn.int32) else torch.float32))


@pytest.mark.parametrize("odd_w", [5, 7])
def test_pixel_shuffle_odd_width(device, odd_w):
    x = torch.randn(1, 12, 4, odd_w).to(torch.bfloat16)
    out = ttnn.pixel_shuffle(ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device), 2)
    assert torch.equal(ttnn.to_torch(out).float(), F.pixel_shuffle(x.float(), 2))


def test_pixel_shuffle_tile_output_and_memory_configs(device):
    x = torch.randn(1, 16, 32, 32).to(torch.bfloat16)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
    out = ttnn.pixel_shuffle(tx, 4, output_layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    assert out.layout == ttnn.TILE_LAYOUT and list(out.shape) == [1, 1, 128, 128]
    assert torch.equal(ttnn.to_torch(out).float(), F.pixel_shuffle(x.float(), 4))
    hs = _hs(8, 16, 128)  # output [1, 1, 128, 128] row-major sharded: 16 rows per core
    out2 = ttnn.pixel_shuffle(tx, 4, memory_config=hs)
    assert torch.equal(ttnn.to_torch(out2).float(), F.pixel_shuffle(x.float(), 4))


@pytest.mark.parametrize("order_enum,order_name", ORDERS)
@pytest.mark.parametrize("r", [2, 3])
def test_round_trip_with_pixel_unshuffle(device, order_enum, order_name, r):
    x = torch.randn(1, 4, 12 * r, 10 * r).to(torch.bfloat16)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    down = ttnn.pixel_unshuffle(tx, r, channel_order=order_enum)
    up = ttnn.pixel_shuffle(down, r, channel_order=order_enum)
    assert torch.equal(ttnn.to_torch(up).float(), x.float())


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "N, H, W, C, r, in_cores, out_cores, order_enum, order_name, padded",
    [
        (1, 80, 144, 192, 2, 60, 60, SPATIAL_MAJOR, "spatial_major", None),  # block C: 80x144 -> 160x288
        (1, 40, 72, 192, 2, 45, 45, SPATIAL_MAJOR, "spatial_major", None),
        (1, 20, 36, 192, 2, 15, 15, SPATIAL_MAJOR, "spatial_major", None),
        (1, 10, 18, 192, 2, 10, 10, SPATIAL_MAJOR, "spatial_major", None),
        (1, 80, 144, 192, 2, 60, 64, SPATIAL_MAJOR, "spatial_major", None),  # output on another grid
        (1, 16, 24, 32, 3, 8, 12, SPATIAL_MAJOR, "spatial_major", None),  # r = 3
        (2, 12, 20, 64, 2, 5, 16, SPATIAL_MAJOR, "spatial_major", None),  # N = 2
        (1, 40, 72, 192, 2, 45, 45, CHANNEL_MAJOR, "channel_major", None),  # torch order: strided gather path
        (1, 16, 24, 24, 2, 8, 8, SPATIAL_MAJOR, "spatial_major", 32),  # padded output channels
        (1, 16, 24, 24, 2, 8, 8, CHANNEL_MAJOR, "channel_major", 32),
    ],
)
def test_pixel_shuffle_channels_last(device, N, H, W, C, r, in_cores, out_cores, order_enum, order_name, padded):
    torch.manual_seed(0)
    x = torch.randn(N, H, W, C * r * r).to(torch.bfloat16)
    ref = golden_nhwc(x.float(), r, order_name)
    rows_in = -(-N * H * W // in_cores)
    rows_out = -(-N * H * W * r * r // out_cores)
    rows_out = -(-rows_out // r) * r
    cp = padded or C
    tx = ttnn.from_torch(x, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=_hs(in_cores, rows_in, C * r * r))
    out = ttnn.pixel_shuffle(tx, r, memory_config=_hs(out_cores, rows_out, cp), channel_order=order_enum, channels_last=True, padded_channels=padded)
    assert list(out.shape) == [N, H * r, W * r, cp]
    res = ttnn.to_torch(out).float()
    assert torch.equal(res[..., :C], ref), f"max diff {(res[..., :C] - ref).abs().max()}"
    if cp > C:
        assert torch.equal(res[..., C:], torch.zeros_like(res[..., C:]))


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_channels_last_round_trip(device):
    """pixel_shuffle(channels_last) inverts pixel_unshuffle(channels_last) (NCHW in, NHWC out) back to NHWC."""
    x = torch.randn(1, 8, 32, 48).to(torch.bfloat16)  # NCHW, C = 8
    tx = ttnn.from_torch(x, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    down = ttnn.pixel_unshuffle(tx, 2, memory_config=_hs(8, 48, 32), channels_last=True, padded_channels=32)  # [1, 16, 24, 32]
    up = ttnn.pixel_shuffle(down, 2, memory_config=_hs(8, 192, 8), channels_last=True)  # [1, 32, 48, 8]
    assert torch.equal(ttnn.to_torch(up).float(), x.float().permute(0, 2, 3, 1))


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("H, W, Cin, Cout, cores, out_cores", [(80, 144, 192, 192, 60, 60), (10, 18, 192, 192, 6, 6), (40, 72, 192, 192, 45, 60)])
def test_conv_transpose_as_matmul_plus_pixel_shuffle(device, H, W, Cin, Cout, cores, out_cores):
    """conv_transpose2d(k=2, s=2) == pixel_shuffle(SPATIAL_MAJOR, channels_last)(linear(x, W1)) with
    W1[ci, (dy*2+dx)*Cout + co] = Wt[ci, co, dy, dx]; the linear writes row-major output (untilize_out)."""
    torch.manual_seed(0)
    r = 2
    x = torch.randn(1, Cin, H, W).to(torch.bfloat16).float()
    wt = (torch.randn(Cin, Cout, r, r) / (Cin * 4) ** 0.5).to(torch.bfloat16).float()
    ref = F.conv_transpose2d(x, wt, stride=r)  # [1, Cout, 2H, 2W]
    w1 = wt.permute(2, 3, 1, 0).reshape(Cout * r * r, Cin).T.contiguous()  # [K, N]
    rows_in = -(-(-(-H * W // cores)) // 32) * 32
    m = rows_in // 32
    compute = ttnn.init_device_compute_kernel_config(device.arch(), math_fidelity=ttnn.MathFidelity.HiFi3, fp32_dest_acc_en=True)
    x_nhwc = x.permute(0, 2, 3, 1).reshape(1, 1, H * W, Cin).contiguous().to(torch.bfloat16)
    tx = ttnn.from_torch(x_nhwc, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=_hs(cores, rows_in, Cin))
    tw = ttnn.from_torch(w1.to(torch.bfloat16), ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    cfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 8), in0_block_w=Cin // 32, out_subblock_h=1, out_subblock_w=4, out_block_h=min(m, 2), out_block_w=Cout * 4 // 32,
        per_core_M=m, per_core_N=Cout * 4 // 32, fuse_batch=True, fused_activation=None, mcast_in0=False, gather_in0=False, hop_cores=ttnn.CoreRangeSet([]), num_global_cb_receivers=0, untilize_out=True)
    y = ttnn.matmul(tx, tw, memory_config=_hs(cores, rows_in, Cout * 4), program_config=cfg, compute_kernel_config=compute, dtype=ttnn.bfloat16)
    y4 = ttnn.reshape(y, [1, H, W, Cout * 4])
    rows_out = -(-H * W * 4 // out_cores)
    rows_out = -(-rows_out // r) * r
    up = ttnn.pixel_shuffle(y4, r, memory_config=_hs(out_cores, rows_out, Cout), channel_order=SPATIAL_MAJOR, channels_last=True)
    res = ttnn.to_torch(up).float().permute(0, 3, 1, 2)
    assert_with_pcc(ref, res, 0.999)
