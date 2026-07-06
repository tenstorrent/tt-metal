# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Comprehensive tests for ttnn.pixel_unshuffle — dedicated NCHW direct-gather kernel.
#
# Golden reference (reshape -> permute -> reshape), parameterized by channel_order:
#
#   x: [N, C, H, W]
#   reshape -> [N, C, H/r, r, W/r, r]          (split H=(h_out,rh), W=(w_out,rw))
#   CHANNEL_MAJOR:  permute (0,1,3,5,2,4) -> [N, C, r, r, H/r, W/r]
#                  reshape -> [N, C*r*r, H/r, W/r]   c_out = c*r^2 + rh*r + rw
#                  (identical to torch.nn.functional.pixel_unshuffle)
#   SPATIAL_MAJOR:  permute (0,3,5,1,2,4) -> [N, r, r, C, H/r, W/r]
#                  reshape -> [N, C*r*r, H/r, W/r]   c_out = rh*(r*C) + rw*C + c
#                  (matches ONNX SpaceToDepth channel ordering)
#
# Both orderings produce the same shape and coincide when C == 1.
#
# Kernel properties exercised:
#   - Reads NCHW ROW_MAJOR input directly; TILE input is untilized first
#   - Writes NCHW output with any MemoryConfig via TensorAccessor (sharded output ok)
#   - Sharded input accepted natively (TensorAccessor resolves page_id across cores)
#   - output_layout=TILE_LAYOUT tilizes the ROW_MAJOR kernel output
#   - channel_order selects the c_out -> (c_in, rh, rw) decode in BOTH reader and writer

import pytest
import torch
import torch.nn.functional as F
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


# ---------------------------------------------------------------------------
# Golden + channel-order parameterization
# ---------------------------------------------------------------------------

CHANNEL_MAJOR = ttnn.PixelUnshuffleChannelOrder.CHANNEL_MAJOR
SPATIAL_MAJOR = ttnn.PixelUnshuffleChannelOrder.SPATIAL_MAJOR

# (enum, name) pairs used to parameterize + drive the golden.
CHANNEL_ORDERS = [
    pytest.param(CHANNEL_MAJOR, "channel_major", id="channel_major"),
    pytest.param(SPATIAL_MAJOR, "spatial_major", id="spatial_major"),
]


def golden_pixel_unshuffle(x, r, order_name):
    """reshape -> permute -> reshape reference for both channel orderings."""
    N, C, H, W = x.shape
    t = x.reshape(N, C, H // r, r, W // r, r)  # [N, C, h_out, rh, w_out, rw]
    if order_name == "channel_major":
        t = t.permute(0, 1, 3, 5, 2, 4).contiguous()  # [N, C, r, r, H/r, W/r]
    elif order_name == "spatial_major":
        t = t.permute(0, 3, 5, 1, 2, 4).contiguous()  # [N, r, r, C, H/r, W/r]
    else:
        raise ValueError(f"unknown channel order {order_name!r}")
    return t.reshape(N, C * r * r, H // r, W // r)


def test_golden_anchor_matches_torch():
    """Sanity anchor: the channel-major golden equals torch.pixel_unshuffle, and
    spatial-major diverges for C>1 but coincides for C=1."""
    x = torch.randn(2, 3, 8, 8)
    assert torch.allclose(golden_pixel_unshuffle(x, 2, "channel_major"), F.pixel_unshuffle(x, 2))
    assert not torch.allclose(
        golden_pixel_unshuffle(x, 2, "spatial_major"), golden_pixel_unshuffle(x, 2, "channel_major")
    )
    x1 = torch.randn(1, 1, 8, 8)
    assert torch.allclose(
        golden_pixel_unshuffle(x1, 2, "spatial_major"), golden_pixel_unshuffle(x1, 2, "channel_major")
    )


def _torch_dtype_for(ttnn_dtype):
    return {
        ttnn.bfloat16: torch.bfloat16,
        ttnn.float32: torch.float32,
        ttnn.uint16: torch.int16,
        ttnn.int32: torch.int32,
    }[ttnn_dtype]


def _make_input(ttnn_dtype, shape, seed):
    torch.manual_seed(seed)
    tdt = _torch_dtype_for(ttnn_dtype)
    if tdt in (torch.int16, torch.int32):
        # non-negative so signed/unsigned interpretation agrees for uint16
        return torch.randint(0, 4000, shape, dtype=tdt)
    return torch.randn(*shape, dtype=tdt)


# ---------------------------------------------------------------------------
# Section A — full sweep: shape × factor × layout × dtype × channel_order
#            (golden is reshape->permute->reshape for the selected order)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "N,C,H,W",
    [
        (1, 1, 64, 64),  # C=1 (orderings coincide)
        (1, 2, 48, 768),  # BEV UV-like, C=2
        (1, 3, 64, 64),  # RGB conv, C=3
        (1, 16, 32, 32),  # C=16
        (2, 4, 32, 32),  # batch > 1
    ],
)
@pytest.mark.parametrize("downscale_factor", [2, 4])
@pytest.mark.parametrize("input_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("order_enum,order_name", CHANNEL_ORDERS)
def test_pixel_unshuffle_correctness(
    device, N, C, H, W, downscale_factor, input_layout, input_dtype, order_enum, order_name
):
    """Correctness across shape/factor/layout/dtype for both channel orderings."""
    r = downscale_factor
    if H % r != 0 or W % r != 0:
        pytest.skip(f"H={H} or W={W} not divisible by r={r}")
    if input_layout == ttnn.TILE_LAYOUT and (H < 32 or W < 32):
        pytest.skip("TILE layout requires H,W >= 32")
    if input_dtype == ttnn.float32 and input_layout == ttnn.TILE_LAYOUT:
        pytest.skip("float32 TILE not always supported in this environment")

    x = _make_input(input_dtype, (N, C, H, W), seed=42)
    golden = golden_pixel_unshuffle(x.float(), r, order_name)

    tt_in = ttnn.from_torch(
        x, dtype=input_dtype, layout=input_layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_out = ttnn.pixel_unshuffle(tt_in, downscale_factor=r, channel_order=order_enum)
    result = ttnn.to_torch(ttnn.to_layout(tt_out, ttnn.ROW_MAJOR_LAYOUT)).float()

    assert list(result.shape) == [N, C * r * r, H // r, W // r], f"shape {list(result.shape)}"
    assert_with_pcc(golden, result, pcc=0.99)


# ---------------------------------------------------------------------------
# Section B — memory configs: input/output × {DRAM, L1} × channel_order
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("input_mem_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
@pytest.mark.parametrize("output_mem_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
@pytest.mark.parametrize("input_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("order_enum,order_name", CHANNEL_ORDERS)
def test_pixel_unshuffle_memory_configs(
    device, input_mem_config, output_mem_config, input_layout, order_enum, order_name
):
    """All DRAM/L1 in×out combinations, both layouts, both orderings."""
    N, C, H, W, r = 1, 4, 32, 32, 2
    x = _make_input(ttnn.bfloat16, (N, C, H, W), seed=11)
    golden = golden_pixel_unshuffle(x.float(), r, order_name)

    tt_in = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=input_layout, device=device, memory_config=input_mem_config)
    tt_out = ttnn.pixel_unshuffle(tt_in, downscale_factor=r, memory_config=output_mem_config, channel_order=order_enum)

    assert tt_out.memory_config().buffer_type == output_mem_config.buffer_type
    result = ttnn.to_torch(ttnn.to_layout(tt_out, ttnn.ROW_MAJOR_LAYOUT)).float()
    assert_with_pcc(golden, result, pcc=0.99)


# ---------------------------------------------------------------------------
# Section C — dtype sweep × channel_order (bfloat16, float32, uint16, int32)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.uint16, ttnn.int32])
@pytest.mark.parametrize("order_enum,order_name", CHANNEL_ORDERS)
def test_pixel_unshuffle_dtypes(device, dtype, order_enum, order_name):
    """Every supported dtype for both orderings — output dtype must be preserved."""
    N, C, H, W, r = 1, 4, 32, 32, 2
    x = _make_input(dtype, (N, C, H, W), seed=10)
    golden = golden_pixel_unshuffle(x.float(), r, order_name)

    tt_in = ttnn.from_torch(
        x, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_out = ttnn.pixel_unshuffle(tt_in, downscale_factor=r, channel_order=order_enum)

    assert tt_out.dtype == dtype, f"output dtype {tt_out.dtype} != input {dtype}"
    result = ttnn.to_torch(tt_out).float()
    assert_with_pcc(golden, result, pcc=0.99)


# ---------------------------------------------------------------------------
# Section D — downscale_factor sweep × channel_order (incl. r=1 identity, r=3)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("r", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("order_enum,order_name", CHANNEL_ORDERS)
def test_pixel_unshuffle_downscale_factors(device, r, order_enum, order_name):
    N, C, H, W = 1, 3, r * 8, r * 8
    x = _make_input(ttnn.bfloat16, (N, C, H, W), seed=r)
    golden = golden_pixel_unshuffle(x.float(), r, order_name)

    tt_in = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_out = ttnn.pixel_unshuffle(tt_in, downscale_factor=r, channel_order=order_enum)
    result = ttnn.to_torch(tt_out).float()

    assert list(result.shape) == [N, C * r * r, H // r, W // r]
    assert_with_pcc(golden, result, pcc=0.99)


# ---------------------------------------------------------------------------
# Section E — TILE output × channel_order
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "N,C,H,W,r",
    [
        (1, 1, 64, 64, 4),
        (1, 32, 48, 768, 2),
        (1, 3, 64, 64, 2),
        (2, 4, 32, 32, 2),
    ],
)
@pytest.mark.parametrize("input_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("order_enum,order_name", CHANNEL_ORDERS)
def test_pixel_unshuffle_tile_output(device, N, C, H, W, r, input_layout, order_enum, order_name):
    """output_layout=TILE_LAYOUT tilizes the kernel output, both orderings."""
    x = _make_input(ttnn.bfloat16, (N, C, H, W), seed=1)
    golden = golden_pixel_unshuffle(x.float(), r, order_name)

    tt_in = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=input_layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_out = ttnn.pixel_unshuffle(tt_in, downscale_factor=r, output_layout=ttnn.TILE_LAYOUT, channel_order=order_enum)

    assert tt_out.layout == ttnn.TILE_LAYOUT
    result = ttnn.to_torch(ttnn.to_layout(tt_out, ttnn.ROW_MAJOR_LAYOUT)).float()
    assert list(result.shape) == [N, C * r * r, H // r, W // r]
    assert_with_pcc(golden, result, pcc=0.99)


def test_pixel_unshuffle_tile_output_l1(device):
    """TILE output in L1 for both orderings."""
    for order_enum, order_name in [(CHANNEL_MAJOR, "channel_major"), (SPATIAL_MAJOR, "spatial_major")]:
        x = _make_input(ttnn.bfloat16, (1, 2, 64, 64), seed=30)
        golden = golden_pixel_unshuffle(x.float(), 2, order_name)
        tt_in = ttnn.from_torch(
            x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        tt_out = ttnn.pixel_unshuffle(
            tt_in,
            downscale_factor=2,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_layout=ttnn.TILE_LAYOUT,
            channel_order=order_enum,
        )
        assert tt_out.layout == ttnn.TILE_LAYOUT
        assert tt_out.memory_config().buffer_type == ttnn.BufferType.L1
        result = ttnn.to_torch(ttnn.to_layout(tt_out, ttnn.ROW_MAJOR_LAYOUT)).float()
        assert_with_pcc(golden, result, pcc=0.99)


# ---------------------------------------------------------------------------
# Section F — sharded output (HEIGHT_SHARDED L1) × channel_order
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "N,C,H,W,r",
    [
        (1, 1, 32, 32, 2),
        (1, 4, 32, 32, 2),
        (1, 2, 64, 64, 2),
    ],
)
@pytest.mark.parametrize("order_enum,order_name", CHANNEL_ORDERS)
def test_pixel_unshuffle_height_sharded_output(device, N, C, H, W, r, order_enum, order_name):
    """Kernel writes directly into HEIGHT_SHARDED L1 output, both orderings."""
    x = _make_input(ttnn.bfloat16, (N, C, H, W), seed=3)
    golden = golden_pixel_unshuffle(x.float(), r, order_name)

    C_out, Ho, Wo = C * r * r, H // r, W // r
    total_sticks = N * C_out * Ho
    n_cores = min(total_sticks, 8)
    shard_h = (total_sticks + n_cores - 1) // n_cores
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(n_cores - 1, 0))})
    shard_spec = ttnn.ShardSpec(grid, [shard_h, Wo], ttnn.ShardOrientation.ROW_MAJOR)
    sharded_cfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

    tt_in = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_out = ttnn.pixel_unshuffle(tt_in, downscale_factor=r, memory_config=sharded_cfg, channel_order=order_enum)

    assert tt_out.memory_config().is_sharded()
    result = ttnn.to_torch(tt_out).float()
    assert list(result.shape) == [N, C_out, Ho, Wo]
    assert_with_pcc(golden, result, pcc=0.99)


# ---------------------------------------------------------------------------
# Section G — sharded input accepted natively × channel_order
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("N,C,H,W,r", [(1, 1, 32, 32, 2), (1, 4, 32, 32, 2)])
@pytest.mark.parametrize("order_enum,order_name", CHANNEL_ORDERS)
def test_pixel_unshuffle_sharded_input(device, N, C, H, W, r, order_enum, order_name):
    """Sharded L1 input read natively via TensorAccessor, both orderings."""
    x = _make_input(ttnn.bfloat16, (N, C, H, W), seed=5)
    golden = golden_pixel_unshuffle(x.float(), r, order_name)

    total_input_sticks = N * C * H
    n_cores = min(total_input_sticks, 4)
    shard_h = (total_input_sticks + n_cores - 1) // n_cores
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(n_cores - 1, 0))})
    shard_spec = ttnn.ShardSpec(grid, [shard_h, W], ttnn.ShardOrientation.ROW_MAJOR)
    sharded_in_cfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

    tt_in = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_in_sharded = ttnn.to_memory_config(tt_in, sharded_in_cfg)
    tt_out = ttnn.pixel_unshuffle(tt_in_sharded, downscale_factor=r, channel_order=order_enum)
    result = ttnn.to_torch(tt_out).float()

    assert list(result.shape) == [N, C * r * r, H // r, W // r]
    assert_with_pcc(golden, result, pcc=0.99)


# ---------------------------------------------------------------------------
# Section H — channel sweep × channel_order
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("C", [1, 3, 16, 32, 64])
@pytest.mark.parametrize("order_enum,order_name", CHANNEL_ORDERS)
def test_pixel_unshuffle_channel_sweep(device, C, order_enum, order_name):
    N, H, W, r = 1, 32, 32, 2
    x = _make_input(ttnn.bfloat16, (N, C, H, W), seed=C)
    golden = golden_pixel_unshuffle(x.float(), r, order_name)

    tt_in = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_out = ttnn.pixel_unshuffle(tt_in, downscale_factor=r, channel_order=order_enum)
    result = ttnn.to_torch(tt_out).float()

    assert list(result.shape) == [1, C * 4, 16, 16]
    assert_with_pcc(golden, result, pcc=0.99)


# ---------------------------------------------------------------------------
# Section I — batch sweep × channel_order
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("N", [1, 2, 4])
@pytest.mark.parametrize("order_enum,order_name", CHANNEL_ORDERS)
def test_pixel_unshuffle_batch_sizes(device, N, order_enum, order_name):
    x = _make_input(ttnn.bfloat16, (N, 2, 32, 32), seed=N)
    golden = golden_pixel_unshuffle(x.float(), 2, order_name)

    tt_in = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_out = ttnn.pixel_unshuffle(tt_in, downscale_factor=2, channel_order=order_enum)
    result = ttnn.to_torch(tt_out).float()

    assert list(result.shape) == [N, 8, 16, 16]
    assert_with_pcc(golden, result, pcc=0.99)


# ---------------------------------------------------------------------------
# Section J — default channel_order is CHANNEL_MAJOR (backward compatible)
# ---------------------------------------------------------------------------


def test_pixel_unshuffle_default_is_channel_major(device):
    """Omitting channel_order must behave exactly like CHANNEL_MAJOR (== torch)."""
    x = _make_input(ttnn.bfloat16, (1, 4, 32, 32), seed=99)
    golden_cm = golden_pixel_unshuffle(x.float(), 2, "channel_major")

    tt_in = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    y_default = ttnn.to_torch(ttnn.pixel_unshuffle(tt_in, downscale_factor=2)).float()
    y_explicit = ttnn.to_torch(ttnn.pixel_unshuffle(tt_in, downscale_factor=2, channel_order=CHANNEL_MAJOR)).float()

    assert_with_pcc(golden_cm, y_default, pcc=0.99)
    assert torch.equal(y_default, y_explicit)


# ---------------------------------------------------------------------------
# Section K — BEV model shapes × channel_order (primary use case)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,N,C,H,W,r",
    [
        ("Y_path", 1, 1, 1536, 1536, 4),
        ("UV_path", 1, 32, 48, 768, 2),
    ],
)
@pytest.mark.parametrize("input_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("order_enum,order_name", CHANNEL_ORDERS)
@pytest.mark.slow
def test_pixel_unshuffle_bev_shapes(device, name, N, C, H, W, r, input_layout, order_enum, order_name):
    """Full BEV Y/UV shapes, both layouts and orderings."""
    x = _make_input(ttnn.bfloat16, (N, C, H, W), seed=7)
    golden = golden_pixel_unshuffle(x.float(), r, order_name)

    tt_in = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=input_layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_out = ttnn.pixel_unshuffle(tt_in, downscale_factor=r, channel_order=order_enum)
    result = ttnn.to_torch(tt_out).float()

    assert list(result.shape) == [N, C * r * r, H // r, W // r]
    assert_with_pcc(golden, result, pcc=0.99)


# ---------------------------------------------------------------------------
# Section K2 — dedicated BEV pipeline shapes (always run), both orderings × layouts
#   These are the exact block_C shapes the channel-ordering fix targets:
#     Y-path : [1, 1, 1280, 2304] r=4 -> [1, 16, 320, 576]
#     UV-path: [1, 2,  640, 1152] r=2 -> [1,  8, 320, 576]   (C=2 → order matters)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("input_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("order_enum,order_name", CHANNEL_ORDERS)
def test_pixel_unshuffle_bev_y_path(device, input_layout, order_enum, order_name):
    """BEV Y-path: C=1, r=4. Orderings coincide (C=1) but both must produce the golden."""
    N, C, H, W, r = 1, 1, 1280, 2304, 4
    x = _make_input(ttnn.bfloat16, (N, C, H, W), seed=7)
    golden = golden_pixel_unshuffle(x.float(), r, order_name)

    tt_in = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=input_layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_out = ttnn.pixel_unshuffle(tt_in, downscale_factor=r, channel_order=order_enum)
    result = ttnn.to_torch(tt_out).float()

    assert list(result.shape) == [N, C * r * r, H // r, W // r] == [1, 16, 320, 576]
    assert_with_pcc(golden, result, pcc=0.99)


@pytest.mark.parametrize("input_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("order_enum,order_name", CHANNEL_ORDERS)
def test_pixel_unshuffle_bev_uv_path(device, input_layout, order_enum, order_name):
    """BEV UV-path: C=2, r=2 — the ordering-sensitive case that motivated the fix.
    CHANNEL_MAJOR and SPATIAL_MAJOR produce genuinely different channel data here."""
    N, C, H, W, r = 1, 2, 640, 1152, 2
    x = _make_input(ttnn.bfloat16, (N, C, H, W), seed=8)
    golden = golden_pixel_unshuffle(x.float(), r, order_name)

    tt_in = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=input_layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_out = ttnn.pixel_unshuffle(tt_in, downscale_factor=r, channel_order=order_enum)
    result = ttnn.to_torch(tt_out).float()

    assert list(result.shape) == [N, C * r * r, H // r, W // r] == [1, 8, 320, 576]
    assert_with_pcc(golden, result, pcc=0.99)


def test_pixel_unshuffle_bev_uv_orderings_differ(device):
    """Guard: for the BEV UV shape (C=2), the two orderings must NOT be identical —
    otherwise the channel_order flag would be a no-op on the real model."""
    N, C, H, W, r = 1, 2, 640, 1152, 2
    x = _make_input(ttnn.bfloat16, (N, C, H, W), seed=8)
    tt_in = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    y_cm = ttnn.to_torch(ttnn.pixel_unshuffle(tt_in, downscale_factor=r, channel_order=CHANNEL_MAJOR)).float()
    y_sm = ttnn.to_torch(ttnn.pixel_unshuffle(tt_in, downscale_factor=r, channel_order=SPATIAL_MAJOR)).float()
    assert not torch.allclose(y_cm, y_sm), "CHANNEL_MAJOR and SPATIAL_MAJOR must differ for C=2"


# ---------------------------------------------------------------------------
# Section L — program cache hit path × channel_order
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order_enum,order_name", CHANNEL_ORDERS)
def test_pixel_unshuffle_program_cache(device, order_enum, order_name):
    """Run twice to exercise the program-cache-hit validation path."""
    x = _make_input(ttnn.bfloat16, (1, 4, 32, 32), seed=2)
    golden = golden_pixel_unshuffle(x.float(), 2, order_name)
    for _ in range(2):
        tt_in = ttnn.from_torch(
            x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        tt_out = ttnn.pixel_unshuffle(tt_in, downscale_factor=2, channel_order=order_enum)
        result = ttnn.to_torch(tt_out).float()
        assert_with_pcc(golden, result, pcc=0.99)


# ---------------------------------------------------------------------------
# Section M — error-path / validation tests
# ---------------------------------------------------------------------------


def test_pixel_unshuffle_invalid_rank(device):
    """3D input must raise — pixel_unshuffle requires exactly 4D [N,C,H,W]."""
    x = torch.randn(3, 32, 32, dtype=torch.bfloat16)
    tt_in = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    with pytest.raises(RuntimeError):
        ttnn.pixel_unshuffle(tt_in, downscale_factor=2)


def test_pixel_unshuffle_h_not_divisible(device):
    x = torch.randn(1, 1, 33, 32, dtype=torch.bfloat16)
    tt_in = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    with pytest.raises(RuntimeError):
        ttnn.pixel_unshuffle(tt_in, downscale_factor=2)


def test_pixel_unshuffle_w_not_divisible(device):
    x = torch.randn(1, 1, 32, 33, dtype=torch.bfloat16)
    tt_in = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    with pytest.raises(RuntimeError):
        ttnn.pixel_unshuffle(tt_in, downscale_factor=2)


def test_pixel_unshuffle_zero_r(device):
    x = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)
    tt_in = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    with pytest.raises((RuntimeError, ValueError)):
        ttnn.pixel_unshuffle(tt_in, downscale_factor=0)
