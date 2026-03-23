# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Universal input support tests for Conv-family OPs (#31714).
Tests: conv2d, pool2d, upsample (conv context).

Note: ``ttnn::halo`` and ``SlidingWindowConfig`` are C++ internals used by ``ttnn.conv2d`` (L1 / sliding-window
path); they are not exposed on the Python ``ttnn`` module (no ``ttnn.halo`` / ``ttnn.SlidingWindowConfig``).

Note: Conv ops have unique requirements - activation must typically be sharded.
This tests whether interleaved inputs are also accepted (or auto-resharded).

Conv2d requirements (see trial-and-error / #31714):
- Device must be opened with non-zero L1_SMALL (module uses l1_small_size) — DRAM path still runs
  per-slice conv2d_L1 which allocates L1_SMALL.
- Weights: host ROW_MAJOR (OIHW), or device tensors produced by ttnn.prepare_conv_weights. Passing
  raw TILE weights on device without the prepared [1,1,KhKwCi,Co] shape fails host validation on reprocess.
- Bias: host ROW_MAJOR ``[1,1,1,Co]``, or tensors produced by ``ttnn.prepare_conv_bias`` (host bias in, device TILE out).
"""

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

# Conv DRAM path runs conv2d_L1 per slice; L1 path uses L1_SMALL for config/scratch.
pytestmark = pytest.mark.use_module_device({"l1_small_size": 16384})

TILE_H = 32
TILE_W = 32


def make_memory_config(strategy, shape):
    H, W = shape[-2], shape[-1]
    if strategy == "dram":
        return ttnn.DRAM_MEMORY_CONFIG
    elif strategy == "l1":
        return ttnn.L1_MEMORY_CONFIG
    elif strategy == "height_sharded":
        num_cores = max(1, H // TILE_H)
        num_cores = min(num_cores, 8)
        shard_h = H // num_cores
        return ttnn.create_sharded_memory_config(
            [shard_h, W],
            core_grid=ttnn.CoreGrid(y=num_cores, x=1),
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
    elif strategy == "width_sharded":
        num_cores = max(1, W // TILE_W)
        num_cores = min(num_cores, 8)
        shard_w = W // num_cores
        return ttnn.create_sharded_memory_config(
            [H, shard_w],
            core_grid=ttnn.CoreGrid(y=1, x=num_cores),
            strategy=ttnn.ShardStrategy.WIDTH,
            use_height_and_width_as_shard_shape=True,
        )
    elif strategy == "block_sharded":
        num_cores_h = max(1, H // TILE_H)
        num_cores_h = min(num_cores_h, 4)
        num_cores_w = max(1, W // TILE_W)
        num_cores_w = min(num_cores_w, 4)
        shard_h = H // num_cores_h
        shard_w = W // num_cores_w
        return ttnn.create_sharded_memory_config(
            [shard_h, shard_w],
            core_grid=ttnn.CoreGrid(y=num_cores_h, x=num_cores_w),
            strategy=ttnn.ShardStrategy.BLOCK,
            use_height_and_width_as_shard_shape=True,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def tt_flat_nhwc_family_output_to_nchw(tt_output, batch_size, out_h, out_w, channels):
    """Convert ttnn conv-family output to torch NCHW.

    ttnn.conv2d and ttnn.max_pool2d return activations as [1, 1, batch*out_h*out_w, C]
    with spatial data in NHWC order (rows vary fastest over H, then W, then batch).
    PyTorch golden tensors are NCHW [N, C, H, W]. Do not use torch.reshape alone to
    match the flat TT shape — that scrambles channel vs spatial ordering.
    """
    tt_result = ttnn.to_torch(tt_output)
    return tt_result.reshape(batch_size, out_h, out_w, channels).permute(0, 3, 1, 2)


def tt_conv2d_output_to_nchw(tt_output, batch_size, out_h, out_w, out_channels):
    """See :func:`tt_flat_nhwc_family_output_to_nchw`."""
    return tt_flat_nhwc_family_output_to_nchw(tt_output, batch_size, out_h, out_w, out_channels)


def tt_max_pool2d_output_to_nchw(tt_output, batch_size, out_h, out_w, channels):
    """Same output layout as conv2d; see :func:`tt_flat_nhwc_family_output_to_nchw`."""
    return tt_flat_nhwc_family_output_to_nchw(tt_output, batch_size, out_h, out_w, channels)


ALL_MEMORY_STRATEGIES = ["dram", "l1", "height_sharded", "width_sharded", "block_sharded"]


# =============================================================================
# Conv2d - input activation memory config
# =============================================================================


@pytest.mark.parametrize("input_memory_strategy", ALL_MEMORY_STRATEGIES)
def test_conv2d_input_memory(device, input_memory_strategy):
    """Test conv2d with activation in different memory configs.

    Weights: host ROW_MAJOR (conv2d prepares internally). Do not pass raw TILE on device without prepare_conv_weights.
    """
    batch_size = 1
    in_channels = 32
    out_channels = 32
    input_h = 32
    input_w = 32
    kernel_size = 3
    pad = 1

    torch_input = torch.randn(batch_size, in_channels, input_h, input_w, dtype=torch.bfloat16)
    torch_weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, dtype=torch.bfloat16)

    torch_input_nhwc = torch_input.permute(0, 2, 3, 1)
    flat_shape = [1, 1, batch_size * input_h * input_w, in_channels]
    torch_input_flat = torch_input_nhwc.reshape(flat_shape)

    input_mem = make_memory_config(input_memory_strategy, flat_shape)
    tt_input = ttnn.from_torch(
        torch_input_flat, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=input_mem
    )
    tt_weight = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    tt_output = ttnn.conv2d(
        input_tensor=tt_input,
        weight_tensor=tt_weight,
        in_channels=in_channels,
        out_channels=out_channels,
        device=device,
        bias_tensor=None,
        kernel_size=(kernel_size, kernel_size),
        stride=(1, 1),
        padding=(pad, pad),
        batch_size=batch_size,
        input_height=input_h,
        input_width=input_w,
    )

    torch_output = torch.nn.functional.conv2d(torch_input.float(), torch_weight.float(), padding=pad).bfloat16()
    out_h = (input_h + 2 * pad - kernel_size) // 1 + 1
    out_w = (input_w + 2 * pad - kernel_size) // 1 + 1
    tt_result = tt_conv2d_output_to_nchw(tt_output, batch_size, out_h, out_w, out_channels)

    assert_with_pcc(torch_output, tt_result, 0.97)


# =============================================================================
# Conv2d - weights: host ROW_MAJOR vs prepare_conv_weights (device TILE)
# =============================================================================
# Note: Varying arbitrary DRAM/L1/sharded memory on raw [O,I,kH,kW] TILE weights is not supported —
# conv2d expects host ROW_MAJOR or pre-prepared device weights (see prepare_conv_weights).
# Same idea for bias: host ROW_MAJOR [1,1,1,Co] or prepare_conv_bias (see test_conv2d_bias_preparation_paths).


@pytest.mark.parametrize("weight_setup", ["host_row_major", "prepare_conv_weights"])
def test_conv2d_weight_preparation_paths(device, weight_setup):
    """Valid weight paths: host ROW_MAJOR, or ttnn.prepare_conv_weights (matches activation memory config)."""
    batch_size = 1
    in_channels = 32
    out_channels = 32
    input_h = 32
    input_w = 32
    kernel_size = 3
    pad = 1

    torch_input = torch.randn(batch_size, in_channels, input_h, input_w, dtype=torch.bfloat16)
    torch_weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, dtype=torch.bfloat16)

    torch_input_nhwc = torch_input.permute(0, 2, 3, 1)
    flat_shape = [1, 1, batch_size * input_h * input_w, in_channels]
    torch_input_flat = torch_input_nhwc.reshape(flat_shape)

    input_mem = ttnn.DRAM_MEMORY_CONFIG
    tt_input = ttnn.from_torch(
        torch_input_flat, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=input_mem
    )

    if weight_setup == "host_row_major":
        tt_weight = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    else:
        weight_host = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        conv_config = ttnn.Conv2dConfig(weights_dtype=ttnn.bfloat16)
        tt_weight = ttnn.prepare_conv_weights(
            weight_tensor=weight_host,
            weights_format="OIHW",
            input_memory_config=input_mem,
            input_layout=ttnn.ROW_MAJOR_LAYOUT,
            in_channels=in_channels,
            out_channels=out_channels,
            batch_size=batch_size,
            input_height=input_h,
            input_width=input_w,
            kernel_size=(kernel_size, kernel_size),
            stride=(1, 1),
            padding=(pad, pad),
            dilation=(1, 1),
            has_bias=False,
            groups=1,
            device=device,
            input_dtype=ttnn.bfloat16,
            conv_config=conv_config,
        )

    tt_output = ttnn.conv2d(
        input_tensor=tt_input,
        weight_tensor=tt_weight,
        in_channels=in_channels,
        out_channels=out_channels,
        device=device,
        bias_tensor=None,
        kernel_size=(kernel_size, kernel_size),
        stride=(1, 1),
        padding=(pad, pad),
        batch_size=batch_size,
        input_height=input_h,
        input_width=input_w,
    )

    torch_output = torch.nn.functional.conv2d(torch_input.float(), torch_weight.float(), padding=pad).bfloat16()
    out_h = (input_h + 2 * pad - kernel_size) // 1 + 1
    out_w = (input_w + 2 * pad - kernel_size) // 1 + 1
    tt_result = tt_conv2d_output_to_nchw(tt_output, batch_size, out_h, out_w, out_channels)

    assert_with_pcc(torch_output, tt_result, 0.97)


@pytest.mark.parametrize("bias_setup", ["host_row_major", "prepare_conv_bias"])
def test_conv2d_bias_preparation_paths(device, bias_setup):
    """Valid bias paths: host ROW_MAJOR [1,1,1,Co], or ttnn.prepare_conv_bias (matches activation memory config)."""
    batch_size = 1
    in_channels = 32
    out_channels = 32
    input_h = 32
    input_w = 32
    kernel_size = 3
    pad = 1

    torch_input = torch.randn(batch_size, in_channels, input_h, input_w, dtype=torch.bfloat16)
    torch_weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, dtype=torch.bfloat16)
    torch_bias = torch.randn(1, 1, 1, out_channels, dtype=torch.bfloat16)

    torch_input_nhwc = torch_input.permute(0, 2, 3, 1)
    flat_shape = [1, 1, batch_size * input_h * input_w, in_channels]
    torch_input_flat = torch_input_nhwc.reshape(flat_shape)

    input_mem = ttnn.DRAM_MEMORY_CONFIG
    tt_input = ttnn.from_torch(
        torch_input_flat, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=input_mem
    )
    tt_weight = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    if bias_setup == "host_row_major":
        tt_bias = ttnn.from_torch(torch_bias, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    else:
        bias_host = ttnn.from_torch(torch_bias, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        conv_config = ttnn.Conv2dConfig(weights_dtype=ttnn.bfloat16)
        tt_bias = ttnn.prepare_conv_bias(
            bias_tensor=bias_host,
            input_memory_config=input_mem,
            input_layout=ttnn.ROW_MAJOR_LAYOUT,
            in_channels=in_channels,
            out_channels=out_channels,
            batch_size=batch_size,
            input_height=input_h,
            input_width=input_w,
            kernel_size=(kernel_size, kernel_size),
            stride=(1, 1),
            padding=(pad, pad),
            dilation=(1, 1),
            groups=1,
            device=device,
            input_dtype=ttnn.bfloat16,
            conv_config=conv_config,
        )

    tt_output = ttnn.conv2d(
        input_tensor=tt_input,
        weight_tensor=tt_weight,
        in_channels=in_channels,
        out_channels=out_channels,
        device=device,
        bias_tensor=tt_bias,
        kernel_size=(kernel_size, kernel_size),
        stride=(1, 1),
        padding=(pad, pad),
        batch_size=batch_size,
        input_height=input_h,
        input_width=input_w,
    )

    torch_output = torch.nn.functional.conv2d(
        torch_input.float(),
        torch_weight.float(),
        bias=torch_bias.reshape(-1).float(),
        padding=pad,
    ).bfloat16()
    out_h = (input_h + 2 * pad - kernel_size) // 1 + 1
    out_w = (input_w + 2 * pad - kernel_size) // 1 + 1
    tt_result = tt_conv2d_output_to_nchw(tt_output, batch_size, out_h, out_w, out_channels)

    assert_with_pcc(torch_output, tt_result, 0.97)


# =============================================================================
# Pool2d (max_pool2d) - input memory config
# =============================================================================


@pytest.mark.parametrize("input_memory_strategy", ALL_MEMORY_STRATEGIES)
def test_max_pool2d_input_memory(device, input_memory_strategy):
    """Test max_pool2d with input in different memory configs.
    Pool2d currently requires sharded ROW_MAJOR input."""
    batch_size = 1
    channels = 32
    input_h = 32
    input_w = 32
    kernel_size = 2
    stride = 2

    torch_input = torch.randn(batch_size, channels, input_h, input_w, dtype=torch.bfloat16)

    torch_input_nhwc = torch_input.permute(0, 2, 3, 1)
    flat_shape = [1, 1, batch_size * input_h * input_w, channels]
    torch_input_flat = torch_input_nhwc.reshape(flat_shape)

    input_mem = make_memory_config(input_memory_strategy, flat_shape)
    tt_input = ttnn.from_torch(
        torch_input_flat, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=input_mem
    )

    tt_output = ttnn.max_pool2d(
        input_tensor=tt_input,
        batch_size=batch_size,
        input_h=input_h,
        input_w=input_w,
        channels=channels,
        kernel_size=(kernel_size, kernel_size),
        stride=(stride, stride),
        padding=(0, 0),
        dilation=(1, 1),
        ceil_mode=False,
    )

    torch_output = torch.nn.functional.max_pool2d(torch_input.float(), kernel_size, stride).bfloat16()
    _, _, out_h, out_w = torch_output.shape
    tt_result = tt_max_pool2d_output_to_nchw(tt_output, batch_size, out_h, out_w, channels)

    assert_with_pcc(torch_output, tt_result, 0.99)


@pytest.mark.parametrize("input_memory_strategy", ALL_MEMORY_STRATEGIES)
def test_max_pool2d_tile_input(device, input_memory_strategy):
    """Test max_pool2d with TILE layout input (currently requires ROW_MAJOR)."""
    batch_size = 1
    channels = 32
    input_h = 32
    input_w = 32
    kernel_size = 2
    stride = 2

    torch_input = torch.randn(batch_size, channels, input_h, input_w, dtype=torch.bfloat16)

    torch_input_nhwc = torch_input.permute(0, 2, 3, 1)
    flat_shape = [1, 1, batch_size * input_h * input_w, channels]
    torch_input_flat = torch_input_nhwc.reshape(flat_shape)

    input_mem = make_memory_config(input_memory_strategy, flat_shape)
    tt_input = ttnn.from_torch(
        torch_input_flat, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=input_mem
    )

    tt_output = ttnn.max_pool2d(
        input_tensor=tt_input,
        batch_size=batch_size,
        input_h=input_h,
        input_w=input_w,
        channels=channels,
        kernel_size=(kernel_size, kernel_size),
        stride=(stride, stride),
        padding=(0, 0),
        dilation=(1, 1),
        ceil_mode=False,
    )

    torch_output = torch.nn.functional.max_pool2d(torch_input.float(), kernel_size, stride).bfloat16()
    _, _, out_h, out_w = torch_output.shape
    tt_result = tt_max_pool2d_output_to_nchw(tt_output, batch_size, out_h, out_w, channels)

    assert_with_pcc(torch_output, tt_result, 0.99)


# =============================================================================
# Halo - input memory config
# =============================================================================


@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
def test_halo_input_memory(device, layout):
    pytest.skip(
        "``ttnn::halo`` and ``SlidingWindowConfig`` are C++ internals - no python APIs.  \
    testing halo = testing conv2d with sharded activations + padding / layout so the internal halo path is already covered in tests."
    )
    batch_size = 1
    channels = 32
    input_h = 8
    input_w = 8

    torch_input = torch.randn(batch_size, channels, input_h, input_w, dtype=torch.bfloat16)

    torch_input_nhwc = torch_input.permute(0, 2, 3, 1)
    flat_shape = [1, 1, batch_size * input_h * input_w, channels]
    torch_input_flat = torch_input_nhwc.reshape(flat_shape)

    input_mem = make_memory_config(input_memory_strategy, flat_shape)
    tt_input = ttnn.from_torch(
        torch_input_flat, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=input_mem
    )

    mem = tt_input.memory_config()
    shard_spec = mem.shard_spec
    assert shard_spec is not None

    cfg = ttnn.SlidingWindowConfig()
    cfg.batch_size = batch_size
    cfg.channels = channels
    cfg.input_hw = (input_h, input_w)
    cfg.window_hw = (3, 3)
    cfg.stride_hw = (1, 1)
    cfg.padding = (1, 1, 1, 1)
    cfg.dilation_hw = (1, 1)
    cfg.num_cores_nhw = num_cores_nhw_for_shard_mem_config(mem)
    cfg.core_range_set = shard_spec.grid
    cfg.snap_to_tile = True

    transpose_mcast = shard_spec.orientation == ttnn.ShardOrientation.COL_MAJOR
    tt_output = ttnn.halo(tt_input, cfg, 0, False, transpose_mcast, True, False)

    tt_result = ttnn.to_torch(tt_output)
    assert tt_result.numel() > 0
