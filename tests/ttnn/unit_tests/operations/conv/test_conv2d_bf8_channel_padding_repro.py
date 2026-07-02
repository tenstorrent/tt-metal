# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Minimal repro for the conv2d BF8 channel-padding corruption.

Root cause
----------
For a conv with channel-unaligned input (e.g. in_channels=3, padded to 8 for
alignment) and a *block-float* (bfloat8_b) output, the activation is tilized to
bf8, which shares one exponent across a 16-element block. When the input arrives
already on device (DRAM/L1 interleaved), the conv reshard takes the on-device
path: `create_device_tensor` (UNINITIALIZED L1) + `to_memory_config`
(interleaved_to_sharded). That reshard copies only the 3 real channels and the
per-tap padding lanes keep whatever stale memory was there. If that garbage is
large-magnitude, it dominates the shared bf8 exponent and crushes the real
channels -> wrong result.

The host-input path is safe because it zero-pads via `ttnn::pad(..., 0)`.

Why the priming is required
---------------------------
On clean hardware the reshard's padding region often happens to be ~0, so the
bug is *latent* and the test passes. To make it deterministic we first dirty
DRAM and L1 with large-magnitude values and free them, so the reshard's
uninitialized padding picks up that garbage.

Expected behavior
-----------------
- Without a fix (bf8 ACT_TILIZED, on-device reshard not zeroing padding): FAILS
  (PCC collapses, often ~0.1-0.6).
- With a correct fix (zero the reshard padding / clamp the stale read): PASSES.
"""

import math

import torch
import pytest

import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout


def _prime_memory_with_garbage(device):
    """Dirty DRAM and L1 with large-magnitude values, then free them, so the
    conv's on-device reshard reuses that memory for its (uninitialized) channel
    padding. Makes the otherwise-latent corruption deterministic."""
    garbage = []
    for mem_cfg in (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG):
        for _ in range(8):
            t = torch.full((1, 1, 2048, 256), 1.0e4, dtype=torch.bfloat16)
            garbage.append(
                ttnn.from_torch(t, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_cfg)
            )
    for t in garbage:
        ttnn.deallocate(t)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("in_channels", [3])  # unaligned -> padded to 8 -> exposed
@pytest.mark.parametrize(
    "output_dtype",
    [
        ttnn.bfloat8_b,  # block-float output -> shared exponent -> corrupted by garbage padding
        ttnn.bfloat16,  # control: per-element exponent -> garbage padding can't poison real channels
    ],
)
def test_conv2d_bf8_channel_padding_corruption(device, in_channels, output_dtype):
    batch_size = 1
    out_channels = 32
    H = W = 64
    kernel, stride, padding = 3, 1, 1

    torch.manual_seed(0)
    torch_input_nchw = torch.randn(batch_size, in_channels, H, W, dtype=torch.bfloat16).float()
    torch_weight = torch.randn(out_channels, in_channels, kernel, kernel, dtype=torch.bfloat16).float()
    torch_out_nchw = torch.nn.functional.conv2d(torch_input_nchw, torch_weight, stride=stride, padding=padding)
    ref_nhwc = torch.permute(torch_out_nchw, (0, 2, 3, 1))

    torch_input_nhwc = torch.permute(torch_input_nchw, (0, 2, 3, 1))

    # Dirty memory so the on-device reshard's channel padding holds garbage.
    _prime_memory_with_garbage(device)

    # Input placed ON DEVICE (DRAM) so the conv takes the on-device (no-zero) reshard path.
    tt_input = ttnn.from_torch(torch_input_nhwc, ttnn.bfloat16, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tt_weight = ttnn.from_torch(torch_weight, ttnn.float32)

    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat8_b,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    )
    compute_config = ttnn.init_device_compute_kernel_config(device.arch(), math_fidelity=ttnn.MathFidelity.LoFi)

    [tt_out_dev, [out_h, out_w], [_w, _b]] = ttnn.conv2d(
        input_tensor=tt_input,
        weight_tensor=tt_weight,
        in_channels=in_channels,
        out_channels=out_channels,
        device=device,
        kernel_size=(kernel, kernel),
        stride=(stride, stride),
        padding=(padding, padding),
        batch_size=batch_size,
        input_height=H,
        input_width=W,
        conv_config=conv_config,
        compute_config=compute_config,
        dtype=output_dtype,
        return_output_dim=True,
        return_weights_and_bias=True,
    )
    ttnn.synchronize_device(device)

    out = ttnn.to_torch(ttnn.from_device(tt_out_dev))
    out = out.reshape(batch_size, out_h, out_w, out.shape[-1])[:, :, :, :out_channels]

    passing, pcc_msg = check_with_pcc_without_tensor_printout(ref_nhwc, out, pcc=0.99)
    assert passing, f"conv2d ic={in_channels} {output_dtype} corrupted by uninitialized channel padding: PCC={pcc_msg}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("in_channels", [3])  # unaligned -> padded to 8 -> exposed
@pytest.mark.parametrize(
    "output_dtype",
    [
        ttnn.bfloat8_b,  # block-float output -> shared exponent -> corrupted by garbage padding
        ttnn.bfloat16,  # control: per-element exponent -> garbage padding can't poison real channels
    ],
)
def test_conv2d_bf8_channel_padding_corruption_presharded_no_i2s(device, in_channels, output_dtype):
    """Corruption from a pre-sharded input whose channel padding holds garbage, where
    that tensor did NOT come from interleaved_to_sharded.

    Motivation: PR #48493 fixes `interleaved_to_sharded` so that it zeroes the channel
    padding when resharding on device. But that only sanitizes tensors that pass
    *through* I2S. If a HEIGHT_SHARDED tensor with garbage in its physical channel
    padding reaches conv by some other route (a prior sharded op, an uninitialized
    allocation, a fold/upstream shard that left padding dirty), conv's bf8 activation
    tilize is still corrupted -- so fixing I2S alone does not fully resolve the bug.

    Here we "spawn" that tensor directly with `ttnn.from_torch` into an L1
    HEIGHT_SHARDED config: a physically 8-wide shard whose first 3 lanes hold the real
    channels and whose 5 pad lanes hold large-magnitude garbage. Conv is called with
    in_channels=3, so `prepare_conv2d_weights` zeroes the weight rows for the pad
    lanes; bf16 is therefore correct (garbage x 0), but bf8 shares one exponent across
    the 16-element block, so the garbage pad crushes the real channels -> wrong result.

    The op chain is Halo -> Conv2d with NO InterleavedToSharded at all (verified via
    --profile): the sharded tensor is produced by a host->device transfer, not I2S,
    and conv consumes it directly (reshard_if_not_optimal=False, no override needed).
    """
    batch_size = 1
    out_channels = 32
    H = W = 64
    kernel, stride, padding = 3, 1, 1

    torch.manual_seed(0)
    torch_input_nchw = torch.randn(batch_size, in_channels, H, W, dtype=torch.bfloat16).float()
    torch_weight = torch.randn(out_channels, in_channels, kernel, kernel, dtype=torch.bfloat16).float()
    torch_out_nchw = torch.nn.functional.conv2d(torch_input_nchw, torch_weight, stride=stride, padding=padding)
    ref_nhwc = torch.permute(torch_out_nchw, (0, 2, 3, 1))

    # NHWC, flattened to [1, 1, N*H*W, C] row-major -- the shape conv shards on.
    torch_input_nhwc = torch.permute(torch_input_nchw, (0, 2, 3, 1)).reshape(1, 1, batch_size * H * W, in_channels)

    # --- Build the HEIGHT_SHARDED L1 config (shard width = padded channel count) ---
    padded_channels = ((in_channels + 7) // 8) * 8  # 3 -> 8 (row-major L1 alignment)
    nhw = batch_size * H * W
    num_cores = 8
    grid = ttnn.num_cores_to_corerangeset(num_cores, device.compute_with_storage_grid_size(), row_wise=True)
    shard_height = ((math.ceil(nhw / num_cores) + 31) // 32) * 32

    input_shard_mem_config = ttnn.create_sharded_memory_config(
        shape=(1, 1, shard_height, padded_channels),
        core_grid=grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    # Spawn the pre-sharded tensor directly via from_torch (NO interleaved_to_sharded):
    # a width-`padded_channels` buffer with the real channels in lanes [0:in_channels]
    # and garbage in the pad lanes [in_channels:padded_channels]. This is the stale
    # padding a fixed-I2S would never have touched, because the tensor never went
    # through I2S.
    torch_input_padded = torch.zeros(1, 1, nhw, padded_channels, dtype=torch.float32)
    torch_input_padded[..., :in_channels] = torch_input_nhwc
    torch_input_padded[..., in_channels:] = 1.0e4  # garbage in the channel padding lanes
    tt_input = ttnn.from_torch(
        torch_input_padded,
        ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=input_shard_mem_config,
    )
    tt_weight = ttnn.from_torch(torch_weight, ttnn.float32)

    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat8_b,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        # No override_sharding_config and reshard_if_not_optimal=False: conv accepts
        # the already-sharded input as-is and does not add its own reshard (no I2S).
        reshard_if_not_optimal=False,
    )
    compute_config = ttnn.init_device_compute_kernel_config(device.arch(), math_fidelity=ttnn.MathFidelity.LoFi)

    [tt_out_dev, [out_h, out_w], [_w, _b]] = ttnn.conv2d(
        input_tensor=tt_input,
        weight_tensor=tt_weight,
        in_channels=in_channels,
        out_channels=out_channels,
        device=device,
        kernel_size=(kernel, kernel),
        stride=(stride, stride),
        padding=(padding, padding),
        batch_size=batch_size,
        input_height=H,
        input_width=W,
        conv_config=conv_config,
        compute_config=compute_config,
        dtype=output_dtype,
        return_output_dim=True,
        return_weights_and_bias=True,
    )
    ttnn.synchronize_device(device)

    out = ttnn.to_torch(ttnn.from_device(tt_out_dev))
    out = out.reshape(batch_size, out_h, out_w, out.shape[-1])[:, :, :, :out_channels]

    passing, pcc_msg = check_with_pcc_without_tensor_printout(ref_nhwc, out, pcc=0.99)
    assert passing, (
        f"conv2d ic={in_channels} {output_dtype} (pre-sharded, no-I2S input) corrupted by "
        f"garbage channel padding: PCC={pcc_msg}"
    )
