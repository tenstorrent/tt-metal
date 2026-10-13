# #59149 headroom round, conv set: conv2d with a row-major output (the helper untilizes partials over 8 tiles per core)
# and the halo of a tiled height-sharded conv2d, with ids free of spaces; device profiler.
import os as _os
import sys as _sys

if not (_os.environ.get("HWLOCK_HELD") or _os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH")):
    _sys.exit("not under hwlock")
import pytest
import torch
import ttnn


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], ids=["l1s"], indirect=True)
@pytest.mark.parametrize("layout", ["rm", "tile"])
@pytest.mark.parametrize("out_channels", [288, 512])
def test_conv2d_out_layout(device, out_channels, layout):
    n, h, w, c = 16, 28, 28, 64
    torch.manual_seed(0)
    x = ttnn.from_torch(torch.randn(1, 1, n * h * w, c).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
    wt = ttnn.from_torch(torch.randn(out_channels, c, 3, 3).bfloat16(), dtype=ttnn.bfloat16)
    cfg = ttnn.Conv2dConfig(weights_dtype=ttnn.bfloat16, shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                            output_layout=ttnn.ROW_MAJOR_LAYOUT if layout == "rm" else ttnn.TILE_LAYOUT)
    cc = ttnn.init_device_compute_kernel_config(device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                fp32_dest_acc_en=False, packer_l1_acc=False)
    ttnn.conv2d(input_tensor=x, weight_tensor=wt, in_channels=c, out_channels=out_channels, device=device,
                kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), batch_size=n, input_height=h, input_width=w,
                conv_config=cfg, compute_config=cc, dtype=ttnn.bfloat16)
    ttnn.synchronize_device(device)


# The halo of a height-sharded 3x3 conv2d on a tiled input untilizes C / 32 tiles per row: 8 at 256 channels (control).
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], ids=["l1s"], indirect=True)
@pytest.mark.parametrize("dtype", ["bf16", "bfp8_b"])
@pytest.mark.parametrize("channels", [256])
def test_conv_halo_untilize(device, channels, dtype):
    n, h, w = 16, 14, 14
    torch.manual_seed(0)
    x = ttnn.from_torch(torch.randn(1, 1, n * h * w, channels).bfloat16(),
                        dtype=ttnn.bfloat16 if dtype == "bf16" else ttnn.bfloat8_b,
                        layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
    wt = ttnn.from_torch(torch.randn(channels, channels, 3, 3).bfloat16(), dtype=ttnn.bfloat16)
    ttnn.conv2d(input_tensor=x, weight_tensor=wt, in_channels=channels, out_channels=channels, device=device,
                kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), batch_size=n, input_height=h, input_width=w,
                conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.bfloat16,
                                              shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED))
    ttnn.synchronize_device(device)


