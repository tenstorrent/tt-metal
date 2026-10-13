# #59149 headroom round: production callers of the fast untilize with rows wider than eight tiles (dense input, 16-bit
# DEST), which the face-pair passes reach, plus eight-tile and block-float controls; under the device profiler.
import os as _os
import sys as _sys

if not (_os.environ.get("HWLOCK_HELD") or _os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH")):
    _sys.exit("not under hwlock")
import pytest
import torch
import ttnn

from tests.ttnn.unit_tests.operations.fused.test_group_norm import run_sdxl_base_group_norm_test
from tests.ttnn.unit_tests.operations.transformers.test_paged_fused_update_cache import (
    run_test_paged_fused_update_cache_decode,
)
from tests.ttnn.nightly.unit_tests.operations.transformers.test_paged_update_cache import run_test_update_cache_decode


# ttnn.untilize of a height-sharded tensor, 64 cores, 4 tile rows per core: the LLK-bound shape class.
@pytest.mark.parametrize("dtype", ["bf16", "bfp8"])
@pytest.mark.parametrize("wt", [8, 10, 12, 16, 18, 24])
def test_sharded_untilize(device, wt, dtype):
    h, w = 64 * 128, wt * 32
    mem = ttnn.create_sharded_memory_config(
        (1, 1, h, w), core_grid=ttnn.CoreGrid(y=8, x=8), strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR)
    torch.manual_seed(0)
    x = torch.randn(1, 1, h, w).bfloat16()
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16 if dtype == "bf16" else ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT,
                        device=device, memory_config=mem)
    out = ttnn.untilize(t, memory_config=mem, use_multicore=True)
    ttnn.synchronize_device(device)
    if dtype == "bf16":
        assert torch.equal(ttnn.to_torch(out).float(), x.float())


# SDXL UNet group norm at 2560 channels: block-sharded 8x8, 10 tiles per core, row-major in place (Welford: 32-bit DEST).
@pytest.mark.parametrize("use_welford", [False, True], ids=["tile_reduction", "welford"])
def test_sdxl_group_norm_2560(device, use_welford):
    run_sdxl_base_group_norm_test(device, 1, 2560, 32, 32, use_welford, ttnn.ROW_MAJOR_LAYOUT, True)


# Gemma 4 decode rope on the full head dim (256 sliding, 512 global): the cos/sin untilize at 8 and 16 tiles.
@pytest.mark.parametrize("head_dim", [256, 512])
def test_rotary_decode(device, head_dim):
    torch.manual_seed(0)
    x = ttnn.from_torch(torch.randn(1, 8, 32, head_dim).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=device)
    cos = ttnn.from_torch(torch.randn(1, 1, 2048, head_dim).bfloat16(), dtype=ttnn.bfloat16,
                          layout=ttnn.TILE_LAYOUT, device=device)
    sin = ttnn.from_torch(torch.randn(1, 1, 2048, head_dim).bfloat16(), dtype=ttnn.bfloat16,
                          layout=ttnn.TILE_LAYOUT, device=device)
    ttnn.experimental.rotary_embedding(x, cos, sin, 129)
    ttnn.synchronize_device(device)


# Gemma 4 decode KV writes at head dim 256 and 512 (8 and 16 tiles).
@pytest.mark.parametrize("head_dim", [256, 512])
def test_paged_fused_update_cache(device, head_dim):
    run_test_paged_fused_update_cache_decode(True, 127, 64, head_dim, 2048, 8, 1, ttnn.bfloat16, ttnn.bfloat16,
                                             device, 0.9995, row_major=False)


@pytest.mark.parametrize("head_dim", [256, 512])
def test_paged_update_cache(device, head_dim):
    run_test_update_cache_decode(127, True, head_dim, 2048, 4, 8, ttnn.bfloat16, ttnn.bfloat16, device)


# ND-sharded identical-spec untilize (DeepSeek MLA cache width 576 is 18 tiles; 256 is 8).
@pytest.mark.parametrize("width", [256, 576])
def test_untilize_nd_identical(device, width):
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))])
    mem = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1,
                            nd_shard_spec=ttnn.NdShardSpec(shard_shape=[2, 32, width], grid=grid,
                                                           orientation=ttnn.ShardOrientation.ROW_MAJOR))
    torch.manual_seed(0)
    x = ttnn.from_torch(torch.randn(1, 16, 64, width).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=device, memory_config=mem)
    out = ttnn.untilize(x, memory_config=mem)
    ttnn.synchronize_device(device)
    assert torch.equal(ttnn.to_torch(out).float(), ttnn.to_torch(x).float())


# Hunyuan Image 3 VAE conv3d with a 16-bit DEST: C_out_block 256 and 512 (8 and 16 tiles) in the row-major output.
@pytest.mark.parametrize("c_out", [256, 512])
def test_conv3d_16bit(device, c_out):
    c_in, c_in_block, d, h, w = 64, 32, 4, 16, 16
    torch.manual_seed(0)
    x = ttnn.from_torch(torch.randn(1, d, h, w, c_in).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                        device=device)
    cfg = ttnn.Conv3dConfig(weights_dtype=ttnn.bfloat16, output_layout=ttnn.ROW_MAJOR_LAYOUT, T_out_block=1,
                            W_out_block=1, H_out_block=1, C_out_block=c_out, C_in_block=c_in_block,
                            compute_with_storage_grid_size=device.compute_with_storage_grid_size())
    wt = ttnn.experimental.prepare_conv3d_weights(
        weight_tensor=ttnn.from_torch(torch.randn(c_out, c_in, 3, 3, 3).bfloat16(), dtype=ttnn.bfloat16),
        groups=1, C_in_block=c_in_block, alignment=32, device=device)
    bias = ttnn.from_torch(torch.randn(1, c_out).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device)
    cc = ttnn.init_device_compute_kernel_config(device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2,
                                                fp32_dest_acc_en=False, packer_l1_acc=False)
    ttnn.experimental.conv3d(input_tensor=x, weight_tensor=wt, device=device, bias_tensor=bias, dtype=ttnn.bfloat16,
                             output_channels=c_out, kernel_size=(3, 3, 3), stride=(1, 1, 1), groups=1,
                             padding=(1, 1, 1), dilation=(1, 1, 1), padding_mode="zeros", config=cfg,
                             compute_kernel_config=cc)
    ttnn.synchronize_device(device)
