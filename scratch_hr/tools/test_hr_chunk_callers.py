# Shadow (not checked in), #58736 second review: helper callers at production widths that no checked-in test reaches.
import pytest
import torch
import ttnn

from tests.ttnn.unit_tests.operations.transformers.test_paged_fused_update_cache import (
    run_test_paged_fused_update_cache_decode,
)
from tests.ttnn.nightly.unit_tests.operations.transformers.test_paged_update_cache import run_test_update_cache_decode


# DeepSeek V3 decode MLA (mla1d.py): wq_b's output, width-sharded [32, 192] on 8x2 cores, untilized to height-sharded
# rows of 3072 on 4x8 cores; the multi-core untilize runs the helper at 6 tiles.
def test_wqb_untilize(device):
    x = ttnn.from_torch(torch.randn(1, 1, 32, 3072).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=device,
                        memory_config=ttnn.create_sharded_memory_config(
                            (1, 1, 32, 3072), core_grid=ttnn.CoreGrid(y=8, x=2),
                            strategy=ttnn.ShardStrategy.WIDTH, orientation=ttnn.ShardOrientation.ROW_MAJOR))
    out_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                                ttnn.ShardSpec(ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0),
                                                                                 ttnn.CoreCoord(3, 7))}),
                                               (1, 3072), ttnn.ShardOrientation.ROW_MAJOR))
    out = ttnn.untilize(x, memory_config=out_mem)
    ttnn.synchronize_device(device)
    assert torch.equal(ttnn.to_torch(out).float(), ttnn.to_torch(x).float())


# Gemma 4 decode rope on the full head dim (256 sliding, 512 global): the cos/sin untilize runs the helper at 8 and 16.
@pytest.mark.parametrize("head_dim", [256, 512])
def test_rotary_decode_wide(device, head_dim):
    x = ttnn.from_torch(torch.randn(1, 8, 32, head_dim).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=device)
    cos = ttnn.from_torch(torch.randn(1, 1, 2048, head_dim).bfloat16(), dtype=ttnn.bfloat16,
                          layout=ttnn.TILE_LAYOUT, device=device)
    sin = ttnn.from_torch(torch.randn(1, 1, 2048, head_dim).bfloat16(), dtype=ttnn.bfloat16,
                          layout=ttnn.TILE_LAYOUT, device=device)
    ttnn.experimental.rotary_embedding(x, cos, sin, 129)
    ttnn.synchronize_device(device)


# Hunyuan Image 3 VAE conv3d with a 16-bit DEST: C_out_block 256 and 512 (8 and 16 tiles) in the row-major output.
@pytest.mark.parametrize("c_out", [256, 512])
def test_conv3d_16bit(device, c_out):
    c_in, c_in_block, d, h, w = 64, 32, 4, 16, 16
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


# Gemma 4 decode KV writes at head_dim 256 and 512 (8 and 16 tiles): paged fused (tiled and row-major input) and paged.
@pytest.mark.parametrize("row_major", [False])
@pytest.mark.parametrize("head_dim", [256, 512])
def test_paged_fused_update_cache_wide(device, head_dim, row_major):
    run_test_paged_fused_update_cache_decode(True, 127, 64, head_dim, 2048, 8, 1, ttnn.bfloat16, ttnn.bfloat16,
                                             device, 0.9995, row_major=row_major)


@pytest.mark.parametrize("head_dim", [256, 512])
def test_paged_update_cache_wide(device, head_dim):
    run_test_update_cache_decode(127, True, head_dim, 2048, 4, 8, ttnn.bfloat16, ttnn.bfloat16, device)


# ttnn.update_cache (kv_cache) decode at head_dim 256: no Blackhole model found at 5 tiles or more; ELF coverage only.
def test_kv_update_cache_wide(device):
    cache = ttnn.from_torch(torch.randn(32, 1, 2048, 256).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                            device=device)
    x = ttnn.from_torch(torch.randn(1, 1, 32, 256).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=device)
    ttnn.update_cache(cache, x, 127)
    ttnn.synchronize_device(device)


# ND-sharded input and output with the same spec (untilize's ND-identical factory, the legacy variable-blocks kernel):
# shard rows of 8 and 18 tiles (the DeepSeek MLA cache width 576 is 18 tiles).
@pytest.mark.parametrize("width", [256, 576])
def test_untilize_nd_identical(device, width):
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))])
    mem = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1,
                            nd_shard_spec=ttnn.NdShardSpec(shard_shape=[2, 32, width], grid=grid,
                                                           orientation=ttnn.ShardOrientation.ROW_MAJOR))
    x = ttnn.from_torch(torch.randn(1, 16, 64, width).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=device, memory_config=mem)
    out = ttnn.untilize(x, memory_config=mem)
    ttnn.synchronize_device(device)
    assert torch.equal(ttnn.to_torch(out).float(), ttnn.to_torch(x).float())


# Retile (32x32 to 16x32 tiles) of rows wider than one eight-tile chunk: 16 and 32 tiles per block.
@pytest.mark.parametrize("width", [512, 1024])
def test_retile_wide(device, width):
    torch.manual_seed(42)
    x = torch.rand([1, 1, 128, width], dtype=torch.bfloat16)
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, tile=ttnn.Tile([32, 32]))
    y = ttnn.tilize(t, tile=ttnn.Tile([16, 32]))
    assert torch.equal(ttnn.to_torch(y), x)
