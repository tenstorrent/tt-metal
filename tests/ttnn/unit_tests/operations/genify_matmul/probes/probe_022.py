import torch, ttnn, sys

sys.path.insert(0, ".")
from eval.golden_tests.genify_matmul.helpers import activation_memory_config, output_memory_config
from eval.golden_tests.genify_matmul.feature_spec import INPUTS

device = ttnn.open_device(device_id=0)
g = device.compute_with_storage_grid_size()
print("grid", g.x, g.y, "dram", device.dram_grid_size().x)
for a_shape, b_shape in INPUTS:
    mc = activation_memory_config("block_sharded", list(a_shape), ttnn.bfloat16, device, {})
    out_shape = list(a_shape[:-1]) + [b_shape[-1]]
    omc = output_memory_config("block_sharded", mc, out_shape, ttnn.bfloat16, device, {})
    print(a_shape, b_shape, mc.shard_spec, "|", omc.shard_spec.shape)
# ragged-M shard page layout: res5-like shard [224, 128] on (8,10) -> use smaller [64, 32] for M=96 on 2 cols?
from eval.sharding import shard_config

M, K = 96, 64
mc = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    ttnn.BufferType.L1,
    ttnn.ShardSpec(
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))]),
        [64, 32],
        ttnn.ShardOrientation.ROW_MAJOR,
    ),
)
A = torch.arange(M * K).reshape(1, 1, M, K).float()
t = ttnn.from_torch(A, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
print("addr", t.buffer_address(), "pages", t.buffer_num_pages(), "aligned page", t.buffer_aligned_page_size())
back = ttnn.to_torch(t)
print("roundtrip ok", torch.equal(back, A))
ttnn.close_device(device)
