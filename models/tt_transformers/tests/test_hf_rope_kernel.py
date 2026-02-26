from functools import partial

import torch

import ttnn
from models.common.utility_functions import torch_random
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt

# Decode phase: input shape [1, batch, num_heads, head_dim]
input_shape = [1, 8, 8, 64]
batch_size = input_shape[1]
num_heads = input_shape[2]
head_dim = input_shape[3]

# cos/sin shape for decode: [1, batch, 1, head_dim]
cos_sin_shape = [1, batch_size, 1, head_dim]

input_dtype = ttnn.bfloat16
input_layout = ttnn.TILE_LAYOUT

device = ttnn.open_device(device_id=0)

# Generate input tensors
torch_input_tensor = gen_func_with_cast_tt(partial(torch_random, low=-100, high=100, dtype=torch.float32), input_dtype)(
    input_shape
)
torch_cos_cache_tensor = gen_func_with_cast_tt(partial(torch_random, low=-1, high=1, dtype=torch.float32), input_dtype)(
    cos_sin_shape
)
torch_sin_cache_tensor = gen_func_with_cast_tt(partial(torch_random, low=-1, high=1, dtype=torch.float32), input_dtype)(
    cos_sin_shape
)

# Create HEIGHT_SHARDED memory config for decode
input_shard_height = num_heads * 32  # TILE_HEIGHT
input_shard_width = head_dim

cos_sin_shard_height = 32  # TILE_HEIGHT
cos_sin_shard_width = head_dim

num_cores = min(batch_size, device.compute_with_storage_grid_size().x * device.compute_with_storage_grid_size().y)
core_grid = ttnn.CoreGrid(x=num_cores, y=1)
shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))})

input_shard_spec = ttnn.ShardSpec(shard_grid, [input_shard_height, input_shard_width], ttnn.ShardOrientation.ROW_MAJOR)
input_sharded_mem_config = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec
)

cos_sin_shard_spec = ttnn.ShardSpec(
    shard_grid, [cos_sin_shard_height, cos_sin_shard_width], ttnn.ShardOrientation.ROW_MAJOR
)
cos_sin_sharded_mem_config = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, cos_sin_shard_spec
)

# Convert to device tensors
input_tensor = ttnn.from_torch(
    torch_input_tensor,
    dtype=input_dtype,
    layout=input_layout,
    device=device,
    memory_config=input_sharded_mem_config,
)
cos_cache_tensor = ttnn.from_torch(
    torch_cos_cache_tensor,
    dtype=input_dtype,
    layout=input_layout,
    device=device,
    memory_config=cos_sin_sharded_mem_config,
)
sin_cache_tensor = ttnn.from_torch(
    torch_sin_cache_tensor,
    dtype=input_dtype,
    layout=input_layout,
    device=device,
    memory_config=cos_sin_sharded_mem_config,
)

# Execute rotary embedding (output uses same sharded config as input for decode)
output_tensor = ttnn.experimental.rotary_embedding_hf(
    input_tensor, cos_cache_tensor, sin_cache_tensor, is_decode=True, memory_config=input_sharded_mem_config
)

output_tensor = ttnn.to_torch(output_tensor)
print(f"Output shape: {output_tensor.shape}")

ttnn.close_device(device)
