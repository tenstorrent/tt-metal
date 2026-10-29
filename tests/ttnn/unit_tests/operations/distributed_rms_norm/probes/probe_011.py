import ttnn
from eval.golden_tests.distributed_rms_norm.conftest import _FabricMeshCache
import ttnn.operations.distributed_rms_norm.distributed_rms_norm_program_descriptor as pd

cache = _FabricMeshCache()
mesh = cache.get()
v = ttnn.get_memory_view(mesh, ttnn.BufferType.L1)
print(
    "PROBE before",
    pd.l1_budget(mesh),
    v.largest_contiguous_bytes_free_per_bank,
    v.total_bytes_per_bank,
    v.block_table[:3],
)
mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
t = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 32, 32 * 108 * 4]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, mc)
v = ttnn.get_memory_view(mesh, ttnn.BufferType.L1)
print("PROBE after", pd.l1_budget(mesh), v.largest_contiguous_bytes_free_per_bank, v.block_table)
