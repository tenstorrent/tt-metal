import ttnn

device = ttnn.open_device(device_id=0)
import math, torch, ttnn
from eval.golden_tests.genify_matmul.helpers import create_ttnn_input_tensor
from ttnn.operations.genify_matmul import genify_matmul
from ttnn.operations.genify_matmul import genify_matmul_program_descriptor as pd

real = pd.create_block_sharded_program_descriptor


def spy(*a, **kw):
    d = real(*a, **kw)
    for k in d.kernels:
        print("KERNEL", k.kernel_source.split("/")[-1], "CT", list(k.compile_time_args), "DEF", list(k.defines))
    for c in d.cbs:
        print("CB", c.total_size, [(f.buffer_index, f.page_size) for f in c.format_descriptors])
    print("SEMS", len(d.semaphores))
    return d


pd.create_block_sharded_program_descriptor = spy
m, k, n = 1, 2048, 256
A = torch.randn(1, 1, m, k)
B = torch.randn(k, n) / math.sqrt(k)
grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))])
mc = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    ttnn.BufferType.L1,
    ttnn.ShardSpec(grid, [32, 256], ttnn.ShardOrientation.ROW_MAJOR),
)
omc = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    ttnn.BufferType.L1,
    ttnn.ShardSpec(grid, [32, 32], ttnn.ShardOrientation.ROW_MAJOR),
)
ta = create_ttnn_input_tensor(A, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=mc)
tb = create_ttnn_input_tensor(B, device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
bias = create_ttnn_input_tensor(torch.randn(1, n), device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, math_approx_mode=False)
out = genify_matmul(
    ta, tb, bias=bias, activation="relu", compute_kernel_config=cfg, memory_config=omc, dtype=ttnn.bfloat8_b
)
ttnn.close_device(device)
