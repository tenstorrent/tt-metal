import torch, ttnn, math
from eval.golden_tests.genify_matmul.helpers import create_ttnn_input_tensor
from ttnn.operations.genify_matmul import genify_matmul_program_descriptor as pd

device = ttnn.open_device(device_id=0)
A = torch.randn(1, 32, 512, 128)
B = torch.randn(1, 32, 128, 1024) / math.sqrt(128)
ta = create_ttnn_input_tensor(A, device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
tb = create_ttnn_input_tensor(
    B, device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
)
out = ttnn.allocate_tensor_on_device(
    ttnn.Shape([1, 32, 512, 1024]), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
)
geom, grid, blk, g = pd.plan(ta, tb, out, False, device)
print(geom)
print(grid)
print(blk)
print(g)
print("budget", pd.l1_cb_budget(device, [ta, tb, out]))
print(ttnn.get_arch_name() if hasattr(ttnn, "get_arch_name") else device.arch())
ttnn.close_device(device)
