import torch, ttnn
import torch.nn.functional as F
from ttnn.operations.groupnorm_sc_N_1_HW_C import groupnorm_sc_N_1_HW_C

device = ttnn.open_device(device_id=0)
torch.manual_seed(0)
shape = (1, 1, 32, 32)
x = torch.randn(shape, dtype=torch.float32)
gamma = torch.randn((1, 1, 1, 32), dtype=torch.float32)
beta = torch.randn((1, 1, 1, 32), dtype=torch.float32)
x_ncl = x.reshape(1, 32, 32).permute(0, 2, 1).contiguous()
y_ref = (
    F.group_norm(x_ncl, num_groups=1, weight=gamma.reshape(32), bias=beta.reshape(32), eps=1e-5)
    .permute(0, 2, 1)
    .reshape(1, 1, 32, 32)
    .contiguous()
)
x_tt = ttnn.from_torch(
    x.to(torch.bfloat16),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
g_tt = ttnn.from_torch(
    gamma.to(torch.bfloat16),
    dtype=ttnn.bfloat16,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
b_tt = ttnn.from_torch(
    beta.to(torch.bfloat16),
    dtype=ttnn.bfloat16,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
print("=== fp32_acc=True, NO UnpackToDestFp32 tags ===")
y_tt = groupnorm_sc_N_1_HW_C(x_tt, 1, gamma=g_tt, beta=b_tt, eps=1e-5)
y_out = ttnn.to_torch(y_tt).to(torch.float32)
print("y_out[0,0,0,:8] =", y_out[0, 0, 0, :8].tolist())
print("y_ref[0,0,0,:8] =", y_ref[0, 0, 0, :8].tolist())
print("diff max =", (y_out - y_ref).abs().max().item())
ttnn.close_device(device)
