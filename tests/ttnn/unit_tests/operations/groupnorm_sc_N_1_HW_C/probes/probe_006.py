import torch, ttnn

device = ttnn.open_device(device_id=0)
from ttnn.operations.groupnorm_sc_N_1_HW_C import groupnorm_sc_N_1_HW_C

# Reproduce the failing golden cell: (1,1,64,17) fp32 input, gamma_only bf8b TILE gamma
torch.manual_seed(0)
x = torch.randn(1, 1, 64, 17, dtype=torch.float32)
g_bf16 = torch.randn(1, 1, 1, 17, dtype=torch.bfloat16)  # golden uses bf16 reference for bf8b


def gn(x, gamma):
    xp = x.to(torch.float32).squeeze(1).permute(0, 2, 1)
    y = torch.nn.functional.group_norm(xp, 1, weight=gamma.to(torch.float32).reshape(-1), eps=1e-5)
    return y.permute(0, 2, 1).unsqueeze(1)


expected = gn(x, g_bf16)

# bf8b round-trip of gamma to measure the quantization floor
tt_g = ttnn.from_torch(g_bf16, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
g_q = ttnn.to_torch(tt_g).to(torch.float32)[..., :17]
floor = gn(x, g_q)
floor_rms = ((floor - expected).pow(2).mean().sqrt() / expected.std()).item()

tt_x = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
out = ttnn.to_torch(groupnorm_sc_N_1_HW_C(tt_x, 1, gamma=tt_g)).to(torch.float32)
op_rms = ((out - expected).pow(2).mean().sqrt() / expected.std()).item()
op_vs_floor = ((out - floor).pow(2).mean().sqrt() / expected.std()).item()
print(f"probe floor_rms={floor_rms:.5f} op_rms={op_rms:.5f} op_vs_floor={op_vs_floor:.5f}")
ttnn.close_device(device)
