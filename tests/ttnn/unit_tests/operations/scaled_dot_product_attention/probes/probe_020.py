import math, os, torch, ttnn
from tests.ttnn.utils_for_testing import comp_pcc
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

device = ttnn.open_device(device_id=0)
B, H, S, D = 1, 10, 9472, 128
scale = 1.0 / math.sqrt(D)


def rnd(*sh, seed):
    torch.manual_seed(seed)
    n = torch.randn(sh)
    b = torch.bernoulli(torch.full(sh, 0.001))
    return n + b * torch.randn(sh) * 10.0


tq = rnd(B, H, S, D, seed=1234)
tk = rnd(B, H, S, D, seed=1235)
tv = rnd(B, H, S, D, seed=1236)
q = ttnn.from_torch(tq, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
k = ttnn.from_torch(tk, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
v = ttnn.from_torch(tv, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
cfg = ttnn.ComputeConfigDescriptor(
    math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, math_approx_mode=False
)
out = scaled_dot_product_attention(q, k, v, compute_kernel_config=cfg)
res = ttnn.to_torch(out).float()
ref = torch.nn.functional.scaled_dot_product_attention(
    tq.float(), tk.float(), tv.float(), attn_mask=None, is_causal=False, scale=scale
)
_, pcc = comp_pcc(ref, res, 0.9)
print(f"RESULT_PCC EXPMODE={os.environ.get('TTNN_SDPA_EXP_MODE')} PCC={pcc}")
ttnn.close_device(device)
