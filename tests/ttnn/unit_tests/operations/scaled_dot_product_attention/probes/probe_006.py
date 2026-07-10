import torch, ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


def run(shape, dtype=ttnn.bfloat8_b):
    torch.manual_seed(0)
    B, H, S, D = shape
    Q = torch.randn(shape, dtype=torch.bfloat16)
    K = torch.randn(shape, dtype=torch.bfloat16)
    V = torch.randn(shape, dtype=torch.bfloat16)
    m = torch.zeros(B, 1, S, S, dtype=torch.bfloat16)
    m.masked_fill_(torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1), float("-inf"))
    exp = torch.nn.functional.scaled_dot_product_attention(Q.float(), K.float(), V.float(), attn_mask=m.float())
    to = lambda t: ttnn.from_torch(
        t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
    )
    out = ttnn.to_torch(
        scaled_dot_product_attention(to(Q), to(K), to(V), attn_mask=to(m), compute_kernel_config=cfg)
    ).float()
    e = exp.flatten().float()
    o = out.flatten().float()
    pcc = torch.corrcoef(torch.stack([e, o]))[0, 1].item()
    diff = (exp - out).abs()
    rms = torch.sqrt((diff**2).mean()).item() / torch.sqrt((exp**2).mean()).item()
    print(f"{shape} bf8b+custom: PCC={pcc:.5f} rel_rms={rms:.4f}")


device = ttnn.open_device(device_id=0)
try:
    for s in [(1, 1, 128, 64), (1, 1, 32, 32), (1, 8, 256, 64), (1, 12, 512, 64), (2, 4, 256, 128)]:
        run(s)
finally:
    ttnn.close_device(device)
