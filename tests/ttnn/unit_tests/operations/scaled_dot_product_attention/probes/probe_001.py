import torch, ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention, validate

dev = ttnn.open_device(device_id=0)
try:
    b, nh, nkv, s, d = 2, 8, 1, 160, 64
    Q = torch.randn(b, nh, s, d)
    K = torch.randn(b, nkv, s, d)
    V = torch.randn(b, nkv, s, d)
    mask = torch.zeros(1, 1, s, s)  # bcast batch + head
    to = lambda t: ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False
    )
    try:
        out = scaled_dot_product_attention(to(Q), to(K), to(V), attn_mask=to(mask), compute_kernel_config=cfg)
        print("RAN OK - no rejection! shape", out.shape)
    except Exception as e:
        print("EXC TYPE:", type(e).__name__)
        print("EXC MSG:", str(e)[:300])
finally:
    ttnn.close_device(dev)
