import torch, ttnn, re
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention as sdpa

dev = ttnn.open_device(device_id=0)


def run(D, dt, mask, accf):
    shape = (1, 1, 128, D)
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=accf
    )
    to = lambda t: ttnn.from_torch(
        t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    kw = {}
    if mask:
        kw["attn_mask"] = to(torch.zeros(1, 1, 128, 128))
    try:
        out = sdpa(
            to(torch.randn(*shape)), to(torch.randn(*shape)), to(torch.randn(*shape)), compute_kernel_config=cfg, **kw
        )
        print(f"D={D} {dt} mask={mask} acc={accf} OK")
    except Exception as e:
        m = re.search(r"grow to (\d+) B", str(e))
        print(f"D={D} {dt} mask={mask} acc={accf} FAIL grow_to={m.group(1) if m else '?'}")


try:
    run(1024, ttnn.bfloat16, False, True)
    run(1024, ttnn.bfloat16, True, True)  # +mask CB -> +65536 to sum
    run(256, ttnn.float32, False, True)
    run(256, ttnn.float32, True, True)
    run(512, ttnn.float32, False, True)
    run(1024, ttnn.bfloat8_b, False, True)
finally:
    ttnn.close_device(dev)
