import torch, ttnn, re
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention as sdpa
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention_program_descriptor as pd

dev = ttnn.open_device(device_id=0)


def ref(Q, K, V):
    return torch.nn.functional.scaled_dot_product_attention(Q.float(), K.float(), V.float())


def run(D, dt, mask):
    shape = (1, 1, 128, D)
    acc = False if dt == ttnn.bfloat16 else True  # fp32 needs acc True (excl otherwise); bf8b either
    if dt == ttnn.float32:
        acc = True
    if dt == ttnn.bfloat8_b:
        acc = True
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=acc
    )
    Qt, Kt, Vt = torch.randn(*shape), torch.randn(*shape), torch.randn(*shape)
    to = lambda t: ttnn.from_torch(
        t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    kw = {}
    if mask:
        kw["attn_mask"] = to(torch.zeros(1, 1, 128, 128))
    # capture chosen chunks
    sqt, skvt, dht = shape[2] // 32, shape[2] // 32, D // 32
    in_df = dt
    out_df = dt
    interm_df = ttnn.float32 if dt == ttnn.float32 else ttnn.bfloat16
    sq, sk = pd._pick_chunks(sqt, skvt, dht, mask, in_df, out_df, interm_df, ttnn.bfloat16)
    try:
        out = sdpa(to(Qt), to(Kt), to(Vt), compute_kernel_config=cfg, **kw)
        r = ttnn.to_torch(out).float()
        g = (
            ref(Qt, Kt, Vt)
            if not mask
            else torch.nn.functional.scaled_dot_product_attention(
                Qt.float(), Kt.float(), Vt.float(), attn_mask=torch.zeros(1, 1, 128, 128)
            )
        )
        pcc = torch.corrcoef(torch.stack([r.flatten(), g.flatten()]))[0, 1].item()
        print(f"D={D} {str(dt).split('.')[-1]} mask={mask} chunks=({sq},{sk}) OK pcc={pcc:.5f}")
    except Exception as e:
        m = re.search(r"grow to (\d+) B", str(e))
        print(
            f"D={D} {str(dt).split('.')[-1]} mask={mask} chunks=({sq},{sk}) FAIL grow_to={m.group(1) if m else str(e)[:120]}"
        )


try:
    for dt in [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b]:
        for D in [128, 256, 512, 1024]:
            for mask in [False, True]:
                run(D, dt, mask)
finally:
    ttnn.close_device(dev)
