import torch, ttnn
from ttnn.operations.g6_to_layout import g6_to_layout

device = ttnn.open_device(device_id=0)


def gs(shape, seed=460, sseed=461):
    g = torch.Generator()
    g.manual_seed(seed)
    base = torch.randn(shape, generator=g)
    g2 = torch.Generator()
    g2.manual_seed(sseed)
    k = torch.randint(-8, 9, (*shape[:-1], shape[-1] // 16), generator=g2)
    return (base * torch.pow(2.0, k).repeat_interleave(16, dim=-1)).to(torch.bfloat16)


for name, shape, gen in [
    ("c16", (1, 1, 1, 256), "randn"),
    ("Q3", (1, 1, 32, 256), "gs"),
    ("w200", (1, 2, 70, 208), "randn"),
]:
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16) if gen == "randn" else gs(shape)
    out = (
        ttnn.to_torch(
            g6_to_layout(
                ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device),
                ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat8_b,
            )
        )
        .reshape(shape)
        .double()
    )
    host = ttnn.to_torch(ttnn.from_torch(x, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)).reshape(shape).double()
    ref = x.double()
    pcc = torch.corrcoef(torch.stack([out.flatten(), ref.flatten()]))[0, 1].item()
    rrms = ((out - ref).pow(2).mean().sqrt() / ref.std()).item()
    rrms_host = ((host - ref).pow(2).mean().sqrt() / ref.std()).item()
    print(
        f"RESULT {name} pcc={pcc:.6f} rel_rms={rrms:.5f} host_rel_rms={rrms_host:.5f} max|dev-host|={((out-host).abs().max()).item():.4g} exact_vs_host={(out==host).float().mean().item():.4f}"
    )
ttnn.close_device(device)
