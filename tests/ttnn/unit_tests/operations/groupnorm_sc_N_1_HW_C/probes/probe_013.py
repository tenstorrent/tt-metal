import torch, ttnn
from ttnn.operations.groupnorm_sc_N_1_HW_C import groupnorm_sc_N_1_HW_C


def ref(x, G, gamma=None, beta=None, eps=1e-5):
    xf = x.float()
    N, _, HW, C = xf.shape
    w = gamma.float().reshape(C) if gamma is not None else None
    b = beta.float().reshape(C) if beta is not None else None
    o = torch.nn.functional.group_norm(xf.squeeze(1).permute(0, 2, 1), G, weight=w, bias=b, eps=eps)
    return o.permute(0, 2, 1).unsqueeze(1)


def pcc(a, b):
    a = a.double().flatten() - a.double().mean()
    b = b.double().flatten() - b.double().mean()
    return float((a @ b) / (a.norm() * b.norm()))


TD = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16, ttnn.bfloat8_b: torch.bfloat16}
device = ttnn.open_device(device_id=0)
try:
    print(
        "=== bf8b x, c_non_aligned, gamma_beta, RM affine (the two failing golden cells) + writer out_block check ==="
    )
    for shape, G, adt in (
        ((1, 1, 64, 50), 1, ttnn.bfloat16),
        ((1, 1, 128, 100), 1, ttnn.float32),
        ((2, 1, 64, 47), 1, ttnn.bfloat16),
        ((1, 1, 64, 200), 8, ttnn.bfloat16),
        ((1, 1, 4096, 320), 32, ttnn.bfloat16),
    ):
        C = shape[-1]
        for dt in (ttnn.bfloat8_b, ttnn.bfloat16):
            torch.manual_seed(0)
            x = torch.randn(shape).to(TD[dt])
            g = torch.randn(1, 1, 1, C).to(TD[adt])
            b = torch.randn(1, 1, 1, C).to(TD[adt])
            tx = ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
            tg = ttnn.from_torch(g, dtype=adt, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
            tb = ttnn.from_torch(b, dtype=adt, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
            xr = ttnn.to_torch(tx).float() if dt == ttnn.bfloat8_b else x
            outs = [ttnn.to_torch(groupnorm_sc_N_1_HW_C(tx, G, gamma=tg, beta=tb)).float() for _ in range(3)]
            e = ref(xr, G, g, b)
            d = (outs[0] - e).abs()
            rms = float(torch.sqrt((d**2).mean()) / torch.sqrt((e**2).mean()))
            det = all(torch.equal(outs[0], o) for o in outs[1:])
            print(
                f"x={str(dt)[9:]:9s} {shape} G={G} affine={str(adt)[9:]}/RM: pcc={pcc(outs[0],e):.5f} rms={rms:.4f} max={float(d.max()):.3f} det={det}"
            )
finally:
    ttnn.close_device(device)
