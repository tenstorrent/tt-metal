import torch, ttnn
from ttnn.operations.groupnorm_sc_N_1_HW_C import groupnorm_sc_N_1_HW_C


def ref(x, G, gamma=None, beta=None, eps=1e-5):
    xf = x.float()
    N, _, HW, C = xf.shape
    w = gamma.float().reshape(C) if gamma is not None else None
    b = beta.float().reshape(C) if beta is not None else None
    o = torch.nn.functional.group_norm(xf.squeeze(1).permute(0, 2, 1), G, weight=w, bias=b, eps=eps)
    return o.permute(0, 2, 1).unsqueeze(1)


device = ttnn.open_device(device_id=0)
try:
    shape, G = (1, 1, 128, 1024), 32
    C = shape[-1]
    for xl_name, xl in (("RM", ttnn.ROW_MAJOR_LAYOUT), ("TILE", ttnn.TILE_LAYOUT)):
        for gl_name, gl in (("TILE", ttnn.TILE_LAYOUT), ("RM", ttnn.ROW_MAJOR_LAYOUT)):
            for affine in ("gamma_only", "gamma_beta"):
                torch.manual_seed(0)
                x = torch.randn(shape).to(torch.bfloat16)
                g = torch.randn(1, 1, 1, C).to(torch.bfloat16)
                b = torch.randn(1, 1, 1, C).to(torch.bfloat16) if affine == "gamma_beta" else None
                tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=xl, device=device)
                tg = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=gl, device=device)
                tb = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=gl, device=device) if b is not None else None
                outs = []
                for rep in range(2):
                    o = ttnn.to_torch(groupnorm_sc_N_1_HW_C(tx, G, gamma=tg, beta=tb)).float()
                    outs.append(o)
                e = ref(x, G, g, b)
                d = (outs[0] - e).abs()
                per_ch = d.amax(dim=(0, 1, 2))
                bad = (per_ch > 0.1).nonzero().flatten().tolist()
                rms = float(torch.sqrt((d**2).mean()) / torch.sqrt((e**2).mean()))
                print(
                    f"x={xl_name:4s} gamma={gl_name:4s} {affine:10s} rms={rms:.4f} max={float(d.max()):.3f} bad_channels={len(bad)} {bad[:20]} det={torch.equal(outs[0],outs[1])}"
                )
                if bad and affine == "gamma_only" and xl_name == "RM" and gl_name == "TILE":
                    z = ref(x, G)  # normalized without affine
                    for c in bad[:8]:
                        zc = z[0, 0, :, c]
                        oc = outs[0][0, 0, :, c]
                        fit = float((oc * zc).sum() / (zc * zc).sum())
                        print(
                            f"   ch {c}: tile {c//32} lane {c%32} gamma={float(g[0,0,0,c]):+.4f} fitted_gamma={fit:+.4f} resid={float((oc-fit*zc).abs().max()):.4f}"
                        )
finally:
    ttnn.close_device(device)
