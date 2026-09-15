import torch, ttnn
from ttnn.operations.groupnorm_sc_N_1_HW_C import groupnorm_sc_N_1_HW_C
import ttnn.operations.groupnorm_sc_N_1_HW_C.groupnorm_sc_N_1_HW_C as opmod


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
    # 1. determinism / correctness of the fixed gamma_only path on RM input
    print("=== fix check: RM x, gamma_only ===")
    for shape, G in (((1, 1, 128, 1024), 32), ((4, 1, 128, 256), 8), ((1, 1, 64, 320), 32), ((2, 1, 512, 256), 8)):
        C = shape[-1]
        for gl_name, gl in (("TILE", ttnn.TILE_LAYOUT), ("RM", ttnn.ROW_MAJOR_LAYOUT)):
            torch.manual_seed(0)
            x = torch.randn(shape).to(torch.bfloat16)
            g = torch.randn(1, 1, 1, C).to(torch.bfloat16)
            tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
            tg = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=gl, device=device)
            outs = [ttnn.to_torch(groupnorm_sc_N_1_HW_C(tx, G, gamma=tg)).float() for _ in range(4)]
            e = ref(x, G, g)
            d = (outs[0] - e).abs()
            rms = float(torch.sqrt((d**2).mean()) / torch.sqrt((e**2).mean()))
            det = all(torch.equal(outs[0], o) for o in outs[1:])
            print(f"{shape} G={G} gamma={gl_name}: rms={rms:.4f} max={float(d.max()):.3f} det={det}")

    # 2. dtype probes: temporarily widen SUPPORTED in-process
    print("=== dtype probes (SUPPORTED widened in-process) ===")
    opmod.SUPPORTED["dtype"] = [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b]
    opmod.SUPPORTED["affine_dtype"] = [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b, "none"]
    cases = []
    for dt in (ttnn.float32, ttnn.bfloat8_b):
        for xl in [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT] if dt != ttnn.bfloat8_b else [ttnn.TILE_LAYOUT]:
            for affine in ("no_affine", "gamma_beta", "gamma_only"):
                for adt, al in (
                    (ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
                    (ttnn.float32, ttnn.TILE_LAYOUT),
                    (ttnn.float32, ttnn.ROW_MAJOR_LAYOUT),
                    (ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
                ):
                    if affine == "no_affine" and adt != ttnn.bfloat16:
                        continue
                    cases.append((dt, xl, affine, adt, al))
    # also bf16 x with fp32 / bf8b affine
    for affine in ("gamma_beta", "gamma_only"):
        for adt, al in (
            (ttnn.float32, ttnn.TILE_LAYOUT),
            (ttnn.float32, ttnn.ROW_MAJOR_LAYOUT),
            (ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
        ):
            cases.append((ttnn.bfloat16, ttnn.TILE_LAYOUT, affine, adt, al))
    shapes = (((1, 1, 64, 320), 32), ((2, 1, 256, 1280), 32), ((1, 1, 32, 32), 1))
    for dt, xl, affine, adt, al in cases:
        for shape, G in shapes:
            C = shape[-1]
            torch.manual_seed(0)
            x = torch.randn(shape).to(TD[dt])
            g = torch.randn(1, 1, 1, C).to(TD[adt])
            b = torch.randn(1, 1, 1, C).to(TD[adt])
            try:
                tx = ttnn.from_torch(x, dtype=dt, layout=xl, device=device)
                tg = ttnn.from_torch(g, dtype=adt, layout=al, device=device) if affine != "no_affine" else None
                tb = ttnn.from_torch(b, dtype=adt, layout=al, device=device) if affine == "gamma_beta" else None
                # reference uses the dequantized device values for bf8b
                xr = ttnn.to_torch(tx).float() if dt == ttnn.bfloat8_b else x
                gr = ttnn.to_torch(tg).float() if (tg is not None and adt == ttnn.bfloat8_b) else g
                br = ttnn.to_torch(tb).float() if (tb is not None and adt == ttnn.bfloat8_b) else b
                out = groupnorm_sc_N_1_HW_C(tx, G, gamma=tg, beta=tb)
                o = ttnn.to_torch(out).float()
                e = ref(xr, G, gr if tg is not None else None, br if tb is not None else None)
                d = (o - e).abs()
                rms = float(torch.sqrt((d**2).mean()) / torch.sqrt((e**2).mean()))
                status = f"pcc={pcc(o,e):.5f} rms={rms:.4f} max={float(d.max()):.3f} finite={bool(torch.isfinite(o).all())} out_dtype_ok={out.dtype==dt}"
            except Exception as ex:
                status = f"EXC {type(ex).__name__}: {str(ex)[:160]}"
            print(
                f"x={dt} {('RM' if xl==ttnn.ROW_MAJOR_LAYOUT else 'TILE'):4s} {affine:10s} affine={adt}/{'RM' if al==ttnn.ROW_MAJOR_LAYOUT else 'TILE'} {shape}: {status}"
            )
finally:
    ttnn.close_device(device)
