import importlib, torch, ttnn
from ttnn.operations.groupnorm_sc_N_1_HW_C import groupnorm_sc_N_1_HW_C

opmod = importlib.import_module("ttnn.operations.groupnorm_sc_N_1_HW_C.groupnorm_sc_N_1_HW_C")


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


def run(shape, G, dt, xl, affine, adt, al, device, seed=0):
    C = shape[-1]
    torch.manual_seed(seed)
    x = torch.randn(shape).to(TD[dt])
    g = torch.randn(1, 1, 1, C).to(TD[adt])
    b = torch.randn(1, 1, 1, C).to(TD[adt])
    tx = ttnn.from_torch(x, dtype=dt, layout=xl, device=device)
    tg = ttnn.from_torch(g, dtype=adt, layout=al, device=device) if affine != "no_affine" else None
    tb = ttnn.from_torch(b, dtype=adt, layout=al, device=device) if affine == "gamma_beta" else None
    xr = ttnn.to_torch(tx).float() if dt == ttnn.bfloat8_b else x
    gr = ttnn.to_torch(tg).float() if (tg is not None and adt == ttnn.bfloat8_b) else g
    br = ttnn.to_torch(tb).float() if (tb is not None and adt == ttnn.bfloat8_b) else b
    out = groupnorm_sc_N_1_HW_C(tx, G, gamma=tg, beta=tb)
    o = ttnn.to_torch(out).float()
    e = ref(xr, G, gr if tg is not None else None, br if tb is not None else None)
    d = (o - e).abs()
    rms = float(torch.sqrt((d**2).mean()) / torch.sqrt((e**2).mean()))
    return f"pcc={pcc(o,e):.5f} rms={rms:.4f} max={float(d.max()):.3f} finite={bool(torch.isfinite(o).all())} out_dtype_ok={out.dtype==dt} shape_ok={list(o.shape)==list(shape)}"


device = ttnn.open_device(device_id=0)
try:
    print("=== bf8b probes ===")
    for shape, G in (((1, 1, 64, 320), 32), ((2, 1, 256, 1280), 32), ((1, 1, 32, 32), 1), ((1, 1, 128, 1024), 32)):
        for dt, xl, affine, adt, al in (
            (ttnn.bfloat8_b, ttnn.TILE_LAYOUT, "no_affine", ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
            (ttnn.bfloat8_b, ttnn.TILE_LAYOUT, "gamma_beta", ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
            (ttnn.bfloat8_b, ttnn.TILE_LAYOUT, "gamma_only", ttnn.float32, ttnn.TILE_LAYOUT),
            (ttnn.bfloat8_b, ttnn.TILE_LAYOUT, "gamma_beta", ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
            (ttnn.bfloat16, ttnn.TILE_LAYOUT, "gamma_beta", ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
            (ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, "gamma_only", ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
            (ttnn.float32, ttnn.TILE_LAYOUT, "gamma_beta", ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
        ):
            try:
                s = run(shape, G, dt, xl, affine, adt, al, device)
            except Exception as ex:
                s = f"EXC {type(ex).__name__}: {str(ex)[:200]}"
            print(
                f"x={str(dt)[9:]:9s} {('RM' if xl==ttnn.ROW_MAJOR_LAYOUT else 'TILE'):4s} {affine:10s} affine={str(adt)[9:]}/{'RM' if al==ttnn.ROW_MAJOR_LAYOUT else 'TILE'} {shape}: {s}"
            )

    print("=== alignment probes (SUPPORTED['alignment'] widened in-process) ===")
    opmod.SUPPORTED["alignment"] = ["tile_aligned", "hw_non_aligned", "c_non_aligned"]
    for shape, G in (
        ((1, 1, 17, 64), 1),
        ((1, 1, 50, 128), 1),
        ((2, 1, 100, 128), 1),
        ((1, 1, 64, 17), 1),
        ((1, 1, 64, 50), 1),
        ((1, 1, 128, 100), 1),
        ((2, 1, 64, 47), 1),
        ((1, 1, 64, 48), 2),
        ((1, 1, 64, 80), 4),
        ((1, 1, 128, 144), 4),
        ((1, 1, 64, 200), 8),
        ((2, 1, 64, 48), 2),
    ):
        for xl in (ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT):
            for affine, adt, al in (
                ("no_affine", ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
                ("gamma_beta", ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
                ("gamma_beta", ttnn.bfloat16, ttnn.TILE_LAYOUT),
            ):
                try:
                    s = run(shape, G, ttnn.bfloat16, xl, affine, adt, al, device)
                except Exception as ex:
                    s = f"EXC {type(ex).__name__}: {str(ex)[:200]}"
                print(
                    f"al= x={('RM' if xl==ttnn.ROW_MAJOR_LAYOUT else 'TILE'):4s} {affine:10s} affine/{'RM' if al==ttnn.ROW_MAJOR_LAYOUT else 'TILE'} {shape} G={G}: {s}"
                )

    print("=== compute config refusal ===")
    x = ttnn.from_torch(torch.randn(1, 1, 64, 64), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    for fp32acc, full in ((False, True), (True, False), (True, True)):
        cfg = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=fp32acc,
            dst_full_sync_en=full,
        )
        try:
            o = ttnn.to_torch(groupnorm_sc_N_1_HW_C(x, 2, compute_kernel_config=cfg)).float()
            e = ref(ttnn.to_torch(x), 2)
            print(f"cfg fp32acc={fp32acc} full_sync={full} HiFi2: pcc={pcc(o,e):.5f}")
        except Exception as ex:
            print(
                f"cfg fp32acc={fp32acc} full_sync={full}: EXC {type(ex).__name__} isNotImpl={isinstance(ex, NotImplementedError)}: {str(ex)[:120]}"
            )
finally:
    ttnn.close_device(device)
