# MAX REDUCE_ROW output bits per side, and their comparison against an exact (fp64) reference.
#   run:     python bits.py run <side> <out.pt>
#   compare: python bits.py cmp <ref side>=<a.pt> <side>=<b.pt> ...   (first is the baseline for "better / worse")
# Cases: every bf16 pattern (and every TF32 pattern) one per row with the rest of the row the lowest finite value, 0 or -0,
# plus a dense random layout, at scaler 1.0 (what every production caller passes); and every bf16 scaler bit pattern (and
# a TF32 sample) against two broad data tiles. Scaler tiles use the production layout (row 0 of each face).
import hashlib
import sys

import torch

BF16_LOW = -3.3895313892515355e38
F32_LOW = -3.4028234663852886e38


def bf16_from_bits(b):
    return b.to(torch.int32).to(torch.int16).view(torch.bfloat16)


def rich_tiles():
    """Two broad 32x32 data tiles (bf16 values, as float32)."""
    g = torch.Generator().manual_seed(58705)
    a = torch.zeros(32, 32)
    a[0:4] = torch.randn(4, 32, generator=g) * torch.exp2(torch.randint(-8, 8, (4, 32), generator=g).float())
    a[4] = -torch.arange(1, 33).float()
    a[5] = 0.0
    a[6] = -0.0
    a[7] = torch.where(torch.arange(32) % 2 == 0, torch.tensor(-0.0), -torch.arange(1, 33).float())
    a[8] = torch.where(torch.arange(32) % 2 == 0, torch.tensor(0.0), -torch.arange(1, 33).float())
    a[9] = torch.where(torch.arange(32) % 2 == 0, 1.0, -1.0) * 1.0e38
    a[10] = torch.exp2(torch.arange(96, 128).float())
    a[11] = torch.where(torch.arange(32) % 2 == 0, 1.0, -1.0) * torch.exp2(-126 + (torch.arange(32) % 7).float())
    a[12] = bf16_from_bits(torch.arange(1, 33) | (torch.arange(32) % 2 << 15)).float()
    a[13] = torch.where(torch.arange(32) % 2 == 0, bf16_from_bits(torch.arange(1, 33)).float(), -torch.arange(1, 33).float())
    a[14] = torch.randn(32, generator=g)
    a[14, 5] = float("inf")
    a[15] = -torch.rand(32, generator=g) - 1
    a[15, 7] = float("-inf")
    a[16] = torch.randn(32, generator=g)
    a[16, 9] = float("nan")
    a[17] = float("-inf")
    a[18] = torch.where(torch.arange(32) % 2 == 0, 1.0, 2.0) * (1 + torch.arange(32).float() / 64)
    e = torch.randint(1, 255, (13, 32), generator=g)
    m = torch.randint(0, 128, (13, 32), generator=g)
    s = torch.randint(0, 2, (13, 32), generator=g)
    a[19:32] = bf16_from_bits((s << 15) | (e << 7) | m).float()
    e = torch.randint(0, 255, (32, 32), generator=g)
    m = torch.randint(0, 128, (32, 32), generator=g)
    s = torch.randint(0, 2, (32, 32), generator=g)
    b = bf16_from_bits((s << 15) | (e << 7) | m).float()
    b[torch.rand(32, 32, generator=g) < 0.1] = 0.0
    return [a.bfloat16().float(), b.bfloat16().float()]


def cases():
    """name -> (fmt 'bf16'|'tf32', fp32_dest, data [n,32,32] float32, scalers [n] float32)"""
    out = {}
    p16 = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).float()
    p32 = (torch.arange(0, 1 << 19, dtype=torch.int64) << 13).to(torch.int32).view(torch.float32)
    for fmt, p, low in (("bf16", p16, BF16_LOW), ("tf32", p32, F32_LOW)):
        n = p.numel()
        col = torch.arange(n) % 32
        fills = (("low", low), ("zero", 0.0), ("negzero", -0.0)) if fmt == "bf16" else (("low", low), ("zero", 0.0))
        for name, fill in fills:
            x = torch.full((n, 32), fill)
            x[torch.arange(n), col] = p
            for d in ((False, True) if fmt == "bf16" else (True,)):
                out[f"{fmt} every {name} dest32={d}"] = (fmt, d, x.reshape(n // 32, 32, 32), torch.ones(n // 32))
        g = torch.Generator().manual_seed(23)
        shape = (2048, 32, 32)
        dense = torch.randn(shape, generator=g) * torch.exp2(torch.randint(-8, 8, shape, generator=g).float())
        for d in ((False, True) if fmt == "bf16" else (True,)):
            out[f"{fmt} dense dest32={d}"] = (fmt, d, dense, torch.ones(2048))
    tiles = rich_tiles()
    sc16 = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).float()
    for ti, t in enumerate(tiles):
        for d in (False, True):
            out[f"bf16 scaler sweep tile{ti} dest32={d}"] = ("bf16", d, t.expand(sc16.numel(), 32, 32), sc16)
    g = torch.Generator().manual_seed(29)
    e = torch.arange(256).repeat_interleave(8)
    m = torch.randint(0, 1 << 23, (2048,), generator=g)
    m[::8] = 0
    sc32 = ((e << 23) | m).to(torch.int32).view(torch.float32)
    sc32 = torch.cat([sc32, -sc32])
    for ti, t in enumerate(tiles):
        out[f"tf32 scaler sample tile{ti} dest32=True"] = ("tf32", True, t.expand(sc32.numel(), 32, 32), sc32)
    return out


def run(side, path):
    import ttnn

    from ttnn_kernels import run_case

    device = ttnn.open_device(device_id=0)
    res = {}
    for name, (fmt, d, data, sc) in cases().items():
        s = torch.zeros(sc.numel(), 32, 32)
        s[:, 0, :] = sc[:, None]
        s[:, 16, :] = sc[:, None]
        y = run_case(device, data.contiguous(), s, fmt, d)
        y = y.reshape(-1, 32, 32)
        iv = y.contiguous().view(torch.int16 if y.dtype == torch.bfloat16 else torch.int32)
        rest = iv[:, :, 1:].to(torch.int64)
        w = torch.arange(1, 32 * 31 + 1, dtype=torch.int64).reshape(32, 31)
        res[name] = {"col0": iv[:, :, 0].clone(), "rest_ck": (rest * w).sum(dim=(1, 2)), "rest_nz": (rest != 0).sum(dim=(1, 2)), "dtype": str(y.dtype)}
        print(f"CASE {side} {name} ok tiles {iv.shape[0]} rest_nonzero {int((res[name]['rest_nz'] > 0).sum())}", flush=True)
    torch.save(res, path)
    ttnn.close_device(device)


def to_f64(bits, dt):
    if dt == "torch.bfloat16":
        return bits.to(torch.int16).view(torch.bfloat16).double()
    return bits.to(torch.int32).view(torch.float32).double()


def exact(fmt, d, data, sc, dt):
    """fp64 max over the row of x * s, NaN propagating, rounded to the output dtype."""
    prod = data.double() * sc.double()[:, None, None]
    m = prod.amax(dim=2)
    m = torch.where(torch.isnan(prod).any(dim=2), torch.tensor(float("nan"), dtype=torch.float64), m)
    if dt == "torch.bfloat16":
        return m.to(torch.bfloat16).double()
    return m.to(torch.float32).double()


def err(v, e):
    ve = torch.where(torch.isnan(e), torch.where(torch.isnan(v), 0.0, float("inf")), torch.zeros_like(v))
    same = (v == e) | (torch.isnan(v) & torch.isnan(e))
    diff = (v - e).abs()
    diff = torch.where(torch.isnan(diff), torch.tensor(float("inf"), dtype=torch.float64), diff)
    return torch.where(torch.isnan(e), ve, torch.where(same, 0.0, diff))


def vclass(v):
    out = []
    for x, sg in zip(v.tolist(), torch.signbit(v).tolist()):
        sgn = "-" if sg else "+"
        if x != x:
            out.append("nan")
        elif x in (float("inf"), float("-inf")):
            out.append(sgn + "inf")
        elif x == 0:
            out.append(sgn + "0")
        elif abs(x) < 2.0**-126:
            out.append(sgn + "den")
        else:
            out.append(sgn + "norm")
    return out


def by_class(name, data, sc, mask):
    """Class of the input that drives each differing value: the scaler for the sweeps, the swept pattern otherwise."""
    from collections import Counter
    idx = mask.nonzero()
    if idx.numel() == 0:
        return ""
    t, r = idx[:, 0], idx[:, 1]
    if "scaler" in name:
        v = sc[t]
    elif "every" in name:
        v = data[t, r, (t * 32 + r) % 32]
    else:
        return ""
    return " ".join(f"{k}:{n}" for k, n in sorted(Counter(vclass(v.float())).items()))


def cmp(args):
    sides = [a.split("=", 1) for a in args]
    loaded = {s: torch.load(p, weights_only=True) for s, p in sides}
    base = sides[0][0]
    cs = cases()
    for name, (fmt, d, data, sc) in cs.items():
        r0 = loaded[base].get(name)
        if r0 is None:
            print(f"{name}: missing in {base}")
            continue
        dt = r0["dtype"]
        ex = exact(fmt, d, data, sc, dt)
        e0 = err(to_f64(r0["col0"], dt), ex)
        line = [f"{name}: {r0['col0'].shape[0]} tiles; {base} col0 exact {int((e0 == 0).sum())} of {e0.numel()}"]
        for s, _ in sides[1:]:
            r = loaded[s].get(name)
            if r is None:
                line.append(f"{s} missing")
                continue
            dcol = r["col0"] != r0["col0"]
            drest = (r["rest_ck"] != r0["rest_ck"]) | (r["rest_nz"] != r0["rest_nz"])
            tiles = int((dcol.any(dim=1) | drest).sum())
            v = to_f64(r["col0"], dt)
            v0 = to_f64(r0["col0"], dt)
            e1 = err(v, ex)
            szero = dcol & (v == 0) & (v0 == 0)
            nanp = dcol & torch.isnan(v) & torch.isnan(v0)
            other = dcol & ~szero & ~nanp
            better = other & (e1 < e0)
            worse = other & (e1 > e0)
            same_err = other & (e1 == e0)
            line.append(
                f"{s} vs {base}: tiles differ {tiles} (rest {int(drest.sum())}), values differ {int(dcol.sum())}: "
                f"signed zero {int(szero.sum())}, NaN payload {int(nanp.sum())}, closer to exact {int(better.sum())}, "
                f"equally far {int(same_err.sum())}, farther {int(worse.sum())}"
            )
            if int(dcol.sum()):
                print(f"  CLASS {name} {s} differ by input class: {by_class(name, data, sc, dcol)}")
                if int(worse.sum()):
                    print(f"  CLASS {name} {s} farther by input class: {by_class(name, data, sc, worse)}")
                if int(better.sum()):
                    print(f"  CLASS {name} {s} closer by input class: {by_class(name, data, sc, better)}")
                idx = dcol.nonzero()[:12]
                for t, row in idx.tolist():
                    xs = data[t, row]
                    print(
                        f"  EX {name} {s} tile {t} row {row} scaler {float(sc[t]):.6g} ({hex(int(sc[t].to(torch.bfloat16).view(torch.int16)) & 0xFFFF) if fmt == 'bf16' else hex(int(sc[t].view(torch.int32)) & 0xFFFFFFFF)}) "
                        f"row max {float(xs.max()):.6g} min {float(xs.min()):.6g} | {base} {float(v0[t, row]):.6g} {s} {float(v[t, row]):.6g} exact {float(ex[t, row]):.6g}"
                    )
            if int(worse.sum()):
                for t, row in (worse.nonzero()[:12]).tolist():
                    xs = data[t, row]
                    print(f"  WORSE {name} {s} tile {t} row {row} scaler {float(sc[t]):.6g} row max {float(xs.max()):.6g} | {base} {float(v0[t, row]):.6g} {s} {float(v[t, row]):.6g} exact {float(ex[t, row]):.6g}")
        print(" | ".join(line), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3])
    else:
        cmp(sys.argv[2:])
