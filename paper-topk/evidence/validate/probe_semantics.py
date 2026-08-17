#!/usr/bin/env python3
"""Value-semantics probes for the bf16 special-value mutation seen in topk.

1) identity-op probe: ttnn.identity (SFPU pass-through) on all special bit
   patterns -> does the generic bf16 compute datapath canonicalize?
2) NaN-only topk probe: readback top lanes as bits (largest and smallest).
3) zeros-only topk probe: -0/+0 boundary bits + indices.
All bits printed as hex; PROBE_DONE at end.
"""
import torch
import ttnn


def bf16_from_bits(bits):
    b = torch.as_tensor(bits, dtype=torch.int64)
    b = torch.where(b >= 0x8000, b - 0x10000, b).to(torch.int16)
    return b.view(torch.bfloat16)


def bits_of(t):
    return t.contiguous().view(torch.int16).to(torch.int64) & 0xFFFF


SPECIALS = [0x7FFF, 0x7FC0, 0x7FC1, 0xFFC0, 0xFFC1, 0xFFFF,  # NaNs both signs, payload variants
            0x7F80, 0xFF80,                                    # +-Inf
            0x0000, 0x8000,                                    # +-0
            0x0001, 0x007F, 0x8001, 0x807F,                    # +-subnormals
            0x0080, 0x8080,                                    # +-min normal
            0x3F80, 0xBF80]                                    # +-1.0 controls


def fmt(xs):
    return [f"{b:04x}" for b in xs]


def main():
    dev = ttnn.open_device(device_id=0)
    try:
        # ---- 1) identity op ----
        n = 1024
        row = torch.linspace(-1, 1, n).to(torch.bfloat16)
        row[100 : 100 + len(SPECIALS)] = bf16_from_bits(SPECIALS)
        x = row.reshape(1, 1, 32, 32).clone()
        tt = ttnn.from_torch(x, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=dev)
        out = ttnn.to_torch(ttnn.identity(tt)).flatten()
        ib = bits_of(out)[100 : 100 + len(SPECIALS)].tolist()
        print("IDENTITY in :", fmt(SPECIALS), flush=True)
        print("IDENTITY out:", fmt(ib), flush=True)

        # ---- 2) NaN-only topk, largest & smallest, W=10000 single-core ----
        for largest in (True, False):
            w, k = 10000, 32
            base = torch.linspace(-1, 1, w).to(torch.bfloat16)
            nanspec = [0x7FFF, 0x7FC0, 0xFFC0, 0x7F80, 0xFF80]
            base[100 : 100 + len(nanspec)] = bf16_from_bits(nanspec)
            x = base.expand(32, w).clone().reshape(1, 1, 32, w)
            tt = ttnn.from_torch(x, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=dev)
            v, i = ttnn.topk(tt, k, dim=-1, largest=largest, sorted=True)
            vb = bits_of(ttnn.to_torch(v)).reshape(-1, k)
            ii = ttnn.to_torch(i, dtype=torch.uint16).reshape(-1, k).to(torch.int64)
            print(f"NAN topk largest={largest} in specials@100..: {fmt(nanspec)}", flush=True)
            print(f"  row0 top8 vals: {fmt(vb[0,:8].tolist())}", flush=True)
            print(f"  row0 top8 idx : {ii[0,:8].tolist()}", flush=True)

        # ---- 3) zeros-only topk: negative base so zeros are the top mass ----
        w, k = 10000, 32
        base = (torch.linspace(-2, -1, w)).to(torch.bfloat16)
        zspec = [0x3F80] + [0x0000, 0x8000] * 8  # 1.0 winner + 8x(+0,-0)
        base[100 : 100 + len(zspec)] = bf16_from_bits(zspec)
        x = base.expand(32, w).clone().reshape(1, 1, 32, w)
        tt = ttnn.from_torch(x, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=dev)
        v, i = ttnn.topk(tt, k, dim=-1, largest=True, sorted=True)
        vb = bits_of(ttnn.to_torch(v)).reshape(-1, k)
        ii = ttnn.to_torch(i, dtype=torch.uint16).reshape(-1, k).to(torch.int64)
        print(f"ZEROS topk in specials@100..: {fmt(zspec)}", flush=True)
        print(f"  row0 top20 vals: {fmt(vb[0,:20].tolist())}", flush=True)
        print(f"  row0 top20 idx : {ii[0,:20].tolist()}", flush=True)
        print("PROBE_DONE", flush=True)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
