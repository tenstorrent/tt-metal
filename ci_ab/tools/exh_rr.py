# Output bits of ttnn.embedding_bw (reshuffle_rows on Blackhole) for index patterns that exercise every case of the body:
# random, every row into one embedding, all distinct, runs of repeats, skipped rows (padding past the vocabulary), uint32
# and bfloat16 indices, bf16 and bf8_b outputs; a sha256 per case. usage: python exh_rr.py <out.json>
import hashlib, json, sys
import torch
import ttnn

dev = ttnn.open_device(device_id=0)
res = {}


def h(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:20]


def pattern(name, b, s, v, g):
    n = b * s
    if name == "random":
        return torch.randint(0, v, (b, s), generator=g)
    if name == "one":
        return torch.full((b, s), 5 % v)
    if name == "distinct":
        return torch.randperm(max(v, n), generator=g)[:n].remainder(v).reshape(b, s)
    if name == "runs":
        return (torch.arange(n) // 7).remainder(v).reshape(b, s)
    if name == "sparse":
        return (torch.randint(0, 4, (b, s), generator=g) * 9).remainder(v)
    if name.startswith("k") or name == "mix":
        # k tokens of each 32-token block share a chunk (k live rows per reshuffle call); mix alternates 1 and 8 by block
        out = torch.empty(n, dtype=torch.int64); nch = max(v // 32, 1)
        for blk in range(n // 32):
            k = (1 if blk % 2 else 8) if name == "mix" else int(name[1:])
            sizes = [k] * (32 // k) + ([32 % k] if 32 % k else [])
            ch = torch.randperm(nch, generator=g)[: len(sizes)].tolist()
            ch += [ch[-1]] * (len(sizes) - len(ch))
            toks = []
            for c, m in zip(ch, sizes):
                toks += [min(c * 32 + r, v - 1) for r in torch.randperm(32, generator=g)[:m].tolist()]
            out[blk * 32 : blk * 32 + 32] = torch.tensor(toks)[torch.randperm(32, generator=g)]
        return out.reshape(b, s)
    raise ValueError(name)


try:
    for b, s, e, v in ((2, 64, 160, 96), (3, 32, 384, 320), (2, 1024, 512, 3200), (2, 64, 160, 9), (1, 32, 64, 33), (4, 128, 256, 40), (2, 128, 96, 30522)):
        for pat in ("random", "one", "distinct", "runs", "sparse", "k1", "k2", "k3", "k4", "k5", "k8", "mix"):
            for idt in (ttnn.uint32, ttnn.bfloat16):
                if idt == ttnn.bfloat16 and v > 256:
                    continue
                for odt in (ttnn.bfloat16, ttnn.bfloat8_b):
                    g = torch.Generator().manual_seed(1234)
                    idx = pattern(pat, b, s, v, g)
                    w = torch.randn((v, e), generator=g)
                    grad = torch.randn((1, 1, b * s, e), generator=g)
                    ti = ttnn.from_torch(idx, dtype=idt, device=dev)
                    tw = ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
                    tg = ttnn.from_torch(grad, dtype=odt, layout=ttnn.TILE_LAYOUT, device=dev)
                    y = ttnn.to_torch(ttnn.embedding_bw(ti, tw, tg, dtype=odt))
                    res[f"{b}x{s}x{e}x{v}|{pat}|{idt}|{odt}"] = h(y.to(torch.float32))
        print(b, s, e, v, "done", flush=True)
finally:
    ttnn.close_device(dev)
json.dump(res, open(sys.argv[1], "w"), indent=0)
print("saved", len(res))
