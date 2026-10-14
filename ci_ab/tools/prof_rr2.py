# Device time of ttnn.embedding_bw (reshuffle_rows) for every case in one process, under
# `python -m tracy -r -p --no-web-server -o <dir> prof_rr2.py <cases>`; a signpost names each case before its
# R3_NMEAS measured calls (2 warm-up calls first). kN cases: 8x128 tokens over vocab 30522 with exactly N live rows in
# every 32-row chunk the reader hands to compute (N tokens of a 32-token block share a chunk).
import gc, os, sys
import torch
import ttnn
try:
    from tracy import signpost
except ImportError:
    signpost = None

N_WARM, N_MEAS = 2, int(os.environ.get("R3_NMEAS", "6"))


def rand_idx(b, s, v, g):
    return torch.randint(0, v, (b, s), generator=g)


def k_idx(b, s, v, k, g):
    nchunks = v // 32
    out = torch.empty((b, s), dtype=torch.int64)
    for bi in range(b):
        for blk in range(s // 32):
            sizes = [k] * (32 // k) + ([32 % k] if 32 % k else [])
            chunks = torch.randperm(nchunks, generator=g)[: len(sizes)]
            toks = []
            for c, n in zip(chunks.tolist(), sizes):
                rows = torch.randperm(32, generator=g)[:n]
                toks += [c * 32 + r for r in rows.tolist()]
            perm = torch.randperm(32, generator=g)
            out[bi, blk * 32 : blk * 32 + 32] = torch.tensor(toks)[perm]
    return out


def case(b, s, e, v, k=0):
    def setup(d):
        g = torch.Generator().manual_seed(1234)
        idx = k_idx(b, s, v, k, g) if k else rand_idx(b, s, v, g)
        ti = ttnn.from_torch(idx, dtype=ttnn.uint32, device=d)
        tw = ttnn.from_torch(torch.randn((v, e), generator=g), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=d)
        tg = ttnn.from_torch(torch.randn((1, 1, b * s, e), generator=g), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=d)
        return (ti, tw, tg)

    def run(d, ti, tw, tg):
        return ttnn.embedding_bw(ti, tw, tg, dtype=ttnn.bfloat16)

    return setup, run


CASES = {"emb_2x1024x4096x3200": case(2, 1024, 4096, 3200), "emb_1x2048x2048x32000": case(1, 2048, 2048, 32000),
         "emb_8x128x768x30522": case(8, 128, 768, 30522), "emb_2x64x160x96": case(2, 64, 160, 96)}
for v in (256, 320):
    CASES[f"emb_2x64x160x{v}"] = case(2, 64, 160, v)
    CASES[f"emb_8x128x768x{v}"] = case(8, 128, 768, v)
for k in (1, 2, 3, 4, 8):
    CASES[f"k{k}_8x128x768x30522"] = case(8, 128, 768, 30522, k)


def main():
    names = sys.argv[1:] or list(CASES)
    dev = ttnn.open_device(device_id=0)
    try:
        for name in names:
            setup, run = CASES[name]
            gc.collect()
            try:
                args = setup(dev)
                for _ in range(N_WARM):
                    run(dev, *args)
                ttnn.synchronize_device(dev)
                signpost(header=name)
                for _ in range(N_MEAS):
                    run(dev, *args)
                ttnn.synchronize_device(dev)
                print(f"CASE {name} ok", flush=True)
                del args
            except Exception as e:  # noqa: BLE001
                print(f"CASE {name} FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
        ttnn.ReadDeviceProfiler(dev)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
