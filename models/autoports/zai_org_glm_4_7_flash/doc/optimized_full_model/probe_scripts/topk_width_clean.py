import argparse
import time

import torch

import ttnn

ap = argparse.ArgumentParser()
ap.add_argument("--chunks", type=int, required=True)
ap.add_argument("--warmup", type=int, default=8)
ap.add_argument("--iters", type=int, default=32)
args = ap.parse_args()

device = ttnn.open_device(device_id=0, l1_small_size=32768)
try:
    B, vocab = 32, 154880
    torch.manual_seed(0)
    x = torch.randn(1, 1, B, vocab, dtype=torch.bfloat16)
    ref_top32 = x.float()[0, 0, 0].sort(descending=True).values[:32]
    x_dev = ttnn.from_torch(x, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    chunks = args.chunks
    chunk_w = vocab // chunks
    xs = [ttnn.slice(x_dev, [0, 0, 0, i * chunk_w], [1, 1, B, (i + 1) * chunk_w]) for i in range(chunks)]

    def one_pass():
        vals_list, idx_list = [], []
        for xi in xs:
            v, idx = ttnn.topk(xi, k=32, dim=-1, largest=True, stable=False)
            vals_list.append(v)
            idx_list.append(idx)
        vcat = ttnn.concat(vals_list, dim=3) if chunks > 1 else vals_list[0]
        icat = ttnn.concat(idx_list, dim=3) if chunks > 1 else idx_list[0]
        return vcat, icat, vals_list, idx_list

    for i in range(args.warmup):
        vcat, icat, vl, il = one_pass()
        for v, idx in zip(vl, il):
            ttnn.deallocate(v)
            ttnn.deallocate(idx)
        if chunks > 1:
            ttnn.deallocate(vcat)
            ttnn.deallocate(icat)
    ttnn.synchronize_device(device)

    vcat, icat, vl, il = one_pass()
    got = ttnn.to_torch(vcat).float()[0, 0, 0].sort(descending=True).values[:32]
    ok = torch.allclose(got, ref_top32, atol=2e-2, rtol=2e-2)
    for v, idx in zip(vl, il):
        ttnn.deallocate(v)
        ttnn.deallocate(idx)
    if chunks > 1:
        ttnn.deallocate(vcat)
        ttnn.deallocate(icat)
    ttnn.synchronize_device(device)

    t0 = time.perf_counter()
    for _ in range(args.iters):
        vcat, icat, vl, il = one_pass()
        for v, idx in zip(vl, il):
            ttnn.deallocate(v)
            ttnn.deallocate(idx)
        if chunks > 1:
            ttnn.deallocate(vcat)
            ttnn.deallocate(icat)
    ttnn.synchronize_device(device)
    t1 = time.perf_counter()
    print(f"RESULT chunks={chunks} width={chunk_w} ms_per_call={(t1-t0)/args.iters*1000:.4f} correct={ok}")
finally:
    ttnn.close_device(device)
