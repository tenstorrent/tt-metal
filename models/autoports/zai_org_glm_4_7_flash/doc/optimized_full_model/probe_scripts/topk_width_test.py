import time

import torch

import ttnn

device = ttnn.open_device(device_id=0, l1_small_size=32768)
try:
    B = 32
    vocab = 154880
    x = torch.randn(1, 1, B, vocab, dtype=torch.bfloat16)
    x_dev = ttnn.from_torch(x, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    def bench(chunks, iters=32):
        chunk_w = vocab // chunks
        xs = [ttnn.slice(x_dev, [0, 0, 0, i * chunk_w], [1, 1, B, (i + 1) * chunk_w]) for i in range(chunks)]
        for xi in xs:
            xi_c = ttnn.to_layout(xi, ttnn.TILE_LAYOUT)
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            vals_list, idx_list = [], []
            for xi in xs:
                v, idx = ttnn.topk(xi, k=32, dim=-1, largest=True, stable=False)
                vals_list.append(v)
                idx_list.append(idx)
            if chunks > 1:
                vcat = ttnn.concat(vals_list, dim=3)
                icat = ttnn.concat(idx_list, dim=3)
                ttnn.deallocate(vcat)
                ttnn.deallocate(icat)
            for v, idx in zip(vals_list, idx_list):
                ttnn.deallocate(v)
                ttnn.deallocate(idx)
        ttnn.synchronize_device(device)
        t1 = time.perf_counter()
        for xi in xs:
            ttnn.deallocate(xi)
        return (t1 - t0) / iters * 1000

    for chunks in [4, 2, 1]:
        try:
            ms = bench(chunks)
            print(f"RESULT chunks={chunks} width={vocab//chunks} ms={ms:.4f}")
        except Exception as e:
            print(f"RESULT chunks={chunks} FAILED: {e}")
finally:
    ttnn.close_device(device)
