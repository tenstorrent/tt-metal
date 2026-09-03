import time

import torch

import ttnn

device = ttnn.open_device(device_id=0, l1_small_size=32768)
try:
    B = 32
    vocab = 154880
    torch.manual_seed(0)
    x = torch.randn(1, 1, B, vocab, dtype=torch.bfloat16)
    x_dev = ttnn.from_torch(x, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ref_vals, ref_idx = torch.topk(x.float(), k=32, dim=-1, largest=True)

    def bench(chunks, iters=32, warmup=5):
        chunk_w = vocab // chunks
        xs = [ttnn.slice(x_dev, [0, 0, 0, i * chunk_w], [1, 1, B, (i + 1) * chunk_w]) for i in range(chunks)]

        def one_pass():
            vals_list, idx_list = [], []
            for xi in xs:
                v, idx = ttnn.topk(xi, k=32, dim=-1, largest=True, stable=False)
                vals_list.append(v)
                idx_list.append(idx)
            if chunks > 1:
                vcat = ttnn.concat(vals_list, dim=3)
                icat = ttnn.concat(idx_list, dim=3)
            else:
                vcat, icat = vals_list[0], idx_list[0]
            return vcat, icat, vals_list, idx_list

        # warmup (uncounted) -- compiles programs for this shape
        for _ in range(warmup):
            vcat, icat, vals_list, idx_list = one_pass()
            for v, idx in zip(vals_list, idx_list):
                ttnn.deallocate(v)
                ttnn.deallocate(idx)
            if chunks > 1:
                ttnn.deallocate(vcat)
                ttnn.deallocate(icat)
        ttnn.synchronize_device(device)

        # correctness check on the last warmup pass's re-run (compare against torch)
        vcat, icat, vals_list, idx_list = one_pass()
        # merge per-chunk local index -> global index for correctness check
        got_idx = ttnn.to_torch(icat).to(torch.int64)
        got_vals = ttnn.to_torch(vcat).to(torch.float32)
        if chunks > 1:
            # icat indices are LOCAL to each chunk (0..chunk_w); offset by chunk id is embedded
            # in concat order but not in the index value itself unless the op handles it.
            pass
        # global top-32 by value should match torch's top-32 values (order/index mapping aside)
        got_top32_sorted = got_vals[0, 0, 0].sort(descending=True).values[:32]
        ref_top32_sorted = ref_vals[0, 0, 0].sort(descending=True).values[:32]
        val_match = torch.allclose(got_top32_sorted.float(), ref_top32_sorted.float(), atol=1e-2, rtol=1e-2)
        for v, idx in zip(vals_list, idx_list):
            ttnn.deallocate(v)
            ttnn.deallocate(idx)
        if chunks > 1:
            ttnn.deallocate(vcat)
            ttnn.deallocate(icat)

        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            vcat, icat, vals_list, idx_list = one_pass()
            for v, idx in zip(vals_list, idx_list):
                ttnn.deallocate(v)
                ttnn.deallocate(idx)
            if chunks > 1:
                ttnn.deallocate(vcat)
                ttnn.deallocate(icat)
        ttnn.synchronize_device(device)
        t1 = time.perf_counter()
        for xi in xs:
            ttnn.deallocate(xi)
        return (t1 - t0) / iters * 1000, val_match

    for chunks in [4, 2, 1, 8]:
        ms, ok = bench(chunks)
        print(f"RESULT chunks={chunks} width={vocab//chunks} ms_per_call={ms:.4f} values_match_torch_top32={ok}")
finally:
    ttnn.close_device(device)
