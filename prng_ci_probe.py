# CI experiment for #58772 (not for check-in): the output words of every ttnn op that seeds the Blackhole PRNG, as one
# sha256 per case, so that main and the cycle-counter seed wait can be compared on CI's cards. Fixed seeds, several
# shapes; the manual_seed cases seed in one program and draw in the next. Each case runs twice to show it is stable.
import hashlib

import torch
import ttnn

SEEDS = [1, 1234, 0x7FFFFFFF]
SHAPES = [(1, 1, 32, 32), (1, 1, 256, 512)]


def digest(t):
    a = ttnn.to_torch(t)
    if a.dtype == torch.bfloat16:
        a = a.view(torch.int16)
    elif a.dtype == torch.float32:
        a = a.view(torch.int32)
    return hashlib.sha256(a.contiguous().numpy().tobytes()).hexdigest()[:16]


def ids(shape):
    return f"{shape[-2]}x{shape[-1]}"


def cases(dev):
    for shape in SHAPES:
        for seed in SEEDS:
            yield f"rand_{ids(shape)}_{seed}", lambda: ttnn.rand(shape, dtype=ttnn.bfloat16, device=dev, seed=seed)

            def uni():
                x = ttnn.from_torch(torch.zeros(shape), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
                ttnn.uniform(x, -1.0, 1.0, seed)
                return x

            yield f"uniform_{ids(shape)}_{seed}", uni

            def bern():
                p = ttnn.from_torch(torch.full(shape, 0.5), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
                return ttnn.bernoulli(p, seed, dtype=ttnn.float32)

            yield f"bernoulli_{ids(shape)}_{seed}", bern

            def drop():
                torch.manual_seed(0)
                x = ttnn.from_torch(torch.randn(shape, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
                return ttnn.experimental.dropout(x, probability=0.5, scale=2.0, seed=seed)

            yield f"dropout_{ids(shape)}_{seed}", drop
    for seed in SEEDS:
        yield f"randn_32x64_{seed}", lambda: ttnn.randn((1, 1, 32, 64), dtype=ttnn.bfloat16, device=dev, seed=seed)

    def sampling_inputs():
        torch.manual_seed(0)
        shape = [1, 1, 32, 256]
        vals = ttnn.from_torch(torch.randn(shape), device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        idx = ttnn.from_torch(torch.arange(0, 256, dtype=torch.int32).expand(shape), device=dev, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        k = ttnn.from_torch(torch.tensor([10, 15, 20, 25, 30] * 6 + [10, 20]), device=dev, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        p = ttnn.from_torch(torch.tensor([0.0, 0.3, 0.5, 0.7, 0.9] * 6 + [0.1, 0.8]), device=dev, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        temp = ttnn.from_torch(torch.ones(32), device=dev, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        return vals, idx, k, p, temp

    for seed in SEEDS:
        def samp():
            vals, idx, k, p, temp = sampling_inputs()
            return ttnn.sampling(vals, idx, k=k, p=p, temp=temp, seed=seed)

        yield f"sampling_{seed}", samp

        def mseed():
            vals, idx, k, p, temp = sampling_inputs()
            ttnn.manual_seed(seeds=seed, device=dev)
            return ttnn.sampling(vals, idx, k=k, p=p, temp=temp)

        yield f"manualseed_sampling_{seed}", mseed


def main():
    dev = ttnn.open_device(device_id=0)
    grid = dev.compute_with_storage_grid_size()
    print(f"PRNG_PROBE grid {grid.x}x{grid.y}")
    for name, fn in cases(dev):
        h1, h2 = digest(fn()), digest(fn())
        print(f"PRNG_PROBE {name} {h1} {'stable' if h1 == h2 else 'UNSTABLE ' + h2}")
    ttnn.close_device(dev)


if __name__ == "__main__":
    main()
