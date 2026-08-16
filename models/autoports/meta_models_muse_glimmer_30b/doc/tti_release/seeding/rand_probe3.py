"""Probe 3: bisect the tie-break chain -- which op between manual_seed and sampling
destroys the per-core RNG state?  Runs the first N ops of the chain."""
import collections
import sys

import torch

import ttnn

N = 32
W = 64
TIEBREAK_DELTA_SCALE = 1.0 / 512.0
TIEBREAK_DELTA_FLOOR = 1e-6
TIEBREAK_INDEX_SENTINEL = 2**24


def main():
    nops = int(sys.argv[1])
    num_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    device = ttnn.open_device(device_id=0)
    try:
        vals = torch.arange(W, dtype=torch.float32) * -0.5
        values = ttnn.from_torch(
            vals.expand(1, 1, N, W).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        indices = ttnn.from_torch(
            torch.arange(0, W, dtype=torch.int32).expand(1, 1, N, W).contiguous(),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        idx_tiled = ttnn.from_torch(
            torch.arange(0, W, dtype=torch.int32).expand(1, 1, N, W).contiguous(),
            dtype=ttnn.int32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        k_tensor = ttnn.from_torch(
            torch.tensor([32] * N), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        p_tensor = ttnn.from_torch(
            torch.tensor([1.0] * N), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        temp_tensor = ttnn.from_torch(
            torch.tensor([0.02] * N), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        user_ids = ttnn.from_torch(
            torch.arange(N, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        greedy_col = ttnn.from_torch(
            torch.zeros(1, 1, N, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

        names = [
            "max",
            "eq",
            "lt",
            "abs",
            "mul_scale",
            "add_floor",
            "typecast_idx",
            "mul_sentinel",
            "typecast_off",
            "add_idx",
            "min",
            "eq_idx",
            "typecast_bf16",
            "mul_winner",
            "mul_greedycol",
            "mul_delta",
            "add_boost",
        ]

        def chain(limit):
            st = {}
            i = 0

            def step(fn):
                nonlocal i
                if i >= limit:
                    raise StopIteration
                r = fn()
                i += 1
                return r

            try:
                st["maxv"] = step(lambda: ttnn.max(values, dim=3, keepdim=True))
                st["is_max"] = step(lambda: ttnn.eq(values, st["maxv"]))
                st["not_max"] = step(lambda: ttnn.lt(values, st["maxv"]))
                st["abs_max"] = step(lambda: ttnn.abs(st["maxv"]))
                st["ds"] = step(lambda: ttnn.multiply(st["abs_max"], TIEBREAK_DELTA_SCALE))
                st["delta"] = step(lambda: ttnn.add(st["ds"], TIEBREAK_DELTA_FLOOR))
                st["idx"] = step(lambda: ttnn.typecast(idx_tiled, ttnn.int32))
                st["off"] = step(lambda: ttnn.multiply(st["not_max"], TIEBREAK_INDEX_SENTINEL))
                st["off32"] = step(lambda: ttnn.typecast(st["off"], ttnn.int32))
                st["mi"] = step(lambda: ttnn.add(st["idx"], st["off32"]))
                st["gi"] = step(lambda: ttnn.min(st["mi"], dim=3, keepdim=True))
                st["il32"] = step(lambda: ttnn.eq(st["idx"], st["gi"]))
                st["il"] = step(lambda: ttnn.typecast(st["il32"], ttnn.bfloat16))
                st["iw"] = step(lambda: ttnn.multiply(st["is_max"], st["il"]))
                st["wg"] = step(lambda: ttnn.multiply(st["iw"], greedy_col))
                st["boost"] = step(lambda: ttnn.multiply(st["wg"], st["delta"]))
                st["adj"] = step(lambda: ttnn.add(values, st["boost"]))
            except StopIteration:
                pass

        bad = collections.Counter()
        n_bad = 0
        for s in range(1, num_seeds + 1):
            seed = s * 7919 % 1000000 + 1
            seeds = ttnn.from_torch(
                torch.full((N,), seed, dtype=torch.int64).to(torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
            )
            ttnn.manual_seed(seeds=seeds, device=device, user_ids=user_ids)
            chain(nops)
            out = ttnn.sampling(values, indices, k=k_tensor, p=p_tensor, temp=temp_tensor)
            got = ttnn.to_torch(out).reshape(-1).tolist()[:N]
            mode = collections.Counter(got).most_common(1)[0][0]
            odd = [i for i, v in enumerate(got) if v != mode]
            if odd:
                n_bad += 1
                for i in odd:
                    bad[i] += 1
        last = names[nops - 1] if 0 < nops <= len(names) else "-"
        print(
            f"RESULT nops={nops} (last={last}) seeds={num_seeds} bad_seeds={n_bad} per_user={dict(sorted(bad.items()))}"
        )
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
