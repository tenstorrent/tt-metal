"""Probe 2: does an op running BETWEEN ttnn.manual_seed and ttnn.sampling destroy the
per-core RNG state that manual_seed just installed?

The serving pipeline (models/common/sampling/tt_sampling.py) calls
    manual_seed(...)  ->  _adjust_values_for_tiebreak(...)  ->  sampling(...)
and serving shows users 0 and 22 drawing a different random number from the other 30
given the identical seed.  Probe 1 showed manual_seed+sampling back to back is clean.
"""
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
    which = sys.argv[1] if len(sys.argv) > 1 else "none"
    num_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 20
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

        def f_max():
            ttnn.max(values, dim=3, keepdim=True)

        def f_mul():
            ttnn.multiply(values, 2.0)

        def f_typecast():
            ttnn.typecast(idx_tiled, ttnn.int32)

        def f_eq():
            m = ttnn.max(values, dim=3, keepdim=True)
            ttnn.eq(values, m)

        def f_untilize():
            ttnn.untilize(idx_tiled, use_multicore=True)

        def f_tiebreak():
            maxv = ttnn.max(values, dim=3, keepdim=True)
            is_max = ttnn.eq(values, maxv)
            not_max = ttnn.lt(values, maxv)
            abs_max = ttnn.abs(maxv)
            delta_scaled = ttnn.multiply(abs_max, TIEBREAK_DELTA_SCALE)
            delta = ttnn.add(delta_scaled, TIEBREAK_DELTA_FLOOR)
            idx = ttnn.typecast(idx_tiled, ttnn.int32)
            offset = ttnn.multiply(not_max, TIEBREAK_INDEX_SENTINEL)
            offset_i32 = ttnn.typecast(offset, ttnn.int32)
            masked_idx = ttnn.add(idx, offset_i32)
            greedy_i = ttnn.min(masked_idx, dim=3, keepdim=True)
            is_lowidx_i32 = ttnn.eq(idx, greedy_i)
            is_lowidx = ttnn.typecast(is_lowidx_i32, ttnn.bfloat16)
            is_winner = ttnn.multiply(is_max, is_lowidx)
            winner_gated = ttnn.multiply(is_winner, greedy_col)
            boost = ttnn.multiply(winner_gated, delta)
            ttnn.add(values, boost)

        fillers = {
            "none": None,
            "max": f_max,
            "mul": f_mul,
            "typecast": f_typecast,
            "eq": f_eq,
            "untilize": f_untilize,
            "tiebreak": f_tiebreak,
        }
        filler = fillers[which]

        bad = collections.Counter()
        n_bad = 0
        detail = []
        for s in range(1, num_seeds + 1):
            seed = s * 7919 % 1000000 + 1
            seeds = ttnn.from_torch(
                torch.full((N,), seed, dtype=torch.int64).to(torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
            )
            ttnn.manual_seed(seeds=seeds, device=device, user_ids=user_ids)
            if filler is not None:
                filler()
            out = ttnn.sampling(values, indices, k=k_tensor, p=p_tensor, temp=temp_tensor)
            got = ttnn.to_torch(out).reshape(-1).tolist()[:N]
            mode = collections.Counter(got).most_common(1)[0][0]
            odd = [i for i, v in enumerate(got) if v != mode]
            if odd:
                n_bad += 1
                for i in odd:
                    bad[i] += 1
            if len(detail) < 6:
                detail.append((s, mode, odd, [got[i] for i in odd]))
        print(f"RESULT filler={which} seeds={num_seeds} seeds_with_disagreement={n_bad}")
        print(f"RESULT per-user disagreement counts: {dict(sorted(bad.items()))}")
        for d in detail:
            print(f"RESULT   seed#{d[0]} mode={d[1]} odd={d[2]} vals={d[3]}")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
