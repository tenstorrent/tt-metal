"""Probe 4: is a single ttnn.typecast(int32 -> bfloat16) between manual_seed and
sampling enough to destroy the per-core RNG state?"""
import collections
import sys

import torch

import ttnn

N = 32
W = 64


def main():
    which = sys.argv[1]
    num_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 6
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
        i32 = ttnn.from_torch(
            torch.arange(0, W, dtype=torch.int32).expand(1, 1, N, W).contiguous(),
            dtype=ttnn.int32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        bf = ttnn.from_torch(torch.zeros(1, 1, N, W), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        u32 = ttnn.from_torch(
            torch.arange(0, W, dtype=torch.int64).expand(1, 1, N, W).contiguous().to(torch.int32),
            dtype=ttnn.uint32,
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

        fillers = {
            "none": lambda: None,
            "i32_to_bf16": lambda: ttnn.typecast(i32, ttnn.bfloat16),
            "bf16_to_i32": lambda: ttnn.typecast(bf, ttnn.int32),
            "u32_to_bf16": lambda: ttnn.typecast(u32, ttnn.bfloat16),
            "i32_to_u32": lambda: ttnn.typecast(i32, ttnn.uint32),
            "bf16_to_bf16_add": lambda: ttnn.add(bf, bf),
            "bf16_abs": lambda: ttnn.abs(bf),
            "bf16_exp": lambda: ttnn.exp(bf),
        }
        f = fillers[which]

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
            f()
            out = ttnn.sampling(values, indices, k=k_tensor, p=p_tensor, temp=temp_tensor)
            got = ttnn.to_torch(out).reshape(-1).tolist()[:N]
            mode = collections.Counter(got).most_common(1)[0][0]
            odd = [i for i, v in enumerate(got) if v != mode]
            if odd:
                n_bad += 1
                for i in odd:
                    bad[i] += 1
        print(f"RESULT filler={which} seeds={num_seeds} bad_seeds={n_bad} per_user={dict(sorted(bad.items()))}")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
