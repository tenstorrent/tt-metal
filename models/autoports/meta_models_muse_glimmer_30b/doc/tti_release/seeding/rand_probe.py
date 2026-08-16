"""Device-level probe: does ttnn.manual_seed + ttnn.sampling give every user the same
draw when every user is given the SAME seed and the SAME (near-uniform) distribution?

Serving evidence says slots 0 and 22 disagree with the other 30.  This isolates that to
the two ops, with no model in the loop.

Usage: python /tmp/rand_probe.py [num_seeds]
"""
import collections
import sys

import torch

import ttnn

N = 32
W = 64


def build(device):
    # Near-uniform top-32: distinct, ordered values, flattened by a small temp (=1/T).
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
    k_tensor = ttnn.from_torch(torch.tensor([32] * N), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    p_tensor = ttnn.from_torch(
        torch.tensor([1.0] * N), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    temp_tensor = ttnn.from_torch(
        torch.tensor([0.02] * N), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    user_ids = ttnn.from_torch(
        torch.arange(N, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    return values, indices, k_tensor, p_tensor, temp_tensor, user_ids


def one(device, seed, values, indices, k_tensor, p_tensor, temp_tensor, user_ids, filler=None):
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
    return ttnn.to_torch(out).reshape(-1).tolist()[:N]


def main():
    num_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    device = ttnn.open_device(device_id=0)
    try:
        values, indices, k_tensor, p_tensor, temp_tensor, user_ids = build(device)
        bad = collections.Counter()
        rows = []
        for s in range(1, num_seeds + 1):
            got = one(device, s * 7919 % 1000000 + 1, values, indices, k_tensor, p_tensor, temp_tensor, user_ids)
            mode = collections.Counter(got).most_common(1)[0][0]
            odd = [i for i, v in enumerate(got) if v != mode]
            rows.append((s, mode, odd, [got[i] for i in odd]))
            for i in odd:
                bad[i] += 1
        n_bad = sum(1 for _, _, odd, _ in rows if odd)
        print(f"seeds tried: {num_seeds}, seeds with >=1 disagreeing user: {n_bad}")
        print(f"disagreement count per user slot: {dict(sorted(bad.items()))}")
        for s, mode, odd, vals in rows[:12]:
            print(f"  seed#{s}: mode={mode} odd_users={odd} their_values={vals}")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
