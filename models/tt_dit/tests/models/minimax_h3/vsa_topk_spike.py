# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""R3 topk spike (VSA_SCOPE.md): ttnn.topk at k~200 over ~1802 columns.

Production selection shape per device: scores [1, 14, 226, 1808] bf16
(14 local heads, 226 q-tiles/shard, 1808 global tiles), k = 179 at
sparsity 0.9 (rounded up to 192 for the k%16 route; slice back after).
Run: TT_METAL_HOME=... python vsa_topk_spike.py
"""

import time

import torch

import ttnn

HEADS, ROWS, COLS = 14, 226, 1808
K_EXACT = 179
K_PAD = 192  # topk_large_indices needs k % 16 == 0


def check(name, values_tt, indices_tt, scores, k):
    ref_vals, _ = torch.topk(scores.float(), k, dim=-1)
    got_vals = values_tt[..., :k].float()
    # bf16 ties can reorder indices; compare the selected VALUES sorted per row, and count
    # per-row index-set mismatches beyond ties.
    ok_vals = torch.allclose(torch.sort(got_vals, -1).values, torch.sort(ref_vals, -1).values, atol=0.0)
    # set comparison on a sample of rows
    mism = 0
    for h in range(0, HEADS, 5):
        for r in range(0, ROWS, 50):
            got = set(indices_tt[0, h, r, :k].tolist())
            top = torch.topk(scores[0, h, r].float(), k).indices.tolist()
            ref = set(top)
            if got != ref:
                # tolerate ties: all mismatched members must have equal score to the kth value
                kth = scores[0, h, r].float()[top[-1]].item()
                diff = got.symmetric_difference(ref)
                if any(abs(scores[0, h, r].float()[i].item() - kth) > 0 for i in diff):
                    mism += 1
    print(f"{name}: values_match={ok_vals}, sampled index-set mismatches (beyond ties)={mism}")


def bench(fn, n=10):
    fn()  # compile
    t0 = time.perf_counter()
    for _ in range(n):
        out = fn()
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) / n * 1e3, out


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    try:
        torch.manual_seed(0)
        scores = torch.randn(1, HEADS, ROWS, COLS, dtype=torch.bfloat16)

        # Route A: composite ttnn.topk on TILE input
        tt_scores_tile = ttnn.from_torch(scores, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        def run_tile():
            return ttnn.topk(tt_scores_tile, k=K_PAD, dim=-1, largest=True)
        try:
            ms, (vals, idxs) = bench(run_tile)
            v, i = ttnn.to_torch(vals), ttnn.to_torch(idxs)
            check("ttnn.topk(TILE)", v, i, scores, K_EXACT)
            print(f"ttnn.topk(TILE): {ms:.2f} ms/iter, out dtypes {vals.dtype}/{idxs.dtype}")
        except Exception as e:
            print(f"ttnn.topk(TILE) FAILED: {type(e).__name__}: {str(e)[:500]}")

        # Route B: ttnn.experimental.topk_large_indices on ROW_MAJOR input
        tt_scores_rm = ttnn.from_torch(scores, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)
        def run_rm():
            return ttnn.experimental.topk_large_indices(tt_scores_rm, k=K_PAD)
        try:
            ms, out = bench(run_rm)
            outs = out if isinstance(out, (list, tuple)) else [out]
            print(f"topk_large_indices: {ms:.2f} ms/iter, outputs: {[str(o.shape) + ' ' + str(o.dtype) for o in outs]}")
            if len(outs) == 2:
                v, i = ttnn.to_torch(outs[0]), ttnn.to_torch(outs[1])
            else:
                i = ttnn.to_torch(outs[0])
                v = torch.gather(scores, -1, i.to(torch.int64))
            check("topk_large_indices", v, i, scores, K_EXACT)
        except Exception as e:
            print(f"topk_large_indices FAILED: {type(e).__name__}: {str(e)[:500]}")
    finally:
        ttnn.close_device(device)
    print("SPIKE_DONE")
