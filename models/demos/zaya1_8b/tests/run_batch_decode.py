"""Phase 1 validation + throughput bench for batch-on-M decode (single P150a).
Correctness: B=1 regression vs single-user golden; B=2 same-prompt (both==golden);
B=2 different same-length prompts (each==its own single-user golden). Then bench B=1/2/4/8.
Run: TT_DEVICE=1 /home/yito/work/run_zaya.sh python models/demos/zaya1_8b/tests/run_batch_decode.py
"""
import os
import time
import torch
import ttnn

from models.demos.zaya1_8b.tt.model_args import ZayaWeights
from models.demos.zaya1_8b.tt.model import ZayaModel

GOLDEN = os.path.join(os.path.dirname(__file__), "..", "reference", "golden")
N = 6


def col(steps, u):
    return [s[u] for s in steps]


def main():
    ids = torch.load(os.path.join(GOLDEN, "inputs.pt"), weights_only=False)["input_ids"].to(torch.int32)
    # a 2nd prompt of the SAME length (alter one token, keep in-vocab) for per-user correctness
    ids2 = ids.clone()
    ids2[0, 1] = (int(ids2[0, 1]) + 1000) % 100000
    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape((1, 1)), trace_region_size=0)
    device.enable_program_cache()
    try:
        model = ZayaModel(device, w=ZayaWeights(), verbose=False)

        print("[batch] single-user goldens...", flush=True)
        g1 = model.generate(ids, N)
        g2 = model.generate(ids2, N)
        print(f"  golden(ids)  = {g1}")
        print(f"  golden(ids2) = {g2}")

        print("[batch] B=1 regression...", flush=True)
        b1 = model.generate_batched(ids, 1, N)
        b1u = col(b1, 0)
        print(f"  B=1 user0    = {b1u}  [{'PASS' if b1u == g1 else 'FAIL'}]")

        print("[batch] B=2 same prompt...", flush=True)
        b2 = model.generate_batched(ids, 2, N)
        ok_same = all(col(b2, u) == g1 for u in range(2))
        print(f"  B=2 u0={col(b2,0)} u1={col(b2,1)}  [{'PASS' if ok_same else 'FAIL'}] both==golden")

        print("[batch] B=2 different prompts (same len)...", flush=True)
        bm = model.generate_batched_multi([ids, ids2], N)
        ok0, ok1 = col(bm, 0) == g1, col(bm, 1) == g2
        print(f"  u0={col(bm,0)} [{'PASS' if ok0 else 'FAIL'}]  u1={col(bm,1)} [{'PASS' if ok1 else 'FAIL'}]")

        allok = b1u == g1 and ok_same and ok0 and ok1
        print(f"\n  [{'PASS' if allok else 'FAIL'}] batched decode token-exact per user")

        # ---- throughput bench (decode steps only) ----
        print("\n[batch] throughput (n=6 steps each, after warmup):", flush=True)
        from models.demos.zaya1_8b.tt.cache import ZayaCache
        for B in (1, 2, 4, 8):
            cache = ZayaCache()
            first = model.prefill(ids, cache)
            model._replicate_cache(cache, B)
            cur = [first] * B
            for _ in range(2):                      # warmup
                cur = model.decode_step_batched(cur, cache)
            t = time.time()
            for _ in range(N):
                cur = model.decode_step_batched(cur, cache)
            dt = (time.time() - t) / N
            print(f"  B={B}: {dt*1000:6.1f} ms/step | {1/dt:6.2f} steps/s | "
                  f"{B/dt:7.2f} tok/s aggregate | {1/dt:6.2f} tok/s/user")
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
