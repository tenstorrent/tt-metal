"""Phase 1 (traced): batched ttnn-trace decode — token-exact per user + throughput.
For each B in {1,2,4,8}: capture a B-user trace, check every user's stream == the
single-user non-traced golden (same prompt), and bench ms/step + aggregate tok/s.
Run: TT_DEVICE=1 /home/yito/work/run_zaya.sh python models/demos/zaya1_8b/tests/run_trace_batch.py
"""
import os
import time
import torch
import ttnn

from models.demos.zaya1_8b.tt.model_args import ZayaWeights
from models.demos.zaya1_8b.tt.model import ZayaModel
from models.demos.zaya1_8b.tt.trace import TracedGenerator

GOLDEN = os.path.join(os.path.dirname(__file__), "..", "reference", "golden")
N = 6


def main():
    ids = torch.load(os.path.join(GOLDEN, "inputs.pt"), weights_only=False)["input_ids"].to(torch.int32)
    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape((1, 1)), trace_region_size=300000000)
    device.enable_program_cache()
    try:
        model = ZayaModel(device, w=ZayaWeights(), verbose=False)
        golden = model.generate(ids, N)
        print(f"[trace-batch] non-traced golden = {golden}", flush=True)

        rows = []
        for B in (1, 2, 4, 8):
            tg = TracedGenerator(model, device, MAX=64, B=B)
            steps = tg.generate(ids, N)                       # list of per-step (scalar if B==1 else [B])
            streams = ([[s for s in steps]] if B == 1
                       else [[steps[t][u] for t in range(N)] for u in range(B)])
            ok = all(st == golden for st in streams)
            # bench: time N steps (cache already captured/warm)
            cur = steps[-1]
            t = time.time()
            for _ in range(N):
                cur = tg.step(cur)
            dt = (time.time() - t) / N
            rows.append((B, dt, ok))
            print(f"[trace-batch] B={B}: [{'PASS' if ok else 'FAIL'}] token-exact | "
                  f"{dt*1000:6.1f} ms/step | {1/dt:6.2f} steps/s | {B/dt:7.2f} tok/s aggregate", flush=True)

        allok = all(ok for _, _, ok in rows)
        print(f"\n  [{'PASS' if allok else 'FAIL'}] all batches token-exact vs golden")
        base = rows[0][1]
        print("  scaling (vs B=1 ms/step): " + ", ".join(f"B{b}:{dt/base:.2f}x" for b, dt, _ in rows))
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
