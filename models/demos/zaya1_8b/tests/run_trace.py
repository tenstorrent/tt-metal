"""Validate + benchmark ttnn-traced decode vs non-traced (token-exact + speed).
Run: TT_DEVICE=1 /home/yito/work/run_zaya.sh python models/demos/zaya1_8b/tests/run_trace.py
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
    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape((1, 1)), trace_region_size=200000000)
    device.enable_program_cache()
    try:
        print("[trace] building model...", flush=True)
        model = ZayaModel(device, w=ZayaWeights(), verbose=False)

        print("[trace] non-traced generate...", flush=True)
        ref = model.generate(ids, N)                         # non-traced (validated path)

        print("[trace] TracedGenerator init...", flush=True)
        tg = TracedGenerator(model, device, MAX=64)
        print("[trace] prefill (populate buffers)...", flush=True)
        first = tg.prefill(ids)
        print(f"[trace] prefill done, first={first}; capturing trace...", flush=True)
        tg.capture(first)
        print("[trace] capture done; stepping...", flush=True)
        got = [first]
        cur = first
        for s in range(N - 1):
            cur = tg.step(cur)
            got.append(cur)
            print(f"[trace] step {s} -> {cur}", flush=True)

        print(f"  non-traced: {ref}")
        print(f"  traced    : {got}")
        print(f"  [{'PASS' if ref == got else 'FAIL'}] token-exact: traced == non-traced")

        # timing: traced steps (cache already warm/captured)
        t = time.time()
        cur = got[-1]
        for _ in range(N):
            cur = tg.step(cur)
        tr = (time.time() - t) / N
        print(f"\n=== traced decode: {tr*1000:.1f} ms/tok ({1/tr:.2f} tok/s) ===")
    finally:
        ttnn.close_mesh_device(device)

    raise SystemExit(0 if ref == got else 1)


if __name__ == "__main__":
    main()
