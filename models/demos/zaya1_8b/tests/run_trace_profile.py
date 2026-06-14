"""Profile traced decode step breakdown + validate trace at larger MAX.
Builds the model ONCE, then:
  (A) profile step() sub-parts at MAX=64 (embed-write / pos-write / execute_trace / lm_head+argmax)
  (B) re-capture at MAX=256, token-exact check vs non-traced, bench ms/tok
Run: TT_DEVICE=1 /home/yito/work/run_zaya.sh python models/demos/zaya1_8b/tests/run_trace_profile.py
"""
import os
import time
import torch
import ttnn

from models.demos.zaya1_8b.tt.model_args import ZayaWeights
from models.demos.zaya1_8b.tt.model import ZayaModel
from models.demos.zaya1_8b.tt.trace import TracedGenerator

GOLDEN = os.path.join(os.path.dirname(__file__), "..", "reference", "golden")


def sync(device):
    ttnn.synchronize_device(device)


def profile_steps(tg, device, n=8):
    """Time the 4 sub-parts of a decode step (each isolated with a device sync)."""
    acc = {"emb": 0.0, "pos": 0.0, "exec": 0.0, "head": 0.0}
    # fixed dummy token stream just for timing (correctness validated elsewhere)
    tok = 100
    for _ in range(n):
        t = time.time(); tg._write_token(tok); sync(device); acc["emb"] += time.time() - t
        t = time.time(); tg._set_pos_inputs(tg.ts.pos); sync(device); acc["pos"] += time.time() - t
        t = time.time(); ttnn.execute_trace(device, tg.trace_id, cq_id=0, blocking=True); acc["exec"] += time.time() - t
        tg.ts.pos += 1
        t = time.time()
        logits = tg.model.lm_head(tg.ts.hout)
        tok = int(ttnn.to_torch(ttnn.argmax(logits, dim=-1)).reshape(-1)[0])
        acc["head"] += time.time() - t
    for k in acc:
        acc[k] = acc[k] / n * 1000.0
    return acc


def main():
    ids = torch.load(os.path.join(GOLDEN, "inputs.pt"), weights_only=False)["input_ids"].to(torch.int32)
    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape((1, 1)), trace_region_size=300000000)
    device.enable_program_cache()
    try:
        print("[prof] building model...", flush=True)
        model = ZayaModel(device, w=ZayaWeights(), verbose=False)

        # ---- (A) profile at MAX=64 ----
        print("[prof] MAX=64: prefill+capture...", flush=True)
        tg = TracedGenerator(model, device, MAX=64)
        first = tg.prefill(ids)
        tg.capture(first)
        # a few real steps to warm, then profile
        cur = first
        for _ in range(3):
            cur = tg.step(cur)
        acc = profile_steps(tg, device, n=8)
        tot = sum(acc.values())
        print(f"\n=== step breakdown (MAX=64), n=8 avg ===")
        for k in ("emb", "pos", "exec", "head"):
            print(f"  {k:5s}: {acc[k]:7.2f} ms  ({100*acc[k]/tot:4.1f}%)")
        print(f"  TOTAL: {tot:7.2f} ms/tok")

        # ---- (B) longer context: MAX=256 ----
        N = 6
        print("\n[prof] non-traced reference...", flush=True)
        ref = model.generate(ids, N)
        print("[prof] MAX=256: prefill+capture...", flush=True)
        tg2 = TracedGenerator(model, device, MAX=256)
        f2 = tg2.prefill(ids)
        tg2.capture(f2)
        got = [f2]; cur = f2
        for _ in range(N - 1):
            cur = tg2.step(cur); got.append(cur)
        print(f"  non-traced: {ref}")
        print(f"  traced(256): {got}")
        print(f"  [{'PASS' if ref == got else 'FAIL'}] token-exact @ MAX=256")
        t = time.time(); cur = got[-1]
        for _ in range(N):
            cur = tg2.step(cur)
        tr = (time.time() - t) / N
        print(f"=== traced decode @ MAX=256: {tr*1000:.1f} ms/tok ({1/tr:.2f} tok/s) ===")
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
