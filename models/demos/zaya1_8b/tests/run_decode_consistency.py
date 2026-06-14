"""Validate incremental decode (KV+conv+prev_hs cache) against the validated
no-cache prefill path: cached prefill and each decode step must produce the same
token a full recompute would.
Run: TT_DEVICE=1 /home/yito/work/run_zaya.sh python models/demos/zaya1_8b/tests/run_decode_consistency.py
"""
import os
import torch
import ttnn

from models.demos.zaya1_8b.tt.model_args import ZayaWeights
from models.demos.zaya1_8b.tt.model import ZayaModel
from models.demos.zaya1_8b.tt.cache import ZayaCache

GOLDEN = os.path.join(os.path.dirname(__file__), "..", "reference", "golden")
ok_all = True


def check(name, a, b):
    global ok_all
    ok = (a == b)
    ok_all = ok_all and ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: cache={a} recompute={b}")


def main():
    ids = torch.load(os.path.join(GOLDEN, "inputs.pt"), weights_only=False)["input_ids"].to(torch.int32)
    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape((1, 1)))
    try:
        model = ZayaModel(device, w=ZayaWeights(), verbose=False)

        # 1) cached-prefill must match no-cache next_token
        gold0 = model.next_token(ids)
        cache = ZayaCache()
        t0 = model.prefill(ids, cache)
        check("prefill", t0, gold0)

        # 2) successive decode steps must match full-recompute next_token
        seq = ids
        cur = t0
        for step in range(3):
            seq = torch.cat([seq, torch.tensor([[cur]], dtype=torch.int32)], dim=1)
            dec = model.decode_step(cur, cache)
            gold = model.next_token(seq)
            check(f"decode_step {step} (pos {seq.shape[1]})", dec, gold)
            cur = dec
    finally:
        ttnn.close_mesh_device(device)

    print(f"\n=== decode consistency: {'ALL PASS' if ok_all else 'FAIL'} ===")
    raise SystemExit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
