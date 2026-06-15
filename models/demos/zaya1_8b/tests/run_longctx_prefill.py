"""LongCtx L1: validate FlashAttention-2 prefill — token-exact at golden S, and prefill
scales to long S without the [S,S] mask (the 68GB-at-128K blocker is gone).
Run: TT_DEVICE=1 /home/yito/work/run_zaya.sh python models/demos/zaya1_8b/tests/run_longctx_prefill.py
"""
import os
import time
import torch
import ttnn

from models.demos.zaya1_8b.tt.model_args import ZayaConfig, ZayaWeights
from models.demos.zaya1_8b.tt.model import ZayaModel
from models.demos.zaya1_8b.tt.cache import ZayaCache

GOLDEN = os.path.join(os.path.dirname(__file__), "..", "reference", "golden")


def main():
    ids = torch.load(os.path.join(GOLDEN, "inputs.pt"), weights_only=False)["input_ids"].to(torch.int32)
    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape((1, 1)), l1_small_size=32768)
    device.enable_program_cache()
    try:
        model = ZayaModel(device, w=ZayaWeights(), verbose=False)

        # (1) token-exact: flash prefill must still produce the golden stream
        got = model.generate(ids, 6)
        gold = [9079, 236761, 107, 2717, 108, 2717]
        print(f"[L1] generate = {got}")
        print(f"[L1] [{'PASS' if got == gold else 'FAIL'}] token-exact vs golden (flash prefill)", flush=True)

        # (2) long-S prefill scaling (random ids; just confirm it runs without the S^2 mask)
        if os.environ.get("ZAYA_LONGS", "0") != "1":
            print("[L1] (long-S sweep skipped; set ZAYA_LONGS=1)", flush=True)
            return
        print("[L1] long-S prefill (flash, no [S,S] mask):", flush=True)
        for S in (4096, 16384, 65536, 131072):
            try:
                long_ids = torch.randint(0, ZayaConfig.vocab_size, (1, S), dtype=torch.int32)
                cache = ZayaCache()
                t = time.time()
                tok = model.prefill(long_ids, cache)
                dt = time.time() - t
                kv_gb = 40 * 2 * 2 * S * 128 * 2 / 1e9
                print(f"  S={S:>7}: prefill OK {dt:6.2f}s, next-tok={tok}, KV~{kv_gb:.1f}GB", flush=True)
            except Exception as e:
                print(f"  S={S:>7}: FAILED {type(e).__name__}: {str(e)[:80]}", flush=True)
                break
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
