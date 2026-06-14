"""ZAYA1-8B end-to-end greedy generation demo on a single Blackhole P150a.

Generation mode: prefill-recompute (re-run the full forward on the growing
sequence each step). Correct but unoptimised; the incremental ZayaDynamicCache
(conv_states + prev_hs + KV) decode path is a Phase 6 perf item.

Run: TT_DEVICE=1 /home/yito/work/run_zaya.sh \
       python models/demos/zaya1_8b/demo/demo.py --prompt "The capital of France is" --n 8
"""
import argparse
import torch
import ttnn

from models.demos.zaya1_8b.tt.model_args import ZayaConfig, ZayaWeights, find_snapshot
from models.demos.zaya1_8b.tt.model import ZayaModel


def get_tokenizer():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(find_snapshot())
    except Exception as e:
        print(f"[demo] tokenizer unavailable ({e}); will print token ids only")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--n", type=int, default=8, help="tokens to generate")
    args = ap.parse_args()

    tok = get_tokenizer()
    if tok is not None:
        ids = tok(args.prompt, return_tensors="pt").input_ids
    else:
        ids = torch.tensor([[2, 818, 5279, 529, 7001, 563]])  # canonical prompt fallback
    print(f"[demo] prompt ids {ids.tolist()}")

    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape((1, 1)))
    device.enable_program_cache()   # avoid kernel recompile across decode steps
    try:
        print("[demo] building model once (weights load to device once, reused across steps)...")
        model = ZayaModel(device, w=ZayaWeights(), verbose=False)   # seq-agnostic; build once
        # incremental decode: prefill populates the KV+conv+prev_hs cache, then O(1)/token
        new_ids = model.generate(ids.to(torch.int32), args.n)
        generated = torch.cat([ids, torch.tensor([new_ids])], dim=1)
        for step, nxt in enumerate(new_ids):
            piece = tok.decode([nxt]) if tok else str(nxt)
            print(f"[demo] step {step}: -> {nxt!r} {piece!r}")
        if tok is not None:
            print("\n[demo] full text:\n" + tok.decode(generated[0].tolist()))
        else:
            print(f"\n[demo] ids: {generated.tolist()}")
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
