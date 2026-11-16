"""Full-model batch-1 greedy decode on two chat prompts (128 tokens each): tokens, step time and, given a
reference, teacher-forced top-1 agreement (PLAN.md stage gate "Agreement").

  python greedy_agreement.py <out.json>                          # reference run: greedy tokens + metrics
  python greedy_agreement.py <out.json> ws2 <reference.json>     # + feed the reference tokens, score top-1

Agreement = fraction of positions where this model's top-1 token, given the reference history, equals the next
reference token. To produce the reference from the previous `integration` without switching this checkout: check
it out as a worktree and put a directory holding only a `models` symlink to the worktree's `models/` first on
PYTHONPATH (ttnn still comes from this build):
  git worktree add --detach /tmp/int hub/integration; mkdir /tmp/intpath; ln -s /tmp/int/models /tmp/intpath/models
  PYTHONPATH=/tmp/intpath:$PWD python .../greedy_agreement.py ref.json
The second argument selects decoder paths on code that has the WS2 switches: "ws2" (default) or "base" (WS2
batch-1 attention and norm paths off at runtime); code without them ignores it.
"""
import json
import sys

import torch
from transformers import AutoTokenizer

import ttnn
from models.demos.gemma4_26b_a4b_qb2.tt.generator import build_generator
from models.demos.gemma4_26b_a4b_qb2.tt.model import MODEL_ID, REVISION

PROMPTS = [
    "Explain, step by step, why the sky appears blue during the day and red at sunset.",
    "Write a short Python function that returns the n-th Fibonacci number iteratively, then explain it.",
]


REF = None


def main():
    global REF
    out = sys.argv[1]
    if len(sys.argv) > 3:
        REF = {r["prompt"]: r["tokens"] for r in json.load(open(sys.argv[3]))["runs"]}
    torch.set_num_threads(8)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=1_000_000_000)
    report = {"runs": []}
    try:
        gen = build_generator(None, mesh, max_seq_len=4096)
        modes = sys.argv[2].split(",") if len(sys.argv) > 2 else ["ws2"]
        saved = [
            (getattr(layer.layer.self_attn, "fused_heads", None), getattr(layer, "ws2_norm", None))
            for layer in gen.model.layers
        ]
        for mode in modes:
            for layer, (heads, norm) in zip(gen.model.layers, saved):
                if heads is not None:  # ws2-glue code; integration code has neither switch
                    layer.layer.self_attn.fused_heads = heads and mode == "ws2"
                    layer.ws2_norm = norm if mode == "ws2" else None
            gen._release_trace()
            for text in PROMPTS:
                chat = tok.apply_chat_template(
                    [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False
                )
                ids = tok.encode(chat, add_special_tokens=False)
                runs = []
                for _ in range(2):
                    tokens = gen.generate(list(ids), 128, stop_on_eos=False)
                    runs.append({"tokens": [int(t) for t in tokens], "metrics": dict(gen.metrics)})
                assert runs[0]["tokens"] == runs[1]["tokens"], "nondeterministic greedy decode"
                row = {"mode": mode, "prompt": text, "tokens": runs[1]["tokens"], "metrics": runs[1]["metrics"]}
                if REF is not None:
                    # Teacher forcing: feed the reference tokens, score this model's top-1 against the next one.
                    ref = REF[text]
                    preds = gen.generate(list(ids), len(ref), stop_on_eos=False, next_input=lambda step, tok: ref[step])
                    preds = [int(t) for t in preds]
                    row["teacher_forced"] = preds
                    row["agreement"] = sum(a == b for a, b in zip(preds, ref)) / len(ref)
                    print("AGREEMENT", mode, f"{row['agreement']:.4f}", text[:40], flush=True)
                print(
                    "E2E",
                    mode,
                    json.dumps({"metrics": row["metrics"], "text": tok.decode(row["tokens"])[:300]}),
                    flush=True,
                )
                report["runs"].append(row)
        gen._release_trace()
    finally:
        ttnn.close_mesh_device(mesh)
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
