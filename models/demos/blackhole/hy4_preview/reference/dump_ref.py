# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-layer reference dump (8 fixed prompts, one long enough to activate DSA top-k).

Layer-major streaming: each layer's weights are read once and applied to all prompts. Per layer and
prompt: sha256 + stats of the fp32 input/output iHC streams, router top-8 ids (MoE layers), indexer
top-k hash (full-indexer layers). Layers < --full-layers also save the raw tensors (the long prompt only
its last 32 positions). State is checkpointed so a killed run resumes.

    python -m models.demos.blackhole.hy4_preview.reference.dump_ref OUT_DIR [--layers N] [--full-layers 4]
"""
import hashlib
import json
import os
import sys
import time

import torch

from models.demos.blackhole.hy4_preview.reference.oracle import Conv, Hy4Oracle, decoder_layers_multi

SHORT = [
    "The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n",
    "Question: If a train travels 120 km in 2 hours, what is its average speed?\nAnswer: Let's think step by step.",
    "Explain in one paragraph why the sky is blue.",
    "1, 1, 2, 3, 5, 8, 13, 21, 34,",
    "User: Write a haiku about autumn.\nAssistant:",
    "Der schnelle braune Fuchs springt ueber den faulen Hund. 今天天气很好，我们去公园散步吧。",
]
LONG_UNIT = (
    "Section {i}. The committee reviewed the quarterly figures for region {i}, noting that revenue grew by {a} percent "
    "while costs rose by {b} percent, and agreed to revisit the staffing plan after the next audit. "
)


def long_prompt(n_units=75):
    return "".join(LONG_UNIT.format(i=i, a=(i * 7) % 23 + 1, b=(i * 5) % 17 + 1) for i in range(n_units))


def sha(t):
    return hashlib.sha256(t.detach().float().contiguous().numpy().tobytes()).hexdigest()


def stats(t):
    t = t.float()
    return {
        "sha256": sha(t),
        "mean": t.mean().item(),
        "std": t.std().item(),
        "absmax": t.abs().max().item(),
        "norm_mean": t.norm(dim=-1).mean().item(),
    }


def main():
    out = sys.argv[1]
    nl = int(sys.argv[sys.argv.index("--layers") + 1]) if "--layers" in sys.argv else 4
    nfull = int(sys.argv[sys.argv.index("--full-layers") + 1]) if "--full-layers" in sys.argv else 4
    torch.set_num_threads(int(os.environ.get("HY4_THREADS", "32")))
    os.makedirs(out, exist_ok=True)
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file("/mnt/tt-data/Hy4-Preview/tokenizer.json")
    texts = SHORT + [long_prompt()]
    ids = [torch.tensor([[120000] + tok.encode(t).ids]) for t in texts]
    o = Hy4Oracle()
    conv = Conv()
    state_f = os.path.join(out, "state.pt")
    if os.path.exists(state_f):
        st = torch.load(state_f)
        x4s, start, idx_states = st["x4s"], st["layer"], st["idx"]
    else:
        x4s, start, idx_states = [o.embed_ids(i) for i in ids], 0, [{} for _ in ids]
    meta = {
        "prompts": [
            {"n_tokens": i.shape[1], "ids_sha256": hashlib.sha256(i.numpy().tobytes()).hexdigest()} for i in ids
        ],
        "conv": conv.__dict__,
    }
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    t0 = time.time()
    for L in range(start, nl):
        taps = {}
        xin = x4s
        x4s = decoder_layers_multi(o.ck, L, xin, [conv] * len(ids), idx_states, taps)
        rec = {
            "layer": L,
            "kind": ("dense" if o.cfg.is_dense(L) else "moe") + "/" + o.cfg.indexer_types[L],
            "prompts": [],
        }
        for p in range(len(ids)):
            r = {"in": stats(xin[p]), "out": stats(x4s[p])}
            if taps.get("routes"):
                r["route_sha256"] = sha(taps["routes"][p].float())
            if o.cfg.is_full_indexer(L) and "idx" in idx_states[p]:
                r["indexer_topk_sha256"] = sha(idx_states[p]["idx"].float())
            rec["prompts"].append(r)
            if L < nfull:
                keep = slice(None) if ids[p].shape[1] < 512 else slice(-32, None)
                torch.save(
                    {
                        "in": xin[p][:, keep].clone(),
                        "out": x4s[p][:, keep].clone(),
                        "routes": taps["routes"][p][:, keep].clone() if taps.get("routes") else None,
                    },
                    os.path.join(out, f"layer{L:02d}_p{p}.pt"),
                )
        json.dump(rec, open(os.path.join(out, f"layer{L:02d}.json"), "w"), indent=1)
        torch.save({"x4s": x4s, "layer": L + 1, "idx": idx_states}, state_f + ".tmp")
        os.replace(state_f + ".tmp", state_f)
        print(f"layer {L} {rec['kind']} done {time.time()-t0:.0f}s", flush=True)
    if nl == o.cfg.n_layers:
        res = []
        for p, x4 in enumerate(x4s):
            h, hn = o.head(x4, conv)
            lg = o.logits(hn[:, -1:])
            torch.save({"hc_head": h, "final_normed": hn}, os.path.join(out, f"final_hidden_p{p}.pt"))
            res.append({"last_top5": lg[0, -1].topk(5)[1].tolist(), "last_logits_sha256": sha(lg)})
        json.dump(res, open(os.path.join(out, "final.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
