# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""HF bf16 control at every first TT / HF divergence of the qualitative runs (CPU only, no device).

For each qualitative-suite prompt and each chat autoregressive run, the HF model scores the shared prefix (prompt +
the tokens both produced) and reports its top-5 next-token probabilities and the HF rank / probability of the token
TT chose. Output: <stage doc>/qualitative/hf_divergence_ranks.json (GEMMA4_STAGE_DOC, default doc/full_model).

  python models/autoports/google_gemma_4_12b_it/tests/hf_divergence_ranks.py
"""

import json
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

from models.autoports.google_gemma_4_12b_it.tests.full_model_ref import hf_streamed_logits

D = Path(__file__).resolve().parents[1] / os.environ.get("GEMMA4_STAGE_DOC", "doc/full_model")
HF = "google/gemma-4-12B-it"


def first_div(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)


def main():
    torch.set_num_threads(16)
    tok = AutoTokenizer.from_pretrained(HF)
    cases = []
    items = {it["id"]: it for it in json.loads((D / "qualitative/rendered_prompts.json").read_text())}
    hf = {r["id"]: r for r in json.loads((D / "qualitative/hf_outputs.json").read_text())}
    for r in json.loads((D / "qualitative/tt_outputs.json").read_text()):
        cases.append((f"suite/{r['id']}", items[r["id"]]["token_ids"], r["tt_token_ids"], hf[r["id"]]["token_ids"]))
    for name in ("chat_story", "chat_explain"):
        meta = json.loads((D / f"readiness_autoregressive/{name}/autoregressive_meta.json").read_text())
        cases.append(
            (f"autoregressive/{name}", meta["prompt_token_ids"], meta["tt"]["token_ids"], meta["hf"]["token_ids"])
        )
    # the full bf16 model needs ~24 GB of host memory; stream its layers (tests/full_model_ref.hf_streamed_logits)
    divs = [first_div(tt, ref) for _, _, tt, ref in cases]
    todo = [k for k, d in enumerate(divs) if d is not None]
    seqs = [cases[k][1] + cases[k][3][: divs[k]] for k in todo]
    last = dict(zip(todo, hf_streamed_logits(seqs, [len(q) - 1 for q in seqs])))
    out = []
    for k, (name, prompt, tt, ref) in enumerate(cases):
        d = divs[k]
        if d is None:
            out.append({"case": name, "first_divergence": None})
            continue
        p = torch.softmax(last[k], -1)
        v, i = p.topk(5)
        t = tt[d]
        out.append(
            {
                "case": name,
                "first_divergence": d,
                "hf_top5": [[tok.decode([int(a)]), round(float(b), 4)] for a, b in zip(i, v)],
                "tt_token": tok.decode([t]),
                "tt_token_hf_prob": round(float(p[t]), 4),
                "tt_token_hf_rank": int((p > p[t]).sum()) + 1,
            }
        )
        print(out[-1], flush=True)
    (D / "qualitative/hf_divergence_ranks.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
