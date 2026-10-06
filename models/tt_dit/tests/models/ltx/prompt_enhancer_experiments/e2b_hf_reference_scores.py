# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""HF bf16 CPU greedy rerun of hf_reference.json with per-position raw logits kept.

Greedy generate is teacher forcing on its own argmax sequence, so ``logits[i]`` is the reference
distribution at new-token index ``i`` given the HF prefix. Writes hf_reference_scores.json with the
top-5 ids/logits per position, the full logit vector at the contested index 2, and the gap between
the two contested tokens there.
"""

import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reads the reference and writes the scores under the shared experiment output dir, not the repo tree.
OUT_DIR = os.path.join(
    os.environ.get("LTX_ENHANCER_EXP_DIR", os.path.expanduser("~/ltx_enhancer_experiments")), "e2b_on_dit_handle"
)
os.makedirs(OUT_DIR, exist_ok=True)
REF_PATH = os.path.join(OUT_DIR, "hf_reference.json")
OUT_PATH = os.path.join(OUT_DIR, "hf_reference_scores.json")
MAX_NEW_TOKENS = 64
CONTESTED_INDEX = 2
CONTESTED_TOKENS = (60420, 32165)  # device ' cinematic' vs HF ' documentary'

ref = json.load(open(REF_PATH))
snapshot = ref["model_snapshot"]
tok = AutoTokenizer.from_pretrained(snapshot)
model = AutoModelForCausalLM.from_pretrained(snapshot, dtype=torch.bfloat16).eval()
enc = tok(ref["templated_prompt"], return_tensors="pt")
prompt_ids = enc["input_ids"][0].tolist()
assert prompt_ids == ref["prompt_token_ids"], "prompt tokenization drifted from hf_reference.json"

t0 = time.time()
with torch.no_grad():
    gen = model.generate(
        **enc,
        do_sample=False,
        max_new_tokens=MAX_NEW_TOKENS,
        output_logits=True,
        return_dict_in_generate=True,
    )
secs = time.time() - t0
new_ids = gen.sequences[0, len(prompt_ids) :].tolist()
logits = [step[0].float() for step in gen.logits]  # one [vocab] vector per emitted token

positions = []
for i, vec in enumerate(logits):
    top = torch.topk(vec, 5)
    positions.append(
        {
            "index": i,
            "token_id": new_ids[i],
            "argmax": int(vec.argmax().item()),
            "top5_ids": top.indices.tolist(),
            "top5_logits": [round(v, 4) for v in top.values.tolist()],
            "top1_minus_top2": round(float(top.values[0] - top.values[1]), 4),
        }
    )
c = logits[CONTESTED_INDEX]
a, b = CONTESTED_TOKENS
contested = {
    "index": CONTESTED_INDEX,
    "tokens": list(CONTESTED_TOKENS),
    "token_strs": [tok.decode([a]), tok.decode([b])],
    "logit_a": float(c[a]),
    "logit_b": float(c[b]),
    "gap_a_minus_b": float(c[a] - c[b]),
    "rank_a": int((c > c[a]).sum().item()),
    "rank_b": int((c > c[b]).sum().item()),
}
res = {
    "model_snapshot": snapshot,
    "transformers_version": __import__("transformers").__version__,
    "torch_version": torch.__version__,
    "dtype": "bfloat16",
    "prompt_token_count": len(prompt_ids),
    "new_token_ids": new_ids,
    "matches_hf_reference_new_token_ids": new_ids == ref["new_token_ids"][: len(new_ids)],
    "output_text": tok.decode(new_ids, skip_special_tokens=True),
    "positions": positions,
    "contested": contested,
    "index2_full_logits": [round(v, 4) for v in c.tolist()],
    "seconds": secs,
}
json.dump(res, open(OUT_PATH, "w"), indent=1)
print(json.dumps({k: res[k] for k in ["matches_hf_reference_new_token_ids", "contested", "seconds"]}))
sys.stdout.flush()
