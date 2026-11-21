# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU HF reference for a converted Bonsai 2 checkpoint: top-5 of the first token and 8 greedy tokens for the
128-token prompt that the qwen36 text demo (traced_128) uses. Compare with the TT demo output.

Usage: python models/demos/blackhole/bonsai2/cpu_reference.py HF_DENSE_DIR   (run from the tt-metal root)
"""

import json
import sys

import torch
from transformers import AutoTokenizer
from transformers.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig

d = sys.argv[1]
tok = AutoTokenizer.from_pretrained(d)
data = json.load(open("models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_questions_prefill_128.json"))
ids = tok(data[0]["prompt"], return_tensors="pt")["input_ids"]
while ids.shape[1] < 128:
    ids = torch.cat([ids, ids], 1)
ids = ids[:, :128]
print("PROMPT TAIL:", repr(tok.decode(ids[0, -30:])))
m = Qwen3_5ForCausalLM.from_pretrained(d, config=Qwen3_5TextConfig.from_pretrained(d), dtype=torch.bfloat16).eval()
with torch.no_grad():
    lg = m(ids).logits[0, -1].float()
    top = torch.topk(torch.softmax(lg, -1), 5)
    print("top5 first token:", [(tok.decode([int(i)]), round(float(p), 3)) for p, i in zip(top.values, top.indices)])
    out = m.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=8, do_sample=False)
print("GREEDY:", [tok.decode([int(t)]) for t in out[0, 128:]], "eos=", tok.eos_token_id, out[0, 128:].tolist())
