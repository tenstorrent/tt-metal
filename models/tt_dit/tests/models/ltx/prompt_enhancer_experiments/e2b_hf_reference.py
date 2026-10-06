import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.tt_dit.pipelines.ltx.prompt_enhancer import build_messages

S = "/mnt/models/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/3e22461f65e89153144f8adb70e3b8c2cc9845a7"
import os

OUT_DIR = os.path.join(
    os.environ.get("LTX_ENHANCER_EXP_DIR", os.path.expanduser("~/ltx_enhancer_experiments")), "e2b_on_dit_handle"
)
os.makedirs(OUT_DIR, exist_ok=True)
OUT = os.path.join(OUT_DIR, "hf_reference.json")

tok = AutoTokenizer.from_pretrained(S)
model = AutoModelForCausalLM.from_pretrained(S, dtype=torch.bfloat16).eval()
messages = build_messages("beekeeper", "t2v")
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
enc = tok(text, return_tensors="pt")
prompt_ids = enc["input_ids"][0].tolist()

t0 = time.time()
with torch.no_grad():
    logits = model(**enc).logits[0, -1].float()
    top5 = torch.topk(logits, 5).indices.tolist()
    gen = model.generate(**enc, do_sample=False, max_new_tokens=64)
secs = time.time() - t0
new_ids = gen[0, len(prompt_ids) :].tolist()
out_text = tok.decode(new_ids, skip_special_tokens=True)
res = {
    "model_snapshot": S,
    "transformers_version": __import__("transformers").__version__,
    "torch_version": torch.__version__,
    "dtype": "bfloat16",
    "messages": messages,
    "templated_prompt": text,
    "prompt_token_count": len(prompt_ids),
    "prompt_token_ids": prompt_ids,
    "first_token_top5_ids": top5,
    "first_token_top5_tokens": [tok.decode([i]) for i in top5],
    "new_token_ids": new_ids,
    "output_text": out_text,
    "seconds": secs,
}
json.dump(res, open(OUT, "w"), indent=2)
print(
    json.dumps(
        {k: res[k] for k in ["prompt_token_count", "first_token_top5_ids", "new_token_ids", "output_text", "seconds"]}
    )
)
