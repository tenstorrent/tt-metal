# SPDX-License-Identifier: Apache-2.0
"""Pinned-checkpoint CPU HF native-chat controls for the shared prompt suite."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from .hf_text import load_model

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--output-dir", type=Path, required=True)
p.add_argument("--threads", type=int, default=8)
p.add_argument("--max-new-tokens", type=int, default=512)
a = p.parse_args()
torch.set_num_threads(a.threads)
model, tok = load_model(a.checkpoint)
root = Path(__file__).resolve().parents[1]
suite_path = root / "doc/qualitative_prompts.json"
suite = json.loads(suite_path.read_text())
assert tok.chat_template, "Shared quality verdict requires native checkpoint template"
generation_config = json.loads((Path(a.checkpoint) / "generation_config.json").read_text())
eos = generation_config.get("eos_token_id", tok.eos_token_id)
eos = set(eos if isinstance(eos, list) else [eos])
a.output_dir.mkdir(parents=True, exist_ok=True)
for item in suite["prompts"]:
    rendered = tok.apply_chat_template(item["messages"], tokenize=False, add_generation_prompt=True)
    prompt = tok.encode(rendered, add_special_tokens=False, return_tensors="pt")
    cache = None
    tokens = []
    inp = prompt
    started = time.monotonic()
    with torch.inference_mode():
        for step in range(a.max_new_tokens):
            out = model(input_ids=inp, past_key_values=cache, use_cache=True, logits_to_keep=1)
            cache = out.past_key_values
            token = int(out.logits[0, -1].float().argmax())
            tokens.append(token)
            inp = torch.tensor([[token]])
            if step % 32 == 0:
                print("HF_QUALITY", item["id"], step, flush=True)
            if token in eos:
                break
    record = dict(
        id=item["id"],
        messages=item["messages"],
        rendered_prompt=rendered,
        prompt_tokens=prompt[0].tolist(),
        generated_tokens=tokens,
        completion=tok.decode(tokens, skip_special_tokens=False),
        seconds=time.monotonic() - started,
        hf_model="Qwen/Qwen3.8-27B",
        revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        tokenizer=type(tok).__name__,
        chat_template=True,
        do_sample=False,
        max_new_tokens=a.max_new_tokens,
        dtype="bfloat16",
        eos_ids=sorted(eos),
        stopped_on_eos=tokens[-1] in eos,
        prompt_source_sha256=hashlib.sha256(suite_path.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        chat_template_sha256=hashlib.sha256(tok.get_chat_template().encode()).hexdigest(),
    )
    (a.output_dir / (item["id"] + "_hf.json")).write_text(json.dumps(record, indent=2) + "\n")
    print("HF_QUALITY_DONE", item["id"], len(tokens), record["completion"], flush=True)
