# SPDX-License-Identifier: Apache-2.0
"""CPU HF control for the repeated-text throughput fixture (not performance)."""
import argparse, json, time
from pathlib import Path
import torch
from .hf_text import load_model

parser=argparse.ArgumentParser()
parser.add_argument('--tokens',type=int,default=128)
parser.add_argument('--output',default='models/autoports/qwen_qwen3_8_27b/doc/full_model/benchmark_hf_control.json')
args=parser.parse_args()
torch.set_num_threads(16)
model, tok = load_model()
words = tok.encode('The research team records each observation carefully and checks the evidence before drawing conclusions. ', add_special_tokens=False)
body_length = 4096 - 48
for _ in range(8):
    body = tok.decode((words * ((body_length + len(words) - 1) // len(words)))[:body_length])
    prompt = tok.apply_chat_template([{'role':'user','content':'Continue with a detailed, several-paragraph account of this research process. Context: '+body}], tokenize=True, add_generation_prompt=True, return_dict=False)
    if len(prompt) == 4096:
        break
    body_length += 4096-len(prompt)
assert len(prompt) == 4096
cache = None
tokens = []
inp = torch.tensor([prompt])
start = time.monotonic()
with torch.inference_mode():
    for step in range(args.tokens):
        result = model(input_ids=inp, past_key_values=cache, use_cache=True, logits_to_keep=1)
        cache = result.past_key_values
        token = int(result.logits[0,-1].float().argmax())
        tokens.append(token)
        inp = torch.tensor([[token]])
        if step % 16 == 0:
            print(step, flush=True)
record = dict(scope='HF qualitative control of throughput fixture, not benchmark', hf_model='Qwen/Qwen3.8-27B', revision='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0', input_tokens=prompt, output_tokens=tokens, completion=tok.decode(tokens, skip_special_tokens=False), seconds=time.monotonic()-start, chat_template=True, do_sample=False, stop_on_eos=False, dtype='bfloat16', max_new_tokens=args.tokens)
Path(args.output).write_text(json.dumps(record, indent=2)+'\n')
print(record['completion'], flush=True)
