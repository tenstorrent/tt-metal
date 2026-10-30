# SPDX-License-Identifier: Apache-2.0
"""Run the established shared qualitative messages against a live vLLM server."""
import argparse
import json
from pathlib import Path
import urllib.request
from transformers import AutoTokenizer

parser=argparse.ArgumentParser()
parser.add_argument('--server-url',default='http://localhost:8000')
parser.add_argument('--model',default='/workspace/.cache/qwen38-full')
parser.add_argument('--output',required=True)
args=parser.parse_args()
root=Path('models/autoports/qwen_qwen3_8_27b')
tokenizer=AutoTokenizer.from_pretrained(args.model)
assert tokenizer.chat_template
result=dict(hf_model='Qwen/Qwen3.8-27B',revision='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0',
            prompt_mode='chat',chat_template_present=True,endpoint='/v1/chat/completions',
            source='doc/qualitative_prompts.json',control='readiness_vllm/full_model_controls.json',outputs=[])
for item in json.loads((root/'doc/qualitative_prompts.json').read_text())['prompts']:
    for temperature in (0.,.7):
        payload=dict(model=args.model,messages=item['messages'],max_tokens=256,temperature=temperature,
                     top_p=1.,top_k=1 if temperature==0 else 4)
        request=urllib.request.Request(args.server_url+'/v1/chat/completions',
            json.dumps(payload).encode(),{'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=900) as response:data=json.load(response)
        result['outputs'].append(dict(id=item['id'],messages=item['messages'],
            rendered_prompt=tokenizer.apply_chat_template(item['messages'],tokenize=False,add_generation_prompt=True),
            generation=payload,response=data))
        Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
        print(item['id'],temperature,'complete',flush=True)
result['status']='collected_requires_review'
Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
