# SPDX-License-Identifier: Apache-2.0
import json,time
from pathlib import Path
import torch
from .hf_text import load_model

torch.set_num_threads(16)
model,tok=load_model()
root=Path('models/autoports/qwen_qwen3_8_27b/doc')
suite=json.loads((root/'qualitative_prompts.json').read_text())
out=root/'full_model/qualitative';out.mkdir(exist_ok=True)
for item in suite['prompts']:
    rendered=tok.apply_chat_template(item['messages'],tokenize=False,add_generation_prompt=True)
    prompt=tok.encode(rendered,add_special_tokens=False,return_tensors='pt')
    cache=None;tokens=[];inp=prompt;start=time.monotonic()
    with torch.inference_mode():
        for step in range(128):
            result=model(input_ids=inp,past_key_values=cache,use_cache=True,logits_to_keep=1)
            cache=result.past_key_values
            token=int(result.logits[0,-1].float().argmax())
            tokens.append(token);inp=torch.tensor([[token]])
            if step%16==0: print(item['id'],step,flush=True)
            if token==tok.eos_token_id: break
    record=dict(id=item['id'],messages=item['messages'],rendered_prompt=rendered,prompt_tokens=prompt[0].tolist(),
        generated_tokens=tokens,completion=tok.decode(tokens,skip_special_tokens=False),seconds=time.monotonic()-start,
        hf_model='Qwen/Qwen3.8-27B',revision='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0',tokenizer=type(tok).__name__,
        chat_template=True,do_sample=False,max_new_tokens=128,dtype='bfloat16')
    (out/(item['id']+'_hf.json')).write_text(json.dumps(record,indent=2)+'\n')
    print(item['id'],record['completion'],flush=True)
