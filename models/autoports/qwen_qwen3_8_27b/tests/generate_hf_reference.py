# SPDX-License-Identifier: Apache-2.0
"""Fresh text-only HF reference from the exact multimodal checkpoint text weights."""
import argparse
import json
import time
from pathlib import Path
import torch
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
from safetensors import safe_open
from readiness_check.generate import _load_aime24_prompt, DEFAULT_AIME24_PROMPTS_FILE
from readiness_check.schema import Reference, ReferenceEntry, save_reference

p=argparse.ArgumentParser()
p.add_argument('--checkpoint',default='.cache/qwen38-full')
p.add_argument('--output',default='models/autoports/qwen_qwen3_8_27b/readiness_aime24_chat.refpt')
a=p.parse_args()
torch.set_num_threads(16)
root=Path(a.checkpoint)
cfg=AutoConfig.from_pretrained(root).text_config
cfg._attn_implementation='sdpa'
with torch.device('meta'):
    model=Qwen3_5ForCausalLM(cfg)
index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
keys={k: k.replace('model.language_model.', 'model.') for k in index if k.startswith('model.language_model.') or k=='lm_head.weight'}
for filename in sorted({index[k] for k in keys}):
    with safe_open(root/filename,framework='pt') as f:
        tensors={keys[k]: f.get_tensor(k) for k in keys if index[k]==filename}
        model.load_state_dict(tensors,strict=False,assign=True)
    print('HF_LOAD',filename,flush=True)
if any(v.is_meta for v in model.parameters()):
    raise RuntimeError('Checkpoint missing text model parameters')
# Nonpersistent rotary buffers must be materialized separately from checkpoint parameters.
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
model.model.rotary_emb=Qwen3_5TextRotaryEmbedding(cfg)
model.eval()
tokenizer=AutoTokenizer.from_pretrained(root)
prompt=_load_aime24_prompt(DEFAULT_AIME24_PROMPTS_FILE,0)
rendered=tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True)
tokens=tokenizer.encode(rendered,add_special_tokens=False,return_tensors='pt')
generated=[];topk=[];cache=None;start=time.monotonic()
with torch.inference_mode():
    inp=tokens
    for i in range(100):
        out=model(input_ids=inp,past_key_values=cache,use_cache=True,logits_to_keep=1)
        cache=out.past_key_values
        ranks=out.logits[0,-1].float().topk(100).indices.cpu()
        generated.append(int(ranks[0]));topk.append(ranks.to(torch.int32))
        inp=ranks[:1].reshape(1,1)
        print('HF_TOKEN',i,generated[-1],flush=True)
entry=ReferenceEntry(rendered,tokens,torch.tensor([generated]),torch.stack(topk),tokens.shape[1])
save_reference(Reference(100,'Qwen/Qwen3.8-27B',[entry],{'eos_id':tokenizer.eos_token_id,'pad_id':tokenizer.pad_token_id,'bos_id':tokenizer.bos_token_id}),a.output)
metadata=dict(hf_model='Qwen/Qwen3.8-27B',revision='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0',
    tokenizer=type(tokenizer).__name__,prompt_source='aime24',chat_template=True,generation_length=100,
    top_k=100,prompt_tokens=tokens.shape[1],command='python -m models.autoports.qwen_qwen3_8_27b.tests.generate_hf_reference',
    dtype='bfloat16',scope='HF text-only path, exact text weights from multimodal checkpoint',seconds=time.monotonic()-start,
    completion=tokenizer.decode(generated,skip_special_tokens=False))
Path(a.output+'.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(metadata['completion'],flush=True)
