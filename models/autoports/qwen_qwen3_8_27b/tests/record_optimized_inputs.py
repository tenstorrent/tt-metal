# SPDX-License-Identifier: Apache-2.0
"""CPU-only reference activations for representative decoder precision decisions.

Uses real embeddings and the three preceding HF layers, not a TT full-model implementation.
"""
from pathlib import Path
import json
import argparse
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer, DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding

parser=argparse.ArgumentParser()
parser.add_argument("--token-ids-only", action="store_true")
args=parser.parse_args()
torch.set_num_threads(8)
root=Path('.cache/qwen38-weights')
cfg=AutoConfig.from_pretrained(str(root)).text_config
cfg._attn_implementation='eager'
tokenizer=AutoTokenizer.from_pretrained(str(root))
text=('User: Inspect the source code and explain the failing test. Preserve the context and check the result. '
      'Assistant: I will read the implementation, compare the expected and actual values, and verify the fix.\n')
ids=tokenizer.encode(text,add_special_tokens=False)
(root/"recorded_token_ids.json").write_text(json.dumps(ids))
if args.token_ids_only:
    raise SystemExit(0)
if not (root/"prompt_embeddings.safetensors").exists():
    raise SystemExit("Extract recorded_token_ids.json embedding rows before running again")
ids=(ids*((4226+len(ids)-1)//len(ids)))[:4226]
emb=load_file(str(root/'prompt_embeddings.safetensors'))['embeddings']
x=emb.repeat((4226+len(emb)-1)//len(emb),1)[:4226][None]
rope=Qwen3_5TextRotaryEmbedding(cfg)
cos,sin=rope(x,torch.arange(x.shape[1]).expand(3,1,-1))
with torch.no_grad():
 for i in range(4):
  if i in (0,3):
   torch.save(x,root/f'recorded_layer_{i}.pt')
   print('RECORDED',i,tuple(x.shape),x.float().std().item(),flush=True)
  if i==3:break
  layer=Qwen3_5DecoderLayer(cfg,i).to(torch.bfloat16).eval()
  prefix=f'model.language_model.layers.{i}.'
  layer.load_state_dict({k.removeprefix(prefix):v for k,v in load_file(str(root/f'layer_{i}.safetensors')).items()})
  cache=DynamicCache(config=cfg)
  ys=[]
  for start in range(0,x.shape[1],128):
   ys.append(layer(x[:,start:start+128],position_embeddings=(cos[:,start:start+128],sin[:,start:start+128]),past_key_values=cache))
  x=torch.cat(ys,dim=1)
(root/'recorded_inputs.json').write_text(json.dumps({'source':'real checkpoint embeddings + preceding HF linear-attention layers','text':text,'tokens':len(ids),'token_ids_period':tokenizer.encode(text,add_special_tokens=False),'layer_types':['linear_attention','full_attention']},indent=2)+'\n')
