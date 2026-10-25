# SPDX-License-Identifier: Apache-2.0
"""CPU-only preparation of small reference fixtures from a local checkpoint.

Run on the host, then record_optimized_inputs.py in the model environment.
Does not download weights or execute a full model.
"""
import argparse,json,struct,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path);p.add_argument('--destination',type=Path,default=Path('.cache/qwen38-weights'));args=p.parse_args()
root,out=args.checkpoint,args.destination;out.mkdir(parents=True,exist_ok=True)
index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
def read(name, rows=None):
 with (root/index[name]).open('rb') as f:
  n=struct.unpack('<Q',f.read(8))[0];h=json.loads(f.read(n));r=h[name];a,b=r['data_offsets']
  if rows is None:f.seek(8+n+a);return r,f.read(b-a)
  assert r['dtype']=='BF16' and r['shape'][1]==5120
  raw=[]
  for i in rows:f.seek(8+n+a+i*10240);raw.append(f.read(10240))
  return {'dtype':'BF16','shape':[len(rows),5120]},b''.join(raw)
def write(path,items):
 header={};payload=[];offset=0
 for name,row,raw in items:
  header[name]={'dtype':row['dtype'],'shape':row['shape'],'data_offsets':[offset,offset+len(raw)]};offset+=len(raw);payload.append(raw)
 encoded=json.dumps(header).encode();encoded+=b' '*((-len(encoded))%8)
 with path.open('wb') as f:
  f.write(struct.pack('<Q',len(encoded)));f.write(encoded)
  for raw in payload:f.write(raw)
for name in ('config.json','tokenizer.json','tokenizer_config.json','chat_template.jinja'):
 if (root/name).exists():shutil.copyfile(root/name,out/name)
for layer in (0,1,2,3):
 if (out/f'layer_{layer}.safetensors').exists():continue
 items=[]
 for name in index:
  if name.startswith(f'model.language_model.layers.{layer}.'):
   row,raw=read(name);items.append((name,row,raw))
 write(out/f'layer_{layer}.safetensors',items)
if (out/'recorded_token_ids.json').exists():
 row,raw=read('model.language_model.embed_tokens.weight',json.loads((out/'recorded_token_ids.json').read_text()))
 write(out/'prompt_embeddings.safetensors',[('embeddings',row,raw)])
else:
 print('Run record_optimized_inputs.py once to emit token IDs, then rerun preparation and recording.')
