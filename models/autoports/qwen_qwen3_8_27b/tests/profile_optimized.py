# SPDX-License-Identifier: Apache-2.0
"""Serialized whole-layer 4096+128 profiles, with advice-enabled reports."""
import json, os, subprocess
from pathlib import Path
base=Path('models/autoports/qwen_qwen3_8_27b')
control={'dram_sharded':True,'fidelity':'LoFi','mlp_dtype':'bfloat4_b','attention_dtype':'bfloat4_b','down_dtype':'bfloat4_b','block':4,'sharded_norm':True,'carry_residual':True,'kv_dtype':'bfloat8_b','readers':2,'packed_mlp':True}
for chunk in (128,512,1024):
 for layer in (0,3):
  name=f'profile_c{chunk}_l{layer}'
  policy={**control,'prefill_chunk':chunk}
  if chunk>128:policy.update(prefill_grid=[8,8],prefill_block=8)
  with (base/'doc/optimized_decoder'/f'{name}.log').open('w') as log:
   r=subprocess.run(['bash',str(base/'tests/optimized_run.sh'),'profile',name,'--layer',str(layer),'--recorded','--timing'],env={**os.environ,'QWEN_OPT_POLICY':json.dumps(policy)},stdout=log,stderr=subprocess.STDOUT)
  print(name,r.returncode,flush=True)
  if r.returncode:raise SystemExit(r.returncode)
