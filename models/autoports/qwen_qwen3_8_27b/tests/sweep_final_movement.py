# SPDX-License-Identifier: Apache-2.0
import json, os, subprocess
from pathlib import Path
b=Path('models/autoports/qwen_qwen3_8_27b');d=b/'doc/optimized_decoder'
base=json.loads((d/'g80_r2_l0.policy.json').read_text())
policies={'control':{},'norm80':{'norm_cores':80},'norm40':{'norm_cores':40},'l1inter':{'l1_intermediates':True},'l1packed':{'l1_mlp':True},'sdpa128_32':{'sdpa_chunk':128,'sdpa_cores':32}}
for name,change in policies.items():
 p={**base,**change}
 for layer in ((3,) if name.startswith('sdpa') else (0,3)):
  label=f'movement_{name}_l{layer}';(d/f'{label}.policy.json').write_text(json.dumps(p,indent=2)+'\n')
  with (d/f'{label}.log').open('w') as log:
   r=subprocess.run(['bash',str(b/'tests/optimized_run.sh'),'check',label,'--layer',str(layer),'--seq','4096','--recorded','--timing'],env={**os.environ,'QWEN_OPT_POLICY':json.dumps(p)},stdout=log,stderr=subprocess.STDOUT)
  print(label,r.returncode,flush=True)
