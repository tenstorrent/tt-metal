# SPDX-License-Identifier: Apache-2.0
"""Serialized final-geometry layout, precision and topology comparisons."""
import json, os, subprocess
from pathlib import Path
b=Path('models/autoports/qwen_qwen3_8_27b');d=b/'doc/optimized_decoder'
base=json.loads((d/'g80_r2_l0.policy.json').read_text())
policies={'control':{},'norm80':{'norm_cores':80},'norm40':{'norm_cores':40},
 'separate':{'packed_mlp':False,'gate':{'cores':80,'block':2},'up':{'cores':80,'block':2}},
 'separate40':{'packed_mlp':False,'gate':{'cores':40,'block':4},'up':{'cores':40,'block':4}},
 'split':{'split_mix':True},'act8attention':{'attention_activation':'bfloat8_b'},'act8mlp':{'mlp_activation':'bfloat8_b'},
 'mlphifi2':{'gateup':{**base['gateup'],'fidelity':'HiFi2'},'down':{**base['down'],'fidelity':'HiFi2'}},
 'sdpa128':{'sdpa_chunk':128},'sdpa512':{'sdpa_chunk':512},'sdpa8':{'sdpa_cores':8},'sdpa32':{'sdpa_cores':32}}
for name,changes in policies.items():
 p={**base,**changes}
 for layer in ((3,) if name.startswith('sdpa') else (0,3)):
  label=f'finalscreen_{name}_l{layer}';(d/f'{label}.policy.json').write_text(json.dumps(p,indent=2)+'\n')
  with (d/f'{label}.log').open('w') as log:
   r=subprocess.run(['bash',str(b/'tests/optimized_run.sh'),'check',label,'--layer',str(layer),'--seq','4096','--recorded','--timing'],env={**os.environ,'QWEN_OPT_POLICY':json.dumps(p)},stdout=log,stderr=subprocess.STDOUT)
  print(label,r.returncode,flush=True)
