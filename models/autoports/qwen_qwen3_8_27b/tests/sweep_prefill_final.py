# SPDX-License-Identifier: Apache-2.0
import json,os,subprocess
from pathlib import Path
b=Path('models/autoports/qwen_qwen3_8_27b');d=b/'doc/optimized_decoder'
base={**json.loads((d/'g80_r2_l0.policy.json').read_text()),'l1_intermediates':True,'sdpa_cores':32}
policies={'control':{},'bothl1':{'l1_mlp':True},'hybrid':{'separate_prefill':True},'prefillnorm':{'prefill_sharded_norm':True},'prefillnorm40':{'prefill_sharded_norm':True,'prefill_norm_cores':40},'traced':{}}
for name,changes in policies.items():
 p={**base,**changes}
 for layer in (0,3):
  label=f'prefinal_{name}_l{layer}';(d/f'{label}.policy.json').write_text(json.dumps(p,indent=2)+'\n')
  command=['bash',str(b/'tests/optimized_run.sh'),'check',label,'--layer',str(layer),'--seq','4096','--recorded','--timing']
  if name=='traced':command+=['--trace-prefill']
  with (d/f'{label}.log').open('w') as log:r=subprocess.run(command,env={**os.environ,'QWEN_OPT_POLICY':json.dumps(p)},stdout=log,stderr=subprocess.STDOUT)
  print(label,r.returncode,flush=True)
