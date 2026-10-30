# SPDX-License-Identifier: Apache-2.0
import copy,json,os,subprocess
from pathlib import Path
from ..tt.optimized_decoder import DEFAULT_POLICY
b=Path('models/autoports/qwen_qwen3_8_27b');d=b/'doc/optimized_decoder'
base=copy.deepcopy(DEFAULT_POLICY)
policies={'control':{},'grid110':{'prefill_grid':[11,10]},'grid88':{'prefill_grid':[11,8]},'grid88k16':{'prefill_grid':[11,8],'prefill_block':16},'l1':{'prefill_l1':True},'grid88k16l1':{'prefill_grid':[11,8],'prefill_block':16,'prefill_l1':True},'grid88k10l1':{'prefill_grid':[11,8],'prefill_l1':True},'mix88':{'mix':{**base['mix'],'prefill_grid':[11,8]}}}
for name,changes in policies.items():
 p={**base,**changes}
 for layer in (0,3):
  label=f'reviewprefill_{name}_l{layer}';(d/f'{label}.policy.json').write_text(json.dumps(p,indent=2)+'\n')
  command=['bash',str(b/'tests/optimized_run.sh'),'check',label,'--layer',str(layer),'--seq','4096','--recorded','--timing','--trace-prefill']
  with (d/f'{label}.log').open('w') as log:r=subprocess.run(command,env={**os.environ,'QWEN_OPT_POLICY':json.dumps(p)},stdout=log,stderr=subprocess.STDOUT)
  print(label,r.returncode,flush=True)
