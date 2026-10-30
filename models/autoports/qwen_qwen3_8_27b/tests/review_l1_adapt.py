# SPDX-License-Identifier: Apache-2.0
import copy,json,os,subprocess
from pathlib import Path
from ..tt.optimized_decoder import DEFAULT_POLICY
b=Path('models/autoports/qwen_qwen3_8_27b');d=b/'doc/optimized_decoder'
for layer,chunk in ((0,1280),(3,1024)):
 for use_l1 in (False,True):
  p={**copy.deepcopy(DEFAULT_POLICY),'prefill_grid':[11,10],'prefill_chunk':chunk,'prefill_norm_max_tokens':chunk,'prefill_l1':use_l1,'prefill_l1_max_tokens':chunk,'prefill_block':8}
  p['out']['prefill_block']=24;p['down']['prefill_block']=16
  for attempt in range(2):
   label=f'reviewl1_c{chunk}_{use_l1}_{attempt}_l{layer}';(d/f'{label}.policy.json').write_text(json.dumps(p,indent=2)+'\n')
   command=['bash',str(b/'tests/optimized_run.sh'),'check',label,'--layer',str(layer),'--seq','4096','--recorded','--timing','--trace-prefill']
   with (d/f'{label}.log').open('w') as log:r=subprocess.run(command,env={**os.environ,'QWEN_OPT_POLICY':json.dumps(p)},stdout=log,stderr=subprocess.STDOUT)
   print(label,r.returncode,flush=True)
   if r.returncode==0:break
   p['prefill_block']=4;p['out']['prefill_block']=16;p['down']['prefill_block']=8
