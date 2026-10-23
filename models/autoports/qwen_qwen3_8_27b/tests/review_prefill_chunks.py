# SPDX-License-Identifier: Apache-2.0
import copy,json,os,subprocess
from pathlib import Path
from ..tt.optimized_decoder import DEFAULT_POLICY
b=Path('models/autoports/qwen_qwen3_8_27b');d=b/'doc/optimized_decoder'
base={**copy.deepcopy(DEFAULT_POLICY),'prefill_grid':[11,10]}
policies={'c640':{'prefill_chunk':640,'prefill_norm_max_tokens':640},'c1280':{'prefill_chunk':1280,'prefill_norm_max_tokens':1280},'c1024_l1':{'prefill_chunk':1024,'prefill_norm_max_tokens':1024,'prefill_l1':True,'prefill_l1_max_tokens':1024},'c1280_l1':{'prefill_chunk':1280,'prefill_norm_max_tokens':1280,'prefill_l1':True,'prefill_l1_max_tokens':1280}}
for name,changes in policies.items():
 p={**base,**changes}
 for layer in (0,3):
  label=f'reviewchunks_{name}_l{layer}';(d/f'{label}.policy.json').write_text(json.dumps(p,indent=2)+'\n')
  command=['bash',str(b/'tests/optimized_run.sh'),'check',label,'--layer',str(layer),'--seq','4096','--recorded','--timing','--trace-prefill']
  with (d/f'{label}.log').open('w') as log:r=subprocess.run(command,env={**os.environ,'QWEN_OPT_POLICY':json.dumps(p)},stdout=log,stderr=subprocess.STDOUT)
  print(label,r.returncode,flush=True)
