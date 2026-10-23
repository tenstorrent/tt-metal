# SPDX-License-Identifier: Apache-2.0
"""Paired final-topology check of the closest historical decode competitor."""
import json,os,subprocess
from pathlib import Path
from ..tt.optimized_decoder import default_policy
b=Path('models/autoports/qwen_qwen3_8_27b');d=b/'doc/optimized_decoder'
for layer,kind in ((0,'linear_attention'),(3,'full_attention')):
 for repeat,l1 in enumerate((False,True,True,False)):
  policy=default_policy(kind);policy['l1_mlp']=l1
  label=f'paired_final_l{layer}_{repeat}_l1{int(l1)}'
  (d/f'{label}.policy.json').write_text(json.dumps(policy,indent=2)+'\n')
  with (d/f'{label}.log').open('w') as log:
   r=subprocess.run(['bash',str(b/'tests/optimized_run.sh'),'check',label,'--layer',str(layer),'--seq','4096','--recorded','--timing','--trace-prefill'],env={**os.environ,'QWEN_OPT_POLICY':json.dumps(policy)},stdout=log,stderr=subprocess.STDOUT)
  print(label,r.returncode,flush=True)
  if r.returncode:raise SystemExit(r.returncode)
# Final report's 256-token tail has per_core_M1/per_core_N15, hence1x1
# output/down subblocks. Ten columns permit N16 and subblock width4.
for layer,kind in ((0,'linear_attention'),(3,'full_attention')):
 for roles in (('out',),('down',),('out','down')):
  policy=default_policy(kind)
  for role in roles:policy[role]['prefill_grid']=[10,10]
  label=f'final_subblock_{"_".join(roles)}_l{layer}'
  (d/f'{label}.policy.json').write_text(json.dumps(policy,indent=2)+'\n')
  with (d/f'{label}.log').open('w') as log:
   r=subprocess.run(['bash',str(b/'tests/optimized_run.sh'),'check',label,'--layer',str(layer),'--seq','4096','--recorded','--timing','--trace-prefill'],env={**os.environ,'QWEN_OPT_POLICY':json.dumps(policy)},stdout=log,stderr=subprocess.STDOUT)
  print(label,r.returncode,flush=True)