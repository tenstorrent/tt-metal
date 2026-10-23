# SPDX-License-Identifier: Apache-2.0
import json,os,subprocess
from pathlib import Path
b=Path('models/autoports/qwen_qwen3_8_27b');d=b/'doc/optimized_decoder'
base={**json.loads((d/'g80_r2_l0.policy.json').read_text()),'l1_intermediates':True,'sdpa_cores':32,'separate_prefill':True}
policies={'control':{},'down34k16':{'down':{'cores':34,'block':16}},'down17k32':{'down':{'cores':17,'block':32}},'down8k68':{'down':{'cores':8,'block':68}},'mix40k4':{'mix':{'cores':40,'block':4}},'gateup40k4':{'gateup':{'cores':40,'block':4}},'gateup16k10':{'gateup':{'cores':16,'block':10}},'out32k6':{'out':{'cores':32,'block':6}}}
for name,changes in policies.items():
 p={**base,**changes}
 for layer in (0,3):
  label=f'role_{name}_l{layer}';(d/f'{label}.policy.json').write_text(json.dumps(p,indent=2)+'\n')
  command=['bash',str(b/'tests/optimized_run.sh'),'check',label,'--layer',str(layer),'--seq','4096','--recorded','--timing','--trace-prefill']
  with (d/f'{label}.log').open('w') as log:r=subprocess.run(command,env={**os.environ,'QWEN_OPT_POLICY':json.dumps(p)},stdout=log,stderr=subprocess.STDOUT)
  print(label,r.returncode,flush=True)
for layer in (0,3):
 label=f'baseline_fused_l{layer}'
 with (d/f'{label}.log').open('w') as log:r=subprocess.run(['bash',str(b/'tests/optimized_run.sh'),'profile',label,'--layer',str(layer),'--recorded','--baseline-fused','--trace-prefill'],stdout=log,stderr=subprocess.STDOUT)
 print(label,r.returncode,flush=True)
