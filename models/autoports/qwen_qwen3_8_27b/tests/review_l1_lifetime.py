# SPDX-License-Identifier: Apache-2.0
import json,os,subprocess
from pathlib import Path
b=Path('models/autoports/qwen_qwen3_8_27b');d=b/'doc/optimized_decoder'
for layer,chunk in ((0,1280),(3,1024)):
 p=json.loads((d/f'reviewl1_c{chunk}_True_0_l{layer}.policy.json').read_text());label=f'reviewl1_lifetime_l{layer}'
 with (d/f'{label}.log').open('w') as log:
  r=subprocess.run(['bash',str(b/'tests/optimized_run.sh'),'check',label,'--layer',str(layer),'--seq','4096','--recorded','--timing','--trace-prefill'],env={**os.environ,'QWEN_OPT_POLICY':json.dumps(p)},stdout=log,stderr=subprocess.STDOUT)
 print(label,r.returncode,flush=True)
