# SPDX-License-Identifier: Apache-2.0
import json,os,subprocess
from pathlib import Path
from ..tt.optimized_decoder import DEFAULT_POLICY
b=Path('models/autoports/qwen_qwen3_8_27b');d=b/'doc/optimized_decoder'
for tail in (False,True):
 p={**DEFAULT_POLICY,'history_tile_tail':tail};label='history_tile_'+str(tail).lower()
 with (d/f'{label}.log').open('w') as log:
  r=subprocess.run(['bash',str(b/'tests/optimized_run.sh'),'check',label,'--layer','0','--seq','4096','--recorded','--timing','--trace-prefill'],env={**os.environ,'QWEN_OPT_POLICY':json.dumps(p)},stdout=log,stderr=subprocess.STDOUT)
 print(label,r.returncode,flush=True)
for layer in (0,3):
 label=f'default_b32_l{layer}'
 with (d/f'{label}.log').open('w') as log:r=subprocess.run(['bash',str(b/'tests/optimized_run.sh'),'check',label,'--layer',str(layer),'--batch','32','--seq','129','--recorded'],stdout=log,stderr=subprocess.STDOUT)
 print(label,r.returncode,flush=True)
