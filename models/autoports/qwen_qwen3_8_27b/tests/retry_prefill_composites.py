# SPDX-License-Identifier: Apache-2.0
import json,os,subprocess
from pathlib import Path
b=Path('models/autoports/qwen_qwen3_8_27b');d=b/'doc/optimized_decoder'
base=json.loads((d/'role_control_l0.policy.json').read_text())
policies={'normblock_retry':{'prefill_sharded_norm':True,'prefill_norm_block':True},'sdpa32_32exact':{'prefill_sdpa_q':32,'prefill_sdpa_k':32},'sdpa32_128approx':{'prefill_sdpa_q':32,'prefill_sdpa_k':128,'prefill_sdpa_approx':True},'sdpa128_128approx':{'prefill_sdpa_q':128,'prefill_sdpa_k':128,'prefill_sdpa_approx':True},'sdpa128_256approx':{'prefill_sdpa_q':128,'prefill_sdpa_k':256,'prefill_sdpa_approx':True}}
for name,changes in policies.items():
 p={**base,**changes}
 for layer in ((3,) if name.startswith('sdpa') else (0,3)):
  label=f'composite_{name}_l{layer}';(d/f'{label}.policy.json').write_text(json.dumps(p,indent=2)+'\n')
  command=['bash',str(b/'tests/optimized_run.sh'),'check',label,'--layer',str(layer),'--seq','4096','--recorded','--timing','--trace-prefill']
  with (d/f'{label}.log').open('w') as log:r=subprocess.run(command,env={**os.environ,'QWEN_OPT_POLICY':json.dumps(p)},stdout=log,stderr=subprocess.STDOUT)
  print(label,r.returncode,flush=True)
