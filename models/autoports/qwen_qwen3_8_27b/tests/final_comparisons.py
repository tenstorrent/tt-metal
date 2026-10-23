# SPDX-License-Identifier: Apache-2.0
"""Final default and short/tail comparisons with the candidate timing harness."""
import os,subprocess
from pathlib import Path
b=Path('models/autoports/qwen_qwen3_8_27b');d=b/'doc/optimized_decoder'
for length, baseline in [(4096,False),(513,True),(513,False)]:
 for layer in (0,3):
  label=f'finalscreen_{"fused" if baseline else "default"}_{length}_l{layer}'
  command=['bash',str(b/'tests/optimized_run.sh'),'check',label,'--layer',str(layer),'--seq',str(length),'--recorded','--timing','--trace-prefill']
  if baseline:command+=['--baseline-fused']
  with (d/f'{label}.log').open('w') as log:r=subprocess.run(command,env={**os.environ,'QWEN_OPT_POLICY':'{}'},stdout=log,stderr=subprocess.STDOUT)
  print(label,r.returncode,flush=True)
  if r.returncode:raise SystemExit(r.returncode)
with (d/'reader_final_profile.log').open('w') as log:
 r=subprocess.run(['bash',str(b/'tests/profile_readers.sh'),'accepted'],env={**os.environ,'QWEN_OPT_POLICY':'{}'},stdout=log,stderr=subprocess.STDOUT)
if r.returncode:raise SystemExit(r.returncode)
# Match the final runner hash exactly; preserve earlier baseline captures.
for layer in (0,3):
 label=f'baseline_accepted_l{layer}'
 command=['bash',str(b/'tests/optimized_run.sh'),'profile',label,'--layer',str(layer),'--recorded','--baseline-fused','--trace-prefill']
 with (d/f'{label}.log').open('w') as log:r=subprocess.run(command,env={**os.environ,'QWEN_OPT_POLICY':'{}'},stdout=log,stderr=subprocess.STDOUT)
 print(label,r.returncode,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
