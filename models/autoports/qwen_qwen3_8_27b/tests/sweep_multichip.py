# SPDX-License-Identifier: Apache-2.0
"""Serialized same-precision whole-layer geometry screens; retains each result."""
import json
import argparse
import subprocess
import sys
import time
from pathlib import Path
base=Path('models/autoports/qwen_qwen3_8_27b/doc/multichip_decoder')
parser=argparse.ArgumentParser();parser.add_argument('--name',default='geometry');parser.add_argument('--seq',type=int,default=33);parser.add_argument('--steps',type=int,default=8);parser.add_argument('--prefill-repeats',type=int,default=1);args=parser.parse_args()
policies=json.loads((base/f'{args.name}_candidates.json').read_text())
results=[]
for layer in (0,3):
    for name,policy in policies.items():
        label=f'geometry_{name}_l{layer}'
        cmd=[sys.executable,'-m','models.autoports.qwen_qwen3_8_27b.tests.run_multichip_decoder',
             '--layer',str(layer),'--seq',str(args.seq),'--steps',str(args.steps),'--prefill-repeats',str(args.prefill_repeats),'--policy',json.dumps(policy),'--output',str(base/f'{label}.json')]
        print('START',label,flush=True)
        with (base/f'{label}.log').open('w') as log:
            process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
            start=time.monotonic()
            while process.poll() is None:
                if time.monotonic()-start>120:
                    triage=base/f'triage_{label}';triage.mkdir(exist_ok=True)
                    subprocess.run(['timeout','120',sys.executable,'tools/tt-triage.py','--llm-output',
                        '--llm-output-path',str(triage/'tt-triage.txt'),'--run=dump_callstacks','--run=dump_running_operations'],stdout=log,stderr=subprocess.STDOUT)
                    process.terminate()
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:process.kill();process.wait()
                    break
                time.sleep(1)
        row={'layer':layer,'candidate':name,'policy':policy,'exit_code':process.returncode,'command':cmd}
        if process.returncode==0:
            value=json.loads((base/f'{label}.json').read_text())
            row.update(prefill_host_us=value['multichip']['prefill_host_us'],host_us=value['multichip']['median_decode_host_us'],minimum_pcc=min(value['prefill_pcc'],*value['decode_pcc']))
        else:
            # Every failed multi-chip run gets a bounded recovery before another experiment.
            with (base/f'{label}_reset.log').open('w') as log:
                reset=subprocess.run(['timeout','180','tt-smi','-r'],stdout=log,stderr=subprocess.STDOUT)
                listing=subprocess.run(['timeout','60','tt-smi','-ls','--local'],stdout=log,stderr=subprocess.STDOUT)
            row['reset_exit']=reset.returncode;row['list_exit']=listing.returncode
            if reset.returncode or listing.returncode:
                results.append(row);(base/f'{args.name}_results.json').write_text(json.dumps(results,indent=2)+'\n')
                raise RuntimeError('Hardware recovery needs inspection')
        results.append(row);(base/f'{args.name}_results.json').write_text(json.dumps(results,indent=2)+'\n')
        print('RESULT',row,flush=True)
