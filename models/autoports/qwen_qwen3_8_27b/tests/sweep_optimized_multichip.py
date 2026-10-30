# SPDX-License-Identifier: Apache-2.0
"""Serialized same-precision whole-layer geometry screens; retains each result."""
import json
import argparse
import subprocess
import sys
import time
from pathlib import Path
base=Path('models/autoports/qwen_qwen3_8_27b/doc/optimized_multichip_decoder')
parser=argparse.ArgumentParser();parser.add_argument('--runner',default='run_multichip_decoder');parser.add_argument('--layers',default='0,3');parser.add_argument('--batch',type=int,default=1);parser.add_argument('--name',default='geometry');parser.add_argument('--seq',type=int,default=33);parser.add_argument('--steps',type=int,default=8);parser.add_argument('--prefill-repeats',type=int,default=1);args=parser.parse_args()
policies=json.loads((base/f'{args.name}_candidates.json').read_text())
results=[]
for layer in map(int,args.layers.split(',')):
    for name,policy in policies.items():
        label=f'{args.name}_{name}_l{layer}'
        policy=dict(policy)
        extra=[]
        for option in ('ragged','remap'):
            if policy.pop('_'+option,False): extra += ['--'+option]
        if '_capacity_alignment' in policy: extra += ['--capacity-alignment',str(policy.pop('_capacity_alignment'))]
        if '_l1_background_bytes' in policy: extra += ['--l1-background-bytes',str(policy.pop('_l1_background_bytes'))]
        if '_layout' in policy: extra += ['--layout',policy.pop('_layout')]
        if policy.pop('_output_ag',False): extra += ['--output-ag']
        if '_fused' in policy: extra += ['--fused',policy.pop('_fused')]
        cmd=[sys.executable,'-m',f'models.autoports.qwen_qwen3_8_27b.tests.{args.runner}',
             '--layer',str(layer),'--batch',str(args.batch),'--seq',str(args.seq),'--steps',str(args.steps),'--prefill-repeats',str(args.prefill_repeats),'--policy',json.dumps(policy),'--output',str(base/f'{label}.json')]+extra
        if (base/f'{label}.json').exists(): raise FileExistsError(f'Use a fresh sweep name: {label}')
        print('START',label,flush=True)
        with (base/f'{label}.log').open('w') as log:
            process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
            start=time.monotonic()
            while process.poll() is None:
                if time.monotonic()-start>(900 if args.seq>32768 else 360 if args.batch>1 else 120):
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
        elif (base/f'{label}.json').exists():
            # The runner writes JSON only after both mesh finally blocks close.
            # A numerical acceptance failure does not require a hardware reset.
            row['recovery']='not_required_meshes_closed_before_numerical_gate'
        elif process.returncode==1 and 'Closing devices in cluster completed' in (base/f'{label}.log').read_text() and any(message in (base/f'{label}.log').read_text() for message in ('Statically allocated circular buffers','Statically allocated dataflow buffers','Out of Memory: Not enough space to allocate')):
            # These are host resource-admission failures before the rejected
            # program launches. A completed close proves the prior work drained.
            row['recovery']='not_required_host_capacity_failure_mesh_closed'
        else:
            # Device/API failures get bounded recovery before another experiment.
            with (base/f'{label}_reset.log').open('w') as log:
                reset=subprocess.run(['timeout','180','tt-smi','-r'],stdout=log,stderr=subprocess.STDOUT)
                listing=subprocess.run(['timeout','60','tt-smi','-ls','--local'],stdout=log,stderr=subprocess.STDOUT)
            row['reset_exit']=reset.returncode;row['list_exit']=listing.returncode
            if reset.returncode or listing.returncode:
                results.append(row);(base/f'{args.name}_results.json').write_text(json.dumps(results,indent=2)+'\n')
                raise RuntimeError('Hardware recovery needs inspection')
        results.append(row);(base/f'{args.name}_results.json').write_text(json.dumps(results,indent=2)+'\n')
        print('RESULT',row,flush=True)
