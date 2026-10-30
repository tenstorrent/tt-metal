# SPDX-License-Identifier: Apache-2.0
"""Stage05 branch, batch, state, stack and watcher checks; hardware is serialized."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time

p=argparse.ArgumentParser();p.add_argument('--suite',choices=['batch','boundary','context','stack','watcher','continuation','alternatives','packed','prefill_confirm'],required=True)
p.add_argument('--layout',default='phase_specific');args=p.parse_args()
base=Path('models/autoports/qwen_qwen3_8_27b/doc/optimized_multichip_decoder')
cases=[]
if args.suite=='prefill_confirm':
    for label,policy in [('control',{}),('k20',{'prefill_block':20}),('k20_mix40',{'prefill_block':20,'mix':{'cores':40,'block':4,'readers':2,'prefill_block':40}})]:
        cases.append((f'prefill_confirm_{label}_l3',['--layer','3','--policy',json.dumps(policy),'--prefill-repeats','20','--seq','4096','--steps','128']))
elif args.suite=='packed':
    for layer in (0,3):
        for packed in (False,True):
            label='packed' if packed else 'separate'
            cases.append((f'{label}_prefill_repeated_l{layer}',['--layer',str(layer),'--policy',json.dumps({'separate_prefill':not packed}),'--prefill-repeats','10','--seq','4096','--steps','128']))
elif args.suite=='alternatives':
    for layer in (0,3):
        for fused in ('rs','both'):
            cases.append((f'final_fused_{fused}_l{layer}',['--layer',str(layer),'--layout','sharded','--fused',fused,'--policy','{"fused_ag_1d":true}','--seq','4096','--steps','128']))
    cases.append(('mix_readers3_control',['--layer','0','--layout','replicated','--policy','{"mix":{"cores":80,"block":2,"readers":3}}','--seq','33','--steps','128']))
    cases.append(('mix40_readers3',['--layer','0','--policy','{"mix":{"cores":40,"block":4,"readers":3}}','--seq','33','--steps','128']))
elif args.suite=='batch':
    for batch in (32,16,8,4):
        for layer in (0,3):
            cases.append((f'batch{batch}_l{layer}',['--layer',str(layer),'--batch',str(batch),'--seq','4096','--steps','128']))
elif args.suite=='boundary':
    for layer in (0,3):
        chunk=4096
        for seq in (1,2,3,31,32,33,129,2049,chunk-1,chunk,chunk+1):
            cases.append((f'boundary{seq}_l{layer}',['--layer',str(layer),'--seq',str(seq),'--steps','2']))
    for layer,lengths in [(0,(426,427,4095,4097)),(3,(341,342,4095,4097))]:
        for seq in lengths:cases.append((f'odd_batch3_{seq}_l{layer}',['--layer',str(layer),'--seq',str(seq),'--batch','3','--steps','2']))
    for batch in (9,17):
        for layer in (0,3):
            cases.append((f'odd_batch{batch}_129_l{layer}',['--layer',str(layer),'--seq','129','--batch',str(batch),'--steps','8']))
    for batch in (5,7):
        cases.append((f'rope_boundary_b{batch}_l3',['--layer','3','--seq','129','--batch',str(batch),'--steps','8']))
    for batch in (8,32):
        cases.append((f'ragged_pages_b{batch}_l3',['--layer','3','--seq','129','--batch',str(batch),'--steps','8','--ragged','--remap']))
elif args.suite=='continuation':
    for layer in (0,3):
        for seq in (129,257):cases.append((f'continuation{seq}_l{layer}',['--layer',str(layer),'--seq',str(seq),'--split-prefill','128','--steps','8']))
elif args.suite=='context':
    for layer in (0,3):cases.append((f'context262142_l{layer}',['--layer',str(layer),'--seq','262142','--steps','2']))
elif args.suite=='stack':
    for layer in (0,3):
        for batch in (1,8,32):
            cases.append((f'stack_b{batch}_l{layer}',['--layer',str(layer),'--batch',str(batch),'--seq','129','--steps','16','--stack','--async-stress']))
else:
    for layer in (0,3):
        cases.append((f'watcher_l{layer}',['--layer',str(layer),'--seq','4096','--steps','128']))
        cases.append((f'watcher_b32_l{layer}',['--layer',str(layer),'--batch','32','--seq','33','--steps','8']))
env=os.environ.copy();env['TT_METAL_TRACE_ALLOC_TRACKING']='1';env['TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE']='0'
if args.suite=='watcher':
    env['TT_METAL_WATCHER']='10'
    env['TT_METAL_WATCHER_NOINLINE']='1'
else:env.pop('TT_METAL_WATCHER',None)
results=[]
for label,options in cases:
    cmd=[sys.executable,'-m','models.autoports.qwen_qwen3_8_27b.tests.run_multichip_decoder',
         '--layout',args.layout,'--output',str(base/f'{label}.json'),*options]
    if (base/f'{label}.json').exists(): raise FileExistsError(f'Preserve existing evidence: {label}')
    print('START',label,flush=True)
    with (base/f'{label}.log').open('w') as log:
        process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
        start=time.monotonic();limit=900 if args.suite=='context' else 360
        while process.poll() is None:
            if time.monotonic()-start>limit:
                triage=base/f'triage_{label}';triage.mkdir(exist_ok=True)
                subprocess.run(['timeout','120',sys.executable,'tools/tt-triage.py','--llm-output',
                    '--llm-output-path',str(triage/'tt-triage.txt'),'--run=dump_callstacks','--run=dump_running_operations'],stdout=log,stderr=subprocess.STDOUT)
                process.terminate()
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                break
            time.sleep(1)
    if args.suite=='watcher' and Path('generated/watcher/watcher.log').exists():
        shutil.copyfile('generated/watcher/watcher.log',base/f'{label}.watcher.txt')
    row={'case':label,'exit_code':process.returncode,'command':cmd,'log':str(base/f'{label}.log')}
    if process.returncode==0:
        evidence=json.loads((base/f'{label}.json').read_text())
        row.update(prefill_pcc=evidence['prefill_pcc'],minimum_decode_pcc=min(evidence['decode_pcc']))
    results.append(row);(base/f'{args.suite}_validation.json').write_text(json.dumps(results,indent=2)+'\n')
    print('RESULT',row,flush=True)
    if process.returncode:raise SystemExit(process.returncode)
