"""Serialized candidate runner with retained commands, source and numerical comparisons."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'doc/optimized_decoder'


def compare_outputs(control, candidate):
    a, b = [torch.load(p, weights_only=True) for p in (control, candidate)]
    out = {}
    for key in a:
        x, y = a[key].double().reshape(-1), b[key].double().reshape(-1)
        xc, yc = x-x.mean(), y-y.mean()
        out[key] = dict(pcc=float((xc@yc)/(xc.norm()*yc.norm())), exact=torch.equal(x,y),
                        rmse=float((x-y).square().mean().sqrt()), finite=bool(y.isfinite().all()))
    return out


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('matrix', help='JSON array of {tag,layer,args,control}')
    parser.add_argument('--continue-on-closed-error',action='store_true')
    parser.add_argument('--experimental',action='store_true',help='Use the isolated historical candidate runtime')
    parser.add_argument('--advice',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(8)
    for row in json.loads(Path(args.matrix).read_text()):
        tag=row['tag']; cmd=[sys.executable,'-m', 'models.autoports.qwen_qwen3_8_flash_next.tests.' + ('benchmark_optimized_advice' if args.advice else 'benchmark_optimized_candidates' if args.experimental else 'benchmark_optimized_decoder'),
            '--layer',str(row['layer']),'--tag',tag]+row.get('args',[])
        env=dict(os.environ,TT_METAL_TRACE_ALLOC_TRACKING='0')
        start=time.time()
        print('START',tag,flush=True)
        with (DEST/(tag+'.log')).open('w') as log:
            result=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
        record=dict(command=cmd,returncode=result.returncode,elapsed=time.time()-start)
        if result.returncode==0:
            timing=json.loads((DEST/tag/'timing.json').read_text())
            record.update(prefill_ms=timing['median_prefill_host_ms'],decode_ms=timing['median_traced_decode_ms'])
            if row.get('control'):
                base=json.loads((DEST/row['control']/'timing.json').read_text())
                record['comparison']=compare_outputs(base['outputs'],timing['outputs'])
        (DEST/(tag+'_result.json')).write_text(json.dumps(record,indent=2)+'\n')
        print(tag,json.dumps(record),flush=True)
        if result.returncode:
            print((DEST/(tag+'.log')).read_text()[-3500:],flush=True)
            if not args.continue_on_closed_error or 'Closing devices in cluster completed' not in (DEST/(tag+'.log')).read_text():
                break

if __name__=='__main__': main()
