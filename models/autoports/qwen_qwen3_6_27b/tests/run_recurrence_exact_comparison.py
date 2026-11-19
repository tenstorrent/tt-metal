"""Device exact comparison of frozen recurrence helpers; parent-serialized only."""
import argparse
import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import torch
import ttnn

ROOT=Path(__file__).resolve().parents[1]/'doc/optimized_decoder/candidates'

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--candidate',type=Path,default=ROOT/'gdn_sfpu_update_readout.py')
    ap.add_argument('--reference',type=Path,default=ROOT/'gdn_sfpu_decay_predict.py')
    ap.add_argument('--batches',type=int,nargs='+',default=[1,3,32])
    ap.add_argument('--state-dtypes',nargs='+',choices=['bf16','fp32'],default=['bf16'])
    ap.add_argument('--steps',type=int,default=3);ap.add_argument('--cores',type=int)
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    candidate=load(args.candidate,'update_readout_candidate');reference=load(args.reference,'decay_reference')
    torch.set_num_threads(8);gen=torch.Generator().manual_seed(12047)
    report=dict(candidate=str(args.candidate),reference=str(args.reference),
        candidate_sha256=hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
        reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        require_bit_exact=True,cores=args.cores,cases=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,1),physical_device_ids=[0],trace_region_size=0)
    try:
        def upload(tensor,layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(tensor,device=mesh,dtype=ttnn.bfloat16 if tensor.dtype==torch.bfloat16 else ttnn.float32,layout=layout)
        def read(tensor):return ttnn.to_torch(tensor,mesh_composer=ttnn.ConcatMeshToTensor(mesh,dim=0)).contiguous()
        def mismatches(a,b):
            assert a.dtype==b.dtype and a.shape==b.shape
            integer=torch.int16 if a.dtype==torch.bfloat16 else torch.int32
            return int((a.view(integer)!=b.view(integer)).sum())
        for batch in args.batches:
            for state_dtype in args.state_dtypes:
                dtype=torch.bfloat16 if state_dtype=='bf16' else torch.float32
                for mode in ('normal','zero_kv','zero_state'):
                    initial=(torch.randn((batch*48,128,128),generator=gen)*.03).to(dtype)
                    if mode=='zero_state':initial.zero_()
                    state_ref,state_actual=upload(initial),upload(initial)
                    out_ref=out_actual=None
                    for step in range(args.steps):
                        input_state=initial if step==0 else sr.clone()
                        packed=torch.randn((batch,1,10240),generator=gen).bfloat16()
                        if mode=='zero_kv':packed[...,2048:].zero_()
                        g=(-torch.rand((batch,1,48),generator=gen)*.1).float()
                        beta=torch.rand((batch,1,48),generator=gen).bfloat16()
                        data=upload(packed,ttnn.ROW_MAJOR_LAYOUT);gate=upload(g);beta_t=upload(beta)
                        out_ref=reference._recurrent_sfpu_compact(data,data,data,gate,beta_t,state_ref,
                            normalized=False,packed_qkv=True,output=out_ref,cores=args.cores)
                        out_actual=candidate._recurrent_sfpu_compact(data,data,data,gate,beta_t,state_actual,
                            normalized=False,packed_qkv=True,output=out_actual,cores=args.cores)
                        yr,ya,sr,sa=map(read,(out_ref,out_actual,state_ref,state_actual))
                        row=dict(batch=batch,state_dtype=state_dtype,mode=mode,step=step,
                            output_bit_mismatches=mismatches(yr,ya),state_bit_mismatches=mismatches(sr,sa))
                        report['cases'].append(row);print(row,flush=True)
                        args.output.write_text(json.dumps(report,indent=2)+'\n')
                        if row['output_bit_mismatches'] or row['state_bit_mismatches']:
                            torch.save(dict(packed=packed,g=g,beta=beta,input_state=input_state,
                                output_ref=yr,output_actual=ya,state_ref=sr,state_actual=sa),args.output.with_suffix('.failure.pt'))
                            raise AssertionError('Recurrence candidate must match reference output/state bits exactly')
        report['completed']=True;args.output.write_text(json.dumps(report,indent=2)+'\n')
    finally:ttnn.close_mesh_device(mesh)

if __name__=='__main__':main()
