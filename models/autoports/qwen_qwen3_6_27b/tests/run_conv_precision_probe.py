# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Isolated real-tap decode convolution probe; no latency or full-model claim.

Compare a custom single-launch conv to the frozen primitive _batched_conv and
CPU four-product/left-associated FP32 sum -> BF16 -> SiLU reference. Torch IO is
only diagnostic, outside the production helper. This script opens the device;
run only through the parent stage's authorized serialized device queue.
"""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors import safe_open
import ttnn

ROOT=Path(__file__).resolve().parents[1]
CANDIDATES=ROOT/'doc/optimized_decoder/candidates'
MODEL='Qwen/Qwen3.6-27B'
REVISION='8f33b98937c6031bb49e713dee3e8e185aafd26f'


def load_source(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod
    spec.loader.exec_module(mod)
    return mod


def metric(reference,actual):
    assert reference.shape==actual.shape,(reference.shape,actual.shape)
    a,b=reference.double().flatten(),actual.double().flatten()
    ac,bc=a-a.mean(),b-b.mean();denom=ac.norm()*bc.norm()
    return {'pcc':float((ac*bc).sum()/denom) if denom>0 else None,
            'relative_rms':float((a-b).norm()/a.norm().clamp_min(1e-30)),
            'max_abs':float((a-b).abs().max()),
            'equal_fraction':float((a==b).double().mean()),
            'mismatched_values':int((a!=b).sum())}


def fingerprint(x):
    return hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--candidate',type=Path,default=CANDIDATES/'conv_decode_sfpu.py')
    ap.add_argument('--baseline',type=Path,default=CANDIDATES/'integrated57.py')
    ap.add_argument('--layer',type=int,default=0)
    ap.add_argument('--batch',type=int,nargs='+',default=[1,3,32])
    ap.add_argument('--capture',type=Path,default=ROOT/'doc/optimized_decoder/capture41/layer0_step209.pt')
    ap.add_argument('--steps',type=int,default=2)
    ap.add_argument('--cores',type=int)
    ap.add_argument('--require-baseline-exact',action='store_true')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    assert args.steps>0 and all(1<=b<=32 for b in args.batch)
    torch.set_num_threads(8);torch.manual_seed(6363)
    candidate=load_source(args.candidate,'conv_probe_candidate')
    baseline=load_source(args.baseline,'conv_probe_baseline')
    checkpoint=Path(snapshot_download(MODEL,revision=REVISION,local_files_only=True))
    key=f'model.language_model.layers.{args.layer}.linear_attn.conv1d.weight'
    mapping=json.loads((checkpoint/'model.safetensors.index.json').read_text())['weight_map']
    with safe_open(checkpoint/mapping[key],framework='pt',device='cpu') as f:
        weight=f.get_tensor(key).bfloat16()
    assert tuple(weight.shape)==(10240,1,4)
    taps=[weight[:,0,i].reshape(1,1,10240).contiguous() for i in range(4)]
    capture=torch.load(args.capture,weights_only=False,map_location='cpu') if args.capture.exists() else None
    policy={'conv_packed_rm':True,'conv_rm':True,'conv_round_output':True,
            'conv_functional':False,'conv_round_products':False}
    report={'diagnostic_only_no_perf_claim':True,'model':MODEL,'revision':REVISION,
            'candidate':str(args.candidate),'candidate_sha256':hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
            'baseline':str(args.baseline),'baseline_sha256':hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
            'conv_weight':key,'conv_weight_sha256':fingerprint(weight),'policy':policy,
            'capture':str(args.capture) if capture is not None else None,'rows':[]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,1),physical_device_ids=[0],trace_region_size=0)
    try:
        composer=ttnn.ConcatMeshToTensor(mesh,dim=0)
        def upload(x,layout):
            return ttnn.from_torch(x.bfloat16(),device=mesh,dtype=ttnn.bfloat16,layout=layout)
        def read(x):return ttnn.to_torch(x,mesh_composer=composer).detach().cpu().bfloat16().clone()
        device_taps=[upload(t,ttnn.ROW_MAJOR_LAYOUT) for t in taps]
        context=SimpleNamespace(policy=policy,taps_rm=device_taps)
        for batch in args.batch:
            cases=[]
            for scale in (0.,.01,1.,8.):
                token=(torch.randn(batch,10240)*scale).bfloat16()
                history=(torch.randn(batch,3,10240)*scale).bfloat16()
                cases.append((f'random_scale_{scale}',token,history))
            if capture is not None:
                for source in ('tt','hf'):
                    item=capture[source]
                    token=item['projection_qkv'].reshape(-1,10240).bfloat16()
                    history=item['conv_before'].reshape(-1,3,10240).bfloat16()
                    # Captured users repeat to cover every B without inventing
                    # new captured values. Random cases above have distinct users.
                    indices=torch.arange(batch)%token.shape[0]
                    cases.append((f'capture_{source}_users_repeated',token[indices],history[indices]))
            for name,token,history in cases:
                x=upload(token.unsqueeze(0),ttnn.TILE_LAYOUT)
                state=upload(history,ttnn.ROW_MAJOR_LAYOUT)
                baseline_state=upload(history,ttnn.ROW_MAJOR_LAYOUT)
                state_address=state.buffer_address()
                output=None
                expected_history=history.clone()
                for step in range(args.steps):
                    if step:
                        # New input in the same allocation checks runtime rebinding
                        # and real continuation; candidate reuses output allocation.
                        token=(token.float()+torch.randn_like(token.float())*.03125).bfloat16()
                        ttnn.copy(upload(token.unsqueeze(0),ttnn.TILE_LAYOUT),x)
                    products=[expected_history[:,i].float()*taps[i].reshape(1,-1).float() for i in range(3)]
                    products.append(token.float()*taps[3].reshape(1,-1).float())
                    preactivation=((products[0]+products[1])+products[2])+products[3]
                    cpu=F.silu(preactivation.bfloat16()).reshape(batch,1,10240)
                    expected_history=torch.cat((expected_history[:,1:3],token[:,None]),dim=1)
                    output=candidate.conv_decode_sfpu(x,state,device_taps,output=output,cores=args.cores)
                    native=baseline.OptimizedDecoder._batched_conv(context,x,baseline_state)[0]
                    actual,native_host,actual_history,native_history=map(read,(output,native,state,baseline_state))
                    assert state.buffer_address()==state_address
                    assert torch.isfinite(actual).all()
                    row={'batch':batch,'case':name,'step':step,'token_sha256':fingerprint(token),
                         'candidate_vs_cpu':metric(cpu,actual),'baseline_vs_cpu':metric(cpu,native_host),
                         'candidate_vs_baseline':metric(native_host,actual),
                         'candidate_history_vs_cpu':metric(expected_history,actual_history),
                         'baseline_history_vs_cpu':metric(expected_history,native_history),
                         'stable_history_allocation':True}
                    report['rows'].append(row);save()
                    print(json.dumps(row),flush=True)
                    assert torch.equal(actual_history,expected_history),'Candidate carry mismatch'
                    assert torch.equal(native_history,expected_history),'Primitive carry mismatch'
                    if args.require_baseline_exact:
                        assert torch.equal(actual,native_host),'Candidate output differs from primitive baseline'
        report['completed']=True
        report['all_baseline_bit_exact']=all(r['candidate_vs_baseline']['mismatched_values']==0 for r in report['rows'])
        save()
    finally:
        ttnn.close_mesh_device(mesh)


if __name__=='__main__':main()
