# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Isolated same-input prefill RMSNorm accuracy with real folded HF gamma.

Diagnostic only: no model, latency claim, or runtime fallback. Run under the
existing serial device wrapper. Candidate must export the requested function.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
import ttnn

MODEL = 'Qwen/Qwen3.6-27B'
REVISION = '8f33b98937c6031bb49e713dee3e8e185aafd26f'
SHAPES = ((1,1),(1,17),(1,33),(1,65),(3,33),(32,33),(32,128))


def metrics(reference, actual):
    assert reference.shape == actual.shape
    a,b = reference.double().flatten(),actual.double().flatten()
    ac,bc = a-a.mean(),b-b.mean()
    denominator=ac.norm()*bc.norm()
    return {'pcc':float((ac*bc).sum()/denominator) if denominator>0 else None,
            'relative_rms':float((a-b).norm()/a.norm().clamp_min(1e-30)),
            'max_abs':float((a-b).abs().max()),
            'equal_fraction':float((a==b).double().mean())}


def load_gamma(layer):
    root=Path(snapshot_download(MODEL,revision=REVISION,local_files_only=True))
    mapping=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
    result={}
    for key in ('input_layernorm.weight','post_attention_layernorm.weight'):
        name=f'model.language_model.layers.{layer}.{key}'
        with safe_open(root/mapping[name],framework='pt',device='cpu') as source:
            result[key]=(1+source.get_tensor(name).float()).reshape(1,1,5120)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--candidate',type=Path,required=True)
    ap.add_argument('--function',default='prefill_hidden_rmsnorm_sfpu')
    ap.add_argument('--layer',type=int,default=0)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(194)
    spec=importlib.util.spec_from_file_location('prefill_norm_candidate',args.candidate)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    operation=getattr(module,args.function)
    gamma=load_gamma(args.layer)
    eps=json.loads((Path(__file__).parents[1]/'doc/hf_config.json').read_text())['text_config']['rms_norm_eps']
    report={'diagnostic_only_no_perf_claim':True,'model':MODEL,'revision':REVISION,'layer':args.layer,
            'candidate':str(args.candidate),'candidate_sha256':hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
            'harness_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'function':args.function,'epsilon':eps,'seed':194,'input':'BF16 randn * 0.1, same input for CPU and TT',
            'gamma':'real pinned checkpoint folded as FP32 (1 + weight)',
            'threshold':'relative RMS < 1e-4, PCC >= .995, exact repeated execution and stable supplied output',
            'rows':[]}
    mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,1),physical_device_ids=[0],trace_region_size=0)
    try:
        composer=ttnn.ConcatMeshToTensor(mesh,dim=0)
        device_gamma={key:ttnn.from_torch(value,device=mesh,dtype=ttnn.float32,layout=ttnn.TILE_LAYOUT,
                       memory_config=ttnn.DRAM_MEMORY_CONFIG) for key,value in gamma.items()}
        for batch,seq in SHAPES:
            host=(torch.randn(batch,seq,5120,dtype=torch.bfloat16)*0.1).contiguous()
            x=ttnn.from_torch(host,device=mesh,dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for key,weight in device_gamma.items():
                xf=host.float()
                expected=(xf*torch.rsqrt(xf.square().mean(-1,keepdim=True)+eps)*gamma[key]).bfloat16()
                output=operation(x,weight,epsilon=eps)
                assert tuple(output.shape)==tuple(host.shape)
                assert output.dtype==ttnn.bfloat16
                address=output.buffer_address()
                first=ttnn.to_torch(output,mesh_composer=composer).clone()
                repeated=operation(x,weight,epsilon=eps,output=output)
                assert repeated.buffer_address()==address
                second=ttnn.to_torch(repeated,mesh_composer=composer).clone()
                row={'batch':batch,'seq':seq,'weight':key,**metrics(expected,first),
                     'repeat_equal':torch.equal(first,second),'supplied_output_address_stable':True,
                     'logical_shape':list(output.shape),'padded_shape':list(output.padded_shape)}
                report['rows'].append(row)
                print(row,flush=True)
                del output,repeated,first,second,expected,xf
            del x,host
    finally:
        ttnn.close_mesh_device(mesh)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    assert len(report['rows'])==len(SHAPES)*len(gamma)
    assert all(r['relative_rms']<1e-4 and (r['pcc'] is None or r['pcc']>=.995)
               and r['repeat_equal'] for r in report['rows']), 'Prefill norm precision/determinism gate failed; see JSON'


if __name__=='__main__':main()
