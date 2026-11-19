"""Diagnostic-only real-gamma comparison to the unchanged TTNN post-norm chain.

Authoring this runner does not execute hardware. Run under the serial device
wrapper. Readbacks belong to correctness setup/probing, not measured runtime.
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
from models.autoports.qwen_qwen3_6_27b.tests.run_prefill_norm_probe import metrics,MODEL,REVISION


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--candidate',type=Path,required=True)
    ap.add_argument('--function',default='gdn_affine_gate_flat')
    ap.add_argument('--layer',type=int,default=0)
    ap.add_argument('--cores',type=int)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();torch.set_num_threads(8);torch.manual_seed(619)
    root=Path(snapshot_download(MODEL,revision=REVISION,local_files_only=True))
    mapping=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
    name=f'model.language_model.layers.{args.layer}.linear_attn.norm.weight'
    with safe_open(root/mapping[name],framework='pt',device='cpu') as source:
        weight=source.get_tensor(name).bfloat16().reshape(1,1,128)
    spec=importlib.util.spec_from_file_location('gdn_affine_gate_candidate',args.candidate)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    operation=getattr(module,args.function)
    report={'diagnostic_only_no_perf_claim':True,'model':MODEL,'revision':REVISION,'layer':args.layer,
        'candidate':str(args.candidate),'candidate_sha256':hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
        'harness_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'seed':619,
        'reference':'same-input native RMSNorm BF16 output -> BF16 gamma multiply -> reshape -> FP32 SiLU/gate -> BF16',
        'threshold':'bit-exact against TTNN chain plus stable repeated supplied output','rows':[]}
    mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,1),physical_device_ids=[0],trace_region_size=0)
    try:
        read=lambda t:ttnn.to_torch(t,mesh_composer=ttnn.ConcatMeshToTensor(mesh,dim=0)).clone()
        upload=lambda x:ttnn.from_torch(x,device=mesh,dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,memory_config=ttnn.DRAM_MEMORY_CONFIG)
        gamma=upload(weight)
        ckc=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,fp32_dest_acc_en=True,packer_l1_acc=True)
        for batch in (1,3,31,32):
            output=None
            for width in (16512,17408):
                for draw in range(2):
                    raw=upload(torch.randn(1,batch*48,128,dtype=torch.bfloat16))
                    norm=ttnn.rms_norm(raw,epsilon=1e-6,compute_kernel_config=ckc)
                    host_packed=(torch.randn(1,batch,width,dtype=torch.bfloat16)*(1 if draw==0 else 4)).contiguous()
                    # Include saturated and near-zero gate cases without changing shape.
                    host_packed[:,:,10240:10248]=torch.tensor([-32,-8,-1,-.00390625,0,.00390625,8,32],dtype=torch.bfloat16)
                    packed=upload(host_packed)
                    scaled=ttnn.multiply(norm,gamma,dtype=ttnn.bfloat16)
                    scaled=ttnn.reshape(scaled,(1,batch,6144))
                    expected=ttnn.multiply(ttnn.typecast(packed[:,:,10240:16384],ttnn.float32),
                        ttnn.typecast(scaled,ttnn.float32),dtype=ttnn.bfloat16,
                        input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
                    output=operation(norm,gamma,packed,output=output,cores=args.cores)
                    first=read(output);reference=read(expected);address=output.buffer_address()
                    repeated=operation(norm,gamma,packed,output=output,cores=args.cores)
                    second=read(repeated)
                    row={'batch':batch,'packed_width':width,'draw':draw,**metrics(reference,first),
                        'bit_equal':torch.equal(reference,first),'repeat_equal':torch.equal(first,second),
                        'stable_output_address':repeated.buffer_address()==address,
                        'logical_shape':list(output.shape),'padded_shape':list(output.padded_shape)}
                    report['rows'].append(row);print(row,flush=True)
                    del raw,norm,packed,scaled,expected,repeated
    finally:
        ttnn.close_mesh_device(mesh)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    assert len(report['rows'])==16
    assert all(r['bit_equal'] and r['repeat_equal'] and r['stable_output_address'] for r in report['rows']), 'Post-norm fused probe failed; inspect saved metrics'


if __name__=='__main__':main()
