"""V2 isolated intermediate-aware real-weight beta/decay probe; run only under the device serial wrapper."""
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
from models.autoports.qwen_qwen3_6_27b.tests.run_prefill_norm_probe import metrics, MODEL, REVISION


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--candidate',type=Path,required=True)
    ap.add_argument('--layer',type=int,default=0)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();torch.set_num_threads(8);torch.manual_seed(1703)
    root=Path(snapshot_download(MODEL,revision=REVISION,local_files_only=True))
    index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
    weights={}
    for suffix in ('dt_bias','A_log'):
        name=f'model.language_model.layers.{args.layer}.linear_attn.{suffix}'
        with safe_open(root/index[name],framework='pt',device='cpu') as source:
            weights[suffix]=source.get_tensor(name).float().reshape(1,1,48)
    spec=importlib.util.spec_from_file_location('beta_decay_candidate',args.candidate)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    paths=[Path(__file__),args.candidate,
        Path('tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_sigmoid.h'),
        Path('tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_softplus.h'),
        Path('ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.cpp'),
        Path('ttnn/cpp/ttnn/operations/eltwise/binary/binary.cpp')]
    report={'diagnostic_only':True,'model':MODEL,'revision':REVISION,'layer':args.layer,
        'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'reference':'BF16 sigmoid; FP32 add(dt_bias), native FP32 softplus, multiply(uploaded neg_A)',
        'threshold':'exact logical bits for both outputs, repeat equal, stable caller output addresses','rows':[]}
    mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,1),physical_device_ids=[0],trace_region_size=0)
    try:
        def upload(x,dtype):
            return ttnn.from_torch(x,device=mesh,dtype=dtype,layout=ttnn.TILE_LAYOUT,memory_config=ttnn.DRAM_MEMORY_CONFIG)
        def read(x):return ttnn.to_torch(x,mesh_composer=ttnn.ConcatMeshToTensor(mesh,dim=0)).clone()
        bias=upload(weights['dt_bias'],ttnn.float32);neg=upload(-weights['A_log'].exp(),ttnn.float32)
        for batch in (1,3,31,32):
            outputs=None
            for draw in range(3):
                host=(torch.randn(1,batch,16512)* (1,4,20)[draw]).bfloat16()
                boundary=torch.tensor([-32,-20,-8,-1,-.00390625,0,.00390625,1,8,19.875,20,20.125,32],dtype=torch.bfloat16)
                host[:,:,16384:16397]=boundary;host[:,:,16448:16461]=boundary
                packed=upload(host,ttnn.bfloat16)
                reference_beta=ttnn.reshape(ttnn.sigmoid(packed[:,:,16384:16432]),(batch,1,48))
                value=ttnn.add(packed[:,:,16448:16496],bias,dtype=ttnn.float32)
                softplus=ttnn.softplus(value)
                reference_g=ttnn.reshape(ttnn.multiply(neg,softplus),(batch,1,48))
                outputs=module.gdn_beta_decay(packed,bias,neg,outputs=outputs)
                first=[read(x) for x in outputs];reference=[read(x) for x in (reference_beta,reference_g)]
                addresses=[x.buffer_address() for x in outputs]
                repeated=module.gdn_beta_decay(packed,bias,neg,outputs=outputs)
                second=[read(x) for x in repeated]
                row={'batch':batch,'draw':draw,'stable_addresses':addresses==[x.buffer_address() for x in repeated]}
                for i,name in enumerate(('beta','decay')):
                    row[name]={**metrics(reference[i],first[i]),'bit_equal':torch.equal(reference[i],first[i]),
                        'repeat_equal':torch.equal(first[i],second[i]),'finite':bool(torch.isfinite(first[i]).all())}
                row['intermediates']={}
                cpu_add=host[:,:,16448:16496].float()+weights['dt_bias']
                for stage,name,expected in ((1,'add',value),(2,'softplus',softplus)):
                    temporary=module.gdn_beta_decay(packed,bias,neg,diagnostic_stage=stage)
                    observed=read(temporary[1]);native=read(expected).reshape(batch,1,48)
                    details={**metrics(native,observed),'bit_equal':torch.equal(native,observed)}
                    if stage==1:
                        cpu=cpu_add.reshape(batch,1,48)
                        details['native_vs_cpu_fp32']=metrics(cpu,native)
                        details['fused_vs_cpu_fp32']=metrics(cpu,observed)
                        bad=(native!=observed).flatten().nonzero().flatten()[:8]
                        details['first_mismatches']=[{'index':int(i),'cpu':float(cpu.flatten()[i]),
                            'native':float(native.flatten()[i]),'fused':float(observed.flatten()[i])} for i in bad]
                    row['intermediates'][name]=details
                    del temporary
                report['rows'].append(row);print(json.dumps(row),flush=True)
                del packed,reference_beta,value,softplus,reference_g,repeated
    finally:
        ttnn.close_mesh_device(mesh);args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    assert len(report['rows'])==12
    assert all(r['stable_addresses'] and all(v['bit_equal'] for v in r['intermediates'].values()) and all(r[k]['bit_equal'] and r[k]['repeat_equal'] and r[k]['finite'] for k in ('beta','decay')) for r in report['rows']), 'Exact gate probe failed; inspect per-output evidence'


if __name__=='__main__':main()
