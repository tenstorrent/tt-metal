"""Attribute native norm conversion rounding, with same-input CPU oracles.

Device work is run only by the parent's serialized queue. No latency claim.
Every mode consumes the same native FP32 RMSNorm output. Recoverable API
errors are recorded per mode and do not prevent collecting other controls.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
import ttnn

MODEL='Qwen/Qwen3.6-27B'
REVISION='8f33b98937c6031bb49e713dee3e8e185aafd26f'
ROOT=Path(__file__).resolve().parents[4]
MODEL_ROOT=Path(__file__).resolve().parents[1]
SOURCE_FILES=(
 'ttnn/cpp/ttnn/operations/data_movement/sharded/sharded_to_interleaved/device/sharded_to_interleaved_program_factory.cpp',
 'ttnn/cpp/ttnn/kernel/compute/eltwise_copy_metal2.cpp',
 'ttnn/cpp/ttnn/operations/copy/typecast/typecast.cpp',
 'ttnn/cpp/ttnn/operations/data_movement/clone/device/clone_program_factory.cpp',
 'ttnn/cpp/ttnn/operations/data_movement/clone/device/kernels/compute_kernel.cpp',
 'ttnn/cpp/ttnn/operations/data_movement/copy/device/copy_same_memory_config_program_factory.cpp',
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metric(a,b):
    a,b=a.float(),b.float()
    return {'bit_equal':torch.equal(a,b),'mismatched_values':int((a!=b).sum()),
            'max_abs':float((a-b).abs().max()),
            'relative_rms':float((a-b).norm()/b.norm().clamp_min(1e-30))}


def cpu_oracles(value):
    value=value.contiguous().float()
    bits=value.view(torch.int32).to(torch.int64)&0xffffffff
    trunc_bf16=(bits&0xffff0000).to(torch.int32).view(torch.float32)
    trunc_tf32=(bits&0xffffe000).to(torch.int32).view(torch.float32)
    return {'fp32_to_bf16_rne':value.bfloat16().float(),
            'fp32_to_bf16_truncate':trunc_bf16,
            'fp32_to_tf32_truncate_then_bf16_rne':trunc_tf32.bfloat16().float()}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--layer',type=int,default=3)
    ap.add_argument('--batch',type=int,nargs='+',default=[1,3,32])
    ap.add_argument('--candidate',type=Path,default=MODEL_ROOT/'doc/optimized_decoder/candidates/norm_l1_handoff_v2_on134.py')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--require-clone-exact',action='store_true')
    args=ap.parse_args()
    torch.set_num_threads(8);torch.manual_seed(145145)
    checkpoint=Path(snapshot_download(MODEL,revision=REVISION,local_files_only=True))
    config=json.loads((checkpoint/'config.json').read_text());config=config.get('text_config',config)
    assert config['layer_types'][args.layer]=='full_attention'
    index=json.loads((checkpoint/'model.safetensors.index.json').read_text())['weight_map']
    gammas={}
    for name in ('input_layernorm.weight','post_attention_layernorm.weight'):
        key=f'model.language_model.layers.{args.layer}.{name}'
        with safe_open(checkpoint/index[key],framework='pt',device='cpu') as f:
            gammas[name]=(1+f.get_tensor(key).float()).reshape(1,1,5120)
    report={'model':MODEL,'revision':REVISION,'layer':args.layer,
            'harness':str(Path(__file__).resolve()),'harness_sha256':sha(__file__),
            'candidate':str(args.candidate.resolve()),'candidate_sha256':sha(args.candidate),
            'source_sha256':{name:sha(ROOT/name) for name in SOURCE_FILES},
            'candidate_role':'provenance only; probe executes explicit isolated API modes',
            'scope':'same-FP32 norm conversion attribution; no decoder PCC or performance claim','rows':[]}
    mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,1),physical_device_ids=[0],trace_region_size=0)
    try:
        mem=ttnn.create_sharded_memory_config((32,640),ttnn.CoreGrid(x=8,y=1),ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,use_height_and_width_as_shard_shape=True)
        norm_ck=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,fp32_dest_acc_en=True,packer_l1_acc=True)
        copy_ck=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,fp32_dest_acc_en=False,packer_l1_acc=False)
        program=ttnn.LayerNormShardedMultiCoreProgramConfig(compute_with_storage_grid_size=(8,1),
                subblock_w=4,block_h=1,block_w=20,inplace=False)
        def read(t):
            return ttnn.to_torch(ttnn.to_memory_config(t,ttnn.DRAM_MEMORY_CONFIG),
                       mesh_composer=ttnn.ConcatMeshToTensor(mesh,dim=0)).float()
        for name,gamma in gammas.items():
            weight=ttnn.from_torch(gamma,device=mesh,dtype=ttnn.float32,layout=ttnn.TILE_LAYOUT,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for batch in args.batch:
                assert 1<=batch<=32
                for scale in (0.,.001,1.,32.):
                    x=(torch.randn(1,batch,5120)*scale).bfloat16()
                    token=ttnn.from_torch(x,device=mesh,dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,
                                          memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    xf=ttnn.interleaved_to_sharded(token,mem,output_dtype=ttnn.float32)
                    normalized=ttnn.rms_norm(xf,weight=weight,epsilon=config['rms_norm_eps'],
                                             compute_kernel_config=norm_ck,program_config=program)
                    fp32=read(normalized)
                    oracles=cpu_oracles(fp32)
                    baseline=ttnn.sharded_to_interleaved(normalized,ttnn.DRAM_MEMORY_CONFIG,
                                                        output_dtype=ttnn.bfloat16)
                    expected=read(baseline)
                    row={'norm':name,'batch':batch,'scale':scale,
                         'baseline_vs_cpu':{k:metric(expected,v) for k,v in oracles.items()},'modes':{}}
                    modes={
                        'typecast_fp32':lambda:ttnn.typecast(normalized,ttnn.bfloat16,memory_config=mem),
                        'clone_legacy_copy':lambda:ttnn.clone(normalized,dtype=ttnn.bfloat16,
                                                              memory_config=mem,compute_kernel_config=copy_ck),
                        'assign_legacy_copy':lambda:ttnn.assign(normalized,memory_config=mem,dtype=ttnn.bfloat16),
                    }
                    for mode,operation in modes.items():
                        try:
                            result=operation();actual=read(result)
                            reused=ttnn.to_memory_config(result,mem)
                            row['modes'][mode]={'baseline':metric(actual,expected),
                               'cpu':{k:metric(actual,v) for k,v in oracles.items()},
                               'projection_alias':reused.buffer_address()==result.buffer_address(),
                               'memory_config_match':result.memory_config()==mem}
                            del result,reused
                        except Exception as exc:
                            row['modes'][mode]={'error':type(exc).__name__+': '+str(exc)}
                    report['rows'].append(row)
                    args.output.write_text(json.dumps(report,indent=2)+'\n')
                    del token,xf,normalized,baseline
            del weight
    finally:
        ttnn.close_mesh_device(mesh)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    if args.require_clone_exact:
        assert report['rows'] and all(r['modes'].get('clone_legacy_copy',{}).get('baseline',{}).get('bit_equal',False)
                                     for r in report['rows']), 'clone differs; inspect diagnostic JSON'
    print(json.dumps({'rows':len(report['rows']),'output':str(args.output)}))


if __name__=='__main__':
    main()
