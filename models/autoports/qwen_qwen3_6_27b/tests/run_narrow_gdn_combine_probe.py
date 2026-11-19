"""Same-input native FP32 add -> BF16 bit probe for narrow GDN combine.

Diagnostic only. Execute via serial device wrapper; no performance claims.
"""
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path
import torch
import ttnn
from models.autoports.qwen_qwen3_6_27b.tests.run_projection_precision_probe import compare

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--candidate',type=Path,required=True)
    ap.add_argument('--function',default='narrow_gdn_combine')
    ap.add_argument('--cores',type=int)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();torch.set_num_threads(8);torch.manual_seed(907)
    spec=importlib.util.spec_from_file_location('narrow_gdn_combine_candidate',args.candidate)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    operation=getattr(module,args.function)
    cases=[(1,1,False),(1,1,True),(1,17,False),(1,17,True),(1,32,False),(1,32,True),(1,33,False),(3,33,False),(1,641,False)]
    report={'diagnostic_only_no_perf_claim':True,'candidate':str(args.candidate),
        'candidate_sha256':hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
        'harness_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'seed':907,
        'reference':'native ttnn.add on SAME FP32 primary plus reconstructed full zero-Z residual, output BF16',
        'threshold':'exact BF16 bits, repeated output, stable output address','rows':[]}
    mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,1),physical_device_ids=[0],trace_region_size=0)
    try:
        def read(t):
            if t.is_sharded():t=ttnn.to_memory_config(t,ttnn.DRAM_MEMORY_CONFIG)
            return ttnn.to_torch(t,mesh_composer=ttnn.ConcatMeshToTensor(mesh,dim=0)).clone()
        def upload(t):return ttnn.from_torch(t,device=mesh,dtype=ttnn.float32,layout=ttnn.TILE_LAYOUT,memory_config=ttnn.DRAM_MEMORY_CONFIG)
        grid=ttnn.num_cores_to_corerangeset(40,mesh.compute_with_storage_grid_size(),row_wise=True)
        pmem=ttnn.create_sharded_memory_config((32,416),grid,ttnn.ShardStrategy.WIDTH,use_height_and_width_as_shard_shape=True)
        rmem=ttnn.create_sharded_memory_config((32,288),grid,ttnn.ShardStrategy.WIDTH,use_height_and_width_as_shard_shape=True)
        for batch,seq,sharded in cases:
            ph=torch.randn(batch,seq,16640,dtype=torch.float32)
            rh=torch.randn(batch,seq,11520,dtype=torch.float32)/64
            corners=torch.tensor([-0.0,0.0,1+2**-8,1+3*2**-8,-1-2**-8,-1-3*2**-8,2**-20,-2**-20],dtype=torch.float32)
            ph[...,:8]=corners;ph[...,10240:10248]=corners;ph[...,16384:16392]=corners
            rh[...,:8]=torch.tensor([0.0,-0.0,0.0,0.0,0.0,0.0,-2**-20,2**-20]);rh[...,10240:10248]=0
            full=torch.zeros_like(ph);full[...,:10240]=rh[...,:10240];full[...,16384:16512]=rh[...,10240:10368]
            p,r,f=upload(ph),upload(rh),upload(full)
            if sharded:p,r,f=ttnn.to_memory_config(p,pmem),ttnn.to_memory_config(r,rmem),ttnn.to_memory_config(f,pmem)
            expected=ttnn.add(p,f,dtype=ttnn.bfloat16)
            output=operation(p,r,cores=args.cores);actual=read(output);reference=read(expected)[...,:16512].contiguous()
            address=output.buffer_address();output=operation(p,r,output=output,cores=args.cores);again=read(output)
            row={'batch':batch,'seq':seq,'L1_width_sharded':sharded,
                'bit_exact':torch.equal(actual.contiguous().view(torch.int16),reference.view(torch.int16)),
                'repeat_exact':torch.equal(actual.contiguous().view(torch.int16),again.contiguous().view(torch.int16)),
                'stable_address':output.buffer_address()==address,'comparisons':compare(reference,actual,'packed_gdn')}
            report['rows'].append(row);print(batch,seq,sharded,row['bit_exact'],row['comparisons']['all_valid_channels'],flush=True)
            del p,r,f,output,expected,actual,reference,again,ph,rh,full
    finally:
        ttnn.close_mesh_device(mesh);args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    assert len(report['rows'])==len(cases)
    assert all(r['bit_exact'] and r['repeat_exact'] and r['stable_address'] for r in report['rows'])

if __name__=='__main__':main()
