"""Exact native/candidate QKV split and physical-padding probe.

Parent serializes device execution. No whole-layer latency claim. Native
padding is inspected but unspecified; only candidate padding must be zero.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch
import ttnn

MODEL_ROOT=Path(__file__).resolve().parents[1]
REPO=Path(__file__).resolve().parents[4]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--candidate',type=Path,default=MODEL_ROOT/'doc/optimized_decoder/candidates/qkv_split_batched_staging.py')
    ap.add_argument('--batch',type=int,nargs='+',default=[1,3,31,32])
    ap.add_argument('--repeats',type=int,default=3)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    spec=importlib.util.spec_from_file_location('qkv_batched_probe_candidate',args.candidate)
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    torch.set_num_threads(8);torch.manual_seed(163163)
    native=REPO/'ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_decode/device/kernels/reader_interleaved_tm_tile_layout_nlp_create_qkv_heads_decode.cpp'
    report={'harness_sha256':digest(__file__),'candidate':str(args.candidate),'candidate_sha256':digest(args.candidate),
            'native_reader_sha256':digest(native),'rows':[],
            'scope':'logical exactness/core mapping/padding/changed input; no whole-decoder latency claim',
            'native_padding_contract':'unspecified; reported without equality requirement'}
    mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,1),physical_device_ids=[0],trace_region_size=0)
    try:
        def read(t,padded=False):
            t=ttnn.to_memory_config(t,ttnn.DRAM_MEMORY_CONFIG)
            if padded:
                t=ttnn.reshape(t,ttnn.Shape(t.padded_shape),ttnn.Shape(t.padded_shape))
            return ttnn.to_torch(t,mesh_composer=ttnn.ConcatMeshToTensor(mesh,dim=0))
        for batch in args.batch:
            assert 1<=batch<=32
            rows,tail=divmod(batch,8);ranges=[]
            if rows:ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0,0),ttnn.CoreCoord(7,rows-1)))
            if tail:ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0,rows),ttnn.CoreCoord(tail-1,rows)))
            mem=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED,ttnn.BufferType.L1,
                    ttnn.ShardSpec(ttnn.CoreRangeSet(ranges),[32,256],ttnn.ShardOrientation.ROW_MAJOR))
            for repeat in range(args.repeats):
                # Distinct user/channel data detects transposed8x4 placement,
                # odd-user64B alignment bugs, and stale program-cache addresses.
                x=torch.randn(1,1,batch,8192).bfloat16()
                x=(x.float()+torch.arange(batch).reshape(1,1,batch,1)*.125+repeat*.03125).bfloat16()
                token=ttnn.from_torch(x,device=mesh,dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,
                                      memory_config=ttnn.DRAM_MEMORY_CONFIG)
                baseline=ttnn.experimental.nlp_create_qkv_heads_decode(token,num_heads=24,num_kv_heads=4,memory_config=mem)
                candidate=mod.qkv_split_batched_staging(token,num_heads=24,num_kv_heads=4,memory_config=mem)
                for name,heads,start,old,new in zip(('Q','K','V'),(24,4,4),(0,6144,7168),baseline,candidate):
                    a,b=read(old),read(new)
                    expected=x[0,0,:,start:start+heads*256].reshape(1,batch,heads,256)
                    bp=read(new,True);apad=read(old,True)
                    padding=bp[:,:,heads:,:]
                    row={'batch':batch,'repeat':repeat,'tensor':name,
                         'native_logical_equal':torch.equal(a,b),'cpu_logical_equal':torch.equal(b,expected),
                         'memory_config_equal':old.memory_config()==new.memory_config()==mem,
                         'logical_shape_equal':tuple(old.shape)==tuple(new.shape),
                         'padded_shape_equal':tuple(old.padded_shape)==tuple(new.padded_shape),
                         'candidate_padding_nonzero':int(torch.count_nonzero(padding)),
                         'candidate_padding_finite':bool(torch.isfinite(padding).all()),
                         'native_padding_nonzero':int(torch.count_nonzero(apad[:,:,heads:,:]))}
                    report['rows'].append(row)
                del token,baseline,candidate,old,new
                args.output.write_text(json.dumps(report,indent=2)+'\n')
    finally:
        ttnn.close_mesh_device(mesh)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    assert report['rows'] and all(r['native_logical_equal'] and r['cpu_logical_equal'] and r['memory_config_equal']
                and r['logical_shape_equal'] and r['padded_shape_equal'] and r['candidate_padding_nonzero']==0
                and r['candidate_padding_finite'] for r in report['rows']), 'Inspect QKV split diagnostic JSON'
    print(json.dumps({'rows':len(report['rows']),'all_checks_passed':True,'output':str(args.output)}))


if __name__=='__main__':
    main()
