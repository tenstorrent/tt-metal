# SPDX-License-Identifier: Apache-2.0
"""Deferred reduced owned-cache + caller sparse host/device-map preparation probe."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import torch
import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.model import Model
from models.autoports.qwen_qwen3_8_27b.tt.generator import Generator


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--map-kind',choices=('host','device','refreshed','mutated'),required=True)
    parser.add_argument('--checkpoint',default='.cache/qwen38-probe')
    parser.add_argument('--output')
    args=parser.parse_args()
    if (os.environ.get('TT_METAL_TRACE_ALLOC_TRACKING')!='1' or
        os.environ.get('TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE','0')!='0'):
        raise RuntimeError('Require allocation tracking1 with program cache included0')
    root=Path(__file__).resolve().parents[1]
    target=Path(args.output) if args.output else root/'doc/full_model'/f'owned_sparse_{args.map_kind}.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    result=dict(status='running',map_kind=args.map_kind,scope='layers0/3,B8,context512',checks={},
        source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (Path(__file__),root/'tt/generator.py')},
        runtime=dict(torch=str(torch.__version__),ttnn=str(getattr(ttnn,'__version__','unknown')),
                     trace_alloc_tracking=1,trace_skip_program_cache=0))
    torch.set_num_threads(8)
    ttnn.CONFIG.throw_exception_on_fallback=True
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,4),trace_region_size=0)
    try:
        model=Model(args.checkpoint,mesh,layer_indices=[0,3])
        sparse=torch.full((8,16),-1,dtype=torch.int32);sparse[0,:8]=torch.arange(8)
        supplied=(sparse.clone() if args.map_kind=='host' else
                  model.put(sparse,dtype=ttnn.int32,layout=ttnn.ROW_MAJOR_LAYOUT)
                  if args.map_kind=='device' else None)
        gen=Generator(model,args.checkpoint,batch_size=8,context_length=512,page_table=supplied)
        assert gen.owns_cache
        if args.map_kind=='refreshed':
            gen.refresh_page_table(sparse)
        elif args.map_kind=='mutated':
            # Direct tensor mutation bypasses the generator's host shadow.
            gen.refresh(sparse,gen.page_table)
        bound_map=gen.page_table;bound_cache=gen.kv_cache
        def host(t):
            return [ttnn.to_torch(local) for local in ttnn.get_device_tensors(t)]
        def check_map():
            assert gen.page_table is bound_map and gen.kv_cache is bound_cache
            assert all(torch.equal(value,sparse) for value in host(bound_map))
            if args.map_kind=='host':
                assert torch.equal(supplied,sparse)
            elif args.map_kind=='device':
                assert gen.page_table is supplied
        def cache_samples():
            values={}
            for i,state in enumerate(bound_cache):
                for name,tensor in vars(state).items():
                    for row in (0,8 if name in ('key','value') else 1):
                        start=[0]*len(tensor.shape);start[0]=row
                        end=list(tensor.shape);end[0]=row+1
                        values[i,name,row]=host(ttnn.slice(tensor,start,end,[1]*len(end)))
            return values
        cache_samples()  # Warm diagnostic slice signatures before capture.
        original_decode=gen._decode
        def failing_decode(*args_,**kwargs_):
            assert gen.page_table is not bound_map and gen.kv_cache is bound_cache
            original_decode(*args_,**kwargs_)
            raise RuntimeError('injected owned-cache scratch-map failure')
        print('OWNED_SPARSE_FAILURE',args.map_kind,flush=True)
        gen._decode=failing_decode
        try:
            gen.prepare_decode(prefill_lengths=(1,33,65))
        except RuntimeError as error:
            assert str(error)=='injected owned-cache scratch-map failure'
        else:
            raise AssertionError('Expected warmup failure')
        finally:
            gen._decode=original_decode
        check_map()
        assert gen.model_trace is None and not gen._warming
        result['checks']['failure_restores_custom_map_binding_and_contents']=True
        print('OWNED_SPARSE_PREPARE',args.map_kind,flush=True)
        gen.prepare_decode(prefill_lengths=(1,33,65))
        check_map()
        assert not gen.last_preparation_memory['scratch_specs']
        assert all(not torch.count_nonzero(v) for values in cache_samples().values() for v in values)
        result['checks']['owned_cache_reset_semantics_preserved']=True
        ids={str(k):str(v) for k,v in gen.decode_traces.items()};captures=gen.counters['captures']
        print('OWNED_SPARSE_REPLAY',args.map_kind,flush=True)
        for step in range(3):
            tokens=gen.decode_forward(torch.full((8,1),101,dtype=torch.int64),
                torch.full((8,),step,dtype=torch.int32),page_table=bound_map,kv_cache=bound_cache,
                active_slots=[0],return_logits=False)
            assert not torch.count_nonzero(tokens[1:])
            assert all(v.reshape(-1).tolist()==[step+1]+[-1]*7 for v in host(gen.positions))
            samples=cache_samples()
            assert all(torch.count_nonzero(v) for key,values in samples.items() if key[2]==0 for v in values)
            assert all(not torch.count_nonzero(v) for key,values in samples.items() if key[2]!=0 for v in values)
            assert ids=={str(k):str(v) for k,v in gen.decode_traces.items()} and captures==gen.counters['captures']
        check_map()
        result['checks']['three_replays_use_bound_map_and_update_each_active_sample_only']=True
        result.update(status='pass',preparation_memory=gen.last_preparation_memory,
                      limitations=['Reduced layers; cache checks sample active/inactive rows, not complete pools.'])
        print('OWNED_SPARSE_PASS',args.map_kind,flush=True)
    except BaseException as error:
        result.update(status='fail',error_type=type(error).__name__,error=str(error)[:2000])
        raise
    finally:
        target.write_text(json.dumps(result,indent=2)+'\n')
        ttnn.close_mesh_device(mesh)


if __name__=='__main__':
    main()
