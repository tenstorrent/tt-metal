# SPDX-License-Identifier: Apache-2.0
"""External sparse-pool preparation isolation; run only with serialized TT access.

Default: real layers0/3, B8/context512. --full-stack defaults to B32/context33024
and prepares the complete physical bucket domain. No timing is benchmark evidence.
"""
import argparse
import hashlib
import os
import platform
import subprocess
from datetime import datetime, timezone
import json
from pathlib import Path
import torch
import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.model import Model
from models.autoports.qwen_qwen3_8_27b.tt.generator import Generator


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--full-stack',action='store_true')
    p.add_argument('--batch',type=int)
    p.add_argument('--context',type=int)
    p.add_argument('--checkpoint')
    p.add_argument('--output',default='models/autoports/qwen_qwen3_8_27b/doc/full_model/external_prepare.json')
    a=p.parse_args()
    batch=a.batch or (32 if a.full_stack else 8)
    context=a.context or (33024 if a.full_stack else 512)
    checkpoint=a.checkpoint or ('.cache/qwen38-full' if a.full_stack else '.cache/qwen38-probe')
    result=dict(status='running',batch=batch,context=context,full_stack=a.full_stack,checks={})
    source_root=Path(__file__).resolve().parents[1]
    result['provenance']=dict(started_utc=datetime.now(timezone.utc).isoformat(),
        python=platform.python_version(),torch=str(torch.__version__),ttnn=str(getattr(ttnn,'__version__','unknown')),
        checkpoint=checkpoint,git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        trace_alloc_tracking=os.environ.get('TT_METAL_TRACE_ALLOC_TRACKING'),
        trace_skip_program_cache=os.environ.get('TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE','0'),
        source_sha256={str(path.relative_to(source_root)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [Path(__file__),*[source_root/'tt'/name for name in
                ('generator.py','model.py','multichip_decoder.py','optimized_decoder.py')]]})
    if (os.environ.get('TT_METAL_TRACE_ALLOC_TRACKING')!='1' or
            os.environ.get('TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE','0')!='0'):
        raise RuntimeError('Run with TT_METAL_TRACE_ALLOC_TRACKING=1 and TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0')
    target=Path(a.output);target.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(8)
    ttnn.CONFIG.throw_exception_on_fallback=True
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,4),trace_region_size=0)
    try:
        print('EXTERNAL_BUILD',flush=True)
        model=Model(checkpoint,mesh,layer_indices=None if a.full_stack else [0,3])
        cache=model.allocate_cache(batch_size=batch,context_length=context)
        pages=((context+255)//256)*8
        sparse=torch.full((batch,pages),-1,dtype=torch.int32)
        sparse[0,:8]=torch.arange(8,dtype=torch.int32)
        device_map=model.put(sparse,dtype=ttnn.int32,layout=ttnn.ROW_MAJOR_LAYOUT)
        gen=Generator(model,checkpoint,batch_size=batch,context_length=context,kv_cache=cache,page_table=device_map)
        # Deterministic nonzero sentinels; no host copy of the whole KV pool.
        for state in cache:
            for tensor in vars(state).values():
                ttnn.copy(ttnn.full_like(tensor,0.125),tensor)
        gen.refresh(torch.full((1,1,1,batch),101,dtype=torch.int32),gen.tokens)
        gen.refresh(torch.full((batch,),7,dtype=torch.int32),gen.positions)
        gen.refresh(torch.full((batch,1),7,dtype=torch.int32),gen.rope_indices)
        gen.set_sampling_params(seed=9876)
        addresses=[[v.buffer_address() for v in vars(s).values()] for s in cache]
        def host(t):
            return [ttnn.to_torch(local).clone() for local in ttnn.get_device_tensors(t)]
        def sampled_cache():
            # Sample active slot0 and inactive slot1 (or an unused KV page).
            rows={}
            for layer,state in enumerate(cache):
                for name,tensor in vars(state).items():
                    for index in (0,8 if name in ('key','value') else 1):
                        if index>=tensor.shape[0]:
                            continue
                        start=[0]*len(tensor.shape);start[0]=index
                        end=list(tensor.shape);end[0]=index+1
                        rows[(layer,name,index)]=host(ttnn.slice(tensor,start,end,[1]*len(end)))
            return rows
        before=sampled_cache()  # Also warm these diagnostic slice signatures.
        small={name:host(getattr(gen,name)) for name in
               ('tokens','positions','rope_indices','active_mask','seeds','logits','prefill_output')}
        def preserved(label):
            assert gen.kv_cache is cache and gen.page_table is device_map
            assert addresses==[[v.buffer_address() for v in vars(s).values()] for s in cache]
            current=sampled_cache()
            assert all(torch.equal(x,y) for key in before for x,y in zip(before[key],current[key]))
            assert all(torch.equal(x,sparse) for x in host(device_map))
            for name,values in small.items():
                assert all(torch.equal(x,y) for x,y in zip(values,host(getattr(gen,name)))),name
            result['checks'][label]=True
        lengths=None if a.full_stack else (1,33,65)
        original_decode=gen._decode
        def fail_after_scratch_decode(*args,**kwargs):
            assert gen.kv_cache is not cache and gen.page_table is not device_map
            original_decode(*args,**kwargs)
            raise RuntimeError('injected external preparation failure')
        print('EXTERNAL_INJECT_FAILURE',flush=True)
        gen._decode=fail_after_scratch_decode
        try:
            gen.prepare_decode(prefill_lengths=lengths)
        except RuntimeError as error:
            assert str(error)=='injected external preparation failure'
        else:
            raise AssertionError('Failure injection did not propagate')
        finally:
            gen._decode=original_decode
        preserved('failure_preserves_sampled_cache_map_request_and_rng')
        assert gen.model_trace is None and not gen.decode_traces and not gen._warming
        result['failure_memory']=gen.last_preparation_memory
        if not a.full_stack:
            print('EXTERNAL_CAPTURE_END_FAILURE',flush=True)
            original_end=ttnn.end_trace_capture
            original_release=ttnn.release_trace
            released=[]
            ended=[]
            injected=False
            def track_release(*args,**kwargs):
                original_release(*args,**kwargs)
                released.append(str(args[1]))
            def fail_after_record_end(*args,**kwargs):
                nonlocal injected
                original_end(*args,**kwargs)
                ended.append(str(args[1]))
                if not injected:
                    injected=True
                    raise RuntimeError('injected failure after capture record_end')
            ttnn.end_trace_capture=fail_after_record_end
            ttnn.release_trace=track_release
            try:
                gen.prepare_decode(prefill_lengths=lengths)
            except RuntimeError as error:
                assert str(error)=='injected failure after capture record_end'
            else:
                raise AssertionError('Capture-boundary injection did not propagate')
            finally:
                ttnn.end_trace_capture=original_end
                ttnn.release_trace=original_release
            assert injected and ended and all(trace in released for trace in ended)
            preserved('capture_end_failure_preserves_original_cache_map_request_and_rng')
            assert gen.model_trace is None and gen.sample_trace is None and gen.buffered_sample_trace is None and not gen.decode_traces
            assert not gen._warming
            result['capture_boundary_cleanup_errors']=gen.last_preparation_cleanup_errors
            result['checks']['capture_end_failure_retains_original_exception_and_clears_trace_state']=True
        print('EXTERNAL_PREPARE',flush=True)
        gen.prepare_decode(prefill_lengths=lengths)
        preserved('success_preserves_sampled_cache_map_request_and_rng')
        result['preparation_memory']=gen.last_preparation_memory
        specs=gen.last_preparation_memory['scratch_specs']
        assert {entry['field'] for entry in specs}=={'key','value','conv','recurrent'}
        if not a.full_stack:
            assert len(specs)==4
        print('EXTERNAL_REPLAY_ORIGINAL',flush=True)
        ids={str(k):str(v) for k,v in gen.decode_traces.items()}
        captures=gen.counters['captures']
        for step in range(3):
            out=gen.decode_forward(torch.full((batch,1),101,dtype=torch.int64),
                torch.full((batch,),step,dtype=torch.int32),page_table=device_map,kv_cache=cache,
                active_slots=[0],return_logits=False)
            assert out.shape==(batch,) and not torch.count_nonzero(out[1:])
            assert all(value.reshape(-1).tolist()==[step+1]+[-1]*(batch-1) for value in host(gen.positions))
            current=sampled_cache()
            for key in before:
                if key[2]==0:
                    for device,(x,y) in enumerate(zip(before[key],current[key])):
                        assert not torch.equal(x,y),('Original active cache did not change',key,device,step)
            assert all(torch.equal(x,y) for key in before if key[2]!=0
                       for x,y in zip(before[key],current[key]))
            assert ids=={str(k):str(v) for k,v in gen.decode_traces.items()}
            assert captures==gen.counters['captures']
        result['checks']['three_replays_update_original_active_cache_and_positions_without_recapture']=True
        result['checks']['three_replays_preserve_original_inactive_slots_and_unused_pages']=True
        result.update(status='pass',trace_ids=ids,limitations=[
            'Cache checks sample active/inactive slots and physical pages on all devices, not the complete pool.',
            'Allocator checkpoints are not peak working-set measurements.',
            'No accuracy or performance claim.'])
        print('EXTERNAL_PASS',flush=True)
    except BaseException as error:
        result.update(status='fail',error_type=type(error).__name__,error=str(error)[:2000])
        raise
    finally:
        target.write_text(json.dumps(result,indent=2)+'\n')
        ttnn.close_mesh_device(mesh)


if __name__=='__main__':
    main()
