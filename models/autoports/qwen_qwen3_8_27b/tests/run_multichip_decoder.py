# SPDX-License-Identifier: Apache-2.0
"""Identical-input single-chip TTNN vs TP4 comparison, with mutable traced decode."""
import argparse
import json
import time
import hashlib
import os
import math
from pathlib import Path
import torch
import ttnn
from safetensors.torch import load_file
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
from models.autoports.qwen_qwen3_8_27b.tt.optimized_decoder import OptimizedDecoder, LinearState
from models.autoports.qwen_qwen3_8_27b.tt.multichip_decoder import MultichipDecoder
from models.autoports.qwen_qwen3_8_27b.tests.run_optimized_decoder import pcc, audit_runtime


class DecoderStack:
    def __init__(self,models): self.models=models
    def allocate_state(self,**kwargs): return tuple(m.allocate_state(**kwargs) for m in self.models)
    def snapshot(self,state): return tuple(m.snapshot(s) for m,s in zip(self.models,state))
    def prefill_forward(self,x,*,state,**kwargs):
        for m,s in zip(self.models,state): x=m.prefill_forward(x,state=s,**kwargs)
        return x
    def decode_forward(self,x,*,state,**kwargs):
        for m,s in zip(self.models,state): x=m.decode_forward(x,state=s,**kwargs)
        return x


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--layer',type=int,default=0)
    p.add_argument('--seq',type=int,default=33)
    p.add_argument('--batch',type=int,default=1)
    p.add_argument('--steps',type=int,default=2)
    p.add_argument('--prefill-repeats',type=int,default=1)
    p.add_argument('--capacity-alignment',type=int,default=256)
    p.add_argument('--l1-background-bytes',type=int,default=0)
    p.add_argument('--layout',default='phase_specific')
    p.add_argument('--fused',choices=['ag','rs','both'])
    p.add_argument('--output-ag',action='store_true')
    p.add_argument('--async-stress',action='store_true')
    p.add_argument('--remap',action='store_true')
    p.add_argument('--stack',action='store_true')
    p.add_argument('--split-prefill',type=int,default=0)
    p.add_argument('--policy',default='{}')
    p.add_argument('--baseline-policy',default='{}',help='Optional same-precision single-chip control; default preserves prior BFP4 control')
    p.add_argument('--ragged',action='store_true')
    p.add_argument('--profile',action='store_true')
    p.add_argument('--output',required=True)
    p.add_argument('--baseline-only',action='store_true')
    p.add_argument('--multichip-only',action='store_true')
    args=p.parse_args()
    assert not (args.async_stress and (args.profile or args.remap))
    if args.profile:
        assert args.seq==4096 and args.steps==128 and (args.baseline_only or args.multichip_only)
        from tracy import signpost
    torch.set_num_threads(8); torch.manual_seed(982)
    root=Path(__file__).resolve().parents[1]
    cfg=AutoConfig.from_pretrained(str(root/'doc/functional_decoder/hf_config.json')).text_config
    weights=load_file(f'.cache/qwen38-weights/layer_{args.layer}.safetensors')
    recorded=torch.load(f'.cache/qwen38-weights/recorded_layer_{args.layer}.pt',weights_only=True)
    def inputs(length,start=0):
        idx=(torch.arange(start,start+length)[None,:]+17*torch.arange(args.batch)[:,None])%recorded.shape[1]
        return recorded[0,idx]
    x=inputs(args.seq)
    rope=Qwen3_5TextRotaryEmbedding(cfg)
    with torch.no_grad():
        cos,sin=rope(x,torch.arange(args.seq).expand(3,args.batch,-1))
        dec=[]
        for i in range(args.steps):
            xd=inputs(1,args.seq+i)
            positions=args.seq+i-(torch.arange(args.batch)%11 if args.ragged else torch.zeros(args.batch,dtype=torch.long))
            cd,sd=rope(xd,positions.reshape(1,args.batch,1).expand(3,-1,-1))
            dec.append((xd,cd,sd))
    assert args.capacity_alignment%256==0
    capacity=((args.seq+args.steps+args.capacity_alignment-1)//args.capacity_alignment)*args.capacity_alignment
    pt=torch.randperm(args.batch*capacity//32,dtype=torch.int32).reshape(args.batch,-1)
    outputs={}; state_outputs={}; result={'capacity_tokens':capacity,'capacity_alignment':args.capacity_alignment,'layer_type':cfg.layer_types[args.layer],'seq':args.seq,'batch':args.batch,'steps':args.steps,'layout':args.layout,'fused':args.fused,'stack':args.stack,'split_prefill':args.split_prefill,'policy':json.loads(args.policy),'provenance':{str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),root/'tt/multichip_decoder.py',root/'tt/optimized_decoder.py',root/'tt/decode_workspace.py',root/'tt/fused_decode.py',root/'tt/fused_prefill.py']}}
    multi_cls=MultichipDecoder
    if args.fused:
        from .multichip_fused_candidate import FusedCandidate
        multi_cls=FusedCandidate
        candidate_file=root/"tests/multichip_fused_candidate.py"
        result["provenance"]["tests/multichip_fused_candidate.py"]=hashlib.sha256(candidate_file.read_bytes()).hexdigest()
    if args.output_ag:
        from .multichip_output_ag_candidate import OutputGatherCandidate
        multi_cls=OutputGatherCandidate
        candidate_file=root/'tests/multichip_output_ag_candidate.py'
        result['output_ag']=True
        result['provenance']['tests/multichip_output_ag_candidate.py']=hashlib.sha256(candidate_file.read_bytes()).hexdigest()
    choices=[('baseline',OptimizedDecoder,1),('multichip',multi_cls,4)]
    for name,cls,n in choices:
        if args.baseline_only and n!=1 or args.multichip_only and n!=4: continue
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED if n==1 else ttnn.FabricConfig.FABRIC_1D_RING)
        mesh=ttnn.open_mesh_device(ttnn.MeshShape(1,n),physical_device_ids=[0] if n==1 else [],trace_region_size=0)
        tid=None; tid2=None
        try:
            def allocated():
                view=ttnn.get_memory_view(mesh,ttnn.BufferType.DRAM)
                return {'allocated_bytes':view.total_bytes_allocated_per_bank*view.num_banks,
                        'capacity_bytes':view.total_bytes_per_bank*view.num_banks}
            def l1_view():
                view=ttnn.get_memory_view(mesh,ttnn.BufferType.L1)
                return {'reported_allocated_bytes_per_bank':view.total_bytes_allocated_per_bank,
                    'reported_capacity_bytes_per_bank':view.total_bytes_per_bank,'reported_num_banks':view.num_banks}
            memory={'before_weights':allocated()}
            model=cls.from_state_dict(weights,hf_config=cfg,layer_idx=args.layer,mesh_device=mesh,
                **({'policy':{'residual_layout':args.layout,**json.loads(args.policy)}} if n==4 else {'policy':json.loads(args.baseline_policy)}))
            if args.stack:
                assert not args.fused and not args.remap
                other=3 if args.layer==0 else 0
                next_model=cls.from_state_dict(load_file(f'.cache/qwen38-weights/layer_{other}.safetensors'),
                    hf_config=cfg,layer_idx=other,mesh_device=mesh,
                    **({'policy':{'residual_layout':args.layout,**json.loads(args.policy)},'ccl_workspace':model.ar_workspace} if n==4 else {'policy':json.loads(args.baseline_policy)}))
                model=DecoderStack([model,next_model])
            if n==4 and args.fused: model.prepare_fused(args.batch,args.fused)
            background_l1=None
            if n==4 and args.l1_background_bytes:
                tiles=math.ceil(args.l1_background_bytes/2048)
                background_l1=ttnn.empty([1,1,32,tiles*32],device=mesh,dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,memory_config=ttnn.L1_MEMORY_CONFIG)
                memory['synthetic_background_l1_bytes']=tiles*2048
            if not args.profile:
                model.prefill_forward=audit_runtime(model.prefill_forward)
                model.decode_forward=audit_runtime(model.decode_forward)
            memory['after_weights']=allocated()
            memory['l1_after_weights']=l1_view()
            mapper=ttnn.ReplicateTensorToMesh(mesh)
            first_model=model.models[0] if args.stack else model
            decode_layout=first_model.decode_residual_layout(args.batch) if n==4 else 'replicated'
            def sharded_phase(phase):
                return n==4 and (decode_layout=='sharded' if phase=='decode' else args.layout in ('sharded','phase_specific'))
            def put(t,dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,hidden=False,phase="decode"):
                mp=ttnn.ShardTensorToMesh(mesh,dim=2) if hidden and sharded_phase(phase) else mapper
                return ttnn.from_torch(t.contiguous(),device=mesh,dtype=dtype,layout=layout,mesh_mapper=mp)
            def read(t,phase="decode"):
                ts=ttnn.get_device_tensors(t)
                if sharded_phase(phase): return torch.cat([ttnn.to_torch(z) for z in ts],dim=-1)
                values=[ttnn.to_torch(z) for z in ts]
                if len(values)>1:
                    assert all(torch.equal(values[0],z) for z in values[1:]), 'replica divergence'
                return values[0]
            state=model.allocate_state(batch_size=args.batch,context_length=capacity)
            empty=model.snapshot(state)
            def copy_state(source,destination):
                if isinstance(source,tuple):
                    for a,b in zip(source,destination): copy_state(a,b)
                elif isinstance(source,LinearState):
                    ttnn.copy(source.conv,destination.conv);ttnn.copy(source.recurrent,destination.recurrent)
                else:
                    ttnn.copy(source.key,destination.key);ttnn.copy(source.value,destination.value)
            table=put(pt,ttnn.int32,ttnn.ROW_MAJOR_LAYOUT)
            xt,ct,st=put(x,hidden=True,phase="prefill"),put(cos[:,None]),put(sin[:,None])
            prefill_parts=[]
            if args.split_prefill:
                assert 0 < args.split_prefill < args.seq and args.split_prefill % 128 == 0
                for begin,end in [(0,args.split_prefill),(args.split_prefill,args.seq)]:
                    prefill_parts.append((begin,put(x[:,begin:end],hidden=True,phase="prefill"),
                        put(cos[:,None,begin:end,:]),put(sin[:,None,begin:end,:])))
            def prefill():
                if not prefill_parts:
                    return model.prefill_forward(xt,state=state,page_table=table,cos=ct,sin=st)
                parts=[model.prefill_forward(px,state=state,start_position=begin,page_table=table,cos=pc,sin=ps)
                       for begin,px,pc,ps in prefill_parts]
                return ttnn.concat(parts,dim=1)
            pre=prefill()
            pre_host=read(pre,phase="prefill")
            del pre  # Host copy is complete; capture must not retain an unrelated eager output.
            print(name,'EAGER_PREFILL_DONE',flush=True)
            copy_state(empty,state)
            pre_tid=ttnn.begin_trace_capture(mesh,cq_id=0)
            warm_pre=prefill()
            ttnn.end_trace_capture(mesh,pre_tid,cq_id=0)
            copy_state(empty,state)
            prefill_times=[]
            for repeat in range(args.prefill_repeats):
                copy_state(empty,state)
                ttnn.synchronize_device(mesh)
                if args.profile and repeat==args.prefill_repeats-1: signpost('PERF_PREFILL')
                start=time.perf_counter();ttnn.execute_trace(mesh,pre_tid,cq_id=0,blocking=True)
                prefill_times.append((time.perf_counter()-start)*1e6)
                if args.profile and repeat==args.prefill_repeats-1: signpost('PERF_PREFILL_END')
            prefill_host_us=sorted(prefill_times)[len(prefill_times)//2]
            assert torch.equal(pre_host,read(warm_pre,phase="prefill")), 'prefill replay mismatch'
            ttnn.release_trace(mesh,pre_tid)
            # Prefill outputs have been compared and their trace retired.
            # Decode owns separate inputs/state; do not retain phase-sized L1.
            del warm_pre
            if args.profile: ttnn.ReadDeviceProfiler(mesh)
            print(name,'PREFILL_DONE',flush=True)
            backup=model.snapshot(state)
            def restore(): copy_state(backup,state)
            xd,cd,sd=dec[0]
            dt,dc,ds=put(xd,hidden=True),put(cd[:,None]),put(sd[:,None])
            pos=put(torch.full((args.batch,),args.seq,dtype=torch.int32),ttnn.int32,ttnn.ROW_MAJOR_LAYOUT)
            def forward(): return model.decode_forward(dt,state=state,page_table=table,current_pos=pos,cos=dc,sin=ds)
            updates=[]
            for i,(xd,cd,sd) in enumerate(dec):
                updates.append([(put(xd,hidden=True),dt),(put(cd[:,None]),dc),(put(sd[:,None]),ds),
                    (put((args.seq+i-(torch.arange(args.batch)%11 if args.ragged else torch.zeros(args.batch,dtype=torch.long))).to(torch.int32),ttnn.int32,ttnn.ROW_MAJOR_LAYOUT),pos)])
            remap_data=None
            if args.remap:
                assert cfg.layer_types[args.layer]=='full_attention'
                permutation=torch.randperm(pt.numel())
                remapped_table=put(permutation[pt.long()].to(torch.int32),ttnn.int32,ttnn.ROW_MAJOR_LAYOUT)
                remapped_caches=[]
                for cache in (state.key,state.value):
                    local=[ttnn.to_torch(z) for z in ttnn.get_device_tensors(cache)]
                    remapped=[]
                    for value in local:
                        new=torch.empty_like(value);new[permutation]=value;remapped.append(new)
                    joined=torch.cat(remapped,dim=1)
                    remapped_caches.append(ttnn.from_torch(joined,device=mesh,dtype=cache.dtype,
                        layout=ttnn.TILE_LAYOUT,memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=ttnn.ShardTensorToMesh(mesh,dim=1)))
                remap_data=(remapped_table,remapped_caches)
                ttnn.copy(remapped_table,table)
                for source,destination in zip(remapped_caches,(state.key,state.value)): ttnn.copy(source,destination)
                ttnn.copy(put(pt,ttnn.int32,ttnn.ROW_MAJOR_LAYOUT),table);restore()
            for source,destination in updates[0]: ttnn.copy(source,destination)
            eager=forward();read(eager);restore()
            if args.async_stress:
                # Capture 2 retains capture 1's output. Prepare the same L1
                # liveness before any trace exists; native layout selectors
                # can choose another program signature as free L1 changes.
                retained_warm=forward();read(retained_warm)
                del retained_warm
            del eager
            restore();ttnn.synchronize_device(mesh)
            print('DECODE_WARM_DONE', mesh.num_program_cache_entries(), flush=True)
            tid=ttnn.begin_trace_capture(mesh,cq_id=0)
            traced=forward()
            ttnn.end_trace_capture(mesh,tid,cq_id=0)
            if args.async_stress:
                tid2=ttnn.begin_trace_capture(mesh,cq_id=0)
                traced2=forward()
                ttnn.end_trace_capture(mesh,tid2,cq_id=0)
                # Only this later output survives capture2. It may be clobbered
                # by trace1; consume it only after trace2 and before trace1.
                # All state, inputs, scratch and programs were prepared earlier.
                from ttnn.tools import trace_allocation_tracker
                trace_allocation_tracker.acknowledge_corruptible(traced2)
            restore()
            step_outputs=[];times=[]
            for i,(xd,cd,sd) in enumerate(dec):
                for source,destination in updates[i]: ttnn.copy(source,destination)
                ttnn.synchronize_device(mesh)
                if args.profile and i==args.steps-1: signpost('PERF_DECODE')
                start=time.perf_counter();ttnn.execute_trace(mesh,tid,cq_id=0,blocking=True)
                times.append((time.perf_counter()-start)*1e6)
                if args.profile and i==args.steps-1: signpost('PERF_DECODE_END')
                step_outputs.append(read(traced))
                if args.profile and (i+1)%16==0: ttnn.ReadDeviceProfiler(mesh)
            memory['after_trace']=allocated()
            memory['l1_after_trace']=l1_view()
            if not args.stack and not args.profile:
                def state_host(t,field):
                    values=[ttnn.to_torch(z) for z in ttnn.get_device_tensors(t)]
                    if n==1: return values[0]
                    if field=='conv':
                        return torch.cat([torch.cat([v[...,a:b] for v in values],dim=-1)
                            for a,b in [(0,512),(512,1024),(1024,2560)]],dim=-1)
                    return torch.cat(values,dim=1)
                names=('conv','recurrent') if isinstance(state,LinearState) else ('key','value')
                state_outputs[name]={field:state_host(getattr(state,field),field) for field in names}
            outputs[name]=(pre_host,step_outputs)
            for source,destination in updates[0]: ttnn.copy(source,destination)
            repeats=[]
            for _ in range(3):
                restore();ttnn.execute_trace(mesh,tid,cq_id=0,blocking=True);repeats.append(read(traced))
            assert all(torch.equal(repeats[0],v) for v in repeats[1:]), 'restored replay nondeterminism'
            assert torch.equal(step_outputs[0],repeats[0]), 'changed-input replay mismatch'
            if remap_data:
                restore()
                ttnn.copy(remap_data[0],table)
                for source,destination in zip(remap_data[1],(state.key,state.value)): ttnn.copy(source,destination)
                ttnn.execute_trace(mesh,tid,cq_id=0,blocking=True)
                assert torch.equal(step_outputs[0],read(traced)), 'page remap under live trace mismatch'
            stress=None
            if args.async_stress:
                for source,destination in updates[0]: ttnn.copy(source,destination)
                restore()
                for repeat in range(32):
                    ttnn.execute_trace(mesh,tid if repeat%2==0 else tid2,cq_id=0,blocking=True)
                expected=read(traced2)
                restore()
                for repeat in range(32):
                    ttnn.execute_trace(mesh,tid if repeat%2==0 else tid2,cq_id=0,blocking=False)
                ttnn.synchronize_device(mesh)
                assert torch.equal(expected,read(traced2)), 'alternating asynchronous trace replay differs'
                stress={'trace_ids':[str(tid),str(tid2)],'captures':2,'alternating_replays_per_arm':32,
                    'blocking_vs_nonblocking_bitwise':True,'acknowledged_output':'traced2 only; consumed after trace2 before any trace1',
                    'program_cache_allocations_included':os.environ.get('TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE','0')=='0'}
            watcher_replays=0
            if os.environ.get('TT_METAL_WATCHER'):
                for source,destination in updates[0]: ttnn.copy(source,destination)
                deadline=time.monotonic()+11
                while time.monotonic()<deadline:
                    restore()
                    if remap_data:
                        for source,destination in zip(remap_data[1],(state.key,state.value)): ttnn.copy(source,destination)
                    ttnn.execute_trace(mesh,tid,cq_id=0,blocking=True)
                    watcher_replays+=1
                assert torch.equal(step_outputs[0],read(traced)), 'watcher repeated replay mismatch'
            result[name]={'watcher_stress_replays':watcher_replays,'async_stress':stress,'page_remap_exact' :bool(remap_data),'ragged_positions':args.ragged,'last_decode_host_us':times[-1], 'decode_host_us':times, 'median_decode_host_us':sorted(times)[len(times)//2], 'prefill_host_us':prefill_host_us,'prefill_host_times_us':prefill_times,
                'restored_replays_bitwise':3,'replica_equality':n==4 and decode_layout=='replicated',
                'runtime_fallback_audit':not args.profile,'memory':memory,
                'effective_policy':[m.policy for m in model.models] if args.stack else model.policy}
            owners=model.models if args.stack else [model]
            if n==4:result[name]['unique_ccl_contexts']=len({id(m.ccl) for m in owners})
            decode_owners=[m.decode_implementation(args.batch) for m in owners] if n==4 else owners
            result[name]['decode_residual_layout']=decode_layout
            result[name]['effective_decode_policy']=[m.policy for m in decode_owners] if args.stack else decode_owners[0].policy
            result[name]['effective_sdpa_chunks']=[getattr(m,'effective_sdpa_chunk',None) for m in decode_owners]
            workspaces=[getattr(m,'ar_workspace',None) for m in owners]
            if n==4 and any(workspaces):
                unique={id(w):w for w in workspaces if w is not None}
                assert all(m.ccl is w.ccl for m,w in zip(owners,workspaces) if w is not None)
                result[name]['scratch_workspace']={'owners':len(owners),'unique_pools':len(unique),'slots':2*len(unique),
                    'bytes_per_device':sum(w.scratch_bytes_per_device for w in unique.values()),
                    'batch_capacity':32,'mesh_lifetime_owned':True,'paired_roles':['out','down'],
                    'prefill_rs_dram_bytes_per_device':sum(55705600 for w in unique.values() if w.prefill_rs_buffers),
                    'prefill_rs_roles':sorted({role for w in unique.values() for role in w.prefill_rs_buffers}),
                    'native_rs_batches':sorted({b for w in unique.values() for b in w.native_rs_buffers}),
                    'native_rs_dram_bytes_per_device':sum(655360*len(w.native_rs_buffers) for w in unique.values()),
                    'native_rs_l1_reserved_bytes_per_bank':sum(4096*len(w.native_rs_buffers) for w in unique.values())}
            print(name,'DECODE_DONE',times[-1],flush=True)
        finally:
            if tid2 is not None: ttnn.release_trace(mesh,tid2)
            if tid is not None: ttnn.release_trace(mesh,tid)
            ttnn.close_mesh_device(mesh)
    if len(outputs)==2:
        result['prefill_pcc']=pcc(outputs['baseline'][0],outputs['multichip'][0])
        result['decode_pcc']=[pcc(a,b) for a,b in zip(outputs['baseline'][1],outputs['multichip'][1])]
        if len(state_outputs)==2:
            result['state_pcc']={field:pcc(value,state_outputs['multichip'][field])
                                 for field,value in state_outputs['baseline'].items()}
            result['state_shapes']={field:list(value.shape) for field,value in state_outputs['multichip'].items()}
        print(result,flush=True)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    if len(outputs)==2:
        scores=[result['prefill_pcc'],*result['decode_pcc'],*result.get('state_pcc',{}).values()]
        assert all(math.isfinite(v) and v>=.995 for v in scores)

if __name__=='__main__': main()
