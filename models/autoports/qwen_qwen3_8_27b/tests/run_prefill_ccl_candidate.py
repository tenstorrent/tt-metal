# SPDX-License-Identifier: Apache-2.0
"""Coherent prefill CCL placement/fusion adapter; default decoder stays unchanged."""
import hashlib
import json
import math
from pathlib import Path
import sys
import ttnn
from . import run_multichip_decoder as harness

_IMPORTED_SOURCE_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

class PrefillCCLCandidate(harness.MultichipDecoder):
    @classmethod
    def from_state_dict(cls,*args,**kwargs):
        model=super().from_state_dict(*args,**kwargs)
        owner=model.ar_workspace
        if not hasattr(owner,'prefill_candidate_buffers'):owner.prefill_candidate_buffers={}
        weights={}
        mode=model.policy.get('pf_fused','')
        for role in ('mix','gateup','out','down'):
            if role in ('mix','gateup') and mode not in ('ag','both'):continue
            if role in ('out','down') and mode not in ('rs','both'):continue
            if role not in model.policy.get('pf_fused_roles',['mix','gateup','out','down']):continue
            w=getattr(model,role)
            if role in ('mix','gateup') or model.policy.get('pf_rs_interleaved_weights'):
                w=ttnn.to_memory_config(w,ttnn.DRAM_MEMORY_CONFIG)
            weights[role]=ttnn.reshape(w,[1,1,*w.shape])
        for view in [model,getattr(model,'_prefill_impl',model),getattr(model,'_batched_decode_impl',model)]:
            view.pf_buffers=owner.prefill_candidate_buffers
            view.pf_weights=weights
        return model

    def _prefill(self):return bool(self.policy.get('_prefill_phase'))
    def _memory(self,l1):return ttnn.L1_MEMORY_CONFIG if l1 else ttnn.DRAM_MEMORY_CONFIG
    def _persistent(self,key,make):
        if key not in self.pf_buffers:self.pf_buffers[key]=make()
        return self.pf_buffers[key]
    def _gather(self,x,key=None,*,compact_output=False):
        if not self._prefill():return super()._gather(x,key,compact_output=compact_output)
        shape=list(x.shape)
        norm=isinstance(key,tuple)
        name=('norm','in' if key[1]==id(self.in_norm) else 'post') if norm else key
        memory=self._memory(self.policy.get('pf_norm_l1') if norm else key in self.policy.get('pf_ag_l1_roles',[]))
        dtype=x.dtype if norm else getattr(ttnn,self.policy.get('pf_ccl_role_dtypes',{}).get(key,self.policy.get('pf_ccl_dtype','bfloat16')))
        x=ttnn.reshape(x,[1]*(4-len(shape))+shape)
        if x.dtype!=dtype:x=ttnn.typecast(x,dtype)
        output=None
        if self.policy.get('pf_persistent'):
            signature=('ag',name,tuple(x.shape),str(dtype),str(memory))
            output=self._persistent(signature,lambda:ttnn.empty([*list(x.shape)[:-1],x.shape[-1]*4],device=self.mesh_device,dtype=dtype,layout=ttnn.TILE_LAYOUT,memory_config=memory))
        y=ttnn.experimental.all_gather_async(x,dim=3,persistent_output_buffer=output,
            multi_device_global_semaphore=self.ccl.get_and_cycle_ag_semaphore_handles(),
            barrier_semaphore=self.ccl.get_and_cycle_barrier_semaphore_handle(),
            num_links=self.policy.get('pf_links',2),topology=ttnn.Topology.Ring,memory_config=memory,
            chunks_per_sync=self.policy.get('pf_chunks',10),num_workers_per_link=self.policy.get('pf_norm_workers',self.policy.get('pf_workers',2)) if norm else self.policy.get('pf_workers',2),
            num_buffers_per_channel=self.policy.get('pf_buffers',2))
        if y.dtype!=ttnn.bfloat16 and not (not norm and (self.policy.get('pf_keep_bf8_activation') or key in self.policy.get('pf_keep_bf8_roles',[]))):
            y=ttnn.typecast(y,ttnn.bfloat16)
        return ttnn.reshape(y,[*shape[:-1],shape[-1]*4])

    def _reduce(self,x,key=None):
        if not self._prefill():return super()._reduce(x,key)
        shape=list(x.shape);memory=self._memory(self.policy.get('pf_rs_l1'))
        x=ttnn.reshape(x,[1]*(4-len(shape))+shape)
        dtype=getattr(ttnn,self.policy.get('pf_ccl_role_dtypes',{}).get(key,self.policy.get('pf_ccl_dtype','bfloat16')))
        if x.dtype!=dtype:x=ttnn.typecast(x,dtype)
        buffers=None
        if self.policy.get('pf_persistent'):
            def make():
                intermediate,penult=ttnn.experimental.reduce_scatter_minimal_async_create_intermediate_buffer(x,dim=3,topology=ttnn.Topology.Ring)
                output=ttnn.empty([*list(x.shape)[:-1],x.shape[-1]//4],device=self.mesh_device,dtype=dtype,layout=ttnn.TILE_LAYOUT,memory_config=memory)
                return [intermediate,output,penult]
            buffers=self._persistent(('rs',key,tuple(x.shape),str(dtype),str(memory)),make)
        y=ttnn.experimental.reduce_scatter_minimal_async(x,dim=3,persistent_output_buffers=buffers,
            multi_device_global_semaphore=self.ccl.get_and_cycle_rs_semaphore_handles(),
            barrier_semaphore=self.ccl.get_and_cycle_barrier_semaphore_handle(),
            num_links=self.policy.get('pf_links',2),topology=ttnn.Topology.Ring,memory_config=memory,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,chunks_per_sync=self.policy.get('pf_chunks',10),
            num_workers_per_link=self.policy.get('pf_workers',2),num_buffers_per_channel=self.policy.get('pf_buffers',2))
        if y.dtype!=ttnn.bfloat16:y=ttnn.typecast(y,ttnn.bfloat16)
        return ttnn.reshape(y,[*shape[:-1],shape[-1]//4])

    def _norm(self,x,w):
        if not (self._prefill() and self.policy.get('pf_norm_l1') and x.shape[-1]==1280 and w.shape[-1]==5120):
            return super()._norm(x,w)
        shape=list(x.shape);memory=ttnn.L1_MEMORY_CONFIG
        x=ttnn.reshape(ttnn.to_memory_config(x,memory),[1,*shape])
        stats=ttnn.rms_norm_pre_all_gather(x,compute_kernel_config=self.ckc,dtype=ttnn.bfloat16,memory_config=memory)
        stats=self._gather(stats,('norm',id(w)))
        y=ttnn.rms_norm_post_all_gather(x,stats,epsilon=self.config.rms_norm_eps,
            weight=self.norm_shards[id(w)],compute_kernel_config=self.ckc,memory_config=memory)
        return ttnn.reshape(y,shape)

    def _linear(self,x,w,activation=None,**kwargs):
        if not self._prefill():return super()._linear(x,w,activation,**kwargs)
        role=self.weight_roles[id(w)];mode=self.policy.get('pf_fused','')
        enabled=role in self.policy.get('pf_fused_roles',['mix','gateup','out','down'])
        ag=enabled and role in ('mix','gateup') and x.shape[-1]==1280 and mode in ('ag','both')
        rs=enabled and role in ('out','down') and mode in ('rs','both')
        if ag or rs:return self._fused_projection(x,w,role,ag)
        old_l1=self.policy.get('prefill_l1',False);sentinel=object();old_h=self.policy.get('prefill_out_block_h',sentinel)
        self.policy['prefill_l1']=role in self.policy.get('pf_ag_l1_roles',[]) or role in self.policy.get('pf_mm_l1_roles',[])
        self.policy['prefill_l1_max_tokens']=self.policy.get('prefill_chunk',4096)*32
        block=self.policy.get('pf_role_block_h',{}).get(role)
        group='mlp_activation' if role in ('gateup','down') else 'attention_activation'
        old_activation=self.policy.get(group,sentinel)
        if role in ('mix','gateup') and (self.policy.get('pf_keep_bf8_activation') or role in self.policy.get('pf_keep_bf8_roles',[])):
            self.policy[group]='bfloat8_b'
        if role in self.policy.get('pf_role_activation',{}):
            self.policy[group]=self.policy['pf_role_activation'][role]
        old_cfg=self.policy[role]
        if block is not None:
            self.policy['prefill_out_block_h']=block
            self.policy[role]={**old_cfg,'prefill_subh':min(old_cfg.get('prefill_subh',self.policy.get('prefill_subh',1)),block)}
        try:return super()._linear(x,w,activation,**kwargs)
        finally:
            self.policy['prefill_l1']=old_l1
            self.policy[role]=old_cfg
            if old_activation is sentinel:self.policy.pop(group,None)
            else:self.policy[group]=old_activation
            if old_h is sentinel:self.policy.pop('prefill_out_block_h',None)
            else:self.policy['prefill_out_block_h']=old_h

    def _fused_projection(self,x,w,role,ag):
        shape=list(x.shape);m=math.prod(shape[:-1]);x=ttnn.reshape(x,[1,1,m,shape[-1]])
        if self.policy.get('pf_fused_activation') and role in self.policy.get('pf_fused_activation_roles',['mix','gateup','out','down']):
            x=ttnn.typecast(x,getattr(ttnn,self.policy['pf_fused_activation']))
        weight=self.pf_weights[role]
        gx,gy=self.policy.get('pf_ag_grid',[8,8]) if ag else (8,8)
        pm=math.ceil(m/32/gy);pn=math.ceil(weight.shape[-1]/32/gx)
        block_h=min(pm,self.policy.get('pf_fused_role_block_h',{}).get(role,self.policy.get('pf_fused_block_h',2)))
        while pm%block_h:block_h-=1
        pc=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(compute_with_storage_grid_size=(gx,gy),
            allowed_worker_cores=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0,0),ttnn.CoreCoord(gx-1,gy-1))]),
            in0_block_w=self.policy.get('pf_fused_role_k',{}).get(role,self.policy.get('pf_fused_k',4)),per_core_M=pm,per_core_N=pn,
            out_block_h=block_h,out_block_w=pn,out_subblock_h=self.policy.get('pf_fused_role_subh',{}).get(role,self.policy.get('pf_fused_subh',1)),
            out_subblock_w=(next(v for v in (2,5,3,1) if pn%v==0) if ag else self.policy.get('pf_fused_subw',5)),
            transpose_mcast=False,fuse_batch=False)
        output_dtype=getattr(ttnn,self.policy.get('pf_ag_output_dtype','bfloat16')) if ag else getattr(ttnn,self.policy.get('pf_rs_output_dtype','bfloat16'))
        common=dict(num_links=2,topology=ttnn.Topology.Ring,memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
            program_config=pc,compute_kernel_config=self.matmul_ckc,dtype=output_dtype,
            barrier_semaphore=self.ccl.get_and_cycle_barrier_semaphore_handle())
        if ag:
            _,y=ttnn.experimental.all_gather_matmul_async(x,weight,persistent_output_buffer=None,dim=3,
                multi_device_global_semaphore=self.ccl.get_and_cycle_ag_semaphore_handles(),
                all_gather_core_grid_offset=(0,gy),memory_config_ag=self._memory(role in self.policy.get('pf_ag_l1_roles',[])),
                chunks_per_sync=self.policy.get('pf_chunks',10),num_workers_per_link=1,num_buffers_per_channel=2,**common)
            logical=self.logical_widths[id(w)];y=y[...,:logical]
            if y.dtype!=ttnn.bfloat16:y=ttnn.typecast(y,ttnn.bfloat16)
        else:
            memory=self._memory(self.policy.get('pf_rs_l1'))
            def make():return [ttnn.empty([1,1,m,width],device=self.mesh_device,dtype=output_dtype,layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG if width==5120 else memory) for width in (5120,1280)]
            intermediate,output=self._persistent(('fused_rs',role,m,str(memory),str(output_dtype)),make)
            _,y=ttnn.experimental.matmul_reduce_scatter_async(x,weight,persistent_intermediate_buffer=intermediate,persistent_output_buffer=output,dim=3,
                multi_device_global_semaphore=self.ccl.get_and_cycle_rs_semaphore_handles(),reduce_scatter_core_grid_offset=(0,8),
                memory_config_rs=memory,intermediate_memory_config_rs=ttnn.DRAM_MEMORY_CONFIG,**common)
            logical=1280
            if y.dtype!=ttnn.bfloat16:y=ttnn.typecast(y,ttnn.bfloat16)
        return ttnn.reshape(y,[*shape[:-1],logical])

    def _finish(self,residual,mixer):
        if not self._prefill():return super()._finish(residual,mixer)
        assert self.policy.get('packed_mlp') and not self.policy.get('separate_prefill')
        x=ttnn.add(residual,mixer,memory_config=mixer.memory_config())
        h=self._norm(x,self.post_norm)
        gu=self._linear(h,self.gateup)
        inter=ttnn.multiply(gu[...,:4352],gu[...,4352:],input_tensor_a_activations=[ttnn.UnaryOpType.SILU],memory_config=ttnn.DRAM_MEMORY_CONFIG)
        del h,gu
        mlp=self._linear(inter,self.down)
        return ttnn.add(x,mlp,memory_config=mlp.memory_config())

    def prefill_forward(self,hidden_states,*,state,start_position=0,page_table=None,cos=None,sin=None):
        impl=getattr(self,'_prefill_impl',self)
        if not impl.policy.get('pf_return_l1'):
            return super().prefill_forward(hidden_states,state=state,start_position=start_position,page_table=page_table,cos=cos,sin=sin)
        batch,length,_=hidden_states.shape
        if length<=0 or start_position+length>impl.config.max_position_embeddings:raise ValueError('Invalid sequence extent')
        if start_position%impl.chunk_size:raise ValueError('Continuation starts at a chunk boundary')
        outputs=[];chunk=impl.policy.get('prefill_chunk',impl.chunk_size)
        for offset in range(0,length,chunk):
            n=min(chunk,length-offset);x=ttnn.to_layout(hidden_states[:,offset:offset+n,:],ttnn.TILE_LAYOUT)
            h=impl._norm(x,impl.in_norm)
            if impl.layer_type=='linear_attention':y=impl._delta(h,state)
            else:
                c,s=cos[:,:,offset:offset+n,:],sin[:,:,offset:offset+n,:];physical=((n+31)//32)*32
                if physical!=n:
                    h=ttnn.pad(h,[(0,0),(0,physical-n),(0,0)],value=0)
                    c=ttnn.pad(c,[(0,0),(0,0),(0,physical-n),(0,0)],value=0)
                    s=ttnn.pad(s,[(0,0),(0,0),(0,physical-n),(0,0)],value=0)
                y=impl._attention(h,state,page_table=page_table,cos=c,sin=s,chunk_start=start_position+offset)[:,:n,:]
            del h
            outputs.append(impl._finish(x,y))
        while len(outputs)>1:outputs=[ttnn.concat(outputs[i:i+16],dim=1) if len(outputs[i:i+16])>1 else outputs[i] for i in range(0,len(outputs),16)]
        return outputs[0]

if __name__=='__main__':
    harness.MultichipDecoder=PrefillCCLCandidate
    output=Path(sys.argv[sys.argv.index('--output')+1])
    try:harness.main()
    finally:
        if output.exists():
            r=json.loads(output.read_text());source=Path(__file__)
            r['provenance']['tests/run_prefill_ccl_candidate.py']=_IMPORTED_SOURCE_SHA256
            r['candidate_adapter']='Prefill-only coherent CCL/fusion; hidden-sharded residual, unchanged TP4 decode.'
            output.write_text(json.dumps(r,indent=2)+'\n')
