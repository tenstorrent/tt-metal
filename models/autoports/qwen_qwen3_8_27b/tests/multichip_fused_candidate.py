# SPDX-License-Identifier: Apache-2.0
"""Shape-faithful fused CCL/matmul candidate, consumed by the full decoder chain."""
import math
import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.multichip_decoder import MultichipDecoder

class FusedCandidate(MultichipDecoder):
    def prepare_fused(self, batch, mode):
        self.fused_mode=mode
        self.fused_weights={}
        self.fused_buffers={}
        for role in ('mix','gateup','out','down'):
            weight=getattr(self,role)
            native=self.policy.get("fused_dram_sharded_weights")
            if role in ("mix","gateup") and self.policy.get("fused_ag_interleaved_weights"):
                native=False
            value=weight if native else ttnn.to_memory_config(weight,ttnn.DRAM_MEMORY_CONFIG)
            self.fused_weights[id(weight)]=ttnn.reshape(value,[1,1,*value.shape])
            if role in ('out','down'):
                self.fused_buffers[role]=[
                    ttnn.zeros([1,1,batch,width],device=self.mesh_device,dtype=getattr(ttnn,self.policy.get('fused_output_dtype','bfloat16')),
                        layout=ttnn.TILE_LAYOUT,memory_config=ttnn.L1_MEMORY_CONFIG if self.policy.get("fused_l1_interleaved") and not (width==5120 and self.policy.get('fused_intermediate_dram')) and not (width==1280 and self.policy.get('fused_output_dram')) else ttnn.DRAM_MEMORY_CONFIG)
                    for width in (5120,1280)]
        self.fused_batch=batch

    def _delta(self,x,state):
        if self.policy.get('packed_decode_conv') and x.shape[0]>1 and x.shape[1]==1:
            from .multichip_packed_conv_candidate import packed_delta
            return packed_delta(self,x,state)
        return super()._delta(x,state)

    def _linear(self,x,w,activation=None,**kwargs):
        role=self.weight_roles[id(w)]
        is_decode=(x.shape[1]==1 or kwargs.get('decode_batch') is not None)
        ag=is_decode and x.shape[-1]==1280 and role in ('mix','gateup') and self.fused_mode in ('ag','both')
        rs=is_decode and role in ('out','down') and self.fused_mode in ('rs','both')
        if not (ag or rs): return super()._linear(x,w,activation,**kwargs)
        batch=self.fused_batch
        memory=ttnn.L1_MEMORY_CONFIG if self.policy.get("fused_l1_interleaved") else ttnn.DRAM_MEMORY_CONFIG
        if self.policy.get("fused_compact_input"):
            x=ttnn.to_memory_config(ttnn.reshape(x,[1,1,batch,x.shape[-1]]),memory)
        else:
            x=ttnn.reshape(ttnn.to_memory_config(x,memory),[1,1,batch,x.shape[-1]])
        if self.policy.get('fused_activation'):
            x=ttnn.typecast(x,getattr(ttnn,self.policy['fused_activation']))
        weight=self.fused_weights[id(w)]
        nt=math.ceil(weight.shape[-1]/32/8)
        pc=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(compute_with_storage_grid_size=tuple(self.policy.get('fused_rs_grid',[8,6])),
            in0_block_w=self.policy.get('fused_ag_k',10) if ag else (6 if role=='out' else 17)*self.policy.get('fused_rs_k_multiplier',1),out_subblock_h=1,out_subblock_w=self.policy.get('fused_rs_subw',1) if rs else self.policy.get('fused_ag_subw',1),
            per_core_M=math.ceil(batch/32/self.policy.get('fused_rs_grid',[8,6])[1]),per_core_N=nt,transpose_mcast=False,fuse_batch=False,
            **({'out_block_w':self.policy.get('fused_rs_out_block',10)} if rs else {}))
        if ag and self.policy.get('fused_ag_1d'):
            pc=ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(compute_with_storage_grid_size=(8,4),
                in0_block_w=self.policy.get('fused_ag_k',10),out_subblock_h=1,out_subblock_w=1,
                per_core_M=math.ceil(batch/32),per_core_N=math.ceil(weight.shape[-1]/32/32),
                fuse_batch=True,mcast_in0=True)
        intermediate_memory=ttnn.DRAM_MEMORY_CONFIG if self.policy.get('fused_intermediate_dram') else memory
        ckc=ttnn.WormholeComputeKernelConfig(math_fidelity=getattr(ttnn.MathFidelity,self.policy.get('fused_fidelity','LoFi')),
            math_approx_mode=False,fp32_dest_acc_en=False,packer_l1_acc=True)
        common=dict(num_links=2,topology=ttnn.Topology.Ring,memory_config_mm=intermediate_memory if rs else memory,
            program_config=pc,compute_kernel_config=ckc,dtype=getattr(ttnn,self.policy.get('fused_output_dtype','bfloat16')),
            barrier_semaphore=self.ccl.get_and_cycle_barrier_semaphore_handle())
        if ag:
            _,y=ttnn.experimental.all_gather_matmul_async(x,weight,persistent_output_buffer=None,dim=3,
                multi_device_global_semaphore=self.ccl.get_and_cycle_ag_semaphore_handles(),
                all_gather_core_grid_offset=(0,6),memory_config_ag=memory,
                chunks_per_sync=10,num_workers_per_link=1,num_buffers_per_channel=2,**common)
            logical=self.logical_widths[id(w)]
            y=y[...,:logical]
            return ttnn.reshape(y,[1,batch,logical] if kwargs.get('packed_output') else [batch,1,logical])
        intermediate,output=self.fused_buffers[role]
        _,y=ttnn.experimental.matmul_reduce_scatter_async(x,weight,
            persistent_intermediate_buffer=intermediate,persistent_output_buffer=output,dim=3,
            multi_device_global_semaphore=self.ccl.get_and_cycle_rs_semaphore_handles(),
            reduce_scatter_core_grid_offset=(0,6),memory_config_rs=output.memory_config(),
            intermediate_memory_config_rs=intermediate_memory,**common)
        if self.policy.get('fused_residual_l1'):
            compact_mem=ttnn.create_sharded_memory_config([32,160],ttnn.CoreGrid(x=8,y=1),
                ttnn.ShardStrategy.WIDTH,ttnn.ShardOrientation.ROW_MAJOR,use_height_and_width_as_shard_shape=True)
            y=ttnn.to_memory_config(y,compact_mem)
            expanded=ttnn.create_sharded_memory_config([batch*32,32],ttnn.CoreGrid(x=8,y=5),
                ttnn.ShardStrategy.WIDTH,ttnn.ShardOrientation.ROW_MAJOR,use_height_and_width_as_shard_shape=True)
            y=ttnn.reshape(y,[batch,1,1280],memory_config=expanded)
        else:y=ttnn.reshape(y,[batch,1,1280])
        if y.dtype!=ttnn.bfloat16:y=ttnn.typecast(y,ttnn.bfloat16)
        return self._gather(y) if self.policy['residual_layout']=='replicated' else y
