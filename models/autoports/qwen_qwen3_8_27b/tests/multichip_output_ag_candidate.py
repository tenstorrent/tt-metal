# SPDX-License-Identifier: Apache-2.0
"""Row projection alternative: gathered input, local hidden output, sharded residual."""
import math
import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.multichip_decoder import MultichipDecoder


class OutputGatherCandidate(MultichipDecoder):
    @classmethod
    def from_state_dict(cls,state_dict,**kwargs):
        self=super().from_state_dict(state_dict,**kwargs)
        assert self.policy['residual_layout']=='sharded'
        prefix=f'model.language_model.layers.{self.layer_idx}.'
        if not any(k.startswith(prefix) for k in state_dict): prefix=''
        names={'out':('linear_attn.out_proj.weight' if self.layer_type=='linear_attention' else 'self_attn.o_proj.weight'),
               'down':'mlp.down_proj.weight'}
        import torch
        self.output_sharded_weights={}
        for role,name in names.items():
            parts=[v.T.contiguous() for v in state_dict[prefix+name].float().chunk(4,dim=0)]
            self.output_sharded_weights[role]=ttnn.from_torch(torch.cat(parts,dim=0),device=self.mesh_device,
                dtype=ttnn.bfloat4_b,layout=ttnn.TILE_LAYOUT,memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device,dim=0))
        return self

    def _linear(self,x,w,activation=None,**kwargs):
        role=self.weight_roles[id(w)]
        if role not in ('out','down') or not (x.shape[1]==1 or kwargs.get('decode_batch') is not None):
            return super()._linear(x,w,activation,**kwargs)
        batch=kwargs.get('decode_batch') or x.shape[0]
        x=ttnn.reshape(ttnn.to_memory_config(x,ttnn.DRAM_MEMORY_CONFIG),[1,1,batch,x.shape[-1]])
        weight=self.output_sharded_weights[role]
        weight=ttnn.reshape(weight,[1,1,*list(weight.shape)])
        pc=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(compute_with_storage_grid_size=(8,4),
            in0_block_w=6 if role=='out' else 17,out_subblock_h=1,out_subblock_w=1,
            per_core_M=math.ceil(batch/32),per_core_N=5,transpose_mcast=False,fuse_batch=False)
        if self.policy.get('output_ag_1d'):
            pc=ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(compute_with_storage_grid_size=(8,4),
                in0_block_w=6 if role=='out' else 17,out_subblock_h=1,out_subblock_w=1,
                per_core_M=math.ceil(batch/32),per_core_N=2,fuse_batch=True,mcast_in0=True)
        if self.policy.get('fused_payload_bf8'):
            x=ttnn.typecast(x,ttnn.bfloat8_b)
        _,y=ttnn.experimental.all_gather_matmul_async(x,weight,persistent_output_buffer=None,dim=3,
            multi_device_global_semaphore=self.ccl.get_and_cycle_ag_semaphore_handles(),
            barrier_semaphore=self.ccl.get_and_cycle_barrier_semaphore_handle(),
            all_gather_core_grid_offset=(0,6),memory_config_ag=ttnn.DRAM_MEMORY_CONFIG,
            chunks_per_sync=10,num_workers_per_link=1,num_buffers_per_channel=2,
            num_links=2,topology=ttnn.Topology.Ring,memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
            program_config=pc,compute_kernel_config=self.matmul_ckc,dtype=ttnn.bfloat16)
        return ttnn.reshape(y,[batch,1,1280])
