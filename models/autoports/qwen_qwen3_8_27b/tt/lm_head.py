# SPDX-License-Identifier: Apache-2.0
"""Same-precision DRAM-sharded terminal candidate using the common LMHead1D."""
import math
import torch
import ttnn
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.lm_head.lm_head_1d import LMHead1D,LMHead1DConfig


def build_head(weight,mesh,compute_config,*,chunk=8192,readers=1,block=5,cores=8):
    k,vocab=weight.shape
    local=vocab//mesh.get_num_devices()
    dg=mesh.dram_grid_size()
    dram=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0,0),ttnn.CoreCoord(dg.x-1,dg.y-1))])
    input_grid=ttnn.num_cores_to_corerangeset(cores,mesh.compute_with_storage_grid_size(),row_wise=True)
    mem=ttnn.create_sharded_memory_config([32,k//cores],input_grid,ttnn.ShardStrategy.WIDTH,
        ttnn.ShardOrientation.ROW_MAJOR,use_height_and_width_as_shard_shape=True)
    weights=[];configs=[];sizes=[];mems=[]
    for offset in range(0,local,chunk):
        n=min(chunk,local-offset)
        physical=math.ceil(n/(32*dg.x*readers))*32*dg.x*readers
        pieces=[torch.nn.functional.pad(weight[:,i*local+offset:i*local+offset+n],(0,physical-n)) for i in range(mesh.get_num_devices())]
        wm=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED,ttnn.BufferType.DRAM,
            ttnn.ShardSpec(dram,[k,physical//dg.x],ttnn.ShardOrientation.ROW_MAJOR))
        weights.append(LazyWeight(source=torch.cat(pieces,dim=-1).contiguous(),device=mesh,dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,memory_config=wm,mesh_mapper_config=ttnn.MeshMapperConfig(
                placements=[ttnn.PlacementShard(-1)],mesh_shape_override=ttnn.MeshShape([mesh.get_num_devices()]))))
        configs.append(ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(in0_block_w=block,per_core_M=1,
            per_core_N=physical//(32*dg.x*readers),num_workers_per_dram_bank=readers))
        sizes.append(n);mems.append(wm)
    head=LMHead1D.from_config(LMHead1DConfig(output_weights=weights,mesh_device=mesh,dim=k,max_batch_size=32,
        program_configs=configs,compute_kernel_config=compute_config,output_split_sizes=sizes,lm_head_dtype=ttnn.bfloat16,
        output_memcfg=ttnn.DRAM_MEMORY_CONFIG,input_memcfg=mem,weights_memcfgs=mems))
    head.load_device_weights()
    return head
