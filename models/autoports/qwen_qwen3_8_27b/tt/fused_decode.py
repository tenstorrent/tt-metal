# SPDX-License-Identifier: Apache-2.0
"""Batched TP4 row projections with native DRAM-sharded weights and fused RS.

The legacy native weight reader requires out_block_w == per_core_N. Smaller
output blocks copy wider bank descriptors into undersized circular buffers.
The mesh-lifetime workspace owns two role buffers per supported logical batch.
Serial layers share them under the complete out/down protocol; all signatures
are allocated before capture and retained until every trace retires.
"""
import ttnn


def prepare_native_rs(model):
    if model.ar_workspace is None:
        raise ValueError('Native RS requires the shared mesh-lifetime workspace')
    if model.policy.get('fused_output_dtype','bfloat16')!='bfloat16':
        raise ValueError('The selected native RS workspace uses BF16 outputs')
    model.ar_workspace.prepare_native_rs(range(model.policy['sharded_decode_min_batch'],33))
    model.native_rs_weights={}
    for role in ('out','down'):
        weight=getattr(model,role)
        if weight.shape[-1]!=5120:
            raise ValueError('Native fused RS requires unpadded5120 output channels')
        model.native_rs_weights[role]=ttnn.reshape(weight,[1,1,*weight.shape])


def native_rs_projection(model,x,role,batch):
    if batch not in model.ar_workspace.native_rs_buffers:
        raise ValueError('Prepare the native RS batch before capture')
    x=ttnn.to_memory_config(ttnn.reshape(x,[1,1,batch,x.shape[-1]]),ttnn.L1_MEMORY_CONFIG)
    activation=model.policy.get('fused_activation')
    if activation:x=ttnn.typecast(x,getattr(ttnn,activation))
    multiplier=model.policy.get('fused_rs_k_multiplier',1)
    geometry=model.policy.get('decode_rs_geometry',{})
    gx,gy=geometry.get('grid',[8,1]);pn=(160+gx-1)//gx
    pc=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(compute_with_storage_grid_size=(gx,gy),
        allowed_worker_cores=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0,0),ttnn.CoreCoord(gx-1,gy-1))]),
        in0_block_w=geometry.get(role+'_k',6 if role=='out' else 17)*multiplier,
        out_subblock_h=geometry.get('subh',1),out_subblock_w=geometry.get('subw',5),
        per_core_M=1,per_core_N=pn,out_block_w=pn,transpose_mcast=False,fuse_batch=False)
    ckc=ttnn.WormholeComputeKernelConfig(math_fidelity=getattr(ttnn.MathFidelity,model.policy.get('fused_fidelity','LoFi')),
        math_approx_mode=False,fp32_dest_acc_en=False,packer_l1_acc=True)
    intermediate,output=model.ar_workspace.borrow_native_rs(role,batch)
    _,y=ttnn.experimental.matmul_reduce_scatter_async(x,model.native_rs_weights[role],
        persistent_intermediate_buffer=intermediate,persistent_output_buffer=output,dim=3,
        multi_device_global_semaphore=model.ccl.get_and_cycle_rs_semaphore_handles(),
        barrier_semaphore=model.ccl.get_and_cycle_barrier_semaphore_handle(),
        num_links=2,topology=ttnn.Topology.Ring,reduce_scatter_core_grid_offset=(0,6),
        memory_config_mm=intermediate.memory_config(),memory_config_rs=output.memory_config(),
        intermediate_memory_config_rs=intermediate.memory_config(),program_config=pc,
        compute_kernel_config=ckc,dtype=output.dtype)
    y=ttnn.reshape(y,[batch,1,1280])
    return ttnn.typecast(y,ttnn.bfloat16) if y.dtype!=ttnn.bfloat16 else y
