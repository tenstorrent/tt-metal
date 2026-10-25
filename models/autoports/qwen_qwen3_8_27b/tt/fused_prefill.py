# SPDX-License-Identifier: Apache-2.0
"""Native TP4 fused row projections for full 4096-row single-user chunks.

Other logical shapes keep the same hidden-sharded TP4 contract through separate
matmul/RS. The owner preallocates both roles before capture; layers share it.
"""
import ttnn


def prepare_prefill_rs(model):
    if model.ar_workspace is None:
        raise ValueError('Fused prefill RS requires a shared mesh-lifetime owner')
    model.ar_workspace.prepare_prefill_rs()
    model.prefill_rs_weights={}
    for role in ('out','down'):
        w=getattr(model,role)
        if w.shape[-1]!=5120:raise ValueError('Native RS requires unpadded5120 output channels')
        model.prefill_rs_weights[role]=ttnn.reshape(w,[1,1,*w.shape])


def prefill_rs_projection(model,x,role):
    x=ttnn.reshape(x,[1,1,4096,x.shape[-1]])
    geometry=model.policy['prefill_rs_geometry']
    gx,gy=geometry.get('grid',[8,8]);pn=(160+gx-1)//gx
    pc=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(gx,gy),
        allowed_worker_cores=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0,0),ttnn.CoreCoord(gx-1,gy-1))]),
        in0_block_w=geometry[role+'_k'],per_core_M=16,per_core_N=pn,
        out_block_h=geometry['block_h'],out_block_w=pn,
        out_subblock_h=geometry['subh'],out_subblock_w=geometry['subw'],transpose_mcast=False,fuse_batch=False)
    intermediate,output=model.ar_workspace.borrow_prefill_rs(role)
    _,y=ttnn.experimental.matmul_reduce_scatter_async(x,model.prefill_rs_weights[role],
        persistent_intermediate_buffer=intermediate,persistent_output_buffer=output,dim=3,
        multi_device_global_semaphore=model.ccl.get_and_cycle_rs_semaphore_handles(),
        barrier_semaphore=model.ccl.get_and_cycle_barrier_semaphore_handle(),
        num_links=2,topology=ttnn.Topology.Ring,reduce_scatter_core_grid_offset=(0,8),
        memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,memory_config_rs=ttnn.DRAM_MEMORY_CONFIG,
        intermediate_memory_config_rs=ttnn.DRAM_MEMORY_CONFIG,program_config=pc,
        compute_kernel_config=model.matmul_ckc,dtype=ttnn.bfloat8_b)
    return ttnn.reshape(ttnn.typecast(y,ttnn.bfloat16),[1,4096,1280])
