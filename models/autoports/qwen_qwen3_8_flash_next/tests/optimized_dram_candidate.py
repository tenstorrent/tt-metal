"""Explicit DRAM-bank sharded projection candidate, including layout costs."""
import math
import ttnn


def configure(obj, names, *, cores, blocks, readers, activations=None):
    mesh=obj.device
    banks=mesh.dram_grid_size().x
    dg=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0,0),ttnn.CoreCoord(banks-1,0))])
    obj.dram_projections={}
    for index,(name,count,block) in enumerate(zip(names,cores,blocks)):
        role_readers = readers[index] if isinstance(readers, (tuple,list)) else readers
        w=getattr(obj,name); k,n=tuple(w.shape)[-2:]
        assert k//32%count==0 and k//32//count%block==0,(name,k,count,block)
        shard_n=math.ceil(n/(32*banks*role_readers))*32*role_readers
        mem=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED,ttnn.BufferType.DRAM,
            ttnn.ShardSpec(dg,[k,shard_n],ttnn.ShardOrientation.ROW_MAJOR))
        # Setup-only repacking; the kernel keeps logical N and zero padding internal.
        wdram=ttnn.from_torch(ttnn.to_torch(w),dtype=w.dtype,layout=ttnn.TILE_LAYOUT,device=mesh,memory_config=mem)
        coords=[ttnn.CoreCoord(i%min(8,count),i//min(8,count)) for i in range(count)]
        ag=ttnn.CoreRangeSet([ttnn.CoreRange(c,c) for c in coords])
        imem=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED,ttnn.BufferType.L1,
            ttnn.ShardSpec(ag,[32,k//count],ttnn.ShardOrientation.ROW_MAJOR))
        program=ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(in0_block_w=block,
            per_core_M=1,per_core_N=shard_n//32//role_readers,num_workers_per_dram_bank=role_readers,
            fused_activation=activations.get(name) if activations else None)
        obj.dram_projections[w.buffer_address()]=(wdram,imem,program)


def linear(obj,x,weight):
    compute=getattr(obj,"projection_computes",{}).get(weight.buffer_address(),obj.projection_compute)
    if x.shape[-2]!=1 or math.prod(tuple(x.shape)[:-2])!=1:
        return ttnn.linear(x,weight,compute_kernel_config=compute,memory_config=ttnn.DRAM_MEMORY_CONFIG)
    w,mem,program=obj.dram_projections[weight.buffer_address()]
    original=tuple(x.shape)
    x=ttnn.reshape(x,(1,1,1,original[-1]))
    x=ttnn.to_memory_config(x,mem)
    y=ttnn.linear(x,w,program_config=program,compute_kernel_config=compute,
        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,dtype=ttnn.bfloat16)
    result_memory=getattr(obj,'dram_result_memory','dram')
    if result_memory!='sharded':
        y=ttnn.to_memory_config(y,ttnn.L1_MEMORY_CONFIG if result_memory=='l1' else ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.reshape(y,tuple(original[:-1])+(weight.shape[-1],))


def install(cores=(5,8),blocks=(16,24),readers=1,result_memory="dram"):
    from models.autoports.qwen_qwen3_8_flash_next.tt import optimized_decoder as mod
    for cls,names in [(mod.OptimizedGatedDeltaNet,('packed_weight','out_weight')),
                      (mod.OptimizedQSA,('packed_projection','wo'))]:
        old=cls.from_state_dict.__func__
        def factory(cls,*args,_old=old,_names=names,**kwargs):
            obj=_old(cls,*args,**kwargs)
            configure(obj,_names,cores=cores,blocks=blocks,readers=readers)
            obj.dram_result_memory=result_memory
            return obj
        cls.from_state_dict=classmethod(factory)
        cls._linear=linear
