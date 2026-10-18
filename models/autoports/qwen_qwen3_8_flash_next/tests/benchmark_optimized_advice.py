# SPDX-License-Identifier: Apache-2.0
"""Target remaining profiler advice on the delivered runtime, in a fresh process."""
import argparse
import json
import math
import os
import sys

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests import benchmark_optimized_decoder as benchmark


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--advice',required=True)
    args, remaining=parser.parse_known_args()
    os.environ['QWEN38_ADVICE_VARIANT']=args.advice
    sys.argv=[sys.argv[0]]+remaining
    from models.autoports.qwen_qwen3_8_flash_next.tt import optimized_decoder as runtime
    if args.advice.startswith('compact_l1'):
        compact=runtime.OptimizedMoE._compact
        def l1_compact(self,*a,**kw):
            empty=ttnn.empty
            reduce=ttnn.experimental.deepseek_moe_fast_reduce_nc_fused
            def l1_empty(*a,**kw):
                kw['memory_config']=ttnn.L1_MEMORY_CONFIG
                return empty(*a,**kw)
            def l1_reduce(*a,**kw):
                kw['output_memory_config']=ttnn.L1_MEMORY_CONFIG
                return reduce(*a,**kw)
            if args.advice!='compact_l1_output':ttnn.empty=l1_empty
            if args.advice!='compact_l1_input':ttnn.experimental.deepseek_moe_fast_reduce_nc_fused=l1_reduce
            try:return compact(self,*a,**kw)
            finally:
                ttnn.empty=empty
                ttnn.experimental.deepseek_moe_fast_reduce_nc_fused=reduce
        runtime.OptimizedMoE._compact=l1_compact
    if args.advice=='moe_regions':
        from models.autoports.qwen_qwen3_8_flash_next.tests.optimized_moe_region_candidate import RegionMoE
        runtime.OptimizedMoE=RegionMoE
    if args.advice=='moe_count1'  or args.advice.startswith('moe_row1_'):
        runtime._ROUTING_COUNTS_SOURCE=runtime._ROUTING_COUNTS_SOURCE.replace('out[idx[i]]=32','out[idx[i]]=1')
    if args.advice.startswith('moe_row1_'):
        from models.autoports.qwen_qwen3_8_flash_next.tt import fused_decoder
        workers=int(args.advice.rsplit('_',1)[1])
        factory=runtime.OptimizedMoE.from_state_dict.__func__
        dispatch=runtime.OptimizedMoE._dispatch_input
        def moe_factory(cls,*a,**kw):
            obj=factory(cls,*a,**kw)
            obj._dispatch_coords=[ttnn.CoreCoord(i%8,i//8) for i in range(workers)]
            obj._dispatch_cores=ttnn.CoreRangeSet([ttnn.CoreRange(c,c) for c in obj._dispatch_coords])
            return obj
        def dispatch_one(self,x,counts):
            return dispatch(self,x[...,:1,:],counts)
        runtime.OptimizedMoE.from_state_dict=classmethod(moe_factory)
        runtime.OptimizedMoE._dispatch_input=dispatch_one
        fused_decoder._MOE_DISPATCH_COPY=ROW_DISPATCH.replace('NUM_WORKERS',str(workers))
    if args.advice.startswith('hc_') and 'prefill' not in args.advice and 'split' not in args.advice and 'geom' not in args.advice and 'result' not in args.advice:
        runtime.OptimizedHyperConnection.fidelity = ttnn.MathFidelity.LoFi if args.advice.endswith('lofi') else ttnn.MathFidelity.HiFi2
        if 'bfp8' in args.advice:
            runtime.OptimizedHyperConnection.weight_dtype = ttnn.bfloat8_b
    if args.advice=='hc_result_l1' or args.advice=='hc_split_final':
        runtime.OptimizedHyperConnection.dram_cores=(5,2)
        runtime.OptimizedHyperConnection.dram_blocks=(64,5)
        runtime.OptimizedHyperConnection.dram_readers=1
        if args.advice=='hc_result_l1':runtime.OptimizedHyperConnection.dram_result_memory='l1'
    if args.advice.startswith('hc_geom_'):
        read,up,readers=map(int,args.advice.removeprefix('hc_geom_').split('_'))
        runtime.OptimizedHyperConnection.dram_cores=(read,up)
        runtime.OptimizedHyperConnection.dram_blocks=(320//read,10//up)
        runtime.OptimizedHyperConnection.dram_readers=readers
    if args.advice.startswith('hc_split'):
        factory=runtime.OptimizedHyperConnection.from_state_dict.__func__
        project=runtime.OptimizedHyperConnection._project
        def hc_factory(cls,*a,**kw):
            obj=factory(cls,*a,**kw)
            obj.read_down=obj.read_packed[...,:320]
            obj.read_inject=obj.read_packed[...,320:]
            runtime._configure_dram(obj,('read_packed','up','read_down','read_inject'),
                cores=(5,2,5,5) if args.advice=='hc_split_final' else (20,2,20,20),
                blocks=(64,5,64,64) if args.advice=='hc_split_final' else (16,5,16,16),
                readers=1 if args.advice=='hc_split_final' else 2)
            return obj
        def hc_project(self,x,w,activation=None):
            if w is self.read_packed:
                down=project(self,x,self.read_down)
                if args.advice=='hc_split_mixed':
                    inject=ttnn.linear(ttnn.to_memory_config(x,ttnn.L1_MEMORY_CONFIG),self.read_inject,
                        compute_kernel_config=self.hc_compute,memory_config=ttnn.L1_MEMORY_CONFIG)
                else:inject=project(self,x,self.read_inject)
                return ttnn.concat([down,inject],dim=-1)
            return project(self,x,w,activation)
        runtime.OptimizedHyperConnection.from_state_dict=classmethod(hc_factory)
        runtime.OptimizedHyperConnection._project=hc_project
    if args.advice=='shared_split':
        factory=runtime.OptimizedMoE.from_state_dict.__func__
        shared=runtime.OptimizedMoE._shared_linear
        def shared_factory(cls,*a,**kw):
            obj=factory(cls,*a,**kw)
            obj.shared_parts=[obj.router_shared_packed[...,lo:hi] for lo,hi in ((0,512),(512,1152),(1152,1792),(1792,1824))]
            for width in (512,640,32):
                obj.shared_programs[width]=ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=ttnn.CoreCoord(10,8),in0_block_w=80,
                    out_subblock_h=1,out_subblock_w=1,out_block_h=1,out_block_w=1,
                    per_core_M=1,per_core_N=1,fuse_batch=False,mcast_in0=True)
            return obj
        def split_shared(self,x,w):
            if w is self.router_shared_packed:
                return ttnn.concat([shared(self,x,p) for p in self.shared_parts],dim=-1)
            return shared(self,x,w)
        runtime.OptimizedMoE.from_state_dict=classmethod(shared_factory)
        runtime.OptimizedMoE._shared_linear=split_shared
    if args.advice=='qsa_rms':
        def norm(self,x,weight):
            return ttnn.typecast(ttnn.rms_norm(ttnn.typecast(x,ttnn.float32),weight=weight,
                epsilon=self.eps,compute_kernel_config=self.compute),ttnn.bfloat16)
        runtime.OptimizedQSA._norm=norm
    if args.advice.startswith('qsa_chunk_'):
        import torch
        size=int(args.advice.rsplit('_',1)[1])
        factory=runtime.OptimizedQSA.from_state_dict.__func__
        prefill=runtime.OptimizedQSA.prefill_forward
        def qsa_factory(cls,*a,**kw):
            obj=factory(cls,*a,**kw)
            obj.chunk_size=size
            obj.chunk_offsets=ttnn.from_torch(torch.arange(size).float().reshape(1,1,-1,1),
                dtype=ttnn.float32,layout=ttnn.TILE_LAYOUT,device=obj.device)
            obj.indexer_weights=ttnn.ones((obj.batch_size,1,size+32,4),device=obj.device,
                dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT)
            return obj
        def qsa_prefill(self,*a,**kw):
            kw.setdefault('chunk_size',size)
            return prefill(self,*a,**kw)
        runtime.OptimizedQSA.from_state_dict=classmethod(qsa_factory)
        runtime.OptimizedQSA.prefill_forward=qsa_prefill
    if args.advice=='ple_lookup_parallel':
        from models.autoports.qwen_qwen3_8_flash_next.tests.optimized_ple_lookup_candidate import lookup_rows
        runtime.lookup_rows=lookup_rows
    if args.advice in ('ple_input_l1','ple_both_l1'):
        project=runtime.OptimizedPLE._project
        def ple_l1(self,x):
            if x.shape[-2]!=1 or math.prod(tuple(x.shape)[:-2])!=1:return project(self,x)
            return ttnn.linear(ttnn.to_memory_config(x,ttnn.L1_MEMORY_CONFIG),self.packed_projection,
                compute_kernel_config=self.projection_compute,
                memory_config=ttnn.L1_MEMORY_CONFIG if args.advice=='ple_both_l1' else ttnn.DRAM_MEMORY_CONFIG)
        runtime.OptimizedPLE._project=ple_l1
    if args.advice.startswith('shared_grid_'):
        cores=int(args.advice.removeprefix('shared_grid_'))
        factory=runtime.OptimizedMoE.from_state_dict.__func__
        def shared_grid_factory(cls,*a,**kw):
            obj=factory(cls,*a,**kw)
            for width,k in ((1824,2560),(2560,640)):
                pn=math.ceil(width/32/cores);sub=min(pn,4)
                while pn%sub:sub-=1
                obj.shared_programs[width]=ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=ttnn.CoreCoord(5,cores//5),in0_block_w=k//32,
                    out_subblock_h=1,out_subblock_w=sub,out_block_h=1,out_block_w=pn,
                    per_core_M=1,per_core_N=pn,fuse_batch=False,mcast_in0=True)
            return obj
        runtime.OptimizedMoE.from_state_dict=classmethod(shared_grid_factory)
    if args.advice=='ple_split_l1':
        factory=runtime.OptimizedPLE.from_state_dict.__func__
        project=runtime.OptimizedPLE._project
        def ple_split_factory(cls,*a,**kw):
            obj=factory(cls,*a,**kw)
            obj.key_separate=obj.packed_projection[...,:10240]
            obj.value_separate=obj.packed_projection[...,10240:]
            return obj
        def ple_split(self,x):
            if x.shape[-2]!=1 or math.prod(tuple(x.shape)[:-2])!=1:return project(self,x)
            x=ttnn.to_memory_config(x,ttnn.L1_MEMORY_CONFIG)
            return ttnn.concat([ttnn.linear(x,w,compute_kernel_config=self.projection_compute,
                memory_config=ttnn.L1_MEMORY_CONFIG) for w in (self.key_separate,self.value_separate)],dim=-1)
        runtime.OptimizedPLE.from_state_dict=classmethod(ple_split_factory)
        runtime.OptimizedPLE._project=ple_split
    if args.advice.startswith('shared_fidelity_'):
        fidelity=getattr(ttnn.MathFidelity,args.advice.removeprefix('shared_fidelity_'))
        def shared_fidelity(self,x,weight):
            compute=ttnn.init_device_compute_kernel_config(self.mesh.arch(),math_fidelity=fidelity,
                math_approx_mode=False,fp32_dest_acc_en=True,packer_l1_acc=True)
            return ttnn.linear(ttnn.to_memory_config(x,ttnn.L1_MEMORY_CONFIG),weight,
                program_config=self.shared_programs[weight.shape[-1]],compute_kernel_config=compute,
                memory_config=ttnn.L1_MEMORY_CONFIG)
        runtime.OptimizedMoE._shared_linear=shared_fidelity
    if args.advice.startswith('ple_config_'):
        cores,block=map(int,args.advice.removeprefix('ple_config_').split('_'))
        original_project=runtime.OptimizedPLE._project
        def ple_project(self,x):
            if x.shape[-2]!=1 or math.prod(tuple(x.shape)[:-2])!=1:return original_project(self,x)
            w=self.packed_projection
            pn=math.ceil(w.shape[-1]/32/cores)
            sub=min(pn,4)
            while pn%sub:sub-=1
            gx=11 if cores==110 else 10 if cores>=80 else 5
            config=ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(gx,cores//gx),in0_block_w=block,
                out_subblock_h=1,out_subblock_w=sub,out_block_h=1,out_block_w=pn,
                per_core_M=1,per_core_N=pn,fuse_batch=False,mcast_in0=True)
            return ttnn.linear(ttnn.to_memory_config(x,ttnn.L1_MEMORY_CONFIG),w,
                program_config=config,compute_kernel_config=self.projection_compute,memory_config=ttnn.L1_MEMORY_CONFIG)
        runtime.OptimizedPLE._project=ple_project
    if args.advice=='ple_lofi':
        factory=runtime.OptimizedPLE.from_state_dict.__func__
        def ple_factory(cls,*a,**kw):
            obj=factory(cls,*a,**kw)
            obj.projection_compute=ttnn.init_device_compute_kernel_config(obj.device.arch(),
                math_fidelity=ttnn.MathFidelity.LoFi,math_approx_mode=False,fp32_dest_acc_en=True,packer_l1_acc=True)
            return obj
        runtime.OptimizedPLE.from_state_dict=classmethod(ple_factory)
    linear=ttnn.linear
    reduction=ttnn.sum
    def advised_linear(x,w,*other,**kw):
        k,n=tuple(w.shape)[-2:]
        role='shared' if (k,n) in ((2560,1824),(640,2560)) else 'hc' if (k,n) in ((10240,352),(320,10240)) else 'projection'
        variant=args.advice
        selected=(variant.startswith('shared') and role=='shared') or (variant.startswith('hc_prefill') and role=='hc' and x.shape[-2]>4) or (variant=='prefill_l1' and x.shape[-2]>4)
        if not selected or math.prod(tuple(x.shape)[:-2])!=1 or 'program_config' in kw:
            return linear(x,w,*other,**kw)
        x=ttnn.to_memory_config(x,ttnn.L1_MEMORY_CONFIG)
        kw['memory_config']=ttnn.L1_MEMORY_CONFIG
        if variant.startswith('shared_config'):
            _,_,cores,block=variant.split('_');cores,block=int(cores),int(block)
            per_n=math.ceil(n/32/cores)
            sub=min(per_n,4)
            while per_n%sub:sub-=1
            kw['program_config']=ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(10 if cores>50 else 5,cores//(10 if cores>50 else 5)),
                in0_block_w=min(block,k//32),out_subblock_h=1,out_subblock_w=sub,
                out_block_h=1,out_block_w=per_n,per_core_M=math.ceil(x.shape[-2]/32),per_core_N=per_n,
                fuse_batch=False,mcast_in0=True)
        elif variant in ('hc_prefill_4','hc_prefill_8'):
            block=min(int(variant.rsplit('_',1)[1]),k//32)
            while k//32%block:block-=1
            gx,gy=min(10,math.ceil(n/32)),min(10,math.ceil(x.shape[-2]/32))
            pn=math.ceil(n/32/gx);pm=math.ceil(x.shape[-2]/32/gy)
            sub=min(pn,4)
            while pn%sub:sub-=1
            kw['program_config']=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(gx,gy),in0_block_w=block,
                out_subblock_h=1,out_subblock_w=sub,per_core_M=pm,per_core_N=pn,transpose_mcast=False)
        return linear(x,w,*other,**kw)
    def advised_sum(x,*other,**kw):
        if args.advice=='moe_reduce' and kw.get('dim')==1 and tuple(x.shape)[1]==512:
            output=ttnn.experimental.fast_reduce_nc(x,dims=[1],memory_config=ttnn.L1_MEMORY_CONFIG)
            return output[:,:,:x.shape[2],:]
        return reduction(x,*other,**kw)
    ttnn.linear=advised_linear
    ttnn.sum=advised_sum
    print('ADVICE_VARIANT',json.dumps(vars(args)),flush=True)
    benchmark.main()

ROW_DISPATCH=r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto xa=TensorAccessorArgs<0>();
    constexpr auto ya=TensorAccessorArgs<xa.next_compile_time_args_offset()>();
    constexpr auto ca=TensorAccessorArgs<ya.next_compile_time_args_offset()>();
    const auto x=TensorAccessor(xa,get_arg_val<uint32_t>(0));
    const auto y=TensorAccessor(ya,get_arg_val<uint32_t>(1));
    const auto counts=TensorAccessor(ca,get_arg_val<uint32_t>(2));
    const uint32_t worker=get_arg_val<uint32_t>(3);
    CircularBuffer row_cb(0),count_cb(1);row_cb.reserve_back(1);count_cb.reserve_back(1);Noc noc;
    noc.async_read(x,CoreLocalMem<uint32_t>(row_cb.get_write_ptr()),5120,{.page_id=0},{});
    noc.async_read(counts,CoreLocalMem<uint32_t>(count_cb.get_write_ptr()),2048,{.page_id=0},{});
    noc.async_read_barrier();
    const volatile uint32_t* active=reinterpret_cast<volatile uint32_t*>(count_cb.get_write_ptr());
    for(uint32_t expert=worker;expert<512;expert+=NUM_WORKERS) {
        if(active[expert]==0)continue;
        noc.async_write(CoreLocalMem<uint32_t>(row_cb.get_write_ptr()),y,5120,{}, {.page_id=expert*32});
    }
    noc.async_write_barrier();
}
"""

if __name__=='__main__':main()
