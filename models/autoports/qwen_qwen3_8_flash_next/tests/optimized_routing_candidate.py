"""Logical top-ten routing metadata fusion and compact BF8 reduction experiment."""
import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt.optimized_decoder import OptimizedMoE,_COMPACT_EXPERT_SOURCE

class RoutedMoE(OptimizedMoE):
    reduce_bf8=False
    padded_topk=False
    def _counts(self,indices,scores):
        ids=ttnn.to_layout(ttnn.typecast(indices,ttnn.uint32),ttnn.ROW_MAJOR_LAYOUT)
        scores=ttnn.to_layout(scores,ttnn.ROW_MAJOR_LAYOUT)
        counts=ttnn.empty((512,),dtype=ttnn.uint32,layout=ttnn.ROW_MAJOR_LAYOUT,device=self.mesh)
        core=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0,0),ttnn.CoreCoord(0,0))])
        tensors=[ids,scores,counts];args=[]
        for t in tensors:args.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        k=ttnn.KernelDescriptor(kernel_source=COUNTS,source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=core,compile_time_args=args,runtime_args=[((0,0),[t.buffer_address() for t in tensors])],
            config=ttnn.ReaderConfigDescriptor())
        cb=ttnn.CBDescriptor(total_size=2304,core_ranges=core,format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=0,data_format=ttnn.uint32,page_size=2304)])
        ttnn.generic_op(tensors,ttnn.ProgramDescriptor(kernels=[k],cbs=[cb]))
        return counts,ids,scores
    def _compact(self, down, indices, selected):
        scores = selected
        compact = ttnn.empty((1,10,32,self.hidden), dtype=down.dtype, layout=ttnn.TILE_LAYOUT,
                             device=self.mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        coords=[ttnn.CoreCoord(i%10,i//10) for i in range(80)]
        cores=ttnn.CoreRangeSet([ttnn.CoreRange(c,c) for c in coords])
        tensors=[down,indices,scores,compact]
        args=[]
        for t in tensors: args.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        kernel=ttnn.KernelDescriptor(kernel_source=_COMPACT_EXPERT_SOURCE,source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=cores,compile_time_args=args,
            runtime_args=[((c.x,c.y),[*[t.buffer_address() for t in tensors],i]) for i,c in enumerate(coords)],
            config=ttnn.ReaderConfigDescriptor())
        cb=ttnn.CBDescriptor(total_size=1280,core_ranges=cores,format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=0,data_format=ttnn.uint32,page_size=1280)])
        ttnn.generic_op(tensors,ttnn.ProgramDescriptor(kernels=[kernel],cbs=[cb]))
        if not self.reduce_bf8: compact=ttnn.typecast(compact,ttnn.bfloat16)
        scores=scores[:,:,:,:10]
        reduced=ttnn.experimental.deepseek_moe_fast_reduce_nc_fused(compact,self.compact_indices,self._reduce_mapping,
            reduce_dim=1,split_size=self.hidden,cluster_axis=0,scores_tensor=scores,
            output_memory_config=ttnn.DRAM_MEMORY_CONFIG,compute_kernel_config=self.compute)[0]
        return ttnn.typecast(reduced,ttnn.bfloat16) if self.reduce_bf8 else reduced

    def _group_unified(self,x):
        router_input = x[:, :, :1, :] if self.logical_router else x
        router_shared=ttnn.linear(router_input,self.router_shared_packed,compute_kernel_config=self.compute)
        logits=router_shared[...,:512]
        values,indices=ttnn.topk(ttnn.pad(logits,[(0,0),(0,0),(0,0),(0,512)],float("-inf")) if self.padded_topk else logits,k=32,dim=-1,sorted=True)
        selected=ttnn.softmax(ttnn.add(values,self.top_mask),dim=-1,numeric_stable=True,compute_kernel_config=self.compute)
        counts,indices,selected=self._counts(indices,selected)
        dispatched=self._dispatch_input(x,counts)
        down=ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(dispatched,self.expert_offsets,counts,
            self.expert_ids,self.expert_gates,self.expert_ups,self.expert_downs,32)
        result=self._compact(down,indices,selected)
        sg,su=router_shared[...,512:1152],router_shared[...,1152:1792]
        scale=router_shared[...,1792:1793]
        shared=ttnn.linear(ttnn.multiply(sg,su,input_tensor_a_activations=[ttnn.UnaryOpType.SILU]),
            self.shared_down,compute_kernel_config=self.compute)
        return ttnn.add(result,ttnn.multiply(shared,scale,input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]))

COUNTS=r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto ia=TensorAccessorArgs<0>();
    constexpr auto sa=TensorAccessorArgs<ia.next_compile_time_args_offset()>();
    constexpr auto ca=TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    const auto ids=TensorAccessor(ia,get_arg_val<uint32_t>(0));
    const auto scores=TensorAccessor(sa,get_arg_val<uint32_t>(1));
    const auto counts=TensorAccessor(ca,get_arg_val<uint32_t>(2));
    CircularBuffer cb(0);cb.reserve_back(1);const uint32_t base=cb.get_write_ptr();Noc noc;
    noc.async_read(ids,CoreLocalMem<uint32_t>(base+2048),128,{.page_id=0},{});
    noc.async_read(scores,CoreLocalMem<uint32_t>(base+2176),64,{.page_id=0},{});
    auto out=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    for(uint32_t i=0;i<512;++i)out[i]=0;
    noc.async_read_barrier();
    auto idx=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base+2048);
    auto sc=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base+2176);
    for(uint32_t i=0;i<10;++i) {ASSERT(idx[i]<512);if(sc[i]!=0)out[idx[i]]=32;}
    noc.async_write(CoreLocalMem<uint32_t>(base),counts,2048,{}, {.page_id=0});
    noc.async_write_barrier();
}
"""
