"""Decode-only active expert gather; all public batch lanes remain independent."""
import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt.optimized_decoder import OptimizedMoE


class CompactMoE(OptimizedMoE):
    logical_router = False
    @classmethod
    def from_state_dict(cls, *args, **kwargs):
        import torch
        obj = super().from_state_dict(*args, **kwargs)
        obj.mesh = kwargs['mesh_device']
        obj.compact_indices = ttnn.from_torch(torch.arange(10,dtype=torch.int32).reshape(1,1,1,10),
            dtype=ttnn.uint16,layout=ttnn.ROW_MAJOR_LAYOUT,device=obj.mesh)
        return obj

    def _compact(self, down, indices, selected):
        indices = ttnn.to_layout(ttnn.typecast(indices[:, :, :1, :], ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
        scores = ttnn.to_layout(selected[:, :, :1, :], ttnn.ROW_MAJOR_LAYOUT)
        compact = ttnn.empty((1,10,32,self.hidden), dtype=down.dtype, layout=ttnn.TILE_LAYOUT,
                             device=self.mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        coords=[ttnn.CoreCoord(i%10,i//10) for i in range(80)]
        cores=ttnn.CoreRangeSet([ttnn.CoreRange(c,c) for c in coords])
        tensors=[down,indices,scores,compact]
        args=[]
        for t in tensors: args.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        kernel=ttnn.KernelDescriptor(kernel_source=SOURCE,source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=cores,compile_time_args=args,
            runtime_args=[((c.x,c.y),[*[t.buffer_address() for t in tensors],i]) for i,c in enumerate(coords)],
            config=ttnn.ReaderConfigDescriptor())
        cb=ttnn.CBDescriptor(total_size=1280,core_ranges=cores,format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=0,data_format=ttnn.uint32,page_size=1280)])
        ttnn.generic_op(tensors,ttnn.ProgramDescriptor(kernels=[kernel],cbs=[cb]))
        compact=ttnn.typecast(compact,ttnn.bfloat16)
        scores=scores[:,:,:,:10]
        return ttnn.experimental.deepseek_moe_fast_reduce_nc_fused(compact,self.compact_indices,self._reduce_mapping,
            reduce_dim=1,split_size=self.hidden,cluster_axis=0,scores_tensor=scores,
            output_memory_config=ttnn.DRAM_MEMORY_CONFIG,compute_kernel_config=self.compute)[0]

    def _group_unified(self,x):
        router_input = x[:, :, :1, :] if self.logical_router else x
        router_shared=ttnn.linear(router_input,self.router_shared_packed,compute_kernel_config=self.compute)
        logits=router_shared[...,:512]
        values,indices=ttnn.topk(logits,k=32,dim=-1,sorted=True)
        selected=ttnn.softmax(ttnn.add(values,self.top_mask),dim=-1,numeric_stable=True,compute_kernel_config=self.compute)
        routing=ttnn.scatter(ttnn.zeros_like(logits),dim=3,index=indices,src=selected)
        active=ttnn.max(routing,dim=2,keepdim=True)>0
        counts=ttnn.typecast(ttnn.where(active,32,0),ttnn.uint32)
        counts=ttnn.reshape(ttnn.to_layout(counts,ttnn.ROW_MAJOR_LAYOUT),(self.experts,))
        dispatched=self._dispatch_input(x,counts)
        down=ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(dispatched,self.expert_offsets,counts,
            self.expert_ids,self.expert_gates,self.expert_ups,self.expert_downs,32)
        result=self._compact(down,indices,selected)
        sg,su=router_shared[...,512:1152],router_shared[...,1152:1792]
        scale=router_shared[...,1792:1793]
        shared=ttnn.linear(ttnn.multiply(sg,su,input_tensor_a_activations=[ttnn.UnaryOpType.SILU]),
            self.shared_down,compute_kernel_config=self.compute)
        return ttnn.add(result,ttnn.multiply(shared,scale,input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]))


SOURCE=r'''
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto da=TensorAccessorArgs<0>();
    constexpr auto ia=TensorAccessorArgs<da.next_compile_time_args_offset()>();
    constexpr auto sa=TensorAccessorArgs<ia.next_compile_time_args_offset()>();
    constexpr auto oa=TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    const auto down=TensorAccessor(da,get_arg_val<uint32_t>(0));
    const auto indices=TensorAccessor(ia,get_arg_val<uint32_t>(1));
    const auto scores=TensorAccessor(sa,get_arg_val<uint32_t>(2));
    const auto out=TensorAccessor(oa,get_arg_val<uint32_t>(3));
    const uint32_t col=get_arg_val<uint32_t>(4);
    CircularBuffer cb(0); cb.reserve_back(1);
    const uint32_t base=cb.get_write_ptr();
    Noc noc;
    noc.async_read(indices,CoreLocalMem<uint32_t>(base+1088),128,{.page_id=0},{});
    noc.async_read(scores,CoreLocalMem<uint32_t>(base+1216),64,{.page_id=0},{});
    noc.async_read_barrier();
    auto ids=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base+1088);
    auto weights=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base+1216);
    for(uint32_t e=0;e<10;++e) {
        ASSERT(ids[e]<512);
        if (weights[e]!=0) {
            noc.async_read(down,CoreLocalMem<uint32_t>(base),1088,{.page_id=ids[e]*80+col},{});
            noc.async_read_barrier();
        } else {
            auto words=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
            for(uint32_t j=0;j<272;++j) words[j]=0;
        }
        noc.async_write(CoreLocalMem<uint32_t>(base),out,1088,{}, {.page_id=e*80+col});
        noc.async_write_barrier();
    }
}
'''
