"""Sparse prefill plus native decode, with per-token active-expert compaction."""
import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt.optimized_decoder import OptimizedMoE
from models.autoports.qwen_qwen3_8_flash_next.tests.optimized_sparse_candidate import SparseMoE


class HybridMoE(OptimizedMoE):
    prefill_compact = True
    sparse_cores = 20
    sparse_down_cores = 0
    gate_block = 16
    down_block = 20
    sparse_fidelity = ttnn.MathFidelity.LoFi
    sparse_output_dtype = ttnn.bfloat8_b
    sparse_l1 = True
    sparse_packed = True
    exact_nnz = False
    sparse_fuse_silu = False
    prefill_inplace = False

    @classmethod
    def from_state_dict(cls,state,**kwargs):
        import torch
        obj=super().from_state_dict(state,**kwargs)
        # Native decode consumes individual expert matrices. Sparse prefill uses
        # batched matrices. Both formats are immutable setup allocations.
        helper_cls=type("ConfiguredSparsePrefill",(SparseMoE,),{name:getattr(cls,name) for name in
            ('sparse_cores','sparse_down_cores','gate_block','down_block','sparse_fidelity','sparse_output_dtype','sparse_l1','sparse_packed',
             'dedicated_weight_dtype','down_dtype')})
        helper=helper_cls.from_state_dict(state,**kwargs)
        for name in ('gate_up','down','sparse_compute','programs'):
            setattr(obj,name,getattr(helper,name))
        obj.active_nnz=None
        obj.prefill_reduce_indices=ttnn.from_torch(torch.arange(10,dtype=torch.int32).reshape(1,1,1,10).expand(32,1,1,10).contiguous(),
            dtype=ttnn.uint16,layout=ttnn.ROW_MAJOR_LAYOUT,device=obj.mesh)
        return obj

    _sparse=SparseMoE._sparse

    def _compact_prefill(self,down,indices,selected):
        ids=ttnn.to_layout(ttnn.typecast(indices,ttnn.uint32),ttnn.ROW_MAJOR_LAYOUT)
        scores=ttnn.to_layout(selected,ttnn.ROW_MAJOR_LAYOUT)
        compact=ttnn.empty((1,10,32,self.hidden),dtype=ttnn.bfloat8_b,layout=ttnn.TILE_LAYOUT,
            device=self.mesh,memory_config=ttnn.DRAM_MEMORY_CONFIG)
        coords=[ttnn.CoreCoord(i%10,i//10) for i in range(80)]
        cores=ttnn.CoreRangeSet([ttnn.CoreRange(c,c) for c in coords])
        tensors=[down,ids,scores,compact]; args=[]
        for t in tensors: args.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        kernel=ttnn.KernelDescriptor(kernel_source=SOURCE,source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=cores,compile_time_args=args,
            runtime_args=[((c.x,c.y),[*[t.buffer_address() for t in tensors],i]) for i,c in enumerate(coords)],
            config=ttnn.ReaderConfigDescriptor())
        cb=ttnn.CBDescriptor(total_size=8320,core_ranges=cores,format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=0,data_format=ttnn.uint32,page_size=8320)])
        ttnn.generic_op(tensors,ttnn.ProgramDescriptor(kernels=[kernel],cbs=[cb]))
        compact=ttnn.typecast(compact,ttnn.bfloat16)
        scores=ttnn.reshape(scores[:,:,:,:10],(32,1,1,10))
        return ttnn.experimental.deepseek_moe_fast_reduce_nc_fused(compact,self.prefill_reduce_indices,self._reduce_mapping,
            reduce_dim=1,split_size=self.hidden,cluster_axis=0,scores_tensor=scores,
            output_memory_config=ttnn.DRAM_MEMORY_CONFIG,compute_kernel_config=self.compute)[0]

    def _group(self,x):
        from models.autoports.qwen_qwen3_8_flash_next.tt.fused_decoder import _PackedMoE
        if not self.prefill_compact:
            return SparseMoE._group(self,x)
        rs=ttnn.linear(x,self.router_shared_packed,compute_kernel_config=self.compute)
        logits=rs[...,:512]
        values,indices=ttnn.topk(logits,k=32,dim=-1,sorted=True)
        selected=ttnn.softmax(ttnn.add(values,self.top_mask),dim=-1,numeric_stable=True,compute_kernel_config=self.compute)
        routing=ttnn.scatter(ttnn.zeros_like(logits),dim=3,index=indices,src=selected)
        sparsity=ttnn.to_layout(ttnn.max(routing,dim=2,keepdim=True),ttnn.ROW_MAJOR_LAYOUT)
        gu=self._sparse(x,self.gate_up,sparsity,self.intermediate*2)
        gu=ttnn.reshape(gu,(1,512,32,self.intermediate*2))
        hidden=ttnn.multiply(gu[...,:self.intermediate],gu[...,self.intermediate:],
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],memory_config=ttnn.L1_MEMORY_CONFIG)
        down=self._sparse(hidden,self.down,sparsity,self.hidden,input_sparse=True)
        result=self._compact_prefill(down,indices,selected)
        shared=ttnn.linear(ttnn.multiply(rs[...,512:1152],rs[...,1152:1792],input_tensor_a_activations=[ttnn.UnaryOpType.SILU]),
            self.shared_down,compute_kernel_config=self.compute)
        return ttnn.add(result,ttnn.multiply(shared,rs[...,1792:1793],input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]))


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
    const uint32_t base=cb.get_write_ptr(); Noc noc;
    for(uint32_t r=0;r<32;++r) {
        noc.async_read(indices,CoreLocalMem<uint32_t>(base+2176+r*128),128,{.page_id=r},{});
        noc.async_read(scores,CoreLocalMem<uint32_t>(base+6272+r*64),64,{.page_id=r},{});
    }
    noc.async_read_barrier();
    auto ids=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base+2176);
    auto weights=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base+6272);
    auto src=reinterpret_cast<volatile tt_l1_ptr uint8_t*>(base);
    auto dst=reinterpret_cast<volatile tt_l1_ptr uint8_t*>(base+1088);
    for(uint32_t e=0;e<10;++e) {
        auto zero=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base+1088);
        for(uint32_t j=0;j<272;++j) zero[j]=0;
        for(uint32_t r=0;r<32;++r) {
            const uint32_t expert=ids[r*32+e]; ASSERT(expert<512);
            if (weights[r*32+e]==0) continue;
            noc.async_read(down,CoreLocalMem<uint32_t>(base),1088,{.page_id=expert*80+col},{});
            noc.async_read_barrier();
            for(uint32_t side=0;side<2;++side) {
                const uint32_t face=(r/16)*2+side;
                const uint32_t exp=face*16+r%16;
                dst[exp]=src[exp];
                const uint32_t offset=64+face*256+(r%16)*16;
                for(uint32_t c=0;c<16;++c) dst[offset+c]=src[offset+c];
            }
        }
        noc.async_write(CoreLocalMem<uint32_t>(base+1088),out,1088,{}, {.page_id=e*80+col});
        noc.async_write_barrier();
    }
}
'''
