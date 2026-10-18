# SPDX-License-Identifier: Apache-2.0
"""Direct active-rank regions using the native op's tiled in-place contract."""
import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt.optimized_decoder import OptimizedMoE

class RegionMoE(OptimizedMoE):
    @classmethod
    def from_state_dict(cls,*a,**kw):
        obj=super().from_state_dict(*a,**kw)
        obj.capacity_input=ttnn.zeros((1,1,320,obj.hidden),dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,device=obj.mesh,memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return obj

    def _counts(self,indices,scores):
        ids=ttnn.to_layout(ttnn.typecast(indices,ttnn.uint32),ttnn.ROW_MAJOR_LAYOUT)
        scores=ttnn.to_layout(scores,ttnn.ROW_MAJOR_LAYOUT)
        counts=ttnn.empty((512,),dtype=ttnn.uint32,layout=ttnn.ROW_MAJOR_LAYOUT,device=self.mesh)
        tensors=[ids,scores,counts,self.expert_offsets]
        core=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0,0),ttnn.CoreCoord(0,0))]);args=[]
        for t in tensors:args.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        k=ttnn.KernelDescriptor(kernel_source=COUNTS,source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=core,compile_time_args=args,runtime_args=[((0,0),[t.buffer_address() for t in tensors])],config=ttnn.ReaderConfigDescriptor())
        cb=ttnn.CBDescriptor(total_size=4352,core_ranges=core,format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0,data_format=ttnn.uint32,page_size=4352)])
        ttnn.generic_op(tensors,ttnn.ProgramDescriptor(kernels=[k],cbs=[cb]))
        return counts,ids,scores

    def _dispatch_input(self,x,counts):
        source=ttnn.typecast(x,ttnn.bfloat8_b)
        tensors=[source,self.capacity_input];args=[]
        for t in tensors:args.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        coords=[ttnn.CoreCoord(i%10,i//10) for i in range(80)]
        cores=ttnn.CoreRangeSet([ttnn.CoreRange(c,c) for c in coords])
        k=ttnn.KernelDescriptor(kernel_source=COPY,source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=cores,compile_time_args=args,runtime_args=[((c.x,c.y),[*[t.buffer_address() for t in tensors],i]) for i,c in enumerate(coords)],config=ttnn.ReaderConfigDescriptor())
        cb=ttnn.CBDescriptor(total_size=1088,core_ranges=cores,format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0,data_format=ttnn.uint32,page_size=1088)])
        ttnn.generic_op(tensors,ttnn.ProgramDescriptor(kernels=[k],cbs=[cb]));return self.capacity_input

    def _compact(self,down,indices,selected):
        compact=ttnn.reshape(down,(1,10,32,self.hidden))
        # All ten input regions are refreshed, including zero-score experts;
        # an unexecuted region therefore contains finite input rather than garbage.
        result=ttnn.experimental.deepseek_moe_fast_reduce_nc_fused(compact,self.compact_indices,
            self._reduce_mapping,reduce_dim=1,split_size=self.hidden,cluster_axis=0,
            scores_tensor=selected[:,:,:,:10],output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute)[0]
        return ttnn.typecast(result,ttnn.bfloat16)

COUNTS=r'''
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
 constexpr auto ia=TensorAccessorArgs<0>();constexpr auto sa=TensorAccessorArgs<ia.next_compile_time_args_offset()>();
 constexpr auto ca=TensorAccessorArgs<sa.next_compile_time_args_offset()>();constexpr auto oa=TensorAccessorArgs<ca.next_compile_time_args_offset()>();
 const auto ids=TensorAccessor(ia,get_arg_val<uint32_t>(0));const auto scores=TensorAccessor(sa,get_arg_val<uint32_t>(1));
 const auto counts=TensorAccessor(ca,get_arg_val<uint32_t>(2));const auto offsets=TensorAccessor(oa,get_arg_val<uint32_t>(3));
 CircularBuffer cb(0);cb.reserve_back(1);const uint32_t base=cb.get_write_ptr();Noc noc;
 noc.async_read(ids,CoreLocalMem<uint32_t>(base+4096),128,{.page_id=0},{});
 noc.async_read(scores,CoreLocalMem<uint32_t>(base+4224),64,{.page_id=0},{});
 auto out=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);auto off=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base+2048);
 for(uint32_t i=0;i<512;++i){out[i]=0;off[i]=0;}noc.async_read_barrier();
 auto idx=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base+4096);auto sc=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base+4224);
 for(uint32_t i=0;i<10;++i){ASSERT(idx[i]<512);off[idx[i]]=i*32;if(sc[i]!=0)out[idx[i]]=1;}
 noc.async_write(CoreLocalMem<uint32_t>(base),counts,2048,{}, {.page_id=0});
 noc.async_write(CoreLocalMem<uint32_t>(base+2048),offsets,2048,{}, {.page_id=0});noc.async_write_barrier();
}
'''
COPY=r'''
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
 constexpr auto xa=TensorAccessorArgs<0>();constexpr auto ya=TensorAccessorArgs<xa.next_compile_time_args_offset()>();
 const auto x=TensorAccessor(xa,get_arg_val<uint32_t>(0));const auto y=TensorAccessor(ya,get_arg_val<uint32_t>(1));
 const uint32_t tile=get_arg_val<uint32_t>(2);CircularBuffer cb(0);cb.reserve_back(1);Noc noc;
 noc.async_read(x,CoreLocalMem<uint32_t>(cb.get_write_ptr()),1088,{.page_id=tile},{});noc.async_read_barrier();
 for(uint32_t e=0;e<10;++e){noc.async_write(CoreLocalMem<uint32_t>(cb.get_write_ptr()),y,1088,{}, {.page_id=e*80+tile});}
 noc.async_write_barrier();
}
'''
