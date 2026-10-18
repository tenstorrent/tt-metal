"""QSA packed persistent KV and native sparse SDPA candidate."""
import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt.optimized_decoder import OptimizedQSA
from models.autoports.qwen_qwen3_8_flash_next.tt.fused_decoder import _shared_rope_tables, _shared_apply_rope


class SparseQSA(OptimizedQSA):
    cache_dtype=ttnn.bfloat16
    sparse_chunk=128
    decode_chunk=0
    index_width=2176
    direct_topk=False
    dynamic_topk=False
    sparse_fidelity=ttnn.MathFidelity.HiFi4
    indexer=False
    mask_bounds=False
    indexer_chunk=128

    @classmethod
    def from_state_dict(cls,*args,**kwargs):
        obj=super().from_state_dict(*args,**kwargs)
        obj.pool_tokens=obj.k_cache.shape[0]*obj.page_size
        obj.packed_cache=ttnn.empty((1,1,obj.pool_tokens*2,512),dtype=cls.cache_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,device=obj.device,memory_config=ttnn.DRAM_MEMORY_CONFIG)
        obj.sparse_compute=ttnn.init_device_compute_kernel_config(obj.device.arch(),math_fidelity=cls.sparse_fidelity,
            math_approx_mode=False,fp32_dest_acc_en=True,packer_l1_acc=False)
        if cls.indexer:
            obj.indexer_prefix=ttnn.zeros((obj.batch_size,4,31,128),device=obj.device,dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT)
            obj.indexer_weights=ttnn.ones((obj.batch_size,1,160,4),device=obj.device,dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT)
            obj.indexer_compute=ttnn.init_device_compute_kernel_config(obj.device.arch(),math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,fp32_dest_acc_en=False,packer_l1_acc=False)
        del obj.k_cache,obj.v_cache
        return obj

    def _adapter(self,source,output,positions,page_table,mode,seq,index_width=None):
        width=self.index_width if index_width is None else index_width
        work=self.batch_size*seq*(2 if mode==0 else 1)
        grid=self.device.compute_with_storage_grid_size()
        _,cores,g1,_,n1,n2=ttnn.split_work_to_cores(grid,work,True)
        tensors=[source,output,positions,page_table]
        args=[self.batch_size,seq,self.page_size,self.max_context,self.pool_tokens,
              width,mode,1024 if self.cache_dtype==ttnn.bfloat16 else 512]
        for t in tensors: args.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        rt=[];offset=0
        for i,c in enumerate(ttnn.corerange_to_cores(cores,row_wise=True)):
            n=n1 if i<g1.num_cores() else n2
            rt.append(((c.x,c.y),[*[t.buffer_address() for t in tensors],offset,n]));offset+=n
        scratch=128+self.max_context//self.page_size*4+2048+width*4
        scratch=(scratch+63)//64*64
        cb=ttnn.CBDescriptor(total_size=scratch,core_ranges=cores,format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=0,data_format=ttnn.uint32,page_size=scratch)])
        kernel=ttnn.KernelDescriptor(kernel_source=ADAPTER,source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=cores,compile_time_args=args,runtime_args=rt,config=ttnn.ReaderConfigDescriptor())
        ttnn.generic_op(tensors,ttnn.ProgramDescriptor(kernels=[kernel],cbs=[cb]))

    def _write_packed(self,k,v,positions,page_table):
        packed=ttnn.to_layout(ttnn.concat((v,k),dim=-1),ttnn.ROW_MAJOR_LAYOUT)
        if packed.dtype != self.cache_dtype: packed=ttnn.typecast(packed,self.cache_dtype)
        self._adapter(packed,self.packed_cache,positions,page_table,0,k.shape[2])

    def _sparse_attention(self,q,selected,positions,page_table):
        b,_,s,_=tuple(q.shape)
        k_chunk=self.decode_chunk if s==1 and self.decode_chunk else self.sparse_chunk
        width=((2051+k_chunk-1)//k_chunk)*k_chunk
        indices=ttnn.empty((1,1,b*2*s,width),dtype=ttnn.uint32,layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,memory_config=ttnn.DRAM_MEMORY_CONFIG)
        selected=ttnn.to_layout(ttnn.typecast(selected,ttnn.uint32),ttnn.ROW_MAJOR_LAYOUT)
        self._adapter(selected,indices,positions,page_table,1,s,index_width=width)
        groups=[]
        for batch in range(b):
            for g in range(2):
                chunk=q[batch:batch+1,g*12:(g+1)*12,:,:]
                chunk=ttnn.pad(chunk,[(0,0),(0,20),(0,0),(0,0)],0)
                groups.append(ttnn.concat((ttnn.zeros_like(chunk),chunk),dim=-1))
        folded=ttnn.to_layout(ttnn.concat(groups,dim=2),ttnn.ROW_MAJOR_LAYOUT)
        fmt=ttnn.transformer.SparseKVFormat.BF16 if self.cache_dtype==ttnn.bfloat16 else ttnn.transformer.SparseKVFormat.FP8_E4M3
        out=ttnn.transformer.sparse_sdpa(folded,self.packed_cache,indices,256,kv_format=fmt,
            scale=0.0625,k_chunk_size=k_chunk,compute_kernel_config=self.sparse_compute)
        rows=[]
        for batch in range(b):
            heads=[out[:,:12,(batch*2+g)*s:(batch*2+g+1)*s,:] for g in range(2)]
            rows.append(ttnn.concat(heads,dim=1))
        out=rows[0] if b==1 else ttnn.concat(rows,dim=0)
        out=ttnn.permute(out,(0,2,1,3))
        out=ttnn.reshape(out,(b,1,s,6144))
        return ttnn.to_layout(out,ttnn.TILE_LAYOUT)


    def _chunk_forward(self, hidden_states, *, current_pos, page_table):
        """Vectorized projections/selection with bounded selected-KV attention."""
        batch, seq = self.batch_size, hidden_states.shape[-2]
        x = self._reshape_logical(hidden_states, (batch, 1, seq, 2560))
        positions_packed = self._packed_position(current_pos)
        start = ttnn.permute(self._position_float(self._canonical_position(current_pos)), (1, 0, 2, 3))
        positions = ttnn.add(start, self.chunk_offsets[:, :, :seq, :])

        def heads(value, count, width):
            value = self._reshape_logical(value, (batch, seq, count, width))
            return ttnn.permute(value, (0, 2, 1, 3))

        projections = self._linear(x, self.packed_projection)
        qkv = ttnn.reshape(projections[..., :7168], (batch, seq, 7168))
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            qkv, num_heads=24, num_kv_heads=2, transpose_key=False
        )
        # These three consumers use exactly the same absolute positions.
        rope_cos, rope_sin = _shared_rope_tables(self, positions, batch, seq)
        q = _shared_apply_rope(self, self._norm(q, self.nq), rope_cos, rope_sin)
        k = _shared_apply_rope(self, self._norm(k, self.nk), rope_cos, rope_sin)
        gate = projections[..., 7168:13312]
        index = projections[..., 13312:]
        iq = _shared_apply_rope(self, self._norm(heads(index[..., :512], 4, 128), self.niq), rope_cos, rope_sin)
        raw = index[..., 512:640]
        self._write_packed(k,v,positions_packed,page_table)
        self._transfer(raw,self.raw_key_cache,positions_packed,page_table,mode=0,sequence=seq,heads=1)
        groups = self._transfer(
            self.raw_key_cache,
            self._allocate((batch * seq, 1, 4, 128)),
            positions_packed,
            page_table,
            mode=1,
            sequence=seq,
            heads=1,
        )
        pooled = ttnn.typecast(ttnn.mean(ttnn.typecast(groups, ttnn.float32), dim=2, keepdim=True), ttnn.bfloat16)
        pooled = self._reshape_logical(pooled, (batch, 1, seq, 128))
        block_start = ttnn.multiply(ttnn.floor(ttnn.multiply(positions, 0.25)), 4.0)
        pooled = self._chunk_rope(self._norm(pooled, self.nik), block_start)
        self._transfer(pooled, self.pooled_keys, positions_packed, page_table, mode=2, sequence=seq, heads=1)
        # TTNN batched matmul cannot broadcast only the head axis when B>1.
        # Fold the four query heads into M; keep each request's K bank separate.
        if self.indexer:
            qq=ttnn.to_layout(ttnn.concat((self.indexer_prefix,iq),dim=2),ttnn.TILE_LAYOUT)
            padded=((seq+31+31)//32)*32
            qq=ttnn.pad(qq,[(0,0),(0,0),(0,padded-seq-31),(0,0)],0)
            score_rows=[]
            for lane in range(batch):
                score_rows.append(ttnn.experimental.indexer_score_dsa(qq[lane:lane+1],self.pooled_keys,self.indexer_weights[lane:lane+1,:,:padded,:],
                    chunk_start_idx=self.blocks-32,key_compression_ratio=1,cache_batch_idx=lane,
                    program_config=ttnn.IndexerScoreProgramConfig(q_chunk_size=32,k_chunk_size=self.indexer_chunk,head_group_size=4),
                    compute_kernel_config=self.indexer_compute))
            scores=score_rows[0] if batch==1 else ttnn.concat(score_rows,dim=0)
            scores=scores[:,:,31:31+seq,:]
            if not self.mask_bounds: scores=ttnn.to_layout(scores,ttnn.TILE_LAYOUT)
        else:
            flat_iq = self._reshape_logical(iq, (batch, 1, 4 * seq, 128))
            scores = ttnn.matmul(
                flat_iq, self.pooled_keys, transpose_b=True, activation="relu", compute_kernel_config=self.compute
            )
            scores = self._reshape_logical(scores, (batch, 4, seq, self.blocks))
            scores = ttnn.sum(scores, dim=1, keepdim=True)
        if self.mask_bounds:
            scores=ttnn.to_layout(ttnn.typecast(scores,ttnn.bfloat16),ttnn.ROW_MAJOR_LAYOUT)
            bound=self._mask_scores(scores,positions_packed,seq)
            selected=ttnn.experimental.topk_large_indices(scores,k=512,valid_length_tensor=bound)
        else:
            next_position = ttnn.add(positions, 1.0)
            complete = ttnn.floor(ttnn.multiply(next_position, 0.25))
            scores = ttnn.where(ttnn.lt(self.block_ids, complete, dtype=ttnn.bfloat16), scores, float("-inf"))
            if self.direct_topk:
                bounds={}
                if self.dynamic_topk and batch==1 and seq==1:
                    length=ttnn.reshape(ttnn.to_layout(ttnn.typecast(ttnn.maximum(complete,512),ttnn.uint32),ttnn.ROW_MAJOR_LAYOUT),(1,))
                    bounds['valid_length_tensor']=length
                selected=ttnn.experimental.topk_large_indices(ttnn.to_layout(ttnn.typecast(scores,ttnn.bfloat16),ttnn.ROW_MAJOR_LAYOUT),k=512,**bounds)
            else:
                _, selected = ttnn.topk(ttnn.typecast(scores, ttnn.bfloat16), k=512, dim=-1)
        out = self._sparse_attention(q,selected,positions_packed,page_table)
        out = self._linear(ttnn.multiply(out, gate, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]), self.wo)
        return self._reshape_logical(out, (1, batch, seq, 2560))

    def _mask_scores(self,scores,positions,seq):
        bound=ttnn.empty((1,),dtype=ttnn.uint32,layout=ttnn.ROW_MAJOR_LAYOUT,device=self.device)
        grid=self.device.compute_with_storage_grid_size();workers=grid.x*grid.y
        coords=[ttnn.CoreCoord(i%grid.x,i//grid.x) for i in range(workers)]
        cores=ttnn.CoreRangeSet([ttnn.CoreRange(c,c) for c in coords])
        tensors=[scores,positions,bound];args=[self.batch_size,seq,self.blocks,workers]
        for t in tensors:args.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        k=ttnn.KernelDescriptor(kernel_source=MASK,source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=cores,compile_time_args=args,
            runtime_args=[((c.x,c.y),[*[t.buffer_address() for t in tensors],i]) for i,c in enumerate(coords)],
            config=ttnn.ReaderConfigDescriptor())
        cb=ttnn.CBDescriptor(total_size=768,core_ranges=cores,format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=0,data_format=ttnn.uint32,page_size=768)])
        ttnn.generic_op(tensors,ttnn.ProgramDescriptor(kernels=[k],cbs=[cb]))
        return bound

ADAPTER=r'''
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr uint32_t B=get_compile_time_arg_val(0), S=get_compile_time_arg_val(1);
    constexpr uint32_t P=get_compile_time_arg_val(2), C=get_compile_time_arg_val(3);
    constexpr uint32_t T=get_compile_time_arg_val(4), L=get_compile_time_arg_val(5);
    constexpr uint32_t mode=get_compile_time_arg_val(6), rowbytes=get_compile_time_arg_val(7);
    constexpr auto aa=TensorAccessorArgs<8>();
    constexpr auto oa=TensorAccessorArgs<aa.next_compile_time_args_offset()>();
    constexpr auto pa=TensorAccessorArgs<oa.next_compile_time_args_offset()>();
    constexpr auto ta=TensorAccessorArgs<pa.next_compile_time_args_offset()>();
    const auto src=TensorAccessor(aa,get_arg_val<uint32_t>(0));
    const auto dst=TensorAccessor(oa,get_arg_val<uint32_t>(1));
    const auto pos=TensorAccessor(pa,get_arg_val<uint32_t>(2));
    const auto table=TensorAccessor(ta,get_arg_val<uint32_t>(3));
    const uint32_t begin=get_arg_val<uint32_t>(4), count=get_arg_val<uint32_t>(5);
    CircularBuffer cb(0);cb.reserve_back(1);const uint32_t base=cb.get_write_ptr();
    constexpr uint32_t tablebytes=C/P*4;
    const uint32_t paddr=base, taddr=base+128, iaddr=taddr+tablebytes, oaddr=iaddr+2048;
    auto pp=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(paddr);
    auto pages=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(taddr);
    auto ids=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(iaddr);
    auto out=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(oaddr);
    Noc noc;
    for(uint32_t job=begin;job<begin+count;++job) {
        const uint32_t b=job/(S*(mode==0?2:1));
        const uint32_t s=job%S;
        ASSERT(b<B);
        noc.async_read(pos,CoreLocalMem<uint32_t>(paddr),128,{.page_id=b},{});
        noc.async_read(table,CoreLocalMem<uint32_t>(taddr),tablebytes,{.page_id=b},{});
        noc.async_read_barrier();
        const uint32_t absolute=pp[0]+s;ASSERT(absolute<C);
        if constexpr(mode==0) {
            const uint32_t g=(job/S)%2;
            const uint32_t physical=pages[absolute/P]*P+absolute%P;ASSERT(physical<T);
            noc.async_read(src,CoreLocalMem<uint32_t>(iaddr),rowbytes,{.page_id=job},{});
            noc.async_read_barrier();
            noc.async_write(CoreLocalMem<uint32_t>(iaddr),dst,rowbytes,{}, {.page_id=2*physical+g});
            noc.async_write_barrier();
        } else {
            noc.async_read(src,CoreLocalMem<uint32_t>(iaddr),2048,{.page_id=b*S+s},{});
            noc.async_read_barrier();
            const uint32_t complete=(absolute+1)/4;
            uint32_t cursor=0;
            for(uint32_t e=0;e<512;++e) {
                const uint32_t block=ids[e];
                if(block>=complete) continue;
                for(uint32_t j=0;j<4;++j) {
                    const uint32_t logical=block*4+j;
                    const uint32_t physical=pages[logical/P]*P+logical%P;ASSERT(physical<T);
                    out[cursor++]=physical*2;
                }
            }
            for(uint32_t logical=complete*4;logical<=absolute;++logical) {
                const uint32_t physical=pages[logical/P]*P+logical%P;ASSERT(physical<T);
                out[cursor++]=physical*2;
            }
            ASSERT(cursor>0 && cursor<=2051);
            for(uint32_t j=cursor;j<L;++j) out[j]=0xffffffff;
            noc.async_write(CoreLocalMem<uint32_t>(oaddr),dst,L*4,{}, {.page_id=(b*2)*S+s});
            noc.async_write_barrier();
            for(uint32_t j=0;j<cursor;++j) out[j]+=1;
            noc.async_write(CoreLocalMem<uint32_t>(oaddr),dst,L*4,{}, {.page_id=(b*2+1)*S+s});
            noc.async_write_barrier();
        }
    }
}
'''

MASK=r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr uint32_t B=get_compile_time_arg_val(0),S=get_compile_time_arg_val(1),K=get_compile_time_arg_val(2),W=get_compile_time_arg_val(3);
    constexpr auto sa=TensorAccessorArgs<4>();
    constexpr auto pa=TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    constexpr auto ba=TensorAccessorArgs<pa.next_compile_time_args_offset()>();
    const auto scores=TensorAccessor(sa,get_arg_val<uint32_t>(0));
    const auto positions=TensorAccessor(pa,get_arg_val<uint32_t>(1));
    const auto bound=TensorAccessor(ba,get_arg_val<uint32_t>(2));
    const uint32_t worker=get_arg_val<uint32_t>(3);
    CircularBuffer cb(0);cb.reserve_back(1);const uint32_t base=cb.get_write_ptr();Noc noc;
    auto pos=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    auto values=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base+128);
    if(worker==0) {
        uint32_t last=512;
        for(uint32_t b=0;b<B;++b) {
            noc.async_read(positions,CoreLocalMem<uint32_t>(base),128,{.page_id=b},{});noc.async_read_barrier();
            const uint32_t end=(pos[0]+S)/4;if(end>last)last=end;
        }
        ASSERT(last<=K);
        auto result=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base+640);
        for(uint32_t i=0;i<8;++i) { result[i]=0; }
        result[0]=last;
        noc.async_write(CoreLocalMem<uint32_t>(base+640),bound,32,{}, {.page_id=0});noc.async_write_barrier();
    }
    constexpr uint32_t chunks=(K+255)/256;
    uint32_t loaded=0xffffffff,start=0;
    for(uint32_t job=worker;job<B*S*chunks;job+=W) {
        const uint32_t row=job/chunks,b=row/S,col=(job%chunks)*256;
        if(loaded!=b) {noc.async_read(positions,CoreLocalMem<uint32_t>(base),128,{.page_id=b},{});noc.async_read_barrier();start=pos[0];loaded=b;}
        const uint32_t complete=(start+row%S+1)/4;
        const uint32_t count=K-col<256?K-col:256;
        if(col+count<=complete)continue;
        uint32_t first=0;
        if(col<complete) {
            noc.async_read(scores,CoreLocalMem<uint32_t>(base+128),count*2,{.page_id=row,.offset_bytes=col*2},{});
            noc.async_read_barrier();first=complete-col;
        }
        for(uint32_t i=first;i<count;++i)values[i]=0xff80;
        noc.async_write(CoreLocalMem<uint32_t>(base+128),scores,count*2,{}, {.page_id=row,.offset_bytes=col*2});noc.async_write_barrier();
    }
}
"""
