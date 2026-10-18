"""GDN state movement and mathematically equivalent composite norm trials."""
import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt.optimized_decoder import OptimizedGatedDeltaNet
from models.autoports.qwen_qwen3_8_flash_next.tt.fused_decoder import (
    _PRECISE_GDN_READER,_PRECISE_PREFILL_READER,_PRECISE_GDN_COMPUTE,_precise_recurrence,_scaled_query_l2norm)

class MovementGDN(OptimizedGatedDeltaNet):
    roll=False
    inplace=False
    composite_norm=False
    @classmethod
    def from_state_dict(cls,*args,**kwargs):
        import torch
        obj=super().from_state_dict(*args,**kwargs)
        obj.qscale=ttnn.from_torch(torch.full((1,1,1,128),1/128),dtype=ttnn.float32,layout=ttnn.TILE_LAYOUT,device=obj.device)
        obj.kscale=ttnn.from_torch(torch.full((1,1,1,128),128**-0.5),dtype=ttnn.float32,layout=ttnn.TILE_LAYOUT,device=obj.device)
        return obj
    def _recurrence(self,q,k,v,g,beta):
        b,s,_=q.shape
        q=self._heads(q,b,s,16);k=self._heads(k,b,s,16)
        if self.composite_norm:
            q=ttnn.rms_norm(ttnn.typecast(q,ttnn.float32),weight=self.qscale,epsilon=1e-6/128,compute_kernel_config=self.compute)
            k=ttnn.rms_norm(ttnn.typecast(k,ttnn.float32),weight=self.kscale,epsilon=1e-6/128,compute_kernel_config=self.compute)
        else:
            q=_scaled_query_l2norm(q);k=self._l2norm(k)
        v=ttnn.typecast(self._heads(v,b,s,48),ttnn.float32)
        return (_inplace_recurrence if self.inplace else _precise_recurrence)(self,q,k,v,g,beta)
    def _roll_conv(self,projected):
        b,s,_=projected.shape;workers=80
        coords=[ttnn.CoreCoord(i%10,i//10) for i in range(workers)]
        cores=ttnn.CoreRangeSet([ttnn.CoreRange(c,c) for c in coords])
        tensors=[self.conv_state,projected];args=[b,s,workers]
        for t in tensors:args.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        k=ttnn.KernelDescriptor(kernel_source=ROLL,source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=cores,compile_time_args=args,
            runtime_args=[((c.x,c.y),[*[t.buffer_address() for t in tensors],i]) for i,c in enumerate(coords)],
            config=ttnn.ReaderConfigDescriptor())
        cb=ttnn.CBDescriptor(total_size=6144,core_ranges=cores,format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=0,data_format=ttnn.uint32,page_size=6144)])
        ttnn.generic_op(tensors,ttnn.ProgramDescriptor(kernels=[k],cbs=[cb]))
    def _chunk(self, x):
        b, s, _ = x.shape
        padded = (s + 31) // 32 * 32
        projections = self._linear(x, self.packed_weight)
        projected = projections[..., :10240]
        history = ttnn.concat([self.conv_state, projected], dim=1) if not self.roll or not self.native_conv else None
        if self.native_conv:
            proj = ttnn.pad(projected, [(0, 0), (0, padded - s), (0, 0)], 0) if padded != s else projected
            lanes = []
            for lane in range(b):
                incoming = ttnn.to_layout(self.conv_state[lane : lane + 1], ttnn.ROW_MAJOR_LAYOUT)
                lanes.append(
                    ttnn.experimental.kda.qkv_causal_conv1d_silu(
                        ttnn.to_layout(proj[lane : lane + 1], ttnn.ROW_MAJOR_LAYOUT),
                        incoming,
                        *self.conv_taps,
                        2048,
                        2048,
                        6144,
                        actual_start=self.conv_zero,
                        predecessor_carry=incoming,
                        program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=128),
                        compute_kernel_config=self.compute,
                    )
                )
            q, k, v = [ttnn.concat([row[i] for row in lanes], dim=0) if b > 1 else lanes[0][i] for i in range(3)]
            q, k, v = [ttnn.to_layout(t[:, :s, :], ttnn.TILE_LAYOUT) for t in (q, k, v)]
        else:
            conv = None
            for tap in range(4):
                shifted = ttnn.to_layout(history[:, tap : tap + s, :], ttnn.TILE_LAYOUT)
                term = ttnn.multiply(shifted, self.conv_taps[tap])
                conv = term if conv is None else ttnn.add(conv, term)
            conv = ttnn.silu(conv)
            q, k, v = conv[..., :2048], conv[..., 2048:4096], conv[..., 4096:]
        if self.roll: self._roll_conv(projected)
        else: ttnn.copy(ttnn.to_layout(history[:, s : s + 3, :], ttnn.TILE_LAYOUT), self.conv_state)
        a = ttnn.typecast(ttnn.to_layout(projections[..., 16384:16432], ttnn.TILE_LAYOUT), ttnn.float32)
        g = ttnn.multiply(
            self.a_neg,
            ttnn.softplus(ttnn.add(a, self.dt_bias)),
            activations=[ttnn.UnaryOpType.EXP],
        )  # g carries exp(log_decay).
        beta = ttnn.typecast(
            ttnn.sigmoid(ttnn.to_layout(projections[..., 16448:16496], ttnn.TILE_LAYOUT)), ttnn.float32
        )
        o, new_state = self._recurrence(q, k, v, g, beta)
        if not self.inplace: ttnn.copy(new_state, self.recurrent_state)
        o = ttnn.typecast(o, ttnn.bfloat16)
        gate = projections[..., 10240:16384]
        if self.native_norm:
            if padded != s:
                o = ttnn.pad(o, [(0, 0), (0, 0), (0, padded - s), (0, 0)], 0)
                gate = ttnn.pad(gate, [(0, 0), (0, padded - s), (0, 0)], 0)
            o = ttnn.reshape(o, (b * 48, padded, 128))
            gated = ttnn.experimental.kda.sigmoid_gated_rms_norm(
                o,
                gate,
                self.norm_weight_flat,
                48,
                epsilon=self.norm_eps,
                compute_kernel_config=self.compute,
                output_dtype=ttnn.bfloat16,
            )
            gated = ttnn.to_layout(gated[:, :s, :], ttnn.TILE_LAYOUT)
        else:
            o = ttnn.permute(o, (0, 2, 1, 3))
            normed = ttnn.rms_norm(ttnn.typecast(o, ttnn.float32), epsilon=self.norm_eps)
            normed = ttnn.multiply(ttnn.typecast(normed, ttnn.bfloat16), self.norm_weight)
            gate = self._heads(gate, b, s, 48)
            gated = ttnn.multiply(ttnn.typecast(normed, ttnn.float32), ttnn.sigmoid(ttnn.typecast(gate, ttnn.float32)))
            gated = ttnn.to_layout(ttnn.typecast(gated, ttnn.bfloat16), ttnn.ROW_MAJOR_LAYOUT)
            gated = ttnn.to_layout(ttnn.reshape(gated, [b, s, 6144]), ttnn.TILE_LAYOUT)
        return self._linear(gated, self.out_weight)

def _inplace_recurrence(self, q, k, v, g, beta):
    """Exact token recurrence; prefill keeps FP32 state in aliased local buffers."""
    batch, length = q.shape[0], q.shape[1]
    # Query normalization already applied the scale as its output activation.
    decay = g  # Already exponentiated by the multiply output activation.
    output = ttnn.empty(
        (batch, 48, length, 128),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=self.device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    final = self.recurrent_state
    tensors = [q, k, v, decay, beta, self.recurrent_state, output, final]
    grid = self.device.compute_with_storage_grid_size()
    workers = min(batch * 48, grid.x * grid.y)
    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(workers)]
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    args = []
    for tensor in tensors:
        args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    assignments = [
        (i * (batch * 48) // workers, (i + 1) * (batch * 48) // workers - i * (batch * 48) // workers)
        for i in range(workers)
    ]
    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False, dst_full_sync_en=False
    )
    config.unpack_to_dest_mode = [ttnn.UnpackToDestMode.UnpackToDestFp32] * 64
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=_PRECISE_GDN_READER if length == 1 else _PRECISE_PREFILL_READER,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=cores,
            compile_time_args=args,
            config=ttnn.ReaderConfigDescriptor(),
            runtime_args=[
                ((c.x, c.y), [*[t.buffer_address() for t in tensors], *job, length])
                for c, job in zip(coords, assignments)
            ],
        ),
        ttnn.KernelDescriptor(
            kernel_source=_PRECISE_GDN_COMPUTE,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=cores,
            config=config,
            runtime_args=[((c.x, c.y), [job[1] * length]) for c, job in zip(coords, assignments)],
        ),
    ]
    sizes = [16, 4, 4, 4, 1, 1, 1, 16, 16, 4, 1, 4, 16, 4, 1]
    if length > 1:
        sizes.append(4)
    cbs = [
        ttnn.CBDescriptor(
            total_size=count * 4096,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.float32, page_size=4096)],
        )
        for i, count in enumerate(sizes)
        if not (length > 1 and i in (0, 12))
    ]
    if length > 1:
        cbs.append(
            ttnn.CBDescriptor(
                total_size=16 * 4096,
                core_ranges=cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.float32, page_size=4096) for i in (0, 12)
                ],
            )
        )
    ttnn.generic_op(tensors, ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs))
    return output, final


ROLL=r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr uint32_t B=get_compile_time_arg_val(0), S=get_compile_time_arg_val(1), W=get_compile_time_arg_val(2);
    constexpr auto ca=TensorAccessorArgs<3>();
    constexpr auto pa=TensorAccessorArgs<ca.next_compile_time_args_offset()>();
    const auto cache=TensorAccessor(ca,get_arg_val<uint32_t>(0));
    const auto proj=TensorAccessor(pa,get_arg_val<uint32_t>(1));
    CircularBuffer cb(0);cb.reserve_back(1);const uint32_t base=cb.get_write_ptr();Noc noc;
    auto old=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base);
    auto src=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base+2048);
    auto out=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base+4096);
    for(uint32_t page=get_arg_val<uint32_t>(2);page<B*320;page+=W) {
        const uint32_t b=page/320,c=page%320;
        for(uint32_t i=0;i<1024;++i)out[i]=0;
        if(S<3){noc.async_read(cache,CoreLocalMem<uint32_t>(base),2048,{.page_id=page},{});noc.async_read_barrier();}
        uint32_t loaded=0xffffffff;
        for(uint32_t r=0;r<3;++r) {
            const uint32_t history=S+r;
            const bool from_old=history<3;
            const uint32_t row=from_old?history:history-3;
            if(!from_old && loaded!=row/32) {
                noc.async_read(proj,CoreLocalMem<uint32_t>(base+2048),2048,{.page_id=(b*((S+31)/32)+row/32)*320+c},{});
                noc.async_read_barrier();loaded=row/32;
            }
            for(uint32_t side=0;side<2;++side)for(uint32_t j=0;j<16;++j) {
                const uint32_t offset=(row%32/16)*512+side*256+(row%16)*16+j;
                out[side*256+r*16+j]=from_old?old[offset]:src[offset];
            }
        }
        noc.async_write(CoreLocalMem<uint32_t>(base+4096),cache,2048,{}, {.page_id=page});noc.async_write_barrier();
    }
}
"""
