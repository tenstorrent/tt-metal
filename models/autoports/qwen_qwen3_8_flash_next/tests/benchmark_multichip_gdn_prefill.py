"""Prefill value-column split, retaining each FP32 recurrence chain locally."""

import hashlib
from pathlib import Path
import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt import multichip_decoder as target

original = target._local_recurrence


def split_prefill(self, q, k, v, g, beta):
    """Exact token recurrence; prefill keeps FP32 state in aliased local buffers."""
    batch, length = q.shape[0], q.shape[1]
    if length == 1:
        return original(self, q, k, v, g, beta)
    # Query normalization already applied the scale as its output activation.
    decay = g  # Already exponentiated by the multiply output activation.
    output = ttnn.empty(
        (batch, 12, length, 128),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=self.device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    final = self.recurrent_state
    tensors = [q, k, v, decay, beta, self.recurrent_state, output, final]
    grid = self.device.compute_with_storage_grid_size()
    jobs = batch * 48
    workers = min(jobs, grid.x * grid.y)
    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(workers)]
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    args = []
    for tensor in tensors:
        args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    assignments = [(i * jobs // workers, (i + 1) * jobs // workers - i * jobs // workers) for i in range(workers)]
    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False, dst_full_sync_en=False
    )
    config.unpack_to_dest_mode = [ttnn.UnpackToDestMode.UnpackToDestFp32] * 64
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=READER,
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
            kernel_source=target._LOCAL_DECODE_COMPUTE,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=cores,
            config=config,
            runtime_args=[((c.x, c.y), [job[1] * length]) for c, job in zip(coords, assignments)],
        ),
    ]
    sizes = [4, 4, 4, 1, 1, 1, 1, 4, 4, 1, 1, 1, 4, 1, 1]
    if length == 1:
        sizes = [4, 4, 4, 1, 1, 1, 1, 4, 4, 1, 1, 1, 4, 1, 1]
    if length > 1:
        sizes.append(1)
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
                total_size=4 * 4096,
                core_ranges=cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.float32, page_size=4096) for i in (0, 12)
                ],
            )
        )
    ttnn.generic_op(tensors, ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs))
    return output, final


READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

uint32_t off(uint32_t r, uint32_t c) {
    return (r/16)*512 + (c/16)*256 + (r%16)*16 + c%16;
}
void kernel_main() {
    constexpr auto qa = TensorAccessorArgs<0>();
    constexpr auto ka = TensorAccessorArgs<qa.next_compile_time_args_offset()>();
    constexpr auto va = TensorAccessorArgs<ka.next_compile_time_args_offset()>();
    constexpr auto da = TensorAccessorArgs<va.next_compile_time_args_offset()>();
    constexpr auto ba = TensorAccessorArgs<da.next_compile_time_args_offset()>();
    constexpr auto sa = TensorAccessorArgs<ba.next_compile_time_args_offset()>();
    constexpr auto oa = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    constexpr auto fa = TensorAccessorArgs<oa.next_compile_time_args_offset()>();
    const auto q = TensorAccessor(qa, get_arg_val<uint32_t>(0));
    const auto k = TensorAccessor(ka, get_arg_val<uint32_t>(1));
    const auto v = TensorAccessor(va, get_arg_val<uint32_t>(2));
    const auto decay = TensorAccessor(da, get_arg_val<uint32_t>(3));
    const auto beta = TensorAccessor(ba, get_arg_val<uint32_t>(4));
    const auto state = TensorAccessor(sa, get_arg_val<uint32_t>(5));
    const auto output = TensorAccessor(oa, get_arg_val<uint32_t>(6));
    const auto final = TensorAccessor(fa, get_arg_val<uint32_t>(7));
    const uint32_t first = get_arg_val<uint32_t>(8), count = get_arg_val<uint32_t>(9);
    const uint32_t length=get_arg_val<uint32_t>(10), st=(length+31)/32;
    cb_reserve_back(15,1);
    auto assembled=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(15));
    Noc noc;
    cb_reserve_back(14, 1);
    const uint32_t scratch = get_write_ptr(14);
    auto src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<6, ckernel::PoolType::SUM,
        ckernel::ReduceDim::REDUCE_COL>();
    for (uint32_t job = first; job < first + count; ++job) {
        const uint32_t head=job/4, column=job%4;
        const uint32_t b = head/12, h = head%12;
        for(uint32_t token=0;token<length;++token) {
        cb_reserve_back(0,4);
        for (uint32_t i=0;i<4;++i) {
            if(token==0) noc.async_read(state, CoreLocalMem<uint32_t>(get_write_ptr(0)+i*4096),4096,
                {.page_id=head*16+i*4+column},{});
            // Later tokens reuse the FP32 bytes through aliased CB0/CB12.

        }
        noc.async_read_barrier();
        cb_push_back(0,4);
        // Raw 32-bit replication: no floating point conversion on dataflow RISCs.
        for (uint32_t which=1;which<=3;++which) {
            cb_reserve_back(which,which==3?1:4);
            auto dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(which));
            for (uint32_t tile=0;tile<(which==3?1:4);++tile) {
                uint32_t page = which==3 ? ((b*length+token)+h/32)*4+column : (b*length+token)*4+tile;
                if (which==1) noc.async_read(q,CoreLocalMem<uint32_t>(scratch),4096,{.page_id=page},{});
                if (which==2) noc.async_read(k,CoreLocalMem<uint32_t>(scratch),4096,{.page_id=page},{});
                if (which==3) noc.async_read(v,CoreLocalMem<uint32_t>(scratch),4096,{.page_id=page},{});
                noc.async_read_barrier();
                if (which==3) {
                    for(uint32_t c=0;c<32;++c) dst[tile*1024+off(0,c)]=src[off(h%32,c)];
                } else {
                    for(uint32_t r=0;r<32;++r) {
                        const uint32_t value=src[off(h/3,r)];
                        for(uint32_t c=0;c<16;++c) {
                            dst[tile*1024+off(r,c)]=value;
                            dst[tile*1024+off(r,c+16)]=value;
                        }
                    }
                }
            }
            cb_push_back(which,which==3?1:4);
        }
        for(uint32_t which=4;which<=5;++which) {
            cb_reserve_back(which,1);
            if(which==4) noc.async_read(decay,CoreLocalMem<uint32_t>(scratch),4096,{.page_id=(b*st+token/32)+h/32},{});
            else noc.async_read(beta,CoreLocalMem<uint32_t>(scratch),4096,{.page_id=(b*st+token/32)+h/32},{});
            noc.async_read_barrier();
            const uint32_t value=src[off(token%32,h%32)];
            auto dst=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(which));
            for(uint32_t i=0;i<1024;++i) dst[i]=value;
            cb_push_back(which,1);
        }
        cb_wait_front(12,4);
        if(token+1==length) for(uint32_t i=0;i<4;++i)
            noc.async_write(CoreLocalMem<uint32_t>(get_read_ptr(12)+i*4096),final,4096,{},
                {.page_id=head*16+i*4+column});
        noc.async_write_barrier();
        // Final state is still read by compute until output has been published.
        cb_wait_front(13,1);
        auto out=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(13));
        if(token%32==0) for(uint32_t i=0;i<1024;++i) assembled[i]=0;
        for(uint32_t t=0;t<1;++t) for(uint32_t c=0;c<32;++c)
            assembled[t*1024+off(token%32,c)]=out[t*1024+off(0,c)];
        if(token%32==31 || token+1==length) {
            for(uint32_t t=0;t<1;++t)
                noc.async_write(CoreLocalMem<uint32_t>(get_write_ptr(15)+t*4096),output,4096,{},
                    {.page_id=(head*st+token/32)*4+column});
            noc.async_write_barrier();
        }
        cb_pop_front(13,1);
        cb_pop_front(12,4);
        }
    }
}
"""


def main():
    from models.autoports.qwen_qwen3_8_flash_next.tests import benchmark_multichip_decoder as bench

    target._local_recurrence = split_prefill
    bench.CANDIDATE = dict(
        gdn_prefill_value_splits=4, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    )
    bench.main()


if __name__ == "__main__":
    main()
