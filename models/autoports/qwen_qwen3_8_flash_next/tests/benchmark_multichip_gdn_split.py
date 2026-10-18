"""Split independent GDN state value columns across four cores per head."""

import hashlib
from pathlib import Path
import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt import multichip_decoder as target

original = target._local_recurrence


def split_recurrence(self, q, k, v, g, beta):
    """Exact token recurrence; prefill keeps FP32 state in aliased local buffers."""
    batch, length = q.shape[0], q.shape[1]
    if length != 1:
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
            kernel_source=COMPUTE,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=cores,
            config=config,
            runtime_args=[((c.x, c.y), [job[1] * length]) for c, job in zip(coords, assignments)],
        ),
    ]
    sizes = [4, 4, 4, 1, 1, 1, 1, 4, 4, 1, 1, 1, 4, 1, 1]
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


READER = '\n#include "api/dataflow/dataflow_api.h"\n#include "api/dataflow/noc.h"\n#include "api/dataflow/circular_buffer.h"\n#include "api/core_local_mem.h"\n#include "api/tensor/noc_traits.h"\n#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"\n\nuint32_t off(uint32_t r, uint32_t c) {\n    return (r/16)*512 + (c/16)*256 + (r%16)*16 + c%16;\n}\nvoid kernel_main() {\n    constexpr auto qa = TensorAccessorArgs<0>();\n    constexpr auto ka = TensorAccessorArgs<qa.next_compile_time_args_offset()>();\n    constexpr auto va = TensorAccessorArgs<ka.next_compile_time_args_offset()>();\n    constexpr auto da = TensorAccessorArgs<va.next_compile_time_args_offset()>();\n    constexpr auto ba = TensorAccessorArgs<da.next_compile_time_args_offset()>();\n    constexpr auto sa = TensorAccessorArgs<ba.next_compile_time_args_offset()>();\n    constexpr auto oa = TensorAccessorArgs<sa.next_compile_time_args_offset()>();\n    constexpr auto fa = TensorAccessorArgs<oa.next_compile_time_args_offset()>();\n    const auto q = TensorAccessor(qa, get_arg_val<uint32_t>(0));\n    const auto k = TensorAccessor(ka, get_arg_val<uint32_t>(1));\n    const auto v = TensorAccessor(va, get_arg_val<uint32_t>(2));\n    const auto decay = TensorAccessor(da, get_arg_val<uint32_t>(3));\n    const auto beta = TensorAccessor(ba, get_arg_val<uint32_t>(4));\n    const auto state = TensorAccessor(sa, get_arg_val<uint32_t>(5));\n    const auto output = TensorAccessor(oa, get_arg_val<uint32_t>(6));\n    const auto final = TensorAccessor(fa, get_arg_val<uint32_t>(7));\n    const uint32_t first = get_arg_val<uint32_t>(8), count = get_arg_val<uint32_t>(9);\n    Noc noc;\n    cb_reserve_back(14, 1);\n    const uint32_t scratch = get_write_ptr(14);\n    auto src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);\n    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<6, ckernel::PoolType::SUM,\n        ckernel::ReduceDim::REDUCE_COL>();\n    for (uint32_t job = first; job < first + count; ++job) {\n        const uint32_t head=job/4, column=job%4;\n        const uint32_t b = head/12, h = head%12;\n        cb_reserve_back(0,4);\n        for (uint32_t i=0;i<4;++i)\n            noc.async_read(state, CoreLocalMem<uint32_t>(get_write_ptr(0)+i*4096),4096,\n                {.page_id=head*16+i*4+column},{});\n        noc.async_read_barrier();\n        cb_push_back(0,4);\n        // Raw 32-bit replication: no floating point conversion on dataflow RISCs.\n        for (uint32_t which=1;which<=3;++which) {\n            cb_reserve_back(which,which==3?1:4);\n            auto dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(which));\n            for (uint32_t tile=0;tile<(which==3?1:4);++tile) {\n                uint32_t page = which==3 ? (b+h/32)*4+column : b*4+tile;\n                if (which==1) noc.async_read(q,CoreLocalMem<uint32_t>(scratch),4096,{.page_id=page},{});\n                if (which==2) noc.async_read(k,CoreLocalMem<uint32_t>(scratch),4096,{.page_id=page},{});\n                if (which==3) noc.async_read(v,CoreLocalMem<uint32_t>(scratch),4096,{.page_id=page},{});\n                noc.async_read_barrier();\n                if (which==3) {\n                    for(uint32_t c=0;c<32;++c) dst[tile*1024+off(0,c)]=src[off(h%32,c)];\n                } else {\n                    for(uint32_t r=0;r<32;++r) {\n                        const uint32_t value=src[off(h/3,r)];\n                        for(uint32_t c=0;c<16;++c) {\n                            dst[tile*1024+off(r,c)]=value;\n                            dst[tile*1024+off(r,c+16)]=value;\n                        }\n                    }\n                }\n            }\n            cb_push_back(which,which==3?1:4);\n        }\n        for(uint32_t which=4;which<=5;++which) {\n            cb_reserve_back(which,1);\n            if(which==4) noc.async_read(decay,CoreLocalMem<uint32_t>(scratch),4096,{.page_id=b+h/32},{});\n            else noc.async_read(beta,CoreLocalMem<uint32_t>(scratch),4096,{.page_id=b+h/32},{});\n            noc.async_read_barrier();\n            const uint32_t value=src[off(0,h%32)];\n            auto dst=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(which));\n            for(uint32_t i=0;i<1024;++i) dst[i]=value;\n            cb_push_back(which,1);\n        }\n        cb_wait_front(12,4);\n        for(uint32_t i=0;i<4;++i)\n            noc.async_write(CoreLocalMem<uint32_t>(get_read_ptr(12)+i*4096),final,4096,{},\n                {.page_id=head*16+i*4+column});\n        noc.async_write_barrier();\n        // Final state is still read by compute until output has been published.\n        cb_wait_front(13,1);\n        auto out=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(13));\n        for(uint32_t t=0;t<1;++t) {\n            for(uint32_t r=1;r<32;++r) for(uint32_t c=0;c<32;++c) out[t*1024+off(r,c)]=0;\n            noc.async_write(CoreLocalMem<uint32_t>(get_read_ptr(13)+t*4096),output,4096,{},\n                {.page_id=head*4+column});\n        }\n        noc.async_write_barrier();\n        cb_pop_front(13,1);\n        cb_pop_front(12,4);\n    }\n}\n'
COMPUTE = '\n#include "api/compute/common.h"\n#include "api/compute/compute_kernel_hw_startup.h"\n#include "api/compute/tile_move_copy.h"\n#include "api/compute/eltwise_binary_sfpu.h"\n#include "api/compute/sfpu_binary_bcast.h"\n#include "api/dataflow/circular_buffer.h"\n#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"\n\nvoid load(uint32_t cb,uint32_t tile,uint32_t dst) {\n    reconfig_data_format_srca(cb);\n    copy_init(cb);\n    copy_tile(cb,tile,dst);\n}\nvoid finish(uint32_t cb,uint32_t tile) {\n    tile_regs_commit(); tile_regs_wait(); pack_tile(0,cb,tile); tile_regs_release();\n}\ntemplate<uint32_t Out> void reduce_product() {\n    cb_wait_front(8,4);\n    compute_kernel_lib::reduce<PoolType::SUM,ReduceDim::REDUCE_COL,8,6,Out,\n        compute_kernel_lib::ReduceInputPolicy::NoWaitNoPop,\n        compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,\n        ReduceFp32Mode::Accurate>(compute_kernel_lib::ReduceInputBlockShape::of(4,1));\n    cb_pop_front(8,4);\n}\nvoid product(uint32_t state,uint32_t column) {\n    cb_wait_front(state,4); cb_wait_front(column,4); cb_reserve_back(8,4);\n    pack_reconfig_data_format(8);\n    for(uint32_t i=0;i<4;++i) {\n        tile_regs_acquire(); load(state,i,0); load(column,i,1);\n        mul_binary_tile_init(); mul_binary_tile(0,1,0); finish(8,i);\n    }\n    cb_push_back(8,4);\n}\nvoid kernel_main() {\n    compute_kernel_hw_startup(0,4,7);\n    const uint32_t count=get_arg_val<uint32_t>(0);\n    for(uint32_t job=0;job<count;++job) {\n        cb_wait_front(0,4); cb_wait_front(4,1); cb_reserve_back(7,4);\n        pack_reconfig_data_format(7);\n        for(uint32_t i=0;i<4;++i) {\n            tile_regs_acquire(); load(0,i,0); load(4,0,1);\n            mul_binary_tile_init(); mul_binary_tile(0,1,0); finish(7,i);\n        }\n        cb_push_back(7,4); cb_pop_front(0,4); cb_pop_front(4,1);\n        product(7,2); reduce_product<9>();\n        cb_wait_front(9,1); cb_wait_front(3,1); cb_wait_front(5,1);\n        cb_reserve_back(11,1); pack_reconfig_data_format(11);\n        for(uint32_t i=0;i<1;++i) {\n            tile_regs_acquire(); load(3,i,0); load(9,i,1); load(5,0,2);\n            sub_binary_tile_init(); sub_binary_tile(0,1,0);\n            mul_binary_tile_init(); mul_binary_tile(0,2,0); finish(11,i);\n        }\n        cb_push_back(11,1); cb_pop_front(9,1); cb_pop_front(3,1); cb_pop_front(5,1);\n        cb_wait_front(11,1); cb_reserve_back(12,4); pack_reconfig_data_format(12);\n        for(uint32_t i=0;i<4;++i) {\n            tile_regs_acquire(); load(2,i,0); load(11,0,1); load(7,i,2);\n            sfpu_mul_bcast_row_init(); sfpu_mul_bcast_row(0,1);\n            add_binary_tile_init(); add_binary_tile(0,2,0); finish(12,i);\n        }\n        cb_push_back(12,4); cb_pop_front(11,1); cb_pop_front(7,4); cb_pop_front(2,4);\n        product(12,1); reduce_product<13>(); cb_pop_front(1,4);\n    }\n    cb_pop_front(6,1);\n}\n'


def main():
    from models.autoports.qwen_qwen3_8_flash_next.tests import benchmark_multichip_decoder as bench

    target._local_recurrence = split_recurrence
    bench.CANDIDATE = dict(gdn_value_splits=4, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    bench.main()


if __name__ == "__main__":
    main()
