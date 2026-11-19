# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Isolated joint correction/combine + four-tap convolution, no default wiring.

One work item owns all users for one 32-channel column. Raw histories and
post-convolution QKV remain row-major; corrected Z/beta/alpha remain tiled.
Import is CPU-only. Device compilation and exact native gates are mandatory.
"""


def plan(batch, cores=80):
    if type(batch) is not int or not 1 <= batch <= 32 or type(cores) is not int or not 1 <= cores <= 130:
        raise ValueError("Require batch1..32 and workers1..130")
    n = min(130, cores)
    # Stride columns so the first80 QKV columns occupy80 workers, rather
    # than giving half the workers only cheap Z/correction-tail work.
    return [(i, (129 - i) // n + 1, n) for i in range(n)]


def residual_column(column):
    if not 0 <= column < 130:
        raise ValueError("Corrected column outside0..129")
    return 130 + (column if column < 80 else column - 48) if column < 80 or column >= 128 else None


def tile_index(row, channel):
    if not (0 <= row < 32 and 0 <= channel < 32):
        raise ValueError("Tile coordinate outside0..31")
    return (row // 16) * 512 + (channel // 16) * 256 + (row % 16) * 16 + channel % 16


READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
inline uint32_t ti(uint32_t u,uint32_t c) {return (u/16)*512+(c/16)*256+(u%16)*16+c%16;}
void kernel_main() {
    constexpr auto pa=TensorAccessorArgs<0>();
    constexpr auto ha=TensorAccessorArgs<pa.next_compile_time_args_offset()>();
    constexpr auto a0=TensorAccessorArgs<ha.next_compile_time_args_offset()>();
    constexpr auto a1=TensorAccessorArgs<a0.next_compile_time_args_offset()>();
    constexpr auto a2=TensorAccessorArgs<a1.next_compile_time_args_offset()>();
    constexpr auto a3=TensorAccessorArgs<a2.next_compile_time_args_offset()>();
    const auto packed=TensorAccessor(pa,get_arg_val<uint32_t>(0),SOURCE_BYTES);
    const auto hist=TensorAccessor(ha,get_arg_val<uint32_t>(1),5120);
    const auto t0=TensorAccessor(a0,get_arg_val<uint32_t>(2),5120);
    const auto t1=TensorAccessor(a1,get_arg_val<uint32_t>(3),5120);
    const auto t2=TensorAccessor(a2,get_arg_val<uint32_t>(4),5120);
    const auto t3=TensorAccessor(a3,get_arg_val<uint32_t>(5),5120);
    const uint32_t B=get_arg_val<uint32_t>(6),start=get_arg_val<uint32_t>(7),count=get_arg_val<uint32_t>(8),stride=get_arg_val<uint32_t>(9);
    Noc noc;CircularBuffer p(0),r(1),scratch(12);
    scratch.reserve_back(101);const uint32_t tmp=(scratch.get_write_ptr()+63u)&~63u;
    for(uint32_t item=0;item<count;item++) {
        const uint32_t col=start+item*stride;
        p.reserve_back(1);r.reserve_back(1);
        noc.async_read(packed,p,SOURCE_BYTES,{.page_id=col},{.offset_bytes=0});
        if(col<80 || col>=128) {
            const uint32_t rc=130+(col<80?col:col-48);
            noc.async_read(packed,r,SOURCE_BYTES,{.page_id=rc},{.offset_bytes=0});
        } else {noc.async_write_zeros(r,SOURCE_BYTES);noc.write_zeros_l1_barrier();}
        if(col<80) {
            for(uint32_t id=4;id<=11;id++) if(id!=9) {CircularBuffer cb(id);cb.reserve_back(1);}
            for(uint32_t u=0;u<B;u++) for(uint32_t h=0;h<3;h++)
                noc.async_read(hist,CoreLocalMem<uint32_t>(tmp+(u*3+h)*64),64,
                    {.page_id=u*3+h,.offset_bytes=col*64},{});
            noc.async_read(t0,CoreLocalMem<uint32_t>(tmp+6144),64,{.page_id=0,.offset_bytes=col*64},{});
            noc.async_read(t1,CoreLocalMem<uint32_t>(tmp+6208),64,{.page_id=0,.offset_bytes=col*64},{});
            noc.async_read(t2,CoreLocalMem<uint32_t>(tmp+6272),64,{.page_id=0,.offset_bytes=col*64},{});
            noc.async_read(t3,CoreLocalMem<uint32_t>(tmp+6336),64,{.page_id=0,.offset_bytes=col*64},{});
        }
        noc.async_read_barrier();
        // Remove all logical-user padding before any arithmetic, including Z.
        volatile tt_l1_ptr SOURCE_WORD* pp=reinterpret_cast<volatile tt_l1_ptr SOURCE_WORD*>(p.get_write_ptr());
        volatile tt_l1_ptr SOURCE_WORD* rp=reinterpret_cast<volatile tt_l1_ptr SOURCE_WORD*>(r.get_write_ptr());
        for(uint32_t u=B;u<32;u++) for(uint32_t c=0;c<32;c++) {pp[ti(u,c)]=0;rp[ti(u,c)]=0;}
        if(col<80) {
            volatile tt_l1_ptr uint16_t* src=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tmp);
            for(uint32_t j=0;j<7;j++) {
                const uint32_t id=j<5?j+4:j+5;
                CircularBuffer cb(id);
                volatile tt_l1_ptr uint16_t* dst=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb.get_write_ptr());
                for(uint32_t u=0;u<32;u++) for(uint32_t c=0;c<32;c++)
                    dst[ti(u,c)]=u>=B?0:src[j<3?(u*3+j)*32+c:3072+(j-3)*32+c];
                cb.push_back(1);
            }
        }
        p.push_back(1);r.push_back(1);
    }
}
"""

COMPUTE = r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/dataflow/circular_buffer.h"
inline void round_bf16(uint32_t reg) {
    typecast_tile_init<static_cast<uint32_t>(DataFormat::Float32),static_cast<uint32_t>(DataFormat::Float16_b)>();
    typecast_tile<static_cast<uint32_t>(DataFormat::Float32),static_cast<uint32_t>(DataFormat::Float16_b)>(reg);
}
inline void conv_bf16_silu(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE,true,calculate_silu,(false,8),idst,VectorMode::RC));
}
void kernel_main() {
    const uint32_t start=get_arg_val<uint32_t>(0),count=get_arg_val<uint32_t>(1),stride=get_arg_val<uint32_t>(2);
    compute_kernel_hw_startup(0,2);
    CircularBuffer p(0),r(1),combined(2),out(9),done(15);
    for(uint32_t item=0;item<count;item++) {
        const uint32_t col=start+item*stride;
        p.wait_front(1);r.wait_front(1);combined.reserve_back(1);
        // copy_init selects the copy operation but does not change source dtype.
        // Previous QKV item ended with BF16 inputs; restore joint FP32/BF16.
        reconfig_data_format_srca(0);
        tile_regs_acquire();copy_init(0);copy_tile(0,0,0);copy_init(1);copy_tile(1,0,1);
        add_binary_tile_init();add_binary_tile(0,1,0);round_bf16(0);
        tile_regs_commit();tile_regs_wait();pack_tile(0,2);tile_regs_release();combined.push_back(1);
        if(col<80) {
            for(uint32_t id=4;id<=11;id++) if(id!=9) {CircularBuffer cb(id);cb.wait_front(1);}
            combined.wait_front(1);out.reserve_back(1);
            // History, coefficients and packed current token are all BF16.
            reconfig_data_format_srca(4);tile_regs_acquire();
            for(uint32_t tap=0;tap<4;tap++) {
                const uint32_t work=tap?1:0,in=tap==3?2:tap+4,coeff=tap<2?tap+7:tap+8;
                copy_init(in);copy_tile(in,0,work);copy_init(coeff);copy_tile(coeff,0,2);
                mul_binary_tile_init();mul_binary_tile(work,2,work);
                if(tap) {add_binary_tile_init();add_binary_tile(0,1,0);}
            }
            round_bf16(0);silu_tile_init();conv_bf16_silu(0);
            tile_regs_commit();tile_regs_wait();pack_tile(0,9);tile_regs_release();out.push_back(1);
        }
        // Writer has consumed raw current token and both old history rows.
        done.wait_front(1);done.pop_front(1);p.pop_front(1);r.pop_front(1);
        if(col<80) for(uint32_t id=4;id<=11;id++) if(id!=9) {CircularBuffer cb(id);cb.pop_front(1);}
    }
}
"""

WRITER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
inline uint32_t ti(uint32_t u,uint32_t c) {return (u/16)*512+(c/16)*256+(u%16)*16+c%16;}
void kernel_main() {
    constexpr auto ha=TensorAccessorArgs<0>();
    constexpr auto oa=TensorAccessorArgs<ha.next_compile_time_args_offset()>();
    constexpr auto ra=TensorAccessorArgs<oa.next_compile_time_args_offset()>();
    const auto hist=TensorAccessor(ha,get_arg_val<uint32_t>(0),5120);
    const auto output=TensorAccessor(oa,get_arg_val<uint32_t>(1),5120);
    const auto rest=TensorAccessor(ra,get_arg_val<uint32_t>(2),2048);
    const uint32_t B=get_arg_val<uint32_t>(3),start=get_arg_val<uint32_t>(4),count=get_arg_val<uint32_t>(5),stride=get_arg_val<uint32_t>(6);
    Noc noc;CircularBuffer combined(2),out(9),done(15),scratch(13);
    scratch.reserve_back(129);const uint32_t tmp=(scratch.get_write_ptr()+63u)&~63u;
    for(uint32_t item=0;item<count;item++) {
        const uint32_t col=start+item*stride;
        combined.wait_front(1);
        if(col<80) {
            out.wait_front(1);
            for(uint32_t row=0;row<4;row++) {
                const uint32_t id=row==0?9:row==3?2:row+4;
                CircularBuffer cb(id);
                volatile tt_l1_ptr uint16_t* src=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb.get_read_ptr());
                volatile tt_l1_ptr uint16_t* dst=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tmp+row*2048);
                for(uint32_t u=0;u<B;u++) for(uint32_t c=0;c<32;c++) dst[u*32+c]=src[ti(u,c)];
                for(uint32_t u=0;u<B;u++) {
                    if(row==0) noc.async_write(CoreLocalMem<uint32_t>(tmp+u*64),output,64,
                        {},{.page_id=u,.offset_bytes=col*64});
                    else noc.async_write(CoreLocalMem<uint32_t>(tmp+row*2048+u*64),hist,64,
                        {},{.page_id=u*3+row-1,.offset_bytes=col*64});
                }
            }
        }
        if(!REST_ONLY || col>=80) noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(combined),rest,2048,
            {.offset_bytes=0},{.page_id=REST_ONLY?col-80:col});
        noc.async_write_barrier();combined.pop_front(1);if(col<80)out.pop_front(1);
        done.reserve_back(1);done.push_back(1);
    }
}
"""


def sources(dtype="float32", rest_only=False):
    if dtype not in ("float32", "bfloat16") or type(rest_only) is not bool:
        raise ValueError("Require FP32/BF16 joint dtype and boolean rest_only")
    return (
        READER.replace("SOURCE_BYTES", "4096" if dtype == "float32" else "2048").replace(
            "SOURCE_WORD", "uint32_t" if dtype == "float32" else "uint16_t"
        ),
        COMPUTE,
        WRITER.replace("REST_ONLY", "true" if rest_only else "false"),
    )


def joint_conv(packed, history, taps, *, output=None, rest=None, rest_only=False, cores=80):
    """Return (SiLU-convolved RM QKV[B,1,2560], corrected TILE rest), mutate history.

    Default rest is full corrected [1,B,4160], preserving existing downstream
    slices. rest_only returns [1,B,1600]: Z1536, beta32, alpha32, without rawQKV.
    Joint input is [1,B,N>=6784], FP32/BF16 TILE DRAM or WIDTH_SHARDED L1.
    """
    import ttnn

    batch = history.shape[0]
    spans = plan(batch, cores)
    if tuple(history.shape) != (batch, 3, 2560) or len(taps) != 4:
        raise ValueError("Require history[B,3,2560] and four taps")
    if len(packed.shape) != 3 or tuple(packed.shape)[:2] != (1, batch) or packed.shape[-1] < 6784:
        raise ValueError("Require compact joint[1,B,N>=6784]")
    if tuple(packed.padded_shape)[:2] != (1, 32) or packed.dtype not in (ttnn.float32, ttnn.bfloat16):
        raise ValueError("Require one padded user tile and FP32/BF16 joint")
    memory = packed.memory_config()
    if packed.layout != ttnn.TILE_LAYOUT or not (
        memory == ttnn.DRAM_MEMORY_CONFIG
        or memory.buffer_type == ttnn.BufferType.L1
        and memory.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
    ):
        raise ValueError("Joint must be tiled DRAM or width-sharded L1")
    device = packed.device()
    if device.arch() != ttnn.device.Arch.BLACKHOLE:
        raise ValueError("Blackhole native arithmetic only")
    for tensor, shape in [(history, (batch, 3, 2560)), *[(t, (1, 1, 2560)) for t in taps]]:
        if tuple(tensor.shape) != shape or tensor.dtype != ttnn.bfloat16 or tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise ValueError("History/taps must be BF16 row-major with exact local shape")
        if tensor.device() != device or tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
            raise ValueError("History/taps must be same-mesh DRAM")
    if type(rest_only) is not bool:
        raise ValueError("rest_only must be boolean")
    shapes = ((batch, 1, 2560), (1, batch, 1600 if rest_only else 4160))
    outputs = [output, rest]
    for i, layout in enumerate((ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT)):
        if outputs[i] is None:
            outputs[i] = ttnn.empty(
                shapes[i], dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
        tensor = outputs[i]
        if tuple(tensor.shape) != shapes[i] or tensor.dtype != ttnn.bfloat16 or tensor.layout != layout:
            raise ValueError("Output shape/dtype/layout mismatch")
        if tensor.device() != device or tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
            raise ValueError("Outputs must be owning same-mesh DRAM")
    output, rest = outputs
    inputs = [packed, history, *taps]
    addresses = [t.buffer_address() for t in inputs + outputs]
    if len(addresses) != len(set(addresses)):
        raise ValueError("Aliased inputs/history/outputs unsupported")
    grid = device.compute_with_storage_grid_size()
    if len(spans) > grid.x * grid.y:
        raise ValueError("Workers exceed mesh device grid")
    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(len(spans))]
    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    rr, wr, cr = (ttnn.RuntimeArgs() for _ in range(3))
    for c, (start, count, stride) in zip(coords, spans):
        rr[c.x][c.y] = [t.buffer_address() for t in inputs] + [batch, start, count, stride]
        wr[c.x][c.y] = [
            history.buffer_address(),
            output.buffer_address(),
            rest.buffer_address(),
            batch,
            start,
            count,
            stride,
        ]
        cr[c.x][c.y] = [start, count, stride]
    dtype = "float32" if packed.dtype == ttnn.float32 else "bfloat16"
    reader, compute, writer = sources(dtype, rest_only)

    def accessors(tensors):
        return [arg for t in tensors for arg in ttnn.TensorAccessorArgs(t).get_compile_time_args()]

    def kernel(source, config, rt, ct):
        return ttnn.KernelDescriptor(
            kernel_source=source,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=ranges,
            compile_time_args=ct,
            runtime_args=rt,
            config=config,
        )

    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
    )
    modes = [ttnn.UnpackToDestMode.Default] * 64
    if dtype == "float32":
        modes[0] = modes[1] = ttnn.UnpackToDestMode.UnpackToDestFp32
    config.unpack_to_dest_mode = modes
    kernels = [
        kernel(reader, ttnn.ReaderConfigDescriptor(), rr, accessors(inputs)),
        kernel(writer, ttnn.WriterConfigDescriptor(), wr, accessors([history, output, rest])),
        kernel(compute, config, cr, []),
    ]
    cbs = []
    for i in (0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15):
        data = packed.dtype if i in (0, 1) else ttnn.bfloat16
        page = (4096 if dtype == "float32" else 2048) if i in (0, 1) else (64 if i in (12, 13) else 2048)
        size = 6464 if i == 12 else 8256 if i == 13 else page
        cbs.append(
            ttnn.CBDescriptor(
                total_size=size,
                core_ranges=ranges,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=data, page_size=page)],
            )
        )
    ttnn.generic_op(inputs + outputs, ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return output, rest
