# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Opt-in local12 readout cast/RMS/affine/SiLU gate. No device imports at load.

Native FPU normalization order and BF16 boundaries are retained in source;
hardware equivalence and speed require the companion probe. No production hook.
"""

import math
import struct

# id, capacity tiles, bytes/tile; all non-FP32 entries are BF16.
CB_LAYOUT = (
    (0, 4, 4096),
    (1, 4, 2048),
    (2, 4, 4096),
    (3, 1, 2048),
    (4, 1, 2048),
    (5, 1, 4096),
    (6, 1, 4096),
    (7, 4, 2048),
    (8, 4, 2048),
    (9, 4, 2048),
    (10, 4, 2048),
)

READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr bool TILE_OWNER=get_compile_time_arg_val(0);
    constexpr uint32_t EPS_BITS=get_compile_time_arg_val(1);
    constexpr auto ra=TensorAccessorArgs<2>();
    constexpr auto wa=TensorAccessorArgs<ra.next_compile_time_args_offset()>();
    constexpr auto pa=TensorAccessorArgs<wa.next_compile_time_args_offset()>();
    const auto r=TensorAccessor(ra,get_arg_val<uint32_t>(0),4096);
    const auto w=TensorAccessor(wa,get_arg_val<uint32_t>(1),2048);
    const auto p=TensorAccessor(pa,get_arg_val<uint32_t>(2),2048);
    const uint32_t B=get_arg_val<uint32_t>(3),start=get_arg_val<uint32_t>(4),count=get_arg_val<uint32_t>(5);
    Noc noc; CircularBuffer raw(0),scalar(3),epsilon(4),gamma(8),gate(9);
    scalar.reserve_back(1);epsilon.reserve_back(1);
    volatile tt_l1_ptr uint16_t* sp=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scalar.get_write_ptr());
    volatile tt_l1_ptr uint16_t* ep=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(epsilon.get_write_ptr());
    for(uint32_t i=0;i<1024;i++){sp[i]=0x3f80;ep[i]=0;}
    // Native generate_bcast_col_scalar: truncate FP32 epsilon high16.
    for(uint32_t face=0;face<4;face+=2) for(uint32_t row=0;row<16;row++)
        ep[face*256+row*16]=EPS_BITS>>16;
    scalar.push_back(1);epsilon.push_back(1);
    for(uint32_t task=start;task<start+count;task++) {
        const uint32_t head=TILE_OWNER?task/4:task;
        raw.reserve_back(4);gamma.reserve_back(4);gate.reserve_back(4);
        // Initialize padding before ordinary reads; BH zero uses read-loopback.
        noc.async_write_zeros(raw,4*4096);noc.write_zeros_l1_barrier();
        for(uint32_t j=0;j<4;j++) {
            for(uint32_t user=0;user<B;user++) for(uint32_t face=0;face<2;face++) {
                const uint32_t row=user*12+head;
                const uint32_t src=(row%32/16)*2048+face*1024+(row%16)*64;
                const uint32_t dst=j*4096+(user/16)*2048+face*1024+(user%16)*64;
                noc.async_read(r,raw,64,{.page_id=(row/32)*4+j,.offset_bytes=src},{.offset_bytes=dst});
            }
            noc.async_read(w,gamma,2048,{.page_id=j},{.offset_bytes=j*2048});
            noc.async_read(p,gate,2048,{.page_id=80+head*4+j},{.offset_bytes=j*2048});
        }
        noc.async_read_barrier();
        // Whole gate tiles include public-input padding, which must not leak.
        for(uint32_t j=0;j<4;j++) for(uint32_t user=B;user<32;user++) for(uint32_t face=0;face<2;face++)
            noc.async_write_zeros(gate,32,{.offset_bytes=j*2048+(user/16)*1024+face*512+(user%16)*32});
        noc.write_zeros_l1_barrier();raw.push_back(4);gamma.push_back(4);gate.push_back(4);
    }
}
"""

COMPUTE = r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/bcast.h"
#include "api/compute/reduce.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/pack.h"
#include "api/dataflow/circular_buffer.h"
inline void round_bf16(uint32_t d) {
    typecast_tile_init<static_cast<uint32_t>(DataFormat::Float32),static_cast<uint32_t>(DataFormat::Float16_b)>();
    typecast_tile<static_cast<uint32_t>(DataFormat::Float32),static_cast<uint32_t>(DataFormat::Float16_b)>(d);
}
inline void finish(uint32_t cb,uint32_t index=0) {
    tile_regs_commit();tile_regs_wait();pack_tile<true>(0,cb,index);tile_regs_release();
}
void kernel_main() {
    constexpr bool TILE_OWNER=get_compile_time_arg_val(0);
    const uint32_t start=get_arg_val<uint32_t>(0),count=get_arg_val<uint32_t>(1);
    compute_kernel_hw_startup(0,0,1);
    CircularBuffer raw(0),x(1),square(2),scalar(3),epsilon(4),mean(5),inverse(6),norm(7),gamma(8),gate(9),out(10);
    scalar.wait_front(1);epsilon.wait_front(1);
    for(uint32_t task=start;task<start+count;task++) {
        raw.wait_front(4);gamma.wait_front(4);gate.wait_front(4);
        x.reserve_back(4);reconfig_data_format_srca(0);copy_init(0);pack_reconfig_data_format(1);
        for(uint32_t j=0;j<4;j++){tile_regs_acquire();copy_tile(0,j,0);round_bf16(0);finish(1,j);}
        x.push_back(4);x.wait_front(4);raw.pop_front(4);
        // Native RMS: FPU squares, four tiles in increasing reduction order.
        square.reserve_back(4);reconfig_data_format(1,1);mul_init(1,1,false);pack_reconfig_data_format(2);
        for(uint32_t j=0;j<4;j++){tile_regs_acquire();mul_tiles(1,1,j,j,0);finish(2,j);}
        square.push_back(4);square.wait_front(4);mean.reserve_back(1);
        reconfig_data_format(3,2);tile_regs_acquire();
        reduce_init<PoolType::SUM,ReduceDim::REDUCE_ROW>(2,3,5);
        for(uint32_t j=0;j<4;j++) reduce_tile<PoolType::SUM,ReduceDim::REDUCE_ROW>(2,3,j,0,0);
        reduce_uninit();reconfig_data_format(2,3);
        binop_with_scalar_tile_init();mul_unary_tile(0,0x3c000000u);
        pack_reconfig_data_format(5);finish(5);mean.push_back(1);mean.wait_front(1);square.pop_front(4);
        inverse.reserve_back(1);reconfig_data_format(5,4);add_init(5,4);pack_reconfig_data_format(6);
        tile_regs_acquire();add_tiles(5,4,0,0,0);rsqrt_tile_init();rsqrt_tile<ckernel::RsqrtMode::Default>(0);
        finish(6);inverse.push_back(1);inverse.wait_front(1);mean.pop_front(1);
        // Native unweighted norm ends with a real BF16 pack, before gamma.
        norm.reserve_back(4);reconfig_data_format(1,6);mul_bcast_cols_init(1,6,false);pack_reconfig_data_format(7);
        for(uint32_t j=0;j<4;j++){tile_regs_acquire();mul_tiles_bcast_cols(1,6,j,0,0);finish(7,j);}
        norm.push_back(4);norm.wait_front(4);inverse.pop_front(1);x.pop_front(4);
        const uint32_t first=TILE_OWNER?task%4:0,nt=TILE_OWNER?1:4;
        out.reserve_back(nt);pack_reconfig_data_format(10);
        for(uint32_t i=0;i<nt;i++) {
            const uint32_t j=first+i;
            reconfig_data_format_srca(7);copy_init(7);tile_regs_acquire();copy_tile(7,j,0);
            reconfig_data_format_srca(8);copy_init(8);copy_tile(8,j,1);
            sfpu_mul_bcast_row_init();sfpu_mul_bcast_row<true>(0,1);round_bf16(0);
            reconfig_data_format_srca(9);copy_init(9);copy_tile(9,j,2);
            silu_tile_init();silu_tile<true>(2);mul_binary_tile_init();mul_binary_tile(2,0,0);round_bf16(0);
            finish(10,i);
        }
        out.push_back(nt);norm.pop_front(4);gamma.pop_front(4);gate.pop_front(4);
    }
    scalar.pop_front(1);epsilon.pop_front(1);
}
"""

WRITER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr bool TILE_OWNER=get_compile_time_arg_val(0);
    constexpr auto oa=TensorAccessorArgs<1>();
    const auto output=TensorAccessor(oa,get_arg_val<uint32_t>(0),2048);
    const uint32_t start=get_arg_val<uint32_t>(1),count=get_arg_val<uint32_t>(2);
    Noc noc;CircularBuffer out(10);
    for(uint32_t task=start;task<start+count;task++) {
        const uint32_t nt=TILE_OWNER?1:4,first=TILE_OWNER?task:4*task;
        out.wait_front(nt);
        for(uint32_t j=0;j<nt;j++) noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(out),output,2048,
            {.offset_bytes=j*2048},{.page_id=first+j});
        noc.async_write_barrier();out.pop_front(nt);
    }
}
"""


def task_partition(ownership, available, cores=None):
    assert ownership in ("head", "tile") and available >= 1
    assert cores is None or 1 <= cores <= available
    tasks = 12 if ownership == "head" else 48
    workers = min(tasks, cores or available)
    result, start = [], 0
    for i in range(workers):
        count = tasks // workers + (i < tasks % workers)
        result.append((start, count))
        start += count
    return result


def local_norm_gate(readout_fp32, weight, packed, *, epsilon=1e-6, ownership="head", output=None, cores=None):
    """FP32 [1,B*12,128] -> gated BF16 [1,B,1536], compact decode only."""
    import ttnn

    assert math.isfinite(epsilon) and epsilon > 0
    assert len(readout_fp32.shape) == 3 and readout_fp32.shape[0] == 1 and readout_fp32.shape[2] == 128
    assert readout_fp32.shape[1] % 12 == 0
    batch = readout_fp32.shape[1] // 12
    assert 1 <= batch <= 32 and tuple(weight.shape) == (1, 1, 128)
    assert tuple(readout_fp32.padded_shape) == (1, math.ceil(batch * 12 / 32) * 32, 128)
    assert len(packed.shape) == 3 and tuple(packed.shape)[:2] == (1, batch) and packed.shape[2] >= 4096
    assert tuple(weight.padded_shape) == (1, 32, 128)
    assert packed.padded_shape[-2] == 32 and packed.padded_shape[-1] % 32 == 0
    assert readout_fp32.dtype == ttnn.float32 and weight.dtype == packed.dtype == ttnn.bfloat16
    device = readout_fp32.device()
    for t in (readout_fp32, weight, packed):
        assert t.layout == ttnn.TILE_LAYOUT and t.memory_config() == ttnn.DRAM_MEMORY_CONFIG and t.device() == device
    if output is None:
        output = ttnn.empty(
            (1, batch, 1536),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    assert tuple(output.shape) == (1, batch, 1536) and tuple(output.padded_shape) == (1, 32, 1536)
    assert output.dtype == ttnn.bfloat16 and output.layout == ttnn.TILE_LAYOUT
    assert output.device() == device and output.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    assert output.buffer_address() not in [t.buffer_address() for t in (readout_fp32, weight, packed)]
    grid = device.compute_with_storage_grid_size()
    partition = task_partition(ownership, grid.x * grid.y, cores)
    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(len(partition))]
    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    rr, wr, cr = (ttnn.RuntimeArgs() for _ in range(3))
    for c, (start, count) in zip(coords, partition):
        rr[c.x][c.y] = [t.buffer_address() for t in (readout_fp32, weight, packed)] + [batch, start, count]
        wr[c.x][c.y] = [output.buffer_address(), start, count]
        cr[c.x][c.y] = [start, count]
    flag = int(ownership == "tile")
    rct = [flag, struct.unpack("<I", struct.pack("<f", epsilon))[0]]
    for t in (readout_fp32, weight, packed):
        rct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    wct = [flag, *ttnn.TensorAccessorArgs(output).get_compile_time_args()]
    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
    )
    modes = [ttnn.UnpackToDestMode.Default] * 64
    modes[0] = ttnn.UnpackToDestMode.UnpackToDestFp32
    # Native RMS intermediates use ordinary UnpackToSrc, not direct-DST aliases.
    config.unpack_to_dest_mode = modes

    def kernel(source, cfg, rt, ct):
        return ttnn.KernelDescriptor(
            kernel_source=source,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=ranges,
            runtime_args=rt,
            compile_time_args=ct,
            config=cfg,
        )

    kernels = [
        kernel(READER, ttnn.ReaderConfigDescriptor(), rr, rct),
        kernel(WRITER, ttnn.WriterConfigDescriptor(), wr, wct),
        kernel(COMPUTE, config, cr, [flag]),
    ]
    cbs = [
        ttnn.CBDescriptor(
            total_size=count * page,
            core_ranges=ranges,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=i, data_format=ttnn.float32 if page == 4096 else ttnn.bfloat16, page_size=page
                )
            ],
        )
        for i, count, page in CB_LAYOUT
    ]
    ttnn.generic_op(
        [readout_fp32, weight, packed, output], ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[])
    )
    return output


def cpu_check():
    cases = 0
    for batch in range(1, 33):
        for ownership in ("head", "tile"):
            for cores in (1, 2, 7, 12, 48, 110):
                output_tiles = []
                for start, count in task_partition(ownership, 110, cores):
                    for task in range(start, start + count):
                        head = task // 4 if ownership == "tile" else task
                        columns = [task % 4] if ownership == "tile" else range(4)
                        output_tiles.extend(4 * head + j for j in columns)
                        for user in range(batch):
                            rh = 12 * user + head
                            for j in range(4):
                                for face in range(2):
                                    src = (rh % 32 // 16) * 2048 + face * 1024 + (rh % 16) * 64
                                    dst = j * 4096 + (user // 16) * 2048 + face * 1024 + (user % 16) * 64
                                    assert src % 64 == dst % 64 == 0 and src + 64 <= 4096 and dst + 64 <= 16384
                                    assert (rh // 32) * 4 + j < math.ceil(batch * 12 / 32) * 4
                assert sorted(output_tiles) == list(range(48))
                cases += 1
    assert len(set(i for i, _, _ in CB_LAYOUT)) == len(CB_LAYOUT)
    assert all(count * page % page == 0 for _, count, page in CB_LAYOUT)
    return {
        "status": "pass",
        "scope": "CPU geometry only; hardware unverified",
        "cases": cases,
        "cb_bytes_per_core": sum(count * page for _, count, page in CB_LAYOUT),
    }
