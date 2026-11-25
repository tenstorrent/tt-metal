# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Opt-in compact Q/K norm plus native-boundary partial RoPE; CPU-only import.

Q output is DRAM; K output directly owns cache-update shards on cores32..63.
Current bounded contract is BF16 DRAM trig [B,1,1,64], never host positions.
"""

import struct

READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr uint32_t GB=get_compile_time_arg_val(0);
    constexpr auto qa=TensorAccessorArgs<1>();
    constexpr auto ka=TensorAccessorArgs<qa.next_compile_time_args_offset()>();
    const auto qw=TensorAccessor(qa,get_arg_val<uint32_t>(2),GB);
    const auto kw=TensorAccessor(ka,get_arg_val<uint32_t>(3),GB);
    constexpr auto ca=TensorAccessorArgs<ka.next_compile_time_args_offset()>();
    constexpr auto sa=TensorAccessorArgs<ca.next_compile_time_args_offset()>();
    const auto cos_src=TensorAccessor(ca,get_arg_val<uint32_t>(7),2048);
    const auto sin_src=TensorAccessor(sa,get_arg_val<uint32_t>(8),2048);
    const uint32_t addresses[2]={get_arg_val<uint32_t>(0),get_arg_val<uint32_t>(1)};
    Noc noc; UnicastEndpoint self; CircularBuffer x(0),gamma(1);
    const uint32_t nx=get_arg_val<uint32_t>(5),ny=get_arg_val<uint32_t>(6);
    const uint32_t branch=get_arg_val<uint32_t>(4);
    const uint32_t user=get_arg_val<uint32_t>(9);
    CircularBuffer cos_cb(4),sin_cb(5),scalar(6);
    cos_cb.reserve_back(2);sin_cb.reserve_back(2);scalar.reserve_back(1);
    reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scalar.get_write_ptr())[0]=0xbf80u;
    scalar.push_back(1);
    for(uint32_t tile=0;tile<2;tile++) {
        noc.async_read(cos_src,cos_cb,2048,{.page_id=user*2+tile},{.offset_bytes=tile*2048});
        noc.async_read(sin_src,sin_cb,2048,{.page_id=user*2+tile},{.offset_bytes=tile*2048});
    }
    noc.async_read_barrier();cos_cb.push_back(2);sin_cb.push_back(2);
    {
        x.reserve_back(8);gamma.reserve_back(8);
        for(uint32_t tile=0;tile<8;tile++) {
            noc.async_read(self,x,2048,
                {.noc_x=nx,.noc_y=ny,.addr=addresses[branch]+tile*2048},{.offset_bytes=tile*2048});
            if(branch==0) noc.async_read(qw,gamma,GB,{.page_id=tile},{.offset_bytes=tile*GB});
            else noc.async_read(kw,gamma,GB,{.page_id=tile},{.offset_bytes=tile*GB});
        }
        noc.async_read_barrier();x.push_back(8);gamma.push_back(8);
    }
}
"""

COMPUTE = r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/dataflow/circular_buffer.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
namespace ckl=compute_kernel_lib;
void kernel_main() {
    constexpr uint32_t EPS_BITS=get_compile_time_arg_val(0);
    compute_kernel_hw_startup(0,0,2);
    CircularBuffer x(0),gamma(1),inverse(2),out(3);
    for(uint32_t branch=0;branch<1;branch++) {
        x.wait_front(8);gamma.wait_front(8);inverse.reserve_back(1);
        reconfig_data_format_srca(0);copy_init(0);pack_reconfig_data_format(2);
        tile_regs_acquire();
        for(uint32_t tile=0;tile<8;tile++) {
            const uint32_t work=tile?2:0;
            copy_tile(0,tile,work);copy_tile(0,tile,1);
            mul_binary_tile_init();mul_binary_tile(work,1,work);
            if(tile) { add_binary_tile_init();add_binary_tile(0,2,0); }
        }
        // Reduce columns within each independent head row; never reduce heads
        // or padded rows together. Full FP32 sum of 256 squared BF16 values.
        sfpu_reduce_init<PoolType::SUM,DataFormat::Float32>();
        sfpu_reduce<PoolType::SUM,DataFormat::Float32,ReduceDim::REDUCE_ROW>(0,1,1);
        binop_with_scalar_tile_init();mul_unary_tile(0,0x3b800000u); // 1/256
        binop_with_scalar_tile_init();add_unary_tile(0,EPS_BITS);
        rsqrt_tile_init();rsqrt_tile(0);
        tile_regs_commit();tile_regs_wait();pack_tile(0,2);tile_regs_release();
        inverse.push_back(1);inverse.wait_front(1);out.reserve_back(8);pack_reconfig_data_format(3);
        for(uint32_t tile=0;tile<8;tile++) {
            tile_regs_acquire();
            reconfig_data_format_srca(0);copy_init(0);copy_tile(0,tile,0);
            reconfig_data_format_srca(2);copy_init(2);copy_tile(2,0,2);
            sfpu_mul_bcast_col_init();sfpu_mul_bcast_col<true>(0,2);
            reconfig_data_format_srca(1);copy_init(1);copy_tile(1,tile,1);
            sfpu_mul_bcast_row_init();sfpu_mul_bcast_row<true>(0,1);
            typecast_tile_init<static_cast<uint32_t>(DataFormat::Float32),static_cast<uint32_t>(DataFormat::Float16_b)>();
            typecast_tile<static_cast<uint32_t>(DataFormat::Float32),static_cast<uint32_t>(DataFormat::Float16_b)>(0);
            tile_regs_commit();tile_regs_wait();pack_tile<true>(0,3,tile);tile_regs_release();
        }
        out.push_back(8);inverse.pop_front(1);x.pop_front(8);gamma.pop_front(8);
        // Keep the BF16 normalized CB boundary exactly as in compact_qk_norm.
        // Writer owns CB3's pop after both rotated prefix and untouched tail writes.
        CircularBuffer cos_cb(4),sin_cb(5),scalar(6),rotated(7),cos_product(8),sin_product(9),prefix(10);
        DataflowBuffer norm_dfb(3),cos_dfb(4),sin_dfb(5),scalar_dfb(6),rotated_dfb(7),cos_product_dfb(8),sin_product_dfb(9),prefix_dfb(10);
        out.wait_front(8);cos_cb.wait_front(2);sin_cb.wait_front(2);scalar.wait_front(1);
        rotated.reserve_back(2);cos_product.reserve_back(2);sin_product.reserve_back(2);prefix.reserve_back(2);
        reconfig_data_format(3,6);pack_reconfig_data_format(3,7);
        ckl::eltwise_chain(
            ckl::IterationShape::tiles(1).block_size(1),
            ckl::BinaryFpu<ckl::BinaryFpuOp::Mul,
                ckl::input(3,ckl::WaitPolicy::None,ckl::PopPolicy::None,ckl::InputTileMapping::Block,ckl::DataFormatReconfig::Enabled,ckl::TileAddressing::Offset),
                ckl::input(6,ckl::BroadcastDim::Scalar,ckl::WaitPolicy::None,ckl::PopPolicy::None,ckl::DataFormatReconfig::Enabled)>{norm_dfb,scalar_dfb,1u,0u},
            ckl::CopyTile<ckl::input(3,ckl::WaitPolicy::None,ckl::PopPolicy::None,ckl::InputTileMapping::Block),ckl::Dst::D1>{norm_dfb},
            ckl::PackTile<ckl::output(7,ckl::ReservePolicy::None,ckl::PushPolicy::None,ckl::DataFormatReconfig::Enabled,ckl::TileAddressing::Offset),ckl::Dst::D0>{rotated_dfb,0u},
            ckl::PackTile<ckl::output(7,ckl::ReservePolicy::None,ckl::PushPolicy::None,ckl::DataFormatReconfig::Enabled,ckl::TileAddressing::Offset),ckl::Dst::D1>{rotated_dfb,1u});
        rotated.push_back(2);
        constexpr auto held=[](uint32_t cb) {return ckl::input(cb,ckl::WaitPolicy::Upfront,ckl::PopPolicy::None,ckl::InputTileMapping::Block,ckl::DataFormatReconfig::Disabled);};
        constexpr auto consumed=[](uint32_t cb) {return ckl::input(cb,ckl::WaitPolicy::Upfront,ckl::PopPolicy::AtEnd,ckl::InputTileMapping::Block,ckl::DataFormatReconfig::Disabled);};
        constexpr auto produced=[](uint32_t cb) {return ckl::output(cb,ckl::ReservePolicy::None,ckl::PushPolicy::AtEnd,ckl::DataFormatReconfig::Disabled);};
        reconfig_data_format(7,5);pack_reconfig_data_format(7,9);mul_bcast_rows_init(7,5);
        ckl::eltwise_chain<ckl::InitReconfigOwner::Caller>(
            ckl::IterationShape::tiles(2).block_size(2),
            ckl::BinaryFpu<ckl::BinaryFpuOp::Mul,consumed(7),ckl::input(held(5),ckl::BroadcastDim::Row)>{rotated_dfb,sin_dfb},
            ckl::PackTile<produced(9)>{sin_product_dfb});
        reconfig_data_format(7,3,5,4);pack_reconfig_data_format(9,8);
        ckl::eltwise_chain<ckl::InitReconfigOwner::Caller>(
            ckl::IterationShape::tiles(2).block_size(2),
            ckl::BinaryFpu<ckl::BinaryFpuOp::Mul,held(3),ckl::input(held(4),ckl::BroadcastDim::Row)>{norm_dfb,cos_dfb},
            ckl::PackTile<produced(8)>{cos_product_dfb});
        reconfig_data_format(3,8,4,9);pack_reconfig_data_format(8,10);
        ckl::add<consumed(8),consumed(9),produced(10)>(ckl::IterationShape::tiles(2).block_size(2),cos_product_dfb,sin_product_dfb,prefix_dfb);
        cos_cb.pop_front(2);sin_cb.pop_front(2);scalar.pop_front(1);
    }
}
"""

WRITER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto qa=TensorAccessorArgs<0>();
    const auto q=TensorAccessor(qa,get_arg_val<uint32_t>(0),2048);
    const uint32_t k_addr=get_arg_val<uint32_t>(1),user=get_arg_val<uint32_t>(2),branch=get_arg_val<uint32_t>(3);
    const uint32_t nx=get_arg_val<uint32_t>(4),ny=get_arg_val<uint32_t>(5);
    Noc noc;UnicastEndpoint local;CircularBuffer norm(3),prefix(10);
    prefix.wait_front(2);norm.wait_front(8);
    for(uint32_t tile=0;tile<8;tile++) {
        if(tile<2) {
            if(branch==0) noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(prefix),q,2048,{.offset_bytes=tile*2048},{.page_id=user*8+tile});
            else noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(prefix),local,2048,{.offset_bytes=tile*2048},{.noc_x=nx,.noc_y=ny,.addr=k_addr+tile*2048});
        } else {
            if(branch==0) noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(norm),q,2048,{.offset_bytes=tile*2048},{.page_id=user*8+tile});
            else noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(norm),local,2048,{.offset_bytes=tile*2048},{.noc_x=nx,.noc_y=ny,.addr=k_addr+tile*2048});
        }
    }
    noc.async_write_barrier();prefix.pop_front(2);norm.pop_front(8);
}
"""


def compact_qk_rope(q, k, q_weight, k_weight, cos, sin, *, epsilon, outputs=None):
    """BF16 norm boundary plus native FPU rotary packing; caller-owned outputs optional."""
    import ttnn

    batch = q.shape[1]
    assert 1 <= batch <= 32 and epsilon > 0
    assert tuple(q.shape) == (1, batch, 6, 256) and tuple(k.shape) == (1, batch, 1, 256)
    assert q_weight.dtype == k_weight.dtype and q_weight.dtype in (ttnn.float32, ttnn.bfloat16)
    for w in (q_weight, k_weight):
        assert tuple(w.shape) == (1, 1, 256) and w.layout == ttnn.TILE_LAYOUT
        assert w.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    coords = [ttnn.CoreCoord(i % 8, i // 8) for i in range(batch)]
    core_ranges = []
    rows, tail = divmod(batch, 8)
    if rows:
        core_ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, rows - 1)))
    if tail:
        core_ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, rows), ttnn.CoreCoord(tail - 1, rows)))
    ranges = ttnn.CoreRangeSet(core_ranges)
    expected = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(ranges, [32, 256], ttnn.ShardOrientation.ROW_MAJOR),
    )
    for t in (q, k):
        assert t.dtype == ttnn.bfloat16 and t.layout == ttnn.TILE_LAYOUT and t.memory_config() == expected
    for trig in (cos, sin):
        assert tuple(trig.shape) == (batch, 1, 1, 64)
        assert trig.dtype == ttnn.bfloat16 and trig.layout == ttnn.TILE_LAYOUT
        assert trig.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    k_coords = [ttnn.CoreCoord(i % 8, i // 8) for i in range(32, 32 + batch)]
    k_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in k_coords])
    k_memory = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(k_ranges, [32, 256], ttnn.ShardOrientation.ROW_MAJOR),
    )
    if outputs is None:
        outputs = tuple(
            ttnn.empty(
                tuple(t.shape),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=t.device(),
                memory_config=ttnn.DRAM_MEMORY_CONFIG if i == 0 else k_memory,
            )
            for i, t in enumerate((q, k))
        )
    assert len(outputs) == 2
    for i, (original, t) in enumerate(zip((q, k), outputs)):
        assert tuple(t.shape) == tuple(original.shape) and t.dtype == ttnn.bfloat16
        assert t.layout == ttnn.TILE_LAYOUT and t.memory_config() == (ttnn.DRAM_MEMORY_CONFIG if i == 0 else k_memory)
    assert outputs[0].buffer_address() != outputs[1].buffer_address()
    assert all(
        t.buffer_address() not in tuple(v.buffer_address() for v in (q, k, q_weight, k_weight, cos, sin))
        for t in outputs
    )
    coords_key = tuple(q.device_coords())
    assert all(
        t.device() == q.device() and tuple(t.device_coords()) == coords_key
        for t in (k, q_weight, k_weight, cos, sin, *outputs)
    )
    # Input ranges describe B shard owners; launch 2B workers. Q remains local,
    # K runs on a distinct worker and reads the original owner's immutable shard.
    # Mesh virtual worker coordinates are invariant across this homogeneous TP4 mesh.
    device = q.device()
    grid = device.compute_with_storage_grid_size()
    assert grid.x >= 8 and grid.y >= (32 + batch + 7) // 8
    coords = [ttnn.CoreCoord(i % 8, i // 8) for i in range(batch)] + k_coords
    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    rr, wr, cr = (ttnn.RuntimeArgs() for _ in range(3))
    addresses = [q.buffer_address(), k.buffer_address(), q_weight.buffer_address(), k_weight.buffer_address()]
    for task, c in enumerate(coords):
        branch, user = divmod(task, batch)
        owner = device.worker_core_from_logical_core(ttnn.CoreCoord(user % 8, user // 8))
        rr[c.x][c.y] = addresses + [branch, owner.x, owner.y, cos.buffer_address(), sin.buffer_address(), user]
        destination = device.worker_core_from_logical_core(c)
        wr[c.x][c.y] = [
            outputs[0].buffer_address(),
            outputs[1].buffer_address(),
            user,
            branch,
            destination.x,
            destination.y,
        ]
        cr[c.x][c.y] = []
    gb = 4096 if q_weight.dtype == ttnn.float32 else 2048
    rct, wct = [gb], []
    for w in (q_weight, k_weight, cos, sin):
        rct.extend(ttnn.TensorAccessorArgs(w).get_compile_time_args())
    for t in outputs[:1]:
        wct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
    )
    modes = [ttnn.UnpackToDestMode.Default] * 64
    modes[2] = ttnn.UnpackToDestMode.UnpackToDestFp32
    if gb == 4096:
        modes[1] = ttnn.UnpackToDestMode.UnpackToDestFp32
    config.unpack_to_dest_mode = modes

    def kernel(source, config, args, ct):
        return ttnn.KernelDescriptor(
            kernel_source=source,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=ranges,
            compile_time_args=ct,
            runtime_args=args,
            config=config,
        )

    kernels = [
        kernel(READER, ttnn.ReaderConfigDescriptor(), rr, rct),
        kernel(WRITER, ttnn.WriterConfigDescriptor(), wr, wct),
        kernel(COMPUTE, config, cr, [struct.unpack("<I", struct.pack("<f", epsilon))[0]]),
    ]
    cbs = [
        ttnn.CBDescriptor(
            total_size=count * page,
            core_ranges=ranges,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)],
        )
        for index, count, page, dtype in [
            (0, 8, 2048, ttnn.bfloat16),
            (1, 8, gb, q_weight.dtype),
            (2, 1, 4096, ttnn.float32),
            (3, 8, 2048, ttnn.bfloat16),
            (4, 2, 2048, ttnn.bfloat16),
            (5, 2, 2048, ttnn.bfloat16),
            (6, 1, 2048, ttnn.bfloat16),
            (7, 2, 2048, ttnn.bfloat16),
            (8, 2, 2048, ttnn.bfloat16),
            (9, 2, 2048, ttnn.bfloat16),
            (10, 2, 2048, ttnn.bfloat16),
        ]
    ]
    ttnn.generic_op(
        [q, k, q_weight, k_weight, cos, sin, *outputs], ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[])
    )
    return outputs


def cpu_check():
    import json
    import sys

    for batch in range(1, 33):
        workers = list(range(batch)) + list(range(32, 32 + batch))
        assert len(set(workers)) == 2 * batch
        assert all(0 <= c < 64 for c in workers)
        for user in range(batch):
            assert user != 32 + user
            assert {user * 2, user * 2 + 1}.issubset(set(range(2 * batch)))
            assert [user * 8 + t for t in range(8)] == list(range(user * 8, (user + 1) * 8))
    assert "ttnn" not in sys.modules and "torch" not in sys.modules
    assert "typecast_tile<" in COMPUTE and "ckl::BinaryFpuOp::Mul" in COMPUTE
    assert "ckl::add<" in COMPUTE and "prefix.pop_front(2);norm.pop_front(8)" in WRITER
    print(json.dumps({"status": "pass", "scope": "CPU geometry/source only; hardware unverified", "batches": 32}))


if __name__ == "__main__":
    import sys

    assert sys.argv[1:] == ["--cpu-check"]
    cpu_check()
