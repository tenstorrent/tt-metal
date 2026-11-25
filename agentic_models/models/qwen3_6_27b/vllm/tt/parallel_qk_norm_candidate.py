# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Parallel Q/K RMSNorm candidate for compact TP4 decode.

Input Q/K remain HEIGHT_SHARDED on first B cores in eight-column row order.
Separate Q/K worker cores read each user's owning L1 shard and write DRAM outputs
for the unchanged partial-RoPE consumer. No device imports at module load.
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
    const uint32_t addresses[2]={get_arg_val<uint32_t>(0),get_arg_val<uint32_t>(1)};
    Noc noc; UnicastEndpoint self; CircularBuffer x(0),gamma(1);
    const uint32_t nx=get_arg_val<uint32_t>(5),ny=get_arg_val<uint32_t>(6);
    const uint32_t branch=get_arg_val<uint32_t>(4);
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
    }
}
"""

WRITER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto qa=TensorAccessorArgs<0>();
    constexpr auto ka=TensorAccessorArgs<qa.next_compile_time_args_offset()>();
    const auto q=TensorAccessor(qa,get_arg_val<uint32_t>(0),2048);
    const auto k=TensorAccessor(ka,get_arg_val<uint32_t>(1),2048);
    const uint32_t user=get_arg_val<uint32_t>(2);
    const uint32_t branch=get_arg_val<uint32_t>(3);
    Noc noc;CircularBuffer out(3);
    {
        out.wait_front(8);
        for(uint32_t tile=0;tile<8;tile++) {
            if(branch==0) noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(out),q,2048,
                {.offset_bytes=tile*2048},{.page_id=user*8+tile});
            else noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(out),k,2048,
                {.offset_bytes=tile*2048},{.page_id=user*8+tile});
        }
        noc.async_write_barrier();out.pop_front(8);
    }
}
"""


def compact_qk_norm(q, k, q_weight, k_weight, *, epsilon, outputs=None):
    """Parallel local Q/K norm, BF16 activations and unchanged FP32/BF16 weights.

    Exact current user/head shape and shard ownership are required. Epsilon is
    supplied by the model, not hard-coded. All statistics and affine arithmetic
    are FP32 with one BF16 output rounding. Native reduction order may differ;
    unchanged full-layer gates and focused reference comparisons must pass.
    """
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
    if outputs is None:
        outputs = tuple(
            ttnn.empty(
                tuple(t.shape),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=t.device(),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for t in (q, k)
        )
    assert len(outputs) == 2
    for original, t in zip((q, k), outputs):
        assert tuple(t.shape) == tuple(original.shape) and t.dtype == ttnn.bfloat16
        assert t.layout == ttnn.TILE_LAYOUT and t.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    assert outputs[0].buffer_address() != outputs[1].buffer_address()
    assert all(t.buffer_address() not in (q_weight.buffer_address(), k_weight.buffer_address()) for t in outputs)
    # Input ranges describe B shard owners; launch 2B workers. Q remains local,
    # K runs on a distinct worker and reads the original owner's immutable shard.
    # Mesh virtual worker coordinates are invariant across this homogeneous TP4 mesh.
    device = q.device()
    grid = device.compute_with_storage_grid_size()
    assert grid.x >= 8 and grid.y >= (2 * batch + 7) // 8
    coords = [ttnn.CoreCoord(i % 8, i // 8) for i in range(2 * batch)]
    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    rr, wr, cr = (ttnn.RuntimeArgs() for _ in range(3))
    addresses = [q.buffer_address(), k.buffer_address(), q_weight.buffer_address(), k_weight.buffer_address()]
    for task, c in enumerate(coords):
        branch, user = divmod(task, batch)
        owner = device.worker_core_from_logical_core(ttnn.CoreCoord(user % 8, user // 8))
        rr[c.x][c.y] = addresses + [branch, owner.x, owner.y]
        wr[c.x][c.y] = [outputs[0].buffer_address(), outputs[1].buffer_address(), user, branch]
        cr[c.x][c.y] = []
    gb = 4096 if q_weight.dtype == ttnn.float32 else 2048
    rct, wct = [gb], []
    for w in (q_weight, k_weight):
        rct.extend(ttnn.TensorAccessorArgs(w).get_compile_time_args())
    for t in outputs:
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
        ]
    ]
    ttnn.generic_op(
        [q, k, q_weight, k_weight, *outputs], ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[])
    )
    return outputs


def cpu_check():
    import json
    import sys

    cases = 0
    for batch in range(1, 33):
        tasks = []
        output_owners = set()
        for task in range(2 * batch):
            branch, user = divmod(task, batch)
            worker = (task % 8, task // 8)
            source = (user % 8, user // 8)
            assert worker[0] < 8 and worker[1] < 8
            assert source[1] < 4
            if branch == 0:
                assert worker == source
            else:
                assert worker != source
            for tile in range(8):
                key = (branch, user * 8 + tile)
                assert key not in output_owners
                output_owners.add(key)
                assert tile * 2048 + 2048 <= 16384
            tasks.append(worker)
        assert len(set(tasks)) == 2 * batch
        assert len(output_owners) == 16 * batch
        cases += 1
    assert "ttnn" not in sys.modules and "torch" not in sys.modules
    print(json.dumps({"status": "pass", "scope": "CPU ownership only; no hardware claim", "cases": cases}))


if __name__ == "__main__":
    import sys

    assert sys.argv[1:] == ["--cpu-check"]
    cpu_check()
