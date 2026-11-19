# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Opt-in packed QKVG to normalized/rotated QK, placed V and compact gate.

CPU-only import; exact compute math is shared with the validated QK/RoPE helper.
No source/input aliasing, no cross-worker semaphore, no material split operation.
"""

import struct

try:
    from .parallel_qk_rope_candidate import COMPUTE
except ImportError:  # Direct CPU geometry check, without importing TTNN.
    from parallel_qk_rope_candidate import COMPUTE

READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
void kernel_main() {
    constexpr uint32_t GB=get_compile_time_arg_val(0);
    constexpr auto xa=TensorAccessorArgs<1>();
    constexpr auto qa=TensorAccessorArgs<xa.next_compile_time_args_offset()>();
    constexpr auto ka=TensorAccessorArgs<qa.next_compile_time_args_offset()>();
    constexpr auto ca=TensorAccessorArgs<ka.next_compile_time_args_offset()>();
    constexpr auto sa=TensorAccessorArgs<ca.next_compile_time_args_offset()>();
    const auto packed=TensorAccessor(xa,get_arg_val<uint32_t>(0),2048);
    const auto qw=TensorAccessor(qa,get_arg_val<uint32_t>(1),GB);
    const auto kw=TensorAccessor(ka,get_arg_val<uint32_t>(2),GB);
    const auto cos_src=TensorAccessor(ca,get_arg_val<uint32_t>(3),2048);
    const auto sin_src=TensorAccessor(sa,get_arg_val<uint32_t>(4),2048);
    const uint32_t branch=get_arg_val<uint32_t>(5),user=get_arg_val<uint32_t>(6);
    Noc noc;CircularBuffer x(0),gamma(1),cos_cb(4),sin_cb(5),scalar(6),v(11),scratch(13);
    cos_cb.reserve_back(2);sin_cb.reserve_back(2);scalar.reserve_back(1);
    reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scalar.get_write_ptr())[0]=0xbf80u;
    scalar.push_back(1);
    for(uint32_t tile=0;tile<2;tile++) {
        noc.async_read(cos_src,cos_cb,2048,{.page_id=user*2+tile},{.offset_bytes=tile*2048});
        noc.async_read(sin_src,sin_cb,2048,{.page_id=user*2+tile},{.offset_bytes=tile*2048});
    }
    gamma.reserve_back(8);
    for(uint32_t tile=0;tile<8;tile++) {
        if(branch==0) noc.async_read(qw,gamma,GB,{.page_id=tile},{.offset_bytes=tile*GB});
        else noc.async_read(kw,gamma,GB,{.page_id=tile},{.offset_bytes=tile*GB});
    }
    noc.async_read_barrier();cos_cb.push_back(2);sin_cb.push_back(2);gamma.push_back(8);
    x.reserve_back(8);
    const uint32_t dst=x.get_write_ptr(),tmp=(scratch.get_write_ptr()+63u)&~63u;
    // Whole physical head tiles are owned by this worker. Initialize dead rows;
    // no reduction ever sees adjacent users or physical source padding.
    noc.async_write_zeros(CoreLocalMem<uint32_t>(dst),16384);
    if(branch==0) {v.reserve_back(8);noc.async_write_zeros(CoreLocalMem<uint32_t>(v.get_write_ptr()),16384);}
    noc.write_zeros_l1_barrier();
    const uint32_t nheads=branch==0?6:1,first=branch==0?0:48;
    for(uint32_t phase=0;phase<2;phase++) {
        const uint32_t row=(user/16)*1024+(user%16)*32+phase*512;
        const uint32_t skew=row&63u,aligned=row-skew;
        for(uint32_t i=0;i<nheads*8;i++)
            noc.async_read(packed,CoreLocalMem<uint32_t>(tmp+i*64),64,{.page_id=first+i,.offset_bytes=aligned},{});
        noc.async_read_barrier();
        for(uint32_t i=0;i<nheads*8;i++) {
            const uint32_t head=i/8,tile=i%8;
            tt::data_movement::common::tt_memmove<false,true,true,32>(
                noc,dst+tile*2048+head*32+phase*512,tmp+i*64+skew,32);
        }
        noc.async_read_barrier();
        if(branch==0) {
            for(uint32_t tile=0;tile<8;tile++)
                noc.async_read(packed,CoreLocalMem<uint32_t>(tmp+tile*64),64,{.page_id=56+tile,.offset_bytes=aligned},{});
            noc.async_read_barrier();
            for(uint32_t tile=0;tile<8;tile++)
                tt::data_movement::common::tt_memmove<false,true,true,32>(
                    noc,v.get_write_ptr()+tile*2048+phase*512,tmp+tile*64+skew,32);
            noc.async_read_barrier();
        }
    }
    x.push_back(8);if(branch==0) v.push_back(8);
}
"""

WRITER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr uint32_t B=get_compile_time_arg_val(0);
    constexpr auto qa=TensorAccessorArgs<1>();
    constexpr auto xa=TensorAccessorArgs<qa.next_compile_time_args_offset()>();
    constexpr auto ga=TensorAccessorArgs<xa.next_compile_time_args_offset()>();
    const auto q=TensorAccessor(qa,get_arg_val<uint32_t>(0),2048);
    const auto packed=TensorAccessor(xa,get_arg_val<uint32_t>(6),2048);
    const auto gate=TensorAccessor(ga,get_arg_val<uint32_t>(8),2048);
    const uint32_t k_addr=get_arg_val<uint32_t>(1),user=get_arg_val<uint32_t>(2),branch=get_arg_val<uint32_t>(3);
    const uint32_t nx=get_arg_val<uint32_t>(4),ny=get_arg_val<uint32_t>(5),v_addr=get_arg_val<uint32_t>(7);
    Noc noc;UnicastEndpoint local;CircularBuffer norm(3),prefix(10),v(11),gate_tiles(12);
    // Q worker writes its V shard and a disjoint striped subset of complete gate
    // tiles. Gate copies preserve all physical rows, including padding, exactly.
    // These independent copies overlap compute; the full-family timing includes them.
    if(branch==0) {
        v.wait_front(8);
        noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(v),local,16384,{},
            {.noc_x=nx,.noc_y=ny,.addr=v_addr});
        noc.async_write_barrier();v.pop_front(8);
        for(uint32_t first=user;first<48;first+=8*B) {
            const uint32_t count=((48-first+B-1)/B)<8?((48-first+B-1)/B):8;
            gate_tiles.reserve_back(count);
            for(uint32_t i=0;i<count;i++)
                noc.async_read(packed,gate_tiles,2048,{.page_id=64+first+i*B},{.offset_bytes=i*2048});
            noc.async_read_barrier();gate_tiles.push_back(count);gate_tiles.wait_front(count);
            for(uint32_t i=0;i<count;i++)
                noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(gate_tiles),gate,2048,
                    {.offset_bytes=i*2048},{.page_id=first+i*B});
            noc.async_write_barrier();gate_tiles.pop_front(count);
        }
    }
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


def packed_qk_rope(packed, q_weight, k_weight, cos, sin, *, epsilon, outputs=None):
    """BF16 norm boundary plus native FPU rotary packing; caller-owned outputs optional."""
    import ttnn

    batch = packed.shape[1]
    assert 1 <= batch <= 32 and epsilon > 0
    assert tuple(packed.shape) == (1, batch, 3584)
    assert tuple(packed.padded_shape) == (1, 32, 3584)
    assert packed.dtype == ttnn.bfloat16 and packed.layout == ttnn.TILE_LAYOUT
    assert packed.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    assert q_weight.dtype == k_weight.dtype and q_weight.dtype in (ttnn.float32, ttnn.bfloat16)
    for w in (q_weight, k_weight):
        assert tuple(w.shape) == (1, 1, 256) and w.layout == ttnn.TILE_LAYOUT
        assert tuple(w.padded_shape) == (1, 32, 256)
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
    for trig in (cos, sin):
        assert tuple(trig.shape) == (batch, 1, 1, 64)
        assert tuple(trig.padded_shape) == (batch, 1, 32, 64)
        assert trig.dtype == ttnn.bfloat16 and trig.layout == ttnn.TILE_LAYOUT
        assert trig.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    k_coords = [ttnn.CoreCoord(i % 8, i // 8) for i in range(32, 32 + batch)]
    k_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in k_coords])
    k_memory = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(k_ranges, [32, 256], ttnn.ShardOrientation.ROW_MAJOR),
    )
    shapes = [(1, batch, 6, 256), (1, batch, 1, 256), (1, batch, 1, 256), (1, batch, 1536)]
    memories = [ttnn.DRAM_MEMORY_CONFIG, k_memory, expected, ttnn.DRAM_MEMORY_CONFIG]
    if outputs is None:
        outputs = tuple(
            ttnn.empty(
                shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=packed.device(), memory_config=memory
            )
            for shape, memory in zip(shapes, memories)
        )
    assert len(outputs) == 4
    padded_shapes = [(1, batch, 32, 256)] * 3 + [(1, 32, 1536)]
    for tensor, shape, padded, memory in zip(outputs, shapes, padded_shapes, memories):
        assert tuple(tensor.shape) == shape and tensor.dtype == ttnn.bfloat16
        assert tuple(tensor.padded_shape) == padded
        assert tensor.layout == ttnn.TILE_LAYOUT and tensor.memory_config() == memory
    inputs = (packed, q_weight, k_weight, cos, sin)
    assert len({t.buffer_address() for t in outputs}) == 4
    assert all(t.buffer_address() not in {v.buffer_address() for v in inputs} for t in outputs)
    coords_key = tuple(packed.device_coords())
    assert all(t.device() == packed.device() and tuple(t.device_coords()) == coords_key for t in (*inputs, *outputs))
    # Launch 2B workers; each gathers its user row from packed DRAM tiles.
    # Q also owns V output and striped full gate tiles; K owns offset32 shards.
    # Mesh virtual worker coordinates are invariant across this homogeneous TP4 mesh.
    device = packed.device()
    grid = device.compute_with_storage_grid_size()
    assert grid.x >= 8 and grid.y >= (32 + batch + 7) // 8
    coords = [ttnn.CoreCoord(i % 8, i // 8) for i in range(batch)] + k_coords
    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    rr, wr, cr = (ttnn.RuntimeArgs() for _ in range(3))
    for task, c in enumerate(coords):
        branch, user = divmod(task, batch)
        rr[c.x][c.y] = [t.buffer_address() for t in inputs] + [branch, user]
        destination = device.worker_core_from_logical_core(c)
        wr[c.x][c.y] = [
            outputs[0].buffer_address(),
            outputs[1].buffer_address(),
            user,
            branch,
            destination.x,
            destination.y,
            packed.buffer_address(),
            outputs[2].buffer_address(),
            outputs[3].buffer_address(),
        ]
        cr[c.x][c.y] = []
    gb = 4096 if q_weight.dtype == ttnn.float32 else 2048
    rct, wct = [gb], [batch]
    for tensor in inputs:
        rct.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    for tensor in (outputs[0], packed, outputs[3]):
        wct.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
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
            (11, 8, 2048, ttnn.bfloat16),
            (12, 8, 2048, ttnn.bfloat16),
            (13, 49, 64, ttnn.bfloat16),
        ]
    ]
    ttnn.generic_op([*inputs, *outputs], ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[]))
    return outputs


def cpu_check():
    import json
    import sys

    for batch in range(1, 33):
        workers = list(range(batch)) + list(range(32, 32 + batch))
        assert len(set(workers)) == 2 * batch and max(workers) < 64
        gate_tiles = []
        for user in range(batch):
            gate_tiles += list(range(user, 48, batch))
            for phase in range(2):
                row = (user // 16) * 1024 + (user % 16) * 32 + phase * 512
                aligned = row - row % 64
                assert aligned % 64 == 0 and aligned + 64 <= 2048
                assert aligned <= row and row + 32 <= aligned + 64
                for branch, first, heads in ((0, 0, 6), (1, 48, 1), (2, 56, 1)):
                    for tile in range(heads * 8):
                        head, channel = divmod(tile, 8)
                        # Every byte moved belongs to the same logical user;
                        # source physical padding is never part of the fragment.
                        assert 0 <= first + tile < 64 and head < heads
                        dst = channel * 2048 + head * 32 + phase * 512
                        assert dst + 32 <= 16384
                        for column in range(16):
                            # Decode the tiled source and destination byte addresses
                            # independently; this also covers face boundaries 15/16.
                            src_element = (row + column * 2) // 2
                            src_face, src_in_face = divmod(src_element, 256)
                            src_r, src_c = divmod(src_in_face, 16)
                            assert (src_face // 2) * 16 + src_r == user
                            feature = (first + tile) * 32 + (src_face % 2) * 16 + src_c
                            expected_feature = first * 32 + head * 256 + channel * 32 + phase * 16 + column
                            assert feature == expected_feature
                            dst_element = (dst % 2048 + column * 2) // 2
                            dst_face, dst_in_face = divmod(dst_element, 256)
                            dst_r, dst_c = divmod(dst_in_face, 16)
                            assert (dst_face // 2) * 16 + dst_r == head
                            assert (dst_face % 2) * 16 + dst_c == phase * 16 + column
            for first in range(user, 48, 8 * batch):
                count = min((48 - first + batch - 1) // batch, 8)
                assert all(first + i * batch < 48 for i in range(count))
        assert sorted(gate_tiles) == list(range(48))
    assert "ttnn" not in sys.modules and "torch" not in sys.modules
    assert "typecast_tile<" in COMPUTE and "ckl::BinaryFpuOp::Mul" in COMPUTE
    assert "tt_memmove<false,true,true,32>" in READER
    print(
        json.dumps(
            {
                "status": "pass",
                "scope": "CPU layout/source only; hardware unverified",
                "batches": 32,
                "packed_tiles": 112,
                "gate_tiles": 48,
                "outputs": ["Q DRAM", "K offset32", "V firstB", "gate DRAM"],
            }
        )
    )


if __name__ == "__main__":
    import sys

    assert sys.argv[1:] == ["--cpu-check"]
    cpu_check()
