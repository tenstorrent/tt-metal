# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Opt-in packed BF16 gate/up epilogue; imports TTNN only when called.

Input is [1,rows,8704] with contiguous gate/up halves, never paired weights.
Native Blackhole BF16 silu_tile and mul_binary_tile preserve both existing
reference chains' BF16 roundings. Device bitwise evidence is still required.
"""

HALF = 4352
HALF_TILES = HALF // 32

_READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto xa=TensorAccessorArgs<0>();
    const auto x=TensorAccessor(xa,get_arg_val<uint32_t>(0),2048);
    const uint32_t stride=get_arg_val<uint32_t>(1),start=get_arg_val<uint32_t>(2),count=get_arg_val<uint32_t>(3);
    Noc noc; CircularBuffer gate(0),up(1);
    for(uint32_t t=start;t<start+count;t++) {
        const uint32_t row=t/136,col=t%136;
        gate.reserve_back(1);up.reserve_back(1);
        noc.async_read(x,gate,2048,{.page_id=row*stride+col},{.offset_bytes=0});
        noc.async_read(x,up,2048,{.page_id=row*stride+136+col},{.offset_bytes=0});
        noc.async_read_barrier();
        gate.push_back(1);up.push_back(1);
    }
}
"""

_WRITER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto oa=TensorAccessorArgs<0>();
    const auto output=TensorAccessor(oa,get_arg_val<uint32_t>(0),2048);
    const uint32_t start=get_arg_val<uint32_t>(1),count=get_arg_val<uint32_t>(2);
    Noc noc; CircularBuffer out(2);
    for(uint32_t t=start;t<start+count;t++) {
        out.wait_front(1);
        noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(out),output,2048,
                        {.offset_bytes=0},{.page_id=t});
        noc.async_write_barrier();out.pop_front(1);
    }
}
"""

_PREFIX = r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/dataflow/circular_buffer.h"
void kernel_main() {
    const uint32_t count=get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(0,2);copy_init(0);
    CircularBuffer gate(0),up(1),out(2);
"""

_REGISTER_COMPUTE = _PREFIX + r"""
    for(uint32_t t=0;t<count;t++) {
        gate.wait_front(1);up.wait_front(1);out.reserve_back(1);
        tile_regs_acquire();
        copy_init(0);copy_tile(0,0,0);
        copy_init(1);copy_tile(1,0,1);
        // Native BF16 SiLU explicitly rounds to BF16, including when used
        // as multiply's input activation. Do not select the FP32 sigmoid.
        silu_tile_init();silu_tile<false>(0);
        // Native BF16 MUL applies software RNE and native zero handling.
        mul_binary_tile_init();mul_binary_tile<false>(0,1,0);
        tile_regs_commit();tile_regs_wait();pack_tile(0,2);tile_regs_release();
        gate.pop_front(1);up.pop_front(1);out.push_back(1);
    }
}
"""

_ROUNDTRIP_COMPUTE = _PREFIX + r"""
    CircularBuffer activated(3);
    for(uint32_t t=0;t<count;t++) {
        gate.wait_front(1);activated.reserve_back(1);
        pack_reconfig_data_format(2,3);
        tile_regs_acquire();copy_init(0);copy_tile(0,0,0);
        silu_tile_init();silu_tile<false>(0);
        tile_regs_commit();tile_regs_wait();pack_tile(0,3);tile_regs_release();
        gate.pop_front(1);activated.push_back(1);
        pack_reconfig_data_format(3,2);
        activated.wait_front(1);up.wait_front(1);out.reserve_back(1);
        tile_regs_acquire();
        copy_init(3);copy_tile(3,0,0);copy_init(1);copy_tile(1,0,1);
        mul_binary_tile_init();mul_binary_tile<false>(0,1,0);
        tile_regs_commit();tile_regs_wait();pack_tile(0,2);tile_regs_release();
        activated.pop_front(1);up.pop_front(1);out.push_back(1);
    }
}
"""


def tile_plan(rows, *, input_width=8704, cores=110):
    """CPU-only page mapping and contiguous, exhaustive worker assignment."""
    if type(rows) is not int or rows < 1 or input_width < 8704 or input_width % 32:
        raise ValueError("Require positive rows and tile-aligned physical width >=8704")
    if type(cores) is not int or cores < 1:
        raise ValueError("cores must be positive")
    tiles = ((rows + 31) // 32) * HALF_TILES
    workers = min(tiles, cores)
    spans, start = [], 0
    for i in range(workers):
        count = tiles // workers + (i < tiles % workers)
        spans.append((start, count))
        start += count
    return {
        "rows": rows,
        "output_tiles": tiles,
        "input_stride_tiles": input_width // 32,
        "workers": workers,
        "spans": spans,
        "input_payload_bytes": tiles * 4096,
        "output_payload_bytes": tiles * 2048,
    }


def output_shard_shape(rows, cores):
    """Canonical width-sharded output, including any final shard padding."""
    if type(rows) is not int or rows < 1 or type(cores) is not int or cores < 1:
        raise ValueError("Positive logical rows and storage-core count required")
    return ((rows + 31) // 32 * 32, (HALF_TILES + cores - 1) // cores * 32)


def output_memory_config(packed):
    """Keep the existing buffer tier; preserve width sharding for the next MM.

    TensorAccessor carries each tensor's actual distribution, so the unchanged
    global tile reader/writer support interleaved DRAM/L1 and width-sharded L1.
    No tensor migration is performed here.
    """
    import ttnn

    memory = packed.memory_config()
    if memory in (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG):
        return memory
    if memory.buffer_type != ttnn.BufferType.L1 or memory.memory_layout != ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        raise ValueError("Packed epilogue supports interleaved DRAM/L1 or width-sharded L1")
    spec = memory.shard_spec
    shape = output_shard_shape(packed.shape[1], spec.grid.num_cores())
    if spec.shape[0] != shape[0]:
        raise ValueError("Width-sharded input must cover the complete padded row extent")
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(spec.grid, shape, spec.orientation),
    )


def packed_swiglu(packed, *, round_silu, output=None, cores=None, implementation="register"):
    """One device op, optional caller-owned persistent DRAM/L1 output.

    round_silu names the unchanged source chain (True: explicit SiLU; False:
    input activation). Current BF16 native implementations round SiLU in both.
    The conservative roundtrip variant preserves the intermediate BF16 CB hop;
    register removes that idempotent hop using the same native math functions.
    No host tensor conversion, layout fallback, weights or owner is retained.
    """
    import ttnn

    if type(round_silu) is not bool or implementation not in ("register", "roundtrip"):
        raise ValueError("Select a source rounding contract and supported implementation")
    if len(packed.shape) != 3 or packed.shape[0] != 1 or packed.shape[2] != 8704:
        raise ValueError("Expected compact [1,rows,8704] contiguous gate/up output")
    if packed.dtype != ttnn.bfloat16 or packed.layout != ttnn.TILE_LAYOUT:
        raise ValueError("Only already-rounded BF16 TILE projection output is supported")
    destination = output_memory_config(packed)
    device = packed.device()
    if device.arch() != ttnn.device.Arch.BLACKHOLE:
        raise ValueError("This prototype's native arithmetic audit covers Blackhole only")
    rows = packed.shape[1]
    grid = device.compute_with_storage_grid_size()
    maximum = grid.x * grid.y
    if cores is not None and (type(cores) is not int or not 1 <= cores <= maximum):
        raise ValueError("cores exceeds the compute grid")
    plan = tile_plan(rows, input_width=packed.padded_shape[-1], cores=cores or maximum)
    if output is None:
        output = ttnn.empty(
            (1, rows, HALF),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=destination,
        )
    if tuple(output.shape) != (1, rows, HALF) or output.dtype != ttnn.bfloat16:
        raise ValueError("Persistent output shape/dtype mismatch")
    if output.layout != ttnn.TILE_LAYOUT:
        raise ValueError("Persistent output must be TILE")
    memory = output.memory_config()
    if memory not in (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG):
        if memory.buffer_type != ttnn.BufferType.L1 or memory.memory_layout != ttnn.TensorMemoryLayout.WIDTH_SHARDED:
            raise ValueError("Persistent output must be interleaved DRAM/L1 or width-sharded L1")
        spec = memory.shard_spec
        if tuple(spec.shape) != output_shard_shape(rows, spec.grid.num_cores()):
            raise ValueError("Persistent width-sharded output must use canonical row/channel shards")
    if tuple(output.padded_shape) != (1, (rows + 31) // 32 * 32, HALF):
        raise ValueError("Persistent output must have canonical tile padding")
    if output.device() != device:
        raise ValueError("Persistent output must belong to the same device/mesh")
    if output.buffer_address() == packed.buffer_address():
        raise ValueError("In-place packed input/output aliasing is unsupported")
    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(plan["workers"])]
    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    rr, wr, cr = (ttnn.RuntimeArgs() for _ in range(3))
    for c, (start, count) in zip(coords, plan["spans"]):
        rr[c.x][c.y] = [packed.buffer_address(), plan["input_stride_tiles"], start, count]
        wr[c.x][c.y] = [output.buffer_address(), start, count]
        cr[c.x][c.y] = [count]

    def kernel(source, config, runtime, ct):
        return ttnn.KernelDescriptor(
            kernel_source=source,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=ranges,
            compile_time_args=ct,
            runtime_args=runtime,
            config=config,
        )

    kernels = [
        kernel(_READER, ttnn.ReaderConfigDescriptor(), rr, ttnn.TensorAccessorArgs(packed).get_compile_time_args()),
        kernel(_WRITER, ttnn.WriterConfigDescriptor(), wr, ttnn.TensorAccessorArgs(output).get_compile_time_args()),
        kernel(
            _REGISTER_COMPUTE if implementation == "register" else _ROUNDTRIP_COMPUTE,
            ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False, math_approx_mode=False
            ),
            cr,
            [],
        ),
    ]
    cbs = [
        ttnn.CBDescriptor(
            total_size=4096,
            core_ranges=ranges,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=2048)],
        )
        for i in ((0, 1, 2) if implementation == "register" else (0, 1, 2, 3))
    ]
    ttnn.generic_op([packed, output], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return output
