# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Opt-in SDPA head concatenation and native BF16 sigmoid gate.

Consumes native logical [1,B,6,256], physical [1,B,32,256] SDPA DRAM directly. No restore to
height-sharded heads or material concatenated tensor is required. Device
equivalence and complete-boundary timing are required before integration.
"""

from .packed_swiglu_candidate import _WRITER

READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto ha=TensorAccessorArgs<0>();
    constexpr auto ga=TensorAccessorArgs<ha.next_compile_time_args_offset()>();
    const auto heads=TensorAccessor(ha,get_arg_val<uint32_t>(0),2048);
    const auto gates=TensorAccessor(ga,get_arg_val<uint32_t>(1),2048);
    const uint32_t B=get_arg_val<uint32_t>(2),start=get_arg_val<uint32_t>(3),count=get_arg_val<uint32_t>(4);
    Noc noc; CircularBuffer gate(0),out(1),scratch(3);
    scratch.reserve_back(65);
    const uint32_t tmp=(scratch.get_write_ptr()+63u)&~63u;
    for(uint32_t tile=start;tile<start+count;tile++) {
        gate.reserve_back(1);out.reserve_back(1);
        const uint32_t head=tile/8,col=tile%8;
        // Source rows may start at offset32 mod64. Read an aligned two-row
        // pair into a private slot, then select the live row with integer copies.
        for(uint32_t user=0;user<B;user++) for(uint32_t face=0;face<2;face++) {
            const uint32_t src=(head/16)*1024+face*512+(head%16/2)*64;
            noc.async_read(heads,CoreLocalMem<uint32_t>(tmp+(user*2+face)*64),64,
                {.page_id=user*8+col,.offset_bytes=src},{});
        }
        noc.async_read(gates,gate,2048,{.page_id=tile},{.offset_bytes=0});
        noc.async_read_barrier();
        volatile tt_l1_ptr uint16_t* dst=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(out.get_write_ptr());
        volatile tt_l1_ptr uint16_t* gp=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(gate.get_write_ptr());
        volatile tt_l1_ptr uint16_t* src=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tmp);
        for(uint32_t user=0;user<32;user++) for(uint32_t face=0;face<2;face++) for(uint32_t j=0;j<16;j++) {
            const uint32_t d=(user/16)*512+face*256+(user%16)*16+j;
            dst[d]=user<B?src[(user*2+face)*32+(head%2)*16+j]:0;
            if(user>=B) gp[d]=0;
        }
        gate.push_back(1);out.push_back(1);
    }
}
"""

COMPUTE = r"""
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/dataflow/circular_buffer.h"
void kernel_main() {
    const uint32_t count=get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(0,2);
    CircularBuffer gate(0),heads(1),out(2);
    for(uint32_t tile=0;tile<count;tile++) {
        gate.wait_front(1);heads.wait_front(1);out.reserve_back(1);
        tile_regs_acquire();
        copy_init(0);copy_tile(0,0,0);copy_init(1);copy_tile(1,0,1);
        // Blackhole's false specialization rounds sigmoid to BF16 in DST.
        // Native binary's intervening BF16 pack/reload is therefore exact.
        sigmoid_tile_init<false>();sigmoid_tile<VectorMode::RC,false>(0);
        mul_binary_tile_init();mul_binary_tile<false>(0,1,0);
        tile_regs_commit();tile_regs_wait();pack_tile(0,2);tile_regs_release();
        gate.pop_front(1);heads.pop_front(1);out.push_back(1);
    }
}
"""


def band_sources():
    """Split each output tile between disjoint 16-user row bands.

    Both workers retain ordinary BF16 tile arithmetic. Each writer owns one
    contiguous 1024-byte half of the destination tile, including padded zeros.
    This reduces the serial NoC fragment requests per worker at large batches.
    """
    reader = READER
    replacements = (
        (
            "for(uint32_t tile=start;tile<start+count;tile++) {",
            "for(uint32_t unit=start;unit<start+count;unit++) {\n" "        const uint32_t tile=unit/2,band=unit%2;",
        ),
        (
            "for(uint32_t user=0;user<B;user++) for(uint32_t face=0;face<2;face++) {",
            "for(uint32_t user=band*16;user<B && user<(band+1)*16;user++) for(uint32_t face=0;face<2;face++) {",
        ),
        ("dst[d]=user<B?src[", "dst[d]=(user>=band*16 && user<(band+1)*16 && user<B)?src["),
        ("if(user>=B) gp[d]=0;", "if(user<band*16 || user>=(band+1)*16 || user>=B) gp[d]=0;"),
    )
    for before, after in replacements:
        assert reader.count(before) == 1
        reader = reader.replace(before, after, 1)
    writer = _WRITER
    before = """        noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(out),output,2048,
                        {.offset_bytes=0},{.page_id=t});"""
    assert writer.count(before) == 1
    writer = writer.replace(
        before,
        """        const uint32_t tile=t/2,offset=(t%2)*1024;
        noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(out),output,1024,
                        {.offset_bytes=offset},{.page_id=tile,.offset_bytes=offset});""",
        1,
    )
    return reader, writer


def tile_plan(batch, cores=48, *, row_bands=1):
    if type(batch) is not int or not 1 <= batch <= 32 or type(cores) is not int or cores < 1:
        raise ValueError("Require logical batch1..32 and positive worker count")
    if row_bands not in (1, 2):
        raise ValueError("Require one full tile or two disjoint row bands")
    units = 48 * row_bands
    n = min(units, cores)
    spans, start = [], 0
    for i in range(n):
        count = units // n + (i < units % n)
        spans.append((start, count))
        start += count
    return spans


def face_address(user, head, column_tile, face):
    """CPU-checkable physical tile, aligned byte offset and selected-row skew."""
    if not (0 <= user < 32 and 0 <= head < 6 and 0 <= column_tile < 8 and face in (0, 1)):
        raise ValueError("Invalid live SDPA face coordinate")
    return user * 8 + column_tile, (head // 16) * 1024 + face * 512 + (head % 16 // 2) * 64, (head % 2) * 32


def canonical_output_memory(device):
    """The existing eight-storage-core o_proj input contract, independent of workers."""
    import ttnn

    workers = ttnn.num_cores_to_corerangeset(8, device.compute_with_storage_grid_size(), row_wise=True)
    return ttnn.create_sharded_memory_config(
        (32, 192), workers, ttnn.ShardStrategy.WIDTH, use_height_and_width_as_shard_shape=True
    )


def attention_gate(heads, gate, *, output=None, memory_config=None, cores=48, row_bands=1):
    import ttnn

    if len(gate.shape) != 3:
        raise ValueError("Expected compact gate [1,B,1536]")
    batch = gate.shape[1]
    spans = tile_plan(batch, cores, row_bands=row_bands)
    if tuple(heads.shape) not in ((1, batch, 6, 256), (1, batch, 32, 256)) or tuple(gate.shape) != (1, batch, 1536):
        raise ValueError("Expected native SDPA heads [1,B,6,256] (or physical32 view) and gate [1,B,1536]")
    if tuple(heads.padded_shape) != (1, batch, 32, 256) or tuple(gate.padded_shape) != (1, 32, 1536):
        raise ValueError("Unsupported physical padding")
    for t in (heads, gate):
        if t.dtype != ttnn.bfloat16 or t.layout != ttnn.TILE_LAYOUT or t.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
            raise ValueError("Inputs must be BF16 TILE DRAM")
    device = heads.device()
    if gate.device() != device or device.arch() != ttnn.device.Arch.BLACKHOLE:
        raise ValueError("Require one Blackhole device/mesh")
    destination = memory_config or (output.memory_config() if output is not None else ttnn.DRAM_MEMORY_CONFIG)
    if destination != ttnn.DRAM_MEMORY_CONFIG and destination != canonical_output_memory(device):
        raise ValueError("Output must be DRAM or canonical eight-core width-sharded L1 [32,192]")
    if output is None:
        output = ttnn.empty(
            (1, batch, 1536), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=destination
        )
    if tuple(output.shape) != tuple(gate.shape) or tuple(output.padded_shape) != tuple(gate.padded_shape):
        raise ValueError("Output shape mismatch")
    if (
        output.dtype != ttnn.bfloat16
        or output.layout != ttnn.TILE_LAYOUT
        or output.device() != device
        or output.memory_config() != destination
    ):
        raise ValueError("Output placement mismatch")
    if output.buffer_address() in (heads.buffer_address(), gate.buffer_address()):
        raise ValueError("Input/output aliasing is unsupported")
    grid = device.compute_with_storage_grid_size()
    if cores > grid.x * grid.y:
        raise ValueError("Worker count exceeds grid")
    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(len(spans))]
    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    rr, wr, cr = (ttnn.RuntimeArgs() for _ in range(3))
    for c, (start, count) in zip(coords, spans):
        rr[c.x][c.y] = [heads.buffer_address(), gate.buffer_address(), batch, start, count]
        wr[c.x][c.y] = [output.buffer_address(), start, count]
        cr[c.x][c.y] = [count]

    def kernel(source, config, rt, ct):
        return ttnn.KernelDescriptor(
            kernel_source=source,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=ranges,
            compile_time_args=ct,
            runtime_args=rt,
            config=config,
        )

    accessors = (
        ttnn.TensorAccessorArgs(heads).get_compile_time_args() + ttnn.TensorAccessorArgs(gate).get_compile_time_args()
    )
    reader, writer = (READER, _WRITER) if row_bands == 1 else band_sources()
    kernels = [
        kernel(reader, ttnn.ReaderConfigDescriptor(), rr, accessors),
        kernel(writer, ttnn.WriterConfigDescriptor(), wr, ttnn.TensorAccessorArgs(output).get_compile_time_args()),
        kernel(
            COMPUTE,
            ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False, math_approx_mode=False
            ),
            cr,
            [],
        ),
    ]
    cbs = [
        ttnn.CBDescriptor(
            total_size=4160 if i == 3 else 4096,
            core_ranges=ranges,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=64 if i == 3 else 2048)
            ],
        )
        for i in range(4)
    ]
    ttnn.generic_op([heads, gate, output], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return output
