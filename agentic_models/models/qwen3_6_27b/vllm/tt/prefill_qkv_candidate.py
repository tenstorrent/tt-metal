# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Direct compact QKVG prefill movement; no arithmetic or default wiring.

Input[1,B*S,3584] -> Q[B,6,S,256], K/V[B,1,S,256], gate[1,B*S,1536].
Every output tile has one owner. Public users need no independently padded rows.
"""


def geometry(batch, seq):
    if type(batch) is not int or not 1 <= batch <= 32 or type(seq) is not int or seq < 1:
        raise ValueError("Require public batch1..32 and positive logical sequence")
    mt, flat_mt = (seq + 31) // 32, (batch * seq + 31) // 32
    counts = (batch * 6 * mt * 8, batch * mt * 8, batch * mt * 8, flat_mt * 48)
    if sum(counts) > 0xFFFFFFFF or flat_mt * 112 > 0xFFFFFFFF:
        raise ValueError("Tile indexing exceeds uint32")
    return mt, flat_mt, counts


def partition(batch, seq, cores=110):
    _, _, counts = geometry(batch, seq)
    if type(cores) is not int or cores < 1:
        raise ValueError("Positive integer workers required")
    total = sum(counts)
    workers = min(total, cores)
    return [(total * i // workers, total * (i + 1) // workers - total * i // workers) for i in range(workers)]


def work_item(batch, seq, work):
    """Return output ordinal, output tile, source first logical row/column, live rows."""
    mt, _, counts = geometry(batch, seq)
    if type(work) is not int or not 0 <= work < sum(counts):
        raise ValueError("Work index out of bounds")
    output = 0
    while work >= counts[output]:
        work -= counts[output]
        output += 1
    if output == 3:
        first, column = work // 48 * 32, 64 + work % 48
        return output, work, first, column, min(32, batch * seq - first)
    head_rows, column = divmod(work, 8)
    owner_head, tile_row = divmod(head_rows, mt)
    user, head = divmod(owner_head, 6 if output == 0 else 1)
    first = user * seq + tile_row * 32
    column += head * 8 if output == 0 else 48 if output == 1 else 56
    return output, work, first, column, min(32, seq - tile_row * 32)


KERNEL = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
struct Item {uint32_t output,page,first,col,live;};
inline Item item(uint32_t work,uint32_t B,uint32_t S,uint32_t MT) {
    const uint32_t qt=B*6*MT*8,kt=B*MT*8;
    uint32_t output=0,page=work;
    if(work>=qt) {page-=qt;output=1;}
    if(output==1 && page>=kt) {page-=kt;output=2;}
    if(output==2 && page>=kt) {page-=kt;output=3;}
    if(output==3) {
        const uint32_t first=(page/48)*32,live=B*S-first<32?B*S-first:32;
        return {output,page,first,64+page%48,live};
    }
    const uint32_t width=page%8,headrow=page/8,tr=headrow%MT,owner=headrow/MT;
    const uint32_t user=output==0?owner/6:owner,head=output==0?owner%6:0;
    const uint32_t first=user*S+tr*32,live=S-tr*32<32?S-tr*32:32;
    return {output,page,first,width+(output==0?head*8:output==1?48:56),live};
}
inline uint32_t ti(uint32_t row,uint32_t col) {return (row/16)*512+(col/16)*256+(row%16)*16+col%16;}
void kernel_main() {
    constexpr auto ia=TensorAccessorArgs<0>();
    constexpr auto qa=TensorAccessorArgs<ia.next_compile_time_args_offset()>();
    constexpr auto ka=TensorAccessorArgs<qa.next_compile_time_args_offset()>();
    constexpr auto va=TensorAccessorArgs<ka.next_compile_time_args_offset()>();
    constexpr auto ga=TensorAccessorArgs<va.next_compile_time_args_offset()>();
    const auto input=TensorAccessor(ia,get_arg_val<uint32_t>(0),2048);
    const auto q=TensorAccessor(qa,get_arg_val<uint32_t>(1),2048);
    const auto k=TensorAccessor(ka,get_arg_val<uint32_t>(2),2048);
    const auto v=TensorAccessor(va,get_arg_val<uint32_t>(3),2048);
    const auto gate=TensorAccessor(ga,get_arg_val<uint32_t>(4),2048);
    const uint32_t B=get_arg_val<uint32_t>(5),S=get_arg_val<uint32_t>(6),MT=(S+31)/32;
    const uint32_t start=get_arg_val<uint32_t>(7),count=get_arg_val<uint32_t>(8);
    Noc noc;CircularBuffer scratch(0);scratch.reserve_back(257);
    const uint32_t base=(scratch.get_write_ptr()+63u)&~63u;
    const auto write=[&](Item it,uint32_t address) {
        if(it.output==0)noc.async_write(CoreLocalMem<uint32_t>(address),q,2048,{}, {.page_id=it.page});
        else if(it.output==1)noc.async_write(CoreLocalMem<uint32_t>(address),k,2048,{}, {.page_id=it.page});
        else if(it.output==2)noc.async_write(CoreLocalMem<uint32_t>(address),v,2048,{}, {.page_id=it.page});
        else noc.async_write(CoreLocalMem<uint32_t>(address),gate,2048,{}, {.page_id=it.page});
    };
    if(S%32==0) {
        // Every user and gate tile is aligned: native-style eight tiles/barrier.
        for(uint32_t first=start;first<start+count;first+=8) {
            const uint32_t n=start+count-first<8?start+count-first:8;
            for(uint32_t j=0;j<n;j++) {
                const Item it=item(first+j,B,S,MT);
                noc.async_read(input,CoreLocalMem<uint32_t>(base+j*2048),2048,
                    {.page_id=it.first/32*112+it.col},{});
            }
            noc.async_read_barrier();
            for(uint32_t j=0;j<n;j++)write(item(first+j,B,S,MT),base+j*2048);
            noc.async_write_barrier();
        }
    } else for(uint32_t work=start;work<start+count;work++) {
        const Item it=item(work,B,S,MT);
        const uint32_t phase=it.first%32,first_page=it.first/32*112+it.col;
        noc.async_read(input,CoreLocalMem<uint32_t>(base),2048,{.page_id=first_page},{});
        if(phase+it.live>32)noc.async_read(input,CoreLocalMem<uint32_t>(base+2048),2048,
            {.page_id=first_page+112},{});
        noc.async_read_barrier();
        if(phase==0 && it.live==32)write(it,base);
        else {
            volatile tt_l1_ptr uint16_t* src=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base);
            volatile tt_l1_ptr uint16_t* dst=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base+4096);
            for(uint32_t row=0;row<32;row++)for(uint32_t col=0;col<32;col++) {
                const uint32_t logical=phase+row;
                dst[ti(row,col)]=row<it.live?src[(logical/32)*1024+ti(logical%32,col)]:0;
            }
            write(it,base+4096);
        }
        noc.async_write_barrier();
    }
}
"""


def prefill_qkv(packed, *, batch, seq, outputs=None, cores=110):
    """Move BF16 DRAM packed rows directly into owning final head/gate tensors."""
    import ttnn

    spans = partition(batch, seq, cores)
    mt, flat_mt, _ = geometry(batch, seq)
    if tuple(packed.shape) != (1, batch * seq, 3584) or tuple(packed.padded_shape) != (1, flat_mt * 32, 3584):
        raise ValueError("Require actual compact packed[1,B*S,3584] with tile-only flat-row padding")
    if (
        packed.dtype != ttnn.bfloat16
        or packed.layout != ttnn.TILE_LAYOUT
        or packed.memory_config() != ttnn.DRAM_MEMORY_CONFIG
    ):
        raise ValueError("Packed prefill input must be BF16 TILE DRAM")
    device = packed.device()
    grid = device.compute_with_storage_grid_size()
    if len(spans) > grid.x * grid.y:
        raise ValueError("Worker count exceeds device grid")
    shapes = ((batch, 6, seq, 256), (batch, 1, seq, 256), (batch, 1, seq, 256), (1, batch * seq, 1536))
    padded = ((batch, 6, mt * 32, 256), (batch, 1, mt * 32, 256), (batch, 1, mt * 32, 256), (1, flat_mt * 32, 1536))
    if outputs is None:
        outputs = tuple(
            ttnn.empty(
                shape,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for shape in shapes
        )
    if len(outputs) != 4:
        raise ValueError("Require four independently owning outputs")
    for tensor, shape, pad in zip(outputs, shapes, padded):
        if tuple(tensor.shape) != shape or tuple(tensor.padded_shape) != pad:
            raise ValueError("Output logical or physical shape mismatch")
        if (
            tensor.dtype != ttnn.bfloat16
            or tensor.layout != ttnn.TILE_LAYOUT
            or tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG
        ):
            raise ValueError("Outputs must be BF16 TILE DRAM")
        if tensor.device() != device:
            raise ValueError("All outputs must belong to the input mesh")
    addresses = [t.buffer_address() for t in (packed, *outputs)]
    if len(addresses) != len(set(addresses)):
        raise ValueError("Input/output aliasing unsupported")
    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(len(spans))]
    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    runtime = ttnn.RuntimeArgs()
    for coord, (start, count) in zip(coords, spans):
        runtime[coord.x][coord.y] = addresses + [batch, seq, start, count]
    ct = [arg for tensor in (packed, *outputs) for arg in ttnn.TensorAccessorArgs(tensor).get_compile_time_args()]
    kernel = ttnn.KernelDescriptor(
        kernel_source=KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=ranges,
        compile_time_args=ct,
        runtime_args=runtime,
        config=ttnn.ReaderConfigDescriptor(),
    )
    scratch = ttnn.CBDescriptor(
        total_size=16448,
        core_ranges=ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=64)],
    )
    ttnn.generic_op([packed, *outputs], ttnn.ProgramDescriptor(kernels=[kernel], cbs=[scratch], semaphores=[]))
    return tuple(outputs)
