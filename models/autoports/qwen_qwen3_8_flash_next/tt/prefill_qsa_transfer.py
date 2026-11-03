# SPDX-License-Identifier: Apache-2.0
"""Multi-token (prefill) QSA tile-row transfers, modes 0/1/2 of qsa_gather_bounded.cpp.

The bounded kernel moves one row per serialized step: mode 0/2 read the source tile and the destination
tile, patch one row and write the whole destination tile back, for every token, on B*H*Wt workers
(4 for a serial prefill row); mode 1 builds each query's 4-row group tile with its own full page-table
read and four tile reads. Here a mode 0/2 job is one destination tile (read once, every row of the chunk
patched, written once) and a mode 1 job is one source tile (read once; the up to 8 group tiles it holds
are built once and written to every query of the group behind one barrier). Page addressing, row
placement and bytes are unchanged. Toggle: QWEN_PREFILL_QSA_TRANSFER=0 restores the bounded kernel (default on).
Also the prefill score mask (prefill_mask_scores): -inf only below the top-k search bound.
Toggle: QWEN_PREFILL_QSA_MASK=0 restores OptimizedQSA._mask_scores (default on).
"""

import os

import ttnn


def prefill_transfer_enabled():
    return os.environ.get("QWEN_PREFILL_QSA_TRANSFER", "1") != "0"


SOURCE = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

static uint32_t face_index(uint32_t row, uint32_t col) {
    return (row / 16) * 512 + (col / 16) * 256 + (row % 16) * 16 + col % 16;
}

void kernel_main() {
    constexpr uint32_t mode = get_compile_time_arg_val(0), B = get_compile_time_arg_val(1);
    constexpr uint32_t S = get_compile_time_arg_val(2), H = get_compile_time_arg_val(3);
    constexpr uint32_t Wt = get_compile_time_arg_val(4), page_size = get_compile_time_arg_val(5);
    constexpr uint32_t capacity = get_compile_time_arg_val(6), physical_pages = get_compile_time_arg_val(7);
    constexpr uint32_t ND = get_compile_time_arg_val(8);
    constexpr uint32_t St = (S + 31) / 32, page_tiles = page_size / 32, table_bytes = capacity / page_size * 4;
    constexpr uint32_t span = mode == 2 ? 128 : 32;
    static_assert(mode != 1 || H == 1, "mode 1 gathers the single raw-key head");
    constexpr auto aa = TensorAccessorArgs<9>();
    constexpr auto oa = TensorAccessorArgs<aa.next_compile_time_args_offset()>();
    constexpr auto pa = TensorAccessorArgs<oa.next_compile_time_args_offset()>();
    constexpr auto ta = TensorAccessorArgs<pa.next_compile_time_args_offset()>();
    const auto source = TensorAccessor(aa, get_arg_val<uint32_t>(0));
    const auto output = TensorAccessor(oa, get_arg_val<uint32_t>(1));
    const auto positions = TensorAccessor(pa, get_arg_val<uint32_t>(2));
    const auto table = TensorAccessor(ta, get_arg_val<uint32_t>(3));
    const uint32_t begin = get_arg_val<uint32_t>(4), count = get_arg_val<uint32_t>(5), stride = get_arg_val<uint32_t>(6);
    CircularBuffer cb(0);
    cb.reserve_back(1);
    // [input tile 2048][8 output tiles x 2048][positions 128][page table]
    const uint32_t base = cb.get_write_ptr(), outs = base + 2048, pos_addr = outs + 8 * 2048, table_addr = pos_addr + 128;
    auto in = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base);
    auto pos = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pos_addr);
    auto pages = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(table_addr);
    Noc noc;
    uint32_t loaded = 0xffffffff, start = 0;
    for (uint32_t item = 0; item < count; ++item) {
        const uint32_t job = begin + item * stride;
        const uint32_t col = job % Wt, d = (job / Wt) % ND, head = (job / (Wt * ND)) % H, b = job / (Wt * ND * H);
        ASSERT(b < B);
        if (b != loaded) {
            noc.async_read(positions, CoreLocalMem<uint32_t>(pos_addr), 128, {.page_id = b}, {});
            noc.async_read_barrier();
            start = pos[0];
            loaded = b;
            ASSERT(start + S <= capacity);
            const uint32_t live = ((start + S + page_size - 1) / page_size * 4 + 31) / 32 * 32;
            noc.async_read(table, CoreLocalMem<uint32_t>(table_addr), live < table_bytes ? live : table_bytes, {.page_id = b}, {});
            noc.async_read_barrier();
        }
        const uint32_t first = start / span, last = (start + S - 1) / span;
        if (first + d > last) {
            continue;
        }
        const uint32_t lo = (first + d) * span < start ? start : (first + d) * span;
        const uint32_t hi = (first + d + 1) * span < start + S ? (first + d + 1) * span : start + S;
        if constexpr (mode == 0 || mode == 2) {
            uint32_t output_tile;
            if constexpr (mode == 0) {
                const uint32_t physical = pages[lo / page_size];
                ASSERT(physical < physical_pages);
                output_tile = ((physical * H + head) * page_tiles + lo % page_size / 32) * Wt + col;
            } else {
                output_tile = ((b * H + head) * (capacity / 4 / 32) + lo / 4 / 32) * Wt + col;
            }
            auto out = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(outs);
            bool any = mode == 0;
            if constexpr (mode == 2) {
                for (uint32_t a = lo; a < hi; ++a) {
                    any = any || a % 4 == 3;
                }
            }
            if (!any) {
                continue;
            }
            noc.async_read(output, CoreLocalMem<uint32_t>(outs), 2048, {.page_id = output_tile}, {});
            uint32_t loaded_tile = 0xffffffff;
            for (uint32_t a = lo; a < hi; ++a) {
                if constexpr (mode == 2) {
                    if (a % 4 != 3) {
                        continue;
                    }
                }
                const uint32_t s = a - start;
                if (s / 32 != loaded_tile) {
                    noc.async_read(source, CoreLocalMem<uint32_t>(base), 2048, {.page_id = ((b * H + head) * St + s / 32) * Wt + col}, {});
                    noc.async_read_barrier();
                    loaded_tile = s / 32;
                }
                const uint32_t row = mode == 0 ? a % 32 : (a / 4) % 32;
                for (uint32_t c = 0; c < 32; ++c) {
                    out[face_index(row, c)] = in[face_index(s % 32, c)];
                }
            }
            noc.async_write(CoreLocalMem<uint32_t>(outs), output, 2048, {}, {.page_id = output_tile});
            noc.async_write_barrier();
        } else {
            // Groups (start + q) / 4 * 4 of queries q in [lo - start, hi - start) lie in this source tile.
            const uint32_t tile_base = (first + d) * 32;
            const uint32_t physical = pages[tile_base / page_size];
            ASSERT(physical < physical_pages);
            noc.async_read(source, CoreLocalMem<uint32_t>(base), 2048, {.page_id = (physical * page_tiles + tile_base % page_size / 32) * Wt + col}, {});
            noc.async_read_barrier();
            for (uint32_t g = lo / 4 * 4; g < hi; g += 4) {
                const uint32_t i = (g - tile_base) / 4;
                auto words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(outs + i * 2048);
                for (uint32_t w = 0; w < 512; ++w) {
                    words[w] = 0;
                }
                auto out = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(outs + i * 2048);
                for (uint32_t r = 0; r < 4; ++r) {
                    for (uint32_t c = 0; c < 32; ++c) {
                        out[face_index(r, c)] = in[face_index((g + r) % 32, c)];
                    }
                }
            }
            for (uint32_t a = lo; a < hi; ++a) {
                const uint32_t i = (a / 4 * 4 - tile_base) / 4;
                noc.async_write(CoreLocalMem<uint32_t>(outs + i * 2048), output, 2048, {}, {.page_id = (b * S + a - start) * Wt + col});
            }
            noc.async_write_barrier();
        }
    }
}
"""


def prefill_transfer(attention, source, output, positions, page_table, *, mode, sequence, heads):
    """Same output as QSA._transfer(mode 0/1/2) on the bounded (paged pooled) cache."""
    if mode not in (0, 1, 2) or not attention.paged_pooled or (mode == 1 and heads != 1):
        raise ValueError("Prefill transfer covers bounded modes 0/1/2 only")
    width_tiles = source.shape[-1] // 32
    span = 128 if mode == 2 else 32
    nd = (sequence + span - 1) // span + 1
    work = attention.batch_size * heads * nd * width_tiles
    grid = attention.device.compute_with_storage_grid_size()
    workers = min(work, grid.x * grid.y)
    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(workers)]
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    tensors = [source, output, positions, page_table]
    args = [
        mode,
        attention.batch_size,
        sequence,
        heads,
        width_tiles,
        attention.page_size,
        attention.max_context,
        attention.pool_tokens // attention.page_size,
        nd,
    ]
    for tensor in tensors:
        args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    runtime = [
        ((c.x, c.y), [*[t.buffer_address() for t in tensors], i, (work + workers - 1 - i) // workers, workers])
        for i, c in enumerate(coords)
    ]
    scratch = 2048 + 8 * 2048 + 128 + attention.max_context // attention.page_size * 4
    cb = ttnn.CBDescriptor(
        total_size=scratch,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=scratch)],
    )
    kernel = ttnn.KernelDescriptor(
        kernel_source=SOURCE,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=cores,
        compile_time_args=args,
        runtime_args=runtime,
        config=ttnn.ReaderConfigDescriptor(),
    )
    ttnn.generic_op(tensors, ttnn.ProgramDescriptor(kernels=[kernel], cbs=[cb]))
    return output


def prefill_mask_enabled():
    return os.environ.get("QWEN_PREFILL_QSA_MASK", "1") != "0"


# OptimizedQSA._mask_scores writes -inf into every column of every row from the row's completed-block
# count to the full pooled capacity K (max context / 4), although the following
# topk_large_indices(valid_length_tensor=bound) searches only the first `bound` columns. This variant
# writes the same bound and the same -inf values, but only below the bound; columns at or beyond it are
# never read. The job order covers only the live 256-column chunks.
MASK_SOURCE = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr uint32_t B=get_compile_time_arg_val(0),S=get_compile_time_arg_val(1),K=get_compile_time_arg_val(2),W=get_compile_time_arg_val(3);
    constexpr auto sa=TensorAccessorArgs<4>();
    constexpr auto pa=TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    constexpr auto ba=TensorAccessorArgs<pa.next_compile_time_args_offset()>();
    const auto scores=TensorAccessor(sa,get_arg_val<uint32_t>(0));
    const auto positions=TensorAccessor(pa,get_arg_val<uint32_t>(1));
    const auto bound=TensorAccessor(ba,get_arg_val<uint32_t>(2));
    const uint32_t worker=get_arg_val<uint32_t>(3);
    CircularBuffer cb(0);cb.reserve_back(1);const uint32_t base=cb.get_write_ptr();Noc noc;
    auto pos=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    auto values=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base+128);
    uint32_t last=512;
    for(uint32_t b=0;b<B;++b) {
        noc.async_read(positions,CoreLocalMem<uint32_t>(base),128,{.page_id=b},{});noc.async_read_barrier();
        const uint32_t end=(pos[0]+S)/4;if(end>last)last=end;
    }
    ASSERT(last<=K);
    if(worker==0) {
        auto result=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base+640);
        for(uint32_t i=0;i<8;++i) { result[i]=0; }
        result[0]=last;
        noc.async_write(CoreLocalMem<uint32_t>(base+640),bound,32,{}, {.page_id=0});noc.async_write_barrier();
    }
    const uint32_t chunks=(last+255)/256;
    uint32_t loaded=0xffffffff,start=0;
    for(uint32_t job=worker;job<B*S*chunks;job+=W) {
        const uint32_t row=job/chunks,b=row/S,col=(job%chunks)*256;
        if(loaded!=b) {noc.async_read(positions,CoreLocalMem<uint32_t>(base),128,{.page_id=b},{});noc.async_read_barrier();start=pos[0];loaded=b;}
        const uint32_t complete=(start+row%S+1)/4;
        const uint32_t count=last-col<256?last-col:256;
        if(col+count<=complete)continue;
        uint32_t first=0;
        if(col<complete) {
            noc.async_read(scores,CoreLocalMem<uint32_t>(base+128),count*2,{.page_id=row,.offset_bytes=col*2},{});
            noc.async_read_barrier();first=complete-col;
        }
        for(uint32_t i=first;i<count;++i)values[i]=0xff80;
        noc.async_write(CoreLocalMem<uint32_t>(base+128),scores,count*2,{}, {.page_id=row,.offset_bytes=col*2});noc.async_write_barrier();
    }
}
"""


def prefill_mask_scores(attention, scores, positions, seq):
    """Bound and live-column mask of OptimizedQSA._mask_scores for a multi-token chunk."""
    bound = ttnn.empty((1,), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=attention.device)
    grid = attention.device.compute_with_storage_grid_size()
    workers = grid.x * grid.y
    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(workers)]
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    tensors = [scores, positions, bound]
    args = [attention.batch_size, seq, attention.blocks, workers]
    for t in tensors:
        args.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    kernel = ttnn.KernelDescriptor(
        kernel_source=MASK_SOURCE,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=cores,
        compile_time_args=args,
        runtime_args=[((c.x, c.y), [*[t.buffer_address() for t in tensors], i]) for i, c in enumerate(coords)],
        config=ttnn.ReaderConfigDescriptor(),
    )
    cb = ttnn.CBDescriptor(
        total_size=768,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=768)],
    )
    ttnn.generic_op(tensors, ttnn.ProgramDescriptor(kernels=[kernel], cbs=[cb]))
    return bound
