// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Hy4 B20 fused KV append, writer. Core b (of 4) writes, for every row r < S whose id is not 0xFFFFFFFF, into pool row
// ids[r] (704 B pages, SCALED_FP8 row = 512 e4m3 | 4 fp32 scales | 64 bf16 rope | pad):
//   e4m3 block b   (128 B)  at byte 128 b
//   scale of block b (4 B)  at byte 512 + 4 b
//   rope cols 16 b..16 b+15 (32 B) at byte 528 + 32 b   (from k_pe [1,1,S,64] TILE: tile b/2, face column b%2)
// CT: 0 cb_out (e4m3 tiles), 1 cb_scale (fp32 scale tile, column 0 = scale), 2 cb_stage, 3 S, 4 PAGE, then
// TensorAccessorArgs(kpe), (ids), (pool). Common: kpe addr, ids addr, pool addr. RT: b. stage (4096 B): ids page [0,64)
// | kpe tile [64, 2112) | scale staging 8 x 64 B [2112, 2624) | rope staging 8 x 32 B [2624, 2880)
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_scale = get_compile_time_arg_val(1);
    constexpr uint32_t cb_stage = get_compile_time_arg_val(2);
    constexpr uint32_t S = get_compile_time_arg_val(3);
    constexpr uint32_t PAGE = get_compile_time_arg_val(4);
    constexpr auto kpe_args = TensorAccessorArgs<5>();
    constexpr auto ids_args = TensorAccessorArgs<kpe_args.next_compile_time_args_offset()>();
    constexpr auto pool_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
    constexpr uint32_t KPE_OFF = 64, SC_OFF = 2112, RP_OFF = 2624;
    const uint32_t b = get_arg_val<uint32_t>(0);
    const auto kpe_acc = TensorAccessor(kpe_args, get_common_arg_val<uint32_t>(0), 2048);
    const auto ids_acc = TensorAccessor(ids_args, get_common_arg_val<uint32_t>(1), 64);
    const auto pool_acc = TensorAccessor(pool_args, get_common_arg_val<uint32_t>(2), PAGE);
    Noc noc;
    CircularBuffer stage(cb_stage), out(cb_out), scl(cb_scale);
    stage.reserve_back(1);
    const uint32_t base = stage.get_write_ptr();
    noc.async_read(ids_acc, stage, 64, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = 0});
    noc.async_read(kpe_acc, stage, 2048, {.page_id = b >> 1, .offset_bytes = 0}, {.offset_bytes = KPE_OFF});
    noc.async_read_barrier();
    invalidate_l1_cache();
    out.wait_front(4);
    scl.wait_front(1);
    volatile tt_l1_ptr uint32_t* ids = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    volatile tt_l1_ptr uint32_t* sct = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scl.get_read_ptr());
    volatile tt_l1_ptr uint32_t* kpe = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + KPE_OFF);
    volatile tt_l1_ptr uint32_t* sst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + SC_OFF);
    volatile tt_l1_ptr uint32_t* rst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + RP_OFF);
    for (uint32_t r = 0; r < S; ++r) {
        const uint32_t d = ids[r];
        if (d == 0xFFFFFFFFu) {
            continue;
        }
        noc.async_write(out, pool_acc, 128, {.offset_bytes = r * 128}, {.page_id = d, .offset_bytes = 128 * b});
        // scale (fp32 tile, column 0 of row r): face (r/16)*2, row r%16, 16 words per face row
        sst[r * 16 + b] = sct[(r >> 4) * 512 + (r & 15) * 16];
        noc.async_write(
            use<CircularBuffer::AddrSelector::WRITE_PTR>(stage),
            pool_acc,
            4,
            {.offset_bytes = SC_OFF + r * 64 + 4 * b},
            {.page_id = d, .offset_bytes = 512 + 4 * b});
        // rope piece: bf16 tile face (r/16)*2 + (b&1) (128 words), row r%16: 16 bf16 = 8 words
        const uint32_t fbase = (((r >> 4) << 1) + (b & 1)) * 128 + (r & 15) * 8;
        for (uint32_t w = 0; w < 8; ++w) {
            rst[r * 8 + w] = kpe[fbase + w];
        }
        noc.async_write(
            use<CircularBuffer::AddrSelector::WRITE_PTR>(stage),
            pool_acc,
            32,
            {.offset_bytes = RP_OFF + r * 32},
            {.page_id = d, .offset_bytes = 528 + 32 * b});
    }
    noc.async_write_barrier();
    out.pop_front(4);
    scl.pop_front(1);
}
