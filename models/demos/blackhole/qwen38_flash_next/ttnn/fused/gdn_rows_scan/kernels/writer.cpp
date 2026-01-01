// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Per (value head, state column block) item: copies row r of each o tile into the head's assembled o-rows tiles
// (zeroed first, so rows >= `rows` stay exactly zero), drains the state after r + 1 rows to prefix slot r, then writes
// the gated-norm output. With block_tiles = 1 the four cores of a head exchange o-rows tiles at the same L1 address on
// every core, synchronized by a semaphore, so every core runs the gated norm on identical tiles and writes its own
// column. Compile-time args: 0 rows, 1 block_tiles (the builder's vbt); then TensorAccessorArgs of prefix
// ([rows, 12, 128, 128] fp32) and out ([1, 1, 32, 1536] bf16).  Runtime args: 0 prefix, 1 out addresses, 2 items, then
// per item (head, block) and the peers' (noc x, noc y).  Semaphore 0: the exchange counter.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/semaphore.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "../../kernels/row_mask.h"

namespace {
constexpr uint32_t cb_outs = 25, cb_outg = 26, cb_orows = 29, cb_obf = 31;
constexpr uint32_t rows = get_compile_time_arg_val(0);
constexpr uint32_t block_tiles = get_compile_time_arg_val(1);
constexpr uint32_t value_heads = 12, head_tiles = 4, block_state_tiles = head_tiles * block_tiles;
constexpr uint32_t head_state_tiles = head_tiles * head_tiles;
constexpr uint32_t peers = head_tiles / block_tiles - 1;  // 0 for the whole-head item, 3 for one column block of four
constexpr uint32_t bf16_tile_bytes = 2048, fp32_tile_bytes = 4096;
constexpr uint32_t exchange_semaphore_id = 0;
static_assert(block_tiles == head_tiles || block_tiles == 1, "a whole head or one of four column blocks");

// The state after row + 1 rows -> prefix slot row: the item's `block_state_tiles` tiles from cb_outs.
template <typename Accessor>
FORCE_INLINE void drain_state(const Accessor& prefix, uint32_t row, uint32_t head, uint32_t block) {
    const Noc noc;
    CircularBuffer(cb_outs).wait_front(block_state_tiles);
    for (uint32_t kt = 0; kt < head_tiles; ++kt) {
        for (uint32_t c = 0; c < block_tiles; ++c) {
            const uint32_t page =
                (row * value_heads + head) * head_state_tiles + kt * head_tiles + block * block_tiles + c;
            noc.async_write(
                CircularBuffer(cb_outs),
                prefix,
                prefix.get_aligned_page_size(),
                {.offset_bytes = (kt * block_tiles + c) * fp32_tile_bytes},
                {.page_id = page});
        }
    }
    noc.async_write_barrier();
    CircularBuffer(cb_outs).pop_front(block_state_tiles);
}
}  // namespace

void kernel_main() {
    constexpr auto prefix_args = TensorAccessorArgs<2>();
    constexpr auto out_args = TensorAccessorArgs<prefix_args.next_compile_time_args_offset()>();
    uint32_t arg = 0;
    const uint32_t prefix_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t out_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t items = get_arg_val<uint32_t>(arg++);
    const auto prefix = TensorAccessor(prefix_args, prefix_addr);
    const auto out = TensorAccessor(out_args, out_addr);
    const Noc noc;
    Semaphore<> exchange(exchange_semaphore_id);

    for (uint32_t item = 0; item < items; ++item) {
        const uint32_t head = get_arg_val<uint32_t>(arg++);
        const uint32_t block = get_arg_val<uint32_t>(arg++);
        uint32_t peer_x[head_tiles], peer_y[head_tiles];
        for (uint32_t p = 0; p < peers; ++p) {
            peer_x[p] = get_arg_val<uint32_t>(arg++);
            peer_y[p] = get_arg_val<uint32_t>(arg++);
        }

        CircularBuffer(cb_orows).reserve_back(head_tiles);
        // the item's own column tiles; the peers fill the others
        const uint32_t own_tiles = CircularBuffer(cb_orows).get_write_ptr() + block * block_tiles * bf16_tile_bytes;
        fused_rows::fill_words(own_tiles, block_tiles * bf16_tile_bytes / 4, 0);

        for (uint32_t r = 0; r < rows; ++r) {
            CircularBuffer(cb_obf).wait_front(block_tiles);
            const uint32_t o_tiles = CircularBuffer(cb_obf).get_read_ptr();
            for (uint32_t j = 0; j < block_tiles; ++j) {
                fused_rows::copy_row_bf16(own_tiles + j * bf16_tile_bytes, o_tiles + j * bf16_tile_bytes, r);
            }
            CircularBuffer(cb_obf).pop_front(block_tiles);
            if (r + 1 < rows) {
                drain_state(prefix, r, head, block);
            }
        }
        if constexpr (peers > 0) {
            for (uint32_t p = 0; p < peers; ++p) {
                noc.async_write(
                    CoreLocalMem<uint32_t>(own_tiles),
                    UnicastEndpoint{},
                    block_tiles * bf16_tile_bytes,
                    {},
                    {.noc_x = peer_x[p], .noc_y = peer_y[p], .addr = own_tiles});
            }
            noc.async_write_barrier();
            for (uint32_t p = 0; p < peers; ++p) {
                exchange.up(noc, peer_x[p], peer_y[p], 1);
            }
            exchange.wait_min(peers);
            exchange.set(0);
        }
        // the gated norm may start; the last prefix state drains meanwhile
        CircularBuffer(cb_orows).push_back(head_tiles);
        drain_state(prefix, rows - 1, head, block);
        {
            CircularBuffer(cb_outg).wait_front(head_tiles);
            for (uint32_t c = 0; c < block_tiles; ++c) {
                const uint32_t col = block * block_tiles + c;
                noc.async_write(
                    CircularBuffer(cb_outg),
                    out,
                    out.get_aligned_page_size(),
                    {.offset_bytes = col * bf16_tile_bytes},
                    {.page_id = head * head_tiles + col});
            }
            noc.async_write_barrier();
            CircularBuffer(cb_outg).pop_front(head_tiles);
        }
    }
}
