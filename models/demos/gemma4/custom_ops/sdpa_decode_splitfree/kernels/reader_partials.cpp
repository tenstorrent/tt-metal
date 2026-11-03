// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Phase 1 reader of sdpa_decode_splitfree. For every unit of this core (splitfree_common.hpp: for_each_unit):
// the row's page-table row, the row's Q tile row, then per live chunk the transposed K chunk and the V chunk
// of the unit's KV group through the double-buffered K / V rings. cur_pos is read once into the compute's CB.
//
// compile-time args: the sf:: prefix (24) + TensorAccessorArgs(q, k, v, cur_pos, page_table)
// runtime args: [core, q_addr, k_addr, v_addr, pos_addr, page_table_addr]
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "splitfree_common.hpp"
#include "splitfree_dataflow.hpp"

void kernel_main() {
    using namespace sf;
    Noc noc;
    const uint32_t core = get_arg_val<uint32_t>(0);
    const uint32_t q_addr = get_arg_val<uint32_t>(1);
    const uint32_t k_addr = get_arg_val<uint32_t>(2);
    const uint32_t v_addr = get_arg_val<uint32_t>(3);
    const uint32_t pos_addr = get_arg_val<uint32_t>(4);
    const uint32_t pt_addr = get_arg_val<uint32_t>(5);

    constexpr auto q_args = TensorAccessorArgs<CT_ACCESSORS>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto pos_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr auto pt_args = TensorAccessorArgs<pos_args.next_compile_time_args_offset()>();

    constexpr uint32_t q_tile_bytes = get_tile_size(CB_Q);
    constexpr uint32_t k_tile_bytes = get_tile_size(CB_K);
    constexpr uint32_t v_tile_bytes = get_tile_size(CB_V);
    const auto q_reader = TensorAccessor(q_args, q_addr, q_tile_bytes);
    const auto k_reader = TensorAccessor(k_args, k_addr, k_tile_bytes);
    const auto v_reader = TensorAccessor(v_args, v_addr, v_tile_bytes);
    const auto pt_reader = TensorAccessor(pt_args, pt_addr, PT_STICK);

    // cur_pos: once, for the compute (pushed) and for this kernel's walk (a local copy)
    volatile tt_l1_ptr uint32_t* pos_l1 = read_cur_pos<CB_POS_C, POS_STICK>(noc, pos_args, pos_addr);
    uint32_t pos[B];
    for (uint32_t b = 0; b < B; ++b) {
        pos[b] = pos_l1[b];
    }
    CircularBuffer(CB_POS_C).push_back(1);

    // the page-table row of the current unit's row: one reserved page, overwritten per unit, never pushed
    CircularBuffer cb_pt(CB_PT);
    cb_pt.reserve_back(1);
    const uint32_t pt_l1 = cb_pt.get_write_ptr();
    volatile tt_l1_ptr uint32_t* page_table = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pt_l1);

    CircularBuffer cb_q(CB_Q);
    for_each_unit(core, pos, [&](uint32_t b, uint32_t g, uint32_t /*bg*/, uint32_t u, const Row& r) {
        noc.async_read(pt_reader, CoreLocalMem<uint32_t>(pt_l1), PT_STICK, {.page_id = b}, {});
        noc.async_read_barrier();

        cb_q.reserve_back(DHT);
        uint32_t q_ptr = cb_q.get_write_ptr();
        for (uint32_t c = 0; c < DHT; ++c) {
            noc.async_read(q_reader, CoreLocalMem<uint32_t>(q_ptr), q_tile_bytes, {.page_id = b * DHT + c}, {});
            q_ptr += q_tile_bytes;
        }
        noc.async_read_barrier();
        cb_q.push_back(DHT);

        const uint32_t j_lo = chunk_lo(r, u);
        const uint32_t j_hi = chunk_hi(r, u);
        for (uint32_t j = j_lo; j <= j_hi; ++j) {
            const uint32_t row0 = j * SK;  // the chunk's first tile row of the virtual sequence
            read_k_chunk<CB_K, DHT, SK, NKV, BLOCK_T, CAP_T, READS>(noc, k_reader, g, row0, page_table);
            read_v_chunk<CB_V, DHT, SK, NKV, BLOCK_T, CAP_T, READS>(noc, v_reader, g, row0, page_table);
        }
    });
}
