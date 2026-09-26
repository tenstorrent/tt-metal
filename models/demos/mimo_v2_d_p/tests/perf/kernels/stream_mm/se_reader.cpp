// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Streamed-expert weight reader (BRISC, NOC0): reads this reader's region of its DRAM bank chunk by chunk (sizes from
// se_schedule.hpp) into fixed-size CB slots, BATCH chunks per read barrier. Only the chunk's real bytes are read; a
// slot is SLOT_TILES pages so the CB never wraps inside a chunk.
//
// CT: 0 CB, 1 TILE_BYTES, 2 SLOT_TILES, 3 BATCH, 4 NK_GU, 5 NK_D, 6 PASSES, 7 NUM_EXPERTS, 8 GU_CHUNK_TILES,
//     9 PIPELINED (chunk order, see se_schedule.hpp)
// RT: 0 bank base, 1 bank id, 2 region offset, 3.. down chunk tiles of each pass
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "se_schedule.hpp"
#ifdef SE_DYN
#include "se_dyn.hpp"
#endif
#ifdef SE_ZONES
#include "tools/profiler/kernel_profiler.hpp"
#define SE_MARK(name)            \
    {                            \
        DeviceZoneScopedN(name); \
    }
#else
#define SE_MARK(name)
#endif

void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t slot_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t batch = get_compile_time_arg_val(3);
    constexpr uint32_t nk_gu = get_compile_time_arg_val(4);
    constexpr uint32_t nk_d = get_compile_time_arg_val(5);
    constexpr uint32_t passes = get_compile_time_arg_val(6);
    constexpr uint32_t num_experts = get_compile_time_arg_val(7);
    constexpr uint32_t gu_chunk_tiles = get_compile_time_arg_val(8);
    constexpr bool pipelined = get_compile_time_arg_val(9) != 0;
    constexpr uint32_t slot_bytes = slot_tiles * tile_bytes;

    const uint32_t bank_base = get_arg_val<uint32_t>(0);
    const uint32_t bank_id = get_arg_val<uint32_t>(1);
    const uint32_t bank_offset = get_arg_val<uint32_t>(2);
    uint64_t src = get_noc_addr_from_bank_id<true>(bank_id, bank_base + bank_offset);

    uint32_t in_batch = 0, l1 = 0, nmark = 0;
#ifdef SE_DYN
    // Dynamic counts (gate/up only, plain order): only the active experts' chunks, each at its expert's place in
    // the region. RT 4.. are the se_dyn.hpp args; CB 7 is this RISC's scratch.
    static_assert(nk_d == 0 && passes == 1 && !pipelined);
    SeDyn d;
    se_dyn_load<num_experts>(d, 4, get_write_ptr(tt::CBIndex::c_7), 1);
    const uint32_t chunk_bytes = gu_chunk_tiles * tile_bytes;
    const uint64_t base = src;
    for (uint32_t a = 0; a < d.n_act; ++a) {
        for (uint32_t c = 0; c < nk_gu; ++c) {
            if (in_batch == 0) {
                cb_reserve_back(cb, slot_tiles * batch);
                l1 = get_write_ptr(cb);
            }
            noc_async_read(base + (d.eid[a] * nk_gu + c) * chunk_bytes, l1 + in_batch * slot_bytes, chunk_bytes);
            if (++in_batch == batch) {
                noc_async_read_barrier();
                cb_push_back(cb, slot_tiles * batch);
                in_batch = 0;
            }
        }
    }
    if (false)
#endif
        for_each_chunk<nk_gu, nk_d, passes, num_experts, pipelined>([&](bool is_down, uint32_t p) {
            if (in_batch == 0) {
                cb_reserve_back(cb, slot_tiles * batch);
                l1 = get_write_ptr(cb);
            }
            const uint32_t bytes = (is_down ? get_arg_val<uint32_t>(3 + p) : gu_chunk_tiles) * tile_bytes;
            noc_async_read(src, l1 + in_batch * slot_bytes, bytes);
            src += bytes;
            if (++nmark % 8 == 0) {
                SE_MARK("W_RD");
            }
            if (++in_batch == batch) {
                noc_async_read_barrier();
                cb_push_back(cb, slot_tiles * batch);
                in_batch = 0;
            }
        });
    if (in_batch) {
        noc_async_read_barrier();
        cb_push_back(cb, slot_tiles * batch);  // the forwarder only consumes the chunks that exist
    }
}
