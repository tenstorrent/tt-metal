// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// block_pool sender, a kernel of the block_topk_route program: pushes this device's K/V shard of the gathered buffer
// kv_g [1, 2H, NBG, D] to ring peers over one fabric link in one direction, while the other cores are still pooling.
//
// The K/V units are the first KV_UNITS of the pool's numbering (block_pool_units.hpp); a unit is DT tiles at
// consecutive pages (head NBGT + r NBPT + g) DT of kv_g. The sender of link l takes units l, l + L, l + 2L, ..., waits
// for each unit's round (the round semaphore reaching the number of cores with a K/V unit in that round) before reading
// it from DRAM, and writes it in one scatter packet to every target, at the same pages of the target's kv_g (one mesh
// buffer: same address on every device). A target reached from both directions (the opposite position of an even
// ring) takes the units whose index (u / L) has the parity of this sender. Each target then gets one atomic increment
// of this shard's arrival slot on its gate core, after the data on the same route.
//
// Compile-time args: 0 H, 1 RT, 2 NBPT, 3 NBGT, 4 DT, 5 NCORES (the pool cores), 6 L, then TensorAccessorArgs of kv_g.
// Common runtime args: 0 kv_g address, 1 gate core noc x, 2 gate core noc y.
// Runtime args: 0 ring index r, 1 arrival slot address, 2 link l, 3 parity, 4 number of targets, then (hops, half) per
// target, then the fabric connection [has_fwd, args..., has_bwd, args...] with exactly one of the two set.

#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "block_pool_units.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"

using namespace tt::tt_fabric::linear::experimental;

namespace {
constexpr uint32_t cb_send = 6;
constexpr uint32_t TILE_BYTES = 2048;
constexpr uint32_t MAX_TARGETS = 32;
}  // namespace

void kernel_main() {
    constexpr uint32_t H = get_compile_time_arg_val(0);
    constexpr uint32_t RT = get_compile_time_arg_val(1);
    constexpr uint32_t NBPT = get_compile_time_arg_val(2);
    constexpr uint32_t NBGT = get_compile_time_arg_val(3);
    constexpr uint32_t DT = get_compile_time_arg_val(4);
    constexpr uint32_t NCORES = get_compile_time_arg_val(5);
    constexpr uint32_t L = get_compile_time_arg_val(6);
    constexpr auto kv_args = TensorAccessorArgs<7>();
    static_assert(DT >= 1 && DT <= NOC_SCATTER_WRITE_MAX_CHUNKS, "one packet holds a unit, at most 4 scattered tiles");
    using Layout = block_pool::Layout<H, RT, NBPT>;
    constexpr uint32_t KV_UNITS = Layout::KV_UNITS;

    size_t arg = 0;
    const uint32_t r = get_arg_val<uint32_t>(arg++);
    const uint32_t slot_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t l = get_arg_val<uint32_t>(arg++);
    const uint32_t parity = get_arg_val<uint32_t>(arg++);
    const uint32_t n_targets = get_arg_val<uint32_t>(arg++);
    if (n_targets == 0) {
        return;
    }
    ASSERT(n_targets <= MAX_TARGETS);
    uint8_t hops[MAX_TARGETS];
    bool half[MAX_TARGETS];
    for (uint32_t t = 0; t < n_targets; ++t) {
        hops[t] = static_cast<uint8_t>(get_arg_val<uint32_t>(arg++));
        half[t] = get_arg_val<uint32_t>(arg++) != 0;
    }
    auto fabric = FabricConnectionManager::build_from_args<
        FabricConnectionManager::BuildFromArgsMode::BUILD_AND_OPEN_CONNECTION_START_ONLY>(arg);

    const auto kv_g = TensorAccessor(kv_args, get_common_arg_val<uint32_t>(0));
    const uint64_t slot_noc_addr =
        safe_get_noc_addr(get_common_arg_val<uint32_t>(1), get_common_arg_val<uint32_t>(2), slot_addr, 0);
    auto first_page = [&](uint32_t u) {
        const block_pool::Unit un = Layout::decode(u);
        const uint32_t head = un.t == 2 ? H + un.h : un.h;
        return (head * NBGT + r * NBPT + un.g) * DT;
    };
    auto round_sem = [](uint32_t k) {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(block_pool::ROUND_SEM + k));
    };
    uint32_t rounds_done = 0;
    auto ready = [&](uint32_t u) {
        invalidate_l1_cache();
        while (rounds_done <= u / NCORES && *round_sem(rounds_done) >= Layout::kv_cores<NCORES>(rounds_done)) {
            ++rounds_done;
        }
        return rounds_done > u / NCORES;
    };
    auto wait_ready = [&](uint32_t u) {
        for (; rounds_done <= u / NCORES; ++rounds_done) {
            noc_semaphore_wait_min(round_sem(rounds_done), Layout::kv_cores<NCORES>(rounds_done));
        }
    };

    Noc noc;
    CircularBuffer cbs(cb_send);
    cbs.reserve_back(2 * DT);
    const uint32_t buf = cbs.get_write_ptr();
    auto read_unit = [&](uint32_t u, uint32_t slot) {
        const uint32_t page = first_page(u);
        for (uint32_t n = 0; n < DT; ++n) {
            noc.async_read(
                kv_g,
                CoreLocalMem<uint32_t>(buf + (slot * DT + n) * TILE_BYTES),
                TILE_BYTES,
                {.page_id = page + n},
                {});
        }
    };

    fabric.open_finish();
    auto& link = fabric.has_forward_connection() ? fabric.get_forward_connection() : fabric.get_backward_connection();
    volatile PACKET_HEADER_TYPE* hdr = PacketHeaderPool::allocate_header();

    if (l < KV_UNITS) {
        wait_ready(l);
        read_unit(l, 0);
    }
    uint32_t slot = 0;
    for (uint32_t u = l; u < KV_UNITS; u += L, slot ^= 1) {
        noc.async_read_barrier();
        const uint32_t next = u + L;
        const bool prefetched = next < KV_UNITS && ready(next);
        if (prefetched) {
            read_unit(next, slot ^ 1);
        }
        const uint32_t page = first_page(u);
        uint64_t addrs[NOC_SCATTER_WRITE_MAX_CHUNKS] = {0, 0, 0, 0};
        for (uint32_t n = 0; n < DT; ++n) {
            addrs[n] = tt::tt_fabric::linear::addrgen_detail::get_noc_address(kv_g, page + n, 0);
        }
        const uint32_t src = buf + slot * DT * TILE_BYTES;
        for (uint32_t t = 0; t < n_targets; ++t) {
            if (half[t] && (u / L) % 2 != parity) {
                continue;
            }
            if constexpr (DT == 1) {
                fabric_unicast_noc_unicast_write(
                    &link, hdr, src, TILE_BYTES, tt::tt_fabric::NocUnicastCommandHeader{addrs[0]}, hops[t]);
            } else {
                const uint16_t sizes[NOC_SCATTER_WRITE_MAX_CHUNKS - 1] = {TILE_BYTES, TILE_BYTES, TILE_BYTES};
                fabric_unicast_noc_scatter_write(
                    &link,
                    hdr,
                    src,
                    DT * TILE_BYTES,
                    tt::tt_fabric::NocUnicastScatterCommandHeader(addrs, sizes, static_cast<uint8_t>(DT)),
                    hops[t]);
            }
            // The header and the source slot are reused: their L1 reads must have left before the next packet.
            noc.async_writes_flushed();
        }
        if (next < KV_UNITS && !prefetched) {
            wait_ready(next);
            read_unit(next, slot ^ 1);
        }
    }

    for (uint32_t t = 0; t < n_targets; ++t) {
        fabric_unicast_noc_unicast_atomic_inc(
            &link, hdr, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{slot_noc_addr, 1}, hops[t]);
        noc.async_writes_flushed();
    }
    fabric.close();
}
