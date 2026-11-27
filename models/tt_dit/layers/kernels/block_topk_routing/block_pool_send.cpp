// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// block_pool sender, a kernel of the block_topk_route program: pushes this device's K/V shard of the gathered buffer
// kv_g [1, 2H, NBG, D] to ring peers over one fabric link in one direction, while the other cores are still pooling.
//
// The K/V units are the first KV_UNITS of the pool's numbering (block_pool_units.hpp); a unit is DT tiles at
// consecutive pages (head NBGT + r NBPT + g) DT of kv_g. At launch the sender zeroes its page (SenderPage in
// block_pool_units.hpp; flag word u: unit u is in DRAM, incremented by the pool writer that wrote it) and sets
// QUEUE_GO_SEM; the queue's core (link 0, parity 0) multicasts the page's address to FLAGS_ADDR_SEM of every pool core
// (the page has the same address on every sender), and then every sender multicasts one increment to their
// FLAGS_READY_SEM. The sender of link l takes units l, l + L, l + 2L, ... in completion order: it sweeps its unsent
// units, takes one whose flag is set (marking it taken), reads it from DRAM and writes it in one scatter packet to
// every target, at the same pages of the target's kv_g (one mesh buffer: same address on every device); the next ready
// unit's read is issued before the current one's packets. A target reached from both directions (the opposite position
// of an even ring) takes the units whose index (u / L) has the parity of this sender. Each target then gets one atomic
// increment of this shard's arrival slot on its gate core, after the data on the same route.
//
// Compile-time args: 0 H, 1 RT, 2 NBPT, 3 NBGT, 4 DT, 5 NCORES (the pool cores), 6 L, then TensorAccessorArgs of kv_g.
// Common runtime args: 0 kv_g address, 1 gate core noc x, 2 gate core noc y, then the pool cores as two rectangles,
// each noc x0, y0, x1, y1 and its core count.
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
constexpr uint32_t cb_send = 6, cb_flags = 3;
constexpr uint32_t TILE_BYTES = 2048;
constexpr uint32_t MAX_TARGETS = 32;
constexpr uint32_t TAKEN = 2;
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
    using Page = block_pool::SenderPage<KV_UNITS, NCORES>;

    // Before the early return below: every pool writer waits for every sender's ready increment.
    Noc noc;
    CircularBuffer cbf(cb_flags);
    const uint32_t flags_addr = cbf.get_write_ptr();
    volatile tt_l1_ptr uint32_t* flags = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flags_addr);
    noc.async_write_zeros(CoreLocalMem<uint32_t>(flags_addr), Page::BYTES);
    noc.write_zeros_l1_barrier();
    flags[Page::ADDR_WORD] = flags_addr;
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(block_pool::QUEUE_GO_SEM)) = 1;
    // A NOC_1 multicast rectangle runs from the high corner to the low one.
    auto pool_cores = [&](uint32_t rect, uint32_t addr) {
        const uint32_t x0 = get_common_arg_val<uint32_t>(3 + 5 * rect), y0 = get_common_arg_val<uint32_t>(4 + 5 * rect);
        const uint32_t x1 = get_common_arg_val<uint32_t>(5 + 5 * rect), y1 = get_common_arg_val<uint32_t>(6 + 5 * rect);
        return noc_index == 0 ? get_noc_multicast_addr(x0, y0, x1, y1, addr)
                              : get_noc_multicast_addr(x1, y1, x0, y0, addr);
    };
    // Multicasts from the four senders serialize, so only the queue's core sends the address the readers wait for.
    if (get_arg_val<uint32_t>(2) == 0 && get_arg_val<uint32_t>(3) == 0) {
        for (uint32_t rect = 0; rect < 2; ++rect) {
            noc_semaphore_set_multicast(
                flags_addr + Page::ADDR_WORD * 4,
                pool_cores(rect, get_semaphore(block_pool::FLAGS_ADDR_SEM)),
                get_common_arg_val<uint32_t>(7 + 5 * rect));
        }
        noc_async_write_barrier();
    }
    for (uint32_t rect = 0; rect < 2; ++rect) {
        noc_semaphore_inc_multicast(
            pool_cores(rect, get_semaphore(block_pool::FLAGS_READY_SEM)),
            1,
            get_common_arg_val<uint32_t>(7 + 5 * rect));
    }
    noc_async_atomic_barrier();

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
    uint32_t lowest_untaken = l;
    auto take_ready = [&]() -> uint32_t {
        invalidate_l1_cache();
        while (lowest_untaken < KV_UNITS && flags[lowest_untaken] == TAKEN) {
            lowest_untaken += L;
        }
        for (uint32_t u = lowest_untaken; u < KV_UNITS; u += L) {
            if (flags[u] == 1) {
                flags[u] = TAKEN;
                return u;
            }
        }
        return KV_UNITS;
    };
    auto wait_ready = [&]() {
        uint32_t u = take_ready();
        while (u == KV_UNITS) {
            u = take_ready();
        }
        return u;
    };

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

    uint32_t remaining = l < KV_UNITS ? (KV_UNITS - l + L - 1) / L : 0;
    uint32_t u = KV_UNITS;
    if (remaining) {
        u = wait_ready();
        read_unit(u, 0);
    }
    for (uint32_t slot = 0; remaining; slot ^= 1) {
        noc.async_read_barrier();
        --remaining;
        uint32_t next = remaining ? take_ready() : KV_UNITS;
        if (next < KV_UNITS) {
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
        if (remaining && next == KV_UNITS) {
            next = wait_ready();
            read_unit(next, slot ^ 1);
        }
        u = next;
    }

    for (uint32_t t = 0; t < n_targets; ++t) {
        fabric_unicast_noc_unicast_atomic_inc(
            &link, hdr, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{slot_noc_addr, 1}, hops[t]);
        noc.async_writes_flushed();
    }
    fabric.close();
}
