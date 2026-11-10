// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// block_pool sender, a kernel of the block_topk_route program: pushes this device's K/V shard of the gathered buffer
// kv_g [1, 2H, NBG, D] to ring peers over one fabric link in one direction.
//
// The shard is the 2H * NBPT * DT tiles at tile rows [r NBPT, r NBPT + NBPT) of every head; tile i of it is head
// i / (NBPT DT), offset i % (NBPT DT) inside that head's rows. The block_pool program of the same call has completed,
// so the shard is in DRAM. The core streams its tiles [lo, hi) in packets of up to P tiles and writes each packet to
// every target whose range covers it, at the same pages of the target's kv_g (one mesh buffer: same address on every
// device). Each target then gets one atomic increment of this shard's arrival slot on its gate core, after the data
// on the same route.
//
// Compile-time args: 0 NBPT, 1 NBGT, 2 DT, 3 P, then TensorAccessorArgs of kv_g.
// Common runtime args: 0 kv_g address, 1 gate core noc x, 2 gate core noc y.
// Runtime args: 0 ring index r, 1 arrival slot address, 2 lo, 3 hi, 4 number of targets, then (hops, lo, hi) per
// target, then the fabric connection [has_fwd, args..., has_bwd, args...] with exactly one of the two set.

#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
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
    constexpr uint32_t NBPT = get_compile_time_arg_val(0);
    constexpr uint32_t NBGT = get_compile_time_arg_val(1);
    constexpr uint32_t DT = get_compile_time_arg_val(2);
    constexpr uint32_t P = get_compile_time_arg_val(3);
    constexpr auto kv_args = TensorAccessorArgs<4>();
    constexpr uint32_t HEAD_TILES = NBPT * DT;
    static_assert(P >= 1 && P <= NOC_SCATTER_WRITE_MAX_CHUNKS, "one packet holds at most 4 scattered tiles");

    size_t arg = 0;
    const uint32_t r = get_arg_val<uint32_t>(arg++);
    const uint32_t slot_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t lo = get_arg_val<uint32_t>(arg++);
    const uint32_t hi = get_arg_val<uint32_t>(arg++);
    const uint32_t n_targets = get_arg_val<uint32_t>(arg++);
    if (n_targets == 0) {
        return;
    }
    ASSERT(n_targets <= MAX_TARGETS);
    uint8_t hops[MAX_TARGETS];
    uint32_t t_lo[MAX_TARGETS], t_hi[MAX_TARGETS];
    for (uint32_t t = 0; t < n_targets; ++t) {
        hops[t] = static_cast<uint8_t>(get_arg_val<uint32_t>(arg++));
        t_lo[t] = get_arg_val<uint32_t>(arg++);
        t_hi[t] = get_arg_val<uint32_t>(arg++);
    }
    auto fabric = FabricConnectionManager::build_from_args<
        FabricConnectionManager::BuildFromArgsMode::BUILD_AND_OPEN_CONNECTION_START_ONLY>(arg);

    const auto kv_g = TensorAccessor(kv_args, get_common_arg_val<uint32_t>(0));
    const uint64_t slot_noc_addr =
        safe_get_noc_addr(get_common_arg_val<uint32_t>(1), get_common_arg_val<uint32_t>(2), slot_addr, 0);
    auto page_of = [&](uint32_t i) { return (i / HEAD_TILES * NBGT + r * NBPT) * DT + i % HEAD_TILES; };

    Noc noc;
    CircularBuffer cbs(cb_send);
    cbs.reserve_back(2 * P);
    const uint32_t buf = cbs.get_write_ptr();
    auto read_chunk = [&](uint32_t first, uint32_t slot) {
        const uint32_t n = first + P < hi ? P : hi - first;
        for (uint32_t i = 0; i < n; ++i) {
            noc.async_read(
                kv_g,
                CoreLocalMem<uint32_t>(buf + (slot * P + i) * TILE_BYTES),
                TILE_BYTES,
                {.page_id = page_of(first + i)},
                {});
        }
    };

    fabric.open_finish();
    auto& link = fabric.has_forward_connection() ? fabric.get_forward_connection() : fabric.get_backward_connection();
    volatile PACKET_HEADER_TYPE* hdr = PacketHeaderPool::allocate_header();

    read_chunk(lo, 0);
    uint32_t slot = 0;
    for (uint32_t first = lo; first < hi; first += P, slot ^= 1) {
        noc.async_read_barrier();
        if (first + P < hi) {
            read_chunk(first + P, slot ^ 1);
        }
        const uint32_t end = first + P < hi ? first + P : hi;
        const uint32_t src_base = buf + slot * P * TILE_BYTES;
        for (uint32_t t = 0; t < n_targets; ++t) {
            const uint32_t a = first > t_lo[t] ? first : t_lo[t];
            const uint32_t b = end < t_hi[t] ? end : t_hi[t];
            if (a >= b) {
                continue;
            }
            const uint32_t src = src_base + (a - first) * TILE_BYTES;
            const uint32_t n = b - a;
            uint64_t addrs[NOC_SCATTER_WRITE_MAX_CHUNKS] = {0, 0, 0, 0};
            for (uint32_t i = 0; i < n; ++i) {
                addrs[i] = tt::tt_fabric::linear::addrgen_detail::get_noc_address(kv_g, page_of(a + i), 0);
            }
            if (n == 1) {
                fabric_unicast_noc_unicast_write(
                    &link, hdr, src, TILE_BYTES, tt::tt_fabric::NocUnicastCommandHeader{addrs[0]}, hops[t]);
            } else {
                const uint16_t sizes[NOC_SCATTER_WRITE_MAX_CHUNKS - 1] = {TILE_BYTES, TILE_BYTES, TILE_BYTES};
                fabric_unicast_noc_scatter_write(
                    &link,
                    hdr,
                    src,
                    n * TILE_BYTES,
                    tt::tt_fabric::NocUnicastScatterCommandHeader(addrs, sizes, static_cast<uint8_t>(n)),
                    hops[t]);
            }
            // The header and the source slot are reused: their L1 reads must have left before the next packet.
            noc.async_writes_flushed();
        }
    }

    for (uint32_t t = 0; t < n_targets; ++t) {
        fabric_unicast_noc_unicast_atomic_inc(
            &link, hdr, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{slot_noc_addr, 1}, hops[t]);
        noc.async_writes_flushed();
    }
    fabric.close();
}
