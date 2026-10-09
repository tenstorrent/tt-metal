// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ClusterSemaphore datacopy, chain, and closed-ring tests. All tile operations use Metal compute APIs.
// The DFB supplies tile metadata and a shared L1 allocation. Reserve it once and address tiles explicitly:
// never push/pop credits, since DFB synchronization would mask broken ClusterSemaphore ordering.
// The host initializes the input/ring data before launch; subsequent slot ownership is managed only by
// ClusterSemaphore. PACK publishes a slot with up(), and UNPACK releases it with down().

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/experimental/cluster_semaphore.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    using namespace ckernel;
    constexpr std::uint32_t num_neos = get_arg(args::num_neos);
    constexpr std::uint32_t depth = get_arg(args::depth);
    constexpr std::uint32_t max_tiles = get_arg(args::max_tiles);
    constexpr bool closed_ring = get_arg(args::closed_ring);
    constexpr std::uint32_t batch_tiles = get_arg(args::batch_tiles);
    constexpr bool omit_ready_wait = get_arg(args::omit_ready_wait);
    constexpr bool omit_space_wait = get_arg(args::omit_space_wait);
    constexpr std::uint32_t consumer_skew = get_arg(args::consumer_skew);
    constexpr bool nosync = get_arg(args::nosync);
    constexpr std::uint32_t skew_spins = get_arg(args::skew_spins);
    constexpr std::uint32_t local_capacity = get_arg(args::local_capacity);
    const std::uint32_t num_iters = get_arg(args::num_iters);
    const std::uint32_t report_addr = get_arg(args::report_addr);
    const std::uint32_t active_neos = get_arg(args::active_neos);
    const std::uint32_t neo = get_my_thread_id();
    const bool active = neo < active_neos;
    const bool consumes_ring = closed_ring || neo != 0;
    const bool produces_ring = closed_ring || neo != num_neos - 1;
    const std::uint32_t previous = neo == 0 ? active_neos - 1 : neo - 1;

    DataflowBuffer input(dfb::tiles_in);
    DataflowBuffer output(dfb::tiles_out);
    // Quasar has no descriptor-based 2.0 startup/copy/pack LLKs yet. Keep only those
    // compute operations on the ID-based API; buffer management uses the 2.0 DFB interface.
    compute_kernel_hw_startup(input.get_id(), output.get_id());
    copy_init(input.get_id());
    output.reserve_back(local_capacity);

    // A STRIDED binding gives NEO n a BFD base n tiles above the allocation base. Tiles below are
    // physical, from the allocation base. UNPACK reads tile T at index T - n, but PACK steps num_neos
    // tiles per index, so NEO n can only pack tiles n, n + num_neos, ... Each ring and the output therefore
    // live in their producer's lane: slot s of NEO n's ring is ring_start + s * num_neos + n. The first
    // num_neos tiles are padding (also used for host reports). No FIFO cursor advances in this test.
    const std::uint32_t input_start = num_neos;
    const std::uint32_t ring_start = (input_start + max_tiles + num_neos - 1) / num_neos * num_neos;
    const std::uint32_t output_start = ring_start + depth * num_neos;

#if defined(TRISC_UNPACK) || defined(TRISC_PACK)
    // UNPACK waits on the previous NEO's semaphore; PACK signals its own.
    const std::uint32_t semaphore_index =
#ifdef TRISC_UNPACK
        consumes_ring ? previous : 0;
#else
        neo;
#endif
#ifdef TRISC_PACK
    // NEO 0 sets every semaphore's start value: full in a closed ring, else empty.
    if (neo == 0) {
        for (std::uint32_t index = 0; index < num_neos; ++index) {
            ClusterSemaphore initial(index, depth);
            initial.set(closed_ring && index < active_neos ? depth : 0);
        }
    }
#endif
#endif
    // Drain MATH startup writes before PACK can wait on MATH_PACK.
#ifdef TRISC_MATH
    tensix_sync();
#endif
    // No participant may use a ring until NEO 0 has initialized every semaphore.
    sync_threads();
#if defined(TRISC_UNPACK) || defined(TRISC_PACK)
    ClusterSemaphore sem(semaphore_index, depth);
#endif

    // Datacopy / chain / ring: each NEO copies tiles from its input (or the previous NEO's ring) to its
    // output (or its own ring). Inactive NEOs skip this.
    if (active) {
        for (std::uint32_t i = 0; i < num_iters; ++i) {
#ifdef TRISC_UNPACK
            // Optional delay so the consumer runs behind the producer.
            if (consumes_ring) {
                for (volatile std::uint32_t spin = 0; spin < consumer_skew; ++spin) {
                }
            }
            // At the start of each batch, wait until the producer has published the whole batch.
            if constexpr (!nosync && !omit_ready_wait) {
                if (consumes_ring && i % batch_tiles == 0) {
                    sem.wait_min(batch_tiles);
                }
            }
#endif
            tile_regs_acquire();
            // Read from the previous NEO's ring slot, or from this NEO's input tiles.
            const std::uint32_t src =
                consumes_ring ? ring_start + (i % depth) * num_neos + previous : input_start + i % max_tiles;
            copy_tile(input.get_id(), src - neo, 0);
#ifdef TRISC_UNPACK
            // At the end of each batch, give the slots back to the producer.
            if constexpr (!nosync) {
                if (consumes_ring && (i + 1) % batch_tiles == 0) {
                    sem.down(batch_tiles);
                }
            }
#endif
            tile_regs_commit();
#ifdef TRISC_PACK
            // At the start of each batch, wait until the ring has room for the whole batch.
            if constexpr (!nosync && !omit_space_wait) {
                if (produces_ring && i % batch_tiles == 0) {
                    sem.wait_not_full(batch_tiles);
                }
            }
            // Optional delay on NEO 0 so the first producer runs behind the rest.
            if (neo == 0) {
                for (volatile std::uint32_t spin = 0; spin < skew_spins; ++spin) {
                }
            }
#endif
            tile_regs_wait();
            // Write to this NEO's ring slot, or to the output tiles if this NEO ends the chain.
            const std::uint32_t dst =
                (produces_ring ? ring_start + (i % depth) * num_neos : output_start + (i % max_tiles) * num_neos) + neo;
            pack_tile<true>(0, output.get_id(), (dst - neo) / num_neos);
            tile_regs_release();
#ifdef TRISC_PACK
            // At the end of each batch, publish the packed slots to the next NEO.
            if constexpr (!nosync) {
                if (produces_ring && (i + 1) % batch_tiles == 0) {
                    sem.up(batch_tiles);
                }
            }
#endif
        }
    }
    // All consumers must finish before PACK reports the final count of its outgoing semaphore.
    sync_threads();
#ifdef TRISC_PACK
    // Report how many iterations ran and the final semaphore count for the host to check.
    if (active) {
        auto* report = reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(report_addr + MEM_L1_UNCACHED_BASE);
        report[neo] = num_iters;
        if (produces_ring) {
            report[num_neos + neo] = sem.value();
            // The report preserves the final count; leave hardware clean for the next kernel.
            sem.set(0);
        }
    }
#endif
}
