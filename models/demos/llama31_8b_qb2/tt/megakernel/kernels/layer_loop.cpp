// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#define QB2_ENTRY loop_layer_main
#ifdef LOOP_NATIVE
#define kernel_main loop_layer_main
#endif
#include LOOP_SOURCE
#undef kernel_main
#undef QB2_ENTRY
#include "models/demos/deepseek_v3_b1/unified_kernels/kernel_utils.hpp"
#include "tools/profiler/kernel_profiler.hpp"

#if defined(COMPILE_FOR_NCRISC)
constexpr auto loop_table_args = TensorAccessorArgs<LOOP_CT_OFFSET>();
void snapshot_buffers(uint32_t tt_l1_ptr* state, uint32_t low, uint32_t high) {
    state[256] = low;
    state[257] = high;
    for (uint32_t cb = 0; cb < 64; ++cb) {
        const uint32_t mask = cb < 32 ? low : high;
        if (!(mask & (1u << (cb % 32)))) { continue; }
        const auto& source = get_local_cb_interface(cb);
        state[4 * cb] = (source.fifo_limit - source.fifo_size) << cb_addr_shift;
        state[4 * cb + 1] = source.fifo_size << cb_addr_shift;
        state[4 * cb + 2] = source.fifo_num_pages;
        state[4 * cb + 3] = source.fifo_page_size << cb_addr_shift;
    }
}
void layer_barrier() {
#if defined(PROFILE_KERNEL) && (PROFILE_KERNEL & PROFILER_OPT_DO_SUM)
    const uint32_t before = kernel_profiler::sums[0];
#endif
    {
        // Accumulate all64 barriers without exhausting the512-word marker buffer.
        // The total includes worker waiting; it is not additive model latency.
        DeviceZoneScopedSumN1("DECODER-LAYER-BARRIER-TOTAL");
        const uint32_t arrivals = get_arg_val<uint32_t>(LOOP_RT_OFFSET + 7);
        const uint32_t release = get_arg_val<uint32_t>(LOOP_RT_OFFSET + 8);
        const uint32_t epoch_address = get_arg_val<uint32_t>(LOOP_RT_OFFSET + 9);
        const uint32_t coordinator_x = get_arg_val<uint32_t>(LOOP_RT_OFFSET + 10);
        const uint32_t coordinator_y = get_arg_val<uint32_t>(LOOP_RT_OFFSET + 11);
        const uint32_t cores = get_arg_val<uint32_t>(LOOP_RT_OFFSET + 12);
        const uint32_t index = get_arg_val<uint32_t>(LOOP_RT_OFFSET + 13);
        auto* epoch = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(epoch_address);
#if BOUNDED_LAYER_BARRIER
        const uint32_t generation = *epoch ^ 1u;
#else
        const uint32_t generation = *epoch + 1;
#endif
        *epoch = generation;
        noc_semaphore_inc(get_noc_addr(coordinator_x, coordinator_y, arrivals), 1);
        noc_async_atomic_barrier();
        if (index == 0) {
#if BOUNDED_LAYER_BARRIER
            // Every contributor has flushed its atomic and cannot enter the
            // next generation before release. Reset before publishing release;
            // arrivals stay in [0, cores], and the sense bit cannot overflow.
            noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrivals), cores);
            noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrivals), 0);
#else
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrivals), generation * cores);
#endif
#if LAYER_BARRIER_MULTICAST
            const uint32_t rectangles = get_arg_val<uint32_t>(LOOP_MCAST_RT_OFFSET);
            for (uint32_t i = 0; i < rectangles; ++i) {
                const uint32_t offset = LOOP_MCAST_RT_OFFSET + 1 + 5 * i;
                const uint32_t x0 = get_arg_val<uint32_t>(offset), y0 = get_arg_val<uint32_t>(offset + 1);
                const uint32_t x1 = get_arg_val<uint32_t>(offset + 2), y1 = get_arg_val<uint32_t>(offset + 3);
                const uint32_t destinations = get_arg_val<uint32_t>(offset + 4);
                const uint64_t address = get_noc_multicast_addr(x0, y0, x1, y1, release);
                if (destinations == 1) {
                    noc_async_write(epoch_address, get_noc_addr(x0, y0, release), 4);
                } else if (coordinator_x >= x0 && coordinator_x <= x1 && coordinator_y >= y0 && coordinator_y <= y1) {
                    // Loopback counts the sender; all recipients have a live
                    // global release field, even if they are outside the loop.
                    noc_semaphore_set_multicast_loopback_src(epoch_address, address, destinations);
                } else {
                    noc_semaphore_set_multicast(epoch_address, address, destinations);
                }
            }
#else
            for (uint32_t i = 0; i < cores; ++i) {
                noc_async_write(epoch_address,
                    get_noc_addr(get_arg_val<uint32_t>(LOOP_RT_OFFSET + 14 + 2 * i),
                                 get_arg_val<uint32_t>(LOOP_RT_OFFSET + 15 + 2 * i), release), 4);
            }
#endif
            noc_async_write_barrier();
        }
#if BOUNDED_LAYER_BARRIER
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(release), generation);
#else
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(release), generation);
#endif
    }
#if defined(PROFILE_KERNEL) && (PROFILE_KERNEL & PROFILER_OPT_DO_SUM)
    auto* state = reinterpret_cast<uint32_t tt_l1_ptr*>(get_arg_val<uint32_t>(LOOP_RT_OFFSET));
    state[512 + state[482]++] = kernel_profiler::sums[0] - before;
#endif
}
#endif

#if INLINE_CB_RESET
// Called while every local RISC is held at the existing end boundary.
// Peers only see tensor data and semaphores, never these local interfaces.
// NCRISC resets stream registers only after the cross-core end rendezvous;
// no subsequent producer runs until the unchanged next start rendezvous.
void reset_layer_cb_interfaces(uint32_t tt_l1_ptr* state) {
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
    constexpr bool read = true, write = true, tile_ptr = false;
#elif defined(UCK_CHLKC_UNPACK)
    constexpr bool read = true, write = false, tile_ptr = false;
#elif defined(UCK_CHLKC_PACK)
    constexpr bool read = false, write = true, tile_ptr = true;
#endif
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC) || defined(UCK_CHLKC_UNPACK) || defined(UCK_CHLKC_PACK)
#if defined(COMPILE_FOR_NCRISC)
    constexpr bool stream = true;
#else
    constexpr bool stream = false;
#endif
    unified_kernels::reconfig_cbs_for_mask<read, write, tile_ptr, stream>(state, state[256], 0);
    unified_kernels::reconfig_cbs_for_mask<read, write, tile_ptr, stream>(state, state[257], 32);
#endif
}
#endif

void kernel_main() {
    auto* state = reinterpret_cast<uint32_t tt_l1_ptr*>(get_arg_val<uint32_t>(LOOP_RT_OFFSET));
    // Separate start/end words prevent a fast RISC from consuming a prior
    // barrier's undrained exit count when immediately entering the next one.
    auto* start_sync = reinterpret_cast<volatile uint32_t tt_l1_ptr*>(state + 259);
    auto* end_sync = reinterpret_cast<volatile uint32_t tt_l1_ptr*>(state + 260);
    const uint32_t first = get_arg_val<uint32_t>(LOOP_RT_OFFSET + 5);
    const uint32_t count = get_arg_val<uint32_t>(LOOP_RT_OFFSET + 6);
#if defined(COMPILE_FOR_NCRISC)
    snapshot_buffers(state, get_arg_val<uint32_t>(LOOP_RT_OFFSET + 1), get_arg_val<uint32_t>(LOOP_RT_OFFSET + 2));
#if ALIAS_PROJECTION_CBS && defined(PROJECTION)
    // A multi-format static allocation has one physical extent. Give each
    // phase its original logical ring, preserving exact block wrap points.
    const auto capacity = [&](uint32_t cb, uint32_t pages) {
        state[4 * cb + 1] = pages * state[4 * cb + 3];
        state[4 * cb + 2] = pages;
    };
    capacity(0, 8 * PROJECTION_BUFFERS);
    capacity(4, (CUSTOM_DOWN ? 8 : 7) * PROJECTION_BUFFERS);
    capacity(6, 4 * PROJECTION_BUFFERS);
    capacity(1, 224 * PROJECTION_BUFFERS);
    capacity(3, (CUSTOM_DOWN ? 128 : 112) * PROJECTION_BUFFERS);
    capacity(7, 64 * PROJECTION_BUFFERS);
    capacity(24, 28);
    capacity(25, 16);
#if SHARED_QKV
    capacity(8, 16 * PROJECTION_BUFFERS);
    capacity(9, 96 * PROJECTION_BUFFERS);
    capacity(26, 6);
#endif
#endif
#if defined(PROFILE_KERNEL) && (PROFILE_KERNEL & PROFILER_OPT_DO_SUM)
    state[482] = 0;
#endif
#endif
#if LOOP_PATCH == 5
    const uint32_t initial_residual = get_arg_val<uint32_t>(14);
#endif
#if defined(COMPILE_FOR_NCRISC) && CACHE_LAYER_TABLE && (LOOP_PATCH == 1 || LOOP_PATCH == 2 || LOOP_PATCH == 3 || LOOP_PATCH == 4)
    {
        DeviceZoneScopedN("LAYER-TABLE-BATCH-PREFETCH");
        // Interleaved128-byte rows span DRAM banks: issue each row using its
        // accessor address, then join once. Refill every resident invocation,
        // including inactive warmup; no residency is assumed across prefill.
        const auto cached_table = TensorAccessor(loop_table_args, get_arg_val<uint32_t>(LOOP_RT_OFFSET + 4), 128);
        for (uint32_t row = first; row < first + count; ++row) {
            noc_async_read(cached_table.get_noc_addr(row), reinterpret_cast<uint32_t>(state + 1024 + row * 32), 128);
        }
        noc_async_read_barrier();
    }
#endif
    for (uint32_t layer = 0; layer < count; ++layer) {
        const uint32_t table_row = CACHE_LAYER_TABLE ? 1024 + (first + layer) * 32 : 448;
#if defined(COMPILE_FOR_NCRISC)
        state[300] = 0;
        state[301] = 0;
#if SCRATCH_INIT_ONCE
        // Published through the following local/global start boundary. The
        // selected scratch remains physically stable across this layer loop.
        state[483] = layer;
#endif
        if (layer > 0) {
            uint32_t mask = get_arg_val<uint32_t>(LOOP_RT_OFFSET + 3);
            for (uint32_t semaphore = 0; mask; ++semaphore, mask >>= 1) {
                if (mask & 1) { noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(semaphore)), 0); }
            }
        }
#if !CACHE_LAYER_TABLE && (LOOP_PATCH == 3 || LOOP_PATCH == 4 || defined(PREFETCH_ROLE))
        // Only KV update and attention readers need shared cache addresses.
        // Projections already fetch their own weight row in the native body;
        // norm, fabric and SFPU workers do not consume any table columns here.
        const auto table = TensorAccessor(loop_table_args, get_arg_val<uint32_t>(LOOP_RT_OFFSET + 4), 128);
        noc_async_read(table.get_noc_addr(first + layer), reinterpret_cast<uint32_t>(state + 448), 128);
        noc_async_read_barrier();
#endif
#endif
        // Both cross-RISC and cross-core boundaries are required: a core must
        // finish resetting local semaphores/CBs before any peer can produce.
        unified_kernels::sync_riscs_enter<>(start_sync);
#if defined(COMPILE_FOR_NCRISC)
        layer_barrier();
#endif
        unified_kernels::sync_riscs_exit<>(start_sync);
#if ALIAS_PROJECTION_CBS && defined(PROJECTION)
        // Start release guarantees NCRISC finished the configuration snapshot.
        // Later layers already reconfigure at the preceding end boundary.
        if (layer == 0) { unified_kernels::reconfig_cb_interfaces(state); }
#endif
        auto* rt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_addr(0));
#if LOOP_PATCH == 1
        rt[6] = first + layer;  // MLP/O weight row
#elif LOOP_PATCH == 2
        rt[3] = first + layer;  // QKV weight row
#elif LOOP_PATCH == 3
        rt[1] = state[table_row + LOOP_CACHE_COLUMN];
#elif LOOP_PATCH == 4
        rt[1] = state[table_row + 4];
        rt[2] = state[table_row + 5];
#elif LOOP_PATCH == 5
        rt[14] = layer == 0 ? initial_residual : rt[15];
#ifdef FUSE_EMBEDDING
        rt[EMBED_RT_OFFSET + 2] = layer == 0;
#endif
#elif LOOP_PATCH == 6 && defined(TAIL_RT_OFFSET)
        rt[TAIL_RT_OFFSET + 20] = layer + 1 == count;
#endif
        loop_layer_main();
        unified_kernels::sync_riscs_enter<>(end_sync);
#if defined(COMPILE_FOR_NCRISC)
        layer_barrier();
#endif
#if INLINE_CB_RESET
        if (layer + 1 < count) { reset_layer_cb_interfaces(state); }
#endif
        unified_kernels::sync_riscs_exit<>(end_sync);
#if !INLINE_CB_RESET
        if (layer + 1 < count) {
            // Current Blackhole implementation handles all64 CB indices and
            // synchronizes stream-register reset with producer/consumer RISCs.
            unified_kernels::reconfig_cb_interfaces(state);
        }
#endif
    }
#if defined(COMPILE_FOR_NCRISC) && defined(PROFILE_KERNEL) && (PROFILE_KERNEL & PROFILER_OPT_DO_SUM)
    // Export the total independently of optional marker-buffer capacity.
    // Words480/481 are outside CB snapshots, synchronization and table scratch.
    // The separate sum-capture harness reads these after execution, outside
    // timing signposts. Avoid quick_push on an already full optional buffer.
    state[480] = kernel_profiler::sums[0];
    state[481] = state[482];
#endif
}
