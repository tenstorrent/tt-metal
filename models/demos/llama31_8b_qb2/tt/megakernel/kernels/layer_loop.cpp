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
        const uint32_t generation = *epoch + 1;
        *epoch = generation;
        noc_semaphore_inc(get_noc_addr(coordinator_x, coordinator_y, arrivals), 1);
        noc_async_atomic_barrier();
        if (index == 0) {
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrivals), generation * cores);
            for (uint32_t i = 0; i < cores; ++i) {
                noc_async_write(epoch_address,
                    get_noc_addr(get_arg_val<uint32_t>(LOOP_RT_OFFSET + 14 + 2 * i),
                                 get_arg_val<uint32_t>(LOOP_RT_OFFSET + 15 + 2 * i), release), 4);
            }
            noc_async_write_barrier();
        }
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(release), generation);
    }
#if defined(PROFILE_KERNEL) && (PROFILE_KERNEL & PROFILER_OPT_DO_SUM)
    auto* state = reinterpret_cast<uint32_t tt_l1_ptr*>(get_arg_val<uint32_t>(LOOP_RT_OFFSET));
    state[512 + state[482]++] = kernel_profiler::sums[0] - before;
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
#if defined(PROFILE_KERNEL) && (PROFILE_KERNEL & PROFILER_OPT_DO_SUM)
    state[482] = 0;
#endif
#endif
#if LOOP_PATCH == 5
    const uint32_t initial_residual = get_arg_val<uint32_t>(14);
#endif
    for (uint32_t layer = 0; layer < count; ++layer) {
#if defined(COMPILE_FOR_NCRISC)
        if (layer > 0) {
            uint32_t mask = get_arg_val<uint32_t>(LOOP_RT_OFFSET + 3);
            for (uint32_t semaphore = 0; mask; ++semaphore, mask >>= 1) {
                if (mask & 1) { noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(semaphore)), 0); }
            }
        }
#if LOOP_PATCH == 3 || LOOP_PATCH == 4
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
        auto* rt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_addr(0));
#if LOOP_PATCH == 1
        rt[6] = first + layer;  // MLP/O weight row
#elif LOOP_PATCH == 2
        rt[3] = first + layer;  // QKV weight row
#elif LOOP_PATCH == 3
        rt[1] = state[448 + LOOP_CACHE_COLUMN];
#elif LOOP_PATCH == 4
        rt[1] = state[452];
        rt[2] = state[453];
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
        unified_kernels::sync_riscs_exit<>(end_sync);
        if (layer + 1 < count) {
            // Current Blackhole implementation handles all64 CB indices and
            // synchronizes stream-register reset with producer/consumer RISCs.
            unified_kernels::reconfig_cb_interfaces(state);
        }
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
