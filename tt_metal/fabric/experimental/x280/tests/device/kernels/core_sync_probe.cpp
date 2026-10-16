// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Grid participant kernel for core_sync_barrier hardware measurement. Every
// participant records the timestamp at which it left the barrier, so the host
// can report the spread across the grid rather than a pass bit.
//
// The barrier can be compiled out (BarrierEnabled = 0) without changing
// anything else the kernel does. That build is the null run: the same
// timestamps taken by the same cores with nothing synchronising them, which is
// what separates a clock offset between two free-running counters from skew the
// barrier is answerable for.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
/* Named by path rather than by -I: the ERISC and DRISC launches of this source
 * get no include root beyond its own directory. */
#include "../../../kernels/shared/core_sync_barrier.hpp"
#include "../../../kernels/shared/core_sync_core_family.h"
#include "../../../kernels/shared/core_sync_probe_args.h"
#include "../../../kernels/shared/core_sync_layout.h"
#include "../../../kernels/shared/core_sync_transport_metal.hpp"
#include "../../../include/x280_noc_coord.h"

// Compile-time args:
//   0: is_director (1 if director, 0 if worker)
//   1: total_workers (if director) or worker_index (if worker)
//   2: barrier_enabled (0 runs the null measurement)
//   3: arrival_mode (0 counter, 1 word -- see ArrivalWord below)
//   4: core_family (0 Tensix, 1 idle ethernet, 2 DRISC, 3 active ethernet --
//      this core's own family; read on the worker path only, the director
//      being Tensix)
//
// Runtime args, both roles (core_sync_probe_args.h):
//   0: layout cookie, checked before any other slot; a mismatch returns
//      without writing the result
//   1: run_nonce
//   2: director_ready_base
//   3: result_l1_addr
//   4: arrival_semaphore_id (on the director, the counter every worker increments)
// Director, then: grid_x0, grid_y0, grid_x1, grid_y1, grid_semaphore_id,
//   grid_num_dests, counter_arrivals, endpoint_count, word_arrival_count,
//   then 5 per endpoint: noc_x, noc_y, semaphore_id_or_l1_addr, is_semaphore,
//   core_type,
//   then 2 per word arrival: slot_addr, worker_index.
// Worker, then: director_noc_x, director_noc_y, release_semaphore_id,
//   worker_index, release_l1_addr.

using GridTransport = core_sync::MetalTransport<ProgrammableCoreType::TENSIX>;

// Arrival by private word rather than by counter. It is the tier every worker
// of an X280 director takes, because a NoC atomic cannot reach an L2CPU's
// inbound AXI endpoint (core_sync_transport_x280.hpp, kMayHostCounter).
template <ProgrammableCoreType CoreType>
using WordTransport = core_sync::MetalTransport<CoreType, core_sync::ArrivalKind::Word>;

// A worker's transport, parameterized on the worker's own core family: that is
// the space its release semaphore resolves in.
template <ProgrammableCoreType CoreType>
using CounterTransport = core_sync::MetalTransport<CoreType>;

static constexpr uint32_t ARRIVAL_COUNTER = 0;
static constexpr uint32_t ARRIVAL_WORD = 1;

template <bool BarrierEnabled>
struct GridDirectorConfig {
    static constexpr bool kEnabled = BarrierEnabled;
    static constexpr bool kIsDirector = true;
    static inline GridTransport::Roster kRoster{};
};

template <bool BarrierEnabled, typename Transport>
struct GridWorkerConfig {
    static constexpr bool kEnabled = BarrierEnabled;
    static constexpr bool kIsDirector = false;
    static inline uint32_t kLocalReleaseAddr = 0;
    static inline typename Transport::ArrivalTarget kArrivalTarget{};
};

void kernel_main() {
    constexpr uint32_t is_director = get_compile_time_arg_val(0);
    constexpr uint32_t count_or_idx = get_compile_time_arg_val(1);
    constexpr bool barrier_enabled = get_compile_time_arg_val(2) != 0;
    constexpr uint32_t arrival_mode = get_compile_time_arg_val(3);
    constexpr uint32_t core_family = get_compile_time_arg_val(4);
    constexpr ProgrammableCoreType kWorkerCore =
        core_family == CORE_SYNC_CORE_FAMILY_IDLE_ETH     ? ProgrammableCoreType::IDLE_ETH
        : core_family == CORE_SYNC_CORE_FAMILY_DRAM       ? ProgrammableCoreType::DRAM
        : core_family == CORE_SYNC_CORE_FAMILY_ACTIVE_ETH ? ProgrammableCoreType::ACTIVE_ETH
                                                          : ProgrammableCoreType::TENSIX;
    static_assert(
        core_family == CORE_SYNC_CORE_FAMILY_TENSIX || core_family == CORE_SYNC_CORE_FAMILY_IDLE_ETH ||
            core_family == CORE_SYNC_CORE_FAMILY_DRAM || core_family == CORE_SYNC_CORE_FAMILY_ACTIVE_ETH,
        "core_family names no core family this probe runs on");

    if (!ubench_layout_is_current(
            get_arg_val<uint32_t>(CORE_SYNC_PROBE_ARG_LAYOUT_COOKIE), CORE_SYNC_PROBE_LAYOUT_COOKIE)) {
        return;
    }

#if defined(CORE_SYNC_PROBE_DRISC)
    static_assert(core_family == CORE_SYNC_CORE_FAMILY_DRAM, "the DRISC entry point requires the DRAM core family");
#else
    static_assert(core_family != CORE_SYNC_CORE_FAMILY_DRAM, "the DRAM core family requires the DRISC entry point");
#endif

    const uint32_t run_nonce = get_arg_val<uint32_t>(CORE_SYNC_PROBE_ARG_RUN_NONCE);
    const uint32_t director_ready_base = get_arg_val<uint32_t>(CORE_SYNC_PROBE_ARG_DIRECTOR_READY_BASE);
    const uint32_t result_l1_addr = get_arg_val<uint32_t>(CORE_SYNC_PROBE_ARG_RESULT_L1_ADDR);
    const uint32_t arrival_sem_id = get_arg_val<uint32_t>(CORE_SYNC_PROBE_ARG_ARRIVAL_SEMAPHORE_ID);
    constexpr uint32_t kRole = CORE_SYNC_PROBE_ARG_ROLE_BASE;

    core_sync::Outcome outcome = core_sync::Outcome::ConfigError;
    uint64_t arrival = 0;

    if constexpr (is_director != 0) {
        constexpr uint32_t total_workers = count_or_idx;
        using Config = GridDirectorConfig<barrier_enabled>;

        const x280_noc_rect_t grid_rect{
            {get_arg_val<uint32_t>(kRole + 0), get_arg_val<uint32_t>(kRole + 1)},
            {get_arg_val<uint32_t>(kRole + 2), get_arg_val<uint32_t>(kRole + 3)}};
        const uint32_t grid_sem_id = get_arg_val<uint32_t>(kRole + 4);
        const uint32_t grid_num_dests = get_arg_val<uint32_t>(kRole + 5);
        const uint32_t counter_arrivals = get_arg_val<uint32_t>(kRole + 6);
        const uint32_t endpoint_count = get_arg_val<uint32_t>(kRole + 7);
        const uint32_t word_arrival_count = get_arg_val<uint32_t>(kRole + 8);
        constexpr uint32_t kEndpointsBase = kRole + 9;

        /* The arrays hold total_workers; a larger count is refused rather than
         * written past, and the transport's roster validation refuses an
         * out-of-range coordinate before any wait or release. */
        const bool counts_fit = endpoint_count <= total_workers && word_arrival_count <= total_workers;
        core_sync::MetalEndpoint worker_eps[total_workers];
        for (uint32_t i = 0; counts_fit && i < endpoint_count; ++i) {
            const uint32_t ep = kEndpointsBase + i * 5;
            const x280_noc_coord_t w_noc{get_arg_val<uint32_t>(ep + 0), get_arg_val<uint32_t>(ep + 1)};
            const uint32_t w_target = get_arg_val<uint32_t>(ep + 2);
            const uint32_t w_is_sem = get_arg_val<uint32_t>(ep + 3);
            const uint32_t w_core_type = get_arg_val<uint32_t>(ep + 4);
            worker_eps[i] = core_sync::MetalEndpoint(
                w_noc, w_target, static_cast<ProgrammableCoreType>(w_core_type), w_is_sem != 0);
        }

        const uint32_t words_base = kEndpointsBase + endpoint_count * 5;
        core_sync::WordArrival word_arrivals[total_workers];
        for (uint32_t i = 0; counts_fit && i < word_arrival_count; ++i) {
            word_arrivals[i].addr = get_arg_val<uint32_t>(words_base + i * 2);
            word_arrivals[i].worker_index = get_arg_val<uint32_t>(words_base + 1 + i * 2);
        }

        Config::kRoster = core_sync::MetalRoster{
            worker_eps,
            endpoint_count,
            static_cast<uint32_t>(get_semaphore<ProgrammableCoreType::TENSIX>(arrival_sem_id)),
            counter_arrivals,
            word_arrivals,
            word_arrival_count,
            core_sync::MetalGroup{grid_rect, grid_sem_id, grid_num_dests}};

        if (counts_fit) {
            outcome = core_sync::core_sync_barrier<Config, GridTransport>::sync(0, 0, run_nonce);
        }
        arrival = get_timestamp();
    } else {
        constexpr uint32_t worker_idx = count_or_idx;

        const x280_noc_coord_t director_noc{get_arg_val<uint32_t>(kRole + 0), get_arg_val<uint32_t>(kRole + 1)};
        const uint32_t release_sem_id = get_arg_val<uint32_t>(kRole + 2);
        const uint32_t word_index = get_arg_val<uint32_t>(kRole + 3);
        const uint32_t release_l1_addr = get_arg_val<uint32_t>(kRole + 4);
        /* An out-of-range director coordinate is refused rather than masked
         * into another tile's address. */
        const bool director_valid = x280_noc_coord_valid(director_noc);

        if constexpr (arrival_mode == ARRIVAL_WORD) {
            using Transport = WordTransport<kWorkerCore>;
            using Config = GridWorkerConfig<barrier_enabled, Transport>;

            Config::kArrivalTarget = core_sync::MetalWordArrivalTarget{director_noc, director_ready_base, word_index};
            /* A word worker's release lands on a raw L1 address rather than a
             * semaphore: the director that releases it is an X280, which has no
             * semaphore id to name. */
            Config::kLocalReleaseAddr = release_l1_addr;

            if (director_valid) {
                outcome = core_sync::core_sync_barrier<Config, Transport>::sync(0, 0, run_nonce);
            }
        } else {
            using Transport = CounterTransport<kWorkerCore>;
            using Config = GridWorkerConfig<barrier_enabled, Transport>;

            /* TENSIX names the DIRECTOR's family, which is where the counter
             * lives, not this worker's -- the director is Tensix in every case
             * this probe runs. The kWorkerCore below names this core's own. */
            Config::kArrivalTarget =
                core_sync::MetalCounterArrivalTarget{director_noc, arrival_sem_id, ProgrammableCoreType::TENSIX};
            Config::kLocalReleaseAddr = static_cast<uint32_t>(get_semaphore<kWorkerCore>(release_sem_id));

            if (director_valid) {
                outcome = core_sync::core_sync_barrier<Config, Transport>::sync(0, 0, run_nonce);
            }
        }
        arrival = get_timestamp();
    }

    volatile tt_l1_ptr uint32_t* res = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(result_l1_addr);
    res[CORE_SYNC_PROBE_RESULT_OUTCOME] = static_cast<uint32_t>(outcome);
    res[CORE_SYNC_PROBE_RESULT_EXIT_LO] = static_cast<uint32_t>(arrival & 0xFFFFFFFFu);
    res[CORE_SYNC_PROBE_RESULT_EXIT_HI] = static_cast<uint32_t>(arrival >> 32);
    res[CORE_SYNC_PROBE_RESULT_RUN_NONCE] = run_nonce;
}
