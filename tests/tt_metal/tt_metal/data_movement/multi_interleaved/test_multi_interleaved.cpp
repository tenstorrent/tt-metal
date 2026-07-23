// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <numeric>
#include <random>

#include "multi_device_fixture.hpp"
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "dm_common.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <distributed/mesh_device_impl.hpp>

namespace tt::tt_metal {

using namespace std;
using namespace tt;
using namespace tt::test_utils;

namespace unit_tests::dm::multi_interleaved {
// Test config, i.e. test parameters
struct MultiInterleavedConfig {
    uint32_t test_id = 0;
    uint32_t num_of_transactions = 0;
    uint32_t num_pages = 0;
    uint32_t page_size_bytes = 0;
    DataFormat l1_data_format = DataFormat::Invalid;
    CoreRangeSet cores;
    bool read_kernel = true;
    bool write_kernel = true;
    // false (default): reader on RISCV0/NOC0, writer on RISCV1/NOC1 (the multi_interleaved standard).
    // true: swap to reader on RISCV1/NOC1, writer on RISCV0/NOC0 (the tt-metal default reader NoC).
    bool default_noc = false;
};

/// @brief Does Interleaved buffer --> Reader --> L1 --> Writer --> Interleaved buffer
/// @param mesh_device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool run_dm(const shared_ptr<distributed::MeshDevice>& mesh_device, const MultiInterleavedConfig& test_config) {
    log_info(
        tt::LogTest,
        "num transaction {}, num pages: {}, page size bytes: {}",
        test_config.num_of_transactions,
        test_config.num_pages,
        test_config.page_size_bytes);

    // Get the actual device for this single-device test
    IDevice* device = mesh_device->impl().get_device(0);

    // Program
    Program program = CreateProgram();

    // ===== L1-provider mode selection =====
    // TT_DM_L1_PROVIDERS=N (read-only path only): instead of reading from the interleaved DRAM buffer, the
    // readers read from the L1 of N provider cores, removing the DRAM controllers from the path. The set of
    // provider cores (and thus which readers are reserved) depends on TT_DM_PROVIDER_LAYOUT:
    //   0 = leftcol (default): first N cores of the leftmost column; that whole column is reserved so the
    //       reader set stays a clean rectangle.
    //   1 = mirror: providers live in the two logical columns adjacent to the physical DRAM links
    //       (heatmap x=1 = left DRAM, x=11 = middle DRAM). BOTH whole columns are always reserved from
    //       readers (constant 90-reader set across an N-sweep), so only provider count/placement varies.
    //       N is split ~proportional to the DRAM channels per side (3 left : 4 right), clamped to the 10
    //       rows per column and spread evenly over the rows. The special case N=7 uses the exact
    //       DRAM-mirror rows: left heatmap (1,2)(1,9)(1,11); right heatmap (11,3)(11,4)(11,6)(11,8)
    //       (the top-right provider is placed at y=4; the DRAM channel at y=0 has no Tensix row).
    // All readers read the same small shared resident buffer replicated round-robin across the providers.
    // A matched DRAM baseline (same reader set, but reading DRAM) can be run with TT_DM_RESERVE_COL=1, which
    // reserves the same cores for the active layout without serving from L1.
    const bool read_only = (test_config.read_kernel && !test_config.write_kernel);
    const uint32_t n_requested = read_only ? tt::tt_metal::unit_tests::dm::env_uint("TT_DM_L1_PROVIDERS", 0u) : 0u;
    const uint32_t provider_layout =
        read_only ? tt::tt_metal::unit_tests::dm::env_uint("TT_DM_PROVIDER_LAYOUT", 0u) : 0u;
    const bool provider_mode = (n_requested > 0u);  // actually serve reads from L1 providers
    const bool reserve_only = read_only && tt::tt_metal::unit_tests::dm::env_uint("TT_DM_RESERVE_COL", 0u) != 0u;
    const bool do_reserve = provider_mode || reserve_only;

    CoreRangeSet reader_cores = test_config.cores;
    std::vector<CoreCoord> provider_logical;  // provider logical cores (reserved from readers)
    if (do_reserve) {
        std::vector<CoreCoord> full = corerange_to_cores(test_config.cores);
        uint32_t min_x = full[0].x, min_y = full[0].y, max_x = full[0].x, max_y = full[0].y;
        for (const auto& c : full) {
            min_x = std::min(min_x, (uint32_t)c.x);
            min_y = std::min(min_y, (uint32_t)c.y);
            max_x = std::max(max_x, (uint32_t)c.x);
            max_y = std::max(max_y, (uint32_t)c.y);
        }
        if (provider_layout == 1u) {
            // Mirror layout: providers live in the two logical columns adjacent to the physical DRAM links.
            // Coordinate note: the profiler/heatmap coordinate differs from the kernel NoC coordinate
            // (worker_core_from_logical); for this grid: logical col 0 -> heatmap x=1 (left DRAM), logical
            // col 6 -> heatmap x=11 (middle DRAM); profiler.y == worker.y == logical.y + 2 (workers only at
            // heatmap y=2..11). Both whole columns are ALWAYS reserved from readers (constant 90-reader set
            // across the N-sweep), so only the number/placement of active L1 providers varies.
            constexpr uint32_t kLeftCol = 0u;   // heatmap x=1  (mirrors left DRAM x=0, 3 channels)
            constexpr uint32_t kRightCol = 6u;  // heatmap x=11 (mirrors middle DRAM x=9, 4 channels)
            constexpr uint32_t kNRows = 10u;    // logical rows 0..9 (heatmap y=2..11)
            std::vector<CoreRange> reader_ranges;
            for (const auto& c : full) {
                if ((uint32_t)c.x != kLeftCol && (uint32_t)c.x != kRightCol) {
                    reader_ranges.emplace_back(c);
                }
            }
            reader_cores = CoreRangeSet(reader_ranges);

            // Provider cells are only needed when actually serving from L1.
            if (provider_mode) {
                if (n_requested == 7u) {
                    // Exact DRAM mirror (with the top-right provider tweaked from y=2 to y=4):
                    //   left  -> heatmap (1,2)(1,9)(1,11);  right -> heatmap (11,3)(11,4)(11,6)(11,8)
                    provider_logical = {{0, 0}, {0, 7}, {0, 9}, {6, 1}, {6, 2}, {6, 4}, {6, 6}};
                } else {
                    // Split N proportional to DRAM channels per side (3 left : 4 right), clamped to the
                    // 10 rows per column, then spread each side's providers evenly over the rows.
                    uint32_t left_n = (uint32_t)((n_requested * 3u + 3u) / 7u);  // round(N*3/7)
                    if (left_n > kNRows) {
                        left_n = kNRows;
                    }
                    uint32_t right_n = n_requested - left_n;
                    if (right_n > kNRows) {
                        right_n = kNRows;
                        left_n = n_requested - right_n;
                    }
                    auto spread_rows = [](uint32_t count, uint32_t nrows) {
                        std::vector<uint32_t> rows;
                        if (count == 0) {
                            return rows;
                        }
                        if (count == 1) {
                            rows.push_back(nrows / 2u);
                            return rows;
                        }
                        for (uint32_t i = 0; i < count; i++) {
                            // round(i*(nrows-1)/(count-1)) via integer round-half-up
                            rows.push_back((2u * i * (nrows - 1u) + (count - 1u)) / (2u * (count - 1u)));
                        }
                        return rows;
                    };
                    for (uint32_t y : spread_rows(left_n, kNRows)) {
                        provider_logical.push_back(CoreCoord(kLeftCol, y));
                    }
                    for (uint32_t y : spread_rows(right_n, kNRows)) {
                        provider_logical.push_back(CoreCoord(kRightCol, y));
                    }
                }
            }
        } else {
            // Leftcol layout: first N cores of the leftmost column (min_x), sorted by y.
            std::vector<CoreCoord> col;
            for (const auto& c : full) {
                if ((uint32_t)c.x == min_x) {
                    col.push_back(c);
                }
            }
            std::sort(col.begin(), col.end(), [](const CoreCoord& a, const CoreCoord& b) { return a.y < b.y; });
            uint32_t take = provider_mode ? n_requested : std::min<uint32_t>(7u, (uint32_t)col.size());
            TT_FATAL(
                (uint32_t)col.size() >= take,
                "TT_DM_L1_PROVIDERS={} exceeds cores in the reserved column ({})",
                take,
                col.size());
            provider_logical.assign(col.begin(), col.begin() + take);
            reader_cores = CoreRangeSet(CoreRange(CoreCoord(min_x + 1, min_y), CoreCoord(max_x, max_y)));
        }
    }
    // Kernel/compile provider count: nonzero only when actually serving from L1.
    const uint32_t n_providers = provider_mode ? (uint32_t)provider_logical.size() : 0u;

    const uint32_t num_cores = reader_cores.num_cores();
    const size_t per_core_size_bytes = test_config.num_pages * test_config.page_size_bytes;
    const size_t total_buffer_size_bytes = num_cores * per_core_size_bytes;

    InterleavedBufferConfig interleaved_buffer_config{
        .device = device,
        .size = total_buffer_size_bytes,
        .page_size = test_config.page_size_bytes,
        .buffer_type = BufferType::DRAM};
    std::shared_ptr<Buffer> input_buffer;
    input_buffer = CreateBuffer(interleaved_buffer_config);
    uint32_t input_buffer_address = input_buffer->address();

    auto output_buffer = CreateBuffer(interleaved_buffer_config);
    uint32_t output_buffer_address = output_buffer->address();

    TT_FATAL(input_buffer_address != output_buffer_address, "Input and output buffer addresses must be different");
    TT_FATAL(test_config.read_kernel || test_config.write_kernel, "At least one kernel must run");

    // Input - generate data for all cores
    vector<uint32_t> packed_input = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -100.0f,
        100.0f,
        total_buffer_size_bytes / sizeof(bfloat16),
        chrono::system_clock::now().time_since_epoch().count());

    // Golden output
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    vector<uint32_t> packed_golden = packed_input;

    uint8_t l1_cb_index = CBIndex::c_0;
    bool sync = test_config.read_kernel == test_config.write_kernel;
    const bool enable_phase_counters = tt::tt_metal::unit_tests::dm::phase_counters_enabled();
    // TT_DM_RAND_OFFSET randomizes each reader core's DRAM-bank access order (read-only; compiled out
    // of the reader kernel when 0). Seeded deterministically so runs are reproducible.
    //   0 = off (sequential)
    //   1 = static rotation: random start bank, sequential order, same every transaction iteration
    //   2 = advancing random permutation: random per-core page/bank order whose start advances each
    //       transaction iteration (e.g. D3 D7 D1 D4 ...), decorrelating both start and batch order
    const uint32_t rand_mode = tt::tt_metal::unit_tests::dm::env_uint("TT_DM_RAND_OFFSET", 0u);
    std::mt19937 rand_offset_rng(42u);
    // TT_DM_WARMUP_ITERS runs N uninstrumented passes over the batched read loop before the single
    // measured pass, so timing counters capture steady-state behavior (compiled out of the reader
    // kernel when 0). Only affects the read-only (!sync) path.
    const uint32_t num_warmup_iters = tt::tt_metal::unit_tests::dm::env_uint("TT_DM_WARMUP_ITERS", 0u);
    // TT_DM_PAGE_COUNTERS emits one timestamp marker per noc_async_read (batched !sync path) so the
    // per-page issue cadence (and noc_cmd_buf_ready backpressure) can be reconstructed. Compiled out
    // of the reader kernel when 0.
    const uint32_t enable_page_counters = tt::tt_metal::unit_tests::dm::env_uint("TT_DM_PAGE_COUNTERS", 0u);
    // TT_DM_POSTED_WRITES makes the writer kernel issue posted writes (no per-write ack) instead of the
    // default non-posted writes. Posted writes cannot use the ack-based write barrier (it would hang) or
    // the ack counter, so the writer waits on noc_async_posted_writes_flushed() and instruments
    // NIU_MST_POSTED_WR_REQ_SENT. Write-only (!sync) path is the intended experiment.
    const bool use_posted_writes = tt::tt_metal::unit_tests::dm::env_uint("TT_DM_POSTED_WRITES", 0u) != 0u;
    if (enable_page_counters && !sync) {
        // Per-RISC profiler L1 buffer holds ~PROFILER_L1_OPTIONAL_MARKER_COUNT/2 TS_DATA markers;
        // beyond that the tail is silently dropped. One marker per page => Q is the limiting factor.
        const uint32_t Q = test_config.num_of_transactions * test_config.num_pages;
        constexpr uint32_t kPageMarkerBudget = 110u;
        if (Q > kPageMarkerBudget) {
            log_warning(
                tt::LogTest,
                "TT_DM_PAGE_COUNTERS: Q={} exceeds ~{} per-page markers; the profiler L1 buffer will "
                "overflow and the issue-cadence tail will be dropped for this point.",
                Q,
                kPageMarkerBudget);
        }
    }

    // Injection-rate throttle: spin on the RISC wall clock after every noc_async_read to deflate the
    // per-core issue rate (read-only batched path is the intended experiment; compiled out when off).
    //   TT_DM_INJECT_DELAY=N            -> fixed N-cycle delay after every read (mode 1)
    //   TT_DM_INJECT_DELAY_MIN/_MAX=a,b -> random per-transaction delay in [a,b] cycles (mode 2)
    // If any of the MIN/MAX vars are set it takes precedence (random); otherwise a nonzero fixed value
    // selects fixed mode. Everything zero leaves the kernel on its baseline max-injection path.
    const uint32_t inject_fixed = tt::tt_metal::unit_tests::dm::env_uint("TT_DM_INJECT_DELAY", 0u);
    const uint32_t inject_min_env = tt::tt_metal::unit_tests::dm::env_uint("TT_DM_INJECT_DELAY_MIN", 0u);
    const uint32_t inject_max_env = tt::tt_metal::unit_tests::dm::env_uint("TT_DM_INJECT_DELAY_MAX", 0u);
    uint32_t inject_delay_mode = 0u;
    uint32_t inject_delay_min = 0u;
    uint32_t inject_delay_max = 0u;
    if (inject_min_env > 0u || inject_max_env > 0u) {
        inject_delay_mode = 2u;  // random per-transaction
        inject_delay_min = inject_min_env;
        inject_delay_max = std::max(inject_max_env, inject_min_env);
    } else if (inject_fixed > 0u) {
        inject_delay_mode = 1u;  // fixed
        inject_delay_min = inject_fixed;
        inject_delay_max = inject_fixed;
    }
    // Per-core PRNG seeds for random injection (mode 2). Deterministic across runs for reproducibility;
    // forced nonzero so the kernel's xorshift never sticks at 0.
    std::mt19937 inject_seed_rng(0xC0FFEEu);

    // Staggered issuing: release whole columns/rows of the grid one group at a time (read-only batched
    // path). Two release mechanisms, selected by TT_DM_STAGGER_DELAY:
    //   - event handoff (default, mode 1): each core waits on a progress semaphore until all cores in
    //     preceding groups have finished issuing, then runs, then releases the next group. Adaptive, but
    //     the release travels the shared read-response path so the next group can be drain-gated.
    //   - fixed delay (mode 2, TT_DM_STAGGER_DELAY=N>0): each group spins group_rank*N wall-clock cycles
    //     after the global barrier before issuing. Open-loop time-based staircase, no handoff traffic.
    //   TT_DM_STAGGER=0 off | 1 by column (x, isolates DRAM-distance) | 2 by row (y, isolates congestion)
    //   TT_DM_STAGGER_DIR=0 ascending group index | 1 descending
    //   TT_DM_STAGGER_DELAY=N cycles per group step (0 => event handoff; N>0 => fixed-delay staircase)
    const uint32_t stagger_axis = tt::tt_metal::unit_tests::dm::env_uint("TT_DM_STAGGER", 0u);
    const uint32_t stagger_dir = tt::tt_metal::unit_tests::dm::env_uint("TT_DM_STAGGER_DIR", 0u);
    const uint32_t stagger_delay = tt::tt_metal::unit_tests::dm::env_uint("TT_DM_STAGGER_DELAY", 0u);
    // 0 = off, 1 = event handoff, 2 = fixed per-group cycle delay.
    uint32_t stagger_mode = 0u;
    if ((stagger_axis != 0u) && !sync) {
        stagger_mode = (stagger_delay > 0u) ? 2u : 1u;
    }
    const bool stagger_enabled = (stagger_mode != 0u);

    // ===== Static read-VC override (VC-validation / VC-diversification experiments) =====
    // Two ways to drive the per-core static read VC (kernel programs it once after the barrier):
    //   TT_DM_READ_VC=k        force every core onto a single static request VC k (validation sweep).
    //   TT_DM_READ_VC_BANDS=N  split the distinct rows present into N contiguous equal-count bands (y
    //                          ascending = top->bottom) and put each band on its own VC (band index == VC;
    //                          VC identity is irrelevant, only the partition boundaries matter). Targets the
    //                          row-dominated return-path funnel. Takes precedence over TT_DM_READ_VC.
    // Default (neither set) = off: baseline uses the init VC = 1. Valid VCs are 0..5 for unicast requests.
    const uint32_t read_vc_single = tt::tt_metal::unit_tests::dm::env_uint("TT_DM_READ_VC", 0xFFu);
    const uint32_t read_vc_bands = tt::tt_metal::unit_tests::dm::env_uint("TT_DM_READ_VC_BANDS", 0u);
    // Banding axis: 0 = rows (y, the contention/funnel axis), 1 = columns (x, the DRAM-distance/drain axis).
    const uint32_t read_vc_axis = tt::tt_metal::unit_tests::dm::env_uint("TT_DM_READ_VC_AXIS", 0u);
    const bool read_vc_override =
        test_config.read_kernel && !sync && ((read_vc_bands >= 2u) || (read_vc_single != 0xFFu));

    // Compile-time arguments for kernels
    vector<uint32_t> reader_compile_args = {
        (uint32_t)test_config.num_of_transactions,
        (uint32_t)test_config.num_pages,
        (uint32_t)test_config.page_size_bytes,
        (uint32_t)l1_cb_index,
        (uint32_t)test_config.test_id,
        (uint32_t)sync,
        (uint32_t)test_config.default_noc,
        (uint32_t)enable_phase_counters,
        (uint32_t)rand_mode,
        (uint32_t)num_warmup_iters,
        (uint32_t)(enable_page_counters ? 1u : 0u),
        (uint32_t)inject_delay_mode,
        (uint32_t)inject_delay_min,
        (uint32_t)inject_delay_max,
        (uint32_t)stagger_mode,
        0u,  // placeholder for stagger_sem_id, patched below once the semaphore is created (mode 1 only)
        (uint32_t)(read_vc_override ? 1u : 0u)};  // arg 16: read_vc_override (VC value is a per-core rt arg)

    vector<uint32_t> writer_compile_args = {
        (uint32_t)test_config.num_of_transactions,
        (uint32_t)test_config.num_pages,
        (uint32_t)test_config.page_size_bytes,
        (uint32_t)l1_cb_index,
        (uint32_t)test_config.test_id,
        (uint32_t)sync,
        (uint32_t)test_config.default_noc,
        (uint32_t)enable_phase_counters,
        (uint32_t)(enable_page_counters ? 1u : 0u),
        (uint32_t)(use_posted_writes ? 1u : 0u)};

    if (sync) {
        // Create circular buffers - each core only needs space for its own data
        CircularBufferConfig l1_cb_config =
            CircularBufferConfig(per_core_size_bytes, {{l1_cb_index, test_config.l1_data_format}})
                .set_page_size(l1_cb_index, test_config.page_size_bytes);
        CreateCircularBuffer(program, reader_cores, l1_cb_config);
    }

    std::vector<uint32_t> l1_addrs;
    std::vector<CoreCoord> core_list = corerange_to_cores(reader_cores);
    constexpr uint32_t noc_l1_alignment = 16u;
    constexpr uint32_t barrier_scratch_bytes = 2u * noc_l1_alignment;
    const uint32_t required_l1_bytes = per_core_size_bytes + barrier_scratch_bytes;
    for (auto& core : core_list) {
        auto [l1_addr, l1_size] = get_l1_address_and_size(mesh_device, core);
        TT_FATAL(
            l1_size >= required_l1_bytes,
            "L1 size {} must be >= per_core_size_bytes {} + barrier scratch {} (total {})",
            l1_size,
            per_core_size_bytes,
            barrier_scratch_bytes,
            required_l1_bytes);
        l1_addrs.push_back(l1_addr);
    }

    // ===== Staggered-issue per-core arg =====
    // Group cores by column (x) or row (y) and walk the groups in the chosen direction (ascending or
    // descending index). Each core gets a per-group value folded into a single runtime arg:
    //   mode 1 (event handoff): the cumulative count of cores in all earlier-ordered groups; the kernel
    //                           waits until that many cores have finished issuing before its group starts.
    //   mode 2 (fixed delay):   group_rank * stagger_delay wall-clock cycles; the kernel spins that long
    //                           after the global barrier before issuing (open-loop staircase). Empty
    //                           columns/rows of the block are skipped so the rank stays dense.
    std::vector<uint32_t> stagger_thresholds(num_cores, 0u);
    if (stagger_enabled) {
        std::vector<uint32_t> core_group(num_cores, 0u);
        uint32_t max_key = 0u;
        for (size_t i = 0; i < num_cores; ++i) {
            uint32_t key = (stagger_axis == 1u) ? (uint32_t)core_list[i].x : (uint32_t)core_list[i].y;
            core_group[i] = key;
            max_key = std::max(max_key, key);
        }
        std::vector<uint32_t> group_count(max_key + 1u, 0u);
        for (size_t i = 0; i < num_cores; ++i) {
            group_count[core_group[i]]++;
        }
        std::vector<uint32_t> group_value(max_key + 1u, 0u);
        uint32_t running_prefix = 0u;  // cumulative core count (mode 1)
        uint32_t rank = 0u;            // dense group ordinal (mode 2)
        auto assign_group = [&](uint32_t k) {
            if (group_count[k] == 0u) {
                return;  // no cores in this column/row of the block; keep the staircase dense
            }
            group_value[k] = (stagger_mode == 2u) ? (rank * stagger_delay) : running_prefix;
            running_prefix += group_count[k];
            rank += 1u;
        };
        if (stagger_dir == 0u) {  // ascending group index
            for (uint32_t k = 0u; k <= max_key; ++k) {
                assign_group(k);
            }
        } else {  // descending group index
            for (uint32_t k = max_key + 1u; k-- > 0u;) {
                assign_group(k);
            }
        }
        for (size_t i = 0; i < num_cores; ++i) {
            stagger_thresholds[i] = group_value[core_group[i]];
        }
    }

    // ===== Per-core static read VC (validation constant or contiguous row-band split) =====
    // read_vc_per_core[i] is consumed by the kernel (runtime arg 10) only when read_vc_override is set.
    std::vector<uint32_t> read_vc_per_core(num_cores, 0u);
    if (read_vc_override) {
        if (read_vc_bands >= 2u) {
            // Contiguous equal-count bands over the chosen axis: rows (y) or columns (x).
            auto key = [&](const CoreCoord& c) { return (read_vc_axis == 1u) ? (uint32_t)c.x : (uint32_t)c.y; };
            std::vector<uint32_t> uniq;
            uniq.reserve(core_list.size());
            for (const auto& c : core_list) {
                uniq.push_back(key(c));
            }
            std::sort(uniq.begin(), uniq.end());
            uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
            const uint32_t nkeys = (uint32_t)uniq.size();
            for (size_t i = 0; i < num_cores; ++i) {
                // rank of this core's row/column -> band index -> VC (identity irrelevant).
                const uint32_t rank =
                    (uint32_t)(std::lower_bound(uniq.begin(), uniq.end(), key(core_list[i])) - uniq.begin());
                read_vc_per_core[i] = (rank * read_vc_bands) / nkeys;
            }
        } else {
            for (size_t i = 0; i < num_cores; ++i) {
                read_vc_per_core[i] = read_vc_single;
            }
        }
    }

    // ===== Barrier synchronization setup =====
    // CreateSemaphore allocates semaphores on all specified cores (same ID maps to same L1 offset).
    // We only use the coordinator's semaphore - all cores increment it via NOC and poll until num_cores.
    // Creating on all cores ensures get_semaphore(id) works correctly on every core.
    CoreCoord coordinator_core = core_list[0];
    CoreCoord coordinator_phys = device->worker_core_from_logical_core(coordinator_core);

    uint32_t reader_barrier_sem_id = 0;
    uint32_t writer_barrier_sem_id = 0;

    if (test_config.read_kernel) {
        reader_barrier_sem_id = CreateSemaphore(program, reader_cores, 0);
    }
    if (test_config.write_kernel) {
        writer_barrier_sem_id = CreateSemaphore(program, reader_cores, 0);
    }

    // Dedicated progress semaphore for the event-handoff stagger (mode 1 only; mode 2 is time-based and
    // needs no handoff). Patch its ID into the reader compile args (placeholder at index 15).
    if (test_config.read_kernel && stagger_mode == 1u) {
        uint32_t reader_stagger_sem_id = CreateSemaphore(program, reader_cores, 0);
        reader_compile_args[15] = reader_stagger_sem_id;
    }

    if (read_vc_override) {
        if (read_vc_bands >= 2u) {
            log_info(
                tt::LogTest,
                "multi_interleaved read: {}-way contiguous {}-band static read VC split",
                read_vc_bands,
                (read_vc_axis == 1u) ? "column(x)" : "row(y)");
        } else {
            log_info(tt::LogTest, "multi_interleaved read: forcing all cores onto static read VC = {}", read_vc_single);
        }
    }

    // ===== L1-provider resident data =====
    // Each reader reads the same num_pages pages (page q -> provider q%N, the (q/N)-th page that provider
    // holds). We write those pages into the providers' L1 and use the shared pattern as every reader's
    // golden. provider_base is the readers' unreserved L1 base (identical across cores).
    uint32_t provider_base = 0;
    std::vector<uint32_t> provider_xy_packed;  // physical NoC coords packed x<<16|y, one per provider
    std::vector<uint32_t> shared_input;        // the num_pages resident pages, every reader's golden
    if (provider_mode) {
        TT_FATAL(rand_mode == 0u, "TT_DM_L1_PROVIDERS is mutually exclusive with TT_DM_RAND_OFFSET");
        const size_t per_core_words = per_core_size_bytes / sizeof(uint32_t);
        shared_input.resize(per_core_words);
        std::iota(shared_input.begin(), shared_input.end(), 1u);  // deterministic, content is irrelevant

        auto [prov_l1_addr, prov_l1_size] = get_l1_address_and_size(mesh_device, provider_logical[0]);
        provider_base = prov_l1_addr;
        const uint32_t page_words = test_config.page_size_bytes / sizeof(uint32_t);
        std::string prov_coord_str;
        for (uint32_t p = 0; p < n_providers; ++p) {
            CoreCoord phys = device->worker_core_from_logical_core(provider_logical[p]);
            prov_coord_str += fmt::format(
                " [{}]logical({},{})->phys({},{})", p, provider_logical[p].x, provider_logical[p].y, phys.x, phys.y);
            provider_xy_packed.push_back(((uint32_t)phys.x << 16) | ((uint32_t)phys.y & 0xFFFFu));
            // Pages this provider serves: q where q % N == p, packed in (q/N) order.
            std::vector<uint32_t> prov_data;
            for (uint32_t q = p; q < test_config.num_pages; q += n_providers) {
                prov_data.insert(
                    prov_data.end(),
                    shared_input.begin() + q * page_words,
                    shared_input.begin() + (q + 1) * page_words);
            }
            TT_FATAL(
                provider_base + prov_data.size() * sizeof(uint32_t) <= prov_l1_addr + prov_l1_size,
                "provider resident data exceeds L1");
            detail::WriteToDeviceL1(device, provider_logical[p], provider_base, prov_data);
        }
        log_info(
            tt::LogTest,
            "multi_interleaved read: L1-provider mode, N={} providers (reserved leftmost column), {} readers",
            n_providers,
            num_cores);
        log_info(tt::LogTest, "  provider cores:{}", prov_coord_str);
        // Diagnostic: enumerate DRAM channels and their NoC coords to compute the per-link-column split.
        std::string dram_str;
        int ndram = device->num_dram_channels();
        for (int ch = 0; ch < ndram; ++ch) {
            CoreCoord dl = device->logical_core_from_dram_channel(ch);
            CoreCoord dv = device->virtual_core_from_logical_core(dl, CoreType::DRAM);
            dram_str += fmt::format(" ch{}:logical({},{})->noc({},{})", ch, dl.x, dl.y, dv.x, dv.y);
        }
        log_info(tt::LogTest, "  DRAM channels ({} total):{}", ndram, dram_str);
        // Diagnostic: per-logical-column/row translation for kernel NoC coord (worker_core_from_logical)
        // vs profiler/heatmap coord (virtual_core_from_logical, WORKER), to resolve coordinate mismatches.
        {
            std::vector<CoreCoord> full = corerange_to_cores(test_config.cores);
            std::map<uint32_t, std::pair<uint32_t, uint32_t>> xmap, ymap;  // logical -> (worker.x/y, virtual.x/y)
            for (const auto& c : full) {
                CoreCoord w = device->worker_core_from_logical_core(c);
                CoreCoord v = device->virtual_core_from_logical_core(c, CoreType::WORKER);
                xmap[c.x] = {w.x, v.x};
                ymap[c.y] = {w.y, v.y};
            }
            std::string xs, ys;
            for (auto& [lx, p] : xmap) {
                xs += fmt::format(" L{}->W{}/V{}", lx, p.first, p.second);
            }
            for (auto& [ly, p] : ymap) {
                ys += fmt::format(" L{}->W{}/V{}", ly, p.first, p.second);
            }
            log_info(tt::LogTest, "  X map (logical->worker/virtual):{}", xs);
            log_info(tt::LogTest, "  Y map (logical->worker/virtual):{}", ys);
        }
    }

    // Kernels
    if (test_config.read_kernel) {
        reader_compile_args.push_back((uint32_t)n_providers);  // arg 17: l1_providers_n (0 = DRAM path)
        TensorAccessorArgs(*input_buffer).append_to(reader_compile_args);
        auto reader_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/data_movement/multi_interleaved/kernels/multi_interleaved_read.cpp",
            reader_cores,
            DataMovementConfig{
                // NOC0 (RISCV_0) is the multi_interleaved standard reader NoC; default_noc swaps to
                // RISCV_1/NOC1 (the tt-metal default reader NoC) to compare the return-path routing.
                .processor = test_config.default_noc ? DataMovementProcessor::RISCV_1 : DataMovementProcessor::RISCV_0,
                .noc = test_config.default_noc ? NOC::RISCV_1_default : NOC::RISCV_0_default,
                .compile_args = reader_compile_args});

        for (size_t i = 0; i < num_cores; ++i) {
            // Each core reads from different pages to distribute across DRAM banks
            uint32_t page_offset = i * test_config.num_pages;
            // Use the end of L1 data buffer as scratch space for polling
            uint32_t local_barrier_addr = l1_addrs[i] + per_core_size_bytes;

            std::vector<uint32_t> reader_run_time_args = {
                input_buffer_address,
                l1_addrs[i],
                page_offset,
                reader_barrier_sem_id,  // Semaphore ID, kernel will call get_semaphore() to get address
                coordinator_phys.x,
                coordinator_phys.y,
                num_cores,
                local_barrier_addr,
                (uint32_t)(inject_seed_rng() | 1u),  // per-core inject PRNG seed (nonzero), arg index 8
                stagger_thresholds[i],               // per-core stagger arg (prefix or delay cycles), arg index 9
                read_vc_per_core[i]};                // per-core static read VC (arg index 10; used iff override)

            // Per-core page-visit order, consumed by the reader kernel only when rand_mode != 0.
            // mode 1: rotated identity (random start, sequential order).
            // mode 2: random permutation (random order; kernel advances the start each iteration).
            if (rand_mode != 0) {
                std::vector<uint32_t> page_perm(test_config.num_pages);
                std::iota(page_perm.begin(), page_perm.end(), 0u);
                if (rand_mode == 1) {
                    uint32_t start = (uint32_t)(rand_offset_rng() % test_config.num_pages);
                    for (uint32_t p = 0; p < test_config.num_pages; p++) {
                        page_perm[p] = (p + start) % test_config.num_pages;
                    }
                } else {
                    std::shuffle(page_perm.begin(), page_perm.end(), rand_offset_rng);
                }
                reader_run_time_args.insert(reader_run_time_args.end(), page_perm.begin(), page_perm.end());
            }
            // L1-provider mode: base address (arg 11) + provider physical coords (arg 12..12+N-1), same for
            // every reader. Mutually exclusive with rand_mode above, so there is no arg-index collision.
            if (provider_mode) {
                reader_run_time_args.push_back(provider_base);
                reader_run_time_args.insert(
                    reader_run_time_args.end(), provider_xy_packed.begin(), provider_xy_packed.end());
            }
            tt::tt_metal::SetRuntimeArgs(program, reader_kernel, core_list[i], reader_run_time_args);
        }
    }

    if (test_config.write_kernel) {
        TensorAccessorArgs(*output_buffer).append_to(writer_compile_args);
        auto writer_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/data_movement/multi_interleaved/kernels/multi_interleaved_write.cpp",
            test_config.cores,
            DataMovementConfig{
                // NOC1 (RISCV_1) is the multi_interleaved standard writer NoC; default_noc swaps to
                // RISCV_0/NOC0 so reader and writer stay on distinct RISCs when the reader takes NOC1.
                .processor = test_config.default_noc ? DataMovementProcessor::RISCV_0 : DataMovementProcessor::RISCV_1,
                .noc = test_config.default_noc ? NOC::RISCV_0_default : NOC::RISCV_1_default,
                .compile_args = writer_compile_args});

        for (size_t i = 0; i < num_cores; ++i) {
            // Each core writes to different pages to distribute across DRAM banks
            uint32_t page_offset = i * test_config.num_pages;
            // Use the end of L1 data buffer as scratch space for polling
            uint32_t local_barrier_addr = l1_addrs[i] + per_core_size_bytes + noc_l1_alignment;

            std::vector<uint32_t> writer_run_time_args = {
                output_buffer_address,
                l1_addrs[i],
                page_offset,
                writer_barrier_sem_id,  // Semaphore ID, kernel will call get_semaphore() to get address
                coordinator_phys.x,
                coordinator_phys.y,
                num_cores,
                local_barrier_addr};
            tt::tt_metal::SetRuntimeArgs(program, writer_kernel, core_list[i], writer_run_time_args);
        }
    }

    // Assign unique id
    log_info(tt::LogTest, "Running Test ID: {}, Run ID: {}", test_config.test_id, unit_tests::dm::runtime_host_id);
    program.set_runtime_id(unit_tests::dm::runtime_host_id++);

    // Launch program and record outputs

    if (test_config.read_kernel) {
        detail::WriteToBuffer(input_buffer, packed_input);
        MetalContext::instance().get_cluster().dram_barrier(device->id());
    } else {
        // If not reading, write each core's slice to L1 directly
        const size_t per_core_words = per_core_size_bytes / sizeof(uint32_t);
        for (size_t i = 0; i < num_cores; ++i) {
            vector<uint32_t> core_input(
                packed_input.begin() + i * per_core_words, packed_input.begin() + (i + 1) * per_core_words);
            detail::WriteToDeviceL1(device, core_list[i], l1_addrs[i], core_input);
        }
        MetalContext::instance().get_cluster().l1_barrier(device->id());
    }

    auto mesh_workload = distributed::MeshWorkload();
    vector<uint32_t> coord_data = {0, 0};
    auto target_devices =
        distributed::MeshCoordinateRange(distributed::MeshCoordinate(coord_data));  // Single device at (0,0)
    mesh_workload.add_program(target_devices, std::move(program));

    auto& cq = mesh_device->mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, mesh_workload, false);
    Finish(cq);

    vector<uint32_t> packed_output;
    bool is_equal = false;

    if (test_config.write_kernel) {
        detail::ReadFromBuffer(output_buffer, packed_output);
        is_equal = (packed_output == packed_golden);
        if (!is_equal) {
            log_error(tt::LogTest, "Equality Check failed");
            log_info(tt::LogTest, "Golden vector");
            print_vector<uint32_t>(packed_golden);
            log_info(tt::LogTest, "Output vector");
            print_vector<uint32_t>(packed_output);
        }
    } else {
        // Each core reads pages into L1; verify each reader's L1 against its golden slice. In provider mode
        // every reader reads the same shared pages, so the golden is shared_input for all of them.
        const size_t per_core_words = per_core_size_bytes / sizeof(uint32_t);
        for (size_t i = 0; i < num_cores; ++i) {
            detail::ReadFromDeviceL1(device, core_list[i], l1_addrs[i], per_core_size_bytes, packed_output);
            vector<uint32_t> core_golden = provider_mode ? shared_input
                                                         : vector<uint32_t>(
                                                               packed_golden.begin() + i * per_core_words,
                                                               packed_golden.begin() + (i + 1) * per_core_words);
            is_equal = (packed_output == core_golden);
            if (!is_equal) {
                log_error(tt::LogTest, "Equality Check failed for core {}", i);
                log_info(tt::LogTest, "Golden vector for core {}", i);
                print_vector<uint32_t>(core_golden);
                log_info(tt::LogTest, "Output vector");
                print_vector<uint32_t>(packed_output);
                return is_equal;
            }
        }
    }
    return is_equal;
}

void directed_ideal_test(
    const shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t test_case_id,
    CoreCoord mst_start_coord,
    CoreCoord mst_grid_size,
    bool read,
    bool write,
    bool default_noc = false) {
    // Physical Constraints
    auto [flit_size_bytes, max_transmittable_bytes, max_transmittable_flits] =
        tt::tt_metal::unit_tests::dm::compute_physical_constraints(mesh_device);

    // Parameters
    uint32_t page_size_bytes = 256 * flit_size_bytes;  // 1 packet = 16 kB for BH, 8 kB for WH
    uint32_t num_pages = 16;
    uint32_t num_of_transactions = 16;

    // Cores
    CoreCoord mst_end_coord =
        CoreCoord(mst_start_coord.x + mst_grid_size.x - 1, mst_start_coord.y + mst_grid_size.y - 1);
    CoreRangeSet core_range_set({CoreRange(mst_start_coord, mst_end_coord)});

    // Test config
    unit_tests::dm::multi_interleaved::MultiInterleavedConfig test_config = {
        .test_id = test_case_id,
        .num_of_transactions = num_of_transactions,
        .num_pages = num_pages,
        .page_size_bytes = page_size_bytes,
        .l1_data_format = DataFormat::Float16_b,
        .cores = core_range_set,
        .read_kernel = read,
        .write_kernel = write,
        .default_noc = default_noc};

    // Run
    EXPECT_TRUE(run_dm(mesh_device, test_config));
}

void packet_sizes_test(
    const shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t test_case_id,
    CoreCoord mst_start_coord,
    CoreCoord mst_grid_size,
    bool read,
    bool write,
    bool default_noc = false) {
    // Physical Constraints
    auto [flit_size_bytes, max_transmittable_bytes, max_transmittable_flits] =
        tt::tt_metal::unit_tests::dm::compute_physical_constraints(mesh_device);

    // Parameters
    uint32_t max_page_size_bytes = 256 * flit_size_bytes;  // 1 packet = 16 kB for BH, 8 kB for WH
    uint32_t max_num_pages = 256;
    uint32_t num_of_transactions = 1;
    uint32_t num_pages;

    // Cores
    CoreCoord mst_end_coord =
        CoreCoord(mst_start_coord.x + mst_grid_size.x - 1, mst_start_coord.y + mst_grid_size.y - 1);
    CoreRangeSet core_range_set({CoreRange(mst_start_coord, mst_end_coord)});

    for (uint32_t pages = 1; pages <= max_num_pages; pages *= 4) {
        if (pages > 16) {
            // avoid writing too large of a memory block at once, prefer to overwrite multiple times
            num_of_transactions = pages / 16;
            num_pages = 16;
        } else {
            num_pages = pages;
        }
        for (uint32_t page_size_bytes = flit_size_bytes; page_size_bytes <= max_page_size_bytes; page_size_bytes *= 2) {
            // Test config
            unit_tests::dm::multi_interleaved::MultiInterleavedConfig test_config = {
                .test_id = test_case_id,
                .num_of_transactions = num_of_transactions,
                .num_pages = num_pages,
                .page_size_bytes = page_size_bytes,
                .l1_data_format = DataFormat::Float16_b,
                .cores = core_range_set,
                .read_kernel = read,
                .write_kernel = write,
                .default_noc = default_noc};

            // Run
            EXPECT_TRUE(run_dm(mesh_device, test_config));
        }
    }
}

}  // namespace unit_tests::dm::multi_interleaved

/* ========== Full grid directed ideal ========== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementMultiInterleavedDirectedIdeal) {
    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->impl().get_device(0);

    uint32_t test_case_id = 110;
    CoreCoord mst_start_coord = {0, 0};
    CoreCoord mst_grid_size = {device->compute_with_storage_grid_size().x, device->compute_with_storage_grid_size().y};
    unit_tests::dm::multi_interleaved::directed_ideal_test(
        mesh_device, test_case_id, mst_start_coord, mst_grid_size, true, true);
}

/* ========== Full grid packet sizes sweep ========== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementMultiInterleavedSizes) {
    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->impl().get_device(0);

    uint32_t test_case_id = 111;
    CoreCoord mst_start_coord = {0, 0};
    CoreCoord mst_grid_size = {device->compute_with_storage_grid_size().x, device->compute_with_storage_grid_size().y};
    unit_tests::dm::multi_interleaved::packet_sizes_test(
        mesh_device, test_case_id, mst_start_coord, mst_grid_size, true, true);
}

/* ========== Full grid read kernel directed ideal ========== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementMultiInterleavedReadDirectedIdeal) {
    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->impl().get_device(0);

    uint32_t test_case_id = 112;
    CoreCoord mst_start_coord = {0, 0};
    CoreCoord mst_grid_size = {device->compute_with_storage_grid_size().x, device->compute_with_storage_grid_size().y};
    unit_tests::dm::multi_interleaved::directed_ideal_test(
        mesh_device, test_case_id, mst_start_coord, mst_grid_size, true, false);
}

/* ========== Full grid read kernel packet sizes sweep ========== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementMultiInterleavedReadSizes) {
    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->impl().get_device(0);

    uint32_t test_case_id = 113;
    CoreCoord mst_start_coord = {0, 0};
    CoreCoord mst_grid_size = {device->compute_with_storage_grid_size().x, device->compute_with_storage_grid_size().y};
    unit_tests::dm::multi_interleaved::packet_sizes_test(
        mesh_device, test_case_id, mst_start_coord, mst_grid_size, true, false);
}

/* ========== Full grid write kernel directed ideal ========== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementMultiInterleavedWriteDirectedIdeal) {
    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->impl().get_device(0);

    uint32_t test_case_id = 114;
    CoreCoord mst_start_coord = {0, 0};
    CoreCoord mst_grid_size = {device->compute_with_storage_grid_size().x, device->compute_with_storage_grid_size().y};
    unit_tests::dm::multi_interleaved::directed_ideal_test(
        mesh_device, test_case_id, mst_start_coord, mst_grid_size, false, true);
}

/* ========== Full grid write kernel packet sizes sweep ========== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementMultiInterleavedWriteSizes) {
    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->impl().get_device(0);

    uint32_t test_case_id = 115;
    CoreCoord mst_start_coord = {0, 0};
    CoreCoord mst_grid_size = {device->compute_with_storage_grid_size().x, device->compute_with_storage_grid_size().y};
    unit_tests::dm::multi_interleaved::packet_sizes_test(
        mesh_device, test_case_id, mst_start_coord, mst_grid_size, false, true);
}

/* ========== 2x2 CORE TESTS ========== */

TEST_F(GenericMeshDeviceFixture, TensixDataMovement2x2MultiInterleavedDirectedIdeal) {
    uint32_t test_case_id = 116;
    CoreCoord mst_start_coord = {0, 0};
    CoreCoord mst_grid_size = {2, 2};
    unit_tests::dm::multi_interleaved::directed_ideal_test(
        get_mesh_device(), test_case_id, mst_start_coord, mst_grid_size, true, true);
}

TEST_F(GenericMeshDeviceFixture, TensixDataMovement2x2MultiInterleavedSizes) {
    uint32_t test_case_id = 117;
    CoreCoord mst_start_coord = {0, 0};
    CoreCoord mst_grid_size = {2, 2};
    unit_tests::dm::multi_interleaved::packet_sizes_test(
        get_mesh_device(), test_case_id, mst_start_coord, mst_grid_size, true, true);
}

TEST_F(GenericMeshDeviceFixture, TensixDataMovement2x2MultiInterleavedReadDirectedIdeal) {
    uint32_t test_case_id = 118;
    CoreCoord mst_start_coord = {0, 0};
    CoreCoord mst_grid_size = {2, 2};
    unit_tests::dm::multi_interleaved::directed_ideal_test(
        get_mesh_device(), test_case_id, mst_start_coord, mst_grid_size, true, false);
}

TEST_F(GenericMeshDeviceFixture, TensixDataMovement2x2MultiInterleavedReadSizes) {
    uint32_t test_case_id = 119;
    CoreCoord mst_start_coord = {0, 0};
    CoreCoord mst_grid_size = {2, 2};
    unit_tests::dm::multi_interleaved::packet_sizes_test(
        get_mesh_device(), test_case_id, mst_start_coord, mst_grid_size, true, false);
}

TEST_F(GenericMeshDeviceFixture, TensixDataMovement2x2MultiInterleavedWriteDirectedIdeal) {
    uint32_t test_case_id = 120;
    CoreCoord mst_start_coord = {0, 0};
    CoreCoord mst_grid_size = {2, 2};
    unit_tests::dm::multi_interleaved::directed_ideal_test(
        get_mesh_device(), test_case_id, mst_start_coord, mst_grid_size, false, true);
}

TEST_F(GenericMeshDeviceFixture, TensixDataMovement2x2MultiInterleavedWriteSizes) {
    uint32_t test_case_id = 121;
    CoreCoord mst_start_coord = {0, 0};
    CoreCoord mst_grid_size = {2, 2};
    unit_tests::dm::multi_interleaved::packet_sizes_test(
        get_mesh_device(), test_case_id, mst_start_coord, mst_grid_size, false, true);
}

/* ========== 6x6 CORE TESTS ========== */

TEST_F(GenericMeshDeviceFixture, TensixDataMovement6x6MultiInterleavedDirectedIdeal) {
    uint32_t test_case_id = 122;
    CoreCoord mst_start_coord = {0, 0};
    CoreCoord mst_grid_size = {6, 6};
    unit_tests::dm::multi_interleaved::directed_ideal_test(
        get_mesh_device(), test_case_id, mst_start_coord, mst_grid_size, true, true);
}

TEST_F(GenericMeshDeviceFixture, TensixDataMovement6x6MultiInterleavedSizes) {
    uint32_t test_case_id = 123;
    CoreCoord mst_start_coord = {0, 0};
    CoreCoord mst_grid_size = {6, 6};
    unit_tests::dm::multi_interleaved::packet_sizes_test(
        get_mesh_device(), test_case_id, mst_start_coord, mst_grid_size, true, true);
}

TEST_F(GenericMeshDeviceFixture, TensixDataMovement6x6MultiInterleavedReadDirectedIdeal) {
    uint32_t test_case_id = 124;
    CoreCoord mst_start_coord = {0, 0};
    CoreCoord mst_grid_size = {6, 6};
    unit_tests::dm::multi_interleaved::directed_ideal_test(
        get_mesh_device(), test_case_id, mst_start_coord, mst_grid_size, true, false);
}

TEST_F(GenericMeshDeviceFixture, TensixDataMovement6x6MultiInterleavedReadSizes) {
    uint32_t test_case_id = 125;
    CoreCoord mst_start_coord = {0, 0};
    CoreCoord mst_grid_size = {6, 6};
    unit_tests::dm::multi_interleaved::packet_sizes_test(
        get_mesh_device(), test_case_id, mst_start_coord, mst_grid_size, true, false);
}

TEST_F(GenericMeshDeviceFixture, TensixDataMovement6x6MultiInterleavedWriteDirectedIdeal) {
    uint32_t test_case_id = 126;
    CoreCoord mst_start_coord = {0, 0};
    CoreCoord mst_grid_size = {6, 6};
    unit_tests::dm::multi_interleaved::directed_ideal_test(
        get_mesh_device(), test_case_id, mst_start_coord, mst_grid_size, false, true);
}

TEST_F(GenericMeshDeviceFixture, TensixDataMovement6x6MultiInterleavedWriteSizes) {
    uint32_t test_case_id = 127;
    CoreCoord mst_start_coord = {0, 0};
    CoreCoord mst_grid_size = {6, 6};
    unit_tests::dm::multi_interleaved::packet_sizes_test(
        get_mesh_device(), test_case_id, mst_start_coord, mst_grid_size, false, true);
}

// NOC1-swap counterpart of test 125 (6x6 Read Sizes). Same read-only packet-size sweep over the
// full 6x6 block at (0,0), but the reader runs on RISCV_1/NOC1 instead of RISCV_0/NOC0. NOC1 is the
// coordinate mirror of NOC0 on Blackhole and routes the opposite way around the torus, so the heavy
// DRAM->L1 return streams traverse a different set of links. Compare against test 125 to see the
// effect of the return-path routing on aggregate read bandwidth.
TEST_F(GenericMeshDeviceFixture, TensixDataMovement6x6MultiInterleavedReadSizesNoc1) {
    uint32_t test_case_id = 128;
    CoreCoord mst_start_coord = {0, 0};
    CoreCoord mst_grid_size = {6, 6};
    unit_tests::dm::multi_interleaved::packet_sizes_test(
        get_mesh_device(), test_case_id, mst_start_coord, mst_grid_size, true, false, /*default_noc=*/true);
}

// ====================== Flag-driven read grid sweep; Test id = 129 ======================
// Read-only packet-size sweep over a runtime-configurable core block, so one build can cover
// any topology (row, column, or square/rectangular block) without recompiling the test list:
//   TT_DM_GRID_COLS = cores along x (width of a row); default = full compute grid x
//   TT_DM_GRID_ROWS = cores along y (height of a column); default = full compute grid y
// A row sweep is COLS=k, ROWS=1; a column sweep is COLS=1, ROWS=k; an NxN block is COLS=ROWS=k.
// Both dims are clamped to the harvested grid. Enable the t0..t3 phase counters with
// TT_DM_PHASE_COUNTERS=1. Uses the standard read-only methodology (NOC0/RISCV0).
TEST_F(GenericMeshDeviceFixture, TensixDataMovementMultiInterleavedReadGridSweep) {
    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->impl().get_device(0);
    auto grid = device->compute_with_storage_grid_size();

    uint32_t cols = unit_tests::dm::env_uint("TT_DM_GRID_COLS", (uint32_t)grid.x);
    uint32_t rows = unit_tests::dm::env_uint("TT_DM_GRID_ROWS", (uint32_t)grid.y);
    cols = std::clamp(cols, 1u, (uint32_t)grid.x);
    rows = std::clamp(rows, 1u, (uint32_t)grid.y);
    // TT_DM_DEFAULT_NOC=1 swaps the reader to RISCV1/NOC1 (coordinate mirror, routes the opposite way
    // around the torus) to compare directional return-path routing against the default NOC0/RISCV0.
    bool default_noc = unit_tests::dm::env_uint("TT_DM_DEFAULT_NOC", 0u) != 0u;
    log_info(
        tt::LogTest,
        "Read grid sweep: {} cols x {} rows ({} cores) on grid {}x{}",
        cols,
        rows,
        cols * rows,
        grid.x,
        grid.y);

    uint32_t test_case_id = 129;
    CoreCoord mst_start_coord = {0, 0};
    CoreCoord mst_grid_size = {cols, rows};
    unit_tests::dm::multi_interleaved::packet_sizes_test(
        get_mesh_device(), test_case_id, mst_start_coord, mst_grid_size, true, false, default_noc);
}

}  // namespace tt::tt_metal
