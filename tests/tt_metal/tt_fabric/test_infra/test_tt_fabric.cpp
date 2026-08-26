// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/reflection.hpp>
#include <vector>
#include <string>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <random>
#include <filesystem>
#include <optional>
#include <iomanip>
#include <sstream>
#include <memory>
#include <thread>

#include "tt_fabric_test_context.hpp"
#include "tt_fabric_test_constants.hpp"

#include "tt_metal/impl/context/metal_context.hpp"
#include <llrt/tt_cluster.hpp>
#include <llrt/hal.hpp>

using tt::tt_fabric::fabric_tests::DEFAULT_BUILT_TESTS_DUMP_FILE;
using tt::tt_fabric::fabric_tests::OUTPUT_DIR;

const std::unordered_map<Topology, FabricConfig> TestFixture::topology_to_fabric_config_map = {
    {Topology::NeighborExchange, FabricConfig::FABRIC_1D_NEIGHBOR_EXCHANGE},
    {Topology::Linear, FabricConfig::FABRIC_1D},
    {Topology::Ring, FabricConfig::FABRIC_1D_RING},
    {Topology::Mesh, FabricConfig::FABRIC_2D},
};

const std::unordered_map<std::pair<Topology, std::string>, FabricConfig, tt::tt_fabric::fabric_tests::pair_hash>
    TestFixture::torus_topology_to_fabric_config_map = {
        {{Topology::Torus, "X"}, FabricConfig::FABRIC_2D_TORUS_X},
        {{Topology::Torus, "Y"}, FabricConfig::FABRIC_2D_TORUS_Y},
        {{Topology::Torus, "XY"}, FabricConfig::FABRIC_2D_TORUS_XY},
};

namespace {

// Host-side ethernet link monitor. While a test runs, a background thread polls the PCS_STATUS debug
// register of every active ethernet core and logs up<->down transitions, giving a host-side timeline
// of link drops to correlate with the device-side watcher ring buffer (ETH_LINK_DOWN_RING_BUF_CODE).
//
// Opt-in via TT_FABRIC_MONITOR_ETH_LINKS, since the extra register reads can perturb the bandwidth
// numbers this perf benchmark measures. The thread lives entirely inside this process: killing the
// test (Ctrl-C / SIGKILL) tears it down with the process, so it can never leak or hold the chip.
class EthLinkMonitor {
public:
    // Reads PCS_STATUS the same way the device's is_link_up() does (eth_fw_api.h): value 1 == link up.
    static constexpr uint32_t PCS_STATUS_LINK_UP = 1;

    void start() {
        if (getenv("TT_FABRIC_MONITOR_ETH_LINKS") == nullptr) {
            return;
        }
        stop_.store(false);
        thread_ = std::thread([this] { run(); });
    }

    void stop_and_join() {
        if (thread_.joinable()) {
            stop_.store(true);
            thread_.join();
        }
    }

    ~EthLinkMonitor() { stop_and_join(); }

private:
    struct MonitoredCore {
        tt::ChipId chip_id;
        CoreCoord virtual_core;
        CoreCoord logical_core;
    };

    void run() {
        auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
        const auto& hal = tt::tt_metal::MetalContext::instance().hal();
        const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();

        if (!hal.get_supports_eth_debug_regs()) {
            log_warning(tt::LogTest, "EthLinkMonitor: PCS_STATUS debug reg unsupported on this arch; disabled.");
            return;
        }
        const uint32_t pcs_status_addr = hal.get_eth_debug_reg_addr(tt::tt_metal::EthDebugReg::PCS_STATUS);

        // Enumerate active eth cores once and seed state as "unknown" so the first sample isn't logged.
        std::vector<MonitoredCore> cores;
        for (auto chip_id : cluster.all_chip_ids()) {
            for (const auto& logical_core : control_plane.get_active_ethernet_cores(chip_id)) {
                const auto virtual_core =
                    cluster.get_virtual_coordinate_from_logical_coordinates(chip_id, logical_core, tt::CoreType::ETH);
                cores.push_back({chip_id, virtual_core, logical_core});
            }
        }
        log_info(tt::LogTest, "EthLinkMonitor: polling PCS_STATUS on {} active ethernet core(s).", cores.size());

        std::vector<int> prev_up(cores.size(), -1);  // -1 unknown, 1 up, 0 down
        int rounds = 0;
        while (!stop_.load()) {
            // [SLOT DUMP] Periodically mirror the ERISC L1 debug slot into the test log. Done from this
            // thread because it already holds an open cluster: an out-of-process dump has to construct its
            // own UMD cluster, and that runs topology discovery, which blocks up to 900s waiting for eth
            // training whenever a link is still down -- i.e. precisely the state we are trying to capture.
            // Emits the same "SLOT <dev> <chan> w0..w15 hb0 hb1" format as the teardown dump.
            if ((rounds++ % kSlotDumpEveryRounds) == 0) {
                dump_slots(cores);
            }
            for (size_t i = 0; i < cores.size(); ++i) {
                uint32_t pcs_status = 0;
                try {
                    cluster.read_reg(
                        &pcs_status, tt_cxy_pair(cores[i].chip_id, cores[i].virtual_core), pcs_status_addr);
                } catch (...) {
                    continue;  // Core temporarily unreachable (e.g. remote over a downed link); skip this round.
                }
                const int up = (pcs_status == PCS_STATUS_LINK_UP) ? 1 : 0;
                if (prev_up[i] != -1 && up != prev_up[i]) {
                    log_warning(
                        tt::LogTest,
                        "ETH LINK {}: device {} eth core {} (logical {}), PCS_STATUS={:#x}",
                        up ? "UP (recovered)" : "DOWN",
                        cores[i].chip_id,
                        cores[i].virtual_core.str(),
                        cores[i].logical_core.str(),
                        pcs_status);
                }
                prev_up[i] = up;
            }
            // Sleep in small slices so stop_ is observed promptly.
            for (int slept_ms = 0; slept_ms < kPollIntervalMs && !stop_.load(); slept_ms += kSleepSliceMs) {
                std::this_thread::sleep_for(std::chrono::milliseconds(kSleepSliceMs));
            }
        }
    }

    // Dump the debug slot every ~30s (kPollIntervalMs * 150).
    static constexpr int kSlotDumpEveryRounds = 150;
    static constexpr uint64_t kDbgSlotBase =
        0x6F1F8;  // dev_mem_map.h MEM_AERISC_RESUME_PHASE_BASE -- MUST track it: the region grows
                  // DOWNWARD from MEM_ERISC_FABRIC_ROUTER_RESERVED_BASE, so changing
                  // MEM_AERISC_RESUME_PHASE_SIZE MOVES this base. (SIZE=96 -> base 0x6F200.)
    // 24 words: 16 original + 16/17 send-gate, 18/19 sync tx/rx, 20/21 slot, 22 sync-seen, 23 min-free.
    static constexpr std::size_t kDbgSlotWords = 26;
    static constexpr uint64_t kErisc0Heartbeat = 0x7CC70;

    void dump_slots(const std::vector<MonitoredCore>& cores) {
        auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
        for (const auto& c : cores) {
            std::vector<uint32_t> words(kDbgSlotWords, 0);
            std::vector<uint32_t> hb(2, 0);
            try {
                cluster.read_core(
                    words, kDbgSlotWords * sizeof(uint32_t), tt_cxy_pair(c.chip_id, c.virtual_core), kDbgSlotBase);
                cluster.read_core(hb, 2 * sizeof(uint32_t), tt_cxy_pair(c.chip_id, c.virtual_core), kErisc0Heartbeat);
            } catch (...) {
                continue;
            }
            std::string line = fmt::format("SLOT {} {}", c.chip_id, c.logical_core.y);
            for (auto w : words) {
                line += fmt::format(" 0x{:x}", w);
            }
            line += fmt::format(" 0x{:x} 0x{:x}", hb[0], hb[1]);
            log_info(tt::LogTest, "{}", line);
        }
    }

    static constexpr int kPollIntervalMs = 200;
    static constexpr int kSleepSliceMs = 20;

    std::atomic<bool> stop_{false};
    std::thread thread_;
};

// Teardown dump of the ERISC debug slot for every active ethernet core, using the test's own cluster
// (no second process needed). Emits the same "SLOT <dev> <chan> w0..w15 hb0 hb1" lines the external
// `run_link_control dump_dbg_slot` produces, so one parser handles both.
void dump_erisc_debug_slots_from_host() {
    constexpr uint64_t DBG_SLOT_BASE =
        0x6F1F8;  // dev_mem_map.h MEM_AERISC_RESUME_PHASE_BASE -- MUST track it: the
                  // region grows DOWNWARD from MEM_ERISC_FABRIC_ROUTER_RESERVED_BASE, so
                  // changing MEM_AERISC_RESUME_PHASE_SIZE MOVES this base. (SIZE=96 -> 0x6F200.)
    constexpr std::size_t DBG_SLOT_WORDS = 26;  // incl. 16/17 send-gate, 18/19 tx/rx, 20/21 slot, 22 seen, 23 min-free
    constexpr uint64_t ERISC0_HEARTBEAT = 0x7CC70;

    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();

    log_info(tt::LogTest, "ERISC debug slot dump (teardown): base 0x{:X}, {} words", DBG_SLOT_BASE, DBG_SLOT_WORDS);
    for (auto chip_id : cluster.all_chip_ids()) {
        for (const auto& logical_core : control_plane.get_active_ethernet_cores(chip_id)) {
            const auto virtual_core =
                cluster.get_virtual_coordinate_from_logical_coordinates(chip_id, logical_core, tt::CoreType::ETH);
            std::vector<uint32_t> words(DBG_SLOT_WORDS, 0);
            std::vector<uint32_t> hb(2, 0);
            try {
                cluster.read_core(
                    words, DBG_SLOT_WORDS * sizeof(uint32_t), tt_cxy_pair(chip_id, virtual_core), DBG_SLOT_BASE);
                cluster.read_core(hb, 2 * sizeof(uint32_t), tt_cxy_pair(chip_id, virtual_core), ERISC0_HEARTBEAT);
            } catch (...) {
                continue;
            }
            // [BASE SANITY] word[0] is the resume-phase code, always 0x5E5E_xxxx on a core where the
            // router has run. If it isn't, DBG_SLOT_BASE has drifted from MEM_AERISC_RESUME_PHASE_BASE
            // and every word below is shifted -- which silently produces plausible-looking garbage.
            // Warn once rather than let a misaligned dump be analysed as real data.
            static bool base_warned = false;
            if (!base_warned && words[0] != 0 && (words[0] & 0xFFFF0000u) != 0x5E5E0000u) {
                log_warning(
                    tt::LogTest,
                    "SLOT dump base looks WRONG: word[0]=0x{:x} on device {} core {} (expected 0x5E5Exxxx). "
                    "DBG_SLOT_BASE (0x{:X}) is probably out of sync with MEM_AERISC_RESUME_PHASE_BASE -- "
                    "all dumped words are shifted; do not trust this data.",
                    words[0],
                    chip_id,
                    logical_core.str(),
                    DBG_SLOT_BASE);
                base_warned = true;
            }
            std::string line = fmt::format("SLOT {} {}", chip_id, logical_core.y);
            for (auto w : words) {
                line += fmt::format(" 0x{:x}", w);
            }
            line += fmt::format(" 0x{:x} 0x{:x}", hb[0], hb[1]);
            log_info(tt::LogTest, "{}", line);
        }
    }
}

// [STREAM REG DUMP] Read the doorbell counter directly, from an INDEPENDENT vantage point.
//
// Every observation of this counter so far has come from a single source: the router's own
// get_ptr_val(). That is an uncached NOC_STREAM_READ_REG so it should be trustworthy, but it has
// never been cross-checked, and the worker's write to it has never been read back from anywhere at
// all -- we only know the write was correctly addressed and NIU-acked, which says nothing about the
// counter actually updating.
//
// Stream 22 is the sender-channel free-slots ("doorbell") stream. Two registers form the
// increment-on-write pair:
//     UPDATE    idx 270 -> 0xFFB40000 + 22*0x1000 + 270*4 = 0xFFB56438  (what the WORKER writes)
//     AVAILABLE idx 297 -> 0xFFB40000 + 22*0x1000 + 297*4 = 0xFFB564A4  (what the ROUTER reads)
// Hardware is meant to apply an UPDATE write to the AVAILABLE counter.
//
// Interpretation at the barrier hang (num_buffers == 32):
//   AVAILABLE == 32  -> counter genuinely never moved; the write is acked but not applied
//   AVAILABLE == 31  -> counter DID update and the router's own read is somehow stale -- a different bug
// Immune to the L1 data-cache staleness affecting the slot/header probes, since these are register
// reads over NOC, not L1 dereferences.
void dump_sender_credit_stream_regs() {
    constexpr uint64_t NOC_OVERLAY_START = 0xFFB40000;
    constexpr uint64_t STREAM_REG_SPACE = 0x1000;
    constexpr uint32_t SENDER_CREDITS_STREAM_ID = 22;
    constexpr uint32_t UPDATE_REG_IDX = 270;
    constexpr uint32_t AVAILABLE_REG_IDX = 297;

    const uint64_t update_addr =
        NOC_OVERLAY_START + SENDER_CREDITS_STREAM_ID * STREAM_REG_SPACE + (UPDATE_REG_IDX << 2);
    const uint64_t available_addr =
        NOC_OVERLAY_START + SENDER_CREDITS_STREAM_ID * STREAM_REG_SPACE + (AVAILABLE_REG_IDX << 2);

    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();

    log_info(
        tt::LogTest,
        "Stream {} credit regs: UPDATE=0x{:X} AVAILABLE=0x{:X} (host-side independent read)",
        SENDER_CREDITS_STREAM_ID,
        update_addr,
        available_addr);
    for (auto chip_id : cluster.all_chip_ids()) {
        for (const auto& logical_core : control_plane.get_active_ethernet_cores(chip_id)) {
            const auto virtual_core =
                cluster.get_virtual_coordinate_from_logical_coordinates(chip_id, logical_core, tt::CoreType::ETH);
            uint32_t upd = 0xDEADBEEF;
            uint32_t avail = 0xDEADBEEF;
            try {
                cluster.read_reg(&upd, tt_cxy_pair(chip_id, virtual_core), update_addr);
                cluster.read_reg(&avail, tt_cxy_pair(chip_id, virtual_core), available_addr);
            } catch (...) {
                continue;
            }
            log_info(
                tt::LogTest, "STREAMREG {} {} update=0x{:x} available=0x{:x}", chip_id, logical_core.y, upd, avail);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> input_args(argv, argv + argc);

    auto fixture = std::make_shared<TestFixture>();

    // Parse command line and YAML configurations
    CmdlineParser cmdline_parser(input_args);

    if (cmdline_parser.has_help_option()) {
        cmdline_parser.print_help();
        return 0;
    }
    std::vector<ParsedTestConfig> raw_test_configs;
    tt::tt_fabric::fabric_tests::AllocatorPolicies allocation_policies;
    std::optional<tt::tt_fabric::fabric_tests::PhysicalMeshConfig> physical_mesh_config = std::nullopt;
    bool use_dynamic_policies = true;  // Default to dynamic

    if (auto yaml_path = cmdline_parser.get_yaml_config_path()) {
        YamlConfigParser yaml_parser;
        auto parsed_yaml = yaml_parser.parse_file(yaml_path.value());
        raw_test_configs = std::move(parsed_yaml.test_configs);

        // Check if YAML explicitly provided allocation_policies
        if (parsed_yaml.allocation_policies.has_value()) {
            allocation_policies = parsed_yaml.allocation_policies.value();
            use_dynamic_policies = false;  // User provided explicit policies
        }

        if (parsed_yaml.physical_mesh_config.has_value()) {
            physical_mesh_config = parsed_yaml.physical_mesh_config;
        }
    } else {
        log_error(
            tt::LogTest,
            "No YAML config file path specified. Please use --test_config <file_path> to specify the test config. Use "
            "--help for more information.");
        return 1;
    }

    log_info(tt::LogTest, "Starting Test");

    fixture->init(physical_mesh_config);

    TestContext test_context;
    test_context.init(fixture, allocation_policies, use_dynamic_policies);

    test_context.set_show_workers(cmdline_parser.show_workers());

    // Configure progress monitoring from cmdline flags
    if (cmdline_parser.show_progress()) {
        ProgressMonitorConfig progress_config;
        progress_config.enabled = true;
        progress_config.granular = cmdline_parser.show_progress_detail();
        progress_config.poll_interval_seconds = cmdline_parser.get_progress_interval();
        progress_config.hung_threshold_seconds = cmdline_parser.get_hung_threshold();
        progress_config.hung_confirmation_rounds = cmdline_parser.get_hung_confirmation_rounds();
        progress_config.wait_on_hang = cmdline_parser.wait_on_hang();
        progress_config.summary_file = cmdline_parser.get_validation_summary_file();
        progress_config.detail_file = cmdline_parser.get_validation_detail_file();

        test_context.enable_progress_monitoring(progress_config);
    }

    bool has_bandwidth_tests = std::any_of(raw_test_configs.begin(), raw_test_configs.end(), [](const auto& config) {
        return config.performance_test_mode == PerformanceTestMode::BANDWIDTH;
    });

    // Initialize CSV file for bandwidth results if any of the configs have bandwidth test mode set
    if (has_bandwidth_tests) {
        test_context.initialize_bandwidth_results_csv_file();
    }

    bool has_latency_tests = std::any_of(raw_test_configs.begin(), raw_test_configs.end(), [](const auto& config) {
        return config.performance_test_mode == PerformanceTestMode::LATENCY;
    });

    // Initialize CSV file for latency results if any of the configs have latency test mode set
    if (has_latency_tests) {
        test_context.initialize_latency_results_csv_file();
    }

    cmdline_parser.apply_overrides(raw_test_configs);

    raw_test_configs = tt::tt_fabric::fabric_tests::expand_channel_trimming(std::move(raw_test_configs));

    if (raw_test_configs.empty()) {
        log_fatal(tt::LogTest, "No test configurations loaded or generated. Exiting.");
        return 1;
    }

    std::optional<uint32_t> master_seed = cmdline_parser.get_master_seed();
    if (!master_seed.has_value()) {
        master_seed = test_context.get_randomized_master_seed();
    }

    std::mt19937 gen(master_seed.value());

    // fixture is passed twice since it implements both interfaces
    // the builder object does the initial processing of the tests parsed from yaml/cmd line and tries to fill
    // any gaps/optionals/missing values
    TestConfigBuilder builder(*fixture, *fixture, gen);

    std::ofstream output_stream;
    bool dump_built_tests = cmdline_parser.dump_built_tests();
    if (dump_built_tests) {
        std::filesystem::path dump_file_dir =
            std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
            std::string(OUTPUT_DIR);
        if (!std::filesystem::exists(dump_file_dir)) {
            std::filesystem::create_directory(dump_file_dir);
        }

        std::string dump_file = cmdline_parser.get_built_tests_dump_file_name(DEFAULT_BUILT_TESTS_DUMP_FILE);
        std::filesystem::path dump_file_path = dump_file_dir / dump_file;
        output_stream.open(dump_file_path, std::ios::out | std::ios::trunc);

        // dump physical mesh first
        if (physical_mesh_config.has_value()) {
            YamlTestConfigSerializer::dump(physical_mesh_config.value(), output_stream);
        }

        // dump allocation policies second
        YamlTestConfigSerializer::dump(allocation_policies, output_stream);
    }

    bool device_opened = false;
    uint32_t tests_ran = 0;
    // Skip/execution accounting so the end-of-run summary can explain what actually executed
    // instead of only emitting a single "completed" line on success. See #48783.
    uint32_t groups_skipped_filter = 0;
    uint32_t groups_skipped_platform = 0;
    uint32_t groups_skipped_mesh_passthrough = 0;
    uint32_t groups_skipped_unsupported = 0;
    uint32_t groups_skipped_topology = 0;
    uint32_t tests_executed = 0;
    // Per-group outcome (name, status) for the end-of-run report, in execution order.
    std::vector<std::pair<std::string, std::string>> group_results;
    group_results.reserve(raw_test_configs.size());
    for (auto& test_config : raw_test_configs) {
        if (!cmdline_parser.check_filter(test_config, true)) {
            log_info(tt::LogTest, "Skipping Test Group: {} due to filter policy", test_config.name);
            groups_skipped_filter++;
            group_results.emplace_back(test_config.name, "SKIP:filter");
            continue;
        }
        if (builder.should_skip_test_on_platform(test_config)) {
            log_info(tt::LogTest, "Skipping Test Group: {} due to platform skip policy", test_config.name);
            groups_skipped_platform++;
            group_results.emplace_back(test_config.name, "SKIP:platform");
            continue;
        }
        if (builder.should_skip_test_for_disabled_mesh_passthrough(test_config)) {
            log_info(
                tt::LogTest,
                "Skipping Test Group: {} because sequential_mesh_passthrough requires "
                "TT_METAL_ENABLE_FABRIC_MESH_PASS_THROUGH=1",
                test_config.name);
            groups_skipped_mesh_passthrough++;
            group_results.emplace_back(test_config.name, "SKIP:mesh_passthrough");
            continue;
        }
        log_info(tt::LogTest, "Running Test Group: {}", test_config.name);

        const auto& topology = test_config.fabric_setup.topology;
        const auto& fabric_tensix_config = test_config.fabric_setup.fabric_tensix_config.value();
        if (test_config.performance_test_mode != PerformanceTestMode::NONE) {
            tt::tt_metal::MetalContext::instance().rtoptions().set_enable_fabric_bw_telemetry(true);
        }

        if (test_config.fabric_setup.use_vc2) {
            tt::tt_metal::MetalContext::instance().rtoptions().set_enable_fabric_vc2(true);
        }

        log_info(
            tt::LogTest,
            "Opening devices with topology: {} and fabric_tensix_config: {}",
            topology,
            fabric_tensix_config);

        bool open_devices_success =
            test_context.open_devices(test_config.fabric_setup, test_config.channel_trimming_mode);
        if (!open_devices_success) {
            log_warning(
                tt::LogTest, "Skipping Test Group: {} due to unsupported fabric configuration", test_config.name);
            groups_skipped_unsupported++;
            group_results.emplace_back(test_config.name, "SKIP:unsupported_config");
            continue;
        }

        // Validate device frequencies for performance tests. Validation runs only once
        // since device frequencies are cached in TestFixture for its lifetime.
        if (test_config.performance_test_mode != PerformanceTestMode::NONE) {
            if (!fixture->validate_device_frequencies_for_performance_tests()) {
                test_context.close_devices();
                return 1;  // Hard exit - cannot run performance benchmarks with invalid frequencies
            }
        }

        // Check topology-based skip conditions after devices are opened
        if (builder.should_skip_test_on_topology(test_config)) {
            log_info(tt::LogTest, "Skipping Test Group: {} due to topology skip policy", test_config.name);
            test_context.close_devices();
            groups_skipped_topology++;
            group_results.emplace_back(test_config.name, "SKIP:topology");
            continue;
        }
        tests_ran++;
        device_opened = true;
        group_results.emplace_back(test_config.name, "RAN");

        for (uint32_t iter = 0; iter < test_config.num_top_level_iterations; ++iter) {
            log_info(tt::LogTest, "Starting top-level iteration {}/{}", iter + 1, test_config.num_top_level_iterations);

            log_info(tt::LogTest, "Building tests");
            auto built_tests = builder.build_tests({test_config}, cmdline_parser);

            // Enable telemetry for both benchmark and latency modes to ensure buffer clearing
            test_context.set_telemetry_enabled(test_config.performance_test_mode != PerformanceTestMode::NONE);
            // Set skip_packet_validation flag
            test_context.set_skip_packet_validation(test_config.skip_packet_validation);

            // Set code profiling enabled based on rtoptions
            auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
            test_context.set_code_profiling_enabled(rtoptions.get_enable_fabric_code_profiling_rx_ch_fwd());

            for (auto& built_test : built_tests) {
                log_info(tt::LogTest, "Running Test: {}", built_test.parametrized_name);
                tests_executed++;

                // Prepare allocator and memory maps for this specific test
                test_context.prepare_for_test(built_test);

                // Set performance test mode for each iteration
                test_context.set_performance_test_mode(built_test.performance_test_mode);

                test_context.setup_devices();
                log_info(tt::LogTest, "Device setup complete");

                test_context.process_traffic_config(built_test);
                log_info(tt::LogTest, "Traffic config processed");

                // Setup latency test mode AFTER process_traffic_config so that senders_/receivers_ maps are populated
                if (built_test.performance_test_mode == PerformanceTestMode::LATENCY) {
                    test_context.setup_latency_test_mode(built_test);
                }

                // Clear code profiling buffers before test execution
                if (test_context.get_code_profiling_enabled()) {
                    test_context.clear_code_profiling_buffers();
                }

                if (dump_built_tests) {
                    YamlTestConfigSerializer::dump({built_test}, output_stream);
                }

                log_info(tt::LogTest, "Compiling programs");
                test_context.compile_programs();

                // multi-host barrier to synchronize before starting the test (as we could be clearing out addresses)
                fixture->barrier();

                // Watch ethernet links for the duration of this test (opt-in via TT_FABRIC_MONITOR_ETH_LINKS).
                // Devices/fabric are up by now, so active eth cores can be enumerated and polled.
                EthLinkMonitor eth_link_monitor;
                eth_link_monitor.start();

                log_info(tt::LogTest, "Launching programs");
                test_context.launch_programs();

                log_info(tt::LogTest, "Waiting for programs");
                test_context.wait_for_programs_with_progress();

                eth_link_monitor.stop_and_join();

                // Teardown dump of the ERISC debug slot. Fires on runs that finish (cleanly or via the
                // hang detector); a hard-wedged run killed externally never gets here, which is why the
                // out-of-process `run_link_control dump_dbg_slot` exists as well.
                dump_erisc_debug_slots_from_host();

                dump_sender_credit_stream_regs();

                if (test_context.did_last_test_hang()) {
                    log_error(
                        tt::LogTest,
                        "Test {} HUNG - aborting test suite. System may be in a bad state.",
                        built_test.parametrized_name);
                    test_context.record_hung_test(built_test.parametrized_name);
                    test_context.reset_devices();
                    break;
                }

                log_info(tt::LogTest, "Test {} Finished.", built_test.parametrized_name);

                test_context.process_telemetry_data(built_test);

                // Read and report code profiling results
                if (test_context.get_code_profiling_enabled()) {
                    test_context.read_code_profiling_results();
                    test_context.report_code_profiling_results();
                }

                test_context.validate_results();

                // Performance profiling (bandwidth mode)
                if (test_context.get_performance_test_mode() == PerformanceTestMode::BANDWIDTH) {
                    test_context.profile_results(built_test);
                }

                // Latency measurement (latency test mode)
                if (test_context.get_performance_test_mode() == PerformanceTestMode::LATENCY) {
                    test_context.collect_latency_results();
                    test_context.report_latency_results(built_test);
                }

                if (test_context.get_telemetry_enabled()) {
                    test_context.clear_telemetry();
                }
                // Synchronize across all hosts after running the current test variant
                fixture->barrier();
                test_context.reset_devices();
            }
            if (test_context.did_last_test_hang()) {
                break;
            }
        }
        if (test_context.did_last_test_hang()) {
            break;
        }
    }

    test_context.close_devices();

    tt::tt_metal::MetalContext::instance().rtoptions().set_enable_fabric_bw_telemetry(false);

    // Generate summaries after all tests have run
    if (has_bandwidth_tests) {
        test_context.generate_bandwidth_summary();
    }
    if (has_latency_tests) {
        test_context.generate_latency_summary();
    }

    // Setup CSV files for CI to upload (handles both bandwidth and latency)
    if (has_bandwidth_tests || has_latency_tests) {
        test_context.setup_ci_artifacts();
    }

    // Always emit a run summary so a successful run is auditable -- how many groups ran, how
    // many were skipped and why, and how many individual tests executed -- rather than only a
    // single "completed" line. This also makes a run that executed nothing obvious. See #48783.
    const auto total_groups = raw_test_configs.size();
    const auto groups_skipped_total = groups_skipped_filter + groups_skipped_platform +
                                      groups_skipped_mesh_passthrough + groups_skipped_unsupported +
                                      groups_skipped_topology;
    const auto& failed_tests = test_context.get_all_failed_tests();

    log_info(tt::LogTest, "=========== TEST RUN SUMMARY ===========");
    log_info(tt::LogTest, "Test groups: {} total, {} ran, {} skipped", total_groups, tests_ran, groups_skipped_total);
    if (groups_skipped_total > 0) {
        log_info(
            tt::LogTest,
            "  skipped breakdown: filter={}, platform={}, mesh_passthrough={}, unsupported_config={}, topology={}",
            groups_skipped_filter,
            groups_skipped_platform,
            groups_skipped_mesh_passthrough,
            groups_skipped_unsupported,
            groups_skipped_topology);
    }
    log_info(tt::LogTest, "Per-group results:");
    for (const auto& [group_name, group_status] : group_results) {
        log_info(tt::LogTest, "  [{}] {}", group_status, group_name);
    }
    log_info(tt::LogTest, "Tests: {} executed, {} failed", tests_executed, failed_tests.size());
    if (!failed_tests.empty()) {
        log_error(tt::LogTest, "Failed tests:");
        for (const auto& failed_test : failed_tests) {
            log_error(tt::LogTest, "  - {}", failed_test);
        }
    }
    if (test_context.has_test_failures()) {
        log_error(tt::LogTest, "Result: FAILED - some tests failed golden comparison validation");
    } else if (device_opened) {
        // NOTE: keep this exact phrase. External harnesses (tools/scaleout/exabox/
        // run_fabric_tests.sh and analyze_fabric_results.py) count occurrences of
        // "All tests completed successfully" as the per-rank success marker.
        log_info(tt::LogTest, "All tests completed successfully");
    } else {
        log_warning(
            tt::LogTest, "Result: no test groups executed (all {} filtered, skipped, or unsupported)", total_groups);
    }
    log_info(tt::LogTest, "========================================");

    // Fail the run (non-zero exit) if any tests failed validation.
    if (test_context.has_test_failures()) {
        TT_THROW("Some tests failed golden comparison validation. See summary above.");
    }
    return 0;
}
