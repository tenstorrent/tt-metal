// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Renders one manifest configuration's records per member, and one per
// packet_injection tensix_to_x280 mode and family, through that
// member's own record function and the shared writer, then has the live
// tabulator validate them (tests/host/record_round_trip.py): the campaign
// validator for the configuration, the dispatch mode on every record, and the
// same result tables, holding the expected cells, from
// table mode and the campaign report. Inputs are synthetic but carry the
// configuration's counts, so a record that drops or renames a field the
// tabulator reads fails here rather than in a campaign.

#include "harness/measurement/measurement.hpp"
#include "harness/output/output.hpp"
#include "tests/device/suites/core_sync/records.hpp"
#include "tests/device/suites/eth_credit/records.hpp"
#include "tests/device/suites/p2p_latency/metal_peer.hpp"
#include "tests/device/suites/packet_injection/records.hpp"
#include "tests/device/suites/pipeline_stage/records.hpp"
#include "tests/device/suites/transport_bandwidth/records.hpp"
#include "tests/host/scoped_env.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace {

namespace hout = x280::harness::output;
namespace hmeas = x280::harness::measurement;
namespace peer = x280::suites::p2p_latency::metal_peer;
namespace ps = x280::suites::pipeline_stage::common;
namespace pi = x280::suites::packet_injection;
using hmeas::Counter;
using x280::testing::ScopedEnv;

// AICLK evidence the tabulator accepts: two 10 ms endpoints 10 ms apart, each
// counting exactly 1 GHz inside 10 ns host brackets.
hmeas::AiclkRunEvidence clean_aiclk(Counter reference) {
    const auto endpoint = [reference](uint64_t host, uint64_t ticks) {
        return hmeas::AiclkEndpoint{
            .reference = reference,
            .reference_core = std::array<uint32_t, 2>{1, 2},
            .first = {host, ticks, host + 10},
            .second = {host + 10000000, ticks + 10000000, host + 10000010},
            .telemetry =
                {.host_before_ns = host - 20,
                 .host_after_ns = host - 10,
                 .limit_max_mhz = 1000,
                 .current_mhz = 1000,
                 .arb_max_raw = 1000,
                 .thermal_trip_count = 0,
                 .update_period_ms = 100},
        };
    };
    auto evidence = hmeas::assess_aiclk_run(1000000000, endpoint(100, 1000), endpoint(20000100, 20001000));
    EXPECT_TRUE(evidence.errors.empty()) << hout::clock_json(evidence);
    return evidence;
}

// One participant's two phases of `samples` exchanges, A initiating first.
p2p_latency_result_t p2p_result(uint32_t backend_id, uint32_t participant, uint32_t samples) {
    p2p_latency_result_t r{};
    r.backend_id = backend_id;
    r.path_sentinel = 0x201u;
    r.terminal_state = P2P_LATENCY_STATE_DONE;
    for (uint32_t p = 0; p < 2; ++p) {
        auto& ph = r.phases[p];
        ph.phase_ordinal = p;
        ph.direction = p == 0 ? P2P_LATENCY_DIRECTION_A_TO_B : P2P_LATENCY_DIRECTION_B_TO_A;
        const bool initiator = (participant == P2P_LATENCY_PARTICIPANT_A) == (p == 0);
        ph.role = initiator ? P2P_LATENCY_ROLE_INITIATOR : P2P_LATENCY_ROLE_RESPONDER;
        ph.sample_kind = initiator ? P2P_LATENCY_SAMPLE_KIND_RTT : P2P_LATENCY_SAMPLE_KIND_TURN;
        ph.samples_completed = samples;
        ph.terminal_state = P2P_LATENCY_STATE_DONE;
        for (uint32_t i = 0; i < samples; ++i) {
            ph.raw_deltas[i] = 300u + i;
            ph.secondary_deltas[i] = 20u + i;
        }
    }
    return r;
}

pipeline_stage_result_t stage_result(uint32_t backend_id, uint32_t timed) {
    pipeline_stage_result_t r{};
    r.backend_id = backend_id;
    r.terminal_state = PIPELINE_STAGE_STATE_DONE;
    r.packet_header_bytes = FS_PACKET_HEADER_BYTES;
    r.instrumentation = PIPELINE_STAGE_INSTRUMENTATION_PER_SAMPLE;
    r.warmup_completed = 128u;
    r.timed_count = timed;
    r.items_completed = timed;
    r.raw_batch_delta = 40000u;
    r.secondary_batch_delta = 70000u;
    r.verify_sentinel = PIPELINE_STAGE_VERIFY_MATCH;
    return r;
}

pipeline_stage_samples_t stage_samples(uint32_t timed) {
    pipeline_stage_samples_t s{};
    s.samples_completed = timed;
    for (uint32_t i = 0; i < timed; ++i) {
        s.raw_deltas[i] = 90u + i % 7u;
    }
    return s;
}

class RecordRoundTrip : public testing::Test {
protected:
    void SetUp() override {
        std::string pattern = (std::filesystem::temp_directory_path() / "x280-records-XXXXXX").string();
        ASSERT_NE(::mkdtemp(pattern.data()), nullptr) << std::strerror(errno);
        dir_ = pattern;
        output_.emplace("X280_CHARACTERIZATION_OUTPUT_DIR", dir_.c_str());
        dispatch_.emplace("TT_METAL_SLOW_DISPATCH_MODE", "1");
    }

    void TearDown() override {
        dispatch_.reset();
        output_.reset();
        std::error_code error;
        std::filesystem::remove_all(dir_, error);
        EXPECT_FALSE(error) << dir_ << ": " << error.message();
    }

    // Appends `spec`'s record to the file named for manifest configuration `id`.
    void write(const std::string& id, const hout::RecordSpec& spec) { hout::write_jsonl(id, hout::record(spec)); }

    // Runs the live tabulator over every file written; `files` is how many.
    void validate(size_t files) {
        const std::string command = "python3 '" X280_SOURCE_DIR "/tests/host/record_round_trip.py' '" + dir_.string() +
                                    "' '" X280_SOURCE_DIR "/characterization_manifest.json' " + std::to_string(files);
        EXPECT_EQ(std::system(command.c_str()), 0) << command;
    }

    std::filesystem::path dir_;
    std::optional<ScopedEnv> output_;
    std::optional<ScopedEnv> dispatch_;
};

}  // namespace

TEST_F(RecordRoundTrip, EveryMembersRecordsPassTheLiveTabulator) {
    constexpr uint32_t kP2pSamples = 64;
    const auto tensix_clock = clean_aiclk(Counter::TensixLiveTimer);
    const auto erisc_clock = clean_aiclk(Counter::EriscWallClock);
    // The selection of every member that calibrates when its configuration
    // sets no warm-up, as each one's manifest configuration leaves it.
    const hmeas::WarmupCalibrationRequest pilot{
        .pilot_device_ns = 100000,
        .pilot_iterations = 1024,
        .default_warmup = 64,
        .timed_iterations = 1024,
        .target_device_ns = hmeas::kAiclkTargetDeviceWorkNs,
        .max_warmup = UINT32_MAX,
        .max_device_ns = UINT64_MAX};
    const auto calibrated = hmeas::calibrate_warmup(pilot);

    // p2p_latency: four records per configuration, two participants by two phases.
    {
        const p2p_latency_common_args_t common{
            .first_direction = P2P_LATENCY_DIRECTION_A_TO_B, .payload_size = 4, .warmup_count = 8, .run_nonce = 0x201};
        const auto a = p2p_result(P2P_LATENCY_BACKEND_ID_X280_X280_SAME_TILE, P2P_LATENCY_PARTICIPANT_A, kP2pSamples);
        const auto b = p2p_result(P2P_LATENCY_BACKEND_ID_X280_X280_SAME_TILE, P2P_LATENCY_PARTICIPANT_B, kP2pSamples);
        for (uint32_t p = 0; p < 2; ++p) {
            for (const auto& [who, r] : {std::pair{"A", &a}, std::pair{"B", &b}}) {
                write(
                    "x280_same_tile.AInitiatesFirst",
                    peer::phase_record(
                        "x280_same_tile",
                        {{"participant", who},
                         {"chip_id", 0},
                         {"x280_hw_index", 0},
                         {"x280_noc_x", 8u},
                         {"x280_noc_y", 3u}},
                        *r,
                        p,
                        common,
                        std::nullopt,
                        std::nullopt,
                        P2P_LATENCY_TRANSMIT_WAIT_FOR_COMPLETION,
                        std::nullopt,
                        hmeas::WarmupSelection{.warmup = 8}));
            }
        }
        for (uint32_t p = 0; p < 2; ++p) {
            for (const auto& [who, r, hw] : {std::tuple{"A", &a, 0}, std::tuple{"B", &b, 1}}) {
                write(
                    "x280_peer_tile.AInitiatesFirst",
                    peer::phase_record(
                        "x280_peer_tile",
                        {{"participant", who},
                         {"chip_id", 0},
                         {"hw_index", hw},
                         {"peer_hw_index", 1 - hw},
                         {"x280_noc_x", 8u},
                         {"x280_noc_y", hw == 0 ? 3u : 9u},
                         {"peer_x280_noc_x", 8u},
                         {"peer_x280_noc_y", hw == 0 ? 9u : 3u},
                         {"x280_noc", hout::kX280Noc},
                         {"hops_a_to_b", 6u},
                         {"hops_b_to_a", 4u}},
                        *r,
                        p,
                        common,
                        0,
                        P2P_LATENCY_X280_NOC_ORDERING,
                        P2P_LATENCY_TRANSMIT_WAIT_FOR_COMPLETION,
                        std::nullopt,
                        hmeas::WarmupSelection{.warmup = 8}));
            }
        }
        const auto x280 =
            p2p_result(P2P_LATENCY_BACKEND_ID_X280_METAL_PEER_TILE, P2P_LATENCY_PARTICIPANT_A, kP2pSamples);
        for (const auto& [id, member, name, backend, counter, clock, prefix, noc0] :
             {std::tuple{
                  "tensix_peer_tile.X280InitiatesFirst",
                  "tensix_peer_tile",
                  "Tensix",
                  P2P_LATENCY_BACKEND_ID_TENSIX_TENSIX_PEER_TILE,
                  Counter::TensixLiveTimer,
                  &tensix_clock,
                  "tensix",
                  std::array<uint32_t, 2>{1, 2}},
              std::tuple{
                  "erisc_peer_tile.X280InitiatesFirst",
                  "erisc_peer_tile",
                  "ERISC",
                  P2P_LATENCY_BACKEND_ID_ERISC_ERISC_PEER_TILE,
                  Counter::EriscWallClock,
                  &erisc_clock,
                  "erisc",
                  std::array<uint32_t, 2>{10, 1}},
              std::tuple{
                  "drisc_peer_tile.X280InitiatesFirst",
                  "drisc_peer_tile",
                  "DRISC",
                  P2P_LATENCY_BACKEND_ID_DRISC_DRISC_PEER_TILE,
                  Counter::RiscDebugWallClock,
                  &tensix_clock,
                  "drisc",
                  std::array<uint32_t, 2>{0, 0}}}) {
            // D1's card 0 placement: the X280 at NoC0 (8, 3), each peer at `noc0`, every leg on NoC 0.
            std::vector<hout::Field> placement{
                {"chip_id", 0},
                {"x280_hw_index", 0},
                {"x280_noc_x", 8u},
                {"x280_noc_y", 3u},
                {"x280_noc", hout::kX280Noc}};
            hout::append_placement(placement, prefix, {.logical = noc0, .virt = noc0, .noc0 = noc0, .noc = 0u});
            const auto identity = [&](const char* who) {
                std::vector<hout::Field> fields{{"participant", who}};
                fields.insert(fields.end(), placement.begin(), placement.end());
                return fields;
            };
            // Departure TURN samples are 100 ticks below completion's.
            auto departure = p2p_result(backend, P2P_LATENCY_PARTICIPANT_B, kP2pSamples);
            for (uint32_t i = 0; i < kP2pSamples; ++i) {
                departure.phases[0].raw_deltas[i] -= 100u;
            }
            for (const auto& [config, metal, tx_mode] :
                 {std::tuple{
                      std::string(id),
                      p2p_result(backend, P2P_LATENCY_PARTICIPANT_B, kP2pSamples),
                      P2P_LATENCY_TRANSMIT_WAIT_FOR_COMPLETION},
                  std::tuple{std::string(id) + ".departure", departure, P2P_LATENCY_TRANSMIT_WAIT_FOR_DEPARTURE}}) {
                for (uint32_t p = 0; p < 2; ++p) {
                    write(
                        config,
                        peer::phase_record(
                            member,
                            identity("X280"),
                            x280,
                            p,
                            common,
                            0,
                            P2P_LATENCY_X280_NOC_ORDERING,
                            P2P_LATENCY_TRANSMIT_WAIT_FOR_COMPLETION,
                            std::nullopt,
                            calibrated));
                    write(
                        config,
                        peer::phase_record(
                            member,
                            identity(name),
                            metal,
                            p,
                            common,
                            0,
                            std::nullopt,
                            tx_mode,
                            peer::MetalClock{counter, *clock},
                            calibrated));
                }
            }
        }
    }

    // transport_bandwidth: one record per point.
    {
        transport_bw_result_t r{};
        r.raw_batch_delta = 200000;
        r.secondary_batch_delta = 350000;
        const x280::suites::transport_bandwidth::Point point{
            .transfer_size = 64, .boundary = TRANSPORT_BW_BOUNDARY_DRAINED, .warmup = 128, .samples = 16384};
        transport_bw_x280_to_tensix_result_t hart0{.common = r, .wall_start = 1000, .wall_end = 201000};
        write(
            "bw.x280_to_tensix.64.retired",
            x280::suites::transport_bandwidth::x280_to_tensix_record({hart0}, point, /*store_width=*/4));
        // Hart 1 starts 100 ticks after hart 0 and ends 300 after it.
        transport_bw_x280_to_tensix_result_t hart1{.common = r, .wall_start = 1100, .wall_end = 201300};
        hart1.common.raw_batch_delta = 200200;
        write(
            "bw.x280_to_tensix.h2.64.retired",
            x280::suites::transport_bandwidth::x280_to_tensix_record({hart0, hart1}, point, /*store_width=*/4));
        write(
            "bw.tensix_to_x280.64.drained",
            x280::suites::transport_bandwidth::tensix_to_x280_record(r, point, calibrated, tensix_clock));
        // Two idle-ERISC writers, the second taking twice as long.
        transport_bw_result_t slow = r;
        slow.raw_batch_delta = 400000;
        x280::suites::transport_bandwidth::Point injection = point;
        injection.boundary = TRANSPORT_BW_BOUNDARY_INJECTION;
        write(
            "bw.erisc_to_x280.w2.64.injection",
            x280::suites::transport_bandwidth::erisc_to_x280_record(
                {r, slow}, {{0, 0}, {0, 1}}, injection, calibrated, erisc_clock, 1500));
    }

    // pipeline_stage: route_compute instrumented, and route_forward's one
    // configuration, decide_plain.
    {
        const auto x_samples = stage_samples(4096);
        const auto x_compute = stage_result(PIPELINE_STAGE_BACKEND_ID_X280_ROUTE_COMPUTE, 4096);
        write(
            "pipeline.x280.route_compute.direction_table_instrumented",
            ps::x280_route_compute_record(
                {.result = x_compute,
                 .run_nonce = 0xC01,
                 .warmup = 128,
                 .timed = 4096,
                 .instrumentation = PIPELINE_STAGE_INSTRUMENTATION_PER_SAMPLE,
                 .samples = &x_samples}));
        auto x_forward = stage_result(PIPELINE_STAGE_BACKEND_ID_X280_ROUTE_FORWARD, 16384);
        x_forward.instrumentation = PIPELINE_STAGE_INSTRUMENTATION_OFF;
        write(
            "pipeline.x280.route_forward.decide_plain",
            ps::x280_route_forward_record(
                {.result = x_forward,
                 .run_nonce = 0xF01,
                 .warmup = 128,
                 .timed = 16384,
                 .instrumentation = PIPELINE_STAGE_INSTRUMENTATION_OFF}));
    }

    // packet_injection: run, aggregate and one worker per configuration, two
    // for the paired family; tensix_to_x280 in each of its three modes.
    {
        pi_run_result_t run_res{};
        run_res.state = PI_PHASE_DONE;
        run_res.aggregate_start_mtime = 1000;
        run_res.aggregate_end_mtime = 51000;
        run_res.aggregate_start_rdcycle = 7000;
        run_res.aggregate_end_rdcycle = 1757000;
        pi_producer_result_t producer{};
        producer.throughput_start = 100;
        producer.throughput_end = 90100;
        producer.cpu_start = 200;
        producer.cpu_end = 120200;
        // At B=1 every header is its own counter write: N writes.
        producer.measured_counter_writes = 1000;
        const pi_consumer_slot_t slot{};
        const pi::LimSlotDump unread{};
        const pi_layout_t layout{};
        pi_layout_t paired_layout{};
        paired_layout.risc_shift = 1;
        paired_layout.risc_mask = 1;
        const pi::Worker worker{.index = 0, .slot = slot, .lim_dump = unread, .producer = producer};
        const auto config = [](uint32_t mode, uint32_t W, uint64_t N) {
            x280::harness::regions::packet_injection::Config cfg{};
            cfg.mode = mode;
            cfg.S = 16;
            cfg.B = 1;
            cfg.C = 1;
            cfg.K = 1;
            cfg.W = W;
            cfg.N = N;
            cfg.ordered_tiles = {{0, 0}};
            return cfg;
        };
        // Each run's value is its own, as minted: the tabulator joins a run's records by it.
        uint64_t run_value = 0x50494E4A00000011ull;
        const auto run =
            [&](const x280::harness::regions::packet_injection::Config& cfg, const pi_layout_t& at, uint32_t workers) {
                return pi::Run{
                    .run_value = run_value++,
                    .cfg = cfg,
                    .layout = at,
                    .tile_count = 1,
                    .worker_count = workers,
                    .mtime_hz = 50000000,
                    .result = run_res,
                    .successful = true,
                    .report_successful = true,
                    .tile = {.chip_id = 0, .hw_index = 0, .noc_x = 8, .noc_y = 3}};
            };
        // One tensix_to_x280 configuration's run, aggregate and worker records,
        // worker w taking samples[w].
        const auto write_tensix =
            [&](const char* id, const pi::Run& r, const std::vector<pi::TensixWorkerSamples>& samples) {
                const auto identity = pi::tensix_to_x280_identity(r, {.bridge_history = "unknown"});
                const hmeas::WarmupSelection warmup{.warmup = r.cfg.W};
                write(id, pi::run_record("tensix_to_x280", identity, "run", {}, std::nullopt, tensix_clock, warmup));
                write(id, pi::aggregate_record("tensix_to_x280", identity, r, warmup));
                for (uint32_t w = 0; w < samples.size(); ++w) {
                    const pi::Worker each{.index = w, .slot = slot, .lim_dump = unread, .producer = producer};
                    write(
                        id,
                        pi::run_record(
                            "tensix_to_x280",
                            identity,
                            "worker",
                            pi::tensix_to_x280_worker(
                                r,
                                each,
                                {.logical = std::array<uint32_t, 2>{0, 0},
                                 .virt = std::array<uint32_t, 2>{1, 2 + w},
                                 .noc0 = std::array<uint32_t, 2>{1, 2 + w},
                                 .noc = w},
                                samples[w]),
                            Counter::TensixLiveTimer,
                            tensix_clock,
                            warmup));
                }
            };

        const auto throughput_cfg = config(PI_MODE_THROUGHPUT, 1000, 1000);
        write_tensix("packet.throughput.brisc.T1", run(throughput_cfg, layout, 1), {{}});
        auto paired_cfg = config(PI_MODE_THROUGHPUT, 1000, 1000);
        paired_cfg.family = x280::harness::regions::packet_injection::WorkerFamily::Paired;
        write_tensix("packet.throughput.paired.T1", run(paired_cfg, paired_layout, 2), {{}, {}});
        const auto latency_cfg = config(PI_MODE_LATENCY, 100, 3);
        write_tensix("packet.latency-include.brisc.T1", run(latency_cfg, layout, 1), {{.latency = {360, 300, 330}}});
        const auto backpressure_cfg = config(PI_MODE_BACKPRESSURE, 100, 1000);
        write_tensix(
            "packet.backpressure-include.brisc.T1",
            run(backpressure_cfg, layout, 1),
            {{.credit_stall_cpu_cycles = {7, 4, 6, 5},
              .stall_triggered_rtt_cpu_cycles = {71, 41, 61, 51},
              .credit_stalls = {70, 40, 60, 50},
              .stall_triggered_rtt = {700, 400, 600, 500}}});

        const auto erisc_cfg = config(PI_MODE_THROUGHPUT, static_cast<uint32_t>(calibrated.warmup), 1024);
        producer.measured_counter_writes = 1024;
        const auto erisc = run(erisc_cfg, layout, 1);
        const auto erisc_identity =
            pi::erisc_to_x280_header_identity(erisc, {.writers_requested = 1, .writers = {{0, 8}}});
        const char* erisc_id = "packet.erisc_header.writers1.b1";
        write(
            erisc_id,
            pi::run_record("erisc_to_x280_header", erisc_identity, "run", {}, std::nullopt, erisc_clock, calibrated));
        write(erisc_id, pi::aggregate_record("erisc_to_x280_header", erisc_identity, erisc, calibrated));
        write(
            erisc_id,
            pi::run_record(
                "erisc_to_x280_header",
                erisc_identity,
                "worker",
                pi::erisc_to_x280_header_worker(
                    erisc,
                    worker,
                    {.logical = std::array<uint32_t, 2>{0, 11},
                     .virt = std::array<uint32_t, 2>{31, 25},
                     .noc0 = std::array<uint32_t, 2>{10, 1},
                     .noc = 0u},
                    {31, 25}),
                Counter::EriscWallClock,
                erisc_clock,
                calibrated));
    }

    // core_sync: one record per release shape over the 4x4 block.
    {
        std::vector<x280::suites::core_sync::ReleasedWorker> workers;
        for (uint32_t y = 0; y < 4; ++y) {
            for (uint32_t x = 0; x < 4; ++x) {
                workers.push_back(
                    {.logical_x = x, .logical_y = y, .raw_timestamp = (1ull << 53) + x + 4 * y, .outcome = 0});
            }
        }
        for (const char* release : {"null", "unicast", "multicast"}) {
            write(
                "core_sync.release_skew.fixed4x4",
                x280::suites::core_sync::characterization_record(release, 0x301, 0, workers));
        }
    }

    // eth_credit: one run's throughput record and its diagnostic records.
    {
        ec_result_t forwarder{};
        forwarder.packets_in[0] = 20000;
        forwarder.packets_out[0] = 20000;
        forwarder.first_batch_packets = 4;
        forwarder.first_ts = 1000;
        forwarder.last_ts = 3000000;
        forwarder.status.state = EC_STATE_DONE;
        const x280::suites::eth_credit::ReceiveWindow window{
            .case_name = "throughput_sweep",
            .p = 0,
            .tx_chip = 0,
            .rx_chip = 1,
            .x280_hw_index = 0,
            .tx_erisc =
                {.logical = std::array<uint32_t, 2>{0, 8},
                 .virt = std::array<uint32_t, 2>{28, 25},
                 .noc0 = std::array<uint32_t, 2>{6, 1},
                 .noc = 0u},
            .rx_erisc =
                {.logical = std::array<uint32_t, 2>{0, 5},
                 .virt = std::array<uint32_t, 2>{25, 25},
                 .noc0 = std::array<uint32_t, 2>{14, 1},
                 .noc = 1u},
            .tx_x280 = {.noc0 = std::array<uint32_t, 2>{8, 3}, .noc = hout::kX280Noc},
            .rx_x280 = {.noc0 = std::array<uint32_t, 2>{8, 3}, .noc = hout::kX280Noc},
            .x280_pll_mhz = 1750,
            .shape = {1u, 8u, 8u, 4192u, 4u}};
        const char* ec_id = "eth_credit.throughput";
        write(ec_id, x280::suites::eth_credit::throughput_record(window, forwarder, erisc_clock, erisc_clock));
        // The same window at E1, the Ethernet hop alone, on its ERISC wall clock.
        auto hop = window;
        hop.case_name = "ethernet_hop_sweep";
        hop.path = x280::suites::eth_credit::kEthernetHop;
        write(
            "eth_credit.ethernet_hop",
            x280::suites::eth_credit::throughput_record(hop, forwarder, erisc_clock, erisc_clock));
        // At the sending forwarder, the NoC hop alone, which counts what it sent.
        auto noc_hop = window;
        noc_hop.case_name = "noc_hop_sweep";
        noc_hop.path = x280::suites::eth_credit::kNocHop;
        write(
            "eth_credit.noc_hop",
            x280::suites::eth_credit::throughput_record(noc_hop, forwarder, erisc_clock, erisc_clock));
        // At the receiving forwarder again, the receive side alone.
        auto receive_side = window;
        receive_side.case_name = "receive_side_sweep";
        receive_side.path = x280::suites::eth_credit::kReceiveSide;
        write(
            "eth_credit.receive_side",
            x280::suites::eth_credit::throughput_record(receive_side, forwarder, erisc_clock, erisc_clock));
        // Each write alone at the sending forwarder.
        for (const auto& [id, path] :
             {std::pair{"eth_credit.w1_alone", x280::suites::eth_credit::kW1Alone},
              std::pair{"eth_credit.w2_alone", x280::suites::eth_credit::kW2Alone},
              std::pair{"eth_credit.r1_alone", x280::suites::eth_credit::kR1Alone}}) {
            auto alone = window;
            alone.path = path;
            write(id, x280::suites::eth_credit::throughput_record(alone, forwarder, erisc_clock, erisc_clock));
        }
        ec_result_t kernel{};
        kernel.magic = EC_RESULT_MAGIC;
        kernel.status.state = EC_STATE_DONE;
        write(ec_id, x280::suites::eth_credit::kernel_record("throughput_sweep", 0, "E0", window.tx_erisc, kernel));
        write(ec_id, x280::suites::eth_credit::attribution_record("throughput_sweep", {}, {}, {}));
        write(
            ec_id,
            x280::suites::eth_credit::link_record(
                {.case_name = "throughput_sweep",
                 .chip = 0,
                 .metal_failure = "none",
                 .undrained = "none",
                 .erisc_clock = &erisc_clock}));
    }

    validate(27);
}
