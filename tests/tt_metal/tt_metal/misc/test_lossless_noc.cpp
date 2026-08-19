// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "hostdevcommon/profiler_common.h"
#include "impl/profiler/lossless_noc.hpp"
#include "impl/profiler/profiler.hpp"
#include "tools/profiler/event_metadata.hpp"

namespace {

using tt::tt_metal::profiler::LosslessNocCore;
using tt::tt_metal::profiler::LosslessNocOperation;
using tt::tt_metal::profiler::LosslessNocTransaction;

TEST(LosslessNocTest, DebugDumpPollingExcludesUnsupportedRiscTypes) {
    EXPECT_TRUE(tt::tt_metal::is_supported_debug_dump_risc_type(tracy::RiscType::BRISC, false));
    EXPECT_TRUE(tt::tt_metal::is_supported_debug_dump_risc_type(tracy::RiscType::ERISC, true));
    EXPECT_FALSE(tt::tt_metal::is_supported_debug_dump_risc_type(tracy::RiscType::ERISC, false));
    EXPECT_FALSE(tt::tt_metal::is_supported_debug_dump_risc_type(tracy::RiscType::QUASAR_DM0, false));
}

TEST(LosslessNocTest, ExtendedTimestampedDataReportsCompleteMarkerCount) {
    EXPECT_EQ(tt::tt_metal::timestamped_data_packet_marker_count(kernel_profiler::TS_DATA), 2);
    EXPECT_EQ(tt::tt_metal::timestamped_data_packet_marker_count(kernel_profiler::TS_DATA_16B), 3);
    EXPECT_EQ(tt::tt_metal::timestamped_data_packet_marker_count(kernel_profiler::TS_DATA_24B), 4);
}

TEST(LosslessNocTest, FullUint32ByteCountRoundTripsThroughSizeTrailer) {
    constexpr uint32_t expected_num_bytes = 0xFEDCBA98;
    static_assert(kernel_profiler::TimestampedDataSize<kernel_profiler::TS_DATA_16B>::size == 2);
    static_assert(kernel_profiler::TimestampedDataSize<kernel_profiler::TS_DATA_24B>::size == 3);

    KernelProfilerNocEventMetadata metadata;
    metadata.data.local_event_size_trailer.num_bytes = expected_num_bytes;
    metadata.data.local_event_size_trailer.reserved = 0;

    const KernelProfilerNocEventMetadata decoded(metadata.asU64());
    EXPECT_EQ(decoded.getLocalNocEventSizeTrailer().num_bytes, expected_num_bytes);
    EXPECT_EQ(decoded.getLocalNocEventSizeTrailer().reserved, 0);
}

TEST(LosslessNocTest, SerializesTransactionAsSpecifiedJson) {
    LosslessNocTransaction transaction{
        .operation =
            LosslessNocOperation{
                .runtime_id = 17,
                .trace_id = 23,
                .trace_replay_session_id = 4,
                .name = "matmul",
            },
        .device_id = 0,
        .core = LosslessNocCore{.x = 3, .y = 5},
        .risc = "NCRISC",
        .issue_timestamp = 123456,
        .type = "READ",
        .noc = "NOC_1",
        .vc = 2,
        .destinations = nlohmann::ordered_json::array({{{"x", 7}, {"y", 9}}}),
        .num_bytes = 0xFEDCBA98,
        .debug_metadata =
            {
                {"posted", false},
                {"src_addr", 0x1000},
                {"dst_addr", 0x2000},
                {"counter", 31},
            },
    };

    EXPECT_EQ(
        tt::tt_metal::profiler::serializeLosslessNocTransaction(transaction),
        nlohmann::ordered_json({
            {"operation",
             {
                 {"runtime_id", 17},
                 {"trace_id", 23},
                 {"trace_replay_session_id", 4},
                 {"name", "matmul"},
             }},
            {"device_id", 0},
            {"core", {{"x", 3}, {"y", 5}}},
            {"risc", "NCRISC"},
            {"issue_timestamp", 123456},
            {"type", "READ"},
            {"noc", "NOC_1"},
            {"vc", 2},
            {"destinations", nlohmann::ordered_json::array({{{"x", 7}, {"y", 9}}})},
            {"num_bytes", 0xFEDCBA98},
            {"debug_metadata",
             {
                 {"posted", false},
                 {"src_addr", 0x1000},
                 {"dst_addr", 0x2000},
                 {"counter", 31},
             }},
        }));
}

TEST(LosslessNocTest, BuildsManifestWithIssueCycleAndNpeSemantics) {
    constexpr uint32_t device_frequency_mhz = 1000;
    constexpr uint64_t event_count = 42;

    EXPECT_EQ(
        tt::tt_metal::profiler::makeLosslessNocManifest(device_frequency_mhz, event_count),
        nlohmann::ordered_json({
            {"schema_version", 1},
            {"capture_mode", "non_dropping"},
            {"complete", true},
            {"timestamp_semantics", "issue_cycles"},
            {"device_frequency_mhz", device_frequency_mhz},
            {"events", {{"path", "lossless_noc_events.jsonl"}, {"count", event_count}}},
            {"npe_modeled_semantics",
             {
                 {"input", "noc_trace*.json"},
                 {"input_timestamps", "issue_cycles"},
                 {"completion_timestamps", "modeled_by_tt_npe"},
             }},
        }));
}

TEST(LosslessNocTest, AtomicallyPublishesJsonlBeforeCompleteManifest) {
    const std::filesystem::path output_dir = std::filesystem::path(::testing::TempDir()) / "lossless_noc_artifact_test";
    std::filesystem::remove_all(output_dir);
    std::filesystem::create_directories(output_dir);

    const LosslessNocTransaction transaction{
        .operation =
            LosslessNocOperation{
                .runtime_id = 17,
                .trace_id = std::nullopt,
                .trace_replay_session_id = std::nullopt,
                .name = "matmul",
            },
        .device_id = 0,
        .core = LosslessNocCore{.x = 3, .y = 5},
        .risc = "NCRISC",
        .issue_timestamp = 123456,
        .type = "READ",
        .noc = "NOC_1",
        .vc = 2,
        .destinations = nlohmann::ordered_json::array(),
        .num_bytes = 0xFEDCBA98,
        .debug_metadata = nlohmann::ordered_json::object(),
    };

    tt::tt_metal::profiler::writeLosslessNocArtifactsAtomically(output_dir, {transaction}, 1000);

    std::ifstream events_stream(output_dir / "lossless_noc_events.jsonl");
    std::string event_line;
    ASSERT_TRUE(std::getline(events_stream, event_line));
    EXPECT_EQ(nlohmann::json::parse(event_line)["num_bytes"], 0xFEDCBA98);
    EXPECT_FALSE(std::getline(events_stream, event_line));

    std::ifstream manifest_stream(output_dir / "lossless_noc_manifest.json");
    const auto manifest = nlohmann::json::parse(manifest_stream);
    EXPECT_TRUE(manifest["complete"]);
    EXPECT_EQ(manifest["events"]["count"], 1);
    EXPECT_FALSE(std::filesystem::exists(output_dir / "lossless_noc_events.jsonl.tmp"));
    EXPECT_FALSE(std::filesystem::exists(output_dir / "lossless_noc_manifest.json.tmp"));

    std::filesystem::remove_all(output_dir);
}

}  // namespace
