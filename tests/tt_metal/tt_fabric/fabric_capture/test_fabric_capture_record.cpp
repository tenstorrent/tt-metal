// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The capture record's writer and reader agree: reading a written record and writing it again gives back the same
// JSON, and the reader is strict, failing on a key it doesn't read or can't find. No device.

#include <gtest/gtest.h>

#include <chrono>
#include <exception>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_reads.hpp"
#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_record.hpp"
#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_targets.hpp"

namespace tt::tt_fabric::capture {
namespace {

using json = nlohmann::ordered_json;

// A record with every optional part both set and null, and every location:
//   - M0/C1, found, with a router read in full after two rounds (the second failed), one at another core than the
//     manifest's with no rounds, and one with no core for its channel;
//   - M0/C2, which the reader didn't find;
//   - M0/C3, found as another architecture.
CaptureRecord make_record_with_every_part() {
    const RouterRecord read_router{
        .target =
            {.router =
                 {.path = "M0/C1/E0",
                  .eth_chan = 0,
                  .virtual_core = {20, 25},
                  .stream_register_addresses = {0x1000, 0x1040},
                  .multi_sampled_regions =
                      {{.kind = RegionKind::L1, .address = 0x40, .size_bytes = 160},
                       {.kind = RegionKind::REGISTER, .address = 0x1040, .size_bytes = 4}}},
             .reader_core = CoreXY{20, 25},
             .location = CoreLocation::FOUND,
             .discovery_skipped = true},
        .rounds =
            {{.round = 0,
              .t_ns = 100,
              .reads =
                  {{.address = 0xffb12000, .size_bytes = 4, .t_ns = 100, .bin_offset = 0},
                   {.address = 0x40, .size_bytes = 160, .t_ns = 110, .bin_offset = 4}}},
             {.round = 1,
              .t_ns = 200,
              .reads = {{.address = 0xffb12000, .size_bytes = 4, .t_ns = 200, .error = "read timed out"}}}},
        .reads =
            {{.address = 0xffb12000, .size_bytes = 4, .t_ns = 300, .bin_offset = 164},
             {.address = 0, .size_bytes = 262144, .t_ns = 400, .bin_offset = 168},
             {.address = 0x1000, .size_bytes = 4, .t_ns = 500, .error = "failed"}},
    };
    const RouterRecord moved_router{
        .target =
            {.router = {.path = "M0/C1/W0", .eth_chan = 1, .virtual_core = {21, 25}},
             .reader_core = CoreXY{22, 25},
             .location = CoreLocation::CORE_MISMATCH},
        .reads = {{.address = 0, .size_bytes = 262144, .t_ns = 600, .error = "failed"}},
    };
    const RouterRecord unread_router{
        .target =
            {.router = {.path = "M0/C1/N0", .eth_chan = 9, .virtual_core = {9, 9}},
             .location = CoreLocation::CHANNEL_NOT_FOUND},
    };
    const auto make_unlocated_router = [](const std::string& path) {
        return RouterRecord{.target = {.router = {.path = path, .eth_chan = 0, .virtual_core = {20, 25}}}};
    };
    return CaptureRecord{
        .manifest =
            {.path = "fabric_manifest_rank_1_of_2.json",
             .blake3 = std::string(64, 'a'),
             .run_id = "20261105T221500Z-1a2b3c4d",
             .mpi_rank = 0,
             .world_size = 2,
             .host_name = "bh-glx-120-c02u14",
             .git_sha = "0123456789abcdef0123456789abcdef01234567"},
        .capture_id = "20261105T223000Z",
        .capture_git_sha = "0123456789abcdef0123456789abcdef01234567",
        .cluster_id = "bh-glx-120-c02u14",
        .started_at = "2026-11-05T22:30:00Z",
        .finished_at = "2026-11-05T22:30:09Z",
        .round_options = {.num_rounds = 2, .interval = std::chrono::milliseconds(500)},
        .bin_file = "fabric_capture_rank_1_of_2.bin",
        .bin_size_bytes = 262312,
        .bin_blake3 = std::string(64, 'b'),
        .discovery_skipped_cores = {{.asic_id = 0xa, .core = {20, 25}, .reason = "heartbeat stalled"}},
        .chips =
            {{.path = "M0/C1",
              .asic_id = 0xa,
              .chip = 0,
              .location = ChipLocation::FOUND,
              .routers = {read_router, moved_router, unread_router}},
             {.path = "M0/C2",
              .asic_id = 0xb,
              .location = ChipLocation::MISSING,
              .routers = {make_unlocated_router("M0/C2/E0")}},
             {.path = "M0/C3",
              .asic_id = 0xc,
              .chip = 1,
              .location = ChipLocation::ARCH_MISMATCH,
              .routers = {make_unlocated_router("M0/C3/E0")}}},
    };
}

// What reading `record_json` fails with, or empty when it doesn't fail.
std::string parse_error(const json& record_json) {
    try {
        parse_capture_record(record_json);
    } catch (const std::exception& error) {
        return error.what();
    }
    return {};
}

void expect_parse_fails_with(const json& record_json, std::string_view expected) {
    const std::string message = parse_error(record_json);
    EXPECT_FALSE(message.empty()) << "expected a failure with: " << expected;
    EXPECT_NE(message.find(expected), std::string::npos) << message;
}

// Writing a record, reading it back and writing again gives the same JSON, and reading gives back the record it was
// written from: with every optional part, and with none.
TEST(CaptureRecord, RoundTrips) {
    for (const CaptureRecord& record : {make_record_with_every_part(), CaptureRecord{}}) {
        const json written = make_capture_record_json(record);
        const CaptureRecord parsed = parse_capture_record(written);
        EXPECT_EQ(make_capture_record_json(parsed), written);
        EXPECT_EQ(parsed, record);
    }
}

// A key the writer wrote that the reader doesn't read fails, whatever its value.
TEST(CaptureRecord, FailsOnAKeyItDoesNotRead) {
    json record_json = make_capture_record_json(make_record_with_every_part());
    record_json["capture"]["pid"] = 0;
    expect_parse_fails_with(
        record_json, "Fabric capture record: /capture has keys this reader doesn't read: /capture/pid");
}

// A key the reader reads that the writer didn't write fails, named by its pointer. A chip's path is its key, so its "/"
// is escaped in the pointer.
TEST(CaptureRecord, FailsOnAMissingKey) {
    json record_json = make_capture_record_json(make_record_with_every_part());
    record_json["chips"]["M0/C1"]["routers"]["E0"].erase("eth_chan");
    expect_parse_fails_with(record_json, "Fabric capture record: /chips/M0~1C1/routers/E0/eth_chan is missing");
}

TEST(CaptureRecord, FailsOnAnotherFileKindOrVersion) {
    json record_json = make_capture_record_json(make_record_with_every_part());
    record_json["file_kind"] = "fabric_manifest";
    expect_parse_fails_with(record_json, "/file_kind is \"fabric_manifest\", not \"fabric_capture\"");

    record_json = make_capture_record_json(make_record_with_every_part());
    record_json["capture_version"] = 2;
    expect_parse_fails_with(record_json, "/capture_version is 2, not 1");
}

// Enums and ASIC ids are spelled as the manifest spells them, and nothing else reads as one.
// A round's reads are read as strictly as the full read's.
TEST(CaptureRecord, FailsOnARoundReadItCannotRead) {
    json record_json = make_capture_record_json(make_record_with_every_part());
    record_json["chips"]["M0/C1"]["routers"]["E0"]["rounds"][0]["reads"][0].erase("t_ns");
    expect_parse_fails_with(
        record_json, "Fabric capture record: /chips/M0~1C1/routers/E0/rounds/0/reads/0/t_ns is missing");
}

TEST(CaptureRecord, FailsOnASpellingTheWriterWouldNotWrite) {
    json record_json = make_capture_record_json(make_record_with_every_part());
    record_json["chips"]["M0/C1"]["location"] = "lost";
    expect_parse_fails_with(record_json, "/chips/M0~1C1/location is \"lost\", which names no value this reader knows");

    record_json = make_capture_record_json(make_record_with_every_part());
    record_json["chips"]["M0/C1"]["routers"]["E0"]["multi_sampled_regions"][0]["kind"] = "dram";
    expect_parse_fails_with(
        record_json,
        "/chips/M0~1C1/routers/E0/multi_sampled_regions/0/kind is \"dram\", which names no value this reader knows");

    record_json = make_capture_record_json(make_record_with_every_part());
    record_json["chips"]["M0/C1"]["asic_id"] = "0xa";
    expect_parse_fails_with(
        record_json, "/chips/M0~1C1/asic_id is \"0xa\", not an ASIC id written as 0x and 16 hex digits");
}

}  // namespace
}  // namespace tt::tt_fabric::capture
