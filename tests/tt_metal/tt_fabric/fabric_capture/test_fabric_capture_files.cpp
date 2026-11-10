// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// What a capture's files promise: each read's offset finds its bytes in the .bin, and a capture never replaces another
// or leaves a half-written one behind. No device.

#include <gtest/gtest.h>

#include <cstddef>
#include <exception>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <span>
#include <string>
#include <vector>

#include <unistd.h>

#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_files.hpp"

namespace tt::tt_fabric::capture {
namespace {

// An empty directory of its own for each test, removed afterwards.
class CaptureFilesTest : public ::testing::Test {
protected:
    void SetUp() override {
        directory_ =
            std::filesystem::temp_directory_path() / ("fabric_capture_files_" + std::to_string(::getpid()) + "_" +
                                                      ::testing::UnitTest::GetInstance()->current_test_info()->name());
        std::filesystem::remove_all(directory_);
        std::filesystem::create_directories(directory_);
    }
    void TearDown() override { std::filesystem::remove_all(directory_); }

    std::filesystem::path directory_;
};

std::vector<std::byte> make_bytes(std::initializer_list<int> values) {
    std::vector<std::byte> bytes;
    bytes.reserve(values.size());
    for (const int value : values) {
        bytes.push_back(static_cast<std::byte>(value));
    }
    return bytes;
}

std::vector<std::byte> read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    const std::string chars((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const std::span<const std::byte> bytes = std::as_bytes(std::span(chars));
    return {bytes.begin(), bytes.end()};
}

// Each recorded read's offset finds the bytes it was captured with, and a failed read keeps its error and has no
// offset.
void expect_reads_find_their_bytes(
    const std::vector<ReadRecord>& records,
    const std::vector<CapturedRead>& captured,
    const std::vector<std::byte>& bin) {
    ASSERT_EQ(records.size(), captured.size());
    for (size_t i = 0; i < records.size(); ++i) {
        const ReadRecord& read = records[i];
        EXPECT_EQ(read.address, captured[i].address);
        EXPECT_EQ(read.t_ns, captured[i].t_ns);
        if (captured[i].error.has_value()) {
            EXPECT_FALSE(read.bin_offset.has_value());
            EXPECT_EQ(read.error, captured[i].error);
            continue;
        }
        ASSERT_TRUE(read.bin_offset.has_value());
        const auto start = bin.begin() + static_cast<std::ptrdiff_t>(*read.bin_offset);
        EXPECT_EQ(std::vector<std::byte>(start, start + read.size_bytes), captured[i].bytes);
    }
}

// Every read's offset finds its bytes in the .bin, a round's as well as the full read's.
TEST_F(CaptureFilesTest, EachReadsOffsetFindsItsBytes) {
    const RouterTarget e0{.router = {.path = "M0/C0/E0"}, .location = CoreLocation::FOUND};
    const RouterTarget e1{.router = {.path = "M0/C0/E1"}, .location = CoreLocation::FOUND};
    // E0 has two rounds, the second of which failed, then a full read whose last register read failed. E1 has no
    // rounds and one read.
    const std::vector<CapturedRound> e0_rounds = {
        {.round = 0,
         .t_ns = 1,
         .reads =
             {{.address = 0x10, .size_bytes = 4, .t_ns = 1, .bytes = make_bytes({5, 6, 7, 8})},
              {.address = 0x40, .size_bytes = 2, .t_ns = 2, .bytes = make_bytes({30, 31})}}},
        {.round = 1, .t_ns = 3, .reads = {{.address = 0x10, .size_bytes = 4, .t_ns = 3, .error = "timed out"}}},
    };
    const CapturedCore e0_core{
        .reads = {
            {.address = 0x10, .size_bytes = 4, .t_ns = 4, .bytes = make_bytes({1, 2, 3, 4})},
            {.address = 0, .size_bytes = 6, .t_ns = 5, .bytes = make_bytes({10, 11, 12, 13, 14, 15})},
            {.address = 0x20, .size_bytes = 4, .t_ns = 6, .error = "timed out"},
        }};
    const CapturedCore e1_core{.reads = {{.address = 0, .size_bytes = 2, .t_ns = 7, .bytes = make_bytes({20, 21})}}};

    const std::filesystem::path bin_path = directory_ / "capture.bin";
    CaptureBinWriter writer(bin_path);
    const RouterRecord e0_record = writer.append_router(e0, e0_rounds, e0_core);
    const RouterRecord e1_record = writer.append_router(e1, {}, e1_core);
    const FinishedBin finished = writer.finish();

    const std::vector<std::byte> bin = read_file(bin_path);
    ASSERT_EQ(e0_record.rounds.size(), e0_rounds.size());
    for (size_t round = 0; round < e0_rounds.size(); ++round) {
        EXPECT_EQ(e0_record.rounds[round].round, e0_rounds[round].round);
        EXPECT_EQ(e0_record.rounds[round].t_ns, e0_rounds[round].t_ns);
        expect_reads_find_their_bytes(e0_record.rounds[round].reads, e0_rounds[round].reads, bin);
    }
    expect_reads_find_their_bytes(e0_record.reads, e0_core.reads, bin);
    EXPECT_TRUE(e1_record.rounds.empty());
    expect_reads_find_their_bytes(e1_record.reads, e1_core.reads, bin);
    EXPECT_EQ(finished.size_bytes, bin.size());
    EXPECT_EQ(finished.blake3, compute_blake3_hex(bin));
}

TEST_F(CaptureFilesTest, NeverReplacesACaptureOrLeavesAHalfOne) {
    // directory_ stands in for a run's folder; the capture's files go in its captures/first/.
    const CaptureFilePaths paths = make_capture_file_paths(directory_, "first", 0, 1);
    EXPECT_EQ(paths.bin_path, directory_ / "captures" / "first" / "fabric_capture_rank_1_of_1.bin");

    // A writer that isn't finished leaves no file behind.
    {
        CaptureBinWriter writer(paths.bin_path);
        writer.append_router(
            {.router = {.path = "M0/C0/E0"}, .location = CoreLocation::FOUND},
            {},
            {.reads = {{.size_bytes = 2, .bytes = make_bytes({1, 2})}}});
    }
    EXPECT_TRUE(std::filesystem::is_empty(paths.bin_path.parent_path()));

    // A .bin already under the name isn't replaced, by a new capture or by a writer that started before it appeared.
    CaptureBinWriter writer(paths.bin_path);
    std::ofstream(paths.bin_path) << "an earlier capture";
    EXPECT_THROW(make_capture_file_paths(directory_, "first", 0, 1), std::exception);
    EXPECT_THROW(writer.finish(), std::exception);
    EXPECT_EQ(read_file(paths.bin_path).size(), std::string("an earlier capture").size());
}

}  // namespace
}  // namespace tt::tt_fabric::capture
