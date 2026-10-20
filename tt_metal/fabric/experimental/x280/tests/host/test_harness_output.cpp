// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Host tests for the JSONL writer and the record builder.

#include "harness/output/output.hpp"
#include "tests/host/scoped_env.hpp"

#include <gtest/gtest.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace {

namespace hout = x280::harness::output;

using x280::testing::ScopedEnv;

class HarnessJsonlOutput : public testing::Test {
protected:
    void SetUp() override {
        std::string pattern = (std::filesystem::temp_directory_path() / "x280-jsonl-XXXXXX").string();
        char* created = ::mkdtemp(pattern.data());
        ASSERT_NE(created, nullptr) << std::strerror(errno);
        root_ = created;
        env_.emplace("TT_METAL_HOME", root_.c_str());
        ASSERT_STREQ(std::getenv("TT_METAL_HOME"), root_.c_str());
    }

    void TearDown() override {
        env_.reset();
        if (root_.empty()) {
            return;
        }
        std::error_code error;
        const auto directory = root_ / "characterization";
        if (std::filesystem::is_directory(directory)) {
            for (const auto* name : {"records.jsonl", "blocked.jsonl", "full.jsonl"}) {
                std::filesystem::remove(directory / name, error);
                EXPECT_FALSE(error) << directory / name << ": " << error.message();
            }
        }
        std::filesystem::remove(directory, error);
        EXPECT_FALSE(error) << directory << ": " << error.message();
        std::filesystem::remove(root_, error);
        EXPECT_FALSE(error) << root_ << ": " << error.message();
    }

    std::filesystem::path root_;
    std::optional<ScopedEnv> env_;
};

}  // namespace

TEST_F(HarnessJsonlOutput, CreatesDirectoryAndAppendsCompleteRecords) {
    hout::write_jsonl("records", "\"value\":1");
    hout::write_jsonl("records", "\"value\":2");
    std::ifstream input(root_ / "characterization" / "records.jsonl");
    ASSERT_TRUE(input.is_open());
    std::ostringstream contents;
    contents << input.rdbuf();
    EXPECT_EQ(contents.str(), "{\"value\":1}\n{\"value\":2}\n");
}

TEST_F(HarnessJsonlOutput, ExplicitCampaignDirectoryTakesPrecedence) {
    ScopedEnv selected("X280_CHARACTERIZATION_OUTPUT_DIR", root_.c_str());
    EXPECT_EQ(hout::output_dir(), root_);
}

TEST(HarnessPopulation, PreservesRawSamplesAboveDoublePrecision) {
    const uint64_t tick = (1ULL << 53) + 1;
    EXPECT_EQ(
        hout::sample_population_json({tick, tick + 2, tick + 4}),
        "{\"count\":3,\"units\":\"ticks\","
        "\"raw_ticks\":[9007199254740993,9007199254740995,9007199254740997]}");
}

TEST_F(HarnessJsonlOutput, PropagatesDirectoryCreationFailure) {
    const auto path = root_ / "characterization";
    std::ofstream blocker(path);
    ASSERT_TRUE(blocker.is_open());
    blocker.close();
    EXPECT_THROW(hout::write_jsonl("records", "\"value\":1"), std::filesystem::filesystem_error);
}

TEST_F(HarnessJsonlOutput, ReportsOpenFailureWithPath) {
    const auto path = root_ / "characterization" / "blocked.jsonl";
    std::filesystem::create_directories(path);
    try {
        hout::write_jsonl("blocked", "\"value\":1");
        FAIL() << "opening a directory as a JSONL file must fail";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find(path.string()), std::string::npos);
    }
}

TEST_F(HarnessJsonlOutput, ReportsWriteAndCloseFailuresWithPath) {
    if (!std::filesystem::exists("/dev/full")) {
        GTEST_SKIP() << "/dev/full is required for deterministic write/close failures";
    }
    const auto directory = hout::output_dir();
    const auto path = directory / "full.jsonl";
    std::filesystem::create_symlink("/dev/full", path);
    for (const size_t size : {1u, 1024u * 1024u}) {
        try {
            hout::write_jsonl("full", std::string(size, '0'));
            FAIL() << "writing to /dev/full must fail for body size " << size;
        } catch (const std::runtime_error& error) {
            EXPECT_NE(std::string(error.what()).find(path.string()), std::string::npos);
        }
    }
}

// Spec (record): suite and member, the fields in the order given, then the
// envelope; every value raw. Strings are JSON-escaped.
TEST(HarnessRecord, RendersFieldsThenTheEnvelope) {
    ScopedEnv dispatch("TT_METAL_SLOW_DISPATCH_MODE", "1");
    hout::RecordSpec spec{
        .suite = "suite",
        .member = "member",
        .fields =
            {{"name", "a\"b\n"},
             {"count", 7u},
             {"offset", -3},
             {"flag", true},
             {"none", hout::Value()},
             {"tiles", hout::Value::array({hout::Value::array({1u, 2u})})},
             {"nested", hout::Value::object({{"k", 1u}})},
             {"samples", hout::Value::population({5, 6})},
             {"window",
              hout::Value::throughput(
                  {.start_ticks = 3,
                   .end_ticks = 9,
                   .requests = 2,
                   .completed = true,
                   .useful_bytes_per_request = 96,
                   .slot_bytes_per_request = 128})}},
        .wall_counter = x280::harness::measurement::Counter::X280Mtime,
        .cpu_counter = x280::harness::measurement::Counter::X280Rdcycle,
        .batch = hout::RecordSpec::Batch{.wall_ticks = 10, .cpu_cycles = 20, .items = 2, .logical_bytes = 0},
        .warmup = x280::harness::measurement::WarmupSelection{.warmup = 8},
        .clock = hout::kX280MtimeClock,
    };
    EXPECT_EQ(
        hout::record(spec),
        "\"suite\":\"suite\",\"member\":\"member\",\"name\":\"a\\\"b\\u000a\",\"count\":7,"
        "\"offset\":-3,\"flag\":true,\"none\":null,\"tiles\":[[1,2]],\"nested\":{\"k\":1},"
        "\"samples\":{\"count\":2,\"units\":\"ticks\",\"raw_ticks\":[5,6]},"
        "\"window\":{\"start_ticks\":3,\"end_ticks\":9,\"requests\":2,"
        "\"useful_bytes_per_request\":96,\"slot_bytes_per_request\":128,\"interval_error\":null},"
        "\"measurement_schema\":2,\"wall_counter\":\"x280_mtime\",\"cpu_counter\":\"x280_rdcycle\","
        "\"raw_wall_ticks\":10,\"cpu_cycles\":20,\"items\":2,\"logical_bytes\":0,"
        "\"warmup_source\":\"explicit\","
        "\"clock\":{\"domain\":\"x280_mtime\",\"expected_hz\":50000000,\"conversion_hz\":50000000,"
        "\"errors\":[]},\"tt_metal_slow_dispatch_mode\":\"1\"");
}

// Spec (RecordSpec): schema 2 with clock evidence, 1 with NoConversion, absent
// with no clock; a wall counter needs a clock and NoConversion a wall counter.
// The dispatch mode is null when the variable is unset.
TEST(HarnessRecord, SchemaFollowsTheClock) {
    using x280::harness::measurement::Counter;
    ScopedEnv dispatch("TT_METAL_SLOW_DISPATCH_MODE", nullptr);
    const auto aiclk = hout::record(
        {.suite = "s",
         .member = "m",
         .warmup = hout::NoWarmup{},
         .clock = x280::harness::measurement::AiclkRunEvidence{}});
    EXPECT_NE(
        aiclk.find("\"measurement_schema\":2,\"warmup_source\":\"none\",\"clock\":{\"domain\":\"aiclk\""),
        std::string::npos)
        << aiclk;
    const auto raw = hout::record(
        {.suite = "s",
         .member = "m",
         .wall_counter = Counter::TensixLiveTimer,
         .warmup = hout::NoWarmup{},
         .clock = hout::NoConversion{}});
    EXPECT_EQ(
        raw,
        "\"suite\":\"s\",\"member\":\"m\",\"measurement_schema\":1,"
        "\"wall_counter\":\"tensix_live_timer\",\"warmup_source\":\"none\","
        "\"tt_metal_slow_dispatch_mode\":null");
    EXPECT_EQ(
        hout::record({.suite = "s", .member = "m"}),
        "\"suite\":\"s\",\"member\":\"m\",\"tt_metal_slow_dispatch_mode\":null");
    EXPECT_THROW(
        hout::record({.suite = "s", .member = "m", .wall_counter = Counter::X280Mtime}), std::invalid_argument);
    EXPECT_THROW(
        hout::record({.suite = "s", .member = "m", .warmup = hout::NoWarmup{}, .clock = hout::NoConversion{}}),
        std::invalid_argument);
}

// Spec (RecordSpec): AICLK evidence converts the counter it sampled,
// or, for a DRISC's wall counter, the Tensix live timer that stands in for it.
TEST(HarnessRecord, AiclkEvidenceMustHaveSampledTheWallCounter) {
    using x280::harness::measurement::Counter;
    const auto record = [](Counter wall, Counter sampled) {
        x280::harness::measurement::AiclkRunEvidence evidence{};
        evidence.reference = sampled;
        return hout::record(
            {.suite = "s", .member = "m", .wall_counter = wall, .warmup = hout::NoWarmup{}, .clock = evidence});
    };
    EXPECT_NO_THROW(record(Counter::EriscWallClock, Counter::EriscWallClock));
    EXPECT_NO_THROW(record(Counter::TensixLiveTimer, Counter::TensixLiveTimer));
    EXPECT_NO_THROW(record(Counter::RiscDebugWallClock, Counter::TensixLiveTimer));
    EXPECT_THROW(record(Counter::EriscWallClock, Counter::TensixLiveTimer), std::invalid_argument);
    EXPECT_THROW(record(Counter::TensixLiveTimer, Counter::EriscWallClock), std::invalid_argument);
    EXPECT_THROW(record(Counter::RiscDebugWallClock, Counter::EriscWallClock), std::invalid_argument);
}

// Spec (RecordSpec): a record with a clock states
// how its warm-up was chosen -- explicit, calibrated or none -- and a refused
// calibration is no warm-up selection.
TEST(HarnessRecord, AClockRequiresAWarmupSelection) {
    namespace hmeas = x280::harness::measurement;
    EXPECT_THROW(hout::record({.suite = "s", .member = "m", .clock = hout::kX280MtimeClock}), std::invalid_argument);
    EXPECT_THROW(
        hout::record(
            {.suite = "s",
             .member = "m",
             .warmup = hmeas::WarmupSelection{.refusal = hmeas::WarmupRefusal::InvalidPilot},
             .clock = hout::kX280MtimeClock}),
        std::invalid_argument);
    EXPECT_NE(
        hout::record({.suite = "s", .member = "m", .warmup = hout::NoWarmup{}, .clock = hout::kX280MtimeClock})
            .find("\"warmup_source\":\"none\",\"clock\""),
        std::string::npos);
    EXPECT_NO_THROW(hout::record({.suite = "s", .member = "m"}));
}

static_assert(!std::is_constructible_v<hout::Value, double>, "a record field must not hold a floating-point value");

// Spec (record): a key may appear once per object, envelope keys included.
TEST(HarnessRecord, RefusesADuplicateKeyIncludingAnEnvelopeKey) {
    const hout::RecordSpec cases[] = {
        {.suite = "s", .member = "m", .fields = {{"suite", "t"}}},
        {.suite = "s",
         .member = "m",
         .fields = {{"clock", 1u}},
         .warmup = hout::NoWarmup{},
         .clock = hout::kX280MtimeClock},
        {.suite = "s", .member = "m", .fields = {{"a", 1u}, {"a", 2u}}},
        {.suite = "s", .member = "m", .fields = {{"o", hout::Value::object({{"k", 1u}, {"k", 2u}})}}},
    };
    for (const auto& spec : cases) {
        try {
            hout::record(spec);
            ADD_FAILURE() << "record() accepted a duplicate key";
        } catch (const hout::RecordKeyError& error) {
            EXPECT_EQ(error.fault(), hout::RecordKeyFault::Duplicate) << error.what();
        }
    }
    EXPECT_NO_THROW(
        hout::record({.suite = "s", .member = "m", .fields = {{"o", hout::Value::object({{"k", 1u}})}, {"k", 2u}}}));
}

// Spec (record): suite and member are required, a batch names both counters,
// and an invalid batch interval throws.
TEST(HarnessRecord, RefusesAnIncompleteSpecOrAnInvalidBatch) {
    using x280::harness::measurement::Counter;
    EXPECT_THROW(hout::record({.suite = "s"}), std::invalid_argument);
    const hout::RecordSpec::Batch batch{.wall_ticks = 10, .cpu_cycles = 20, .items = 2};
    EXPECT_THROW(
        hout::record(
            {.suite = "s",
             .member = "m",
             .wall_counter = Counter::X280Mtime,
             .batch = batch,
             .clock = hout::kX280MtimeClock}),
        std::invalid_argument);
    EXPECT_THROW(
        hout::record(
            {.suite = "s",
             .member = "m",
             .wall_counter = Counter::X280Mtime,
             .cpu_counter = Counter::X280Rdcycle,
             .batch = hout::RecordSpec::Batch{.cpu_cycles = 20, .items = 2},
             .warmup = hout::NoWarmup{},
             .clock = hout::kX280MtimeClock}),
        std::runtime_error);
}
