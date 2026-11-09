// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Legacy versus new ttnn::reshape: bit-exactness on first and cached dispatch, host dispatch time, and device
// kernel time. Cases are a curated set of issue reproductions plus, when TT_RESHAPE_WORKLOAD_JSON names a file
// written by tests/ttnn/benchmark/reshape/capture_reshape_workloads.py, every single-device workload in it.
//
//   ./build_Release/bin/tt-nn-validation-basic --gtest_filter="ReshapeNewBenchmark.Correctness*"
//   ./build_Release/bin/tt-nn-validation-basic --gtest_filter="ReshapeNewBenchmark.DispatchPerformance"
//   TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1 \
//       ./build_Release/bin/tt-nn-validation-basic --gtest_filter="ReshapeNewBenchmark.DeviceKernelTime"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include <enchantum/enchantum.hpp>
#include <fmt/format.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/profiler.hpp>
#include <tt-metalium/experimental/tensor_layout_apis_with_custom_alignment.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt_stl/reflection.hpp>

#include "ttnn/operations/data_movement/reshape_view/reshape_force.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape_policy.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::operations::data_movement::reshape_benchmark {
namespace {

using detail::ReshapeRequest;

enum class LegacyStatus {
    Exact,        // legacy output must match the new output bit for bit
    Wrong,        // a known legacy bug: only the new output is checked
    Unsupported,  // legacy rejects the case: only the new output is checked and timed
};

struct ReshapeCase {
    std::string name;
    TensorSpec input_spec;
    ttnn::Shape logical_shape;
    ttnn::Shape padded_shape;
    std::optional<MemoryConfig> memory_config;
    std::optional<PadValue> pad_value;
    std::optional<CoreRangeSet> sub_core_grid;
    bool skip_padding_fill = false;
    LegacyStatus legacy = LegacyStatus::Exact;
};

constexpr const char* kKernelDurationAnalysis = "DEVICE KERNEL DURATION [ns]";

MemoryConfig dram() { return MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM}; }

MemoryConfig l1_sharded(TensorMemoryLayout layout, CoreRange grid, std::array<uint32_t, 2> shard_shape) {
    return MemoryConfig{
        layout, BufferType::L1, ShardSpec(CoreRangeSet(grid), shard_shape, ShardOrientation::ROW_MAJOR)};
}

TensorSpec spec(const ttnn::Shape& shape, DataType dtype, Layout layout, const MemoryConfig& memory_config) {
    return TensorSpec(shape, TensorLayout(dtype, PageConfig(layout), memory_config));
}

ReshapeCase simple_case(
    std::string name,
    TensorSpec input_spec,
    const ttnn::Shape& shape,
    std::optional<MemoryConfig> memory_config = std::nullopt,
    LegacyStatus legacy = LegacyStatus::Exact) {
    return ReshapeCase{
        .name = std::move(name),
        .input_spec = std::move(input_spec),
        .logical_shape = shape,
        .padded_shape = shape,
        .memory_config = std::move(memory_config),
        .legacy = legacy};
}

std::vector<ReshapeCase> curated_cases() {
    std::vector<ReshapeCase> cases;
    const auto bf16 = DataType::BFLOAT16;

    // #38031: DeepSeek q_reshape, width-sharded in, height-sharded out.
    cases.push_back(simple_case(
        "38031_q_reshape",
        spec(
            ttnn::Shape{1, 1, 32, 3072},
            bf16,
            Layout::TILE,
            l1_sharded(TensorMemoryLayout::WIDTH_SHARDED, CoreRange({0, 0}, {7, 1}), {32, 192})),
        ttnn::Shape{1, 32, 16, 192},
        l1_sharded(TensorMemoryLayout::HEIGHT_SHARDED, CoreRange({0, 0}, {7, 0}), {128, 192})));

    // Attention head split and merge.
    cases.push_back(simple_case(
        "head_split", spec(ttnn::Shape{1, 1, 128, 4096}, bf16, Layout::TILE, dram()), ttnn::Shape{1, 128, 32, 128}));
    cases.push_back(simple_case(
        "head_merge", spec(ttnn::Shape{1, 128, 32, 128}, bf16, Layout::TILE, dram()), ttnn::Shape{1, 1, 128, 4096}));
    cases.push_back(simple_case(
        "outer_merge_view",
        spec(ttnn::Shape{1, 32, 128, 128}, bf16, Layout::TILE, dram()),
        ttnn::Shape{1, 1, 4096, 128}));

    // #19272: a row-major height-sharded tensor with row padding; legacy views it and scrambles the rows.
    cases.push_back(simple_case(
        "19272_padded_height_shard",
        spec(
            ttnn::Shape{1, 1, 8, 8},
            bf16,
            Layout::ROW_MAJOR,
            l1_sharded(TensorMemoryLayout::HEIGHT_SHARDED, CoreRange({0, 0}, {0, 0}), {8, 16})),
        ttnn::Shape{1, 2, 4, 8},
        std::nullopt,
        LegacyStatus::Wrong));

    // #36172: ND-sharded input.
    cases.push_back(simple_case(
        "36172_nd_sharded",
        spec(
            ttnn::Shape{2, 64, 64},
            bf16,
            Layout::TILE,
            MemoryConfig(BufferType::L1, NdShardSpec{ttnn::Shape{1, 32, 32}, CoreRangeSet(CoreRange({0, 0}, {3, 3}))})),
        ttnn::Shape{2, 32, 128},
        std::nullopt,
        LegacyStatus::Unsupported));

    // #27843: uint8.
    cases.push_back(simple_case(
        "27843_uint8",
        spec(ttnn::Shape{4}, DataType::UINT8, Layout::ROW_MAJOR, dram()),
        ttnn::Shape{1, 1, 2, 2},
        std::nullopt,
        LegacyStatus::Unsupported));

    // #55237: two calls that differ only in the padded shape; legacy's hash collides on the second.
    const auto rm_30x32 = spec(ttnn::Shape{1, 1, 30, 32}, bf16, Layout::ROW_MAJOR, dram());
    cases.push_back(ReshapeCase{
        .name = "55237_padded_32",
        .input_spec = rm_30x32,
        .logical_shape = ttnn::Shape{1, 1, 32, 30},
        .padded_shape = ttnn::Shape{1, 1, 32, 32}});
    cases.push_back(ReshapeCase{
        .name = "55237_padded_64",
        .input_spec = rm_30x32,
        .logical_shape = ttnn::Shape{1, 1, 32, 30},
        .padded_shape = ttnn::Shape{1, 1, 32, 64},
        .legacy = LegacyStatus::Wrong});

    // #61727's tiled reshape (the trace ordering itself is covered by the Python regression test).
    cases.push_back(
        simple_case("61727_tile", spec(ttnn::Shape{64, 64}, bf16, Layout::TILE, dram()), ttnn::Shape{32, 128}));

    // #18088: the Flux rotary split of the last dim into pairs, tiled and row-major.
    cases.push_back(simple_case(
        "18088_tile_pairs", spec(ttnn::Shape{128, 64}, bf16, Layout::TILE, dram()), ttnn::Shape{128, 32, 2}));
    cases.push_back(simple_case(
        "18088_rm_pairs", spec(ttnn::Shape{128, 64}, bf16, Layout::ROW_MAJOR, dram()), ttnn::Shape{128, 32, 2}));

    // Dtype coverage: block float (composite path), float32 and int32 with padded rows, uint32 row-major.
    cases.push_back(simple_case(
        "bfp8_tile",
        spec(ttnn::Shape{1, 1, 64, 64}, DataType::BFLOAT8_B, Layout::TILE, dram()),
        ttnn::Shape{1, 1, 32, 128}));
    cases.push_back(simple_case(
        "fp32_padded_rows",
        spec(ttnn::Shape{2, 3, 40, 50}, DataType::FLOAT32, Layout::TILE, dram()),
        ttnn::Shape{3, 2, 50, 40}));
    cases.push_back(simple_case(
        "int32_padded_rows",
        spec(ttnn::Shape{2, 3, 40, 50}, DataType::INT32, Layout::TILE, dram()),
        ttnn::Shape{6, 20, 100}));
    cases.push_back(simple_case(
        "uint32_rm", spec(ttnn::Shape{3, 5, 7}, DataType::UINT32, Layout::ROW_MAJOR, dram()), ttnn::Shape{7, 15}));

    return cases;
}

ttnn::Shape shape_from_json(const nlohmann::json& dims) {
    const auto values = dims.get<std::vector<uint32_t>>();
    return ttnn::Shape(ttsl::Span<const uint32_t>(values));
}

// Single-device device-storage workloads from a capture file; the rest cannot be replayed here.
std::vector<ReshapeCase> workload_cases() {
    const char* path = std::getenv("TT_RESHAPE_WORKLOAD_JSON");
    if (path == nullptr) {
        return {};
    }
    std::ifstream file(path);
    TT_FATAL(file.good(), "TT_RESHAPE_WORKLOAD_JSON={} cannot be opened", path);
    const auto workloads = nlohmann::json::parse(file).at("workloads");

    std::vector<ReshapeCase> cases;
    for (const auto& workload : workloads) {
        const auto& input = workload.at("input");
        const auto& request = workload.at("request");
        uint32_t devices = 1;
        for (const auto& extent : input.at("topology").at("distribution_shape")) {
            devices *= extent.get<uint32_t>();
        }
        if (input.at("storage") != "device" || devices != 1) {
            continue;
        }

        const auto dtype = enchantum::cast<DataType>(input.at("dtype").get<std::string>()).value();
        const auto layout = enchantum::cast<Layout>(input.at("layout").get<std::string>()).value();
        const auto tile_dims = input.at("tile").get<std::array<uint32_t, 2>>();
        const auto page_config = layout == Layout::TILE ? PageConfig(layout, Tile(tile_dims)) : PageConfig(layout);
        const auto input_memory_config = ttsl::json::from_json<MemoryConfig>(input.at("memory_config"));
        const auto logical = shape_from_json(input.at("logical_shape"));
        TensorSpec input_spec =
            input.at("padded_shape").is_null()
                ? TensorSpec(logical, TensorLayout(dtype, page_config, input_memory_config))
                : TensorSpec(
                      logical,
                      tt::tt_metal::experimental::tensor_layout_from_padded_shape(
                          dtype, page_config, input_memory_config, logical, shape_from_json(input.at("padded_shape"))));

        std::optional<PadValue> pad_value;
        if (const auto& pad = request.at("pad_value"); !pad.is_null()) {
            pad_value = pad.contains("float") ? PadValue{pad.at("float").get<float>()}
                                              : PadValue{pad.at("uint").get<uint32_t>()};
        }
        cases.push_back(ReshapeCase{
            .name = fmt::format("workload_{}", workload.at("key").get<std::string>().substr(0, 12)),
            .input_spec = std::move(input_spec),
            .logical_shape = shape_from_json(request.at("logical_shape")),
            .padded_shape = shape_from_json(request.at("padded_shape")),
            .memory_config = request.at("memory_config").is_null()
                                 ? std::nullopt
                                 : std::optional(ttsl::json::from_json<MemoryConfig>(request.at("memory_config"))),
            .pad_value = pad_value,
            .sub_core_grid = request.at("sub_core_grid").is_null()
                                 ? std::nullopt
                                 : std::optional(ttsl::json::from_json<CoreRangeSet>(request.at("sub_core_grid"))),
            .skip_padding_fill = request.at("skip_padding_fill").get<bool>()});
    }
    return cases;
}

std::vector<ReshapeCase> all_cases() {
    auto cases = curated_cases();
    auto workloads = workload_cases();
    cases.insert(cases.end(), std::make_move_iterator(workloads.begin()), std::make_move_iterator(workloads.end()));
    return cases;
}

template <typename T, typename Bits>
std::vector<T> random_bits(size_t count, std::mt19937& generator) {
    std::vector<T> values(count);
    for (auto& value : values) {
        value = std::bit_cast<T>(static_cast<Bits>(generator()));
    }
    return values;
}

// Raw random bit patterns wherever the dtype stores elements exactly, so NaN payloads, signed zeros and
// subnormals all appear; block float gets ordinary normals, since its shared exponents cannot hold them.
Tensor make_input(const TensorSpec& input_spec, tt::tt_metal::distributed::MeshDevice* device, uint32_t seed) {
    std::mt19937 generator(seed);
    const size_t count = input_spec.logical_shape().volume();
    switch (input_spec.data_type()) {
        case DataType::BFLOAT16:
            return Tensor::from_vector(random_bits<bfloat16, uint16_t>(count, generator), input_spec, device);
        case DataType::FLOAT32:
            return Tensor::from_vector(random_bits<float, uint32_t>(count, generator), input_spec, device);
        case DataType::INT32:
            return Tensor::from_vector(random_bits<int32_t, uint32_t>(count, generator), input_spec, device);
        case DataType::UINT32:
            return Tensor::from_vector(random_bits<uint32_t, uint32_t>(count, generator), input_spec, device);
        case DataType::UINT16:
            return Tensor::from_vector(random_bits<uint16_t, uint16_t>(count, generator), input_spec, device);
        case DataType::UINT8:
            return Tensor::from_vector(random_bits<uint8_t, uint8_t>(count, generator), input_spec, device);
        case DataType::INT8:
            return Tensor::from_vector(random_bits<int8_t, uint8_t>(count, generator), input_spec, device);
        case DataType::BFLOAT8_B:
        case DataType::BFLOAT4_B: {
            std::normal_distribution<float> normal;
            std::vector<float> values(count);
            for (auto& value : values) {
                value = normal(generator);
            }
            return Tensor::from_vector(std::move(values), input_spec, device);
        }
        default: TT_THROW("reshape benchmark: no input generator for {}", input_spec.data_type());
    }
}

std::vector<uint32_t> logical_bits(const Tensor& tensor) {
    auto widen = [](const auto& values, auto to_bits) {
        std::vector<uint32_t> bits(values.size());
        std::transform(values.begin(), values.end(), bits.begin(), to_bits);
        return bits;
    };
    switch (tensor.dtype()) {
        case DataType::BFLOAT16:
            return widen(tensor.to_vector<bfloat16>(), [](bfloat16 v) { return uint32_t{std::bit_cast<uint16_t>(v)}; });
        case DataType::FLOAT32:
        case DataType::BFLOAT8_B:
        case DataType::BFLOAT4_B:
            return widen(tensor.to_vector<float>(), [](float v) { return std::bit_cast<uint32_t>(v); });
        case DataType::INT32:
            return widen(tensor.to_vector<int32_t>(), [](int32_t v) { return std::bit_cast<uint32_t>(v); });
        case DataType::UINT32: return tensor.to_vector<uint32_t>();
        case DataType::UINT16: return widen(tensor.to_vector<uint16_t>(), [](uint16_t v) { return uint32_t{v}; });
        case DataType::UINT8: return widen(tensor.to_vector<uint8_t>(), [](uint8_t v) { return uint32_t{v}; });
        case DataType::INT8:
            return widen(tensor.to_vector<int8_t>(), [](int8_t v) { return uint32_t{std::bit_cast<uint8_t>(v)}; });
        default: TT_THROW("reshape benchmark: no readback for {}", tensor.dtype());
    }
}

Tensor run_legacy(const ReshapeCase& c, const Tensor& input) {
    return detail::reshape_legacy(
        input,
        c.logical_shape,
        c.padded_shape,
        c.memory_config,
        c.pad_value,
        TileReshapeMapMode::CACHE,
        c.sub_core_grid,
        c.skip_padding_fill);
}

Tensor run_new(const ReshapeCase& c, const Tensor& input) {
    return detail::reshape_new(
        input,
        c.logical_shape,
        c.padded_shape,
        c.memory_config,
        c.pad_value,
        TileReshapeMapMode::CACHE,
        c.sub_core_grid,
        c.skip_padding_fill);
}

bool fits_device(const ReshapeCase& c, const tt::tt_metal::distributed::MeshDevice& device) {
    const auto grid = device.compute_with_storage_grid_size();
    const auto inside = [&](const CoreRangeSet& cores) {
        const auto box = cores.bounding_box();
        return box.end_coord.x < grid.x && box.end_coord.y < grid.y;
    };
    const auto& input_config = c.input_spec.memory_config();
    if (input_config.shard_spec().has_value() && !inside(input_config.shard_spec()->grid)) {
        return false;
    }
    if (input_config.nd_shard_spec().has_value() && !inside(input_config.nd_shard_spec()->grid)) {
        return false;
    }
    return !(
        c.memory_config.has_value() && c.memory_config->shard_spec().has_value() &&
        !inside(c.memory_config->shard_spec()->grid));
}

class ReshapeNewBenchmark : public TTNNFixtureWithSuiteDevice<ReshapeNewBenchmark> {
protected:
    // A reshape preserves row-major logical order, so the input's own elements are the oracle. Block float is
    // re-blocked by the reshape (shared exponents change with the tile rows), so it is checked against legacy only.
    void check(const ReshapeCase& c, const Tensor& input) {
        const Tensor candidate = run_new(c, input);
        EXPECT_EQ(candidate.logical_shape(), c.logical_shape);
        if (!is_block_float(input.dtype())) {
            const auto expected = logical_bits(input);
            const auto actual = logical_bits(candidate);
            ASSERT_EQ(expected.size(), actual.size());
            const auto mismatch = std::mismatch(expected.begin(), expected.end(), actual.begin());
            EXPECT_TRUE(mismatch.first == expected.end())
                << "new reshape differs from the input at flat index " << (mismatch.first - expected.begin())
                << ": input 0x" << std::hex << *mismatch.first << ", output 0x" << *mismatch.second;
        }
        if (c.legacy == LegacyStatus::Exact) {
            const Tensor legacy = run_legacy(c, input);
            const ReshapeRequest request{
                input,
                c.logical_shape,
                c.padded_shape,
                c.memory_config,
                c.pad_value,
                c.sub_core_grid,
                c.skip_padding_fill};
            EXPECT_NO_THROW(detail::compare_reshape_results(request, legacy, candidate));
        }
    }

    template <typename Body>
    void for_each_case(Body body) {
        for (const auto& c : all_cases()) {
            SCOPED_TRACE(c.name);
            if (!fits_device(c, *device_)) {
                std::cout << "[ SKIPPED  ] " << c.name << ": shard grid exceeds the device grid\n";
                continue;
            }
            body(c);
        }
    }
};

TEST_F(ReshapeNewBenchmark, CorrectnessNonCached) {
    for_each_case([&](const ReshapeCase& c) {
        device_->disable_and_clear_program_cache();
        check(c, make_input(c.input_spec, device_, 1));
        device_->enable_program_cache();
    });
}

// The second dispatch uses different tensors of the same spec, so buffer addresses change between the miss and
// the hit, and must not add a program cache entry for the new implementation.
TEST_F(ReshapeNewBenchmark, CorrectnessCached) {
    for_each_case([&](const ReshapeCase& c) {
        check(c, make_input(c.input_spec, device_, 1));
        const auto entries = device_->num_program_cache_entries();
        const Tensor second_input = make_input(c.input_spec, device_, 2);
        const Tensor second_output = run_new(c, second_input);
        EXPECT_EQ(device_->num_program_cache_entries(), entries) << "new reshape missed the cache on a same-spec call";
        check(c, second_input);
    });
}

uint32_t iterations_from_env(const char* name, uint32_t fallback) {
    const char* value = std::getenv(name);
    return value == nullptr ? fallback : static_cast<uint32_t>(std::stoul(value));
}

TEST_F(ReshapeNewBenchmark, DispatchPerformance) {
    const uint32_t iterations = iterations_from_env("TT_RESHAPE_BENCH_ITERATIONS", 2000);
    auto time_ns = [&](auto implementation, const ReshapeCase& c, const Tensor& input) {
        implementation(c, input);
        tt::tt_metal::distributed::Finish(device_->mesh_command_queue());
        const auto start = std::chrono::steady_clock::now();
        for (uint32_t i = 0; i < iterations; ++i) {
            const Tensor output = implementation(c, input);
        }
        tt::tt_metal::distributed::Finish(device_->mesh_command_queue());
        return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count() / iterations;
    };

    for_each_case([&](const ReshapeCase& c) {
        const Tensor input = make_input(c.input_spec, device_, 1);
        // New first, so instruction-cache warmup does not favor it.
        const double new_ns = time_ns(run_new, c, input);
        if (c.legacy == LegacyStatus::Unsupported) {
            std::cout << fmt::format("{:<32} new {:>10.0f} ns/call (legacy unsupported)\n", c.name, new_ns);
            return;
        }
        const double legacy_ns = time_ns(run_legacy, c, input);
        const double overhead = (new_ns / legacy_ns - 1.0) * 100.0;
        std::cout << fmt::format(
            "{:<32} new {:>10.0f} ns/call  legacy {:>10.0f} ns/call  overhead {:+.1f}%\n",
            c.name,
            new_ns,
            legacy_ns,
            overhead);
        EXPECT_LT(overhead, 5.0) << c.name;
    });
}

uint64_t latest_kernel_duration_ns() {
    uint64_t total = 0;
    for (const auto& [chip, programs] : tt::tt_metal::experimental::GetLatestProgramsPerfData()) {
        for (const auto& program : programs) {
            if (const auto it = program.program_analyses_results.find(kKernelDurationAnalysis);
                it != program.program_analyses_results.end()) {
                total += it->second.duration;
            }
        }
    }
    return total;
}

// Sums every program a call launches: legacy issues several for some cases (block float, layout round trips).
TEST_F(ReshapeNewBenchmark, DeviceKernelTime) {
    if (std::getenv("TT_METAL_DEVICE_PROFILER") == nullptr) {
        GTEST_SKIP() << "needs TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 "
                        "TT_METAL_PROFILER_CPP_POST_PROCESS=1";
    }
    const uint32_t iterations = iterations_from_env("TT_RESHAPE_BENCH_DEVICE_ITERATIONS", 10);
    auto kernel_ns = [&](auto implementation, const ReshapeCase& c, const Tensor& input) {
        implementation(c, input);
        tt::tt_metal::distributed::Finish(device_->mesh_command_queue());
        tt::tt_metal::ReadMeshDeviceProfilerResults(*device_);
        for (uint32_t i = 0; i < iterations; ++i) {
            const Tensor output = implementation(c, input);
        }
        tt::tt_metal::distributed::Finish(device_->mesh_command_queue());
        tt::tt_metal::ReadMeshDeviceProfilerResults(*device_);
        return static_cast<double>(latest_kernel_duration_ns()) / iterations;
    };

    for_each_case([&](const ReshapeCase& c) {
        const Tensor input = make_input(c.input_spec, device_, 1);
        const double new_ns = kernel_ns(run_new, c, input);
        if (c.legacy == LegacyStatus::Unsupported) {
            std::cout << fmt::format("{:<32} new {:>10.0f} device ns/call (legacy unsupported)\n", c.name, new_ns);
            return;
        }
        const double legacy_ns = kernel_ns(run_legacy, c, input);
        std::cout << fmt::format(
            "{:<32} new {:>10.0f} device ns/call  legacy {:>10.0f} device ns/call  speedup {:.2f}x\n",
            c.name,
            new_ns,
            legacy_ns,
            new_ns > 0 ? legacy_ns / new_ns : 0.0);
    });
}

}  // namespace
}  // namespace ttnn::operations::data_movement::reshape_benchmark
