// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Integration coverage for the dense 1D/2D multicast matmul factories on their active Metal 2.0
// path. Each case first proves that ttnn::matmul's factory selection reaches the
// create_program_artifacts() implementation under test, then inspects the ProgramSpec it emits:
// every multicast semaphore, CT/RT argument block, and resource binding must be helper-owned, and
// the legacy hand-built multicast resources and arguments must be gone. The same configuration
// then runs through ttnn::matmul and is checked against a host reference.
//
// Set TT_MCAST_ARGUMENT_AUDIT=1 to also print one "MCAST_ARGUMENT_AUDIT <json>" line per case:
// the complete emitted argument snapshot plus warmed artifact-construction timings.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/shape.hpp>
#include "common_test_utils.hpp"
#include "ttnn/kernel_lib/mcast/mcast_protocol.hpp"
#include "ttnn/device.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::operations::matmul::mcast_spec_test {

namespace m2 = tt::tt_metal::experimental;
namespace wire = dataflow_kernel_lib::mcast_wire;
using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRange;
using tt::tt_metal::CoreRangeSet;

class MatmulMcastSpec : public TTNNFixtureWithSuiteDevice<MatmulMcastSpec> {};

struct McastSpecCase {
    std::string name;
    bool two_d = false;
    bool mcast_in0 = true;         // 1D only
    bool transpose_mcast = false;  // 2D only
    // Width (1D) or block (2D) sharded in0 selects the rotating in0 sender.
    std::optional<CoreCoord> in0_shard_grid;
    CoreCoord grid;
    uint32_t M = 0;
    uint32_t K = 0;
    uint32_t N = 0;
    uint32_t in0_block_w = 0;
    uint32_t per_core_M = 0;
    uint32_t per_core_N = 0;
    // Senders each rotating in0 channel cycles through: shard count along the multicast line.
    uint32_t in0_rotating_span = 0;
};

std::ostream& operator<<(std::ostream& os, const McastSpecCase& c) { return os << c.name; }

// Which channel each Metal 2.0 matmul dataflow kernel carries, and whether it may only receive.
struct ChannelKernel {
    std::string prefix;
    bool receiver_only;
};

const std::map<std::string, ChannelKernel>& channel_kernels() {
    static const std::map<std::string, ChannelKernel> kernels = {
        {"in0_sender", {"in0", false}},
        {"in0_mcast_no_work", {"in0", false}},
        {"in0_no_work_in_receiver", {"in0", false}},
        {"in0_no_work_not_in_receiver", {"in0", false}},
        {"in0_receiver", {"in0", true}},
        {"in0_receiver_other_noc", {"in0", true}},
        {"in1_sender_writer", {"in1", false}},
        {"in1_receiver_writer", {"in1", true}},
        {"in1_receiver_writer_other_noc", {"in1", true}},
    };
    return kernels;
}

std::string helper_name(const std::string& prefix, const std::string& field) { return prefix + "_mcast_" + field; }

CoreRangeSet node_set(const m2::Nodes& nodes) {
    return std::visit(
        [](const auto& value) -> CoreRangeSet {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, CoreCoord>) {
                return CoreRangeSet(CoreRange(value, value));
            } else if constexpr (std::is_same_v<T, CoreRange>) {
                return CoreRangeSet(value);
            } else {
                return value;
            }
        },
        nodes);
}

CoreRangeSet kernel_placement(const m2::ProgramSpec& spec, const m2::KernelSpecName& kernel) {
    CoreRangeSet placement;
    for (const auto& unit : spec.work_units) {
        if (std::find(unit.kernels.begin(), unit.kernels.end(), kernel) != unit.kernels.end()) {
            placement = placement.merge(node_set(unit.target_nodes));
        }
    }
    return placement;
}

const m2::ProgramRunArgs::KernelRunArgs* kernel_run_args(
    const m2::ProgramRunArgs& run_args, const m2::KernelSpecName& kernel) {
    for (const auto& args : run_args.kernel_run_args) {
        if (args.kernel == kernel) {
            return &args;
        }
    }
    return nullptr;
}

Tensor make_input(
    tt::tt_metal::distributed::MeshDevice& device,
    const ttnn::Shape& shape,
    const std::vector<float>& data,
    const MemoryConfig& memory_config) {
    const TensorLayout layout(DataType::BFLOAT16, PageConfig(Layout::TILE), memory_config);
    return Tensor::from_vector(data, tt::tt_metal::TensorSpec(shape, layout))
        .to_device(&device, memory_config, ttnn::QueueId(0));
}

MemoryConfig in0_memory_config(const McastSpecCase& c) {
    if (!c.in0_shard_grid) {
        return MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    }
    const CoreCoord shard_grid = *c.in0_shard_grid;
    const CoreRangeSet cores(CoreRange({0, 0}, {shard_grid.x - 1, shard_grid.y - 1}));
    if (!c.two_d) {
        return MemoryConfig{
            tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED,
            BufferType::L1,
            tt::tt_metal::ShardSpec(
                cores,
                {c.M, c.K / static_cast<uint32_t>(shard_grid.x * shard_grid.y)},
                tt::tt_metal::ShardOrientation::ROW_MAJOR)};
    }
    // Transposed multicast lays M blocks along x and K blocks along y.
    const uint32_t m_blocks = c.transpose_mcast ? shard_grid.x : shard_grid.y;
    const uint32_t k_blocks = c.transpose_mcast ? shard_grid.y : shard_grid.x;
    return MemoryConfig{
        tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED,
        BufferType::L1,
        tt::tt_metal::ShardSpec(
            cores,
            {c.M / m_blocks, c.K / k_blocks},
            c.transpose_mcast ? tt::tt_metal::ShardOrientation::COL_MAJOR : tt::tt_metal::ShardOrientation::ROW_MAJOR)};
}

MatmulProgramConfig program_config(const McastSpecCase& c) {
    const CoreRangeSet worker_cores(CoreRange({0, 0}, {c.grid.x - 1, c.grid.y - 1}));
    if (c.two_d) {
        return MatmulMultiCoreReuseMultiCastProgramConfig{
            .compute_with_storage_grid_size = c.grid,
            .in0_block_w = c.in0_block_w,
            .out_subblock_h = 1,
            .out_subblock_w = 1,
            .out_block_h = c.per_core_M,
            .out_block_w = c.per_core_N,
            .per_core_M = c.per_core_M,
            .per_core_N = c.per_core_N,
            .transpose_mcast = c.transpose_mcast,
            .fuse_batch = true,
            .allowed_worker_cores = worker_cores,
        };
    }
    return MatmulMultiCoreReuseMultiCast1DProgramConfig{
        .compute_with_storage_grid_size = c.grid,
        .in0_block_w = c.in0_block_w,
        .out_subblock_h = 1,
        .out_subblock_w = 1,
        .out_block_h = c.per_core_M,
        .out_block_w = c.per_core_N,
        .per_core_M = c.per_core_M,
        .per_core_N = c.per_core_N,
        .fuse_batch = true,
        .mcast_in0 = c.mcast_in0,
        .allowed_worker_cores = worker_cores,
    };
}

std::vector<float> cpu_matmul(
    const std::vector<float>& a, const std::vector<float>& b, uint32_t M, uint32_t K, uint32_t N) {
    std::vector<float> c(static_cast<size_t>(M) * N, 0.0f);
    for (uint32_t i = 0; i < M; ++i) {
        for (uint32_t k = 0; k < K; ++k) {
            const float aik = a[static_cast<size_t>(i) * K + k];
            for (uint32_t j = 0; j < N; ++j) {
                c[static_cast<size_t>(i) * N + j] += aik * b[static_cast<size_t>(k) * N + j];
            }
        }
    }
    return c;
}

std::vector<float> to_float_vector(const Tensor& t) {
    const auto values = t.to_vector<bfloat16>();
    std::vector<float> out(values.size());
    std::transform(values.begin(), values.end(), out.begin(), [](bfloat16 x) { return static_cast<float>(x); });
    return out;
}

std::string node_key(const CoreCoord& node) { return std::to_string(node.x) + "," + std::to_string(node.y); }

nlohmann::json kernel_snapshot(
    const m2::ProgramSpec& spec, const m2::ProgramRunArgs& run_args, const m2::KernelSpec& k) {
    nlohmann::json named_ct = nlohmann::json::object();
    for (const auto& [name, value] : k.compile_time_args) {
        named_ct[name] = value;
    }
    nlohmann::json defines = nlohmann::json::object();
    for (const auto& [name, value] : k.compiler_options.defines) {
        defines[name] = value;
    }
    nlohmann::json semaphores = nlohmann::json::object();
    for (const auto& binding : k.semaphore_bindings) {
        semaphores[binding.accessor_name] = binding.semaphore_spec_name.get();
    }
    nlohmann::json runtime = nlohmann::json::object();
    uint32_t aggregate_rt_words = 0;
    const auto placement = kernel_placement(spec, k.unique_id);
    const auto* args = kernel_run_args(run_args, k.unique_id);
    for (const auto& node : tt::tt_metal::corerange_to_cores(placement, std::nullopt, true)) {
        nlohmann::json named = nlohmann::json::object();
        if (args != nullptr) {
            for (const auto& [name, values] : args->runtime_arg_values) {
                if (const auto it = values.find(node); it != values.end()) {
                    named[name] = it->second;
                }
            }
        }
        std::vector<uint32_t> varargs;
        if (args != nullptr) {
            if (const auto it = args->advanced_options.runtime_varargs.find(node);
                it != args->advanced_options.runtime_varargs.end()) {
                varargs = it->second;
            }
        }
        aggregate_rt_words += static_cast<uint32_t>(named.size() + varargs.size());
        runtime[node_key(node)] = {{"named", std::move(named)}, {"varargs", std::move(varargs)}};
    }
    const std::string source = std::holds_alternative<std::filesystem::path>(k.source)
                                   ? std::get<std::filesystem::path>(k.source).string()
                                   : std::string("<inline>");
    return {
        {"kernel", k.unique_id.get()},
        {"source", source},
        {"placement", placement.str()},
        {"named_ct", std::move(named_ct)},
        {"ct_varargs", k.advanced_options.compile_time_varargs},
        {"ct_words", k.compile_time_args.size() + k.advanced_options.compile_time_varargs.size()},
        {"defines", std::move(defines)},
        {"semaphore_bindings", std::move(semaphores)},
        {"runtime_arg_names", k.runtime_arg_schema.runtime_arg_names},
        {"num_runtime_varargs", k.advanced_options.num_runtime_varargs},
        {"runtime", std::move(runtime)},
        {"aggregate_rt_words", aggregate_rt_words},
    };
}

template <typename Factory>
ttnn::device_operation::ProgramArtifacts build_artifacts(
    const ttnn::prim::MatmulParams& attributes, const ttnn::prim::MatmulInputs& inputs, std::vector<Tensor>& outputs) {
    return Factory::create_program_artifacts(attributes, inputs, outputs);
}

// Checks one multicast channel of the emitted spec and records whether it is present.
void expect_helper_channel(
    const m2::ProgramSpec& spec,
    const m2::ProgramRunArgs& run_args,
    const std::string& prefix,
    const McastSpecCase& c,
    std::optional<bool>& present) {
    const std::array<std::string, 3> roles{"data_ready", "consumer_ready", "signal_source"};
    for (const auto& kernel : spec.kernels) {
        const auto channel = channel_kernels().find(kernel.unique_id.get());
        if (channel == channel_kernels().end() || channel->second.prefix != prefix) {
            continue;
        }
        SCOPED_TRACE(kernel.unique_id.get());
        const auto ct_base = kernel.compile_time_args.find(helper_name(prefix, "ct_base"));
        const auto rt_base = kernel.compile_time_args.find(helper_name(prefix, "rt_base"));
        ASSERT_NE(ct_base, kernel.compile_time_args.end()) << "no helper CT block";
        ASSERT_NE(rt_base, kernel.compile_time_args.end()) << "no helper RT block";
        const auto& ct = kernel.advanced_options.compile_time_varargs;
        ASSERT_LT(ct_base->second, ct.size());
        const uint32_t control = ct[ct_base->second];
        const bool kernel_present = control != wire::ABSENT;
        if (present.has_value()) {
            EXPECT_EQ(*present, kernel_present) << "kernels disagree on the channel's presence";
        }
        present = kernel_present;
        for (const auto& role : roles) {
            EXPECT_TRUE(kernel.compiler_options.defines.contains(helper_name(prefix, role) + "_type"));
        }
        if (!kernel_present) {
            EXPECT_TRUE(kernel.semaphore_bindings.empty());
            continue;
        }
        const std::vector<uint32_t> words(ct.begin() + ct_base->second, ct.end());
        const auto metadata = wire::decode_compile_time_metadata(words, false);
        EXPECT_EQ(control, wire::compile_time_control(metadata));
        EXPECT_NE(metadata.mcast.flags & wire::PRE_HANDSHAKE, 0u);
        const bool can_send = (metadata.kernel.capabilities & wire::CAN_SEND) != 0;
        const bool can_receive = (metadata.kernel.capabilities & wire::CAN_RECEIVE) != 0;
        if (channel->second.receiver_only) {
            EXPECT_FALSE(can_send);
            EXPECT_TRUE(can_receive);
        } else {
            EXPECT_TRUE(can_send);
        }
        const bool rotating = prefix == "in0" && c.in0_shard_grid.has_value();
        EXPECT_EQ(metadata.mcast.rotating_span, rotating ? c.in0_rotating_span : 0u);

        // Helper-owned semaphores, bound by the helper's accessor names.
        std::set<std::string> accessors;
        for (const auto& binding : kernel.semaphore_bindings) {
            accessors.insert(binding.accessor_name);
            EXPECT_EQ(binding.semaphore_spec_name.get(), binding.accessor_name);
        }
        EXPECT_EQ(
            accessors,
            (std::set<std::string>{helper_name(prefix, "data_ready"), helper_name(prefix, "consumer_ready")}));

        // Every placed node carries the operation prefix followed by exactly one helper RT block.
        const uint32_t rt_words = wire::RuntimeLayout(metadata).words;
        EXPECT_EQ(kernel.advanced_options.num_runtime_varargs, rt_base->second + rt_words);
        EXPECT_TRUE(kernel.advanced_options.num_runtime_varargs_per_node.empty());
        const auto* args = kernel_run_args(run_args, kernel.unique_id);
        ASSERT_NE(args, nullptr);
        const auto placement = kernel_placement(spec, kernel.unique_id);
        for (const auto& node : tt::tt_metal::corerange_to_cores(placement)) {
            const auto it = args->advanced_options.runtime_varargs.find(node);
            ASSERT_NE(it, args->advanced_options.runtime_varargs.end()) << node.str();
            EXPECT_EQ(it->second.size(), kernel.advanced_options.num_runtime_varargs) << node.str();
        }

        for (const auto& sem : spec.semaphores) {
            if (accessors.contains(sem.unique_id.get())) {
                EXPECT_TRUE(placement.subtract(node_set(sem.target_nodes)).empty())
                    << sem.unique_id.get() << " does not cover " << placement.str();
                EXPECT_EQ(sem.advanced_options.initial_value, 0u);
            }
        }
    }
}

class MatmulMcastSpecParam : public MatmulMcastSpec, public ::testing::WithParamInterface<McastSpecCase> {};

TEST_P(MatmulMcastSpecParam, ActivePathEmitsHelperOwnedMulticast) {
    const McastSpecCase& c = GetParam();
    auto& device = *device_;
    const auto compute_grid = device.compute_with_storage_grid_size();
    if (c.grid.x > compute_grid.x || c.grid.y > compute_grid.y) {
        GTEST_SKIP() << "needs a " << c.grid.str() << " worker grid";
    }

    std::mt19937 rng(11);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> a_host(static_cast<size_t>(c.M) * c.K);
    std::vector<float> b_host(static_cast<size_t>(c.K) * c.N);
    std::generate(a_host.begin(), a_host.end(), [&] { return dist(rng); });
    std::generate(b_host.begin(), b_host.end(), [&] { return dist(rng); });
    const Tensor a = make_input(device, ttnn::Shape({1, 1, c.M, c.K}), a_host, in0_memory_config(c));
    const Tensor b = make_input(
        device,
        ttnn::Shape({1, 1, c.K, c.N}),
        b_host,
        MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, BufferType::DRAM});
    const auto config = program_config(c);
    const MemoryConfig dram{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};

    ttnn::prim::MatmulParams params{.program_config = config, .output_mem_config = dram};
    const auto attributes = ttnn::prim::create_matmul_attributes(a, b, params, {});
    const ttnn::prim::MatmulInputs inputs{{a, b}, {std::nullopt}, {}};
    ttnn::prim::MatmulDeviceOperation::validate_on_program_cache_miss(attributes, inputs);
    const auto factory = ttnn::prim::MatmulDeviceOperation::select_program_factory(attributes, inputs);
    if (c.two_d) {
        ASSERT_TRUE(std::holds_alternative<ttnn::prim::MatmulMultiCoreReuseMcast2DProgramFactory>(factory));
    } else {
        ASSERT_TRUE(std::holds_alternative<ttnn::prim::MatmulMultiCoreReuseMcast1DProgramFactory>(factory));
    }
    auto outputs = ttnn::prim::MatmulDeviceOperation::create_output_tensors(attributes, inputs);
    const auto build = [&] {
        return c.two_d
                   ? build_artifacts<ttnn::prim::MatmulMultiCoreReuseMcast2DProgramFactory>(attributes, inputs, outputs)
                   : build_artifacts<ttnn::prim::MatmulMultiCoreReuseMcast1DProgramFactory>(
                         attributes, inputs, outputs);
    };
    const auto artifacts = build();
    const auto& spec = artifacts.spec;
    const auto& run_args = artifacts.run_params;

    std::set<std::string> semaphore_names;
    for (const auto& sem : spec.semaphores) {
        semaphore_names.insert(sem.unique_id.get());
    }
    if (const char* audit = std::getenv("TT_MCAST_ARGUMENT_AUDIT"); audit != nullptr && std::string(audit) == "1") {
        std::vector<int64_t> construction_ns;
        for (int iteration = 0; iteration < 25; ++iteration) {
            const auto start = std::chrono::steady_clock::now();
            const auto rebuilt = build();
            const auto elapsed = std::chrono::steady_clock::now() - start;
            if (iteration >= 5) {
                construction_ns.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
            }
        }
        nlohmann::json kernels = nlohmann::json::array();
        for (const auto& kernel : spec.kernels) {
            kernels.push_back(kernel_snapshot(spec, run_args, kernel));
        }
        nlohmann::json snapshot{
            {"case", c.name},
            {"shape_mkn", {c.M, c.K, c.N}},
            {"grid", {c.grid.x, c.grid.y}},
            {"semaphores", semaphore_names},
            {"construction_ns", construction_ns},
            {"kernels", std::move(kernels)},
        };
        std::cout << "MCAST_ARGUMENT_AUDIT " << snapshot.dump() << std::endl;
    }

    // No legacy multicast resources or arguments survive the port.
    const std::set<std::string> legacy_ct{
        "in0_mcast_num_dests",
        "in0_mcast_num_cores",
        "in1_mcast_num_dests",
        "in1_mcast_num_cores",
        "num_x",
        "num_y",
        "transpose_mcast",
        "core_in_in0_receiver_mcast_grid"};
    for (const auto& kernel : spec.kernels) {
        SCOPED_TRACE(kernel.unique_id.get());
        for (const auto& [name, value] : kernel.compile_time_args) {
            EXPECT_FALSE(legacy_ct.contains(name)) << name;
        }
        for (const auto& name : kernel.runtime_arg_schema.runtime_arg_names) {
            EXPECT_EQ(name.find("_mcast_dest_noc_"), std::string::npos) << name;
            EXPECT_EQ(name.find("_mcast_sender_noc_"), std::string::npos) << name;
        }
        EXPECT_FALSE(kernel.compiler_options.defines.contains("SKIP_MCAST"));
        for (const auto& binding : kernel.semaphore_bindings) {
            EXPECT_TRUE(
                binding.accessor_name.starts_with("in0_mcast_") || binding.accessor_name.starts_with("in1_mcast_"))
                << binding.accessor_name;
        }
    }

    std::optional<bool> in0_channel;
    std::optional<bool> in1_channel;
    expect_helper_channel(spec, run_args, "in0", c, in0_channel);
    expect_helper_channel(spec, run_args, "in1", c, in1_channel);
    ASSERT_TRUE(in0_channel.has_value()) << "no kernel carries the in0 channel";
    ASSERT_TRUE(in1_channel.has_value()) << "no kernel carries the in1 channel";
    const bool in0_present = *in0_channel;
    const bool in1_present = *in1_channel;
    EXPECT_EQ(in0_present, c.two_d || c.mcast_in0);
    EXPECT_EQ(in1_present, c.two_d || !c.mcast_in0);

    // Only helper semaphores remain: data-ready and consumer-ready for each present channel.
    std::set<std::string> expected_semaphores;
    for (const auto& [prefix, present] : {std::pair{"in0", in0_present}, std::pair{"in1", in1_present}}) {
        if (present) {
            expected_semaphores.insert(helper_name(prefix, "data_ready"));
            expected_semaphores.insert(helper_name(prefix, "consumer_ready"));
        }
    }
    EXPECT_EQ(semaphore_names, expected_semaphores);

    const Tensor output = ttnn::matmul(a, b, false, false, dram, std::nullopt, config);
    const auto expected = cpu_matmul(to_float_vector(a), to_float_vector(b), c.M, c.K, c.N);
    EXPECT_GE(ttnn::test_utils::pcc(to_float_vector(output), expected), 0.999f);
}

INSTANTIATE_TEST_SUITE_P(
    McastDenseMatmul,
    MatmulMcastSpecParam,
    ::testing::Values(
        McastSpecCase{
            .name = "fixed_1d_mcast_in0",
            .grid = {8, 1},
            .M = 256,
            .K = 1024,
            .N = 1024,
            .in0_block_w = 4,
            .per_core_M = 8,
            .per_core_N = 4},
        McastSpecCase{
            .name = "rotating_1d_mcast_in0",
            .in0_shard_grid = CoreCoord{8, 1},
            .grid = {8, 1},
            .M = 256,
            .K = 1024,
            .N = 1024,
            .in0_block_w = 4,
            .per_core_M = 8,
            .per_core_N = 4,
            .in0_rotating_span = 8},
        // Sixteen width shards feed eight output cores: the second row only sends, from outside the
        // receiver rectangle.
        McastSpecCase{
            .name = "rotating_1d_mcast_in0_sender_only_row",
            .in0_shard_grid = CoreCoord{8, 2},
            .grid = {8, 2},
            .M = 256,
            .K = 2048,
            .N = 1024,
            .in0_block_w = 4,
            .per_core_M = 8,
            .per_core_N = 4,
            .in0_rotating_span = 16},
        McastSpecCase{
            .name = "fixed_1d_mcast_in1",
            .mcast_in0 = false,
            .grid = {8, 1},
            .M = 256,
            .K = 256,
            .N = 128,
            .in0_block_w = 4,
            .per_core_M = 1,
            .per_core_N = 4},
        McastSpecCase{
            .name = "fixed_2d",
            .two_d = true,
            .grid = {4, 4},
            .M = 256,
            .K = 1024,
            .N = 256,
            .in0_block_w = 4,
            .per_core_M = 2,
            .per_core_N = 2},
        McastSpecCase{
            .name = "fixed_2d_transpose_mcast",
            .two_d = true,
            .transpose_mcast = true,
            .grid = {4, 4},
            .M = 256,
            .K = 1024,
            .N = 256,
            .in0_block_w = 4,
            .per_core_M = 2,
            .per_core_N = 2},
        McastSpecCase{
            .name = "rotating_2d",
            .two_d = true,
            .in0_shard_grid = CoreCoord{4, 4},
            .grid = {4, 4},
            .M = 256,
            .K = 1024,
            .N = 256,
            .in0_block_w = 4,
            .per_core_M = 2,
            .per_core_N = 2,
            .in0_rotating_span = 4},
        McastSpecCase{
            .name = "rotating_2d_transpose_mcast",
            .two_d = true,
            .transpose_mcast = true,
            .in0_shard_grid = CoreCoord{4, 4},
            .grid = {4, 4},
            .M = 256,
            .K = 1024,
            .N = 256,
            .in0_block_w = 4,
            .per_core_M = 2,
            .per_core_N = 2,
            .in0_rotating_span = 4}),
    [](const ::testing::TestParamInfo<McastSpecCase>& info) { return info.param.name; });

}  // namespace ttnn::operations::matmul::mcast_spec_test
