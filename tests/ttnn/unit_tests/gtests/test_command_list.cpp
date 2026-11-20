// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Command lists need no warm-up run: each test opens a fresh device and first calls its op while recording.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include <tt-metalium/workload/run_args/program_run_args.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <internal/mesh_device.hpp>

#include "ttnn/command_list.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/normalization/softmax/device/softmax_device_operation.hpp"
#include "ttnn/operations/normalization/softmax/softmax.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::command_list::test {

class CommandListTest : public TTNNFixtureWithDevice {
protected:
    void SetUp() override {
        if (!check_dispatch_mode()) {
            GTEST_SKIP() << "Command lists require fast dispatch";
        }
        TTNNFixtureWithDevice::SetUp();
    }

    Tensor make_input(const std::vector<float>& values) const {
        const TensorSpec spec(
            ttnn::Shape{1, 1, 32, 64}, TensorLayout(DataType::BFLOAT16, Layout::TILE, MemoryConfig{}));
        return Tensor::from_vector(values, spec, device_);
    }

    // Records a single-core row-major ttnn::pad, whose factory allocates an L1 pad-value tensor and hands it to the
    // program cache, then replays the list and checks the output. after_build runs between build() and the replay.
    void expect_pad_replays(const std::function<void()>& after_build) const {
        constexpr uint32_t width = 32;
        constexpr float pad_value = 3.0f;
        std::vector<float> values(32 * width);
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = static_cast<float>(static_cast<int>(i % 11) - 5);
        }
        const Tensor input = Tensor::from_vector(
            values,
            TensorSpec(
                ttnn::Shape{1, 1, 32, width}, TensorLayout(DataType::BFLOAT16, Layout::ROW_MAJOR, MemoryConfig{})),
            device_);

        CommandListBuilder builder(*device_);
        Tensor output;
        {
            auto recording = builder.record_scope();
            output = ttnn::pad(
                input,
                ttsl::SmallVector<std::array<uint32_t, 2>>{{0, 0}, {0, 0}, {0, 0}, {0, width}},
                pad_value,
                /*use_multicore=*/false);
        }
        CommandList command_list = builder.build(QueueId{0});
        builder.deallocate();
        after_build();

        // Same spec as pad's pad-value tensor, so it takes that tensor's L1 if the list let it be freed.
        const Tensor clobber = Tensor::from_vector(
            std::vector<uint32_t>(16, 0),
            TensorSpec(
                ttnn::Shape{1, 1, 1, 16},
                TensorLayout(
                    DataType::UINT32,
                    Layout::ROW_MAJOR,
                    MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::L1})),
            device_);

        command_list.enqueue(/*blocking=*/true);

        const std::vector<float> result = output.to_vector<float>();
        ASSERT_EQ(result.size(), values.size() * 2);
        for (uint32_t row = 0; row < 32; ++row) {
            for (uint32_t col = 0; col < 2 * width; ++col) {
                const float expected = col < width ? values[(row * width) + col] : pad_value;
                EXPECT_EQ(result[(row * 2 * width) + col], expected) << "at row " << row << ", column " << col;
            }
        }

        const auto l1_allocated = [&] {
            return device_->allocator()->get_statistics(BufferType::L1).total_allocated_bytes;
        };
        const size_t l1_held = l1_allocated();
        command_list.deallocate();
        EXPECT_LT(l1_allocated(), l1_held) << "The list should hold pad's pad-value tensor until it is deallocated";
    }
};

TEST_F(CommandListTest, RecordBuildReplay) {
    std::vector<float> values(32 * 64);
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>(static_cast<int>(i % 17) - 8);
    }
    const Tensor input = make_input(values);

    CommandListBuilder builder(*device_);
    Tensor output;
    {
        auto recording = builder.record_scope();
        EXPECT_TRUE(builder.is_recording());
        output = ttnn::relu(input);
    }
    EXPECT_FALSE(builder.is_recording());

    CommandList command_list = builder.build(QueueId{0});
    builder.deallocate();
    EXPECT_EQ(*command_list.cq_id(), 0);

    command_list.enqueue(/*blocking=*/true);

    const std::vector<float> result = output.to_vector<float>();
    ASSERT_EQ(result.size(), values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        EXPECT_EQ(result[i], std::max(values[i], 0.0f)) << "at index " << i;
    }
    command_list.deallocate();
}

TEST_F(CommandListTest, PadReplaysAfterProgramCacheClear) {
    expect_pad_replays([this] { device_->clear_program_cache(); });
}

TEST_F(CommandListTest, PadReplaysWithProgramCacheDisabled) {
    device_->disable_and_clear_program_cache();
    expect_pad_replays([] {});
}

TEST_F(CommandListTest, LegacyOpIsRejected) {
    const Tensor input = make_input(std::vector<float>(32 * 64, 1.0f));
    CommandListBuilder builder(*device_);
    {
        auto recording = builder.record_scope();
        try {
            (void)ttnn::add(input, input);
            FAIL() << "Recording a legacy-factory op should throw";
        } catch (const std::exception& e) {
            EXPECT_THAT(std::string(e.what()), ::testing::HasSubstr("cannot be recorded into a command list"));
        }
        EXPECT_TRUE(builder.is_recording());
    }
    EXPECT_THAT(
        [&] { (void)builder.build(); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("An error interrupted recording")));
}

TEST_F(CommandListTest, ExceptionLeavingRecordScopeRequiresClear) {
    CommandListBuilder builder(*device_);
    EXPECT_THAT(
        [&] {
            auto recording = builder.record_scope();
            throw std::runtime_error("user error");
        },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("user error")));
    EXPECT_THAT(
        [&] { (void)builder.build(); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("An error interrupted recording")));
    builder.clear();
    EXPECT_THAT(
        [&] { (void)builder.build(); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("Cannot build an empty CommandList")));
}

TEST_F(CommandListTest, BuilderLifetimeErrors) {
    CommandListBuilder builder(*device_);
    EXPECT_ANY_THROW(builder.build());
    EXPECT_ANY_THROW(CommandListBuilder second_builder(*device_));
    {
        auto recording = builder.record_scope();
        EXPECT_ANY_THROW(builder.begin_recording());
        EXPECT_ANY_THROW(builder.clear());
        EXPECT_ANY_THROW(builder.deallocate());
    }
    builder.deallocate();
    EXPECT_THAT(
        [&] { (void)builder.build(); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("CommandListBuilder has been deallocated")));
}

CmdListScalar real(double value) { return {value}; }
CmdListScalar integer(int64_t value) { return {value}; }

TEST(CommandListParamsTest, EncodersProduceArgumentWords) {
    EXPECT_EQ(encode_uint32(integer(7)).to_uint32(), 7u);
    EXPECT_EQ(encode_uint32(integer(0xFFFFFFFF)).to_uint32(), 0xFFFFFFFFu);
    EXPECT_EQ(encode_int32_bits(integer(-1)).to_uint32(), 0xFFFFFFFFu);
    EXPECT_EQ(encode_float32_bits(real(1.0)).to_uint32(), 0x3F800000u);
    EXPECT_EQ(encode_float32_bits(integer(2)).to_uint32(), 0x40000000u);
    EXPECT_EQ(encode_float32_bits(real(-std::numeric_limits<double>::infinity())).to_uint32(), 0xFF800000u);
    EXPECT_EQ(encode_bfloat16_pair(real(1.0)).to_uint32(), 0x3F803F80u);
    EXPECT_EQ(encode_bfloat16_low(real(1.0)).to_uint32(), 0x00003F80u);
    EXPECT_EQ(encode_bfloat16_high(real(1.0)).to_uint32(), 0x3F800000u);
    EXPECT_EQ(encode_uint16_pair(integer(0x1234)).to_uint32(), 0x12341234u);
    EXPECT_EQ(encode_uint16_low(integer(0x1234)).to_uint32(), 0x00001234u);
    EXPECT_EQ(encode_uint16_high(integer(0x1234)).to_uint32(), 0x12340000u);
    EXPECT_EQ(encode_float8_e4m3_quad(real(1.0)).to_uint32(), 0x38383838u);
}

TEST(CommandListParamsTest, EncodersRejectValuesTheyCannotRepresent) {
    const auto rejects = [](const CmdListEncodeFn& encode, const CmdListScalar& scalar) {
        EXPECT_THAT(
            [&] { (void)encode(scalar); }, ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("needs")));
    };
    rejects(encode_uint32, integer(-1));
    rejects(encode_uint32, integer(int64_t{1} << 32));
    rejects(encode_uint32, real(3.0));
    rejects(encode_uint16_low, real(1.0));
    rejects(encode_int32_bits, integer(int64_t{1} << 31));
    rejects(encode_uint16_low, integer(0x10000));
    rejects(encode_float32_bits, real(1e39));
    rejects(encode_bfloat16_pair, real(-1e39));
    rejects(encode_bfloat16_low, real(3.4e38));
    rejects(encode_float8_e4m3_quad, real(512.0));
    rejects(encode_float8_e4m3_quad, real(-std::numeric_limits<double>::infinity()));
}

TEST_F(CommandListTest, BindsNeedARecordingBuilder) {
    EXPECT_THAT(
        [] { push_cmd_list_binds("op", {{.name = "a", .invoke_name = "x", .value = integer(1)}}); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("op: CmdListParams need a TTNN CommandListBuilder recording on this thread")));
    EXPECT_ANY_THROW(pop_cmd_list_binds(false));

    CommandListBuilder builder(*device_);
    builder.begin_recording();
    push_cmd_list_binds("op", {{.name = "a", .invoke_name = "x", .value = integer(1)}});
    EXPECT_THAT(
        [&] { builder.end_recording(); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Cannot end recording while op still has CmdListParam binds pushed")));
    pop_cmd_list_binds(false);
    builder.end_recording();
}

TEST_F(CommandListTest, FinishAcceptsBindsThatReachedNoProgram) {
    CommandListBuilder builder(*device_);
    auto recording = builder.record_scope();
    CmdListBindScope scope("ttnn.relu", {{.name = "slope", .invoke_name = "slope", .value = real(0.5)}});
    EXPECT_NO_THROW(scope.finish());
    EXPECT_ANY_THROW(pop_cmd_list_binds(false));
}

TEST_F(CommandListTest, BindScopesNest) {
    CommandListBuilder builder(*device_);
    auto recording = builder.record_scope();
    {
        CmdListBindScope outer("outer_op", {{.name = "a", .invoke_name = "x", .value = integer(1)}});
        {
            CmdListBindScope inner("inner_op", {{.name = "b", .invoke_name = "y", .value = integer(2)}});
            EXPECT_NO_THROW(inner.finish());
        }
        CmdListBindScope empty("empty_op", {});
        EXPECT_NO_THROW(empty.finish());
    }
    EXPECT_ANY_THROW(pop_cmd_list_binds(false));
}

TEST_F(CommandListTest, PushRejectsInvalidBinds) {
    CommandListBuilder builder(*device_);
    auto recording = builder.record_scope();
    EXPECT_ANY_THROW(push_cmd_list_binds("op", {{.name = "", .invoke_name = "x", .value = integer(1)}}));
    EXPECT_ANY_THROW(push_cmd_list_binds("op", {{.name = "a", .invoke_name = "", .value = integer(1)}}));
    const Tensor host_tensor = Tensor::from_vector(
        std::vector<float>(32 * 32, 1.0f),
        TensorSpec(ttnn::Shape{32, 32}, TensorLayout(DataType::FLOAT32, Layout::ROW_MAJOR, MemoryConfig{})));
    EXPECT_THAT(
        [&] { push_cmd_list_binds("op", {{.name = "t", .invoke_name = "input_tensor", .value = host_tensor}}); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("must be an allocated device tensor")));
    EXPECT_ANY_THROW(pop_cmd_list_binds(false));
}

namespace m2 = tt::tt_metal;

m2::KernelSpec make_kernel(const std::string& name, tt::tt_metal::DataMovementProcessor processor) {
    m2::KernelSpec kernel{
        .unique_id = m2::KernelSpecName{name},
        .num_threads = 1,
        .hw_config = m2::DataMovementHardwareConfig{
            .config_1xx = m2::DataMovementHardwareConfig::DataMovement1XXConfig{
                .processor = processor,
                .noc = processor == tt::tt_metal::DataMovementProcessor::RISCV_0 ? tt::tt_metal::NOC::NOC_0
                                                                                 : tt::tt_metal::NOC::NOC_1}}};
    // Writes common runtime argument "value" to the L1 address in runtime argument "address".
    kernel.source = "tests/tt_metal/tt_metal/test_kernels/dataflow/command_list_l1_write.cpp";
    kernel.runtime_arg_schema = {.runtime_arg_names = {"address"}, .common_runtime_arg_names = {"value"}};
    return kernel;
}

// A spec and run args shaped like a factory's, for exercising CmdListBinder without a device.
struct BinderFixture {
    m2::ProgramSpec spec;
    m2::ProgramRunArgs run_params;

    BinderFixture() {
        spec.name = "binder_fixture";
        spec.kernels = {
            make_kernel("reader", tt::tt_metal::DataMovementProcessor::RISCV_0),
            make_kernel("writer", tt::tt_metal::DataMovementProcessor::RISCV_1)};
        const TensorSpec tensor_spec(
            ttnn::Shape{32, 32}, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), MemoryConfig{}));
        spec.tensor_parameters = {
            m2::TensorParameter{.unique_id = m2::TensorParamName{"src"}, .spec = tensor_spec},
            m2::TensorParameter{.unique_id = m2::TensorParamName{"dst"}, .spec = tensor_spec}};
        for (const char* kernel : {"reader", "writer"}) {
            m2::ProgramRunArgs::KernelRunArgs kernel_run_args{.kernel = m2::KernelSpecName{kernel}};
            m2::AddRuntimeArgsForNode(kernel_run_args.runtime_arg_values, m2::NodeCoord{0, 0}, {{"address", 0u}});
            m2::AddRuntimeArgsForNode(kernel_run_args.runtime_arg_values, m2::NodeCoord{1, 0}, {{"address", 0u}});
            kernel_run_args.common_runtime_arg_values = {{"value", 0u}};
            run_params.kernel_run_args.push_back(std::move(kernel_run_args));
        }
    }
};

const m2::KernelSpecName kReader{"reader"};
const m2::KernelSpecName kWriter{"writer"};

TEST(CommandListBinderTest, RecordsBindingsWithOneEncodingPerMetalParam) {
    BinderFixture fixture;
    CmdListBinder binder{fixture.spec, fixture.run_params};
    binder.tensor("input_tensor", m2::TensorParamName{"src"});
    binder.tensor("input_tensor", m2::TensorParamName{"dst"});
    binder.param("scale", encode_float32_bits).runtime(kReader, "address").common_runtime(kReader, "value");
    binder.param("scale", encode_uint16_low, "scale_u16")
        .runtime(kWriter, "address", std::vector<m2::NodeCoord>{{1, 0}});
    const CmdListBindings bindings = binder.finish();

    ASSERT_EQ(bindings.tensors.size(), 2u);
    EXPECT_EQ(bindings.tensors[1].program_spec_param_name, m2::TensorParamName{"dst"});
    ASSERT_EQ(bindings.scalars.size(), 2u);
    EXPECT_EQ(bindings.scalars[0].metal_cmd_list_param_name, "scale");
    ASSERT_EQ(bindings.scalars[0].runtime_args.size(), 1u);
    EXPECT_EQ(bindings.scalars[0].runtime_args[0].nodes.size(), 2u);
    EXPECT_EQ(bindings.scalars[0].common_runtime_args.size(), 1u);
    EXPECT_EQ(bindings.scalars[1].metal_cmd_list_param_name, "scale_u16");
    EXPECT_THAT(bindings.scalars[1].runtime_args[0].nodes, ::testing::ElementsAre(m2::NodeCoord{1, 0}));
    EXPECT_EQ(bindings.scalars[1].encode(integer(1)).to_uint32(), 1u);
}

TEST(CommandListBinderTest, RejectsInvalidBindings) {
    BinderFixture fixture;
    const auto rejects = [&](const auto& bind, const std::string& message) {
        CmdListBinder binder{fixture.spec, fixture.run_params};
        EXPECT_THAT([&] { bind(binder); }, ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(message)));
    };
    rejects(
        [](CmdListBinder& b) {
            b.param("scale");
            b.param("scale");
        },
        "Metal param 'scale' is declared twice");
    rejects(
        [](CmdListBinder& b) {
            b.param("p");
            (void)b.finish();
        },
        "Metal param 'p' binds no kernel argument");
    rejects(
        [](CmdListBinder& b) {
            b.param("a").common_runtime(kReader, "value");
            b.param("b").common_runtime(kReader, "value");
        },
        "common runtime argument 'value' of kernel 'reader' is already bound");
    rejects(
        [](CmdListBinder& b) {
            b.tensor("x", m2::TensorParamName{"src"});
            b.tensor("y", m2::TensorParamName{"src"});
        },
        "TensorParameter 'src' is already bound");
    rejects([](CmdListBinder& b) { b.tensor("x", m2::TensorParamName{"nope"}); }, "no TensorParameter 'nope'");
    rejects([](CmdListBinder& b) { b.param("p").runtime(m2::KernelSpecName{"nope"}, "address"); }, "no kernel 'nope'");
    rejects([](CmdListBinder& b) { b.param("p").runtime(kReader, "nope"); }, "has no runtime argument 'nope'");
    rejects(
        [](CmdListBinder& b) { b.param("p").common_runtime(kReader, "address"); },
        "has no common runtime argument 'address'");
    rejects(
        [](CmdListBinder& b) { b.param("p").runtime(kReader, "address", std::vector<m2::NodeCoord>{{5, 5}}); },
        "do not set runtime argument 'address'");
}

class CommandListDeviceParamsTest : public MultiCommandQueueSingleDeviceFixture {
protected:
    Tensor make_tensor(const ttnn::Shape& shape, float offset) const {
        std::vector<float> values(shape.volume());
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = offset + (static_cast<float>(i % 13) / 4.0f);
        }
        const TensorSpec spec(shape, TensorLayout(DataType::BFLOAT16, Layout::TILE, MemoryConfig{}));
        return Tensor::from_vector(values, spec, device_);
    }
};

TEST_F(CommandListDeviceParamsTest, SoftmaxMapsItsKeywordsDirectly) {
    const Tensor input = make_tensor(ttnn::Shape{1, 1, 32, 64}, 0.0f);
    const Tensor mask = make_tensor(ttnn::Shape{1, 1, 32, 64}, -1.0f);
    using Op = ttnn::prim::SoftmaxDeviceOperation;
    const Op::operation_attributes_t attributes{
        .softmax_type = SoftmaxOperationType::ScaleMaskSoftmax,
        .dim = 3,
        .scale = 0.5f,
        .inplace = false,
        .output_mem_config = MemoryConfig{},
        .program_config = SoftmaxDefaultProgramConfig{},
        .is_causal_mask = false,
        .compute_kernel_config = ttnn::init_device_compute_kernel_config(device_->arch(), std::nullopt),
        .is_scale_causal_mask_hw_dims_softmax = false,
        .numeric_stable = true,
        .has_explicit_compute_kernel_config = false};
    const Op::tensor_args_t tensor_args{.input_tensor = input, .mask = mask};
    Tensor output = Op::create_output_tensors(attributes, tensor_args);
    const auto artifacts =
        Op::SoftmaxProgramFactoryAttentionOptimized::create_program_artifacts(attributes, tensor_args, output);

    const CmdListBindings bindings = Op::SoftmaxProgramFactoryAttentionOptimized::map_cmd_list_parameters(
        attributes, tensor_args, artifacts.spec, artifacts.run_params);

    ASSERT_EQ(bindings.tensors.size(), 2u);
    EXPECT_EQ(bindings.tensors[0].invoke_name, "input_tensor");
    EXPECT_EQ(bindings.tensors[0].program_spec_param_name, m2::TensorParamName{"src"});
    EXPECT_EQ(bindings.tensors[1].invoke_name, "mask");
    ASSERT_EQ(bindings.scalars.size(), 1u);
    EXPECT_EQ(bindings.scalars[0].invoke_name, "scale");
    ASSERT_EQ(bindings.scalars[0].runtime_args.size(), 1u);
    EXPECT_EQ(bindings.scalars[0].runtime_args[0].arg_name, "pre_scale");
    EXPECT_FALSE(bindings.scalars[0].runtime_args[0].nodes.empty());
    EXPECT_EQ(bindings.scalars[0].encode(real(0.5)).to_uint32(), std::bit_cast<uint32_t>(0.5f));
}

TEST_F(CommandListDeviceParamsTest, SoftmaxUpdatesInputAndScaleOnEachQueue) {
    const ttnn::Shape shape{1, 1, 64, 64};
    const Tensor input_a = make_tensor(shape, 0.0f);
    const Tensor input_b = make_tensor(shape, 1.5f);
    const Tensor mask = make_tensor(ttnn::Shape{1, 1, 32, 64}, -2.0f);
    const auto softmax = [&](const Tensor& input, float scale) { return ttnn::scale_mask_softmax(input, scale, mask); };
    const std::vector<float> expected_a = softmax(input_a, 0.5f).to_vector<float>();
    const std::vector<float> expected_b = softmax(input_b, 0.25f).to_vector<float>();

    CommandListBuilder builder(*device_);
    Tensor output;
    {
        auto recording = builder.record_scope();
        CmdListBindScope binds(
            "scale_mask_softmax",
            {{.name = "input", .invoke_name = "input_tensor", .value = input_a},
             {.name = "scale", .invoke_name = "scale", .value = real(0.5)}});
        output = softmax(input_a, 0.5f);
        binds.finish();
    }
    CommandList list_cq0 = builder.build(QueueId{0});
    CommandList list_cq1 = builder.build(QueueId{1});
    builder.deallocate();

    list_cq0.update_args({{"input", input_b}, {"scale", real(0.25)}}, /*blocking=*/true);
    list_cq0.enqueue(/*blocking=*/true);
    EXPECT_EQ(output.to_vector<float>(), expected_b);

    list_cq1.enqueue(/*blocking=*/true);
    EXPECT_EQ(output.to_vector<float>(), expected_a);

    EXPECT_THAT(
        [&] { list_cq0.update_args({{"nope", real(1.0)}}); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Unknown TTNN command-list parameter 'nope'")));
    EXPECT_THAT(
        [&] { list_cq0.update_args({{"scale", input_b}}); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("'scale' is a scalar")));
    list_cq0.deallocate();
    list_cq1.deallocate();
}

// Writes one op argument, "value", to two L1 words with different encodings. With `packed`, writer_a writes it
// uint16-pair encoded to kAddressPacked instead, so one Metal param name has a different encoder per program.
struct FanOutDeviceOperation {
    static constexpr uint32_t kAddressA = 100 * 1024;
    static constexpr uint32_t kAddressB = kAddressA + 64;
    static constexpr uint32_t kAddressPacked = kAddressA + 128;

    struct operation_attributes_t {
        uint32_t value;
        bool packed = false;
    };
    struct tensor_args_t {
        const Tensor& input_tensor;
    };
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    struct Factory {
        static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
            const operation_attributes_t& attributes, const tensor_args_t& /*tensor_args*/, Tensor& /*output*/) {
            m2::ProgramSpec spec{
                .name = "cmd_list_fan_out",
                .kernels =
                    {make_kernel("writer_a", tt::tt_metal::DataMovementProcessor::RISCV_0),
                     make_kernel("writer_b", tt::tt_metal::DataMovementProcessor::RISCV_1)},
                .work_units = {m2::WorkUnitSpec{
                    .name = "main",
                    .kernels = {m2::KernelSpecName{"writer_a"}, m2::KernelSpecName{"writer_b"}},
                    .target_nodes = m2::NodeCoord{0, 0}}}};
            const CmdListScalar value{static_cast<int64_t>(attributes.value)};
            m2::ProgramRunArgs run_params;
            run_params.kernel_run_args = {
                m2::ProgramRunArgs::KernelRunArgs{
                    .kernel = m2::KernelSpecName{"writer_a"},
                    .runtime_arg_values = m2::MakeRuntimeArgsForSingleNode(
                        {0, 0}, {{"address", attributes.packed ? kAddressPacked : kAddressA}}),
                    .common_runtime_arg_values = {{"value", writer_a_encoder(attributes)(value)}}},
                m2::ProgramRunArgs::KernelRunArgs{
                    .kernel = m2::KernelSpecName{"writer_b"},
                    .runtime_arg_values = m2::MakeRuntimeArgsForSingleNode({0, 0}, {{"address", kAddressB}}),
                    .common_runtime_arg_values = {{"value", encode_uint16_pair(value)}}}};
            return {.spec = std::move(spec), .run_params = std::move(run_params)};
        }

        static CmdListEncodeFn writer_a_encoder(const operation_attributes_t& attributes) {
            return attributes.packed ? encode_uint16_pair : encode_uint32;
        }

        static CmdListBindings map_cmd_list_parameters(
            const operation_attributes_t& attributes,
            const tensor_args_t& /*tensor_args*/,
            const m2::ProgramSpec& spec,
            const m2::ProgramRunArgs& run_params) {
            CmdListBinder binder{spec, run_params};
            binder.param("value", writer_a_encoder(attributes)).common_runtime(m2::KernelSpecName{"writer_a"}, "value");
            binder.param("value", encode_uint16_pair, "value_u16_pair")
                .common_runtime(m2::KernelSpecName{"writer_b"}, "value");
            return binder.finish();
        }
    };
    using program_factory_t = std::variant<Factory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&) {
        return Factory{};
    }
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&) {}
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t& tensor_args) {
        return tensor_args.input_tensor.tensor_spec();
    }
    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t&, const tensor_args_t& tensor_args) {
        return tensor_args.input_tensor;
    }
};

TEST_F(CommandListDeviceParamsTest, OneKeywordFansOutToTwoEncodings) {
    const Tensor input = make_tensor(ttnn::Shape{1, 1, 32, 32}, 0.0f);
    auto* device = tt::tt_metal::internal::get_device(*device_, MeshCoordinate(0, 0));
    const auto read_word = [&](uint32_t address) {
        std::vector<uint32_t> result;
        tt::tt_metal::detail::ReadFromDeviceL1(device, CoreCoord{0, 0}, address, sizeof(uint32_t), result);
        return result.at(0);
    };

    CommandListBuilder builder(*device_);
    {
        auto recording = builder.record_scope();
        CmdListBindScope binds("fan_out", {{.name = "v", .invoke_name = "value", .value = integer(3)}});
        (void)ttnn::device_operation::launch<FanOutDeviceOperation>({.value = 3}, {.input_tensor = input});
        binds.finish();
    }
    CommandList list = builder.build(QueueId{0});
    builder.deallocate();

    list.update_args({{"v", integer(0x1234)}}, /*blocking=*/true);
    list.enqueue(/*blocking=*/true);
    EXPECT_EQ(read_word(FanOutDeviceOperation::kAddressA), 0x1234u);
    EXPECT_EQ(read_word(FanOutDeviceOperation::kAddressB), 0x12341234u);
    list.deallocate();
}

TEST_F(CommandListDeviceParamsTest, EachProgramKeepsItsOwnEncoder) {
    const Tensor input = make_tensor(ttnn::Shape{1, 1, 32, 32}, 0.0f);
    auto* device = tt::tt_metal::internal::get_device(*device_, MeshCoordinate(0, 0));
    const auto read_word = [&](uint32_t address) {
        std::vector<uint32_t> result;
        tt::tt_metal::detail::ReadFromDeviceL1(device, CoreCoord{0, 0}, address, sizeof(uint32_t), result);
        return result.at(0);
    };

    CommandListBuilder builder(*device_);
    {
        auto recording = builder.record_scope();
        CmdListBindScope binds("fan_out", {{.name = "v", .invoke_name = "value", .value = integer(3)}});
        (void)ttnn::device_operation::launch<FanOutDeviceOperation>({.value = 3}, {.input_tensor = input});
        (void)ttnn::device_operation::launch<FanOutDeviceOperation>(
            {.value = 3, .packed = true}, {.input_tensor = input});
        binds.finish();
    }
    CommandList list = builder.build(QueueId{0});
    builder.deallocate();

    list.update_args({{"v", integer(0x1234)}}, /*blocking=*/true);
    list.enqueue(/*blocking=*/true);
    EXPECT_EQ(read_word(FanOutDeviceOperation::kAddressA), 0x1234u);
    EXPECT_EQ(read_word(FanOutDeviceOperation::kAddressPacked), 0x12341234u);
    list.deallocate();
}

}  // namespace ttnn::command_list::test
