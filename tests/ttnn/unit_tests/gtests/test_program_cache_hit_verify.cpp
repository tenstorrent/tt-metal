// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <exception>
#include <variant>

#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/cleanup.hpp>

#include "ttnn/config.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/graph/graph_processor.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::device_operation::test {
namespace {

struct TestAttributes {
    uint32_t baked_value = 0;
    uint32_t cb_size = 2048;
    bool second_factory = false;
};

// Bakes `baked_value` into a compile-time argument and sizes a circular buffer by `cb_size`, but its program hash
// covers neither, nor which factory `second_factory` selects. With `PatchesCbSize`, hits resize the circular buffer
// in place, which makes leaving `cb_size` out of the hash correct.
template <bool PatchesCbSize>
struct MisHashedOp {
    using operation_attributes_t = TestAttributes;
    struct tensor_args_t {
        Tensor input;
    };
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    template <int Index>
    struct Factory {
        struct shared_variables_t {
            tt::tt_metal::CBHandle cb{};
        };
        using cached_program_t = CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t& attrs, const tensor_args_t& /*tensor_args*/, tensor_return_value_t&) {
            tt::tt_metal::Program program;
            tt::tt_metal::CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/blank.cpp",
                CoreCoord{0, 0},
                tt::tt_metal::DataMovementConfig{.compile_args = {attrs.baked_value}});
            const auto cb = tt::tt_metal::CreateCircularBuffer(
                program,
                CoreCoord{0, 0},
                tt::tt_metal::CircularBufferConfig(attrs.cb_size, {{0, tt::DataFormat::Float16_b}})
                    .set_page_size(0, 2048));
            return cached_program_t(std::move(program), shared_variables_t{.cb = cb});
        }

        static void override_runtime_arguments(
            cached_program_t& cached_program,
            const operation_attributes_t& attrs,
            const tensor_args_t&,
            tensor_return_value_t&) {
            if constexpr (PatchesCbSize) {
                tt::tt_metal::UpdateCircularBufferTotalSize(
                    cached_program.program, cached_program.shared_variables.cb, attrs.cb_size);
            }
        }
    };
    using program_factory_t = std::variant<Factory<0>, Factory<1>>;

    static program_factory_t select_program_factory(const operation_attributes_t& attrs, const tensor_args_t&) {
        if (attrs.second_factory) {
            return Factory<1>{};
        }
        return Factory<0>{};
    }

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&) {}

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t& tensor_args) {
        return tensor_args.input.tensor_spec();
    }

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
        return create_device_tensor(compute_output_specs(attrs, tensor_args), tensor_args.input.device());
    }

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t& tensor_args) {
        return ttsl::hash::hash_objects_with_default_seed(tensor_args.input.tensor_spec());
    }
};

using UnpatchedOp = MisHashedOp<false>;
using PatchingOp = MisHashedOp<true>;

class ProgramCacheHitVerifyTest : public TTNNFixtureWithDevice {
protected:
    Tensor make_input() const {
        return create_device_tensor(
            tt::tt_metal::TensorSpec(
                ttnn::Shape{1, 1, 32, 32},
                tt::tt_metal::TensorLayout(DataType::BFLOAT16, tt::tt_metal::PageConfig(Layout::TILE), MemoryConfig{})),
            device_);
    }

    static auto verify_hits(bool enabled) {
        const bool previous = ttnn::CONFIG.get<"verify_program_cache_hits">();
        ttnn::CONFIG.set<"verify_program_cache_hits">(enabled);
        return ttsl::make_cleanup([previous]() { ttnn::CONFIG.set<"verify_program_cache_hits">(previous); });
    }
};

TEST_F(ProgramCacheHitVerifyTest, MissingHashFieldIsSilentWhenOff) {
    const auto restore = verify_hits(false);
    const auto input = make_input();
    launch<UnpatchedOp>({.baked_value = 1}, {.input = input});
    EXPECT_NO_THROW(launch<UnpatchedOp>({.baked_value = 2}, {.input = input}));
    EXPECT_EQ(device_->get_program_cache().num_entries(), 1);
}

TEST_F(ProgramCacheHitVerifyTest, MissingHashFieldThrowsWhenOn) {
    const auto restore = verify_hits(true);
    const auto input = make_input();
    launch<UnpatchedOp>({.baked_value = 1}, {.input = input});
    EXPECT_NO_THROW(launch<UnpatchedOp>({.baked_value = 1}, {.input = input}));
    EXPECT_THROW(launch<UnpatchedOp>({.baked_value = 2}, {.input = input}), std::exception);
}

TEST_F(ProgramCacheHitVerifyTest, MissingFactorySelectionThrowsWhenOn) {
    const auto restore = verify_hits(true);
    const auto input = make_input();
    launch<UnpatchedOp>({.second_factory = false}, {.input = input});
    EXPECT_THROW(launch<UnpatchedOp>({.second_factory = true}, {.input = input}), std::exception);
}

TEST_F(ProgramCacheHitVerifyTest, ComparesAfterTheHitsUpdate) {
    const auto restore = verify_hits(true);
    const auto input = make_input();
    launch<UnpatchedOp>({.cb_size = 2048}, {.input = input});
    EXPECT_THROW(launch<UnpatchedOp>({.cb_size = 4096}, {.input = input}), std::exception);

    launch<PatchingOp>({.cb_size = 2048}, {.input = input});
    EXPECT_NO_THROW(launch<PatchingOp>({.cb_size = 4096}, {.input = input}));
}

TEST_F(ProgramCacheHitVerifyTest, SkippedDuringGraphCapture) {
    const auto restore = verify_hits(true);
    const auto input = make_input();
    launch<UnpatchedOp>({.baked_value = 1}, {.input = input});
    ttnn::graph::GraphProcessor::begin_graph_capture(tt::tt_metal::IGraphProcessor::RunMode::NORMAL);
    const auto end_capture = ttsl::make_cleanup([]() { ttnn::graph::GraphProcessor::end_graph_capture(); });
    EXPECT_NO_THROW(launch<UnpatchedOp>({.baked_value = 2}, {.input = input}));
}

TEST_F(ProgramCacheHitVerifyTest, HonestHitsPass) {
    const auto restore = verify_hits(true);
    const auto input = make_input();
    ttnn::add(input, input);
    EXPECT_NO_THROW(ttnn::add(input, input));
    EXPECT_NO_THROW(ttnn::multiply(input, input));
    EXPECT_NO_THROW(ttnn::multiply(input, input));
}

}  // namespace
}  // namespace ttnn::device_operation::test
