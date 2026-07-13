// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <vector>

#include <gtest/gtest.h>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/context/metal_env.hpp>
#include <tt-metalium/experimental/program_descriptor_patching.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program.hpp>

#include "ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::binary_ng::test {

namespace {

using tt::tt_metal::Buffer;
using tt::tt_metal::CoreCoord;
using tt::tt_metal::DataType;
using tt::tt_metal::HalProgrammableCoreType;
using tt::tt_metal::Layout;
using tt::tt_metal::MetalEnv;
using tt::tt_metal::MetalEnvDescriptor;
using tt::tt_metal::MemoryConfig;
using tt::tt_metal::PageConfig;
using tt::tt_metal::Program;
using tt::tt_metal::TensorLayout;
using tt::tt_metal::TensorMemoryLayout;
using tt::tt_metal::BufferType;

bool has_binding(
    const tt::tt_metal::ResolvedBindings& bindings, uint32_t kernel_idx, uint32_t arg_idx, uint32_t tensor_buffer_idx) {
    return std::any_of(bindings.rt_args.begin(), bindings.rt_args.end(), [&](const auto& binding) {
        return binding.kernel_idx == kernel_idx && binding.arg_idx == arg_idx &&
               binding.tensor_buffer_idx == tensor_buffer_idx;
    });
}

}  // namespace

class BinaryNgDescriptorBindingsFixture : public ::testing::Test {
protected:
    std::unique_ptr<MetalEnv> mock_env_;
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> device_holder_;
    tt::tt_metal::distributed::MeshDevice* device_ = nullptr;

    void SetUp() override {
        std::filesystem::path repo_root;
        for (auto current = std::filesystem::path(__FILE__).parent_path(); !current.empty(); current = current.parent_path()) {
            const auto mock_cluster_desc =
                current / "tt_metal" / "third_party" / "umd" / "tests" / "cluster_descriptor_examples" /
                "wormhole_N150.yaml";
            if (std::filesystem::exists(mock_cluster_desc)) {
                repo_root = current;
                break;
            }
        }
        ASSERT_FALSE(repo_root.empty());
        ASSERT_EQ(::setenv("TT_METAL_RUNTIME_ROOT", repo_root.c_str(), 1), 0);

        mock_env_ = std::make_unique<MetalEnv>(MetalEnvDescriptor(std::string("wormhole_N150.yaml")));
        auto mesh_shape = mock_env_->get_system_mesh().shape();
        device_holder_ = mock_env_->create_mesh_device(tt::tt_metal::distributed::MeshDeviceConfig(mesh_shape));
        device_ = device_holder_.get();
        ASSERT_GT(device_->num_devices(), 0u);
        device_->disable_and_clear_program_cache();
    }

    void TearDown() override {
        device_holder_.reset();
        mock_env_.reset();
        ::unsetenv("TT_METAL_RUNTIME_ROOT");
    }
};

TEST_F(BinaryNgDescriptorBindingsFixture, InplaceAddDescriptorRegistersPatchableBufferBindings) {
    auto* device = device_;

    const auto tensor_spec = ttnn::TensorSpec(
        ttnn::Shape(tt::tt_metal::Array2D{32, 64}),
        TensorLayout(
            DataType::BFLOAT16,
            PageConfig(Layout::TILE),
            MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM}));

    const auto input_tensor_a = tt::tt_metal::create_device_tensor(tensor_spec, device);
    const auto input_tensor_b = tt::tt_metal::create_device_tensor(tensor_spec, device);
    auto output_tensor = input_tensor_a;

    const auto worker_grid =
        device->worker_cores(HalProgrammableCoreType::TENSIX, device->get_sub_device_ids().front());

    const auto operation_attributes = BinaryNgDeviceOperation::operation_attributes_t{
        BinaryOpType::ADD,
        {},
        {},
        {},
        std::nullopt,
        input_tensor_a.memory_config(),
        input_tensor_a.dtype(),
        std::nullopt,
        worker_grid,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        SubtileBroadcastType::NONE,
        false,
        false,
        false,
        0.0f,
        0.0f,
        false,
        input_tensor_a.layout(),
        input_tensor_b.layout(),
        output_tensor.layout(),
    };

    const auto tensor_args =
        BinaryNgDeviceOperation::tensor_args_t{input_tensor_a, input_tensor_b, std::optional<Tensor>{output_tensor}};

    const auto descriptor =
        BinaryNgDeviceOperation::ProgramFactory::create_descriptor(operation_attributes, tensor_args, output_tensor);
    Program program{descriptor};

    std::vector<Buffer*> tensor_buffers = {input_tensor_a.buffer(), input_tensor_b.buffer(), output_tensor.buffer()};
    const auto bindings =
        tt::tt_metal::resolve_bindings(program, descriptor, tensor_buffers, /*num_input_buffers=*/2);

    EXPECT_FALSE(bindings.rt_args.empty());

    // reader kernel arg[0] = input A base address
    EXPECT_TRUE(has_binding(bindings, /*kernel_idx=*/0, /*arg_idx=*/0, /*tensor_buffer_idx=*/0));
    // reader kernel arg[15] = input B base address in the interleaved tensor-tensor path
    EXPECT_TRUE(has_binding(bindings, /*kernel_idx=*/0, /*arg_idx=*/15, /*tensor_buffer_idx=*/1));
    // In-place output alias keeps the first input-slot mapping. That preserves a stable
    // cache-hit patch target while still rejecting ambiguous duplicate inputs.
    EXPECT_TRUE(has_binding(bindings, /*kernel_idx=*/1, /*arg_idx=*/0, /*tensor_buffer_idx=*/0));
}

}  // namespace ttnn::operations::binary_ng::test
