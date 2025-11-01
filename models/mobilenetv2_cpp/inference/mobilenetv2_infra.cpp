// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mobilenetv2_infra.h"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "helper_funcs.h"

MobileNetv2TestInfra::MobileNetv2TestInfra(std::shared_ptr<ttnn::MeshDevice> device, int batch_size) :
    device_(device), batch_size_(batch_size) {
    torch::manual_seed(0);

    // Load Torch model
    torch_model_ = loadTorchModel();

    // Load TTNN model
    ttnn_mobilenetv2_model_ = loadTtnnModel(device_, torch_model_, batch_size_);

    // Input tensor setup
    torch::Tensor torch_input_tensor = torch::randn({batch_size_, 224, 224, 3}, torch::kFloat32);
    input_tensor_ = from_torch(torch_input_tensor, ttnn::DataType::BFLOAT16);
    torch_input_tensor_ = torch_input_tensor.permute({0, 3, 1, 2});
    torch_output_tensor_ = torch_model_.forward({torch_input_tensor_}).toTensor();
}

MobileNetv2TestInfra::OneConfResult MobileNetv2TestInfra::setupL1ShardedInput(torch::Tensor torch_input_tensor) {
    if (!isWormholeB0()) {
        throw std::runtime_error("Unsupported device");
    }

    torch_input_tensor = torch_input_tensor.defined() ? torch_input_tensor : torch_input_tensor_;

    auto n = torch_input_tensor.size(0);
    auto c = torch_input_tensor.size(1);
    auto h = torch_input_tensor.size(2);
    auto w = torch_input_tensor.size(3);

    // Sharded memory config setup
    uint32_t num_cores = 64;  // using core grid (8, 8)
    uint32_t shard_h = divup(n * w * h, num_cores);
    ttnn::CoreRangeSet shard_grid({ttnn::CoreRange(ttnn::CoreCoord(0, 0), ttnn::CoreCoord(7, 7))});
    tt::tt_metal::ShardSpec shard_spec(shard_grid, {shard_h, 16}, ttnn::ShardOrientation::ROW_MAJOR);
    ttnn::MemoryConfig input_mem_config(ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ttnn::BufferType::L1, shard_spec);

    torch_input_tensor = torch_input_tensor.permute({0, 2, 3, 1});
    torch_input_tensor = torch_input_tensor.reshape({1, 1, h * w * n, c});
    ttnn::Tensor tt_inputs_host = from_torch(torch_input_tensor, ttnn::DataType::BFLOAT16, ttnn::Layout::ROW_MAJOR);
    tt_inputs_host =
        ttnn::pad(tt_inputs_host, tt::tt_metal::Array4D{1, 1, n * h * w, 16}, tt::tt_metal::Array4D{0, 0, 0, 0}, 0);

    return {tt_inputs_host, input_mem_config};
}

MobileNetv2TestInfra::TwoConfResult MobileNetv2TestInfra::setupDramShardedInput(
    std::shared_ptr<ttnn::MeshDevice> device, torch::Tensor torch_input_tensor) {
    auto [tt_inputs_host, input_mem_config] = setupL1ShardedInput(torch_input_tensor);

    auto dram_grid_size = device->dram_grid_size();
    ttnn::CoreRangeSet dram_shard_grid(
        {ttnn::CoreRange(ttnn::CoreCoord(0, 0), ttnn::CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1))});
    ttnn::Shape input_shape = tt_inputs_host.logical_shape();
    tt::tt_metal::ShardSpec dram_shard_spec(
        dram_shard_grid,
        {divup(tt_inputs_host.logical_volume() / input_shape[-1], dram_grid_size.x * dram_grid_size.y), 16},
        ttnn::ShardOrientation::ROW_MAJOR);
    ttnn::MemoryConfig sharded_mem_config_DRAM(
        ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ttnn::BufferType::DRAM, dram_shard_spec);

    return {tt_inputs_host, sharded_mem_config_DRAM, input_mem_config};
}

void MobileNetv2TestInfra::validate(std::optional<ttnn::Tensor> output_tensor) {
    auto tensor =
        output_tensor.has_value() ? output_tensor.value() : ttnn::from_device(output_tensor_, /*blocking=*/true);
    auto torch_output_tensor = to_torch(tensor);
    std::string pcc_message = assert_with_pcc(torch_output_tensor_, torch_output_tensor, 0.94);

    std::cout << "mobilenetv2 batch_size=" << batch_size_ << ", " << pcc_message << std::endl;
}
