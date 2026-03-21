// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mobilenetv2_infra.h"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "helper_funcs.h"

MobileNetv2TestInfra::MobileNetv2TestInfra(
    std::shared_ptr<ttnn::MeshDevice> device, int batch_size, const std::string& weights_dir) :
    device_(std::move(device)), batch_size_(batch_size) {
    ttnn_mobilenetv2_model_ = loadTtnnModel(device_, weights_dir, batch_size_);

    auto host_input = create_mobilenetv2_host_input(batch_size_, 3, 224, 224);
    input_tensor_ = host_input_to_ttnn(host_input);
}

MobileNetv2TestInfra::OneConfResult MobileNetv2TestInfra::setupL1ShardedInput(
    std::optional<MobileNetV2HostInput> host_input) {
    if (!isWormholeB0()) {
        throw std::runtime_error("Unsupported device");
    }

    const auto input = host_input.value_or(create_mobilenetv2_host_input(batch_size_, 3, 224, 224));
    const auto n = input.batch;
    const auto h = input.height;
    const auto w = input.width;

    uint32_t num_cores = 64;
    uint32_t shard_h = divup(n * w * h, num_cores);
    ttnn::CoreRangeSet shard_grid({ttnn::CoreRange(ttnn::CoreCoord(0, 0), ttnn::CoreCoord(7, 7))});
    tt::tt_metal::ShardSpec shard_spec(shard_grid, {shard_h, 16}, ttnn::ShardOrientation::ROW_MAJOR);
    ttnn::MemoryConfig input_mem_config(ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ttnn::BufferType::L1, shard_spec);

    ttnn::Tensor tt_inputs_host = host_input_to_ttnn(input);
    tt_inputs_host = ttnn::pad(
        tt_inputs_host,
        tt::tt_metal::Array4D{1, 1, static_cast<uint32_t>(n * h * w), 16},
        tt::tt_metal::Array4D{0, 0, 0, 0},
        0);

    return {tt_inputs_host, input_mem_config};
}

MobileNetv2TestInfra::TwoConfResult MobileNetv2TestInfra::setupDramShardedInput(
    const std::shared_ptr<ttnn::MeshDevice>& device, std::optional<MobileNetV2HostInput> host_input) {
    auto [tt_inputs_host, input_mem_config] = setupL1ShardedInput(std::move(host_input));

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
    auto host_tensor =
        tensor.layout() == ttnn::Layout::ROW_MAJOR ? tensor : ttnn::to_layout(tensor, ttnn::Layout::ROW_MAJOR);
    auto values = host_tensor.to_vector<float>();
    TT_FATAL(!values.empty(), "Output tensor is empty");
    std::cout << "mobilenetv2 batch_size=" << batch_size_ << ", output_numel=" << values.size() << std::endl;
}
