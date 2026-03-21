// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef MOBILENETV2_CPP_HELPER_FUNCS
#define MOBILENETV2_CPP_HELPER_FUNCS

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "ttnn/device.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

class TtMobileNetV2;

struct MobileNetV2HostInput {
    std::vector<float> nhwc_flattened_data;
    uint32_t batch;
    uint32_t channels;
    uint32_t height;
    uint32_t width;
};

MobileNetV2HostInput create_mobilenetv2_host_input(
    int batch = 1, int input_channels = 3, int input_height = 224, int input_width = 224);

ttnn::Tensor host_input_to_ttnn(const MobileNetV2HostInput& input);

std::unordered_map<std::string, ttnn::Tensor> create_mobilenetv2_model_parameters(
    const std::string& weights_dir, const std::shared_ptr<ttnn::MeshDevice>& device);

uint32_t get_ttbuffer_address(const ttnn::Tensor& tensor);
uint32_t divup(uint32_t x, uint32_t y);
bool isWormholeB0();
std::shared_ptr<TtMobileNetV2> loadTtnnModel(
    const std::shared_ptr<ttnn::MeshDevice>& device, const std::string& weights_dir, int batch_size);

#endif  // MOBILENETV2_CPP_HELPER_FUNCS
