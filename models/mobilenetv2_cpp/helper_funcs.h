// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef MOBILENETV2_CPP_HELPER_FUNCS
#define MOBILENETV2_CPP_HELPER_FUNCS

#include <optional>
#include <tuple>
#include <vector>
#include <string>
#include <unordered_map>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device.hpp"
#include "ttnn/types.hpp"
#include "torch/torch.h"
#include "torch/script.h"

class TtMobileNetV2;

ttnn::Tensor from_torch(
    const at::Tensor& tensor,
    std::optional<ttnn::DataType> dtype = std::nullopt,
    std::optional<ttnn::Layout> layout = ttnn::Layout::ROW_MAJOR);

at::Tensor to_torch(const ttnn::Tensor& tensor, const bool padded_output = false);

std::tuple<at::Tensor, ttnn::Tensor> create_mobilenetv2_input_tensors(
    int batch = 1, int input_channels = 3, int input_height = 224, int input_width = 224);

std::unordered_map<std::string, ttnn::Tensor> create_mobilenetv2_model_parameters(
    const torch::jit::Module& model, const std::shared_ptr<ttnn::MeshDevice>& device);

uint32_t get_ttbuffer_address(const ttnn::Tensor& tensor);

std::string assert_with_pcc(const at::Tensor& golden, const at::Tensor& calculated, const double pcc_threshold = 0.99);

uint32_t divup(uint32_t x, uint32_t y);

bool isWormholeB0();

torch::jit::Module loadTorchModel();

std::shared_ptr<TtMobileNetV2> loadTtnnModel(
    const std::shared_ptr<ttnn::MeshDevice>& device, const torch::jit::Module& torch_model, int batch_size);

#endif  // MOBILENETV2_CPP_HELPER_FUNCS
