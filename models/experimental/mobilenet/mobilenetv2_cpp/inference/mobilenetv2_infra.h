// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef MOBILENETV2_CPP_INFERENCE_MOBILENETV2_INFRA
#define MOBILENETV2_CPP_INFERENCE_MOBILENETV2_INFRA

#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include "helper_funcs.h"
#include "ttnn_mobilenetv2.h"
#include "ttnn/types.hpp"

class MobileNetv2TestInfra : public std::enable_shared_from_this<MobileNetv2TestInfra> {
public:
    using OneConfResult = std::pair<ttnn::Tensor, ttnn::MemoryConfig>;
    using TwoConfResult = std::tuple<ttnn::Tensor, ttnn::MemoryConfig, ttnn::MemoryConfig>;

    MobileNetv2TestInfra(std::shared_ptr<ttnn::MeshDevice> device, int batch_size, const std::string& weights_dir = "");

    void setInputTensor(const ttnn::Tensor& input_tensor) { input_tensor_ = input_tensor; }

    const ttnn::Tensor& getInputTensor() const { return input_tensor_; }

    const ttnn::Tensor& getOutputTensor() const { return output_tensor_; }

    void run() { output_tensor_ = (*ttnn_mobilenetv2_model_)(input_tensor_); }

    OneConfResult setupL1ShardedInput(std::optional<MobileNetV2HostInput> host_input = std::nullopt);

    TwoConfResult setupDramShardedInput(
        const std::shared_ptr<ttnn::MeshDevice>& device, std::optional<MobileNetV2HostInput> host_input = std::nullopt);

    void validate(std::optional<ttnn::Tensor> output_tensor = std::nullopt);

    void deallocOutput() { output_tensor_.deallocate(true); }

private:
    std::shared_ptr<ttnn::MeshDevice> device_;
    int batch_size_;
    std::shared_ptr<TtMobileNetV2> ttnn_mobilenetv2_model_;
    ttnn::Tensor input_tensor_;
    ttnn::Tensor output_tensor_;
};

#endif  // MOBILENETV2_CPP_INFERENCE_MOBILENETV2_INFRA
