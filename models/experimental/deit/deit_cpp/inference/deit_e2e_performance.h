// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <memory>
#include <string>
#include <optional>
#include <vector>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/device.hpp>
#include <ttnn/events.hpp>
#include <ttnn/operations/trace.hpp>

class TtDeiTForImageClassification;

class DeiTTrace2CQ {
public:
    DeiTTrace2CQ();
    ~DeiTTrace2CQ();

    void initialize_deit_trace_2cqs_inference(
        const std::shared_ptr<ttnn::MeshDevice>& device, int batch_size, const std::string& model_path);

    void execute_deit_trace_2cqs_inference(const ttnn::Tensor& input_host);
    ttnn::Tensor get_output();
    void release_deit_trace_2cqs_inference();

private:
    std::shared_ptr<ttnn::MeshDevice> device_;
    std::unique_ptr<TtDeiTForImageClassification> tt_model_;
    std::optional<ttnn::MeshTraceId> trace_id_;
    ttnn::Tensor tt_input_host_;
    ttnn::Tensor tt_input_device_;
    ttnn::Tensor tt_output_;
    std::vector<ttnn::Tensor> outputs_;
    ttnn::MeshEvent op_event_;
    ttnn::MeshEvent write_event_;
};
