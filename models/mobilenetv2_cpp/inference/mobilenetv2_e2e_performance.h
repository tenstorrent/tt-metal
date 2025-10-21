// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef MOBILENETV2_CPP_INFERENCE_MOBILENETV2_E2E_PERFOMANCE
#define MOBILENETV2_CPP_INFERENCE_MOBILENETV2_E2E_PERFOMANCE

#include <memory>
#include "ttnn/device.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/events.hpp"
#include "ttnn/operations/trace.hpp"
#include "mobilenetv2_infra.h"

class MobileNetV2Trace2CQ {
public:
    MobileNetV2Trace2CQ();
    void initialize_mobilenetv2_trace_2cqs_inference(std::shared_ptr<ttnn::MeshDevice> device, int device_batch_size);
    void execute_mobilenetv2_trace_2cqs_inference(const ttnn::Tensor& tt_inputs_host);
    void release_mobilenetv2_trace_2cqs_inference();
    ttnn::Tensor get_output();

private:
    std::shared_ptr<ttnn::MeshDevice> device_ptr_;
    ttnn::Tensor m_tt_inputs_host;
    ttnn::Tensor m_tt_image_res;
    ttnn::Tensor m_input_tensor;
    ttnn::MeshEvent op_event;
    ttnn::MeshEvent write_event;
    std::optional<ttnn::MeshTraceId> tid;
    ttnn::MemoryConfig m_input_mem_config;
    std::shared_ptr<MobileNetv2TestInfra> test_infra;
    std::vector<ttnn::Tensor> outputs;
};

#endif  // MOBILENETV2_CPP_INFERENCE_MOBILENETV2_E2E_PERFOMANCE
