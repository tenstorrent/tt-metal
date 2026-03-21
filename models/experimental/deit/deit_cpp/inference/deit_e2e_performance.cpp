// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "deit_e2e_performance.h"
#include <torch/torch.h>
#include <torch/script.h>
#include <tt-metalium/host_api.hpp>
#include <ttnn/distributed/api.hpp>
#include "../tt_cpp/deit_config.h"
#include "../tt_cpp/deit_for_image_classification.h"
#include "../helper_funcs.h"

DeiTTrace2CQ::DeiTTrace2CQ() :
    device_(nullptr),
    op_event_(0, nullptr, 0, ttnn::MeshCoordinateRange({})),
    write_event_(0, nullptr, 0, ttnn::MeshCoordinateRange({})) {}

DeiTTrace2CQ::~DeiTTrace2CQ() = default;

void DeiTTrace2CQ::initialize_deit_trace_2cqs_inference(
    const std::shared_ptr<ttnn::MeshDevice>& device, int batch_size, const std::string& model_path) {
    device_ = device;
    auto model = torch::jit::load(model_path);
    model.eval();

    std::unordered_map<std::string, torch::Tensor> state_dict;
    for (const auto& pair : model.named_parameters()) {
        state_dict[pair.name] = pair.value;
    }

    DeiTConfig config;
    tt_model_ = std::make_unique<TtDeiTForImageClassification>(config, state_dict, "model.", device_);

    auto dummy = torch::randn({batch_size, 224, 224, 16});
    tt_input_host_ = helper_funcs::from_torch(dummy, ttnn::DataType::BFLOAT16, ttnn::Layout::ROW_MAJOR);
    tt_input_device_ = tt_input_host_.to_device(device_.get(), ttnn::L1_MEMORY_CONFIG);

    tt_model_->forward(tt_input_device_, nullptr, false, false, true);

    trace_id_ = ttnn::operations::trace::begin_trace_capture(device_.get(), ttnn::QueueId(0));
    auto [logits, _, __] = tt_model_->forward(tt_input_device_, nullptr, false, false, true);
    tt_output_ = logits;
    ttnn::operations::trace::end_trace_capture(device_.get(), *trace_id_, ttnn::QueueId(0));

    op_event_ = ttnn::events::record_mesh_event(device_.get(), ttnn::QueueId(0));
    write_event_ = ttnn::events::record_mesh_event(device_.get(), ttnn::QueueId(1));
}

void DeiTTrace2CQ::execute_deit_trace_2cqs_inference(const ttnn::Tensor& input_host) {
    ttnn::events::wait_for_mesh_event(ttnn::QueueId(1), op_event_);
    tt::tt_metal::copy_to_device(input_host, tt_input_device_, ttnn::QueueId(1));
    write_event_ = ttnn::events::record_mesh_event(device_.get(), ttnn::QueueId(1));

    ttnn::events::wait_for_mesh_event(ttnn::QueueId(0), write_event_);
    op_event_ = ttnn::events::record_mesh_event(device_.get(), ttnn::QueueId(0));
    ttnn::operations::trace::execute_trace(device_.get(), *trace_id_, ttnn::QueueId(0), false);
    outputs_.push_back(ttnn::from_device(tt_output_, false));
}

ttnn::Tensor DeiTTrace2CQ::get_output() {
    tt::tt_metal::distributed::Synchronize(device_.get(), 0);
    if (outputs_.empty()) {
        return ttnn::Tensor();
    }
    return outputs_.back();
}

void DeiTTrace2CQ::release_deit_trace_2cqs_inference() {
    if (trace_id_.has_value()) {
        ttnn::operations::trace::release_trace(device_.get(), *trace_id_);
    }
}
