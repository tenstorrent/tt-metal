// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mobilenetv2_e2e_performance.h"
#include "helper_funcs.h"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/sharded/reshard/reshard.hpp"
#include "ttnn/distributed/api.hpp"

MobileNetV2Trace2CQ::MobileNetV2Trace2CQ() :
    device_ptr_(nullptr),
    op_event(0, nullptr, 0, ttnn::MeshCoordinateRange({})),
    write_event(0, nullptr, 0, ttnn::MeshCoordinateRange({})),
    test_infra(nullptr) {}

void MobileNetV2Trace2CQ::initialize_mobilenetv2_trace_2cqs_inference(
    std::shared_ptr<ttnn::MeshDevice> device, int device_batch_size) {
    test_infra = std::make_shared<MobileNetv2TestInfra>(device, device_batch_size);

    this->device_ptr_ = device;
    auto [tt_inputs_host, sharded_mem_config_DRAM, input_mem_config] = (*test_infra).setupDramShardedInput(device);
    m_tt_inputs_host = std::move(tt_inputs_host);
    m_input_mem_config = input_mem_config;

    m_tt_image_res = m_tt_inputs_host.to_device(device_ptr_.get(), sharded_mem_config_DRAM);
    op_event = ttnn::events::record_mesh_event(device_ptr_.get(), ttnn::QueueId(0));
    ttnn::events::wait_for_mesh_event(ttnn::QueueId(1), op_event);

    // First run configures convs JIT
    tt::tt_metal::write_tensor(m_tt_inputs_host, m_tt_image_res, false, ttnn::QueueId(1));
    write_event = ttnn::events::record_mesh_event(device_ptr_.get(), ttnn::QueueId(1));
    ttnn::events::wait_for_mesh_event(ttnn::QueueId(0), write_event);

    (*test_infra).setInputTensor(ttnn::to_memory_config(m_tt_image_res, m_input_mem_config));
    op_event = ttnn::events::record_mesh_event(device_ptr_.get(), ttnn::QueueId(0));
    (*test_infra).run();
    (*test_infra).validate();
    (*test_infra).deallocOutput();

    // Optimized run
    ttnn::events::wait_for_mesh_event(ttnn::QueueId(1), op_event);
    tt::tt_metal::write_tensor(m_tt_inputs_host, m_tt_image_res, false, ttnn::QueueId(1));
    write_event = ttnn::events::record_mesh_event(device_ptr_.get(), ttnn::QueueId(1));
    ttnn::events::wait_for_mesh_event(ttnn::QueueId(0), write_event);

    (*test_infra).setInputTensor(ttnn::to_memory_config(m_tt_image_res, m_input_mem_config));
    op_event = ttnn::events::record_mesh_event(device_ptr_.get(), ttnn::QueueId(0));
    (*test_infra).run();
    (*test_infra).validate();

    // Capture
    ttnn::events::wait_for_mesh_event(ttnn::QueueId(1), op_event);
    tt::tt_metal::write_tensor(m_tt_inputs_host, m_tt_image_res, false, ttnn::QueueId(1));
    write_event = ttnn::events::record_mesh_event(device_ptr_.get(), ttnn::QueueId(1));
    ttnn::events::wait_for_mesh_event(ttnn::QueueId(0), write_event);
    (*test_infra).setInputTensor(ttnn::to_memory_config(m_tt_image_res, m_input_mem_config));
    op_event = ttnn::events::record_mesh_event(device_ptr_.get(), ttnn::QueueId(0));
    auto trace_input_addr = get_ttbuffer_address((*test_infra).getInputTensor());
    ttnn::TensorSpec spec = (*test_infra).getInputTensor().tensor_spec();
    // Important dealloc here: Deallocate the previous output tensor here so that we will allocate our input tensor at
    // the right address afterwards
    (*test_infra).deallocOutput();
    tid = ttnn::operations::trace::begin_trace_capture(device_ptr_.get(), /*cq_id=*/ttnn::QueueId(0));
    (*test_infra).run();
    m_input_tensor = tt::tt_metal::allocate_tensor_on_device(spec, device_ptr_.get());
    auto allocate_addr = get_ttbuffer_address(m_input_tensor);
    TT_FATAL(trace_input_addr == allocate_addr, "trace input addr allocate_tensor_on_device error!");
    ttnn::operations::trace::end_trace_capture(device_ptr_.get(), tid.value(), /*cq_id=*/ttnn::QueueId(0));
}

void MobileNetV2Trace2CQ::execute_mobilenetv2_trace_2cqs_inference(const ttnn::Tensor& tt_inputs_host) {
    assert((test_infra && tid.has_value()) && "call initialize_mobilenetv2_trace_2cqs_inference first!");

    ttnn::events::wait_for_mesh_event(ttnn::QueueId(1), op_event);
    tt::tt_metal::write_tensor(tt_inputs_host, m_tt_image_res, false, ttnn::QueueId(1));
    write_event = ttnn::events::record_mesh_event(device_ptr_.get(), ttnn::QueueId(1));
    ttnn::events::wait_for_mesh_event(ttnn::QueueId(0), write_event);

    m_input_tensor = ttnn::reshard(m_tt_image_res, m_input_mem_config, m_input_tensor);
    op_event = ttnn::events::record_mesh_event(device_ptr_.get(), ttnn::QueueId(0));
    ttnn::operations::trace::execute_trace(
        device_ptr_.get(), tid.value(), /*cq_id=*/ttnn::QueueId(0), /*blocking=*/false);
    outputs.push_back(ttnn::from_device((*test_infra).getOutputTensor(), /*blocking=*/false));
}

ttnn::Tensor MobileNetV2Trace2CQ::get_output() {
    tt::tt_metal::distributed::Synchronize(device_ptr_.get(), 0);
    if (outputs.empty()) {
        return ttnn::Tensor();
    } else {
        return outputs.back();
    }
}

void MobileNetV2Trace2CQ::release_mobilenetv2_trace_2cqs_inference() {
    if (tid.has_value()) {
        ttnn::operations::trace::release_trace(device_ptr_.get(), tid.value());
    }
}
