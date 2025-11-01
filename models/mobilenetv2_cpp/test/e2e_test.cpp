// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <vector>
#include <chrono>
#include <fmt/format.h>
#include "ttnn/device.hpp"
#include "inference/mobilenetv2_e2e_performance.h"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "helper_funcs.h"
#include "util/profiler.hpp"
#include "torch/torch.h"

// Function to measure time in seconds
double measure_time(
    std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point end) {
    return std::chrono::duration<double>(end - start).count();
}

void test_run_mobilenetv2_trace_2cqs_inference(std::shared_ptr<ttnn::MeshDevice> device, int batch_size) {
    // Prepare input tensor
    torch::Tensor torch_input_tensor =
        torch::randn({batch_size, 3, 224, 224}, torch::TensorOptions().dtype(torch::kFloat32));

    int n = torch_input_tensor.size(0);
    int c = torch_input_tensor.size(1);
    int h = torch_input_tensor.size(2);
    int w = torch_input_tensor.size(3);

    // Permute dimensions: (n, h, w, c)
    torch_input_tensor = torch_input_tensor.permute({0, 2, 3, 1});

    // Reshape tensor: (1, 1, h * w * n, c)
    torch_input_tensor = torch_input_tensor.reshape({1, 1, h * w * n, c});

    // Convert to TTNN tensor
    ttnn::Tensor tt_inputs_host = from_torch(torch_input_tensor, ttnn::DataType::BFLOAT16, ttnn::Layout::ROW_MAJOR);

    // Pad TTNN tensor
    tt_inputs_host = tt_inputs_host.pad(tt::tt_metal::Shape{1, 1, n * h * w, 16}, tt::tt_metal::Shape{0, 0, 0, 0}, 0);

    MobileNetV2Trace2CQ mobilenetv2_trace_2cq;

    // Initialize MobileNetV2 inference
    mobilenetv2_trace_2cq.initialize_mobilenetv2_trace_2cqs_inference(device, batch_size);

    // Perform inference iterations
    int inference_iter_count = 10;
    Profiler pf;
    for (int iter = 0; iter < inference_iter_count; ++iter) {
        pf.start("inference time");
        mobilenetv2_trace_2cq.execute_mobilenetv2_trace_2cqs_inference(tt_inputs_host);
        pf.stop("inference time");
    }

    pf.start("sync output");
    mobilenetv2_trace_2cq.get_output();
    pf.stop("sync output");

    // Release resources
    mobilenetv2_trace_2cq.release_mobilenetv2_trace_2cqs_inference();

    // Calculate average inference time
    auto inference_time = pf.get("inference time");
    auto sync_output_time = pf.get("sync output");
    double inference_time_avg = (inference_time + sync_output_time) / inference_iter_count;

    double fps = batch_size / inference_time_avg;
    std::cout << fmt::format(
                     "ttnn_mobilenetv2_224x224_batch_size_{}. One inference iteration time (sec): {:.6f}, FPS: {:.2f}, "
                     "inference time (sec): {:.6f}, sync output time(sec): {:.6f}",
                     batch_size,
                     inference_time_avg,
                     fps,
                     inference_time,
                     sync_output_time)
              << std::endl;
}

int main() {
    auto device = ttnn::MeshDevice::create_unit_mesh(
        0,
        /*l1_small_size=*/24576,
        /*trace_region_size=*/6434816,
        /*num_command_queues=*/2,
        /*dispatch_core_config=*/tt::tt_metal::DispatchCoreConfig(tt::tt_metal::DispatchCoreType::ETH));
    int batch_size = 1;
    device->enable_program_cache();

    test_run_mobilenetv2_trace_2cqs_inference(device, batch_size);

    device->disable_and_clear_program_cache();
    return 0;
}
