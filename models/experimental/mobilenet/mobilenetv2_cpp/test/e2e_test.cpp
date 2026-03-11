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

double measure_time(
    std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point end) {
    return std::chrono::duration<double>(end - start).count();
}

void test_run_mobilenetv2_trace_2cqs_inference(
    const std::shared_ptr<ttnn::MeshDevice>& device, int batch_size, const std::string& weights_dir) {
    auto host_input = create_mobilenetv2_host_input(batch_size, 3, 224, 224);
    ttnn::Tensor tt_inputs_host = host_input_to_ttnn(host_input);

    int n = batch_size;
    int h = 224;
    int w = 224;

    tt_inputs_host = tt_inputs_host.pad(
        tt::tt_metal::Shape{1, 1, static_cast<uint32_t>(n * h * w), 16}, tt::tt_metal::Shape{0, 0, 0, 0}, 0);

    MobileNetV2Trace2CQ mobilenetv2_trace_2cq;
    mobilenetv2_trace_2cq.initialize_mobilenetv2_trace_2cqs_inference(device, batch_size, weights_dir);

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

    mobilenetv2_trace_2cq.release_mobilenetv2_trace_2cqs_inference();

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

int main(int argc, char** argv) {
    auto device = ttnn::MeshDevice::create_unit_mesh(
        0,
        /*l1_small_size=*/24576,
        /*trace_region_size=*/6434816,
        /*num_command_queues=*/2,
        /*dispatch_core_config=*/tt::tt_metal::DispatchCoreConfig(tt::tt_metal::DispatchCoreType::ETH));
    int batch_size = 1;
    device->enable_program_cache();

    std::string weights_dir = "";
    if (argc > 1) {
        weights_dir = argv[1];
    }

    test_run_mobilenetv2_trace_2cqs_inference(device, batch_size, weights_dir);

    device->disable_and_clear_program_cache();
    return 0;
}
