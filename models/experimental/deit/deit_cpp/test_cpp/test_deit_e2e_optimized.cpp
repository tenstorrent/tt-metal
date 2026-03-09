// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <fmt/format.h>
#include <torch/torch.h>
#include <ttnn/device.hpp>
#include "../inference/deit_e2e_performance.h"
#include "../helper_funcs.h"

void test_run_deit_trace_2cqs_inference(
    const std::shared_ptr<ttnn::MeshDevice>& device, int batch_size, const std::string& model_path) {
    auto torch_input = torch::randn({batch_size, 224, 224, 3});
    auto padded = torch::zeros({batch_size, 224, 224, 16});
    using namespace torch::indexing;
    padded.index_put_({Slice(), Slice(), Slice(), Slice(0, 3)}, torch_input);

    ttnn::Tensor tt_input_host = helper_funcs::from_torch(padded, ttnn::DataType::BFLOAT16, ttnn::Layout::ROW_MAJOR);

    DeiTTrace2CQ deit_trace_2cq;
    deit_trace_2cq.initialize_deit_trace_2cqs_inference(device, batch_size, model_path);

    int inference_iter_count = 10;
    helper_funcs::Profiler pf;
    for (int iter = 0; iter < inference_iter_count; ++iter) {
        pf.start("inference_time");
        deit_trace_2cq.execute_deit_trace_2cqs_inference(tt_input_host);
        pf.stop("inference_time");
    }

    pf.start("sync_output");
    deit_trace_2cq.get_output();
    pf.stop("sync_output");

    deit_trace_2cq.release_deit_trace_2cqs_inference();

    auto inference_time = pf.get("inference_time");
    auto sync_output_time = pf.get("sync_output");
    double inference_time_avg = (inference_time + sync_output_time) / inference_iter_count;
    double fps = batch_size / inference_time_avg;

    std::cout << fmt::format(
                     "ttnn_deit_224x224_batch_size_{}. One inference iteration time (sec): {:.6f}, FPS: {:.2f}, "
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
        0, 24576, 6434816, 2, tt::tt_metal::DispatchCoreConfig(tt::tt_metal::DispatchCoreType::ETH));

    device->enable_program_cache();

    std::string model_path = (argc > 1) ? argv[1] : "";
    test_run_deit_trace_2cqs_inference(device, 1, model_path);

    device->disable_and_clear_program_cache();
    return 0;
}
