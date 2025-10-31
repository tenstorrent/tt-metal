// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <iostream>
#include "helper_funcs.h"
#include "inference/mobilenetv2_infra.h"
#include "util/profiler.hpp"

int main() {
    auto device = ttnn::MeshDevice::create_unit_mesh(
        0,
        /*l1_small_size=*/24576,
        /*trace_region_size=*/6434816,
        /*num_command_queues=*/2,
        /*dispatch_core_config=*/tt::tt_metal::DispatchCoreConfig(tt::tt_metal::DispatchCoreType::ETH));
    int batch_size = 1;
    auto infra = std::make_shared<MobileNetv2TestInfra>(device, batch_size);

    // First run configures convs JIT
    (*infra).run();
    (*infra).validate();
    (*infra).deallocOutput();

    // Optimized run
    Profiler profiler;
    for (int i = 0; i < 10; i++) {
        profiler.start("inference_time");
        (*infra).run();
        profiler.stop("inference_time");
    }

    auto inference_time = profiler.get("inference_time");
    double inference_time_avg = inference_time / 10.0;
    double fps = batch_size / inference_time_avg;
    std::cout << fmt::format(
                     "ttnn_mobilenetv2_224x224_batch_size_{}. One inference iteration time (sec): {:.6f}, FPS: {:.2f}",
                     batch_size,
                     inference_time_avg,
                     fps)
              << std::endl;
    return 0;
}
