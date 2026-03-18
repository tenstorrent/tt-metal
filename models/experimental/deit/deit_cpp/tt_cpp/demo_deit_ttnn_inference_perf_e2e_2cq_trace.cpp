// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "deit_test_infra.hpp"
#include "ttnn/device.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/sharded/reshard/reshard.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/profiler_types.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <chrono>
#include <iostream>
#include <iomanip>

using namespace ttnn;
using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

// Helper function to get logits from output (if tuple)
Tensor get_logits(const Tensor& output) {
    // In C++, DeitTestInfra::run returns a single Tensor, so just return it.
    // If it returned a tuple, we would need to extract it.
    return output;
}

// Helper to deallocate output
void deallocate_output(Tensor& output) { output.deallocate(); }

// Helper to dump device profiler
void dump_device_profiler(tt::tt_metal::distributed::MeshDevice* device) {
    tt::tt_metal::ReadMeshDeviceProfilerResults(*device, tt::tt_metal::ProfilerReadState::NORMAL);
}

void run_trace_2cq_model(
    MeshDevice* device,
    std::shared_ptr<deit_inference::DeitTestInfra> test_infra,
    const std::string& model_name,
    int num_warmup_iterations,
    int num_measurement_iterations,
    int batch_size) {
    std::cout << "Running trace 2CQ model with model_name: " << model_name << std::endl;
    // Setup inputs
    auto [tt_inputs_host, sharded_mem_config_DRAM, input_mem_config] = test_infra->setup_dram_sharded_input();

    // Initial move to device (DRAM)
    auto tt_image_res = ttnn::to_device(tt_inputs_host, device, sharded_mem_config_DRAM);

    // Initialize events
    // Python: op_event = ttnn.record_event(device, 0)
    auto op_event = device->mesh_command_queue(0).enqueue_record_event();

    std::cout << "Compiling..." << std::endl;
    auto start_compile = std::chrono::high_resolution_clock::now();

    // Compile run
    // Python: ttnn.wait_for_event(1, op_event)
    device->mesh_command_queue(1).enqueue_wait_for_event(op_event);

    // Python: ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res, 1)
    tt::tt_metal::copy_to_device(tt_inputs_host, tt_image_res, QueueId(1));

    // Python: write_event = ttnn.record_event(device, 1)
    auto write_event = device->mesh_command_queue(1).enqueue_record_event();

    // Python: ttnn.wait_for_event(0, write_event)
    device->mesh_command_queue(0).enqueue_wait_for_event(write_event);

    // Python: test_infra.input_tensor = ttnn.to_memory_config(tt_image_res, input_mem_config)
    test_infra->input_tensor = ttnn::to_memory_config(tt_image_res, input_mem_config, std::nullopt);

    // Python: spec = test_infra.input_tensor.spec
    auto spec = test_infra->input_tensor.tensor_spec();

    // Python: op_event = ttnn.record_event(device, 0)
    op_event = device->mesh_command_queue(0).enqueue_record_event();

    // Python: compile_output = test_infra.run()
    auto compile_output = test_infra->run();

    // Python: _ = ttnn.from_device(get_logits(compile_output), blocking=True)
    auto logits = get_logits(compile_output).cpu();

    auto end_compile = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> compile_duration = end_compile - start_compile;
    std::cout << "Compile done in " << compile_duration.count() << "s" << std::endl;

    dump_device_profiler(device);

    // Cache run
    std::cout << "Running cache pass..." << std::endl;
    auto start_cache = std::chrono::high_resolution_clock::now();
    device->mesh_command_queue(1).enqueue_wait_for_event(op_event);
    tt::tt_metal::copy_to_device(tt_inputs_host, tt_image_res, QueueId(1));
    write_event = device->mesh_command_queue(1).enqueue_record_event();
    device->mesh_command_queue(0).enqueue_wait_for_event(write_event);

    test_infra->input_tensor = ttnn::to_memory_config(tt_image_res, input_mem_config, std::nullopt);

    op_event = device->mesh_command_queue(0).enqueue_record_event();
    deallocate_output(test_infra->output_tensor);
    auto cache_output = test_infra->run();
    logits = get_logits(cache_output).cpu();

    auto end_cache = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cache_duration = end_cache - start_cache;
    std::cout << "Cache pass done in " << cache_duration.count() << "s" << std::endl;

    dump_device_profiler(device);

    // Capture
    std::cout << "Capturing trace..." << std::endl;
    device->mesh_command_queue(1).enqueue_wait_for_event(op_event);
    tt::tt_metal::copy_to_device(tt_inputs_host, tt_image_res, QueueId(1));
    write_event = device->mesh_command_queue(1).enqueue_record_event();

    device->mesh_command_queue(0).enqueue_wait_for_event(write_event);
    test_infra->input_tensor = ttnn::to_memory_config(tt_image_res, input_mem_config, std::nullopt);
    op_event = device->mesh_command_queue(0).enqueue_record_event();
    deallocate_output(test_infra->output_tensor);

    auto reshard_out = tt::tt_metal::create_device_tensor(spec, device);
    auto tid = tt::tt_metal::distributed::BeginTraceCapture(device, 0);
    auto tt_output_res = test_infra->run();
    auto tt_output_logits = get_logits(tt_output_res);
    device->end_mesh_trace(0, tid);
    std::cout << "Trace captured. Trace ID: " << tid << std::endl;

    dump_device_profiler(device);

    // Warmup
    std::cout << "Warmup..." << std::endl;

    for (int i = 0; i < num_warmup_iterations; ++i) {
        device->mesh_command_queue(1).enqueue_wait_for_event(op_event);
        tt::tt_metal::copy_to_device(tt_inputs_host, tt_image_res, QueueId(1));
        write_event = device->mesh_command_queue(1).enqueue_record_event();
        device->mesh_command_queue(0).enqueue_wait_for_event(write_event);

        reshard_out = ttnn::reshard(tt_image_res, input_mem_config, reshard_out);

        op_event = device->mesh_command_queue(0).enqueue_record_event();

        device->replay_mesh_trace(0, tid, true);

        dump_device_profiler(device);
    }

    tt::tt_metal::distributed::Synchronize(device, std::nullopt);

    // Measurement
    std::cout << "Measuring performance..." << std::endl;
    auto start_run = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_measurement_iterations; ++i) {
        device->mesh_command_queue(1).enqueue_wait_for_event(op_event);
        tt::tt_metal::copy_to_device(tt_inputs_host, tt_image_res, QueueId(1));
        write_event = device->mesh_command_queue(1).enqueue_record_event();
        device->mesh_command_queue(0).enqueue_wait_for_event(write_event);

        reshard_out = ttnn::reshard(tt_image_res, input_mem_config, reshard_out);

        op_event = device->mesh_command_queue(0).enqueue_record_event();

        device->replay_mesh_trace(0, tid, false);
    }

    tt::tt_metal::distributed::Synchronize(device, std::nullopt);
    auto end_run = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> run_duration = end_run - start_run;

    double avg_inference_time = run_duration.count() / num_measurement_iterations;
    double first_iter_time = compile_duration.count() + cache_duration.count();
    double compile_time = first_iter_time - 2 * avg_inference_time;

    std::cout << "ttnn_deit_base_batch_size" << batch_size << " inference time (avg): " << avg_inference_time << " s"
              << std::endl;
    std::cout << "ttnn_deit_base_batch_size" << batch_size << " compile time: " << compile_time << " s" << std::endl;
    std::cout << "Samples per second: " << 1.0 / avg_inference_time * batch_size << std::endl;

    dump_device_profiler(device);

    // Release trace
    device->release_mesh_trace(tid);
}

int main(int argc, char** argv) {
    int batch_size = 1;  // Default batch size

    if (argc > 1) {
        try {
            batch_size = std::stoi(argv[1]);
        } catch (const std::exception& e) {
            std::cerr << "Invalid batch size argument: " << argv[1] << ", using default: 1" << std::endl;
        }
    }

    // Initialize device
    // Assuming 8x8 mesh or similar, but for demo we usually use whatever is available
    // ttnn::open_device or MeshDevice::create
    // The Python demo uses a pytest fixture which handles device creation.
    // Here we need to create it manually.
    // Assuming a standard configuration.

    // We need to know the number of devices.
    // Get total system memory or skip logic since device count is not needed directly

    size_t l1_small_size = 32768;
    size_t trace_region_size = 1700000;
    size_t num_command_queues = 2;

    // Create MeshDevice
    // Note: This API might vary depending on the exact checkout, but standard is create
    // We will use a helper if available, or just create it.
    // MeshDevice::create_unit_mesh(device_id, l1_small, trace_region, num_cqs)
    // Using device 0.

    std::shared_ptr<MeshDevice> device =
        MeshDevice::create_unit_mesh(0, l1_small_size, trace_region_size, num_command_queues);

    std::cout << "Device initialized." << std::endl;

    std::string model_name;
    if (argc > 2) {
        model_name = argv[2];
    } else {
        model_name = "models/experimental/deit/deit_cpp/deit_model/deit_teacher_model.pt";
    }
    auto test_infra = deit_inference::create_test_infra(device.get(), batch_size, model_name);

    int num_warmup_iterations = 5;
    int num_measurement_iterations = 15;

    run_trace_2cq_model(
        device.get(), test_infra, model_name, num_warmup_iterations, num_measurement_iterations, batch_size);

    // Device destructor handles closing
    return 0;
}
