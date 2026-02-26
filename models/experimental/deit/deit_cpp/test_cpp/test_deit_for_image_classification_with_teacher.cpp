// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>

#include <torch/torch.h>
#include <torch/script.h>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/types.hpp>
#include <ttnn/device.hpp>
#include <ttnn/distributed/api.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>

#include "../tt_cpp/deit_config.h"
#include "../tt_cpp/deit_for_image_classification_with_teacher.h"
#include "../helper_funcs.h"
#include "../image_utils.h"
#include <iomanip>
#include <ttnn/operations/trace.hpp>
#include <ttnn/api/ttnn/events.hpp>

namespace {

/**
 * Load DeiT for Image Classification with Teacher model from file
 * @param model_path Path to the model file
 * @return Pair of state_dict and PyTorch model
 */
std::pair<std::unordered_map<std::string, torch::Tensor>, torch::jit::script::Module>
load_deit_image_classification_with_teacher_model(const std::string& model_path) {
    try {
        // Load the traced model
        torch::jit::script::Module model = torch::jit::load(model_path);
        model.eval();

        // Extract state dict from the model
        std::unordered_map<std::string, torch::Tensor> state_dict;

        // Get all named parameters from the model
        for (const auto& param : model.named_parameters()) {
            state_dict[param.name] = param.value;
        }

        // Get all named buffers from the model
        for (const auto& buffer : model.named_buffers()) {
            state_dict[buffer.name] = buffer.value;
        }

        std::cout << "Loaded DeiT with Teacher model with " << state_dict.size() << " parameters" << std::endl;

        return std::make_pair(state_dict, model);
    } catch (const std::exception& e) {
        std::cerr << "Failed to load model from " << model_path << ": " << e.what() << std::endl;
        throw;
    }
}

/**
 * Test DeiT for Image Classification with Teacher inference
 * @param model_path Path to the model file
 */
void test_deit_for_image_classification_with_teacher_inference(const std::string& model_path) {
    const double pcc_threshold = 0.95;

    // Initialize device
    auto device = ttnn::MeshDevice::create_unit_mesh(0,
                                                    /*l1_small_size=*/24576,
                                                    /*trace_region_size=*/6434816,
                                                    /*num_command_queues=*/2,
                                                    /*dispatch_core_config=*/tt::tt_metal::DispatchCoreConfig(tt::tt_metal::DispatchCoreType::ETH));

    // Setup base address
    std::string base_address = "model.";

    // Load state dict and model
    auto [state_dict, model] = load_deit_image_classification_with_teacher_model(model_path);

    // Use a sample image path for testing (you can replace this with any valid image file)
    std::string test_image_path ="models/experimental/deit/deit_cpp/deit_model/input_image.jpg";

    torch::Tensor pixel_values = image_utils::load_and_preprocess_image(test_image_path);

    std::cout << "Using load_and_preprocess_image function for input preprocessing" << std::endl;

    // Get PyTorch model output for reference
    torch::Tensor torch_averaged_logits, torch_cls_logits, torch_distillation_logits;
    try {
        // Call the model with pixel_values
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(pixel_values);

        auto output = model.forward(inputs);

        // Extract outputs from the model
        // The output structure depends on the traced model format
        if (output.isTuple()) {
            auto output_tuple = output.toTuple();
            torch_averaged_logits = output_tuple->elements()[0].toTensor(); // Averaged logits
            torch_cls_logits = output_tuple->elements()[1].toTensor(); // CLS logits
            torch_distillation_logits = output_tuple->elements()[2].toTensor(); // Distillation logits
        } else {
            // If only one output, assume it's the averaged logits
            torch_averaged_logits = output.toTensor();
        }

        std::cout << "PyTorch averaged logits shape: " << torch_averaged_logits.sizes() << std::endl;
        if (torch_cls_logits.defined()) {
            std::cout << "PyTorch CLS logits shape: " << torch_cls_logits.sizes() << std::endl;
        }
        if (torch_distillation_logits.defined()) {
            std::cout << "PyTorch distillation logits shape: " << torch_distillation_logits.sizes() << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Failed to get PyTorch model output: " << e.what() << std::endl;
        return;
    }

    // Create DeiT config
    DeiTConfig config;

    // Setup TT model
    TtDeiTForImageClassificationWithTeacher tt_model(config, state_dict, base_address, device);

    // Convert input to TT tensor
    // Permute to NHWC for Conv2d input (required by TtDeiTPatchEmbeddings)
    auto pixel_values_nhwc = pixel_values.permute({0, 2, 3, 1}).contiguous();
    auto tt_input = helper_funcs::from_torch(pixel_values_nhwc, ttnn::DataType::BFLOAT16, ttnn::Layout::ROW_MAJOR).to_device(device.get());

    // Run TT model inference
    std::optional<ttnn::Tensor> head_mask = std::nullopt;
    bool output_attentions = false;
    bool output_hidden_states = false;
    bool return_dict = true;

    auto [tt_averaged_logits, tt_cls_logits, tt_distillation_logits, attention_weights, hidden_states] = tt_model.forward(
        tt_input,
        head_mask.has_value() ? &head_mask.value() : nullptr,
        output_attentions,
        output_hidden_states,
        return_dict
    );

    // Convert TT outputs back to torch tensors
    auto tt_averaged_logits_host = ttnn::from_device(tt_averaged_logits);
    auto tt_averaged_output_torch = helper_funcs::to_torch(tt_averaged_logits_host);

    auto tt_cls_logits_host = ttnn::from_device(tt_cls_logits);
    auto tt_cls_output_torch = helper_funcs::to_torch(tt_cls_logits_host);

    auto tt_distillation_logits_host = ttnn::from_device(tt_distillation_logits);
    auto tt_distillation_output_torch = helper_funcs::to_torch(tt_distillation_logits_host);

    // Ensure output shapes match for comparison
    if (tt_averaged_output_torch.dim() > torch_averaged_logits.dim()) {
        tt_averaged_output_torch = tt_averaged_output_torch.squeeze(0); // Remove batch dimension if needed
    }
    if (tt_cls_output_torch.dim() > torch_cls_logits.dim()) {
        tt_cls_output_torch = tt_cls_output_torch.squeeze(0);
    }
    if (tt_distillation_output_torch.dim() > torch_distillation_logits.dim()) {
        tt_distillation_output_torch = tt_distillation_output_torch.squeeze(0);
    }

    // Compute PCC between PyTorch and TT outputs
    double pcc_averaged = helper_funcs::compute_pcc(torch_averaged_logits, tt_averaged_output_torch);
    double pcc_cls = torch_cls_logits.defined() ? helper_funcs::compute_pcc(torch_cls_logits, tt_cls_output_torch) : 0.0;
    double pcc_distillation = torch_distillation_logits.defined() ? helper_funcs::compute_pcc(torch_distillation_logits, tt_distillation_output_torch) : 0.0;

    // Log results
    std::cout << "PCC between PyTorch and TT averaged logits: " << pcc_averaged << std::endl;
    if (torch_cls_logits.defined()) {
        std::cout << "PCC between PyTorch and TT CLS logits: " << pcc_cls << std::endl;
    }
    if (torch_distillation_logits.defined()) {
        std::cout << "PCC between PyTorch and TT distillation logits: " << pcc_distillation << std::endl;
    }

    std::cout << "PyTorch averaged logits shape: " << torch_averaged_logits.sizes() << std::endl;
    std::cout << "TT averaged logits shape: " << tt_averaged_output_torch.sizes() << std::endl;
    std::cout << "TT CLS logits shape: " << tt_cls_output_torch.sizes() << std::endl;
    std::cout << "TT distillation logits shape: " << tt_distillation_output_torch.sizes() << std::endl;

    // Check if PCC meets threshold
    bool passed = pcc_averaged >= pcc_threshold;
    if (torch_cls_logits.defined()) {
        passed = passed && (pcc_cls >= pcc_threshold);
    }
    if (torch_distillation_logits.defined()) {
        passed = passed && (pcc_distillation >= pcc_threshold);
    }

    if (passed) {
        std::cout << "PASSED: DeiT For Image Classification with Teacher test" << std::endl;
    } else {
        std::cout << "FAILED: One or more PCC values are below threshold (" << pcc_threshold << ")" << std::endl;
    }

    // Profiling
    helper_funcs::Profiler profiler;
    int batch_size = 1;

    // Enable program cache for performance
    device->enable_program_cache();

    // Prepare inputs for Trace/2CQ
    auto tt_input_host = helper_funcs::from_torch(pixel_values_nhwc, ttnn::DataType::BFLOAT16, ttnn::Layout::ROW_MAJOR);

    // Warmup run
    {
         tt_model.forward(
            tt_input,
            head_mask.has_value() ? &head_mask.value() : nullptr,
            output_attentions,
            output_hidden_states,
            return_dict
        );
    }

    // Trace Capture
    auto tid = ttnn::operations::trace::begin_trace_capture(device.get(), ttnn::QueueId(0));
    auto [tt_averaged_logits_trace, tt_cls_logits_trace, tt_distillation_logits_trace, attention_weights_trace, hidden_states_trace] = tt_model.forward(
        tt_input,
        head_mask.has_value() ? &head_mask.value() : nullptr,
        output_attentions,
        output_hidden_states,
        return_dict
    );
    ttnn::operations::trace::end_trace_capture(device.get(), tid, ttnn::QueueId(0));

    // Events for 2CQ
    auto op_event = ttnn::events::record_mesh_event(device.get(), ttnn::QueueId(0));
    auto write_event = ttnn::events::record_mesh_event(device.get(), ttnn::QueueId(1));

    for (int i = 0; i < 10; i++) {
        profiler.start("inference_time");
        
        // CQ1: Update Input (Async Copy)
        ttnn::events::wait_for_mesh_event(ttnn::QueueId(1), op_event);
        tt::tt_metal::copy_to_device(tt_input_host, tt_input, ttnn::QueueId(1));
        write_event = ttnn::events::record_mesh_event(device.get(), ttnn::QueueId(1));

        // CQ0: Execute Trace
        ttnn::events::wait_for_mesh_event(ttnn::QueueId(0), write_event);
        op_event = ttnn::events::record_mesh_event(device.get(), ttnn::QueueId(0));
        ttnn::operations::trace::execute_trace(device.get(), tid, ttnn::QueueId(0), false);
        
        profiler.stop("inference_time");

        profiler.start("sync_output");
        auto tt_averaged_logits_host_prof = ttnn::from_device(tt_averaged_logits_trace);
        profiler.stop("sync_output");
    }

    ttnn::operations::trace::release_trace(device.get(), tid);
    device->disable_and_clear_program_cache();

    double inference_time_total = profiler.get("inference_time");
    double sync_time_total = profiler.get("sync_output");
    double inference_time_avg = (inference_time_total) / 10.0;
    double fps = batch_size / inference_time_avg;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "ttnn_deit_for_image_classification_with_teacher_batch_size_" << batch_size 
              << ". One inference iteration time (sec): " << inference_time_avg 
              << ", FPS: " << std::setprecision(2) << fps
              << ", inference time (sec): " << std::setprecision(6) << (inference_time_total / 10.0)
              << ", sync output time(sec): " << (sync_time_total / 10.0) << std::endl;

    // Clean up device resources
    device->close();
}


} // anonymous namespace

int main(int argc, char** argv) {
    try {
        std::cout << "Starting DeiT for Image Classification with Teacher test..." << std::endl;

        if (argc != 2) {
            std::cerr << "Usage: " << argv[0] << " <model_path>" << std::endl;
            return -1;
        }
        std::string model_path = argv[1];

        std::cout << "Using model path: " << model_path << std::endl;

        // Test DeiT with Teacher inference
        test_deit_for_image_classification_with_teacher_inference(model_path);

        std::cout << "DeiT for Image Classification with Teacher test completed successfully!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with error: " << e.what() << std::endl;
        return 1;
    }
}
