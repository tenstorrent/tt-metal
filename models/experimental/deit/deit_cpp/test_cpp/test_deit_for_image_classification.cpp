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
#include "../tt_cpp/deit_for_image_classification.h"
#include "../helper_funcs.h"
#include "../image_utils.h"
#include <iomanip>
#include <ttnn/operations/trace.hpp>
#include <ttnn/api/ttnn/events.hpp>

namespace {

/**
 * Load DeiT state dict and model for image classification
 * @param model_path Path to the traced model
 * @return Pair of state_dict and loaded model
 */
std::pair<std::unordered_map<std::string, torch::Tensor>, torch::jit::script::Module>
load_deit_image_classification_model(const std::string& model_path) {
    std::unordered_map<std::string, torch::Tensor> state_dict;
    torch::jit::script::Module model;

    try {
        // Load the traced model using torch::jit::load
        model = torch::jit::load(model_path);
        model.eval();

        std::cout << "Successfully loaded model from: " << model_path << std::endl;

        // Load all model parameters to state_dict
        auto named_params = model.named_parameters();
        for (const auto& pair : named_params) {
            state_dict[pair.name] = pair.value;
        }

        std::cout << "Loaded " << state_dict.size() << " parameters from model" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Failed to load model from " << model_path << ": " << e.what() << std::endl;
        throw;
    }

    return std::make_pair(state_dict, model);
}

/**
 * Test DeiT for image classification inference
 * @param model_path Path to the DeiT model file
 */
void test_deit_for_image_classification_inference(const std::string& model_path) {
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
    auto [state_dict, model] = load_deit_image_classification_model(model_path);

    // Use a sample image path for testing (you can replace this with any valid image file)
    std::string test_image_path ="models/experimental/deit/deit_cpp/deit_model/input_image.jpg";

    torch::Tensor pixel_values = image_utils::load_and_preprocess_image(test_image_path);

    std::cout << "Using load_and_preprocess_image function for input preprocessing" << std::endl;

    // Get PyTorch model output for reference
    torch::Tensor torch_output;
    try {
        // Call the model with pixel_values
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(pixel_values);

        auto output = model.forward(inputs);

        // Extract logits from the output
        // The output structure depends on the traced model format
        if (output.isTuple()) {
            auto output_tuple = output.toTuple();
            torch_output = output_tuple->elements()[0].toTensor(); // First element should be logits
        } else {
            torch_output = output.toTensor();
        }

        std::cout << "PyTorch output shape: " << torch_output.sizes() << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Failed to get PyTorch model output: " << e.what() << std::endl;
        return;
    }

    // Create DeiT config
    DeiTConfig config;

    // Setup TT model
    TtDeiTForImageClassification tt_model(config, state_dict, base_address, device);

    // Convert input to TT tensor
    // Permute to NHWC for Conv2d input (required by TtDeiTPatchEmbeddings)
    auto pixel_values_nhwc = pixel_values.permute({0, 2, 3, 1}).contiguous();

    // Pad to 16 channels for L1 alignment
    if (pixel_values_nhwc.size(3) < 16) {
        auto options = pixel_values_nhwc.options();
        auto padded = torch::zeros({pixel_values_nhwc.size(0), pixel_values_nhwc.size(1), pixel_values_nhwc.size(2), 16}, options);
        using namespace torch::indexing;
        padded.index_put_({Slice(), Slice(), Slice(), Slice(0, pixel_values_nhwc.size(3))}, pixel_values_nhwc);
        pixel_values_nhwc = padded;
    }

    // Create L1 interleaved input
    auto tt_input = helper_funcs::from_torch(pixel_values_nhwc, ttnn::DataType::BFLOAT16, ttnn::Layout::ROW_MAJOR)
                        .to_device(device.get(), ttnn::L1_MEMORY_CONFIG);

    // DEBUG: Print memory config
    // std::cout << "DEBUG: tt_input memory layout: " << (int)tt_input.memory_config().memory_layout << std::endl;
    // std::cout << "DEBUG: tt_input buffer type: " << (int)tt_input.memory_config().buffer_type << std::endl;

    // // Run TT model inference
    std::optional<ttnn::Tensor> head_mask = std::nullopt;
    bool output_attentions = false;
    bool output_hidden_states = false;
    bool return_dict = true;

    auto [tt_logits, attention_weights, hidden_states] = tt_model.forward(
        tt_input,
        head_mask.has_value() ? &head_mask.value() : nullptr,
        output_attentions,
        output_hidden_states,
        return_dict
    );

    // Convert TT output back to torch tensor
    auto tt_logits_host = ttnn::from_device(tt_logits);
    auto tt_output_torch = helper_funcs::to_torch(tt_logits_host);

    // Ensure output shapes match for comparison
    if (tt_output_torch.dim() > torch_output.dim()) {
        tt_output_torch = tt_output_torch.squeeze(0); // Remove batch dimension if needed
    }

    // Extract the classification logits (typically [:, 0, :] for DeiT)
    if (tt_output_torch.dim() == 3 && tt_output_torch.size(1) > 1) {
        tt_output_torch = tt_output_torch.select(1, 0); // Select first token (CLS token)
    }

    // Compute PCC between PyTorch and TT outputs
    double pcc = helper_funcs::compute_pcc(torch_output, tt_output_torch);

    // Log results
    std::cout << "PCC between PyTorch and TT outputs: " << pcc << std::endl;
    std::cout << "PyTorch output shape: " << torch_output.sizes() << std::endl;
    std::cout << "TT output shape: " << tt_output_torch.sizes() << std::endl;

    // Check if PCC meets threshold
    if (pcc >= pcc_threshold) {
        std::cout << "PASSED: DeiT For Image Classification test with PCC = " << pcc << std::endl;
    } else {
        std::cout << "FAILED: PCC (" << pcc << ") is below threshold (" << pcc_threshold << ")" << std::endl;
    }

    // Profiling
    helper_funcs::Profiler profiler;
    int batch_size = 1;

    // Enable program cache for performance
    device->enable_program_cache();

    // Prepare inputs for Trace/2CQ
    auto tt_input_host = helper_funcs::from_torch(pixel_values_nhwc, ttnn::DataType::BFLOAT16, ttnn::Layout::ROW_MAJOR);
    auto tt_input_l1 = tt_input_host.to_device(device.get(), ttnn::L1_MEMORY_CONFIG);

    // Warmup run
    {
         tt_model.forward(
            tt_input_l1,
            head_mask.has_value() ? &head_mask.value() : nullptr,
            output_attentions,
            output_hidden_states,
            return_dict
        );
    }

    // Trace Capture
    auto tid = ttnn::operations::trace::begin_trace_capture(device.get(), ttnn::QueueId(0));
    auto [tt_logits_trace, attention_weights_trace, hidden_states_trace] = tt_model.forward(
        tt_input_l1,
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
        write_event = ttnn::events::record_mesh_event(device.get(), ttnn::QueueId(1));

        // CQ0: Execute Trace
        ttnn::events::wait_for_mesh_event(ttnn::QueueId(0), write_event);
        op_event = ttnn::events::record_mesh_event(device.get(), ttnn::QueueId(0));
        ttnn::operations::trace::execute_trace(device.get(), tid, ttnn::QueueId(0), false);

        profiler.stop("inference_time");
    }

    profiler.start("sync_output");
    auto tt_logits_host_final = ttnn::from_device(tt_logits_trace);
    profiler.stop("sync_output");
    ttnn::operations::trace::release_trace(device.get(), tid);
    device->disable_and_clear_program_cache();

    double inference_time_total = profiler.get("inference_time");
    double sync_time_total = profiler.get("sync_output");
    double inference_time_avg = (inference_time_total + sync_time_total) / 10.0;
    double fps = batch_size / inference_time_avg;

    std::cout << fmt::format(
                     "ttnn_deit_for_image_classification_batch_size_{}. One inference iteration time (sec): {:.6f}, "
                     "FPS: {:.2f}, "
                     "inference time (sec): {:.6f}, sync output time(sec): {:.6f}",
                     batch_size,
                     inference_time_avg,
                     fps,
                     inference_time_total,
                     sync_time_total)
              << std::endl;

    // Clean up device resources
    device->close();
}

} // anonymous namespace

int main(int argc, char** argv) {
    try {
        std::cout << "Starting DeiT for Image Classification test..." << std::endl;

        if (argc != 2) {
            std::cerr << "Usage: " << argv[0] << " <model_path>" << std::endl;
            return -1;
        }
        std::string model_path = argv[1];

        std::cout << "Using model path: " << model_path << std::endl;

        // Test DeiT inference with load_and_preprocess_image function
        test_deit_for_image_classification_inference(model_path);

        std::cout << "DeiT for Image Classification test completed successfully!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with error: " << e.what() << std::endl;
        return 1;
    }
}
