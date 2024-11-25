// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "modules/multi_layer_perceptron.hpp"
#include "ops/losses.hpp"
#include "ttnn/graph/graph_consts.hpp"
#include "ttnn/graph/graph_operation_queries.hpp"
#include "ttnn/graph/graph_processor.hpp"
#include "ttnn/graph/graph_trace_utils.hpp"

using ttml::autograd::TensorPtr;

namespace {

using namespace ttnn::graph;

long long extract_peak_DRAM_memory_usage(const nlohmann::json& trace) {
    long long total_buffer = 0;
    long long peak_memory_usage = 0;
    std::vector<std::string> current_op;

    for (size_t i = 0; i < trace.size(); ++i) {
        const auto& v = trace[i];

        if (v[kNodeType] == kNodeFunctionStart) {
            if (current_op.empty()) {
                while (++i < trace.size()) {
                    const auto& inner_v = trace[i];
                    if (inner_v[kNodeType] == "buffer" && inner_v[kParams][kType] == "DRAM") {
                        total_buffer += std::stoll(inner_v[kParams][kSize].get<std::string>());
                    } else if (inner_v[kNodeType] == kNodeTensor) {
                        continue;
                    } else {
                        break;
                    }
                }
                --i;  // adjust for loop increment
            }
            current_op.push_back(v[kParams][kName]);
        } else if (v[kNodeType] == kNodeBufferAllocate && v[kParams][kType] == "DRAM") {
            total_buffer += stoll(v[kParams][kSize].get<std::string>());
        } else if (v[kNodeType] == kNodeBufferDeallocate) {
            auto connection = v[kConnections][0].get<int>();
            auto buffer = trace[connection];
            if (buffer[kParams][kType] == "DRAM") {
                total_buffer -= stoll(buffer[kParams][kSize].get<std::string>());
            }
        } else if (v[kNodeType] == kNodeFunctionEnd) {
            current_op.pop_back();
        }

        peak_memory_usage = std::max(peak_memory_usage, total_buffer);
    }

    return peak_memory_usage;
}

}  // namespace

int main() {
    const size_t num_targets = 10;
    const uint32_t batch_size = 128;
    const size_t num_features = 784;
    auto* device = &ttml::autograd::ctx().get_device();

    auto batch = ttml::autograd::create_tensor(
        ttml::core::zeros(ttml::core::create_shape({batch_size, 1, 1, num_features}), device));
    auto target = ttml::autograd::create_tensor(
        ttml::core::zeros(ttml::core::create_shape({batch_size, 1, 1, num_targets}), device));

    auto model_params = ttml::modules::MultiLayerPerceptronParameters{
        .input_features = num_features, .hidden_features = {128}, .output_features = num_targets};
    auto model = ttml::modules::MultiLayerPerceptron(model_params);

    auto mode = tt::tt_metal::IGraphProcessor::RunMode::NO_DISPATCH;
    ttnn::graph::GraphProcessor graph_processor(mode);
    graph_processor.begin_graph_capture(mode);
    auto output = model(batch);
    auto loss = ttml::ops::cross_entropy_loss(output, target);
    auto forward_trace = graph_processor.end_graph_capture();
    auto forward_peak_l1_memory_usage = ttnn::graph::extract_peak_L1_memory_usage(forward_trace);
    auto forward_peak_DRAM_memory_usage = extract_peak_DRAM_memory_usage(forward_trace);

    auto call = [&] {
        loss->backward();
        return 0;
    };
    auto backward_trace = ttnn::graph::query_trace(call);
    auto backward_peak_l1_memory_usage = ttnn::graph::extract_peak_L1_memory_usage(backward_trace);
    auto backward_peak_DRAM_memory_usage = extract_peak_DRAM_memory_usage(backward_trace);

    auto pretty_forward_trace = forward_trace.dump(4);
    auto pretty_backward_trace = backward_trace.dump(4);

    const std::string path = "/home/ubuntu/graph_traces/";
    std::ofstream forward_trace_file(fmt::format("{}/forward_trace.json", path));
    forward_trace_file << pretty_forward_trace;
    forward_trace_file.close();

    std::ofstream backward_trace_file(fmt::format("{}/backward_trace.json", path));
    backward_trace_file << pretty_backward_trace;
    backward_trace_file.close();

    fmt::print("Forward peak L1 memory usage (in MB): {}\n", forward_peak_l1_memory_usage / 1024.0 / 1024.0);
    fmt::print("Forward peak DRAM memory usage (in MB): {}\n", forward_peak_DRAM_memory_usage / 1024.0 / 1024.0);
    fmt::print("Backward peak L1 memory usage (in MB): {}\n", backward_peak_l1_memory_usage / 1024.0 / 1024.0);
    fmt::print("Backward peak DRAM memory usage (in MB): {}\n", backward_peak_DRAM_memory_usage / 1024.0 / 1024.0);
    fmt::print("Forward trace saved to: {}/forward_trace.json\n", path);
    fmt::print("Backward trace saved to: {}/backward_trace.json\n", path);
    fmt::print("Capture complete\n");

    return 0;
}
