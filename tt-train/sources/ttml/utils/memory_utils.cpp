// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "memory_utils.hpp"

#include "ttnn/graph/graph_consts.hpp"
#include "ttnn/graph/graph_processor.hpp"

namespace ttml::utils {
using namespace ttnn::graph;

// Very simple implementation:
// - Only tracks (de)allocations. If allocation we increase the current usage, if deallocation we decrease it.
// This means that all tensors created before the trace capture are not tracked.
DRAMUsage extract_DRAM_usage(const nlohmann::json& trace) {
    DRAMUsage result;
    std::unordered_map<std::string, long long> current_buffer;

    for (size_t i = 0; i < trace.size(); ++i) {
        const auto& v = trace[i];

        if (v[kNodeType] == kNodeBufferAllocate && v[kParams][kType] == "DRAM") {
            auto device_id = v[kParams][kDeviceId].get<std::string>();
            size_t buffer_size = std::stoll(v[kParams][kSize].get<std::string>());
            fmt::print("[kNodeBufferAllocate] Found buffer: size: {} device: {}\n", buffer_size, device_id);
            current_buffer[device_id] += buffer_size;
        } else if (v[kNodeType] == kNodeBufferDeallocate) {
            auto connection = v[kConnections][0].get<int>();
            auto buffer = trace[connection];
            if (buffer[kParams][kType] == "DRAM") {
                auto device_id = v[kParams][kDeviceId].get<std::string>();
                size_t buffer_size = std::stoll(buffer[kParams][kSize].get<std::string>());
                fmt::print("[kNodeBufferDeallocate] Deallocated buffer: size: {} device: {}\n", buffer_size, device_id);
                current_buffer[device_id] -= buffer_size;
            }
        }

        // Track peak per device
        for (auto& [device_id, total] : current_buffer) {
            result.peak[device_id] = std::max(result.peak[device_id], total);
        }
    }

    // Current usage is whatever remains allocated at end of trace
    result.current = current_buffer;
    return result;
}

// L1Usage extract_L1_usage(const nlohmann::json& trace) {
//     L1Usage result;
//     std::unordered_map<std::string, long long> current_cb;
//     std::unordered_map<std::string, long long> current_buffer;

//     for (size_t i = 0; i < trace.size(); ++i) {
//         const auto& v = trace[i];

//         if (v[kNodeType] == kNodeCBAllocate) {
//             auto device_id = v[kParams][kDeviceId].get<std::string>();
//             current_cb[device_id] += std::stoll(v[kParams][kSize].get<std::string>());
//             result.peak_cb[device_id] = std::max(result.peak_cb[device_id], current_cb[device_id]);
//         } else if (v[kNodeType] == kNodeCBDeallocateAll) {
//             auto device_id = v[kParams][kDeviceId].get<std::string>();
//             current_cb[device_id] = 0;
//         } else if (v[kNodeType] == kNodeBufferAllocate && v[kParams][kType] == "L1") {
//             auto device_id = v[kParams][kDeviceId].get<std::string>();
//             current_buffer[device_id] += std::stoll(v[kParams][kSize].get<std::string>());
//             result.peak_buffer[device_id] = std::max(result.peak_buffer[device_id], current_buffer[device_id]);
//         } else if (v[kNodeType] == kNodeBufferDeallocate) {
//             auto connection = v[kConnections][0].get<int>();
//             auto buffer = trace[connection];
//             if (buffer[kParams][kType] == "L1") {
//                 auto device_id = v[kParams][kDeviceId].get<std::string>();
//                 current_buffer[device_id] -= std::stoll(buffer[kParams][kSize].get<std::string>());
//             }
//         }

//         // Track peak total (CB + buffer) per device
//         for (const auto& [device_id, cb] : current_cb) {
//             auto total = cb + current_buffer[device_id];
//             result.peak_total[device_id] = std::max(result.peak_total[device_id], total);
//         }
//         for (const auto& [device_id, buf] : current_buffer) {
//             if (current_cb.find(device_id) == current_cb.end()) {
//                 result.peak_total[device_id] = std::max(result.peak_total[device_id], buf);
//             }
//         }
//     }

//     // Current L1 buffer usage at end of trace (CBs are typically deallocated)
//     result.current = current_buffer;
//     return result;
// }

// namespace MemoryUsageTracker {
// static std::shared_ptr<ttnn::graph::GraphProcessor> graph_processor;
// static nlohmann::json trace;

// void start_capture() {
//     auto mode = tt::tt_metal::IGraphProcessor::RunMode::NORMAL;
//     graph_processor = std::make_shared<ttnn::graph::GraphProcessor>(mode);
//     graph_processor->begin_graph_capture(mode);
// }

// void end_capture() {
//     trace = graph_processor->end_graph_capture();
// }

// DRAMUsage get_DRAM_usage() {
//     if (trace.empty()) {
//         fmt::print("WARNING: Calling get_DRAM_usage() before trace capture\n");
//     }
//     return extract_DRAM_usage(trace);
// }

// L1Usage get_L1_usage() {
//     if (trace.empty()) {
//         fmt::print("WARNING: Calling get_L1_usage() before trace capture\n");
//     }
//     return extract_L1_usage(trace);
// }

// void print_memory_usage() {
//     auto dram_usage = extract_DRAM_usage(trace);
//     auto l1_usage = extract_L1_usage(trace);

//     fmt::print("=== Memory Usage Summary ===\n");

//     // Print DRAM usage per device
//     for (const auto& [dev_id, peak] : dram_usage.peak) {
//         fmt::print(
//             "Device {}: Peak DRAM {:.2f} MB, Current DRAM {:.2f} MB\n",
//             dev_id,
//             peak / 1024.0 / 1024.0,
//             dram_usage.current[dev_id] / 1024.0 / 1024.0);
//     }

//     // Print L1 usage per device
//     for (const auto& [dev_id, peak_total] : l1_usage.peak_total) {
//         fmt::print(
//             "Device {}: Peak L1 CB {:.2f} MB, Peak L1 Buffer {:.2f} MB, Peak L1 Total {:.2f} MB, Current L1 {:.2f} "
//             "MB\n",
//             dev_id,
//             l1_usage.peak_cb[dev_id] / 1024.0 / 1024.0,
//             l1_usage.peak_buffer[dev_id] / 1024.0 / 1024.0,
//             peak_total / 1024.0 / 1024.0,
//             l1_usage.current[dev_id] / 1024.0 / 1024.0);
//     }
// }
// }  // namespace MemoryUsageTracker

// }  // namespace ttml::utils
