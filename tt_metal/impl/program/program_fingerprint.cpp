// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/program_fingerprint.hpp>

#include <algorithm>
#include <cstdint>
#include <map>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include <enchantum/enchantum.hpp>
#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <tt-metalium/program.hpp>

#include "impl/buffers/circular_buffer.hpp"
#include "impl/buffers/semaphore.hpp"
#include "impl/kernels/kernel.hpp"
#include "program_impl.hpp"

namespace tt::tt_metal::experimental {

namespace {

// FNV-1a: a hash of the kernel's text that is the same in every process.
uint64_t fnv1a(const std::string& text) {
    uint64_t h = 1469598103934665603ULL;
    for (unsigned char c : text) {
        h ^= c;
        h *= 1099511628211ULL;
    }
    return h;
}

// A core range set as sorted [x0, y0, x1, y1] rectangles.
nlohmann::json ranges(const CoreRangeSet& set) {
    std::vector<std::vector<uint32_t>> out;
    for (const auto& r : set.ranges()) {
        out.push_back(
            {static_cast<uint32_t>(r.start_coord.x),
             static_cast<uint32_t>(r.start_coord.y),
             static_cast<uint32_t>(r.end_coord.x),
             static_cast<uint32_t>(r.end_coord.y)});
    }
    std::sort(out.begin(), out.end());
    return out;
}

// The kernel's config as text, in the form of its config hash (DataMovementKernel::config_hash and the others, which
// are not public): processor and NoC for data movement and Ethernet kernels, the math settings for compute kernels.
std::string config_text(const Kernel& kernel) {
    return std::visit(
        [](const auto& c) -> std::string {
            using T = std::decay_t<decltype(c)>;
            if constexpr (std::is_same_v<T, DataMovementConfig>) {
                return fmt::format(
                    "{}_{}_{}",
                    enchantum::to_string(c.processor),
                    enchantum::to_string(c.noc),
                    enchantum::to_string(c.noc_mode));
            } else if constexpr (std::is_same_v<T, EthernetConfig>) {
                return fmt::format(
                    "eth_{}_{}_{}_{}",
                    enchantum::to_string(c.processor),
                    enchantum::to_string(c.noc),
                    enchantum::to_string(c.noc_mode),
                    enchantum::to_string(c.eth_mode));
            } else if constexpr (std::is_same_v<T, ComputeConfig>) {
                return fmt::format(
                    "compute_{}_{}_{}_{}_{}",
                    enchantum::to_string(c.math_fidelity),
                    c.fp32_dest_acc_en,
                    c.dst_full_sync_en,
                    c.math_approx_mode,
                    c.bfp8_pack_precise);
            } else {
                return "other";
            }
        },
        kernel.config());
}

nlohmann::json kernel_json(Kernel& kernel) {
    nlohmann::json k;
    const auto& src = kernel.kernel_source();
    k["name"] = kernel.name();
    k["source"] = src.source_type_ == KernelSource::FILE_PATH ? src.path_.string() : std::string("inline");
    std::string text;
    try {
        text = src.get_content();
    } catch (const std::exception&) {
        text = src.source_;  // the file is gone: hash what the kernel was created from
    }
    k["source_hash"] = fmt::format("{:016x}", fnv1a(text));
    k["core_type"] = std::string(enchantum::to_string(kernel.get_kernel_programmable_core_type()));
    k["processor_class"] = std::string(enchantum::to_string(kernel.get_kernel_processor_class()));
    k["config"] = config_text(kernel);
    k["cores"] = ranges(kernel.core_range_set());
    k["ct_args"] = kernel.compile_time_args();
    const auto named = kernel.named_compile_time_args();
    k["named_ct_args"] = std::map<std::string, uint32_t>(named.begin(), named.end());
    k["defines"] = kernel.defines();
    nlohmann::json rt = nlohmann::json::array();
    for (const auto& core : kernel.cores_with_runtime_args()) {  // a std::set: sorted
        rt.push_back({{"core", {core.x, core.y}}, {"args", kernel.runtime_args(core)}});
    }
    k["rt_args"] = std::move(rt);
    k["common_rt_args"] = kernel.common_runtime_args();
    return k;
}

nlohmann::json cb_json(const CircularBufferImpl& cb) {
    nlohmann::json c;
    c["cores"] = ranges(cb.core_ranges());
    c["total_size"] = cb.size();
    c["globally_allocated"] = cb.globally_allocated();
    c["address"] = cb.address();
    std::vector<uint32_t> indices(cb.buffer_indices().begin(), cb.buffer_indices().end());
    std::sort(indices.begin(), indices.end());
    nlohmann::json buffers = nlohmann::json::array();
    for (uint32_t i : indices) {
        buffers.push_back(
            {{"index", i},
             {"page_size", cb.page_size(i)},
             {"data_format", std::string(enchantum::to_string(cb.data_format(i)))}});
    }
    c["buffers"] = std::move(buffers);
    return c;
}

}  // namespace

std::string program_fingerprint_json(Program& program) {
    auto& impl = program.impl();
    nlohmann::json p;
    nlohmann::json kernels = nlohmann::json::array();
    for (size_t handle = 0; handle < impl.num_kernels(); ++handle) {
        auto kernel = impl.get_kernel(static_cast<KernelHandle>(handle));
        kernels.push_back(kernel ? kernel_json(*kernel) : nlohmann::json(nullptr));
    }
    p["kernels"] = std::move(kernels);
    nlohmann::json cbs = nlohmann::json::array();
    for (const auto& cb : impl.circular_buffers()) {  // creation order: the order of the program's CBs
        cbs.push_back(cb_json(*cb));
    }
    p["cbs"] = std::move(cbs);
    nlohmann::json sems = nlohmann::json::array();
    for (const auto& s : impl.semaphores()) {
        sems.push_back(
            {{"cores", ranges(s.core_range_set())},
             {"id", s.id()},
             {"initial_value", s.initial_value()},
             {"core_type", std::string(enchantum::to_string(s.core_type()))}});
    }
    p["semaphores"] = std::move(sems);
    return p.dump();
}

}  // namespace tt::tt_metal::experimental
