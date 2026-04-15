// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "simple_trace_allocator.hpp"

#include <atomic>
#include <deque>
#include <filesystem>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <set>
#include <string>
#include <string_view>
#include <tuple>

#include "hal/generated/dev_msgs.hpp"
#include "impl/context/metal_context.hpp"
#include "llrt/hal.hpp"
#include <tt-logger/tt-logger.hpp>

namespace tt::tt_metal {

namespace {

std::string_view core_type_name(HalProgrammableCoreType core_type) {
    switch (core_type) {
        case HalProgrammableCoreType::TENSIX: return "TENSIX";
        case HalProgrammableCoreType::ACTIVE_ETH: return "ACTIVE_ETH";
        case HalProgrammableCoreType::IDLE_ETH: return "IDLE_ETH";
        case HalProgrammableCoreType::DRAM: return "DRAM";
        case HalProgrammableCoreType::COUNT: return "COUNT";
    }
    return "UNKNOWN";
}

}  // namespace

std::pair<std::optional<uint32_t>, std::optional<uint32_t>> SimpleTraceAllocator::RegionAllocator::allocate_region(
    uint32_t size, uint32_t trace_idx, uint32_t data_type, uint64_t program_id) {
    std::optional<uint32_t> best_addr;
    float best_cost = std::numeric_limits<float>::infinity();
    std::optional<uint32_t> best_region_sync_idx;
    uint32_t addr = 0;
    auto outer_it = regions_.begin();
    if (size == 0) {
        return {std::nullopt, 0};
    }

    // Set of regions that have no future uses, so they can be evicted to avoid cluttering up regions_.
    std::set<uint32_t> marked_for_deletion;

    // Once we've filled up the entire launch message buffer, we'll sync on that rather than on any older regions.
    constexpr uint32_t max_stall_history_size = dev_msgs::launch_msg_buffer_num_entries;

    // Iterate over possible placements, including the very beginning of the ringbuffer and starting immediately after
    // every region. One of these placements must be the best, since any other placement would be overlap the same or a
    // smaller number of allocations by moving it forward to one of those positions.
    // Then attempt to calculate the placement with the smallest total cost.
    // TODO: sweepline algorithm, so the best postion can be calculated in linear time relative to the number of
    // regions.
    while (true) {
        if (addr + size > ringbuffer_size_) {
            break;
        }
        float cost = 0;
        std::optional<uint32_t> region_sync_idx;
        bool now_in_use = false;
        // outer_it must be the first region that could possibly overlap [addr, addr + size), since in the last
        // iteration we selected addr as the first address after the old version of outer_it.
        for (auto it = outer_it; it != regions_.end(); ++it) {
            auto region = *it;
            if (region.first >= addr + size) {
                break;
            }
            if (intersects(addr, size, region.first, region.second.size)) {
                if (region.second.trace_idx == trace_idx) {
                    now_in_use = true;
                    break;
                }
                auto& next_use_idx = extra_data_[region.second.trace_idx].next_use_idx[region.second.data_type];
                if (next_use_idx.has_value()) {
                    if (*next_use_idx == trace_idx) {
                        // Really try to avoid evicting a buffer that will be used by this program.
                        constexpr uint32_t current_node_eviction_penalty = 1000000000;
                        cost += current_node_eviction_penalty;
                    } else {
                        // Similar to Belady's algorithm, we want to evict the region that will be used the farthest in
                        // the future, so cost is inversely proportional to distance to next use. Also take into account
                        // the size, since that's roughly proportional to the cost of adding the region back in. We
                        // could instead divide by the size, similar to the Belady-size algorithm, but has worked worse
                        // in simulation.
                        cost += region.second.size * 1.0f / (*next_use_idx - trace_idx);
                    }
                } else if (trace_idx - region.second.trace_idx > max_stall_history_size) {
                    // Region has no future uses and is too old to worry about stalls.
                    marked_for_deletion.insert(region.first);
                }
                region_sync_idx = merge_syncs(region_sync_idx, region.second.trace_idx);
            }
        }
        if (!now_in_use) {
            if (region_sync_idx.has_value()) {
                // Avoid evicting something that was last used recently, as that can cause a stall that is very bad for
                // performance. This is critical for avoiding gaps between ops, so it's given a very high cost (the
                // highest cost for a program is normally around 10,000).
                constexpr uint32_t desired_write_ahead = std::min(dev_msgs::launch_msg_buffer_num_entries, 7u);
                constexpr float stall_badness = 100000000;
                static_assert(
                    max_stall_history_size > desired_write_ahead,
                    "max_history_size must be greater than desired_write_ahead");
                int region_idx_diff = trace_idx - *region_sync_idx;
                if (region_idx_diff < desired_write_ahead) {
                    // Stall badness is exponential.
                    cost += stall_badness * (1 << (desired_write_ahead - region_idx_diff));
                }
            }
            if (cost < best_cost) {
                best_cost = cost;
                best_addr = addr;
                best_region_sync_idx = region_sync_idx;
            }
            if (cost == 0) {
                break;
            }
        }
        if (outer_it == regions_.end()) {
            break;
        }
        addr = outer_it->first + outer_it->second.size;
        outer_it++;
    }

    for (const auto& addr : marked_for_deletion) {
        auto it = regions_.find(addr);
        program_ids_memory_map_[it->second.data_type].erase(it->second.program_id);
        regions_.erase(it);
    }

    if (!best_addr.has_value()) {
        return {std::nullopt, best_addr};
    }

    // Evict overlapped regions.
    auto it = regions_.begin();
    while (it != regions_.end()) {
        if (intersects(*best_addr, size, it->first, it->second.size)) {
            program_ids_memory_map_[it->second.data_type].erase(it->second.program_id);
            it = regions_.erase(it);
        } else {
            ++it;
        }
    }
    regions_[*best_addr] = {trace_idx, data_type, size, program_id};
    return {best_region_sync_idx, best_addr};
}

void SimpleTraceAllocator::allocate_trace_programs(const Hal& hal, std::vector<TraceNode*>& trace_nodes) {
    std::map<uint64_t, uint32_t> program_ids_use_map;
    extra_data_.resize(trace_nodes.size());

    std::set<SubDeviceId> sub_device_ids;
    for (size_t i = trace_nodes.size(); i-- > 0;) {
        auto& node = *trace_nodes[i];
        auto it = program_ids_use_map.find(node.program->get_id());
        if (it != program_ids_use_map.end()) {
            // Binary is reused, but the nonbinary is not.
            extra_data_[i].next_use_idx[ExtraData::kBinary] = it->second;
        }
        program_ids_use_map[node.program->get_id()] = static_cast<uint32_t>(i);
        sub_device_ids.insert(node.sub_device_id);
    }
    for (const auto& sub_device_id : sub_device_ids) {
        for (auto& allocator : region_allocators_) {
            allocator.reset_allocator();
        }
        allocate_trace_programs_on_subdevice(hal, trace_nodes, sub_device_id);
    }
}

void SimpleTraceAllocator::allocate_trace_programs_on_subdevice(
    const Hal& hal, std::vector<TraceNode*>& trace_nodes, SubDeviceId sub_device_id) {
    uint32_t expected_workers_completed = 0;
    uint32_t programmable_core_count = hal.get_programmable_core_type_count();
    // For core types where the binary goes to a fixed L1 address (not stored in the config buffer),
    // track the last trace index that used each core type so we can sync before overwriting.
    std::vector<std::optional<uint32_t>> last_fixed_addr_sync_idx(programmable_core_count);
    bool first_program_dispatched = false;
    std::optional<uint32_t> last_stall_idx;
    std::deque<size_t> subdevice_launch_window;

    for (size_t i = 0; i < trace_nodes.size(); i++) {
        auto& node = *trace_nodes[i];
        if (node.sub_device_id != sub_device_id) {
            continue;
        }

        std::optional<uint32_t> nonbinary_sync_idx;
        std::optional<uint32_t> binary_sync_idx;
        // Reinitialize dispatch_metadata. TraceNodes may be shared across device ranges in a mesh
        // trace, so the caller (record_end) relies on this reinitialization to clear stale values
        // from a previous device range's processing.
        node.dispatch_metadata = TraceDispatchMetadata{};
        node.dispatch_metadata.binary_kernel_config_addrs.resize(programmable_core_count);
        node.dispatch_metadata.nonbinary_kernel_config_addrs.resize(programmable_core_count);

        bool all_binaries_cached = true;

        for (uint32_t index = 0; index < programmable_core_count; index++) {
            auto core_type = hal.get_programmable_core_type(index);
            if (!hal.has_programmable_core_type(core_type) || core_type == HalProgrammableCoreType::IDLE_ETH) {
                continue;
            }
            ProgramConfig& program_config = node.program->get_program_config(index);
            bool binary_in_config = hal.get_core_kernel_stored_in_config_buffer(core_type);

            // Only Tensix has a dedicated binary write offset in the dispatcher, so only Tensix
            // can place its binary at a separately-allocated address. For other core types the
            // dispatcher writes the binary at a fixed offset from the non-binary base, so the
            // entire config (non-binary + binary) must be allocated as one contiguous region.
            bool has_separate_binary_offset = (core_type == HalProgrammableCoreType::TENSIX);

            uint32_t non_binary_size;
            if (has_separate_binary_offset && binary_in_config) {
                non_binary_size = program_config.kernel_text_offset;
            } else {
                non_binary_size = node.program->get_program_config_sizes()[index];
            }
            uint32_t binary_size = program_config.kernel_text_size;
            auto& allocator = region_allocators_[index];

            uint64_t pid = node.program->get_id();
            auto [rta_sync_idx, rta_addr] = allocator.allocate_region(non_binary_size, i, ExtraData::kNonBinary, pid);

            nonbinary_sync_idx = merge_syncs(nonbinary_sync_idx, rta_sync_idx);

            uint32_t binary_addr = 0;

            if (has_separate_binary_offset && binary_in_config && binary_size > 0) {
                // Binary is stored in the config buffer and the dispatcher has a dedicated write
                // offset for this core type; allocate separately so it can be cached across
                // invocations of the same program.
                if (auto mem_addr = allocator.get_region(ExtraData::kBinary, pid)) {
                    binary_addr = *mem_addr;
                    allocator.update_region_trace_idx(*mem_addr, i);
                } else {
                    all_binaries_cached = false;
                    auto res = allocator.allocate_region(binary_size, i, ExtraData::kBinary, pid);
                    if (!res.second.has_value()) {
                        // Clear the allocator and try again. Should succeed unless the total size
                        // of the program is larger than the config buffer.
                        allocator.reset_allocator();
                        std::tie(rta_sync_idx, rta_addr) =
                            allocator.allocate_region(non_binary_size, i, ExtraData::kNonBinary, pid);
                        res = allocator.allocate_region(binary_size, i, ExtraData::kBinary, pid);
                        TT_ASSERT(res.second.has_value(), "Failed to allocate binary region");
                        TT_ASSERT(
                            !subdevice_launch_window.empty(),
                            "Failed to allocate binary region on first program on sub-device");
                        auto last_subdevice_idx = static_cast<uint32_t>(subdevice_launch_window.back());
                        binary_sync_idx = merge_syncs(binary_sync_idx, last_subdevice_idx);
                        nonbinary_sync_idx = merge_syncs(nonbinary_sync_idx, last_subdevice_idx);
                    } else {
                        binary_sync_idx = merge_syncs(res.first, binary_sync_idx);
                    }
                    binary_addr = *res.second;
                    allocator.add_region(ExtraData::kBinary, pid, binary_addr);
                }
            } else if (!binary_in_config && !node.program->get_kernel_groups(index).empty()) {
                // Binary goes to a fixed L1 address (not in the config buffer). Must sync with the
                // previous program that used this fixed address before overwriting it.
                all_binaries_cached = false;
                if (last_fixed_addr_sync_idx[index].has_value()) {
                    binary_sync_idx = merge_syncs(binary_sync_idx, last_fixed_addr_sync_idx[index]);
                }
                last_fixed_addr_sync_idx[index] = i;
            } else if (!has_separate_binary_offset && !node.program->get_kernel_groups(index).empty()) {
                // Binary is included in the non-binary allocation (no separate binary write offset
                // for this core type). The binary is always re-sent as part of the full config.
                all_binaries_cached = false;
            }

            TT_ASSERT(rta_addr.has_value(), "Failed to allocate non-binary region");
            node.dispatch_metadata.nonbinary_kernel_config_addrs[index] = {
                .addr = *rta_addr + ringbuffer_starts_[index]};
            node.dispatch_metadata.binary_kernel_config_addrs[index] = {
                .addr = binary_addr + ringbuffer_starts_[index]};
        }

        node.dispatch_metadata.send_binary = !all_binaries_cached;
        extra_data_[i].finished_sync_count = expected_workers_completed + node.num_workers;

        // Subtract 1 because we don't want to overwrite watcher data for the last program to complete executing.
        constexpr uint32_t max_queued_programs = dev_msgs::launch_msg_buffer_num_entries - 1;

        // Do adjustments to the sync index to ensure we don't overflow the worker launch message buffer. The launch
        // message buffer is written to after the binary.
        if (subdevice_launch_window.size() >= max_queued_programs) {
            binary_sync_idx = merge_syncs(binary_sync_idx, static_cast<uint32_t>(subdevice_launch_window.front()));
        }

        if (!first_program_dispatched) {
            // The first program to be dispatched should stall on 0, since there may be undetermined commands in the
            // ringbuffer before this we want to wait for. In particular in the mesh device case we can add go messages
            // for unused nodes before replaying the trace.
            node.dispatch_metadata.sync_count = 0;
            node.dispatch_metadata.stall_first = true;
            first_program_dispatched = true;
        }

        // Only one sync count can currently be specified, so pick the latest one.
        // nonbinary sync requires stall_first (before writing config data); binary-only sync uses
        // stall_before_program (after config, before binary/launch).
        bool needs_nonbinary_sync =
            nonbinary_sync_idx.has_value() && (!last_stall_idx.has_value() || *nonbinary_sync_idx > *last_stall_idx);
        bool needs_binary_sync =
            binary_sync_idx.has_value() && (!last_stall_idx.has_value() || *binary_sync_idx > *last_stall_idx);
        if (needs_nonbinary_sync || needs_binary_sync) {
            uint32_t combined_sync_idx = *merge_syncs(nonbinary_sync_idx, binary_sync_idx);
            node.dispatch_metadata.sync_count = extra_data_[combined_sync_idx].finished_sync_count;
            if (needs_nonbinary_sync) {
                node.dispatch_metadata.stall_first = true;
            } else {
                node.dispatch_metadata.stall_before_program = true;
            }
            last_stall_idx = combined_sync_idx;
        }
        expected_workers_completed += node.num_workers;
        subdevice_launch_window.push_back(i);
        if (subdevice_launch_window.size() > max_queued_programs) {
            subdevice_launch_window.pop_front();
        }
    }
}

void dump_trace_allocation_info(
    const Hal& hal,
    const std::vector<SimpleTraceAllocator::RingbufferConfig>& ringbuffer_configs,
    const std::vector<TraceNode*>& trace_nodes) {
    static std::atomic<uint32_t> dump_idx = 0;

    nlohmann::json trace_alloc_info;
    trace_alloc_info["schema_version"] = 1;
    trace_alloc_info["ringbuffer_configs"] = nlohmann::json::array();
    trace_alloc_info["core_types"] = nlohmann::json::array();
    trace_alloc_info["trace_nodes"] = nlohmann::json::array();
    trace_alloc_info["results"] = nlohmann::json::array();

    for (const auto& ringbuffer_config : ringbuffer_configs) {
        trace_alloc_info["ringbuffer_configs"].push_back(
            {{"start", ringbuffer_config.start}, {"size", ringbuffer_config.size}});
    }

    for (uint32_t index = 0; index < hal.get_programmable_core_type_count(); ++index) {
        auto core_type = hal.get_programmable_core_type(index);
        bool binary_in_config = hal.get_core_kernel_stored_in_config_buffer(core_type);
        bool has_separate_binary_offset = core_type == HalProgrammableCoreType::TENSIX;
        bool skip = !hal.has_programmable_core_type(core_type) || core_type == HalProgrammableCoreType::IDLE_ETH;
        trace_alloc_info["core_types"].push_back(
            {{"index", index},
             {"core_type", core_type_name(core_type)},
             {"binary_in_config", binary_in_config},
             {"has_separate_binary_offset", has_separate_binary_offset},
             {"skip", skip}});
    }

    for (const auto* trace_node : trace_nodes) {
        auto& program = *trace_node->program;
        nlohmann::json trace_node_info{
            {"program_id", program.get_id()},
            {"sub_device_id", static_cast<uint32_t>(*trace_node->sub_device_id)},
            {"num_workers", trace_node->num_workers},
            {"per_core_type", nlohmann::json::array()}};
        nlohmann::json result_info{
            {"send_binary", trace_node->dispatch_metadata.send_binary},
            {"sync_count", trace_node->dispatch_metadata.sync_count},
            {"stall_first", static_cast<bool>(trace_node->dispatch_metadata.stall_first)},
            {"stall_before_program", static_cast<bool>(trace_node->dispatch_metadata.stall_before_program)},
            {"nonbinary_addrs", nlohmann::json::array()},
            {"binary_addrs", nlohmann::json::array()}};

        for (uint32_t index = 0; index < hal.get_programmable_core_type_count(); ++index) {
            auto core_type = hal.get_programmable_core_type(index);
            const auto& program_config = program.get_program_config(index);
            bool binary_in_config = hal.get_core_kernel_stored_in_config_buffer(core_type);
            bool has_separate_binary_offset = core_type == HalProgrammableCoreType::TENSIX;
            uint32_t nonbinary_size =
                (has_separate_binary_offset && binary_in_config) ? program_config.kernel_text_offset
                                                                 : program.get_program_config_sizes()[index];
            uint32_t binary_size = program_config.kernel_text_size;
            bool has_kernel_groups = !program.get_kernel_groups(index).empty();

            trace_node_info["per_core_type"].push_back(
                {{"index", index},
                 {"nonbinary_size", nonbinary_size},
                 {"binary_size", binary_size},
                 {"has_kernel_groups", has_kernel_groups}});

            uint32_t nonbinary_addr = 0;
            uint32_t binary_addr = 0;
            if (index < trace_node->dispatch_metadata.nonbinary_kernel_config_addrs.size()) {
                nonbinary_addr = trace_node->dispatch_metadata.nonbinary_kernel_config_addrs[index].addr;
            }
            if (index < trace_node->dispatch_metadata.binary_kernel_config_addrs.size()) {
                binary_addr = trace_node->dispatch_metadata.binary_kernel_config_addrs[index].addr;
            }
            result_info["nonbinary_addrs"].push_back(nonbinary_addr);
            result_info["binary_addrs"].push_back(binary_addr);
        }

        trace_alloc_info["trace_nodes"].push_back(std::move(trace_node_info));
        trace_alloc_info["results"].push_back(std::move(result_info));
    }

    const auto output_dir =
        std::filesystem::path(MetalContext::instance().rtoptions().get_logs_dir()) / "generated" / "trace_alloc";
    std::error_code error_code;
    std::filesystem::create_directories(output_dir, error_code);
    if (error_code) {
        log_warning(
            tt::LogMetal,
            "Failed to create trace allocation dump directory '{}': {}",
            output_dir.string(),
            error_code.message());
        return;
    }

    const auto output_path =
        output_dir / ("trace_alloc_" + std::to_string(dump_idx.fetch_add(1, std::memory_order_relaxed)) + ".json");
    std::ofstream output_file(output_path);
    if (!output_file.is_open()) {
        log_warning(tt::LogMetal, "Failed to open trace allocation dump file '{}'", output_path.string());
        return;
    }
    output_file << trace_alloc_info.dump(2);
}

}  // namespace tt::tt_metal
