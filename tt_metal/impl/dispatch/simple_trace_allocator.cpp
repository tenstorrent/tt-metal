// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "simple_trace_allocator.hpp"

#include "impl/context/metal_context.hpp"

namespace tt::tt_metal {

std::pair<std::optional<uint32_t>, std::optional<uint32_t>> SimpleTraceAllocator::RegionAllocator::allocate_region(
    uint32_t size, uint32_t trace_idx, uint32_t data_type) {
    std::optional<uint32_t> best_addr;
    float best_cost = 999999995904;
    std::optional<uint32_t> best_region_sync_idx;
    uint32_t addr = 0;
    auto outer_it = regions_.begin();
    if (size == 0) {
        return {std::nullopt, 0};
    }

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
                        cost += 10000000000;
                    } else {
                        cost += region.second.size * 1.0f / (*next_use_idx - trace_idx);
                    }
                }
                region_sync_idx = merge_syncs(region_sync_idx, region.second.trace_idx);
            }
        }
        if (!now_in_use) {
            if (region_sync_idx.has_value()) {
                constexpr uint32_t kDesiredWriteAhead = 4;
                constexpr uint32_t kStallBadness = 100000000;
                int region_idx_diff = trace_idx - *region_sync_idx;
                if (region_idx_diff < kDesiredWriteAhead) {
                    // Stall badness is exponential
                    cost += kStallBadness * (1 << (kDesiredWriteAhead - region_idx_diff));
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

    if (!best_addr.has_value()) {
        return {std::nullopt, best_addr};
    }

    auto it = regions_.begin();
    while (it != regions_.end()) {
        if (intersects(*best_addr, size, it->first, it->second.size)) {
            program_ids_memory_map_.erase((*trace_nodes_)[it->second.trace_idx].program->get_id());
            it = regions_.erase(it);
        } else {
            ++it;
        }
    }
    regions_[*best_addr] = {trace_idx, data_type, size};
    return {best_region_sync_idx, best_addr};
}

void SimpleTraceAllocator::allocate_trace_programs(IDevice* device, std::vector<TraceNode>& trace_nodes) {
    const auto& hal = MetalContext::instance().hal();
    worker_region_allocator_.set_trace_nodes(&trace_nodes);
    active_eth_region_allocator_.set_trace_nodes(&trace_nodes);
    std::map<uint64_t, uint32_t> program_ids_use_map;
    extra_data_.resize(trace_nodes.size());

    std::set<SubDeviceId> sub_device_ids;
    for (int i = trace_nodes.size() - 1; i >= 0; i--) {
        auto& node = trace_nodes[i];
        auto it = program_ids_use_map.find(node.program->get_id());
        if (it != program_ids_use_map.end()) {
            // Binary is reused, but the nonbinary is not.
            extra_data_[i].next_use_idx[ExtraData::kBinary] = it->second;
        }
        program_ids_use_map[node.program->get_id()] = i;
        sub_device_ids.insert(node.sub_device_id);
    }
    for (auto& sub_device_id : sub_device_ids) {
        worker_region_allocator_.reset_allocator();
        active_eth_region_allocator_.reset_allocator();
        allocate_trace_programs_on_subdevice(device, trace_nodes, sub_device_id);
    }
}

void SimpleTraceAllocator::allocate_trace_programs_on_subdevice(
    IDevice* device, std::vector<TraceNode>& trace_nodes, SubDeviceId sub_device_id) {
    const auto& hal = MetalContext::instance().hal();

    uint32_t expected_workers_completed = 0;
    std::optional<uint32_t> last_active_eth_sync_idx;
    for (size_t i = 0; i < trace_nodes.size(); i++) {
        auto& node = trace_nodes[i];
        auto& program = *node.program;
        if (node.sub_device_id != sub_device_id) {
            continue;
        }
        auto sub_device_id = node.sub_device_id;

        std::optional<uint32_t> sync_idx;
        uint32_t programmable_core_count_ = hal.get_programmable_core_type_count();
        node.dispatch_metadata.binary_kernel_config_addrs.resize(programmable_core_count_);
        node.dispatch_metadata.nonbinary_kernel_config_addrs.resize(programmable_core_count_);

        for (auto& core_type : {HalProgrammableCoreType::TENSIX, HalProgrammableCoreType::ACTIVE_ETH}) {
            uint32_t index = hal.get_programmable_core_type_index(core_type);
            ProgramConfig& program_config = node.program->get_program_config(index);
            uint32_t non_binary_size = core_type == HalProgrammableCoreType::TENSIX
                                           ? program_config.kernel_text_offset
                                           : node.program->get_program_config_sizes()[index];
            uint32_t binary_size = program_config.kernel_text_size;
            auto& allocator =
                core_type == HalProgrammableCoreType::TENSIX ? worker_region_allocator_ : active_eth_region_allocator_;

            auto [rta_sync_idx, rta_addr] = allocator.allocate_region(non_binary_size, i, ExtraData::kNonBinary);

            sync_idx = merge_syncs(sync_idx, rta_sync_idx);

            uint32_t binary_addr = 0;

            // Only tensix binaries are stored in the kernel config buffer. Active ethernet binaries have a fixed
            // address.
            if (core_type == HalProgrammableCoreType::TENSIX) {
                if (auto mem_addr = allocator.get_region(node.program->get_id())) {
                    binary_addr = *mem_addr;
                    node.dispatch_metadata.send_binary = false;
                    allocator.update_region_trace_idx(*mem_addr, i);
                } else {
                    auto res = allocator.allocate_region(binary_size, i, ExtraData::kBinary);
                    if (!res.second.has_value()) {
                        // Clear the allocator and try again. Should succeed unless the total size of the program is
                        // larger than the config buffer.
                        allocator.reset_allocator();
                        std::tie(rta_sync_idx, rta_addr) =
                            allocator.allocate_region(non_binary_size, i, ExtraData::kNonBinary);
                        res = allocator.allocate_region(binary_size, i, ExtraData::kBinary);
                        TT_ASSERT(res.second.has_value(), "Failed to allocate binary region");
                        TT_ASSERT(i > 0, "Failed to allocate binary region on first program");
                        sync_idx = merge_syncs(sync_idx, i - 1);
                    } else {
                        sync_idx = merge_syncs(res.first, sync_idx);
                    }
                    binary_addr = *res.second;
                    allocator.add_region(node.program->get_id(), binary_addr);
                }
            }
            TT_ASSERT(rta_addr.has_value(), "Failed to allocate non-binary region");
            auto& ringbuffer_start =
                core_type == HalProgrammableCoreType::TENSIX ? worker_ringbuffer_start_ : active_eth_ringbuffer_start_;
            node.dispatch_metadata.nonbinary_kernel_config_addrs[index] = {.addr = *rta_addr + ringbuffer_start};
            node.dispatch_metadata.binary_kernel_config_addrs[index] = {.addr = binary_addr + ringbuffer_start};
        }

        bool has_active_eth_kernel = program.runs_on_noc_unicast_only_cores();
        uint32_t num_workers = 0;
        if (program.runs_on_noc_multicast_only_cores()) {
            num_workers += device->num_worker_cores(HalProgrammableCoreType::TENSIX, sub_device_id);
        }
        if (program.runs_on_noc_unicast_only_cores()) {
            num_workers += device->num_worker_cores(HalProgrammableCoreType::ACTIVE_ETH, sub_device_id);
        }
        extra_data_[i].finished_sync_count = expected_workers_completed + num_workers;

        // Subtract 1 because we don't want to overwrite watcher data for the last program to complete executing.
        constexpr uint32_t max_queued_programs = launch_msg_buffer_num_entries - 1;

        // Do adjustments to the sync index to ensure we don't overflow the launch message buffer.
        int final_sync_idx = static_cast<int>(i) - max_queued_programs;
        if (sync_idx.has_value()) {
            final_sync_idx = std::max(final_sync_idx, static_cast<int>(sync_idx.value()));
        }
        // Do adjustments to the sync index to ensure we don't overwrite the previous ethernet binary (since ethernet
        // doesn't use the ringbuffer).
        if (has_active_eth_kernel) {
            if (last_active_eth_sync_idx.has_value()) {
                final_sync_idx = std::max(final_sync_idx, static_cast<int>(last_active_eth_sync_idx.value()));
            }
            last_active_eth_sync_idx = i;
            node.dispatch_metadata.send_binary = true;
        }

        if (final_sync_idx >= 0) {
            node.dispatch_metadata.sync_count = extra_data_[final_sync_idx].finished_sync_count;
            node.dispatch_metadata.stall_first = true;
        }
        expected_workers_completed += num_workers;
    }
}

}  // namespace tt::tt_metal
