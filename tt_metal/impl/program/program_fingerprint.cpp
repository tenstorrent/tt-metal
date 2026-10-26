// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/program_fingerprint.hpp>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iterator>
#include <map>
#include <optional>
#include <ranges>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/program.hpp>

#include "impl/buffers/circular_buffer.hpp"
#include "impl/buffers/semaphore.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/dataflow_buffer/dataflow_buffer_impl.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"

namespace tt::tt_metal::experimental::program_fingerprint {

namespace {

std::string cores_to_string(const CoreRangeSet& cores) {
    return fmt::format(
        "{}",
        fmt::join(cores.ranges() | std::views::transform([](const CoreRange& range) { return range.str(); }), ","));
}

// "key=value" per entry of an ordered map, so the result does not depend on insertion order.
template <typename Map>
auto key_values(const Map& map) {
    return map |
           std::views::transform([](const auto& entry) { return fmt::format("{}={}", entry.first, entry.second); });
}

std::string describe_kernel(const Kernel& kernel) {
    const auto& source = kernel.kernel_source();
    const std::string source_id = source.source_type_ == KernelSource::FILE_PATH
                                      ? source.source_
                                      : fmt::format("inline#{:x}", std::hash<std::string>{}(source.source_));

    std::vector<uint32_t> compile_time_args;
    kernel.process_compile_time_args([&](const std::vector<uint32_t>& values) { compile_time_args = values; });

    std::map<std::string, uint32_t> named_compile_time_args;
    kernel.process_named_compile_time_args([&](const std::unordered_map<std::string, uint32_t>& named_args) {
        named_compile_time_args.insert(named_args.begin(), named_args.end());
    });

    std::map<std::string, std::string> defines;
    // ProgramImpl::compile adds ALIGN_LOCAL_CBS_TO_REMOTE_CBS from the circular buffers, which are described on their
    // own; a program that was never compiled does not have it.
    kernel.process_defines([&](const std::string& define, const std::string& value) {
        if (define != "ALIGN_LOCAL_CBS_TO_REMOTE_CBS") {
            defines[define] = value;
        }
    });

    return fmt::format(
        "kernel {} source={} cores={} processor_class={} compile_args=[{}] named_args=[{}] defines=[{}] opt={}/{} "
        "settings_hash={:x}",
        kernel.name(),
        source_id,
        cores_to_string(kernel.core_range_set()),
        static_cast<int>(kernel.get_kernel_processor_class()),
        fmt::join(compile_time_args, ","),
        fmt::join(key_values(named_compile_time_args), ","),
        fmt::join(key_values(defines), ","),
        kernel.get_compiler_opt_level(),
        kernel.get_linker_opt_level(),
        kernel.compute_hash());
}

std::string describe_tile(const std::optional<Tile>& tile) {
    if (!tile.has_value()) {
        return "none";
    }
    const auto face_shape = tile->get_face_shape();
    return fmt::format(
        "{}x{} face={}x{} faces={} partial_face={} narrow={} transpose={}/{}",
        tile->get_height(),
        tile->get_width(),
        face_shape[0],
        face_shape[1],
        tile->get_num_faces(),
        tile->get_partial_face(),
        tile->get_narrow_tile(),
        tile->get_transpose_within_face(),
        tile->get_transpose_of_faces());
}

std::string describe_circular_buffer(const CircularBufferImpl& cb) {
    std::vector<uint8_t> indices(cb.buffer_indices().begin(), cb.buffer_indices().end());
    std::ranges::sort(indices);
    const auto per_index =
        indices | std::views::transform([&cb](uint8_t index) {
            const auto& tile = cb.tile(index);
            return fmt::format(
                "{}:page={},format={},tile={}",
                index,
                cb.page_size(index),
                static_cast<int>(cb.data_format(index)),
                describe_tile(tile));
        });
    return fmt::format(
        "circular_buffer cores={} size={} globally_allocated={} global_cb={} indices=[{}]",
        cores_to_string(cb.core_ranges()),
        cb.size(),
        cb.globally_allocated(),
        cb.is_global_circular_buffer(),
        fmt::join(per_index, ";"));
}

// What the factory configured. Not the groups, stride, remapper or addresses the program fills in when it is
// finalized, which a freshly built program does not have yet.
std::string describe_dataflow_buffer(const experimental::dfb::detail::DataflowBufferImpl& dfb) {
    const auto& config = dfb.config;
    return fmt::format(
        "dataflow_buffer id={} cores={} entry_size={} num_entries={} producer=mask:{:x},n:{},pattern:{},block:{} "
        "consumer=mask:{:x},n:{},pattern:{},block:{} implicit_sync={}/{} format={} tile={} tensix_scope={} "
        "borrows_memory={} relay={} alias_of={} aliases=[{}]",
        dfb.id,
        cores_to_string(dfb.core_ranges),
        config.entry_size,
        config.num_entries,
        config.producer_risc_mask,
        config.num_producers,
        static_cast<int>(config.pap),
        config.producer_block_size,
        config.consumer_risc_mask,
        config.num_consumers,
        static_cast<int>(config.cap),
        config.consumer_block_size,
        config.enable_producer_implicit_sync,
        config.enable_consumer_implicit_sync,
        static_cast<int>(config.data_format),
        describe_tile(config.tile),
        config.tensix_scope.has_value() ? std::to_string(static_cast<int>(*config.tensix_scope)) : "none",
        config.borrows_memory,
        config.is_relay,
        dfb.alias_primary_id.has_value() ? std::to_string(*dfb.alias_primary_id) : "none",
        fmt::join(dfb.alias_secondary_ids, ","));
}

}  // namespace

std::vector<std::string> describe(Program& program) {
    auto& impl = program.impl();
    std::vector<std::string> lines;

    const auto& hal = MetalContext::instance(impl.get_context_id()).hal();
    for (uint32_t index = 0; index < hal.get_programmable_core_type_count(); index++) {
        // Handles are assigned in creation order, so sorting by handle is stable across rebuilds.
        const auto& kernels = impl.get_kernels(index);
        using HandleAndKernel = std::pair<KernelHandle, const Kernel*>;
        std::vector<HandleAndKernel> sorted;
        sorted.reserve(kernels.size());
        std::ranges::transform(kernels, std::back_inserter(sorted), [](const auto& entry) {
            return HandleAndKernel{entry.first, entry.second.get()};
        });
        std::ranges::sort(sorted, {}, &HandleAndKernel::first);
        for (const auto& [handle, kernel] : sorted) {
            lines.push_back(fmt::format("core_type {} handle {} {}", index, handle, describe_kernel(*kernel)));
        }
    }

    for (const auto& cb : impl.circular_buffers()) {
        lines.push_back(describe_circular_buffer(*cb));
    }

    for (const auto& dfb : impl.dataflow_buffers()) {
        lines.push_back(describe_dataflow_buffer(*dfb));
    }

    for (const auto& semaphore : impl.semaphores()) {
        lines.push_back(fmt::format(
            "semaphore id={} cores={} core_type={} initial_value={}",
            semaphore.id(),
            cores_to_string(semaphore.core_range_set()),
            static_cast<int>(semaphore.core_type()),
            semaphore.initial_value()));
    }
    return lines;
}

std::vector<std::string> describe(distributed::MeshWorkload& workload) {
    using RangeAndProgram = std::pair<distributed::MeshCoordinateRange, Program*>;
    std::vector<RangeAndProgram> programs;
    std::ranges::transform(workload.get_programs(), std::back_inserter(programs), [](auto& entry) {
        return RangeAndProgram{entry.first, &entry.second};
    });
    // std::less<>, not the default std::ranges::less: MeshCoordinateRange only defines operator<.
    std::ranges::sort(programs, std::less<>{}, &RangeAndProgram::first);

    std::vector<std::string> lines;
    for (const auto& [range, program] : programs) {
        std::ostringstream prefix;
        prefix << range;
        for (auto& line : describe(*program)) {
            lines.push_back(fmt::format("{} {}", prefix.str(), line));
        }
    }
    return lines;
}

}  // namespace tt::tt_metal::experimental::program_fingerprint
