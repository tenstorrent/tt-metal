// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/metal2_host_api/program_spec/construction/resource/resource.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_distribution_spec.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>  // fmt::formatter<tt::DataFormat> for TT_FATAL messages
#include <hostdevcommon/tensor_accessor/arg_config.hpp>
#include <tt_stl/assert.hpp>

#include "impl/metal2_host_api/helpers.hpp"
#include "impl/metal2_host_api/llk_metadata.hpp"
#include "impl/context/metal_context.hpp"

namespace tt::tt_metal::experimental {

namespace {

std::optional<LLKMetadata> LLKMetadataFromScratchpad(const ScratchpadSpec& spec) {
    if (!spec.data_format_metadata.has_value()) {
        TT_FATAL(
            !spec.tile_format_metadata.has_value(),
            "Scratchpad '{}' need to have a configured data_format_metadata for it's tile_format_metadata to be "
            "respected",
            spec.unique_id);
        return std::nullopt;
    }
    const Tile tile = spec.tile_format_metadata.value_or(Tile{});
    return LLKMetadata{.format = *spec.data_format_metadata, .tile = tile};
}

}  // namespace

ScratchpadBindingsForKernel ResolveScratchpadBindingsForKernel(
    const KernelSpec& kernel,
    const std::unordered_map<ScratchpadSpecName, const ScratchpadSpec*>& scratchpad_by_name,
    size_t scratchpad_base_crta_word) {
    ScratchpadBindingsForKernel out;
    out.handles.reserve(kernel.scratchpad_bindings.size());

    size_t crta_word_index = scratchpad_base_crta_word;
    for (const auto& binding : kernel.scratchpad_bindings) {
        const ScratchpadSpec* scratchpad_spec = scratchpad_by_name.at(binding.scratchpad_spec_name);

        ScratchpadBindingHandle handle;
        handle.accessor_name = binding.accessor_name;
        handle.size_bytes = scratchpad_spec->size_per_node;
        handle.addr_crta_word = static_cast<uint32_t>(crta_word_index);
        handle.llk_metadata = LLKMetadataFromScratchpad(*scratchpad_spec);
        // handle.allocated_address stays 0 until allocate_scratchpads runs.
        out.handles.push_back(std::move(handle));
        crta_word_index += 1;  // one address word per scratchpad binding
    }

    out.section_words = static_cast<uint32_t>(kernel.scratchpad_bindings.size());
    return out;
}

}  // namespace tt::tt_metal::experimental
