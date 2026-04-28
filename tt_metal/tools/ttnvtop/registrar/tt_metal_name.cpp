// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Bridge implementation: turn a tt-metal ProgramImpl into a short
// human-readable name for the ttnvtop registry. Kept in its own
// translation unit (compiled into libtt_metal via
// tt_metal/impl/sources.cmake) so that the standalone registrar
// library (`ttnvtop_register`) never takes a tt-metal dependency.

#include "tt_metal_name.hpp"

#include <filesystem>
#include <string>

#include "impl/program/program_impl.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/kernels/kernel_source.hpp"
#include "impl/context/metal_context.hpp"
#include "hal_types.hpp"

namespace ttnvtop {

std::string ttnvtop_program_name(const tt::tt_metal::detail::ProgramImpl& program) {
    // Walk every programmable core type and collect basenames (sans extension)
    // of all file-backed kernels in the program. Joined with '|' so the viewer
    // can show "compute|reader|writer" in one label.
    //
    // Order is deterministic: programmable_core_type index ascending, kernel
    // handle ascending within each type. Duplicates suppressed (large grids
    // often have the same kernel placed on many cores; we only want to
    // surface unique source files).
    //
    // Total length is capped at kRegistryNameMax-1 so the registry entry
    // doesn't get truncated mid-segment.

    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    const uint32_t n_types = hal.get_programmable_core_type_count();

    auto& mutable_program = const_cast<tt::tt_metal::detail::ProgramImpl&>(program);

    std::string out;
    auto append = [&out](const std::string& seg) {
        // Already present? skip (cheap: single occurrence check).
        std::string needle = out.empty() ? seg : "|" + seg;
        if (out.find(seg) != std::string::npos) {
            return;
        }
        // Cap at 95 chars so the writer's strncpy(name, src, 95) doesn't
        // chop a segment in half.
        if (out.size() + needle.size() > 95) {
            return;
        }
        out += needle;
    };

    for (uint32_t i = 0; i < n_types; ++i) {
        const auto& kernels = mutable_program.get_kernels(i);
        for (const auto& [handle, kernel] : kernels) {
            if (!kernel) {
                continue;
            }
            const auto& src = kernel->kernel_source();
            if (src.source_type_ != tt::tt_metal::KernelSource::SourceType::FILE_PATH) {
                continue;
            }
            append(src.path_.stem().string());
        }
    }

    if (out.empty()) {
        out = std::string("prog_") + std::to_string(static_cast<uint32_t>(program.get_runtime_id()));
    }
    return out;
}

}  // namespace ttnvtop
