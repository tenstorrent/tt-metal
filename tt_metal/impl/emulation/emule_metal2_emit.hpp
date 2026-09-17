// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
//
// Metal-2.0 named-args / binding-handle snapshot + JIT-wrapper namespace emission.
// Moved out of emulated_program_runner.cpp. The emitted text must stay
// text-equivalent to genfiles.cpp's write_kernel_{args,bindings}_generated_header.

#include <cstdint>
#include <map>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "impl/kernels/kernel.hpp"  // Kernel, SemScope, SemaphoreBindingHandle, SemBindingEntry, emit_semaphore_binding_tokens
#include "emule_program_descriptor.hpp"  // tt_emule::Bindings

namespace tt::tt_metal::emule {

// Captures a Metal 2.0 kernel's named bindings. Drives both the JIT wrapper's
// namespace emission (see emit_metal2_namespaces) and the JIT cache key
// (see cache_key_suffix). Empty for legacy kernels.
struct Metal2BindingsSnapshot {
    // TA bindings are kept in insertion order (matches genfiles.cpp's vector);
    // their CRTA position drives the get_common_vararg offset.
    struct TaEntry {
        std::string name;
        uint32_t cta_offset;
        uint32_t addr_crta_offset;
    };
    // Scratchpad bindings, in insertion order (matches genfiles.cpp's vector).
    struct ScratchEntry {
        std::string name;
        uint32_t size_bytes;
        uint32_t addr_crta_word;
    };

    bool is_metal2 = false;
    std::vector<std::string> runtime_arg_names;
    std::vector<std::string> common_runtime_arg_names;
    std::map<std::string, uint32_t> dfb_accessors;
    std::map<std::string, bool> dfb_accessor_is_relay;
    std::map<std::string, uint8_t> dfb_accessor_prefetcher_pipe_id;
    std::map<std::string, SemaphoreBindingHandle> sem_accessors;
    std::vector<TaEntry> ta_accessors;
    std::vector<ScratchEntry> scratch_accessors;

    // Distinguishes kernels that share source/CTAs/defines but bind different
    // IDs — without this they collide on cache key and the second silently
    // reuses the first's .so.
    std::string cache_key_suffix() const {
        std::string s;
        for (const auto& [name, id] : dfb_accessors) {
            s += ":dfb:" + name + "=" + std::to_string(id);
            if (dfb_accessor_is_relay.contains(name) && dfb_accessor_is_relay.at(name)) {
                s += ":relay";
                if (dfb_accessor_prefetcher_pipe_id.contains(name) &&
                    dfb_accessor_prefetcher_pipe_id.at(name) != 0xFF) {
                    s += ":prefetcher_pipe" + std::to_string(dfb_accessor_prefetcher_pipe_id.at(name));
                }
            }
        }
        for (const auto& [name, h] : sem_accessors) {
            s += ":sem:" + name + "=" + std::to_string(h.id) + "@" + std::to_string(static_cast<int>(h.scope));
        }
        for (const auto& ta : ta_accessors) {
            s += ":ta:" + ta.name + "=" + std::to_string(ta.cta_offset) + "," + std::to_string(ta.addr_crta_offset);
        }
        for (const auto& sp : scratch_accessors) {
            s += ":scratch:" + sp.name + "=" + std::to_string(sp.size_bytes) + "," + std::to_string(sp.addr_crta_word);
        }
        for (const auto& name : runtime_arg_names) {
            s += ":rta:" + name;
        }
        for (const auto& name : common_runtime_arg_names) {
            s += ":crta:" + name;
        }
        return s;
    }
};

// Build the Metal-2.0 binding snapshot from the marshalled POD bindings (no private Kernel read).
Metal2BindingsSnapshot snapshot_from_bindings(const tt_emule::Bindings& b);

// Emits args::/dfb::/sem::/tensor:: namespaces into the JIT wrapper, replacing
// kernel_args_generated.h + kernel_bindings_generated.h that upstream's JIT
// build produces. Must stay text-equivalent to genfiles.cpp's
// write_kernel_{args,bindings}_generated_header.
void emit_metal2_namespaces(
    std::ostream& f,
    const Metal2BindingsSnapshot& s,
    const std::unordered_map<std::string, uint32_t>& named_compile_args);

}  // namespace tt::tt_metal::emule
