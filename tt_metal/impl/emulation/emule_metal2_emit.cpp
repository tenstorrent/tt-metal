// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "emule_metal2_emit.hpp"

#include <algorithm>  // std::sort

#include <tt_stl/assert.hpp>  // TT_FATAL

namespace tt::tt_metal::emule {

Metal2BindingsSnapshot build_metal2_snapshot(const tt::tt_metal::Kernel& kernel) {
    Metal2BindingsSnapshot s;
    s.is_metal2 = kernel.is_metal2_kernel();
    s.runtime_arg_names = kernel.get_runtime_arg_names();
    s.common_runtime_arg_names = kernel.get_common_runtime_arg_names();
    kernel.process_dataflow_buffer_binding_handles(
        [&s](const std::string& name, uint16_t id, bool is_relay, uint8_t prefetcher_pipe_id) {
            s.dfb_accessors[name] = id;
            s.dfb_accessor_is_relay[name] = is_relay;
            s.dfb_accessor_prefetcher_pipe_id[name] = prefetcher_pipe_id;
        });
    kernel.process_semaphore_binding_handles(
        [&s](const std::string& name, uint16_t id, SemScope scope, uint32_t total_binder_harts) {
            s.sem_accessors[name] = {id, scope, total_binder_harts};
        });
    kernel.process_tensor_binding_handles(
        // Match the genfiles.cpp pattern: drop num_runtime_field_crta_words. Emule's
        // snapshot doesn't yet model per-binding runtime CRTA words, and the
        // downstream `named_crta_words` math in emit_metal2_namespaces still assumes
        // 1 word per binding — so a dynamic-shape kernel would silently get its
        // CRTAs decoded at the wrong offsets. Static-shape kernels pass
        // num_rt_words == 0 and are unaffected. Fail loudly on dynamic-shape until
        // snapshot + cache key + get_common_vararg offset math are wired up to
        // consume the per-binding count.
        [&s](const std::string& name, uint32_t cta_off, uint32_t addr_crta_off, uint32_t num_rt_words) {
            TT_FATAL(
                num_rt_words == 0,
                "Emule does not yet support dynamic-shape Metal 2.0 tensor bindings "
                "(binding '{}' has num_runtime_field_crta_words={}). Wire the per-"
                "binding word count through Metal2BindingsSnapshot::TaEntry, the "
                "cache key, and emit_metal2_namespaces' get_common_vararg base "
                "before enabling this path.",
                name,
                num_rt_words);
            s.ta_accessors.push_back({name, cta_off, addr_crta_off});
        });
    kernel.process_scratchpad_binding_handles(
        [&s](const std::string& name, uint32_t size_bytes, uint32_t addr_crta_word) {
            s.scratch_accessors.push_back({name, size_bytes, addr_crta_word});
        });
    return s;
}

void emit_metal2_namespaces(
    std::ostream& f,
    const Metal2BindingsSnapshot& s,
    const std::unordered_map<std::string, uint32_t>& named_compile_args) {
    const bool has_args =
        !s.runtime_arg_names.empty() || !s.common_runtime_arg_names.empty() || !named_compile_args.empty();
    std::vector<tt::tt_metal::SemBindingEntry> sem_entries;
    sem_entries.reserve(s.sem_accessors.size());
    for (const auto& [name, h] : s.sem_accessors) {
        sem_entries.push_back({name, h.id, h.scope});
    }
    if (has_args) {
        f << "#include \"experimental/kernel_args.h\"\n";
    }
    if (!s.dfb_accessors.empty()) {
        f << "#include \"api/dataflow/dataflow_buffer.h\"\n";
    }
    if (!s.ta_accessors.empty()) {
        f << "#include \"api/tensor/tensor_binding_token.h\"\n";
    }
    if (!s.scratch_accessors.empty()) {
        f << "#include \"api/scratchpad.h\"\n";
    }

    if (has_args) {
        f << "namespace args {\n";
        uint32_t rta_offset = 0;
        for (const auto& name : s.runtime_arg_names) {
            f << "constexpr ::experimental::RtaArg<uint32_t> " << name << "{" << rta_offset << "};\n";
            rta_offset += sizeof(uint32_t);
        }
        uint32_t crta_offset = 0;
        for (const auto& name : s.common_runtime_arg_names) {
            f << "constexpr ::experimental::CrtaArg<uint32_t> " << name << "{" << crta_offset << "};\n";
            crta_offset += sizeof(uint32_t);
        }
        // Sort CTAs for deterministic wrapper output.
        std::vector<std::pair<std::string, uint32_t>> cta_entries(named_compile_args.begin(), named_compile_args.end());
        std::sort(cta_entries.begin(), cta_entries.end());
        for (const auto& [name, value] : cta_entries) {
            // Dotted keys cannot name flat args:: constants.
            // Blaze constants are emitted separately from named_ct_arg_namespaces.
            if (name.find('.') != std::string::npos) {
                continue;
            }
            f << "constexpr ::experimental::CtaVal<uint32_t> " << name << "{" << value << "u};\n";
        }
        f << "}  // namespace args\n";
    }
    if (!s.dfb_accessors.empty()) {
        f << "namespace dfb {\n";
        for (const auto& [name, id] : s.dfb_accessors) {
            const bool is_relay = s.dfb_accessor_is_relay.contains(name) && s.dfb_accessor_is_relay.at(name);
            if (is_relay) {
                const uint8_t prefetcher_pipe_id = s.dfb_accessor_prefetcher_pipe_id.contains(name)
                                                       ? s.dfb_accessor_prefetcher_pipe_id.at(name)
                                                       : 0xFF;
                f << "constexpr RelayDFBBindingToken " << name << "{" << id;
                if (prefetcher_pipe_id != 0xFF) {
                    f << ", " << static_cast<uint32_t>(prefetcher_pipe_id);
                }
                f << "};\n";
            } else {
                f << "constexpr DFBBindingToken " << name << "{" << id << "};\n";
            }
        }
        f << "}  // namespace dfb\n";
    }
    if (!sem_entries.empty()) {
        tt::tt_metal::emit_semaphore_binding_tokens(f, sem_entries);
    }
    if (!s.ta_accessors.empty()) {
        f << "namespace tensor {\n";
        for (const auto& ta : s.ta_accessors) {
            f << "using " << ta.name << "_t = ::tensor_accessor::TensorBindingToken<" << ta.cta_offset << "u, "
              << ta.addr_crta_offset << "u>;\n";
            f << "constexpr " << ta.name << "_t " << ta.name << "{};\n";
        }
        f << "}  // namespace tensor\n";
    }
    if (!s.scratch_accessors.empty()) {
        f << "namespace scratch {\n";
        for (const auto& sp : s.scratch_accessors) {
            f << "constexpr ScratchpadBindingToken " << sp.name << "{" << sp.addr_crta_word << "u, " << sp.size_bytes
              << "u};\n";
        }
        f << "}  // namespace scratch\n";
    }

    // Vararg helpers — always emitted for Metal 2.0 kernels (mirrors
    // genfiles.cpp). The CRTA buffer layout is [user-named CRTAs,
    // TensorBinding addresses, scratchpad addresses, varargs], so
    // get_common_vararg's base skips past the named CRTAs, the binding
    // section, and the scratchpad section.
    if (s.is_metal2) {
        const uint32_t named_rta_words = static_cast<uint32_t>(s.runtime_arg_names.size());
        const uint32_t named_crta_words = static_cast<uint32_t>(
            s.common_runtime_arg_names.size() + s.ta_accessors.size() + s.scratch_accessors.size());
        f << "FORCE_INLINE uint32_t get_vararg(uint32_t idx) { "
          << "return get_arg_val<uint32_t>(" << named_rta_words << " + idx); }\n";
        f << "FORCE_INLINE uint32_t get_common_vararg(uint32_t idx) { "
          << "return get_common_arg_val<uint32_t>(" << named_crta_words << " + idx); }\n";
    }
}

}  // namespace tt::tt_metal::emule
