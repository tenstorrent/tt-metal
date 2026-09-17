// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "emule_metal2_emit.hpp"

#include <algorithm>  // std::sort

namespace tt::tt_metal::emule {

Metal2BindingsSnapshot snapshot_from_bindings(const tt_emule::Bindings& b) {
    Metal2BindingsSnapshot s;
    s.is_metal2 = b.is_metal2;
    s.runtime_arg_names = b.rta_names;
    s.common_runtime_arg_names = b.crta_names;
    for (const auto& d : b.dfb) {
        s.dfb_accessors[d.name] = d.dfb_id;
        s.dfb_accessor_is_relay[d.name] = d.is_relay;
        s.dfb_accessor_prefetcher_pipe_id[d.name] = d.prefetcher_pipe;
    }
    for (const auto& sm : b.sem) {
        s.sem_accessors[sm.name] = {
            sm.sem_id, static_cast<SemScope>(static_cast<uint8_t>(sm.scope)), sm.total_binder_harts};
    }
    for (const auto& t : b.tensor) {
        s.ta_accessors.push_back({t.name, t.cta_offset, t.addr_crta_offset});
    }
    for (const auto& sp : b.scratch) {
        s.scratch_accessors.push_back({sp.name, sp.size_bytes, sp.addr_crta_word});
    }
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
