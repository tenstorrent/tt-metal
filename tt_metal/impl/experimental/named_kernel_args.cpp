// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// EXPERIMENTAL: Named kernel-args — temporary, Blaze-only.

#include <tt-metalium/experimental/named_kernel_args.hpp>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt_stl/assert.hpp>

#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "jit_build/jit_build_settings.hpp"

namespace tt::tt_metal::experimental {

void process_named_args(Program& program, const KernelDescriptor& kernel_descriptor, uint32_t kernel_handle) {
    const auto& named_args = kernel_descriptor.named_args;

    // Set per-core runtime args: positional values followed by named values.
    // Build merged vectors per core, then set once.
    if (!named_args.named_per_core_runtime_args.empty() || !named_args.named_per_core_runtime_arg_arrays.empty()) {
        std::map<CoreCoord, std::vector<uint32_t>> core_to_args;
        for (const auto& [core, positional] : kernel_descriptor.runtime_args) {
            core_to_args[core] = positional;
        }
        for (const auto& arg : named_args.named_per_core_runtime_args) {
            for (const auto& [core, value] : arg.core_values) {
                core_to_args[core].push_back(value);
            }
        }
        for (const auto& arg : named_args.named_per_core_runtime_arg_arrays) {
            for (const auto& [core, values] : arg.core_values) {
                core_to_args[core].insert(core_to_args[core].end(), values.begin(), values.end());
            }
        }
        for (const auto& [core, merged] : core_to_args) {
            SetRuntimeArgs(program, kernel_handle, core, merged);
        }
    } else {
        for (const auto& [core_coord, core_runtime_args] : kernel_descriptor.runtime_args) {
            SetRuntimeArgs(program, kernel_handle, core_coord, core_runtime_args);
        }
    }

    // Set common runtime args: positional values followed by named scalars, then named arrays.
    if (!named_args.named_common_runtime_args.empty() || !named_args.named_common_runtime_arg_arrays.empty()) {
        std::vector<uint32_t> merged_common_rt_args;
        merged_common_rt_args.insert(
            merged_common_rt_args.end(),
            kernel_descriptor.common_runtime_args.begin(),
            kernel_descriptor.common_runtime_args.end());
        for (const auto& arg : named_args.named_common_runtime_args) {
            merged_common_rt_args.push_back(arg.value);
        }
        for (const auto& arg : named_args.named_common_runtime_arg_arrays) {
            merged_common_rt_args.insert(merged_common_rt_args.end(), arg.values.begin(), arg.values.end());
        }
        SetCommonRuntimeArgs(program, kernel_handle, merged_common_rt_args);
    } else {
        SetCommonRuntimeArgs(program, kernel_handle, kernel_descriptor.common_runtime_args);
    }

    // Build namespace maps for JIT header generation.
    // Names use "ns.field" convention -- split on '.' to produce namespace hierarchy.
    auto validate_identifier = [](const std::string& id, const std::string& context) {
        TT_FATAL(
            !id.empty() && (std::isalpha(id[0]) || id[0] == '_') &&
                std::all_of(id.begin(), id.end(), [](char c) { return std::isalnum(c) || c == '_'; }),
            "Named arg {}: '{}' is not a valid C++ identifier",
            context,
            id);
    };
    auto split_name = [&validate_identifier](const std::string& name) -> std::pair<std::string, std::string> {
        auto dot = name.find('.');
        if (dot == std::string::npos) {
            validate_identifier(name, "field");
            return {"", name};
        }
        auto ns = name.substr(0, dot);
        auto field = name.substr(dot + 1);
        validate_identifier(ns, "namespace");
        validate_identifier(field, "field");
        return {ns, field};
    };

    auto kernel = program.impl().get_kernel(kernel_handle);

    // RT namespace map: rt::get<rt::ns::field>()
    if (!named_args.named_common_runtime_args.empty() || !named_args.named_per_core_runtime_args.empty() ||
        !named_args.named_common_runtime_arg_arrays.empty() || !named_args.named_per_core_runtime_arg_arrays.empty()) {
        NamedRuntimeArgNamespaces rt_ns_map;

        // Common scalars: one slot each
        uint32_t common_index = static_cast<uint32_t>(kernel_descriptor.common_runtime_args.size());
        for (const auto& arg : named_args.named_common_runtime_args) {
            auto [ns, field] = split_name(arg.name);
            rt_ns_map[ns].push_back({field, common_index, 1, RuntimeArgDispatch::COMMON});
            common_index += 1;
        }
        // Common arrays: N contiguous slots each
        for (const auto& arg : named_args.named_common_runtime_arg_arrays) {
            auto [ns, field] = split_name(arg.name);
            uint32_t len = static_cast<uint32_t>(arg.values.size());
            rt_ns_map[ns].push_back({field, common_index, len, RuntimeArgDispatch::COMMON});
            common_index += len;
        }

        // Per-core scalars: one slot each
        uint32_t per_core_index = 0;
        if (!kernel_descriptor.runtime_args.empty()) {
            per_core_index = static_cast<uint32_t>(kernel_descriptor.runtime_args[0].second.size());
        }
        for (const auto& arg : named_args.named_per_core_runtime_args) {
            auto [ns, field] = split_name(arg.name);
            rt_ns_map[ns].push_back({field, per_core_index, 1, RuntimeArgDispatch::PER_CORE});
            per_core_index += 1;
        }
        // Per-core arrays: N contiguous slots each
        for (const auto& arg : named_args.named_per_core_runtime_arg_arrays) {
            auto [ns, field] = split_name(arg.name);
            uint32_t len = arg.core_values.empty() ? 0 : static_cast<uint32_t>(arg.core_values[0].second.size());
            rt_ns_map[ns].push_back({field, per_core_index, len, RuntimeArgDispatch::PER_CORE});
            per_core_index += len;
        }

        kernel->set_named_runtime_arg_namespaces(rt_ns_map);
    }

    // CT namespace map: ct::ns::field (plain constexpr values)
    if (!kernel_descriptor.named_compile_time_args.empty()) {
        NamedCTArgNamespaces ct_ns_map;
        std::unordered_map<std::string, uint32_t> seen_ct_args;
        for (const auto& [name, value] : kernel_descriptor.named_compile_time_args) {
            auto it = seen_ct_args.find(name);
            if (it != seen_ct_args.end()) {
                TT_FATAL(
                    it->second == value,
                    "named_compile_time_arg '{}' is defined twice with conflicting values ({} vs {}). "
                    "Each CT arg name must be unique across all sub-lists.",
                    name,
                    it->second,
                    value);
                continue;  // same value -- silently skip the duplicate
            }
            seen_ct_args.emplace(name, value);
            auto [ns, field] = split_name(name);
            ct_ns_map[ns].emplace_back(field, value);
        }
        kernel->set_named_ct_arg_namespaces(ct_ns_map);
    }
}

// Emits named_args_generated.h into the given directory.
// Returns true if a header was written (i.e. the kernel has named args).
// Mirrors `write_named_args_generated_header()` in jit_build/genfiles.cpp
// for the emulated (non-silicon) JIT path.
bool emit_named_args_header(
    const std::string& dir,
    const NamedCTArgNamespaces& named_ct_arg_namespaces,
    const NamedRuntimeArgNamespaces& named_runtime_arg_namespaces) {
    std::set<std::string> all_ns;
    for (const auto& [ns, _] : named_ct_arg_namespaces) {
        all_ns.insert(ns);
    }
    for (const auto& [ns, _] : named_runtime_arg_namespaces) {
        if (!ns.empty()) {
            all_ns.insert(ns);
        }
    }
    std::ostringstream header_ct;
    for (const auto& ns : all_ns) {
        if (!ns.empty()) {
            header_ct << "struct " << ns << " {\n";
        }
        if (auto it = named_ct_arg_namespaces.find(ns); it != named_ct_arg_namespaces.end()) {
            for (const auto& [field, value] : it->second) {
                header_ct << "    static constexpr uint32_t " << field << " = " << value << ";\n";
            }
        }
        if (auto it = named_runtime_arg_namespaces.find(ns); it != named_runtime_arg_namespaces.end()) {
            for (const auto& entry : it->second) {
                const char* dispatch_str = entry.dispatch == RuntimeArgDispatch::COMMON ? "rt_args::Dispatch::COMMON"
                                                                                        : "rt_args::Dispatch::PER_CORE";
                if (entry.length > 1) {
                    header_ct << "    static constexpr rt_args::ArrayArg " << entry.field << " = {" << entry.index
                              << ", " << entry.length << ", " << dispatch_str << "};\n";
                } else {
                    header_ct << "    static constexpr rt_args::Arg " << entry.field << " = {" << entry.index << ", "
                              << dispatch_str << "};\n";
                }
            }
        }
        if (!ns.empty()) {
            header_ct << "};\n";
        }
    }
    auto ct_str = header_ct.str();
    if (ct_str.empty()) {
        return false;
    }
    std::ofstream f(dir + "/named_args_generated.h");
    f << "#pragma once\n#include \"api/rt_arg.h\"\n\n";
    f << "namespace ct_args {\n" << ct_str << "}\n";
    return true;
}

}  // namespace tt::tt_metal::experimental
