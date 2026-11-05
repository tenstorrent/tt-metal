// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_kernel_inputs.hpp"

#include <filesystem>
#include <numeric>
#include <string>

#include <fmt/format.h>
#include <tt_stl/assert.hpp>

#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_fields.hpp"

namespace tt::tt_fabric::manifest {

namespace {

std::string describe_reader(const std::source_location& reader) {
    return fmt::format("{}:{}", std::filesystem::path(reader.file_name()).filename().string(), reader.line());
}

}  // namespace

RouterArgs::RouterArgs(std::span<const KernelInputs> kernel_inputs_by_erisc) :
    kernel_inputs_by_erisc_(kernel_inputs_by_erisc), all_eriscs_(kernel_inputs_by_erisc.size()) {
    TT_FATAL(!kernel_inputs_by_erisc_.empty(), "Fabric manifest: a router has no kernel inputs to read");
    std::iota(all_eriscs_.begin(), all_eriscs_.end(), 0);
}

uint32_t RouterArgs::num_eriscs() const { return static_cast<uint32_t>(kernel_inputs_by_erisc_.size()); }

std::optional<uint32_t> RouterArgs::find_erisc_arg(
    uint32_t erisc, std::string_view name, std::source_location reader) const {
    TT_FATAL(
        manifest_reads_arg(name),
        "Fabric manifest: {} reads {}, which no field table reads and k_router_pass_args does not list "
        "(fabric_manifest_fields.hpp); add it to k_router_pass_args",
        describe_reader(reader),
        name);
    TT_FATAL(
        erisc < num_eriscs(),
        "Fabric manifest: {} reads {} from ERISC{}, but the router has {} ERISCs",
        describe_reader(reader),
        name,
        erisc,
        num_eriscs());
    const auto& named_args = kernel_inputs_by_erisc_[erisc].named_compile_args;
    const auto arg = named_args.find(std::string(name));
    if (arg == named_args.end()) {
        return std::nullopt;
    }
    return arg->second;
}

uint32_t RouterArgs::read_erisc_arg(uint32_t erisc, std::string_view name, std::source_location reader) const {
    const auto value = find_erisc_arg(erisc, name, reader);
    TT_FATAL(
        value.has_value(),
        "Fabric manifest: {} reads {}, but ERISC{} wasn't given it",
        describe_reader(reader),
        name,
        erisc);
    return *value;
}

// Reads `name` from each of `eriscs`, which must all have the same value.
uint32_t RouterArgs::read_arg_shared_by(
    const std::vector<uint32_t>& eriscs, std::string_view name, std::source_location reader) const {
    const uint32_t value = read_erisc_arg(eriscs.front(), name, reader);
    for (const uint32_t erisc : eriscs) {
        const uint32_t erisc_value = read_erisc_arg(erisc, name, reader);
        TT_FATAL(
            erisc_value == value,
            "Fabric manifest: {} reads {} as one value, but ERISC{} has {} and ERISC{} has {}",
            describe_reader(reader),
            name,
            eriscs.front(),
            value,
            erisc,
            erisc_value);
    }
    return value;
}

uint32_t RouterArgs::read_router_arg(std::string_view name, std::source_location reader) const {
    return read_arg_shared_by(all_eriscs_, name, reader);
}

std::vector<uint32_t> RouterArgs::find_eriscs_with_flag(std::string_view flag_name, std::source_location reader) const {
    std::vector<uint32_t> eriscs;
    for (const uint32_t erisc : all_eriscs_) {
        if (read_erisc_arg(erisc, flag_name, reader) != 0) {
            eriscs.push_back(erisc);
        }
    }
    return eriscs;
}

std::vector<uint32_t> RouterArgs::find_eriscs_servicing_sender(
    uint32_t sender_index, std::source_location reader) const {
    return find_eriscs_with_flag(fmt::format("IS_SENDER_CHANNEL_{}_SERVICED", sender_index), reader);
}

std::vector<uint32_t> RouterArgs::find_eriscs_servicing_receiver(
    uint32_t receiver_index, std::source_location reader) const {
    return find_eriscs_with_flag(fmt::format("IS_RECEIVER_CHANNEL_{}_SERVICED", receiver_index), reader);
}

tt::tt_metal::NOC RouterArgs::read_erisc_noc(uint32_t erisc) const {
    TT_FATAL(
        erisc < num_eriscs(),
        "Fabric manifest: reads ERISC{}'s NoC, but the router has {} ERISCs",
        erisc,
        num_eriscs());
    return kernel_inputs_by_erisc_[erisc].noc;
}

bool RouterArgs::has_define(std::string_view name, std::source_location reader) const {
    TT_FATAL(
        manifest_reads_define(name),
        "Fabric manifest: {} checks the define {}, which k_router_pass_defines does not list "
        "(fabric_manifest_fields.hpp); add it to k_router_pass_defines",
        describe_reader(reader),
        name);
    const std::string define(name);
    const bool defined = kernel_inputs_by_erisc_.front().defines.contains(define);
    for (const uint32_t erisc : all_eriscs_) {
        TT_FATAL(
            kernel_inputs_by_erisc_[erisc].defines.contains(define) == defined,
            "Fabric manifest: {} checks the define {}, but ERISC0 {} it and ERISC{} {}",
            describe_reader(reader),
            name,
            defined ? "has" : "doesn't have",
            erisc,
            defined ? "doesn't" : "does");
    }
    return defined;
}

}  // namespace tt::tt_fabric::manifest
