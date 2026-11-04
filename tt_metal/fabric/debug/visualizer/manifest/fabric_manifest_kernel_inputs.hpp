// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <source_location>
#include <span>
#include <string_view>
#include <vector>

#include "tt_metal/fabric/fabric_router_builder.hpp"

namespace tt::tt_fabric::manifest {

// Reads a router's kernel inputs by name. A read fails when the arg is missing, or when the ERISCs it reads from were
// given different values, and the error names the code that read it. Nothing is recorded: which args the manifest
// reads is checked in the source by check_manifest_kernel_inputs.py.
//
// Channels are indexed flat across VCs, as RouterVcShape::flat_sender_id and flat_receiver_id number them.
class RouterArgs {
public:
    explicit RouterArgs(std::span<const KernelInputs> kernel_inputs_by_erisc);

    // An arg every ERISC was given, with the same value.
    uint32_t read_router_arg(
        std::string_view name, std::source_location reader = std::source_location::current()) const;
    // An arg the builder emits only in some configs. Empty when no ERISC was given it.
    std::optional<uint32_t> find_router_arg(
        std::string_view name, std::source_location reader = std::source_location::current()) const;
    // A channel's arg, read from the ERISCs that service the channel, or from ERISC 0 when none do. Other ERISCs may
    // be given a placeholder.
    uint32_t read_sender_arg(
        std::string_view name,
        uint32_t sender_index,
        std::source_location reader = std::source_location::current()) const;
    uint32_t read_receiver_arg(
        std::string_view name,
        uint32_t receiver_index,
        std::source_location reader = std::source_location::current()) const;
    uint32_t read_erisc_arg(
        uint32_t erisc, std::string_view name, std::source_location reader = std::source_location::current()) const;

    // The ERISCs the builder assigned a channel to (IS_SENDER_CHANNEL_<i>_SERVICED, IS_RECEIVER_CHANNEL_<i>_SERVICED).
    std::vector<uint32_t> find_eriscs_servicing_sender(
        uint32_t sender_index, std::source_location reader = std::source_location::current()) const;
    std::vector<uint32_t> find_eriscs_servicing_receiver(
        uint32_t receiver_index, std::source_location reader = std::source_location::current()) const;

    tt::tt_metal::NOC read_erisc_noc(uint32_t erisc) const;
    // Whether the router was built with `name` defined. Fails when ERISCs differ.
    bool has_define(std::string_view name, std::source_location reader = std::source_location::current()) const;
    uint32_t num_eriscs() const;
    std::span<const KernelInputs> get_kernel_inputs_by_erisc() const { return kernel_inputs_by_erisc_; }

private:
    std::optional<uint32_t> find_erisc_arg(uint32_t erisc, std::string_view name) const;
    uint32_t read_arg_shared_by(
        const std::vector<uint32_t>& eriscs, std::string_view name, std::source_location reader) const;
    uint32_t read_channel_arg(
        std::string_view name, const std::vector<uint32_t>& servicing_eriscs, std::source_location reader) const;
    std::vector<uint32_t> find_eriscs_with_flag(std::string_view flag_name, std::source_location reader) const;

    std::span<const KernelInputs> kernel_inputs_by_erisc_;
    std::vector<uint32_t> all_eriscs_;
};

}  // namespace tt::tt_fabric::manifest
