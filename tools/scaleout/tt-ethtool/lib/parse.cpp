// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tools/scaleout/tt-ethtool/lib/operations.hpp"

#include <charconv>
#include <stdexcept>
#include <string_view>
#include <system_error>

#include <fmt/core.h>

#include "umd/device/arch/architecture_implementation.hpp"
#include "umd/device/cluster_descriptor.hpp"
#include "umd/device/types/arch.hpp"

namespace tt_ethtool {

namespace {

template <typename T>
T parse_integer(std::string_view s, std::string_view what) {
    T value{};
    auto* begin = s.data();
    auto* end = s.data() + s.size();
    auto result = std::from_chars(begin, end, value);
    if (result.ec != std::errc{} || result.ptr != end || s.empty()) {
        throw std::invalid_argument(fmt::format("{}: '{}' is not a valid integer", what, s));
    }
    return value;
}

}  // namespace

LinkRef parse_link_ref(std::string_view input) {
    const auto colon = input.find(':');
    if (colon == std::string_view::npos) {
        throw std::invalid_argument(fmt::format("link spec '{}' must be in the form <chip>:<channel>", input));
    }
    return LinkRef{
        .chip_id = parse_integer<tt::ChipId>(input.substr(0, colon), "chip id"),
        .channel = parse_integer<std::uint32_t>(input.substr(colon + 1), "channel"),
    };
}

void validate_link_ref(tt::umd::ClusterDescriptor& desc, LinkRef link) {
    const auto& all_chips = desc.get_all_chips();
    if (!all_chips.contains(link.chip_id)) {
        throw std::invalid_argument(fmt::format("Chip {} not found on this host.", link.chip_id));
    }
    const tt::ARCH arch = desc.get_arch(link.chip_id);
    const std::uint32_t num_eth = tt::umd::architecture_implementation::create(arch)->get_num_eth_channels();
    if (link.channel >= num_eth) {
        throw std::invalid_argument(fmt::format(
            "Channel {} out of range for chip {} ({}; has {} channels).",
            link.channel,
            link.chip_id,
            tt::arch_to_str(arch),
            num_eth));
    }
}

}  // namespace tt_ethtool
