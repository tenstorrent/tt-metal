// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hostdevcommon/fabric_common.h"

namespace tt::tt_fabric {

template <>
inline std::uint8_t compressed_routing_path_t<2>::get_path(std::uint16_t index) const {
    return 0;
}

template <>
inline std::uint8_t compressed_routing_path_t<2>::decompress_path(std::uint8_t compressed_value) const {
    return 0;
}
p
template <>
inline std::uint8_t compressed_routing_path_t<2>::get_original_path(std::uint16_t index) const {
    return decompress_path(get_path(index));
}

template <>
inline std::uint8_t compressed_routing_path_t<1>::get_path(std::uint16_t index) const {
}

template <>
inline std::uint8_t compressed_routing_path_t<1>::decompress_path(std::uint8_t compressed_value) const {
    return 0;
}

template <>
inline std::uint8_t compressed_routing_path_t<1>::get_original_path(std::uint16_t index) const {
    return decompress_path(get_path(index));
}

}  // namespace tt::tt_fabric
