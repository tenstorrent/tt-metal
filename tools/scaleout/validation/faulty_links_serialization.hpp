// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <cstdint>
#include "tools/scaleout/validation/faulty_links.hpp"

namespace tt::scaleout::validation {

std::vector<uint8_t> serialize_faulty_links_to_bytes(const std::vector<::FaultyLink>& faulty_links);
std::vector<::FaultyLink> deserialize_faulty_links_from_bytes(const std::vector<uint8_t>& data);

}  // namespace tt::scaleout::validation
