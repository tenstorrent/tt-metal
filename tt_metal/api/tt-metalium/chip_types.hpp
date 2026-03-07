// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace tt {
namespace tt_metal {

// ChipId identifies a physical Tenstorrent device. Defined locally to avoid
// leaking UMD headers (umd/device/types/cluster_descriptor_types.hpp) into
// the tt-metalium public API.
using ChipId = uint32_t;

}  // namespace tt_metal
}  // namespace tt
