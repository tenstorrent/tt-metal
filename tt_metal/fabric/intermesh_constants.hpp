// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tt::tt_fabric {
namespace intermesh_constants {
// TODO: remove once UMD can provide all intermesh links

// Constants used to derive the intermesh ethernet links config
static constexpr uint32_t MULTI_MESH_ENABLED_VALUE = 0x2;
static constexpr uint32_t LINK_CONNECTED_MASK = 0x1;
static constexpr uint32_t MULTI_MESH_MODE_MASK = 0xFF;
static constexpr uint32_t INTERMESH_ETH_LINK_BITS_SHIFT = 8;
static constexpr uint32_t INTERMESH_ETH_LINK_BITS_MASK = 0xFFFF;
static constexpr uint32_t LOCAL_BOARD_ID_OFFSET = 256;
static constexpr uint32_t REMOTE_BOARD_ID_OFFSET = 288;
static constexpr uint32_t REMOTE_ETH_CHAN_ID_OFFSET = 304;

}  // namespace intermesh_constants
}  // namespace tt::tt_fabric
