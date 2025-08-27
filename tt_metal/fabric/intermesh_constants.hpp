// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tt::tt_fabric {
namespace intermesh_constants {
// TODO: remove once UMD can provide all intermesh links

// Constants used to derive the intermesh ethernet links config
static constexpr uint32_t LOCAL_BOARD_ID_OFFSET = 256;
static constexpr uint32_t REMOTE_BOARD_ID_OFFSET = 288;
static constexpr uint32_t REMOTE_ETH_CHAN_ID_OFFSET = 304;

}  // namespace intermesh_constants
}  // namespace tt::tt_fabric
