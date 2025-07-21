// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tt::tt_fabric {
// routing mode macro for (mainly) kernel code
#define ROUTING_MODE_UNDEFINED 0x0000
#define ROUTING_MODE_1D 0x0001
#define ROUTING_MODE_2D 0x0002
#define ROUTING_MODE_3D 0x0004
#define ROUTING_MODE_LINE 0x0008
#define ROUTING_MODE_RING 0x0010
#define ROUTING_MODE_MESH 0x0020
#define ROUTING_MODE_TORUS 0x0040
#define ROUTING_MODE_LOW_LATENCY 0x0080
#define ROUTING_MODE_DYNAMIC 0x0100

// PUSH/PULL is for 2D and will be
// TODO: remove when tt_fabric removes these notion
#define ROUTING_MODE_PUSH 0x0200
#define ROUTING_MODE_PULL 0x0400

}  // namespace tt::tt_fabric
