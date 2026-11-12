// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// A rectangle of nodes as Watcher checks multicasts against it. Dependency-free, so the binding tokens can carry the
// nodes of their DFBs and semaphores in builds without Watcher too.
#pragma once

#include <cstdint>

// A node's NoC coordinates, which fit in 8 bits each.
struct DebugNodeRangeCoord {
    uint8_t x;
    uint8_t y;
};

// A rectangle of nodes, as the NoC coordinates of its corners.
struct DebugNodeRange {
    DebugNodeRangeCoord start;
    DebugNodeRangeCoord end;
};
