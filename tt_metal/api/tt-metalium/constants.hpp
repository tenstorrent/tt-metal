// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Tile constants are shared between host and device code.
// The canonical location is hostdevcommon/tile_constants.h
// This header re-exports them for backwards compatibility with host code.
#include <hostdevcommon/tile_constants.h>
