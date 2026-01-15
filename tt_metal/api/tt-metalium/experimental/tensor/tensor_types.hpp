// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tt::tt_metal {

// TODO:
// DataType and associated utilities.
// (Renamed from types.hpp.)
// There prob is a metal alternative we should adopt here instead.
struct DataType {};
enum class Layout { ROW_MAJOR = 0, TILE = 1, INVALID = 2 };

}  // namespace tt::tt_metal
