// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tt {

// ChipId identifies a physical Tenstorrent device in a cluster (e.g., chip 0, chip 1).
// Defined locally to avoid leaking UMD headers into the tt-metalium public API.
using ChipId = int;

}  // namespace tt
