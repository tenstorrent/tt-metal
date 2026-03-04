// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tt::tt_fabric {

// Entry point: checks all guards internally (runtime option, Mock device, buffer enabled).
// Safe to call unconditionally — returns immediately if capture is not active.
void export_channel_trimming_capture();

}  // namespace tt::tt_fabric
