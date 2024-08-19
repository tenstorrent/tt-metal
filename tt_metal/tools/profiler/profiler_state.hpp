// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tt {

namespace tt_metal {

// Get global device profiling state based on build flag and environment variables
bool getDeviceProfilerState ();

}  // namespace tt_metal

}  // namespace tt
