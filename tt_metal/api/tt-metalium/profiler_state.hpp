// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tt {

namespace tt_metal {

// Get global device profiling state based on build flag and environment variables
bool getDeviceProfilerState();

}  // namespace tt_metal

}  // namespace tt
