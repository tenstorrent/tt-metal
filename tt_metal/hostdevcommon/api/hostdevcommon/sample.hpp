// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <hostdevcommon/streaming_profiler_types.hpp>

namespace tt::tt_metal::experimental::streaming_profiler {

/**
 * @brief The metadata of a sample, a metric's value, which a kernel sends the external process:
 *
 *     DevicePublish<SampleMetadata{"bytes_processed_total"_s}, kSendToExternalProcess>(bytes);
 */
struct SampleMetadata {
    String name;
};

}  // namespace tt::tt_metal::experimental::streaming_profiler
