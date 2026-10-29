// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// The metadata type the example's kernel and its external process share. In actual use it belongs in hostdevcommon/,
// which kernels and host code both include; the example keeps it beside its sources.
#pragma once

#include <hostdevcommon/device_observability_types.hpp>

namespace tt::tt_metal::experimental::device_observability {

/**
 * @brief The metadata of a sample, a metric's value, which a kernel sends the external process:
 *
 *     DevicePublish<SampleMetadata{"bytes_processed_total"_s}, kSendToExternalProcess>(bytes);
 */
struct SampleMetadata {
    String name;
};

}  // namespace tt::tt_metal::experimental::device_observability
