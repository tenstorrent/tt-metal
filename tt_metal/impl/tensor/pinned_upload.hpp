// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

// Pinned upload is the host-to-device write path for large tensors. The default path has the host CPU copy the
// tensor into the command queue's issue ring, a chunk at a time, for the prefetcher to forward to the device. The
// pinned path instead pins the tensor's own host pages through the kernel driver and maps them into the device's NOC
// address space (`experimental::PinnedMemory`), so the prefetcher reads the data straight out of the host buffer over
// PCIe and the CPU copies nothing. `experimental::PinnedMemoryCache` keeps each pin, keyed by host address, so later
// uploads from the same buffer reuse it. The path needs a system where pinned memory can be mapped to the NOC
// (`experimental::GetMemoryPinningParameters`); elsewhere, or when a pin fails, uploads take the copy path.
namespace tt::tt_metal::pinned_upload {

// Pinning a buffer locks every page of it and sets up a device-visible mapping for it, which costs more than copying a
// small tensor through the issue ring. Uploads at or below this size take the copy path.
inline constexpr size_t k_pin_write_threshold_bytes = 32 * 1024 * 1024;  // 32 MB

}  // namespace tt::tt_metal::pinned_upload
