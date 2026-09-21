// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Impl-internal DRISC L1 accessor used by the tensor prefetcher manager.

#pragma once

#include <tt-metalium/hal_types.hpp>

#include <tt-metalium/experimental/prefetcher_pipe.hpp>

namespace tt::tt_metal {
namespace experimental {
// DRISC L1 address of `pipe`'s sender config page. The Tensor prefetcher stamps it into the header
// of every request routed to that pipe's sender, so the DRISC kernel can find its endpoint state.
DeviceAddr sender_state_drisc_l1_base(const PrefetcherPipe& pipe);

}  // namespace experimental
}  // namespace tt::tt_metal
