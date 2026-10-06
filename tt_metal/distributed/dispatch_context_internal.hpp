// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tt::tt_metal::experimental {

// True between a successful DispatchContext::initialize_fast_dispatch and the matching
// terminate_fast_dispatch. Internal: MeshDeviceImpl uses it to refuse sub-device manager loads while a
// manual Fast Dispatch session is open. Defined in dispatch_context.cpp.
bool is_manual_fast_dispatch_session_active();

}  // namespace tt::tt_metal::experimental
