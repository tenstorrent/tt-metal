// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

namespace tt::tt_metal {

// Identifier shared by reporting artifacts from one process. TTNN_RUN_SESSION_ID
// can stamp the same value across every rank of a distributed run.
const std::string& get_or_create_session_id();

}  // namespace tt::tt_metal
