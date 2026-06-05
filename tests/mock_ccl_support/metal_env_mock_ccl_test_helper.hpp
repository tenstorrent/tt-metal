// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <string>

namespace tt::tt_metal::distributed {
class MeshDevice;
}

namespace tt::tt_metal::test_support {

struct OpConstraintQueryResult {
    bool success = false;
    std::optional<std::string> error_message;
};

OpConstraintQueryResult run_all_gather_constraint_query(distributed::MeshDevice* device);

}  // namespace tt::tt_metal::test_support
