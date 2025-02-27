// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tt {

enum class WorkExecutorMode {
    SYNCHRONOUS = 0,
    ASYNCHRONOUS = 1,
};

enum class WorkerState {
    RUNNING = 0,
    TERMINATE = 1,
    IDLE = 2,
};

class WorkExecutor;

}  // namespace tt
