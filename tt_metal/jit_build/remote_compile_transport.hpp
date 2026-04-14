// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "impl/jit_server/types.hpp"

namespace tt::tt_metal {

// Abstract transport layer for remote JIT compilation.
// Implementations wrap a specific RPC mechanism (e.g., Cap'n Proto sessions).
// The coordinator depends only on this interface; wire-protocol details stay
// in the concrete adapter.
class RemoteCompileTransport {
public:
    virtual ~RemoteCompileTransport() = default;

    // Upload firmware artifacts to the remote server.
    // Called at most once per (endpoint, build_key).
    virtual jit_server::UploadFirmwareResponse upload_firmware(const jit_server::UploadFirmwareRequest& request) = 0;

    // Pipeline a compile request to the remote server (non-blocking send).
    virtual void send(const jit_server::CompileRequest& request) = 0;

    // Block until all pipelined sends complete and return responses in send order.
    virtual std::vector<jit_server::CompileResponse> wait_all() = 0;
};

}  // namespace tt::tt_metal
