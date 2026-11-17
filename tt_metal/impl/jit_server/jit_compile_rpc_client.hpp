// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "impl/jit_server/types.hpp"

namespace tt::tt_metal::jit_server {

// The server could not be reached, the connection dropped, or a reply did not arrive within
// TT_METAL_JIT_SERVER_TIMEOUT_S. Unlike a reply with success == false (a genuine compile error),
// building the kernel locally instead is a valid recovery.
class RemoteCompileTransportError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class JitCompileRpcClient {
public:
    explicit JitCompileRpcClient(std::string endpoint);

    static bool enabled();
    static std::vector<std::string> endpoints_from_env();
    static std::string endpoint_from_env();

    CompileResponse compile(const CompileRequest& request) const;
    std::vector<CompileResponse> compile_batch(const std::vector<CompileRequest>& requests) const;
    UploadFirmwareResponse upload_firmware(const UploadFirmwareRequest& request) const;

private:
    std::string endpoint_;
};

// Holds a single persistent connection for incremental pipelined sends.
// Usage: send() requests as they become ready, then wait_all() to collect responses.
// Connecting and every wait throw RemoteCompileTransportError on a transport failure or timeout.
class JitCompileRpcSession {
public:
    explicit JitCompileRpcSession(const std::string& endpoint);
    ~JitCompileRpcSession();

    void send(const CompileRequest& request);
    CompileResponse send_and_wait(const CompileRequest& request);
    std::vector<CompileResponse> wait_all();

    UploadFirmwareResponse upload_firmware(const UploadFirmwareRequest& request);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace tt::tt_metal::jit_server
