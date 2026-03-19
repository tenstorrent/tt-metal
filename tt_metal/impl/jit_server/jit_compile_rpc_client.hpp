// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "impl/jit_server/types.hpp"

namespace tt::tt_metal::jit_server {

class JitCompileRpcClient {
public:
    explicit JitCompileRpcClient(std::string endpoint);

    static bool enabled();
    static std::string endpoint_from_env();

    CompileResponse compile(const CompileRequest& request) const;
    std::vector<CompileResponse> compile_batch(const std::vector<CompileRequest>& requests) const;

private:
    std::string endpoint_;
};

// Holds a single persistent connection for incremental pipelined sends.
// Usage: send() requests as they become ready, then wait_all() to collect responses.
class JitCompileRpcSession {
public:
    explicit JitCompileRpcSession(const std::string& endpoint);
    ~JitCompileRpcSession();

    void send(const CompileRequest& request);
    CompileResponse send_and_wait(const CompileRequest& request);
    std::vector<CompileResponse> wait_all();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace tt::tt_metal::jit_server
