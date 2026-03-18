// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include "impl/jit_server/types.hpp"

namespace tt::tt_metal::jit_server {

class JitCompileRpcClient {
public:
    explicit JitCompileRpcClient(std::string endpoint);

    static bool enabled();
    static std::string endpoint_from_env();

    PrepareGenfilesResponse prepareGenfiles(const PrepareGenfilesRequest& request) const;
    CompileResponse compile(const CompileRequest& request) const;

private:
    std::string endpoint_;
};

}  // namespace tt::tt_metal::jit_server
