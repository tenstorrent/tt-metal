// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>

#include "jit_build/remote_compile_transport.hpp"

namespace tt::tt_metal::jit_server {
class JitCompileRpcSession;
}

namespace tt::tt_metal {

// Adapts an existing JitCompileRpcSession to the RemoteCompileTransport interface
// so the coordinator in jit_build can drive RPC without knowing Cap'n Proto details.
class RpcTransportAdapter final : public RemoteCompileTransport {
public:
    explicit RpcTransportAdapter(const std::string& endpoint);
    ~RpcTransportAdapter() override;

    RpcTransportAdapter(const RpcTransportAdapter&) = delete;
    RpcTransportAdapter& operator=(const RpcTransportAdapter&) = delete;

    jit_server::UploadFirmwareResponse upload_firmware(const jit_server::UploadFirmwareRequest& request) override;
    void send(const jit_server::CompileRequest& request) override;
    std::vector<jit_server::CompileResponse> wait_all() override;

    // Factory function suitable for RemoteCompileCoordinator::TransportFactory.
    static std::unique_ptr<RemoteCompileTransport> create(const std::string& endpoint);

private:
    std::unique_ptr<jit_server::JitCompileRpcSession> session_;
};

}  // namespace tt::tt_metal
