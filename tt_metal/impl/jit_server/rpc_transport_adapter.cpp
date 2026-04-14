// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/jit_server/rpc_transport_adapter.hpp"

#include "impl/jit_server/jit_compile_rpc_client.hpp"

namespace tt::tt_metal {

RpcTransportAdapter::RpcTransportAdapter(const std::string& endpoint) :
    session_(std::make_unique<jit_server::JitCompileRpcSession>(endpoint)) {}

RpcTransportAdapter::~RpcTransportAdapter() = default;

jit_server::UploadFirmwareResponse RpcTransportAdapter::upload_firmware(
    const jit_server::UploadFirmwareRequest& request) {
    return session_->upload_firmware(request);
}

void RpcTransportAdapter::send(const jit_server::CompileRequest& request) { session_->send(request); }

std::vector<jit_server::CompileResponse> RpcTransportAdapter::wait_all() { return session_->wait_all(); }

std::unique_ptr<RemoteCompileTransport> RpcTransportAdapter::create(const std::string& endpoint) {
    return std::make_unique<RpcTransportAdapter>(endpoint);
}

}  // namespace tt::tt_metal
