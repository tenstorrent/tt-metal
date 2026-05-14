// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "impl/jit_server/jit_broker_types.hpp"

namespace tt::tt_metal::jit_server {

class JitBrokerRpcClient {
public:
    explicit JitBrokerRpcClient(std::string endpoint);

    static std::string endpoint_from_env();

    BrokerAssignResponse assign(const BrokerAssignRequest& request) const;
    FirmwareUploadAction claim_firmware_upload(std::uint64_t build_key, const std::string& server_endpoint) const;
    void release_firmware_upload(std::uint64_t build_key, const std::string& server_endpoint, bool success) const;
    void wait_firmware_ready(std::uint64_t build_key, const std::string& server_endpoint) const;
    void register_server(const std::string& server_endpoint) const;
    void report_cache_state(
        const std::string& server_endpoint,
        const std::vector<KernelKey>& kernel_keys,
        const std::vector<std::uint64_t>& firmware_build_keys) const;
    void release(std::uint64_t handle, const KernelKey& kernel_key, bool was_real_compile) const;

private:
    std::string endpoint_;
};

class JitBrokerRpcSession {
public:
    explicit JitBrokerRpcSession(const std::string& endpoint);
    ~JitBrokerRpcSession();

    BrokerAssignResponse assign(const BrokerAssignRequest& request);
    FirmwareUploadAction claim_firmware_upload(std::uint64_t build_key, const std::string& server_endpoint);
    void release_firmware_upload(std::uint64_t build_key, const std::string& server_endpoint, bool success);
    void wait_firmware_ready(std::uint64_t build_key, const std::string& server_endpoint);
    void register_server(const std::string& server_endpoint);
    void report_cache_state(
        const std::string& server_endpoint,
        const std::vector<KernelKey>& kernel_keys,
        const std::vector<std::uint64_t>& firmware_build_keys);
    void release(std::uint64_t handle, const KernelKey& kernel_key, bool was_real_compile);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace tt::tt_metal::jit_server
