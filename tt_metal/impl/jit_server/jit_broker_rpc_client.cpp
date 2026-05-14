// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/jit_server/jit_broker_rpc_client.hpp"

#include <capnp/ez-rpc.h>
#include <cstdlib>
#include <stdexcept>
#include <utility>

#include "impl/jit_server/broker.capnp.h"

namespace tt::tt_metal::jit_server {

namespace {

constexpr const char* kJitBrokerEndpointEnv = "TT_METAL_JIT_BROKER_ENDPOINT";

FirmwareState read_firmware_state(broker_rpc::FirmwareState state) {
    return state == broker_rpc::FirmwareState::PRESENT ? FirmwareState::PRESENT : FirmwareState::ABSENT;
}

FirmwareUploadAction read_upload_action(broker_rpc::FirmwareUploadAction action) {
    switch (action) {
        case broker_rpc::FirmwareUploadAction::SKIP_ALREADY_PRESENT: return FirmwareUploadAction::SKIP_ALREADY_PRESENT;
        case broker_rpc::FirmwareUploadAction::YOU_UPLOAD: return FirmwareUploadAction::YOU_UPLOAD;
        case broker_rpc::FirmwareUploadAction::WAIT_FOR_OTHER: return FirmwareUploadAction::WAIT_FOR_OTHER;
    }
    return FirmwareUploadAction::WAIT_FOR_OTHER;
}

template <typename TKeyBuilder>
void fill_kernel_key(TKeyBuilder builder, const KernelKey& key) {
    builder.setBuildKey(key.build_key);
    builder.setKernelName(key.kernel_name);
}

template <typename TRequestBuilder>
void fill_assign_request(TRequestBuilder& builder, const BrokerAssignRequest& request) {
    builder.setBuildKey(request.build_key);
    auto kernels = builder.initKernelKeys(request.kernel_keys.size());
    for (std::size_t i = 0; i < request.kernel_keys.size(); ++i) {
        kernels.set(i, request.kernel_keys[i]);
    }
}

BrokerAssignResponse read_assign_response(broker_rpc::JitDispatchBroker::AssignResults::Reader reader) {
    BrokerAssignResponse response;
    auto assignments = reader.getAssignments();
    response.assignments.reserve(assignments.size());
    for (auto assignment_reader : assignments) {
        BrokerAssignment assignment;
        assignment.server_endpoint = assignment_reader.getServerEndpoint().cStr();
        assignment.handle = assignment_reader.getHandle();
        assignment.firmware_state = read_firmware_state(assignment_reader.getFirmwareState());
        response.assignments.push_back(std::move(assignment));
    }
    return response;
}

template <typename TRequestBuilder>
void fill_report_cache_state_request(
    TRequestBuilder& builder,
    const std::string& server_endpoint,
    const std::vector<KernelKey>& kernel_keys,
    const std::vector<std::uint64_t>& firmware_build_keys) {
    builder.setServerEndpoint(server_endpoint);
    auto keys_builder = builder.initKernelKeys(kernel_keys.size());
    for (std::size_t i = 0; i < kernel_keys.size(); ++i) {
        fill_kernel_key(keys_builder[i], kernel_keys[i]);
    }
    auto fw_builder = builder.initFirmwareBuildKeys(firmware_build_keys.size());
    for (std::size_t i = 0; i < firmware_build_keys.size(); ++i) {
        fw_builder.set(i, firmware_build_keys[i]);
    }
}

template <typename TRequestBuilder>
void fill_release_request(
    TRequestBuilder& builder, std::uint64_t handle, const KernelKey& kernel_key, bool was_real_compile) {
    builder.setHandle(handle);
    fill_kernel_key(builder.initKernelKey(), kernel_key);
    builder.setWasRealCompile(was_real_compile);
}

}  // namespace

JitBrokerRpcClient::JitBrokerRpcClient(std::string endpoint) : endpoint_(std::move(endpoint)) {}

std::string JitBrokerRpcClient::endpoint_from_env() {
    const char* endpoint = std::getenv(kJitBrokerEndpointEnv);
    if (endpoint == nullptr) {
        return {};
    }
    return endpoint;
}

BrokerAssignResponse JitBrokerRpcClient::assign(const BrokerAssignRequest& request) const {
    try {
        capnp::EzRpcClient client(endpoint_);
        auto cap = client.getMain<broker_rpc::JitDispatchBroker>();
        auto rpc_request = cap.assignRequest();
        fill_assign_request(rpc_request, request);
        auto result = rpc_request.send().wait(client.getWaitScope());
        return read_assign_response(result);
    } catch (const kj::Exception& e) {
        throw std::runtime_error("Broker assign failed at " + endpoint_ + ": " + e.getDescription().cStr());
    }
}

FirmwareUploadAction JitBrokerRpcClient::claim_firmware_upload(
    std::uint64_t build_key, const std::string& server_endpoint) const {
    try {
        capnp::EzRpcClient client(endpoint_);
        auto cap = client.getMain<broker_rpc::JitDispatchBroker>();
        auto rpc_request = cap.claimFirmwareUploadRequest();
        rpc_request.setBuildKey(build_key);
        rpc_request.setServerEndpoint(server_endpoint);
        auto result = rpc_request.send().wait(client.getWaitScope());
        return read_upload_action(result.getAction());
    } catch (const kj::Exception& e) {
        throw std::runtime_error(
            "Broker claimFirmwareUpload failed at " + endpoint_ + ": " + e.getDescription().cStr());
    }
}

void JitBrokerRpcClient::release_firmware_upload(
    std::uint64_t build_key, const std::string& server_endpoint, bool success) const {
    capnp::EzRpcClient client(endpoint_);
    auto cap = client.getMain<broker_rpc::JitDispatchBroker>();
    auto rpc_request = cap.releaseFirmwareUploadRequest();
    rpc_request.setBuildKey(build_key);
    rpc_request.setServerEndpoint(server_endpoint);
    rpc_request.setSuccess(success);
    rpc_request.send().wait(client.getWaitScope());
}

void JitBrokerRpcClient::wait_firmware_ready(std::uint64_t build_key, const std::string& server_endpoint) const {
    capnp::EzRpcClient client(endpoint_);
    auto cap = client.getMain<broker_rpc::JitDispatchBroker>();
    auto rpc_request = cap.waitFirmwareReadyRequest();
    rpc_request.setBuildKey(build_key);
    rpc_request.setServerEndpoint(server_endpoint);
    rpc_request.send().wait(client.getWaitScope());
}

void JitBrokerRpcClient::register_server(const std::string& server_endpoint) const {
    capnp::EzRpcClient client(endpoint_);
    auto cap = client.getMain<broker_rpc::JitDispatchBroker>();
    auto rpc_request = cap.registerServerRequest();
    rpc_request.setServerEndpoint(server_endpoint);
    rpc_request.send().wait(client.getWaitScope());
}

void JitBrokerRpcClient::report_cache_state(
    const std::string& server_endpoint,
    const std::vector<KernelKey>& kernel_keys,
    const std::vector<std::uint64_t>& firmware_build_keys) const {
    capnp::EzRpcClient client(endpoint_);
    auto cap = client.getMain<broker_rpc::JitDispatchBroker>();
    auto rpc_request = cap.reportCacheStateRequest();
    fill_report_cache_state_request(rpc_request, server_endpoint, kernel_keys, firmware_build_keys);
    rpc_request.send().wait(client.getWaitScope());
}

void JitBrokerRpcClient::release(std::uint64_t handle, const KernelKey& kernel_key, bool was_real_compile) const {
    capnp::EzRpcClient client(endpoint_);
    auto cap = client.getMain<broker_rpc::JitDispatchBroker>();
    auto rpc_request = cap.releaseRequest();
    fill_release_request(rpc_request, handle, kernel_key, was_real_compile);
    rpc_request.send().wait(client.getWaitScope());
}

struct JitBrokerRpcSession::Impl {
    capnp::EzRpcClient client;
    broker_rpc::JitDispatchBroker::Client cap;

    explicit Impl(const std::string& endpoint) :
        client(endpoint), cap(client.getMain<broker_rpc::JitDispatchBroker>()) {}
};

JitBrokerRpcSession::JitBrokerRpcSession(const std::string& endpoint) : impl_(std::make_unique<Impl>(endpoint)) {}

JitBrokerRpcSession::~JitBrokerRpcSession() = default;

BrokerAssignResponse JitBrokerRpcSession::assign(const BrokerAssignRequest& request) {
    auto rpc_request = impl_->cap.assignRequest();
    fill_assign_request(rpc_request, request);
    auto result = rpc_request.send().wait(impl_->client.getWaitScope());
    return read_assign_response(result);
}

FirmwareUploadAction JitBrokerRpcSession::claim_firmware_upload(
    std::uint64_t build_key, const std::string& server_endpoint) {
    auto rpc_request = impl_->cap.claimFirmwareUploadRequest();
    rpc_request.setBuildKey(build_key);
    rpc_request.setServerEndpoint(server_endpoint);
    auto result = rpc_request.send().wait(impl_->client.getWaitScope());
    return read_upload_action(result.getAction());
}

void JitBrokerRpcSession::release_firmware_upload(
    std::uint64_t build_key, const std::string& server_endpoint, bool success) {
    auto rpc_request = impl_->cap.releaseFirmwareUploadRequest();
    rpc_request.setBuildKey(build_key);
    rpc_request.setServerEndpoint(server_endpoint);
    rpc_request.setSuccess(success);
    rpc_request.send().wait(impl_->client.getWaitScope());
}

void JitBrokerRpcSession::wait_firmware_ready(std::uint64_t build_key, const std::string& server_endpoint) {
    auto rpc_request = impl_->cap.waitFirmwareReadyRequest();
    rpc_request.setBuildKey(build_key);
    rpc_request.setServerEndpoint(server_endpoint);
    rpc_request.send().wait(impl_->client.getWaitScope());
}

void JitBrokerRpcSession::register_server(const std::string& server_endpoint) {
    auto rpc_request = impl_->cap.registerServerRequest();
    rpc_request.setServerEndpoint(server_endpoint);
    rpc_request.send().wait(impl_->client.getWaitScope());
}

void JitBrokerRpcSession::report_cache_state(
    const std::string& server_endpoint,
    const std::vector<KernelKey>& kernel_keys,
    const std::vector<std::uint64_t>& firmware_build_keys) {
    auto rpc_request = impl_->cap.reportCacheStateRequest();
    fill_report_cache_state_request(rpc_request, server_endpoint, kernel_keys, firmware_build_keys);
    rpc_request.send().wait(impl_->client.getWaitScope());
}

void JitBrokerRpcSession::release(std::uint64_t handle, const KernelKey& kernel_key, bool was_real_compile) {
    auto rpc_request = impl_->cap.releaseRequest();
    fill_release_request(rpc_request, handle, kernel_key, was_real_compile);
    rpc_request.send().wait(impl_->client.getWaitScope());
}

}  // namespace tt::tt_metal::jit_server
