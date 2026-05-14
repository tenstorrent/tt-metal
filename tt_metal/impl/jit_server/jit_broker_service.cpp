// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/jit_server/jit_broker_service.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

#include <tt-logger/tt-logger.hpp>

namespace tt::tt_metal::jit_server {

namespace {

broker_rpc::FirmwareState to_rpc_state(FirmwareState state) {
    return state == FirmwareState::PRESENT ? broker_rpc::FirmwareState::PRESENT : broker_rpc::FirmwareState::ABSENT;
}

KernelKey read_kernel_key(broker_rpc::KernelKey::Reader reader) {
    KernelKey key;
    key.build_key = reader.getBuildKey();
    key.kernel_name = reader.getKernelName().cStr();
    return key;
}

}  // namespace

std::string JitBrokerService::pick_least_loaded_server() const {
    if (known_servers_.empty()) {
        throw std::runtime_error("No servers registered in JIT broker");
    }
    const auto cmp = [this](const std::string& lhs, const std::string& rhs) {
        const std::uint64_t lhs_load = load_.contains(lhs) ? load_.at(lhs) : 0;
        const std::uint64_t rhs_load = load_.contains(rhs) ? load_.at(rhs) : 0;
        if (lhs_load != rhs_load) {
            return lhs_load < rhs_load;
        }
        return lhs < rhs;
    };
    return *std::min_element(known_servers_.begin(), known_servers_.end(), cmp);
}

void JitBrokerService::notify_firmware_waiters(const FirmwareKey& key) {
    auto it = firmware_waiters_.find(key);
    if (it == firmware_waiters_.end()) {
        return;
    }
    for (auto& fulfiller : it->second) {
        fulfiller->fulfill();
    }
    firmware_waiters_.erase(it);
}

kj::Promise<void> JitBrokerService::assign(AssignContext context) {
    const auto params = context.getParams();
    const std::uint64_t build_key = params.getBuildKey();
    auto keys = params.getKernelKeys();
    auto assignments = context.getResults().initAssignments(keys.size());
    for (std::size_t i = 0; i < keys.size(); ++i) {
        KernelKey key{build_key, keys[i].cStr()};
        std::string chosen;
        if (auto it = cache_dir_.find(key); it != cache_dir_.end()) {
            chosen = it->second;
        } else if (auto it = pending_.find(key); it != pending_.end()) {
            chosen = it->second;
        } else {
            chosen = pick_least_loaded_server();
            pending_[key] = chosen;
        }
        load_[chosen] += 1;
        const std::uint64_t handle = next_handle_++;
        issued_handles_[handle] = {chosen, key};

        auto entry = assignments[i];
        entry.setServerEndpoint(chosen);
        entry.setHandle(handle);
        const bool has_firmware = firmware_have_[build_key].contains(chosen);
        entry.setFirmwareState(to_rpc_state(has_firmware ? FirmwareState::PRESENT : FirmwareState::ABSENT));
    }
    return kj::READY_NOW;
}

kj::Promise<void> JitBrokerService::claimFirmwareUpload(ClaimFirmwareUploadContext context) {
    const auto params = context.getParams();
    const std::uint64_t build_key = params.getBuildKey();
    const std::string server = params.getServerEndpoint().cStr();
    broker_rpc::FirmwareUploadAction action = broker_rpc::FirmwareUploadAction::WAIT_FOR_OTHER;
    FirmwareKey key{build_key, server};
    if (firmware_have_[build_key].contains(server)) {
        action = broker_rpc::FirmwareUploadAction::SKIP_ALREADY_PRESENT;
    } else if (!firmware_pending_.contains(key)) {
        firmware_pending_.insert(key);
        action = broker_rpc::FirmwareUploadAction::YOU_UPLOAD;
    }
    context.getResults().setAction(action);
    return kj::READY_NOW;
}

kj::Promise<void> JitBrokerService::releaseFirmwareUpload(ReleaseFirmwareUploadContext context) {
    const auto params = context.getParams();
    FirmwareKey key{params.getBuildKey(), params.getServerEndpoint().cStr()};
    firmware_pending_.erase(key);
    if (params.getSuccess()) {
        firmware_have_[key.build_key].insert(key.server_endpoint);
    }
    notify_firmware_waiters(key);
    return kj::READY_NOW;
}

kj::Promise<void> JitBrokerService::waitFirmwareReady(WaitFirmwareReadyContext context) {
    const auto params = context.getParams();
    FirmwareKey key{params.getBuildKey(), params.getServerEndpoint().cStr()};
    if (firmware_have_[key.build_key].contains(key.server_endpoint) || !firmware_pending_.contains(key)) {
        return kj::READY_NOW;
    }
    auto paf = kj::newPromiseAndFulfiller<void>();
    firmware_waiters_[key].push_back(kj::mv(paf.fulfiller));
    return kj::mv(paf.promise);
}

kj::Promise<void> JitBrokerService::registerServer(RegisterServerContext context) {
    const std::string server = context.getParams().getServerEndpoint().cStr();
    if (!load_.contains(server)) {
        known_servers_.push_back(server);
        load_[server] = 0;
    }
    return kj::READY_NOW;
}

kj::Promise<void> JitBrokerService::reportCacheState(ReportCacheStateContext context) {
    const auto params = context.getParams();
    const std::string server = params.getServerEndpoint().cStr();
    if (!load_.contains(server)) {
        known_servers_.push_back(server);
        load_[server] = 0;
    }

    auto keys = params.getKernelKeys();
    for (auto key_reader : keys) {
        cache_dir_[read_kernel_key(key_reader)] = server;
    }

    auto firmware_build_keys = params.getFirmwareBuildKeys();
    for (auto build_key : firmware_build_keys) {
        firmware_have_[build_key].insert(server);
    }
    return kj::READY_NOW;
}

kj::Promise<void> JitBrokerService::release(ReleaseContext context) {
    const auto params = context.getParams();
    const std::uint64_t handle = params.getHandle();
    auto it = issued_handles_.find(handle);
    if (it == issued_handles_.end()) {
        throw std::runtime_error("Broker release called with unknown handle");
    }
    const auto [server, key] = it->second;
    issued_handles_.erase(it);
    if (load_[server] > 0) {
        load_[server] -= 1;
    }
    if (params.getWasRealCompile()) {
        cache_dir_[key] = server;
        pending_.erase(key);
    }
    return kj::READY_NOW;
}

}  // namespace tt::tt_metal::jit_server
