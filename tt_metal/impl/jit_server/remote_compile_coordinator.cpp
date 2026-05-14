// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/jit_server/remote_compile_coordinator.hpp"

#include <filesystem>
#include <fstream>
#include <unordered_set>
#include <utility>

#include <tt_stl/assert.hpp>

#include "jit_build/build.hpp"
#include "jit_build/build_env_manager.hpp"
#include "jit_build/jit_build_utils.hpp"

namespace tt::tt_metal {

namespace fs = std::filesystem;

// -- Static global state --
std::mutex RemoteCompileCoordinator::s_dedup_mutex_;
std::unordered_map<std::size_t, std::shared_future<void>> RemoteCompileCoordinator::s_dedup_cache_;
std::mutex RemoteCompileCoordinator::s_fw_gate_mutex_;
std::unordered_map<std::string, std::shared_future<void>> RemoteCompileCoordinator::s_fw_gate_;
std::unordered_map<uint64_t, jit_server::UploadFirmwareRequest> RemoteCompileCoordinator::s_fw_cache_;

RemoteCompileCoordinator::RemoteCompileCoordinator(
    std::vector<std::string> endpoints, ChipId device_build_id, uint64_t build_key) :
    endpoints_(std::move(endpoints)),
    dispatch_mode_(DispatchMode::STATIC_HASH),
    device_build_id_(device_build_id),
    build_key_(build_key) {
    TT_FATAL(!endpoints_.empty(), "RemoteCompileCoordinator requires at least one endpoint");
}

RemoteCompileCoordinator::RemoteCompileCoordinator(
    std::string broker_endpoint, ChipId device_build_id, uint64_t build_key) :
    broker_endpoint_(std::move(broker_endpoint)),
    dispatch_mode_(DispatchMode::BROKER),
    device_build_id_(device_build_id),
    build_key_(build_key),
    broker_session_(std::make_unique<jit_server::JitBrokerRpcSession>(broker_endpoint_)) {
    TT_FATAL(!broker_endpoint_.empty(), "RemoteCompileCoordinator broker endpoint must not be empty");
}

RemoteCompileCoordinator::~RemoteCompileCoordinator() = default;

void RemoteCompileCoordinator::submit(
    std::size_t kernel_hash, const std::function<KernelCompileDescriptor()>& make_descriptor) {
    // Check the cross-invocation dedup cache before generating files for this hash.
    std::shared_future<void> existing_future;
    std::shared_ptr<std::promise<void>> new_promise;
    {
        std::lock_guard lock(s_dedup_mutex_);
        auto it = s_dedup_cache_.find(kernel_hash);
        if (it != s_dedup_cache_.end()) {
            existing_future = it->second;
        } else {
            new_promise = std::make_shared<std::promise<void>>();
            existing_future = new_promise->get_future().share();
            s_dedup_cache_[kernel_hash] = existing_future;
        }
    }

    submitted_.push_back(existing_future);

    if (!new_promise) {
        return;
    }

    try {
        auto descriptor = make_descriptor();
        ready_to_dispatch_.push_back({std::move(descriptor), std::move(new_promise)});
    } catch (...) {
        {
            std::lock_guard lock(s_dedup_mutex_);
            s_dedup_cache_.erase(kernel_hash);
        }
        new_promise->set_exception(std::current_exception());
        throw;
    }
}

void RemoteCompileCoordinator::finish() {
    if (!ready_to_dispatch_.empty()) {
        try {
            if (dispatch_mode_ == DispatchMode::BROKER) {
                dispatch_via_broker();
            } else {
                dispatch_static_hash();
            }
        } catch (...) {
            fail_unfinished_dispatch(std::current_exception());
            throw;
        }
    }

    for (auto& [endpoint, pending] : pending_by_endpoint_) {
        if (pending.empty()) {
            continue;
        }

        try {
            TT_FATAL(
                sessions_by_endpoint_.contains(endpoint) && sessions_by_endpoint_[endpoint] != nullptr,
                "Internal error: missing transport session for endpoint {}",
                endpoint);

            auto responses = sessions_by_endpoint_[endpoint]->wait_all();
            TT_FATAL(
                responses.size() == pending.size(),
                "Response count mismatch for endpoint {}: expected {} got {}",
                endpoint,
                pending.size(),
                responses.size());

            for (std::size_t i = 0; i < pending.size(); ++i) {
                auto& resp = responses[i];
                auto& pend = pending[i];
                TT_FATAL(
                    resp.success,
                    "Remote JIT compile failed for kernel {} (endpoint {}): {}",
                    pend.descriptor.request.kernel_name,
                    endpoint,
                    resp.error_message);
                TT_FATAL(
                    resp.elf_blobs.size() == pend.descriptor.expected_elf_paths.size(),
                    "Expected {} ELF blobs but got {} for kernel {}",
                    pend.descriptor.expected_elf_paths.size(),
                    resp.elf_blobs.size(),
                    pend.descriptor.request.kernel_name);

                for (std::size_t j = 0; j < pend.descriptor.expected_elf_paths.size(); ++j) {
                    write_elf_blob(pend.descriptor.expected_elf_paths[j], resp.elf_blobs[j]);
                }

                pend.dedup_promise->set_value();
                pend.dedup_promise.reset();
            }
        } catch (...) {
            fail_unfinished_dispatch(std::current_exception());
            throw;
        }
    }

    for (auto& future : submitted_) {
        future.get();
    }

    submitted_.clear();
    ready_to_dispatch_.clear();
    pending_by_endpoint_.clear();
}

void RemoteCompileCoordinator::fail_unfinished_dispatch(const std::exception_ptr& ex) {
    std::lock_guard lock(s_dedup_mutex_);
    auto reject = [&](PendingKernel& pending) {
        if (!pending.dedup_promise) {
            return;
        }
        s_dedup_cache_.erase(pending.descriptor.kernel_hash);
        try {
            pending.dedup_promise->set_exception(ex);
        } catch (const std::future_error& ex) {
            // Promise already fulfilled/rejected by another path.
            (void)ex;
        }
        pending.dedup_promise.reset();
    };

    for (auto& pending : ready_to_dispatch_) {
        reject(pending);
    }
    for (auto& [endpoint, pending] : pending_by_endpoint_) {
        (void)endpoint;
        for (auto& item : pending) {
            reject(item);
        }
    }
}

void RemoteCompileCoordinator::dispatch_static_hash() {
    for (auto& pending : ready_to_dispatch_) {
        const std::string& endpoint = endpoints_[pending.descriptor.kernel_hash % endpoints_.size()];
        ensure_firmware_uploaded_static(endpoint);
        ensure_session(endpoint).send(pending.descriptor.request);
        pending_by_endpoint_[endpoint].push_back(std::move(pending));
    }
    ready_to_dispatch_.clear();
}

void RemoteCompileCoordinator::dispatch_via_broker() {
    TT_FATAL(broker_session_ != nullptr, "Internal error: broker session not initialized");
    jit_server::BrokerAssignRequest request;
    request.build_key = build_key_;
    request.kernel_keys.reserve(ready_to_dispatch_.size());
    for (const auto& pending : ready_to_dispatch_) {
        request.kernel_keys.push_back(pending.descriptor.request.kernel_name);
    }

    auto response = broker_session_->assign(request);
    TT_FATAL(
        response.assignments.size() == ready_to_dispatch_.size(),
        "Broker assignments mismatch: expected {} got {}",
        ready_to_dispatch_.size(),
        response.assignments.size());

    std::unordered_set<std::string> needs_firmware;
    for (std::size_t i = 0; i < ready_to_dispatch_.size(); ++i) {
        auto& pending = ready_to_dispatch_[i];
        const auto& assignment = response.assignments[i];
        pending.descriptor.request.handle = assignment.handle;
        if (assignment.firmware_state == jit_server::FirmwareState::ABSENT) {
            needs_firmware.insert(assignment.server_endpoint);
        }
    }

    ensure_firmware_uploaded_via_broker(needs_firmware);

    for (std::size_t i = 0; i < ready_to_dispatch_.size(); ++i) {
        auto& pending = ready_to_dispatch_[i];
        const auto& endpoint = response.assignments[i].server_endpoint;
        ensure_session(endpoint).send(pending.descriptor.request);
        pending_by_endpoint_[endpoint].push_back(std::move(pending));
    }
    ready_to_dispatch_.clear();
}

jit_server::JitCompileRpcSession& RemoteCompileCoordinator::ensure_session(const std::string& endpoint) {
    auto it = sessions_by_endpoint_.find(endpoint);
    if (it == sessions_by_endpoint_.end()) {
        it =
            sessions_by_endpoint_.emplace(endpoint, std::make_unique<jit_server::JitCompileRpcSession>(endpoint)).first;
    }
    return *it->second;
}

const jit_server::UploadFirmwareRequest& RemoteCompileCoordinator::get_firmware_request() {
    std::lock_guard lock(s_fw_gate_mutex_);
    auto it = s_fw_cache_.find(build_key_);
    if (it == s_fw_cache_.end()) {
        const auto& dev_env = BuildEnvManager::get_instance().get_device_build_env(device_build_id_);

        jit_server::UploadFirmwareRequest req;
        req.build_key = build_key_;
        for (const auto& fw_state : dev_env.firmware_build_states) {
            const auto& fw_path = fw_state.get_weakened_firmware_name();
            if (!fs::exists(fw_path)) {
                continue;
            }
            jit_server::FirmwareArtifact artifact;
            artifact.target_name = fw_state.get_target_name();
            artifact.file_name = fs::path(fw_path).filename().string();
            artifact.is_kernel_object = fw_state.get_firmware_is_kernel_object();
            artifact.data = tt::jit_build::utils::read_file_bytes(fw_path);
            req.artifacts.push_back(std::move(artifact));
        }
        it = s_fw_cache_.emplace(build_key_, std::move(req)).first;
    }
    return it->second;
}

void RemoteCompileCoordinator::ensure_firmware_uploaded_static(const std::string& endpoint) {
    const std::string gate_key = endpoint + ":" + std::to_string(build_key_);

    std::shared_future<void> existing_future;
    std::shared_ptr<std::promise<void>> gate_promise;
    {
        std::lock_guard lock(s_fw_gate_mutex_);
        auto it = s_fw_gate_.find(gate_key);
        if (it != s_fw_gate_.end()) {
            existing_future = it->second;
        } else {
            gate_promise = std::make_shared<std::promise<void>>();
            existing_future = gate_promise->get_future().share();
            s_fw_gate_[gate_key] = existing_future;
        }
    }

    if (!gate_promise) {
        existing_future.get();
        return;
    }

    try {
        const auto& fw_request = get_firmware_request();
        auto& session = ensure_session(endpoint);
        auto response = session.upload_firmware(fw_request);
        TT_FATAL(response.success, "Firmware upload failed for endpoint {}: {}", endpoint, response.error_message);
        gate_promise->set_value();
    } catch (...) {
        {
            std::lock_guard lock(s_fw_gate_mutex_);
            s_fw_gate_.erase(gate_key);
        }
        gate_promise->set_exception(std::current_exception());
        throw;
    }
}

void RemoteCompileCoordinator::ensure_firmware_uploaded_via_broker(const std::unordered_set<std::string>& endpoints) {
    TT_FATAL(broker_session_ != nullptr, "Internal error: broker session not initialized");
    if (endpoints.empty()) {
        return;
    }

    const auto& fw_request = get_firmware_request();
    for (const auto& endpoint : endpoints) {
        const auto action = broker_session_->claim_firmware_upload(build_key_, endpoint);
        if (action == jit_server::FirmwareUploadAction::SKIP_ALREADY_PRESENT) {
            continue;
        }
        if (action == jit_server::FirmwareUploadAction::WAIT_FOR_OTHER) {
            broker_session_->wait_firmware_ready(build_key_, endpoint);
            continue;
        }

        TT_FATAL(action == jit_server::FirmwareUploadAction::YOU_UPLOAD, "Unexpected firmware upload action");
        try {
            auto& session = ensure_session(endpoint);
            auto response = session.upload_firmware(fw_request);
            TT_FATAL(response.success, "Firmware upload failed for endpoint {}: {}", endpoint, response.error_message);
            broker_session_->release_firmware_upload(build_key_, endpoint, true);
        } catch (...) {
            broker_session_->release_firmware_upload(build_key_, endpoint, false);
            throw;
        }
    }
}

void RemoteCompileCoordinator::write_elf_blob(const std::string& path, const jit_server::ElfBlob& blob) {
    fs::create_directories(fs::path(path).parent_path());
    tt::jit_build::utils::FileRenamer tmp(path);
    std::ofstream elf_file(tmp.path(), std::ios::binary);
    TT_FATAL(elf_file.is_open(), "Cannot write ELF to {}", tmp.path());
    elf_file.write(reinterpret_cast<const char*>(blob.data.data()), static_cast<std::streamsize>(blob.data.size()));
}

}  // namespace tt::tt_metal
