// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/jit_server/remote_compile_coordinator.hpp"

#include <filesystem>
#include <fstream>
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
    device_build_id_(device_build_id),
    build_key_(build_key),
    sessions_(endpoints_.size()),
    pending_by_endpoint_(endpoints_.size()) {
    TT_FATAL(!endpoints_.empty(), "RemoteCompileCoordinator requires at least one endpoint");
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

    // This kernel is new — assign endpoint and pipeline the send.
    const std::size_t ep_idx = kernel_hash % endpoints_.size();

    try {
        auto descriptor = make_descriptor();
        ensure_session(ep_idx);
        ensure_firmware_uploaded(ep_idx);

        sessions_[ep_idx]->send(descriptor.request);
        pending_by_endpoint_[ep_idx].push_back({std::move(descriptor), std::move(new_promise)});
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
    // Phase 1: Collect responses per endpoint, write ELFs and markers, fulfill dedup promises.
    for (std::size_t ep_idx = 0; ep_idx < pending_by_endpoint_.size(); ++ep_idx) {
        auto& pending = pending_by_endpoint_[ep_idx];
        if (pending.empty()) {
            continue;
        }

        try {
            TT_FATAL(
                sessions_[ep_idx] != nullptr,
                "Internal error: missing transport session for endpoint {}",
                endpoints_[ep_idx]);

            auto responses = sessions_[ep_idx]->wait_all();
            TT_FATAL(
                responses.size() == pending.size(),
                "Response count mismatch for endpoint {}: expected {} got {}",
                endpoints_[ep_idx],
                pending.size(),
                responses.size());

            for (std::size_t i = 0; i < pending.size(); ++i) {
                auto& resp = responses[i];
                auto& pend = pending[i];
                TT_FATAL(
                    resp.success,
                    "Remote JIT compile failed for kernel {} (endpoint {}): {}",
                    pend.descriptor.request.kernel_name,
                    endpoints_[ep_idx],
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
            auto ex = std::current_exception();
            std::lock_guard lock(s_dedup_mutex_);
            for (auto& pend : pending) {
                if (pend.dedup_promise) {
                    s_dedup_cache_.erase(pend.descriptor.kernel_hash);
                    pend.dedup_promise->set_exception(ex);
                }
            }
            throw;
        }
    }

    // Phase 2: Wait for any dedup'd kernels fulfilled by other coordinator instances.
    for (auto& future : submitted_) {
        future.get();
    }

    // Clear per-batch state.
    submitted_.clear();
    for (auto& v : pending_by_endpoint_) {
        v.clear();
    }
}

void RemoteCompileCoordinator::ensure_session(std::size_t endpoint_index) {
    if (!sessions_[endpoint_index]) {
        sessions_[endpoint_index] = std::make_unique<jit_server::JitCompileRpcSession>(endpoints_[endpoint_index]);
    }
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

void RemoteCompileCoordinator::ensure_firmware_uploaded(std::size_t endpoint_index) {
    const std::string gate_key = endpoints_[endpoint_index] + ":" + std::to_string(build_key_);

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
        ensure_session(endpoint_index);
        auto response = sessions_[endpoint_index]->upload_firmware(fw_request);
        TT_FATAL(
            response.success,
            "Firmware upload failed for endpoint {}: {}",
            endpoints_[endpoint_index],
            response.error_message);
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

void RemoteCompileCoordinator::write_elf_blob(const std::string& path, const jit_server::ElfBlob& blob) {
    fs::create_directories(fs::path(path).parent_path());
    tt::jit_build::utils::FileRenamer tmp(path);
    std::ofstream elf_file(tmp.path(), std::ios::binary);
    TT_FATAL(elf_file.is_open(), "Cannot write ELF to {}", tmp.path());
    elf_file.write(reinterpret_cast<const char*>(blob.data.data()), static_cast<std::streamsize>(blob.data.size()));
}

}  // namespace tt::tt_metal
