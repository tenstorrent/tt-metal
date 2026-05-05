// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "impl/jit_server/jit_compile_rpc_client.hpp"
#include "impl/jit_server/types.hpp"
#include <umd/device/types/cluster_descriptor_types.hpp>

namespace tt::tt_metal {

// Input descriptor for one kernel to be compiled remotely.
// Built by program.cpp during kernel prep; consumed by the coordinator.
struct KernelCompileDescriptor {
    std::size_t kernel_hash = 0;
    jit_server::CompileRequest request;
    // Local file paths where the resulting ELF blobs should be written.
    std::vector<std::string> expected_elf_paths;
};

// Orchestrates remote JIT compilation for a batch of kernels.
//
// One instance is created per ProgramImpl::compile invocation. It handles:
//   - Cross-invocation in-flight dedup (same kernel hash -> compile once)
//   - Upload-once firmware gate per (endpoint, build_key)
//   - Endpoint sharding (kernel_hash % N)
//   - Pipelined RPC sends during submit, batched response collection in finish
//   - ELF blob materialization to disk
//
// program.cpp prepares descriptors; this class owns all mechanics.
class RemoteCompileCoordinator {
public:
    RemoteCompileCoordinator(std::vector<std::string> endpoints, ChipId device_build_id, uint64_t build_key);
    ~RemoteCompileCoordinator();

    RemoteCompileCoordinator(const RemoteCompileCoordinator&) = delete;
    RemoteCompileCoordinator& operator=(const RemoteCompileCoordinator&) = delete;

    // Submit a kernel for remote compilation.
    // Deduplicates against prior submissions (even from previous batches in this process).
    // Sends the RPC immediately for new kernels; deduped kernels just record the future.
    // make_descriptor() is invoked only by the owning caller.
    void submit(std::size_t kernel_hash, const std::function<KernelCompileDescriptor()>& make_descriptor);

    // Collect all outstanding RPC responses, write ELF blobs to disk,
    // and wait for any dedup'd kernels to complete.
    void finish();

private:
    const jit_server::UploadFirmwareRequest& get_firmware_request();
    void ensure_firmware_uploaded(std::size_t endpoint_index);
    void ensure_session(std::size_t endpoint_index);
    void write_elf_blob(const std::string& path, const jit_server::ElfBlob& blob);

    // -- Configuration (immutable after construction) --
    std::vector<std::string> endpoints_;
    ChipId device_build_id_;
    uint64_t build_key_;

    // -- Per-batch state (between submit/finish) --
    // submit() is called sequentially from the program compile loop;
    // finish() runs after all submits complete.  No mutex needed.
    struct PendingKernel {
        KernelCompileDescriptor descriptor;
        std::shared_ptr<std::promise<void>> dedup_promise;
    };
    std::vector<std::unique_ptr<jit_server::JitCompileRpcSession>> sessions_;
    std::vector<std::vector<PendingKernel>> pending_by_endpoint_;

    std::vector<std::shared_future<void>> submitted_;

    // -- Global state (persists across batches within the process) --
    static std::mutex s_dedup_mutex_;
    static std::unordered_map<std::size_t, std::shared_future<void>> s_dedup_cache_;

    static std::mutex s_fw_gate_mutex_;
    static std::unordered_map<std::string, std::shared_future<void>> s_fw_gate_;
    // Firmware request cache: built at most once per build_key.
    static std::unordered_map<uint64_t, jit_server::UploadFirmwareRequest> s_fw_cache_;
};

}  // namespace tt::tt_metal
