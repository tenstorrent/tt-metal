// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <kj/async.h>

#include "impl/jit_server/broker.capnp.h"
#include "impl/jit_server/jit_broker_types.hpp"

namespace tt::tt_metal::jit_server {

class JitBrokerService final : public broker_rpc::JitDispatchBroker::Server {
public:
    kj::Promise<void> assign(AssignContext context) override;
    kj::Promise<void> claimFirmwareUpload(ClaimFirmwareUploadContext context) override;
    kj::Promise<void> releaseFirmwareUpload(ReleaseFirmwareUploadContext context) override;
    kj::Promise<void> waitFirmwareReady(WaitFirmwareReadyContext context) override;
    kj::Promise<void> registerServer(RegisterServerContext context) override;
    kj::Promise<void> reportCacheState(ReportCacheStateContext context) override;
    kj::Promise<void> release(ReleaseContext context) override;

private:
    std::string pick_least_loaded_server() const;
    void notify_firmware_waiters(const FirmwareKey& key);

    std::unordered_map<KernelKey, std::string, KernelKeyHash> cache_dir_;
    std::unordered_map<KernelKey, std::string, KernelKeyHash> pending_;
    std::unordered_map<std::uint64_t, std::unordered_set<std::string>> firmware_have_;
    std::unordered_map<FirmwareKey, std::vector<kj::Own<kj::PromiseFulfiller<void>>>, FirmwareKeyHash>
        firmware_waiters_;
    std::unordered_set<FirmwareKey, FirmwareKeyHash> firmware_pending_;
    std::unordered_map<std::string, std::uint64_t> load_;
    std::vector<std::string> known_servers_;
    std::unordered_map<std::uint64_t, std::pair<std::string, KernelKey>> issued_handles_;
    std::uint64_t next_handle_ = 1;
};

}  // namespace tt::tt_metal::jit_server
