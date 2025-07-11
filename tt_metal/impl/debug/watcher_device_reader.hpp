// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <core_coord.hpp>
// FIXME: ARCH_NAME specific, needed for several pointer types here
#include "dev_msgs.h"
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <umd/device/tt_soc_descriptor.h>
#include <umd/device/types/cluster_descriptor_types.h>

namespace tt::tt_metal {

constexpr uint64_t DEBUG_SANITIZE_NOC_SENTINEL_OK_64 = 0xbadabadabadabada;
constexpr uint32_t DEBUG_SANITIZE_NOC_SENTINEL_OK_32 = 0xbadabada;
constexpr uint16_t DEBUG_SANITIZE_NOC_SENTINEL_OK_16 = 0xbada;
constexpr uint8_t DEBUG_SANITIZE_NOC_SENTINEL_OK_8 = 0xda;

// Struct containing relevant info for stack usage
struct stack_usage_info_t {
    CoreDescriptor core;
    uint16_t stack_free = uint16_t(~0);
    uint16_t kernel_id;
};

class WatcherDeviceReader {
public:
    WatcherDeviceReader(FILE* f, chip_id_t device_id, const std::vector<std::string>& kernel_names);
    ~WatcherDeviceReader();
    void Dump(FILE* file = nullptr);

private:
    // Functions for dumping each watcher feature to the log
    void DumpCore(CoreDescriptor& logical_core, bool is_active_eth_core);
    void DumpL1Status(CoreDescriptor& core, const launch_msg_t* launch_msg);
    void DumpNocSanitizeStatus(
        CoreDescriptor& core, const std::string& core_str, const mailboxes_t* mbox_data, int noc);
    void DumpAssertStatus(CoreDescriptor& core, const std::string& core_str, const mailboxes_t* mbox_data);
    void DumpAssertTrippedDetails(CoreDescriptor& core, const std::string& error_msg, const mailboxes_t* mbox_data);
    void DumpPauseStatus(CoreDescriptor& core, const std::string& core_str, const mailboxes_t* mbox_data);
    void DumpRingBuffer(CoreDescriptor& core, const mailboxes_t* mbox_data, bool to_stdout);
    void DumpRunState(CoreDescriptor& core, const launch_msg_t* launch_msg, uint32_t state);
    void DumpLaunchMessage(CoreDescriptor& core, const mailboxes_t* mbox_data);
    void DumpWaypoints(CoreDescriptor& core, const mailboxes_t* mbox_data, bool to_stdout);
    void DumpSyncRegs(CoreDescriptor& core);
    void DumpStackUsage(CoreDescriptor& core, const mailboxes_t* mbox_data);
    void ValidateKernelIDs(CoreDescriptor& core, const launch_msg_t* launch);

    // Helper functions
    void LogRunningKernels(CoreDescriptor& core, const launch_msg_t* launch_msg);
    std::string GetKernelName(CoreDescriptor& core, const launch_msg_t* launch_msg, uint32_t type);

    FILE* f;
    chip_id_t device_id;
    const std::vector<std::string>& kernel_names;

    // Information that needs to be kept around on a per-dump basis
    std::set<std::pair<CoreCoord, riscv_id_t>> paused_cores;
    std::map<riscv_id_t, stack_usage_info_t> highest_stack_usage;
    std::map<int, bool> used_kernel_names;
    std::map<CoreCoord, uint32_t> logical_core_to_eth_link_retraining_count;
};

}  // namespace tt::tt_metal
