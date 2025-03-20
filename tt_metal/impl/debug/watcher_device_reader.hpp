// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <core_coord.hpp>
#include "umd/device/tt_soc_descriptor.h"
#include <hal.hpp>

// FIXME: ARCH_NAME specific, needed for several pointer types here
#include <dev_msgs.h>

namespace tt::watcher {

constexpr uint64_t DEBUG_SANITIZE_NOC_SENTINEL_OK_64 = 0xbadabadabadabada;
constexpr uint32_t DEBUG_SANITIZE_NOC_SENTINEL_OK_32 = 0xbadabada;
constexpr uint16_t DEBUG_SANITIZE_NOC_SENTINEL_OK_16 = 0xbada;
constexpr uint8_t DEBUG_SANITIZE_NOC_SENTINEL_OK_8 = 0xda;

// Struct containing relevant info for stack usage
typedef struct {
    CoreDescriptor core;
    uint16_t stack_usage;
    uint16_t kernel_id;
} stack_usage_info_t;

class WatcherDeviceReader {
public:
    WatcherDeviceReader(
        FILE* f,
        chip_id_t device_id,
        std::vector<std::string>& kernel_names,
        void (*set_watcher_exception_message)(const std::string&));
    ~WatcherDeviceReader();
    void Dump(FILE* file = nullptr);

private:
    // Functions for dumping each watcher feature to the log
    void DumpCore(CoreDescriptor& logical_core, bool is_active_eth_core);
    void DumpL1Status(CoreDescriptor& core, const launch_msg_t* launch_msg);
    void DumpNocSanitizeStatus(
        CoreDescriptor& core, const std::string& core_str, const mailboxes_t* mbox_data, int noc);
    void DumpAssertStatus(CoreDescriptor& core, const std::string& core_str, const mailboxes_t* mbox_data);
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
    std::vector<std::string>& kernel_names;
    void (*set_watcher_exception_message)(const std::string&);

    // Information that needs to be kept around on a per-dump basis
    std::set<std::pair<CoreCoord, riscv_id_t>> paused_cores;
    std::map<riscv_id_t, stack_usage_info_t> highest_stack_usage;
    std::map<int, bool> used_kernel_names;
    std::map<CoreCoord, uint32_t> logical_core_to_eth_link_retraining_count;
};

}  // namespace tt::watcher
