// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <unordered_set>
#include <vector>

#include "core_coord.hpp"
#include <tt_stl/span.hpp>
// clang-format off
#include "hal.hpp"
#include "tt_memory.h"
#include <umd/device/types/cluster_descriptor_types.hpp>
#include <umd/device/types/xy_pair.hpp>

// clang-format on

// llrt = lower-level runtime
namespace tt::llrt {

// Return a reference to a potentially shared binary image.
// The images are cached by path name only.
const ll_api::memory& get_risc_binary(
    const std::string& path,
    ll_api::memory::Loading loading = ll_api::memory::Loading::DISCRETE,
    const std::function<void(ll_api::memory&)>& update_callback = nullptr);

CoreCoord logical_core_from_ethernet_core(ChipId chip_id, CoreCoord& ethernet_core);

tt_metal::HalProgrammableCoreType get_core_type(ChipId chip_id, const CoreCoord& virtual_core);

void send_reset_go_signal(ChipId chip, const CoreCoord& virtual_core);

void write_launch_msg_to_core(
    ChipId chip,
    CoreCoord core,
    tt_metal::dev_msgs::launch_msg_t::View msg,
    tt_metal::dev_msgs::go_msg_t::ConstView go_msg,
    bool send_go = true);

bool test_load_write_read_risc_binary(
    const ll_api::memory& mem,
    ChipId chip_id,
    const CoreCoord& core,
    uint32_t core_type_idx,
    uint32_t processor_class_idx,
    uint32_t processor_type_idx);
void write_binary_to_address(const ll_api::memory& mem, ChipId chip_id, const CoreCoord& core, uint32_t address);

namespace internal_ {

void wait_until_cores_done(
    ChipId device_id, int run_state, std::unordered_set<CoreCoord>& not_done_phys_cores, int timeout_ms = 0);

// Send a message to the ethernet firmware mailbox, if supported
// Possible message types can be queried from the Hal. See tt::tt_metal::FWMailboxMsg
// Maximum number of args depends on the architecture. Args not provided will be set to zero.
void send_msg_to_eth_mailbox(
    ChipId device_id,
    const CoreCoord& virtual_core,
    tt_metal::FWMailboxMsg msg_type,
    int mailbox_index,
    std::vector<uint32_t> args,
    bool wait_for_ack = true,
    int timeout_ms = 10000);

// Return to base firmware and wait for a heartbeat from the active ethernet core, if supported
// Default timeout time empirically chosen to be 10 seconds to avoid timeouts
void return_to_base_firmware_and_wait_for_heartbeat(
    ChipId device_id, const CoreCoord& virtual_core, int timeout_ms = 10000);

void set_metal_eth_fw_run_flag(ChipId device_id, const CoreCoord& virtual_core, bool enable);

}  // namespace internal_

}  // namespace tt::llrt
