// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
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

tt::tt_metal::CoreCoord logical_core_from_ethernet_core(ChipId chip_id, tt::tt_metal::CoreCoord& ethernet_core);

tt_metal::HalProgrammableCoreType get_core_type(ChipId chip_id, const tt::tt_metal::CoreCoord& virtual_core);

void send_reset_go_signal(ChipId chip, const tt::tt_metal::CoreCoord& virtual_core);

void write_launch_msg_to_core(
    ChipId chip,
    tt::tt_metal::CoreCoord core,
    tt_metal::dev_msgs::launch_msg_t::View msg,
    tt_metal::dev_msgs::go_msg_t::ConstView go_msg,
    bool send_go = true);

bool test_load_write_read_risc_binary(
    const ll_api::memory& mem,
    ChipId chip_id,
    const tt::tt_metal::CoreCoord& core,
    uint32_t core_type_idx,
    uint32_t processor_class_idx,
    uint32_t processor_type_idx);

bool test_load_multicast_write_risc_binary(
    const ll_api::memory& mem,
    tt::ChipId chip_id,
    const tt::tt_metal::CoreCoord& start_core,
    const tt::tt_metal::CoreCoord& end_core,
    uint32_t core_type_idx,
    uint32_t processor_class_idx,
    uint32_t processor_type_idx);

void write_binary_to_address(const ll_api::memory& mem, ChipId chip_id, const tt::tt_metal::CoreCoord& core, uint32_t address);

namespace internal_ {

void wait_until_cores_done(
    ChipId device_id, int run_state, std::unordered_set<tt::tt_metal::CoreCoord>& not_done_phys_cores, int timeout_ms = 0);

void wait_for_idle(ChipId device_id, const std::vector<std::vector<tt::tt_metal::CoreCoord>>& logical_cores);

// In test mode, return a watcher fault message if one was recorded, so host waits can unwind.
std::optional<std::string> get_watcher_error_message_in_test_mode(ChipId device_id);

// In test mode, throw if watcher detected a device-side fault so host waits can unwind.
void throw_if_watcher_tripped_in_test_mode(ChipId device_id);

// Send a message to the ethernet firmware mailbox, if supported
// Possible message types can be queried from the Hal. See tt::tt_metal::FWMailboxMsg
// Maximum number of args depends on the architecture. Args not provided will be set to zero.
void send_msg_to_eth_mailbox(
    ChipId device_id,
    const tt::tt_metal::CoreCoord& virtual_core,
    tt_metal::FWMailboxMsg msg_type,
    int mailbox_index,
    std::vector<uint32_t> args,
    bool wait_for_ack = true,
    int timeout_ms = 10000);

// Return to base firmware and wait for a heartbeat from the active ethernet core, if supported
// Default timeout time empirically chosen to be 20 seconds to avoid timeouts
void return_to_base_firmware_and_wait_for_heartbeat(
    ChipId device_id, const tt::tt_metal::CoreCoord& virtual_core, int timeout_ms = 20000);

void set_metal_eth_fw_run_flag(ChipId device_id, const tt::tt_metal::CoreCoord& virtual_core, bool enable);

}  // namespace internal_

}  // namespace tt::llrt
