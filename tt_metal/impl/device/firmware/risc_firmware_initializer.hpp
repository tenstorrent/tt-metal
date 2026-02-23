// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "firmware_initializer.hpp"
#include "llrt/hal/generated/dev_msgs.hpp"
#include "umd/device/types/cluster_descriptor_types.hpp"

namespace tt::tt_fabric {
class ControlPlane;
}  // namespace tt::tt_fabric

namespace tt::tt_metal {

class dispatch_core_manager;

class RiscFirmwareInitializer final : public FirmwareInitializer {
public:
    static constexpr InitializerKey key = InitializerKey::Risc;

    ~RiscFirmwareInitializer() override;

    RiscFirmwareInitializer(
        std::shared_ptr<const ContextDescriptor> descriptor,
        tt_fabric::ControlPlane& control_plane,
        dispatch_core_manager& dispatch_core_manager,
        size_t fw_compile_hash);

    void init(const std::vector<Device*>& devices, const std::unordered_set<InitializerKey>& init_done) override;
    void init_by_device_ids(const std::set<tt::ChipId>& device_ids) override;
    void configure() override;
    void teardown() override;
    bool is_initialized() const override;

    // Place cores on the device into a reset state except for the cores in the ignore_virtual_cores set.
    // Used by MetalContext during teardown to put firmware into reset.
    void assert_cores(tt::ChipId device_id, std::unordered_set<CoreCoord>& ignore_virtual_cores);

    // If simulator is enabled, send exit signal to active ethernet cores and wait for them to stop.
    // Used by MetalContext during teardown (same order as init).
    void teardown_simulator_ethernet_cores();

    void build_risc_fw(const std::set<tt::ChipId>& device_ids);
    void launch_risc_fw(const std::set<tt::ChipId>& device_ids);

private:
    void clear_l1_state(tt::ChipId device_id);
    void clear_dram_state(tt::ChipId device_id);
    void clear_launch_messages_on_eth_cores(tt::ChipId device_id);
    // Take the cores out of reset state. This should be called after setting the correct program counter for execution.
    void reset_cores(tt::ChipId device_id);

    void assert_active_ethernet_cores_to_reset(tt::ChipId device_id);
    void assert_tensix_workers_impl(tt::ChipId device_id, const std::unordered_set<CoreCoord>* ignore_virtual_cores);
    void assert_inactive_ethernet_cores(tt::ChipId device_id);

    CoreCoord virtual_noc0_coordinate(tt::ChipId device_id, uint8_t noc_index, CoreCoord coord);
    void generate_device_bank_to_noc_tables(tt::ChipId device_id);
    void generate_worker_logical_to_virtual_map(tt::ChipId device_id);
    void initialize_device_bank_to_noc_tables(
        tt::ChipId device_id,
        const HalProgrammableCoreType& core_type,
        CoreCoord virtual_core,
        std::optional<CoreCoord> end_core);
    void initialize_worker_logical_to_virtual_tables(
        tt::ChipId device_id, const HalProgrammableCoreType& core_type, CoreCoord start_core, CoreCoord end_core);
    void initialize_firmware(
        tt::ChipId device_id,
        const HalProgrammableCoreType& core_type,
        CoreCoord virtual_core,
        dev_msgs::launch_msg_t::View launch_msg,
        dev_msgs::go_msg_t::ConstView go_msg,
        std::optional<CoreCoord> end_core = std::nullopt);
    void initialize_and_launch_firmware(tt::ChipId device_id);
    dev_msgs::core_info_msg_t populate_core_info_msg(
        tt::ChipId device_id, HalProgrammableCoreType programmable_core_type) const;
    uint32_t get_active_erisc_launch_flag_addr();
    bool erisc_app_still_running(tt::ChipId device_id, CoreCoord virtual_core);
    void erisc_send_exit_signal(tt::ChipId device_id, CoreCoord virtual_core, bool is_idle_eth);

    tt_fabric::ControlPlane* control_plane_;
    dispatch_core_manager& dispatch_core_manager_;
    [[maybe_unused]] size_t fw_compile_hash_;
    uint8_t num_hw_cqs_;
    size_t worker_l1_unreserved_start_;

    std::unordered_map<tt::ChipId, std::vector<int32_t>> dram_bank_offset_map_;
    std::unordered_map<tt::ChipId, std::vector<int32_t>> l1_bank_offset_map_;
    std::unordered_map<tt::ChipId, std::vector<uint16_t>> dram_bank_to_noc_xy_;
    std::unordered_map<tt::ChipId, std::vector<uint16_t>> l1_bank_to_noc_xy_;
    std::unordered_map<tt::ChipId, std::vector<uint8_t>> worker_logical_col_to_virtual_col_;
    std::unordered_map<tt::ChipId, std::vector<uint8_t>> worker_logical_row_to_virtual_row_;

    bool initialized_ = false;
};

}  // namespace tt::tt_metal
