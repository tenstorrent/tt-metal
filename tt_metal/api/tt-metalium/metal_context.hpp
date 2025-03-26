// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <dispatch_core_common.hpp>
#include <tt-metalium/core_descriptor.hpp>  // For chip_id_t
#include <tt-metalium/hal_types.hpp>        // For HalProgrammableCoreType
#include <tt-metalium/dev_msgs.h>           // For go_msg_t
#include <tt-metalium/allocator_types.hpp>  // For BankMapping

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tt::tt_metal {

class MetalContext {
public:
    MetalContext& operator=(const MetalContext&) = delete;
    MetalContext& operator=(MetalContext&& other) noexcept = delete;
    MetalContext(const MetalContext&) = delete;
    MetalContext(MetalContext&& other) noexcept = delete;

    static void initialize(
        const DispatchCoreConfig& dispatch_core_config, uint8_t num_hw_cqs, BankMapping l1_bank_remap) noexcept;
    static MetalContext& instance();

private:
    MetalContext(const DispatchCoreConfig& dispatch_core_config, uint8_t num_hw_cqs, BankMapping l1_bank_remap);
    ~MetalContext();

    // Functions to generate, and then write bank to noc tables, used by FW.
    void generate_device_bank_to_noc_tables(chip_id_t device_id);
    void initialize_device_bank_to_noc_tables(
        chip_id_t device_id, const HalProgrammableCoreType& core_type, CoreCoord virtual_core);

    void initialize_firmware(
        chip_id_t device_id,
        const HalProgrammableCoreType& core_type,
        CoreCoord virtual_core,
        launch_msg_t* launch_msg,
        go_msg_t* go_msg);
    void initialize_and_launch_firmware(chip_id_t device_id);

    uint8_t num_hw_cqs_;
    BankMapping l1_bank_remap_;
    DispatchCoreConfig dispatch_core_config_;

    // Used to track which FW has been build already
    std::unordered_set<uint32_t> firmware_built_keys_;

    // Written to device as part of FW init, device-specific
    std::unordered_map<chip_id_t, std::vector<int32_t>> dram_bank_offset_map_;
    std::unordered_map<chip_id_t, std::vector<int32_t>> l1_bank_offset_map_;
    std::unordered_map<chip_id_t, std::vector<uint16_t>> dram_bank_to_noc_xy_;
    std::unordered_map<chip_id_t, std::vector<uint16_t>> l1_bank_to_noc_xy_;

    static MetalContext* _inst;
};

}  // namespace tt::tt_metal
