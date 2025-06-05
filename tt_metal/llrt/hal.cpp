// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "llrt/hal.hpp"

#include <assert.hpp>

#include "hal_types.hpp"
#include <umd/device/types/arch.h>

namespace tt {

namespace tt_metal {

// Hal Constructor determines the platform architecture by using UMD
// Once it knows the architecture it can self initialize architecture specific memory maps
Hal::Hal(tt::ARCH arch, bool is_base_routing_fw_enabled) : arch_(arch) {
    switch (this->arch_) {
        case tt::ARCH::WORMHOLE_B0: initialize_wh(is_base_routing_fw_enabled); break;

        case tt::ARCH::BLACKHOLE: initialize_bh(); break;

        case tt::ARCH::QUASAR: TT_THROW("HAL doesn't support Quasar"); break;

        default: /*TT_THROW("Unsupported arch for HAL")*/; break;
    }
}

uint32_t Hal::get_programmable_core_type_index(HalProgrammableCoreType programmable_core_type_index) const {
    uint32_t index = static_cast<uint32_t>(programmable_core_type_index);

    // TODO: this assumes unused entries occur at the end
    // Assumes unused indices go at the end
    if (index >= core_info_.size()) {
        return -1;
    } else {
        return index;
    }
}

uint32_t Hal::get_num_risc_processors() const {
    uint32_t num_riscs = 0;
    for (uint32_t core_idx = 0; core_idx < core_info_.size(); core_idx++) {
        uint32_t num_processor_classes = core_info_[core_idx].get_processor_classes_count();
        for (uint32_t processor_class_idx = 0; processor_class_idx < num_processor_classes; processor_class_idx++) {
            num_riscs += core_info_[core_idx].get_processor_types_count(processor_class_idx);
        }
    }
    return num_riscs;
}

HalCoreInfoType::HalCoreInfoType(
    HalProgrammableCoreType programmable_core_type,
    CoreType core_type,
    const std::vector<std::vector<HalJitBuildConfig>>& processor_classes,
    const std::vector<DeviceAddr>& mem_map_bases,
    const std::vector<uint32_t>& mem_map_sizes,
    const std::vector<uint32_t>& eth_fw_mailbox_msgs,
    bool supports_cbs,
    bool supports_receiving_multicast_cmds) :
    programmable_core_type_(programmable_core_type),
    core_type_(core_type),
    processor_classes_(processor_classes),
    mem_map_bases_(mem_map_bases),
    mem_map_sizes_(mem_map_sizes),
    eth_fw_mailbox_msgs_{eth_fw_mailbox_msgs},
    supports_cbs_(supports_cbs),
    supports_receiving_multicast_cmds_(supports_receiving_multicast_cmds) {}

uint32_t generate_risc_startup_addr(uint32_t firmware_base) {
    // Options for handling brisc fw not starting at mem[0]:
    // 1) Program the register for the start address out of reset - no reset PC register on GS/WH/BH
    // 2) Encode a jump in crt0 for mem[0]
    // 3) Write the jump to mem[0] here
    // This does #3.  #1 may be best, #2 gets messy (elf files
    // drop any section before .init, crt0 needs ifdefs, etc)
    constexpr uint32_t jal_opcode = 0x6f;
    constexpr uint32_t jal_max_offset = 0x0007ffff;
    uint32_t opcode = jal_opcode;
    TT_FATAL(
        firmware_base < jal_max_offset,
        "Base FW address {} should be below JAL max offset {}",
        firmware_base,
        jal_max_offset);
    // See riscv spec for offset encoding below
    uint32_t jal_offset_bit_20 = 0;
    uint32_t jal_offset_bits_10_to_1 = (firmware_base & 0x7fe) << 20;
    uint32_t jal_offset_bit_11 = (firmware_base & 0x800) << 9;
    uint32_t jal_offset_bits_19_to_12 = (firmware_base & 0xff000) << 0;
    uint32_t jal_offset =
        jal_offset_bit_20 |
        jal_offset_bits_10_to_1 |
        jal_offset_bit_11 |
        jal_offset_bits_19_to_12;

    return jal_offset | opcode;
}

}  // namespace tt_metal
}  // namespace tt
