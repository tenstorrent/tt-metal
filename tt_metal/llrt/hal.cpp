// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "llrt/hal.hpp"

#include <assert.hpp>

#include <cstdint>
#include <enchantum/iostream.hpp>

#include "hal/generated/dev_msgs.hpp"
#include "hal_types.hpp"
#include <umd/device/types/arch.h>

namespace tt {

namespace tt_metal {

std::ostream& operator<<(std::ostream& os, const HalProcessorIdentifier& processor) {
    using enchantum::iostream_operators::operator<<;
    return os << processor.core_type << "_" << processor.processor_class << "_" << processor.processor_type;
}

bool operator<(const HalProcessorIdentifier& lhs, const HalProcessorIdentifier& rhs) {
    return std::tie(lhs.core_type, lhs.processor_class, lhs.processor_type) <
           std::tie(rhs.core_type, rhs.processor_class, rhs.processor_type);
}

bool operator==(const HalProcessorIdentifier& lhs, const HalProcessorIdentifier& rhs) {
    return std::tie(lhs.core_type, lhs.processor_class, lhs.processor_type) ==
           std::tie(rhs.core_type, rhs.processor_class, rhs.processor_type);
}

// Hal Constructor determines the platform architecture by using UMD
// Once it knows the architecture it can self initialize architecture specific memory maps
Hal::Hal(tt::ARCH arch, bool is_base_routing_fw_enabled) : arch_(arch) {
    switch (this->arch_) {
        case tt::ARCH::WORMHOLE_B0: initialize_wh(is_base_routing_fw_enabled); break;

        case tt::ARCH::QUASAR:  // TODO create quasar hal
        case tt::ARCH::BLACKHOLE: initialize_bh(); break;

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

uint32_t Hal::get_total_num_risc_processors() const {
    uint32_t num_riscs = 0;
    for (uint32_t core_idx = 0; core_idx < core_info_.size(); core_idx++) {
        num_riscs += this->get_num_risc_processors(this->core_info_[core_idx].programmable_core_type_);
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
    bool supports_receiving_multicast_cmds,
    dev_msgs::Factory dev_msgs_factory) :
    programmable_core_type_(programmable_core_type),
    core_type_(core_type),
    processor_classes_(processor_classes),
    mem_map_bases_(mem_map_bases),
    mem_map_sizes_(mem_map_sizes),
    eth_fw_mailbox_msgs_{eth_fw_mailbox_msgs},
    supports_cbs_(supports_cbs),
    supports_receiving_multicast_cmds_(supports_receiving_multicast_cmds),
    dev_msgs_factory_(std::move(dev_msgs_factory)) {}

uint32_t HalCoreInfoType::get_processor_index(
    HalProcessorClassType processor_class, uint32_t processor_type_idx) const {
    uint32_t processor_class_idx = utils::underlying_type<HalProcessorClassType>(processor_class);
    // TODO(HalProcessorClassType): fix this after DM0 and DM1 are the same processor class
    if (processor_class == HalProcessorClassType::DM) {
        TT_ASSERT(processor_type_idx < static_cast<uint32_t>(HalProcessorClassType::COMPUTE));
        processor_class_idx = processor_type_idx;
        processor_type_idx = 0;
    }
    uint32_t processor_index = 0;
    for (uint32_t i = 0; i < processor_class_idx; i++) {
        processor_index += this->get_processor_types_count(i);
    }
    TT_ASSERT(processor_type_idx < this->get_processor_types_count(processor_class_idx));
    return processor_index + processor_type_idx;
}

std::pair<HalProcessorClassType, uint32_t> HalCoreInfoType::get_processor_class_and_type_from_index(
    uint32_t processor_index) const {
    uint32_t processor_class_idx = 0;
    for (; processor_class_idx < this->processor_classes_.size(); processor_class_idx++) {
        auto processor_count = get_processor_types_count(processor_class_idx);
        if (processor_index < processor_count) {
            break;
        }
        processor_index -= processor_count;
    }
    TT_ASSERT(processor_class_idx < this->processor_classes_.size());
    // TODO(HalProcessorClassType): fix this after DM0 and DM1 are the same processor class
    if (processor_class_idx < static_cast<uint32_t>(HalProcessorClassType::COMPUTE)) {
        return {HalProcessorClassType::DM, processor_class_idx};
    }
    return {static_cast<HalProcessorClassType>(processor_class_idx), processor_index};
}

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

std::size_t std::hash<tt::tt_metal::HalProcessorIdentifier>::operator()(
    const tt::tt_metal::HalProcessorIdentifier& processor) const {
    auto hasher = std::hash<int>();
    std::size_t hash = 0;
    hash ^= hasher(static_cast<int>(processor.core_type));
    hash ^= hasher(static_cast<int>(processor.processor_class)) << 1;
    hash ^= hasher(processor.processor_type) << 2;
    return hash;
}
