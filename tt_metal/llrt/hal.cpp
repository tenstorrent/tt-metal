// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "llrt/hal.hpp"

#include <tt_stl/assert.hpp>

#include <cstdint>
#include <enchantum/iostream.hpp>

#include "hal_types.hpp"
#include <umd/device/types/arch.hpp>

namespace tt::tt_metal {

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
Hal::Hal(
    tt::ARCH arch,
    bool is_base_routing_fw_enabled,
    bool enable_2_erisc_mode,
    uint32_t profiler_dram_bank_size_per_risc_bytes) :
    arch_(arch) {
    switch (this->arch_) {
        case tt::ARCH::WORMHOLE_B0:
            initialize_wh(is_base_routing_fw_enabled, profiler_dram_bank_size_per_risc_bytes);
            break;

        case tt::ARCH::QUASAR: initialize_qa(profiler_dram_bank_size_per_risc_bytes); break;

        case tt::ARCH::BLACKHOLE: initialize_bh(enable_2_erisc_mode, profiler_dram_bank_size_per_risc_bytes); break;

        default: /*TT_THROW("Unsupported arch for HAL")*/; break;
    }
}

uint64_t Hal::get_pcie_addr_lower_bound() const { return pcie_addr_lower_bound_; }

uint64_t Hal::get_pcie_addr_upper_bound() const { return pcie_addr_upper_bound_; }

uint32_t Hal::get_programmable_core_type_index(HalProgrammableCoreType programmable_core_type_index) const {
    uint32_t index = static_cast<uint32_t>(programmable_core_type_index);

    // TODO: this assumes unused entries occur at the end
    // Assumes unused indices go at the end
    if (index >= core_info_.size()) {
        return -1;
    }
    return index;
}

uint32_t Hal::get_total_num_risc_processors() const {
    uint32_t num_riscs = 0;
    for (uint32_t core_idx = 0; core_idx < core_info_.size(); core_idx++) {
        num_riscs += this->get_num_risc_processors(this->core_info_[core_idx].programmable_core_type_);
    }
    return num_riscs;
}

uint32_t HalCoreInfoType::get_processor_index(
    HalProcessorClassType processor_class, uint32_t processor_type_idx) const {
    uint32_t processor_class_idx = ttsl::as_underlying_type<HalProcessorClassType>(processor_class);
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
    return {static_cast<HalProcessorClassType>(processor_class_idx), processor_index};
}

const std::string& HalCoreInfoType::get_processor_class_name(uint32_t processor_index, bool is_abbreviated) const {
    auto [processor_class, processor_type_idx] = get_processor_class_and_type_from_index(processor_index);
    uint32_t processor_class_idx = ttsl::as_underlying_type<HalProcessorClassType>(processor_class);
    TT_ASSERT(ttsl::as_underlying_type<HalProcessorClassType>(processor_class) < this->processor_classes_names_.size());
    if (is_abbreviated) {
        return this->processor_classes_names_[processor_class_idx][processor_type_idx].first;
    }
    return this->processor_classes_names_[processor_class_idx][processor_type_idx].second;
}

uint32_t HalCoreInfoType::get_processor_class_num_fw_binaries(uint32_t processor_class_idx) const {
    TT_ASSERT(processor_class_idx < this->processor_classes_num_fw_binaries_.size());
    return this->processor_classes_num_fw_binaries_[processor_class_idx];
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
    uint32_t jal_offset = jal_offset_bit_20 | jal_offset_bits_10_to_1 | jal_offset_bit_11 | jal_offset_bits_19_to_12;

    return jal_offset | opcode;
}

HalProcessorSet Hal::parse_processor_set_spec(std::string_view spec) const {
    HalProcessorSet set;

    // TODO: might need a new syntax for new architectures.
    // Current syntax hardcodes the RISC-V names for WH/BH.
    // Either keep this syntax but move it to hal/tt-1xx, and create the new one in hal/tt-2xx,
    // or break compatibility and use the new syntax for all architectures.
    if (spec.find("BR") != std::string_view::npos) {
        set.add(HalProgrammableCoreType::TENSIX, 0);
    }
    if (spec.find("NC") != std::string_view::npos) {
        set.add(HalProgrammableCoreType::TENSIX, 1);
    }
    if (spec.find("TR0") != std::string_view::npos) {
        set.add(HalProgrammableCoreType::TENSIX, 2);
    }
    if (spec.find("TR1") != std::string_view::npos) {
        set.add(HalProgrammableCoreType::TENSIX, 3);
    }
    if (spec.find("TR2") != std::string_view::npos) {
        set.add(HalProgrammableCoreType::TENSIX, 4);
    }
    if (spec.find("TR*") != std::string_view::npos) {
        set.add(HalProgrammableCoreType::TENSIX, 2);
        set.add(HalProgrammableCoreType::TENSIX, 3);
        set.add(HalProgrammableCoreType::TENSIX, 4);
    }
    if (spec.find("ER0") != std::string_view::npos) {
        set.add(HalProgrammableCoreType::ACTIVE_ETH, 0);
        set.add(HalProgrammableCoreType::IDLE_ETH, 0);
    }
    if (spec.find("ER1") != std::string_view::npos) {
        set.add(HalProgrammableCoreType::ACTIVE_ETH, 1);
        set.add(HalProgrammableCoreType::IDLE_ETH, 1);
    }
    if (spec.find("ER*") != std::string_view::npos) {
        set.add(HalProgrammableCoreType::ACTIVE_ETH, 0);
        set.add(HalProgrammableCoreType::ACTIVE_ETH, 1);
        set.add(HalProgrammableCoreType::IDLE_ETH, 0);
        set.add(HalProgrammableCoreType::IDLE_ETH, 1);
    }
    if (set.empty()) {
        TT_THROW("Invalid RISC selection: \"{}\". Valid values are BR,NC,TR0,TR1,TR2,TR*,ER0,ER1,ER*.", spec);
    }
    return set;
}

uint32_t Hal::make_go_msg_u32(
    uint8_t signal, uint8_t master_x, uint8_t master_y, uint8_t dispatch_message_offset) const {
    uint32_t go_msg_u32_val = 0;
    // We know go_msg_t is the same for all core types, so we can use TENSIX's factory.
    auto go_msg = get_dev_msgs_factory(HalProgrammableCoreType::TENSIX)
                      .create_view<dev_msgs::go_msg_t>(reinterpret_cast<std::byte*>(&go_msg_u32_val));
    TT_ASSERT(go_msg.size() == sizeof(uint32_t));
    go_msg.signal() = signal;
    go_msg.master_x() = master_x;
    go_msg.master_y() = master_y;
    go_msg.dispatch_message_offset() = dispatch_message_offset;
    return go_msg_u32_val;
}

}  // namespace tt::tt_metal

std::size_t std::hash<tt::tt_metal::HalProcessorIdentifier>::operator()(
    const tt::tt_metal::HalProcessorIdentifier& processor) const {
    auto hasher = std::hash<int>();
    std::size_t hash = 0;
    hash ^= hasher(static_cast<int>(processor.core_type));
    hash ^= hasher(static_cast<int>(processor.processor_class)) << 1;
    hash ^= hasher(processor.processor_type) << 2;
    return hash;
}
