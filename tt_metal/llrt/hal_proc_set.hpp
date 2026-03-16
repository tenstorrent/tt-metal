// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <tt-metalium/hal_types.hpp>  // HalProgrammableCoreType, NumHalProgrammableCoreTypes

namespace tt::tt_metal {

// A set of processors distinguishing programmable core type and index within that core type.
// See Hal::get_processor_index and Hal::get_processor_class_and_type_from_index.
class HalProcessorSet {
private:
    std::array<uint32_t, NumHalProgrammableCoreTypes> masks_{};

public:
    void add(HalProgrammableCoreType core_type, uint32_t processor_index) {
        masks_[static_cast<size_t>(core_type)] |= (1u << processor_index);
    }
    bool contains(HalProgrammableCoreType core_type, uint32_t processor_index) const {
        return (masks_[static_cast<size_t>(core_type)] & (1u << processor_index)) != 0;
    }
    bool empty() const {
        for (const auto& mask : masks_) {
            if (mask != 0) {
                return false;
            }
        }
        return true;
    }
    // Returns the bitmask of processors for the given core type.
    // Bit i set <=> processor index i is in the set.
    uint32_t get_processor_mask(HalProgrammableCoreType core_type) const {
        return masks_[static_cast<size_t>(core_type)];
    }
};

}  // namespace tt::tt_metal
