// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>

namespace x280 {

// Names follow a contiguous, zero-based enum declaration.
template <typename Enum, std::size_t Size>
constexpr const char* enum_name(Enum value, const std::array<const char*, Size>& names) noexcept {
    const auto index = static_cast<std::size_t>(value);
    return index < names.size() ? names[index] : "unknown";
}

inline std::string hex(uint64_t value) {
    std::ostringstream out;
    out << "0x" << std::hex << value;
    return out.str();
}

inline std::string label_chip_l2cpu(int chip_id, int hw_index) {
    return "chip " + std::to_string(chip_id) + " L2CPU " + std::to_string(hw_index);
}

inline std::string label_noc0_xy(uint64_t noc_x, uint64_t noc_y) {
    return "noc0 (" + std::to_string(noc_x) + "," + std::to_string(noc_y) + ")";
}

inline std::string format_trap_compact(uint64_t mcause, uint64_t mepc, uint64_t mtval) {
    return "trap mcause " + hex(mcause) + " mepc " + hex(mepc) + " mtval " + hex(mtval);
}

inline std::string format_readback_values(uint64_t wrote, uint64_t observed) {
    return "wrote " + hex(wrote) + ", read " + hex(observed);
}

[[noreturn]] inline void fail(std::string message) { throw std::runtime_error("x280: " + message); }

}  // namespace x280
