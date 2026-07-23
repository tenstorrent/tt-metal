// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ostream>
#include <cstdint>

namespace tt::tt_metal {

enum class MathFidelity : uint8_t {
    LoFi = 0,
    HiFi2 = 2,
    HiFi3 = 3,
    HiFi4 = 4,
    Invalid = 0xff,
};

inline std::ostream& operator<<(std::ostream& os, const MathFidelity& fidelity) {
    switch (fidelity) {
        case MathFidelity::LoFi: os << "LoFi"; break;
        case MathFidelity::HiFi2: os << "HiFi2"; break;
        case MathFidelity::HiFi3: os << "HiFi3"; break;
        case MathFidelity::HiFi4: os << "HiFi4"; break;
        case MathFidelity::Invalid: os << "Invalid"; break;
        default: throw std::invalid_argument("Unknown format");
    }
    return os;
}

/**
 * Specifies mode of operation for unpacking directly to Dest register.
 * Default mode enables all dataformats (except Float32) to be unpacked into Dest. Buffers
 * with Default mode can be used to unpack to SRCA/B or Dest.
 * UnpackToDestFp32 enables unpacking Float32 data to Dest with full precision, but makes
 * the buffer incompatible with unpacking to SRCA/B.
 */
enum class UnpackToDestMode : uint8_t { UnpackToDestFp32, Default };

/**
 * Selects where the Unpacker places a consumed buffer's data.
 *   UnpackToSrc  — into the SrcA/SrcB register files (the default). Feeds the FPU directly,
 *                  and the SFPU after a copy to Dest; operand precision is reduced to the
 *                  SrcA/B register format.
 *   UnpackToDest — directly into the Dest register, preserving full 32-bit precision for data
 *                  consumed by the SFPU. The buffer is then unavailable to the FPU, whose
 *                  operands must come from SrcA/SrcB.
 */
enum class UnpackMode : uint8_t { UnpackToSrc, UnpackToDest };

/**
 * Selects a relative accuracy / performance tradeoff for an operation.
 *   Precise     — favor accuracy over speed (a more precise, slower result).
 *   Approximate — favor speed over accuracy (a less precise, faster result).
 * The choice is relative: neither value denotes an absolute precision.
 */
enum class Precision : uint8_t { Approximate, Precise };

}  // namespace tt::tt_metal

template <>
struct std::hash<tt::tt_metal::MathFidelity> {
    std::size_t operator()(const tt::tt_metal::MathFidelity& obj) const noexcept {
        return static_cast<std::size_t>(obj);
    }
};

// Adding to tt::tt_metal namespace as we transition to moving this out of global namespace eventually.
using MathFidelity [[deprecated("Use tt::tt_metal::MathFidelity")]] = tt::tt_metal::MathFidelity;
using UnpackToDestMode [[deprecated("Use tt::tt_metal::UnpackToDestMode")]] = tt::tt_metal::UnpackToDestMode;
