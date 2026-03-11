// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Lightweight test header for auto-packetization silicon tests.
// Defines the AutoPacketFamily enum (all 9 families), RawTestParams struct,
// and helpers for kernel path selection and family categorization.

#pragma once

#include <cstdint>
#include <string>
#include <tt-metalium/core_coord.hpp>

namespace tt::tt_fabric::test {

// All 9 auto-packetizing wrapper families.
enum class AutoPacketFamily {
    UnicastWrite,
    UnicastScatter,
    UnicastFusedAtomicInc,
    UnicastFusedScatterAtomicInc,
    MulticastWrite,
    MulticastScatter,
    MulticastFusedAtomicInc,
    MulticastFusedScatterAtomicInc,
    SparseMulticast,
};

// Minimal test configuration for raw-size silicon tests.
struct RawTestParams {
    uint32_t mesh_id;
    ChipId src_chip;
    ChipId dst_chip;
    uint32_t tensor_bytes;
    CoreCoord sender_core;
    CoreCoord receiver_core;
    AutoPacketFamily family;
    bool use_dram_dst = false;
};

// Maps each AutoPacketFamily to its device kernel .cpp path.
// Paths are relative to project root (as used by CreateKernel).
// Note: multicast/sparse kernel files are created by Plan 02 and may not exist on disk yet.
inline std::string family_kernel_path(AutoPacketFamily family) {
    const std::string base = "tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/kernels/";
    switch (family) {
        case AutoPacketFamily::UnicastWrite:
            return base + "unicast_tx_writer_raw.cpp";
        case AutoPacketFamily::UnicastScatter:
            return base + "scatter_unicast_tx_writer_raw.cpp";
        case AutoPacketFamily::UnicastFusedAtomicInc:
            return base + "fused_atomic_inc_unicast_tx_writer_raw.cpp";
        case AutoPacketFamily::UnicastFusedScatterAtomicInc:
            return base + "fused_scatter_atomic_inc_unicast_tx_writer_raw.cpp";
        case AutoPacketFamily::MulticastWrite:
            return base + "multicast_tx_writer_raw.cpp";
        case AutoPacketFamily::MulticastScatter:
            return base + "scatter_multicast_tx_writer_raw.cpp";
        case AutoPacketFamily::MulticastFusedAtomicInc:
            return base + "fused_atomic_inc_multicast_tx_writer_raw.cpp";
        case AutoPacketFamily::MulticastFusedScatterAtomicInc:
            return base + "fused_scatter_atomic_inc_multicast_tx_writer_raw.cpp";
        case AutoPacketFamily::SparseMulticast:
            return base + "sparse_multicast_tx_writer_raw.cpp";
    }
    return "";
}

// Returns true if the family uses scatter semantics (2-destination writes).
inline bool family_is_scatter(AutoPacketFamily family) {
    switch (family) {
        case AutoPacketFamily::UnicastScatter:
        case AutoPacketFamily::UnicastFusedScatterAtomicInc:
        case AutoPacketFamily::MulticastScatter:
        case AutoPacketFamily::MulticastFusedScatterAtomicInc:
            return true;
        default:
            return false;
    }
}

// Returns true if the family fuses data write with atomic_inc (no separate atomic_inc needed).
inline bool family_is_fused(AutoPacketFamily family) {
    switch (family) {
        case AutoPacketFamily::UnicastFusedAtomicInc:
        case AutoPacketFamily::UnicastFusedScatterAtomicInc:
        case AutoPacketFamily::MulticastFusedAtomicInc:
        case AutoPacketFamily::MulticastFusedScatterAtomicInc:
            return true;
        default:
            return false;
    }
}

}  // namespace tt::tt_fabric::test
