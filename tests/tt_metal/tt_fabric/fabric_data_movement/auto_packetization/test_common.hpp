// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Lightweight test header for auto-packetization silicon tests.
// Defines the AutoPacketFamily enum (all 9 families), RawTestParams struct,
// and helpers for kernel path selection and family categorization.

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <tt-metalium/core_coord.hpp>
#include "tests/tt_metal/tt_fabric/fabric_data_movement/runner_common.hpp"

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

// TX_OP_* values mirroring tx_kernel_common.h device constants.
// Passed as the send_op RT arg to the unified unicast/multicast kernels.
static constexpr uint32_t TX_OP_WRITE                   = 0;
static constexpr uint32_t TX_OP_SCATTER_WRITE            = 1;
static constexpr uint32_t TX_OP_FUSED_ATOMIC_INC         = 2;
static constexpr uint32_t TX_OP_FUSED_SCATTER_ATOMIC_INC = 3;

// Maps each AutoPacketFamily to its TX_OP_* dispatch value.
// SparseMulticast has its own kernel and does not use send_op.
inline uint32_t family_tx_op(AutoPacketFamily family) {
    switch (family) {
        case AutoPacketFamily::UnicastWrite:
        case AutoPacketFamily::MulticastWrite:
            return TX_OP_WRITE;
        case AutoPacketFamily::UnicastScatter:
        case AutoPacketFamily::MulticastScatter:
            return TX_OP_SCATTER_WRITE;
        case AutoPacketFamily::UnicastFusedAtomicInc:
        case AutoPacketFamily::MulticastFusedAtomicInc:
            return TX_OP_FUSED_ATOMIC_INC;
        case AutoPacketFamily::UnicastFusedScatterAtomicInc:
        case AutoPacketFamily::MulticastFusedScatterAtomicInc:
            return TX_OP_FUSED_SCATTER_ATOMIC_INC;
        default:
            return TX_OP_WRITE;
    }
}

// Maps each AutoPacketFamily to its device kernel .cpp path.
// All unicast/multicast variants share tx_writer_raw.cpp (selected via IS_MULTICAST define).
// SparseMulticast uses its own dedicated kernel.
inline std::string family_kernel_path(AutoPacketFamily family) {
    const std::string base = "tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/kernels/";
    if (family == AutoPacketFamily::SparseMulticast) {
        return base + "sparse_multicast_tx_writer_raw.cpp";
    }
    return base + "tx_writer_raw.cpp";
}

// Returns true if the family requires the IS_MULTICAST compile-time define.
inline bool family_is_multicast(AutoPacketFamily family) {
    switch (family) {
        case AutoPacketFamily::MulticastWrite:
        case AutoPacketFamily::MulticastScatter:
        case AutoPacketFamily::MulticastFusedAtomicInc:
        case AutoPacketFamily::MulticastFusedScatterAtomicInc:
        case AutoPacketFamily::SparseMulticast:
            return true;
        default:
            return false;
    }
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

// make_tx_pattern and verify_payload_words are provided by runner_common.hpp

}  // namespace tt::tt_fabric::test
