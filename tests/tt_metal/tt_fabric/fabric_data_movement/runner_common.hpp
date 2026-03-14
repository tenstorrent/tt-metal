// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Shared test types and helpers for fabric data movement runners.
// Used by addrgen_write runners and auto_packetization runners.

#pragma once

#include <cstdint>
#include <iomanip>
#include <string>
#include <vector>
#include <gtest/gtest.h>

namespace tt::tt_fabric::test {

// ---------------------------------------------------------------------------
// Decomposed test variant: replaces both AddrgenApiVariant and AutoPacketFamily
// ---------------------------------------------------------------------------

enum class CastMode : uint8_t {
    Unicast,
    Multicast,
    LinearUnicast,
    LinearMulticast,
    SparseMulticast,
};

enum class WriteOp : uint8_t {
    Write,
    Scatter,
    FusedAtomicInc,
    FusedScatterAtomicInc,
};

enum class StateMode : uint8_t {
    Stateless,
    WithState,
    SetState,
};

enum class ConnectionMode : uint8_t {
    Direct,
    ConnMgr,
};

struct FabricTestVariant {
    CastMode cast;
    WriteOp op;
    StateMode state = StateMode::Stateless;
    ConnectionMode conn = ConnectionMode::Direct;

    bool is_unicast() const { return cast == CastMode::Unicast || cast == CastMode::LinearUnicast; }
    bool is_multicast() const {
        return cast == CastMode::Multicast || cast == CastMode::LinearMulticast ||
               cast == CastMode::SparseMulticast;
    }
    bool is_linear() const { return cast == CastMode::LinearUnicast || cast == CastMode::LinearMulticast; }
    bool is_scatter() const { return op == WriteOp::Scatter || op == WriteOp::FusedScatterAtomicInc; }
    bool is_fused() const { return op == WriteOp::FusedAtomicInc || op == WriteOp::FusedScatterAtomicInc; }
    bool is_conn_mgr() const { return conn == ConnectionMode::ConnMgr; }
    bool is_stateful() const { return state != StateMode::Stateless; }
};

// ---------------------------------------------------------------------------
// String conversion for test naming
// ---------------------------------------------------------------------------

inline std::string to_string(CastMode c) {
    switch (c) {
        case CastMode::Unicast: return "Unicast";
        case CastMode::Multicast: return "Multicast";
        case CastMode::LinearUnicast: return "LinearUnicast";
        case CastMode::LinearMulticast: return "LinearMulticast";
        case CastMode::SparseMulticast: return "SparseMulticast";
    }
    return "Unknown";
}

inline std::string to_string(WriteOp o) {
    switch (o) {
        case WriteOp::Write: return "Write";
        case WriteOp::Scatter: return "Scatter";
        case WriteOp::FusedAtomicInc: return "FusedAtomicInc";
        case WriteOp::FusedScatterAtomicInc: return "FusedScatterAtomicInc";
    }
    return "Unknown";
}

inline std::string to_string(StateMode s) {
    switch (s) {
        case StateMode::Stateless: return "Stateless";
        case StateMode::WithState: return "WithState";
        case StateMode::SetState: return "SetState";
    }
    return "Unknown";
}

inline std::string to_string(ConnectionMode c) {
    switch (c) {
        case ConnectionMode::Direct: return "Direct";
        case ConnectionMode::ConnMgr: return "ConnMgr";
    }
    return "Unknown";
}

inline std::string to_string(const FabricTestVariant& v) {
    std::string name = to_string(v.cast) + to_string(v.op);
    if (v.state != StateMode::Stateless) {
        name += "_" + to_string(v.state);
    }
    if (v.conn == ConnectionMode::ConnMgr) {
        name += "_ConnMgr";
    }
    return name;
}

// ---------------------------------------------------------------------------
// TX_OP_* dispatch values (mirroring tx_kernel_common.h device constants)
// ---------------------------------------------------------------------------

static constexpr uint32_t TX_OP_WRITE = 0;
static constexpr uint32_t TX_OP_SCATTER_WRITE = 1;
static constexpr uint32_t TX_OP_FUSED_ATOMIC_INC = 2;
static constexpr uint32_t TX_OP_FUSED_SCATTER_ATOMIC_INC = 3;

inline uint32_t to_tx_op(WriteOp op) {
    switch (op) {
        case WriteOp::Write: return TX_OP_WRITE;
        case WriteOp::Scatter: return TX_OP_SCATTER_WRITE;
        case WriteOp::FusedAtomicInc: return TX_OP_FUSED_ATOMIC_INC;
        case WriteOp::FusedScatterAtomicInc: return TX_OP_FUSED_SCATTER_ATOMIC_INC;
    }
    return TX_OP_WRITE;
}

// ---------------------------------------------------------------------------
// Kernel path selection (auto-packetization)
// ---------------------------------------------------------------------------

inline std::string family_kernel_path(const FabricTestVariant& v) {
    const std::string base = "tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/kernels/";
    if (v.cast == CastMode::SparseMulticast) {
        return base + "sparse_multicast_tx_writer_raw.cpp";
    }
    return base + "tx_writer_raw.cpp";
}

// ---------------------------------------------------------------------------
// Data verification helpers
// ---------------------------------------------------------------------------

inline std::vector<uint32_t> make_tx_pattern(size_t n_words) {
    std::vector<uint32_t> tx(n_words);
    for (size_t i = 0; i < n_words; ++i) {
        tx[i] = 0xA5A50000u + static_cast<uint32_t>(i);
    }
    return tx;
}

inline void verify_payload_words(
    const std::vector<uint32_t>& rx,
    const std::vector<uint32_t>& tx,
    size_t word_offset = 0,
    size_t n_words = 0) {
    size_t count = (n_words > 0) ? n_words : tx.size();
    if (word_offset == 0 && n_words == 0 && rx.size() != tx.size()) {
        ADD_FAILURE() << "RX size mismatch: got " << rx.size() << " words, expected " << tx.size();
        return;
    }
    for (size_t i = 0; i < count; ++i) {
        if (rx[i + word_offset] != tx[i]) {
            ADD_FAILURE() << "Data mismatch at word " << i << " (offset " << word_offset << "): got 0x" << std::hex
                          << rx[i + word_offset] << ", exp 0x" << tx[i] << std::dec;
            return;
        }
    }
}

inline bool validate_word_alignment_or_fail(uint32_t tensor_bytes) {
    if ((tensor_bytes % 4) != 0) {
        ADD_FAILURE() << "tensor_bytes must be a multiple of 4 (word-aligned) for verification.";
        return false;
    }
    return true;
}

}  // namespace tt::tt_fabric::test
