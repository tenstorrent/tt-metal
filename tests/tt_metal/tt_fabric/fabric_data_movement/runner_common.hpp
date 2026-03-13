// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Shared test helpers for fabric data movement runners.
// Used by addrgen_write runners and auto_packetization runners.

#pragma once

#include <cstdint>
#include <iomanip>
#include <vector>
#include <gtest/gtest.h>

namespace tt::tt_fabric::test {

// Generate deterministic TX pattern: 0xA5A50000 + i
inline std::vector<uint32_t> make_tx_pattern(size_t n_words) {
    std::vector<uint32_t> tx(n_words);
    for (size_t i = 0; i < n_words; ++i) {
        tx[i] = 0xA5A50000u + static_cast<uint32_t>(i);
    }
    return tx;
}

// Validate RX payload equals TX payload word-by-word.
// Optional word_offset and n_words for partial verification.
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

// Validate tensor_bytes is word-aligned.
inline bool validate_word_alignment_or_fail(uint32_t tensor_bytes) {
    if ((tensor_bytes % 4) != 0) {
        ADD_FAILURE() << "tensor_bytes must be a multiple of 4 (word-aligned) for verification.";
        return false;
    }
    return true;
}

}  // namespace tt::tt_fabric::test
