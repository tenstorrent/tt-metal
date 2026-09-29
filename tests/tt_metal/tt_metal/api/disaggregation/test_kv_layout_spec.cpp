// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <variant>

#include <tt-metalium/shape.hpp>
#include <tt-metalium/tensor/spec/layout/tensor_layout.hpp>
#include <tt-metalium/tensor/spec/memory_config/memory_config.hpp>
#include <tt-metalium/tensor/spec/tensor_spec.hpp>

#include "internal/disaggregation/kv_layout_spec.hpp"

namespace tt::tt_metal::internal::disaggregation {
namespace {

// A minimal interleaved TensorSpec — has_sequence() reads only `temporal`, so the tensor is a stand-in
// (KvLayoutSpec has no default ctor because TensorSpec requires a shape + layout).
TensorSpec dummy_tensor() {
    return TensorSpec(
        Shape{1, kTile, kTile}, TensorLayout(DataType::BFLOAT8_B, PageConfig(Layout::TILE), MemoryConfig{}));
}

// Skeleton-level tests: exercise the collapsed spec's self-contained parts (TemporalPolicy,
// AddressingMode, has_sequence) and the GenerationPolicy defaults. The TensorSpec-dependent
// derivations (sequence_axis / feature_width / chunk_size_bytes) are covered end-to-end with real
// ND-sharded TensorSpecs in test_to_chunk_map.cpp.

constexpr uint32_t EXPECTED_WINDOW_TOKENS = 4096;

// --- TemporalPolicy ---

TEST(KvLayoutSpec, CPU_TemporalPolicyDefaultsToDense) {
    TemporalPolicy policy;  // std::variant default-constructs its first alternative
    EXPECT_TRUE(std::holds_alternative<temporal::Dense>(policy));
}

TEST(KvLayoutSpec, CPU_TemporalPolicyWindowCarriesWidth) {
    TemporalPolicy policy = temporal::Window{.width = WindowTokens{EXPECTED_WINDOW_TOKENS}};
    ASSERT_TRUE(std::holds_alternative<temporal::Window>(policy));
    EXPECT_EQ(std::get<temporal::Window>(policy).width.get(), EXPECTED_WINDOW_TOKENS);
}

TEST(KvLayoutSpec, CPU_TemporalPolicyRollingIsTagless) {
    // Conv / token-shift ring: tagless. The kernel_size-1 ring depth is the conv-state tensor's own
    // axis ([B, kernel_size-1, width]), derived from the tensor, not stored on the policy.
    TemporalPolicy policy = temporal::Rolling{};
    EXPECT_TRUE(std::holds_alternative<temporal::Rolling>(policy));
}

TEST(KvLayoutSpec, CPU_TemporalPolicyRecurrentIsNone) {
    // KDA / Mamba recurrent summary: positionless, addressed by checkpoint tag.
    TemporalPolicy policy = temporal::None{};
    EXPECT_TRUE(std::holds_alternative<temporal::None>(policy));
}

// --- has_sequence(): derived from temporal, not stored ---

TEST(KvLayoutSpec, CPU_HasSequenceFromTemporal) {
    KvLayoutSpec attn{.tensor = dummy_tensor(), .temporal = temporal::Dense{}};
    EXPECT_TRUE(attn.has_sequence());

    KvLayoutSpec swa{.tensor = dummy_tensor(),
                     .temporal = temporal::Window{.width = WindowTokens{EXPECTED_WINDOW_TOKENS}}};
    EXPECT_TRUE(swa.has_sequence());

    // Recurrent / conv summaries have no per-token sequence axis.
    KvLayoutSpec conv{.tensor = dummy_tensor(), .temporal = temporal::Rolling{}};
    EXPECT_FALSE(conv.has_sequence());

    KvLayoutSpec ssm{.tensor = dummy_tensor(), .temporal = temporal::None{}};
    EXPECT_FALSE(ssm.has_sequence());
}

// --- GenerationPolicy / AddressingMode defaults ---

TEST(KvLayoutSpec, CPU_GenerationPolicyDefaultsToOptimalBankOrder) {
    GenerationPolicy policy;
    EXPECT_EQ(policy.bank_order, BankOrder::Optimal);
    EXPECT_EQ(policy.bank_scheme, BankScheme::Natural);
}

TEST(KvLayoutSpec, CPU_AddressingSupportsBothSlotAndPaged) {
    // Both representations are first-class; the spec carries which one applies per tensor.
    EXPECT_NE(AddressingMode::Slot, AddressingMode::Paged);
}

// --- arch: the layout's target, and the source of the DRAM bank count / OPTIMAL order ---

TEST(KvLayoutSpec, CPU_ArchDrivesDramBankCount) {
    KvLayoutSpec spec{.tensor = dummy_tensor(), .temporal = temporal::Dense{}};
    EXPECT_EQ(spec.arch, tt::ARCH::Invalid);  // must be set explicitly per layout

    // Bank count + OPTIMAL order are arch facts (mirroring the SoC descriptor), not per-call params.
    EXPECT_EQ(num_dram_banks(tt::ARCH::BLACKHOLE), 8u);
    EXPECT_EQ(num_dram_banks(tt::ARCH::WORMHOLE_B0), 6u);
    EXPECT_EQ(optimal_bank_order(tt::ARCH::BLACKHOLE).size(), 8u);
}

}  // namespace
}  // namespace tt::tt_metal::internal::disaggregation
