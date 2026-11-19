// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// The legacy logical80/padded96 concat untilizes and retilizes BF16. Its
// Blackhole conversion maps exponent-zero values (including -0) to +0 and
// NaNs to same-sign infinity. Preserve that selected behavior, not idealized
// bit-copy semantics. Exhaustive65536-pattern and per-rank Q1 evidence:
// doc/optimized_full_model/gr_compaction_canonicalization_v1.json.
constexpr uint32_t canonicalize_legacy_bf16(uint32_t bits) {
    const uint32_t exponent = bits & 0x7f80u;
    if (exponent == 0) {
        return 0;
    }
    if (exponent == 0x7f80u) {
        return bits & 0xff80u;
    }
    return bits;
}

void kernel_main() {
    constexpr auto ia = TensorAccessorArgs<0>();
    constexpr auto oa = TensorAccessorArgs<ia.next_compile_time_args_offset()>();
    const auto input = TensorAccessor(ia, get_arg_val<uint32_t>(0), 2048);
    const auto output = TensorAccessor(oa, get_arg_val<uint32_t>(1), 2048);
    const uint32_t first = get_arg_val<uint32_t>(2);
    const uint32_t stride = get_arg_val<uint32_t>(3);
    const uint32_t count = get_arg_val<uint32_t>(4);
    cb_reserve_back(0, 1);
    const uint32_t buffer = get_write_ptr(0);
    auto* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buffer);
    for (uint32_t tile = first; tile < 10; tile += stride) {
        // Exclusive whole-tile ownership, including deterministic row padding.
        if (count < 32) {
            for (uint32_t i = 0; i < 512; ++i) {
                words[i] = 0;
            }
        }
        for (uint32_t half = 0; half < 2; ++half) {
            const uint32_t live = count > half * 16 ? count - half * 16 : 0;
            for (uint32_t side = 0; side < 2; ++side) {
                const uint32_t face = tile * 2 + side;
                const uint32_t source_face = face + face / 5;
                const uint64_t source = input.get_noc_addr(source_face / 2) + (half * 2 + source_face % 2) * 512;
                const uint32_t destination = buffer + (half * 2 + side) * 512;
                if (live >= 16) {
                    noc_async_read(source, destination, 512);
                } else {
                    for (uint32_t row = 0; row < live; ++row) {
                        noc_async_read(source + row * 32, destination + row * 32, 32);
                    }
                }
            }
        }
        noc_async_read_barrier();
        // Only live rows need normalization; padding was already zeroed.
        // Each word holds two independent BF16 values. No floating-point
        // instruction, pack conversion, reduction or inter-worker dependency.
        for (uint32_t half = 0; half < 2; ++half) {
            const uint32_t remaining = count > half * 16 ? count - half * 16 : 0;
            const uint32_t live = remaining < 16 ? remaining : 16;
            for (uint32_t side = 0; side < 2; ++side) {
                const uint32_t first_word = (half * 2 + side) * 128;
                for (uint32_t word = 0; word < live * 8; ++word) {
                    const uint32_t packed = words[first_word + word];
                    words[first_word + word] =
                        canonicalize_legacy_bf16(packed & 0xffffu) | (canonicalize_legacy_bf16(packed >> 16) << 16);
                }
            }
        }
        noc_async_write(buffer, output.get_noc_addr(tile), 2048);
        noc_async_write_barrier();
    }
}
