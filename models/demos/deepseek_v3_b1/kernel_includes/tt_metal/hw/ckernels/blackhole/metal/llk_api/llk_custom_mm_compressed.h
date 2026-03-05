// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_unpack_compressed.h"

// NOTE: Requires custom_mm.h to be included before this header.
// Depends on _llk_math_custom_mm_, semaphore::UNPACK_SYNC, TT_MOP, etc.

namespace compressed {

// ---------------------------------------------------------------------------
// Custom MM with compressed in1 (per-tile format reconfig in software loop)
//
// Uses custom_mm_block init/uninit for MOP setup, but replaces the standard
// _run_ with a custom version that does reconfig_unpack_srca + variable
// address increment per context. ct_dim=1 only. kt_dim must be even.
//
// SrcB (in0/activations) auto-advances via HW counters.
// SrcA (in1/weights) address and format set per context in software loop.
// ---------------------------------------------------------------------------

/**
 * @brief Reconfig SrcA format for use inside custom_mm MOP loop.
 *
 * Uses direct cfg[] writes instead of cfg_reg_rmw_tensix to avoid
 * tensix instruction pipeline latency. Must be called AFTER
 * wait_for_next_context() when the unpacker is paused at semaphore.
 */
FORCE_INLINE void reconfig_custom_mm_srca(
    volatile uint* cfg, uint32_t fmt_idx, uint32_t reg0_base, uint32_t reg2_base) {
    uint32_t src_format = DATA_FORMATS[fmt_idx];
    cfg[THCON_SEC0_REG0_TileDescriptor_ADDR32] = reg0_base | src_format;
    cfg[THCON_SEC0_REG2_Out_data_format_ADDR32] = reg2_base | src_format;
}

/** @brief Reconfig SrcA with pre-resolved DataFormat value (no lookup). */
FORCE_INLINE void reconfig_custom_mm_srca_raw(
    volatile uint* cfg, uint32_t src_format, uint32_t reg0_base, uint32_t reg2_base) {
    cfg[THCON_SEC0_REG0_TileDescriptor_ADDR32] = reg0_base | src_format;
    cfg[THCON_SEC0_REG2_Out_data_format_ADDR32] = reg2_base | src_format;
}

/** @brief Reconfig SrcA input format only (REG0). REG2 stays unchanged. */
FORCE_INLINE void reconfig_custom_mm_srca_input_only(volatile uint* cfg, uint32_t src_format, uint32_t reg0_base) {
    cfg[THCON_SEC0_REG0_TileDescriptor_ADDR32] = reg0_base | src_format;
}

// ---------------------------------------------------------------------------
// Constexpr template-unrolled custom MM (zero runtime format lookup overhead)
//
// Format array is passed as positional CTAs. Template recursion unrolls
// the MOP loop so each tile's format is a compile-time constant.
// ---------------------------------------------------------------------------

/**
 * @brief Process one pair of K tiles (ctx0 + ctx1) with compile-time format.
 */
template <size_t PAIR_IDX, size_t NUM_PACKED, const std::array<uint32_t, NUM_PACKED>& FMT_PACKED>
FORCE_INLINE void _custom_mm_compressed_pair_(
    volatile uint* cfg, uint32_t& address_a, uint32_t reg0_base, uint32_t reg2_base) {
    UNPACK(({
        constexpr uint32_t K = PAIR_IDX * 2;
        constexpr uint32_t fmt0 =
            (FMT_PACKED[K / TILES_PER_UINT32] >> ((K % TILES_PER_UINT32) * ASSIGN_BITS)) & ASSIGN_MASK;
        constexpr uint32_t sz0 = TILE_SIZES[fmt0] >> cb_addr_shift;
        constexpr uint32_t fmt1 =
            (FMT_PACKED[(K + 1) / TILES_PER_UINT32] >> (((K + 1) % TILES_PER_UINT32) * ASSIGN_BITS)) & ASSIGN_MASK;
        constexpr uint32_t sz1 = TILE_SIZES[fmt1] >> cb_addr_shift;

        wait_for_next_context(2);
        reconfig_custom_mm_srca(cfg, fmt0, reg0_base, reg2_base);
        cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address_a;
        address_a += sz0;
        semaphore_post(semaphore::UNPACK_SYNC);

        wait_for_next_context(2);
        reconfig_custom_mm_srca(cfg, fmt1, reg0_base, reg2_base);
        cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = address_a;
        address_a += sz1;
        semaphore_post(semaphore::UNPACK_SYNC);
    }));
}

/**
 * @brief Unroll all K pairs using fold expression.
 */
template <size_t NUM_PACKED, const std::array<uint32_t, NUM_PACKED>& FMT_PACKED, size_t... PAIR_IDXS>
FORCE_INLINE void _custom_mm_compressed_unroll_impl_(
    volatile uint* cfg,
    uint32_t& address_a,
    uint32_t reg0_base,
    uint32_t reg2_base,
    std::index_sequence<PAIR_IDXS...>) {
    (_custom_mm_compressed_pair_<PAIR_IDXS, NUM_PACKED, FMT_PACKED>(cfg, address_a, reg0_base, reg2_base), ...);
}

/**
 * @brief Unroll all context pairs for KT_DIM × CT_DIM tiles.
 */
template <uint32_t KT_DIM, uint32_t CT_DIM, size_t NUM_PACKED, const std::array<uint32_t, NUM_PACKED>& FMT_PACKED>
FORCE_INLINE void _custom_mm_compressed_unroll_(
    volatile uint* cfg, uint32_t& address_a, uint32_t reg0_base, uint32_t reg2_base) {
    _custom_mm_compressed_unroll_impl_<NUM_PACKED, FMT_PACKED>(
        cfg, address_a, reg0_base, reg2_base, std::make_index_sequence<KT_DIM * CT_DIM / 2>{});
}

// ---------------------------------------------------------------------------
// Compact constexpr: per-format non-inlined pair functions + constexpr dispatch
//
// Instead of unrolling 112 unique code blocks, generate 4 shared functions
// (one per format pair type) and a constexpr sequence of calls.
// Code size: ~200 bytes (4 functions) + ~450 bytes (112 calls) ≈ 650 bytes
// ---------------------------------------------------------------------------

/**
 * @brief Per-format pair processor. Template on FMT_IDX so each format gets
 *        one shared function body. __attribute__((noinline)) prevents the compiler
 *        from duplicating the body at each call site.
 */
/**
 * @brief Process n_pairs with a fixed constexpr format in a tight loop.
 *        noinline so only one copy per format exists. n_pairs is runtime.
 *        Inner loop: all immediates (tile size, reconfig values), no data loads.
 */
template <uint32_t FMT_IDX>
__attribute__((noinline)) void _run_fmt_(
    volatile uint* cfg, uint32_t& address_a, uint32_t reg0_base, uint32_t reg2_base, uint32_t n_pairs) {
    UNPACK(({
        constexpr uint32_t sz = TILE_SIZES[FMT_IDX] >> cb_addr_shift;
        reconfig_custom_mm_srca(cfg, FMT_IDX, reg0_base, reg2_base);
        // Use local copy to keep address_a in a register (avoid ref load-modify-store)
        uint32_t addr = address_a;
        for (uint32_t i = 0; i < n_pairs; i++) {
            wait_for_next_context(2);
            cfg[THCON_SEC0_REG3_Base_address_ADDR32] = addr;
            addr += sz;
            semaphore_post(semaphore::UNPACK_SYNC);
            wait_for_next_context(2);
            cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = addr;
            addr += sz;
            semaphore_post(semaphore::UNPACK_SYNC);
        }
        address_a = addr;
    }));
}

// Helper: extract format for tile K from packed array
template <size_t K, size_t NUM_PACKED, const std::array<uint32_t, NUM_PACKED>& FMT_PACKED>
constexpr uint32_t _get_fmt_() {
    return (FMT_PACKED[K / TILES_PER_UINT32] >> ((K % TILES_PER_UINT32) * ASSIGN_BITS)) & ASSIGN_MASK;
}

// ---------------------------------------------------------------------------
// Compile-time run detection: scan FMT_PACKED at compile time, emit one
// noinline call per format run. Each call has a runtime pair count but
// constexpr format/tile-size (all immediates in the inner loop).
//
// For uniform bfp8: 1 call to _run_fmt_<0>(..., 112)
// For mostly bfp4 with some bfp8: few calls, e.g. _run_fmt_<1>(..., 100), _run_fmt_<0>(..., 12)
// For fully interleaved: many calls but each to a shared ~80 byte function body
// ---------------------------------------------------------------------------

// Check if a pair (2 tiles) is uniform (both same format)
template <size_t PAIR_IDX, size_t NUM_PACKED, const std::array<uint32_t, NUM_PACKED>& FMT_PACKED>
constexpr bool _pair_uniform_() {
    return _get_fmt_<PAIR_IDX * 2, NUM_PACKED, FMT_PACKED>() == _get_fmt_<PAIR_IDX * 2 + 1, NUM_PACKED, FMT_PACKED>();
}

// Find the number of consecutive same-format PAIRS starting at PAIR_START
// (where each pair is also internally uniform)
template <size_t PAIR_START, size_t TOTAL_PAIRS, size_t NUM_PACKED, const std::array<uint32_t, NUM_PACKED>& FMT_PACKED>
constexpr size_t _pair_run_length_() {
    if constexpr (!_pair_uniform_<PAIR_START, NUM_PACKED, FMT_PACKED>()) {
        return 0;  // first pair itself is mixed — no uniform run
    } else if constexpr (PAIR_START + 1 >= TOTAL_PAIRS) {
        return 1;
    } else {
        constexpr uint32_t fmt = _get_fmt_<PAIR_START * 2, NUM_PACKED, FMT_PACKED>();
        constexpr uint32_t next_fmt = _get_fmt_<(PAIR_START + 1) * 2, NUM_PACKED, FMT_PACKED>();
        if constexpr (next_fmt != fmt || !_pair_uniform_<PAIR_START + 1, NUM_PACKED, FMT_PACKED>()) {
            return 1;
        } else {
            return 1 + _pair_run_length_<PAIR_START + 1, TOTAL_PAIRS, NUM_PACKED, FMT_PACKED>();
        }
    }
}

// Threshold: runs shorter than this use noinline call, longer use noinline call too.
// The distinction is that runs >= MIN_LONG_RUN use the loop-based _run_fmt_ (1 call),
// while runs < MIN_LONG_RUN get inlined as constexpr pairs (zero call overhead, small code).
constexpr size_t MIN_LONG_RUN = 4;  // pairs (8 tiles)

// Inline N constexpr pairs starting at PAIR_START (for short runs at format boundaries)
template <size_t PAIR_START, size_t N, size_t NUM_PACKED, const std::array<uint32_t, NUM_PACKED>& FMT_PACKED>
FORCE_INLINE void _inline_pairs_(volatile uint* cfg, uint32_t& address_a, uint32_t reg0_base, uint32_t reg2_base) {
    if constexpr (N > 0) {
        constexpr uint32_t fmt0 = _get_fmt_<PAIR_START * 2, NUM_PACKED, FMT_PACKED>();
        constexpr uint32_t fmt1 = _get_fmt_<PAIR_START * 2 + 1, NUM_PACKED, FMT_PACKED>();
        constexpr uint32_t sz0 = TILE_SIZES[fmt0] >> cb_addr_shift;
        constexpr uint32_t sz1 = TILE_SIZES[fmt1] >> cb_addr_shift;
        UNPACK(({
            wait_for_next_context(2);
            reconfig_custom_mm_srca(cfg, fmt0, reg0_base, reg2_base);
            cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address_a;
            address_a += sz0;
            semaphore_post(semaphore::UNPACK_SYNC);
            wait_for_next_context(2);
            reconfig_custom_mm_srca(cfg, fmt1, reg0_base, reg2_base);
            cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = address_a;
            address_a += sz1;
            semaphore_post(semaphore::UNPACK_SYNC);
        }));
        _inline_pairs_<PAIR_START + 1, N - 1, NUM_PACKED, FMT_PACKED>(cfg, address_a, reg0_base, reg2_base);
    }
}

// Count consecutive short runs (runs < MIN_LONG_RUN) starting at PAIR_START.
// Returns total number of pairs covered by these short runs.
template <size_t PAIR_START, size_t TOTAL_PAIRS, size_t NUM_PACKED, const std::array<uint32_t, NUM_PACKED>& FMT_PACKED>
constexpr size_t _short_region_pairs_() {
    if constexpr (PAIR_START >= TOTAL_PAIRS) {
        return 0;
    } else {
        constexpr size_t run = _pair_run_length_<PAIR_START, TOTAL_PAIRS, NUM_PACKED, FMT_PACKED>();
        if constexpr (run == 0) {
            // Mixed pair: 1 pair, continue scanning
            return 1 + _short_region_pairs_<PAIR_START + 1, TOTAL_PAIRS, NUM_PACKED, FMT_PACKED>();
        } else if constexpr (run < MIN_LONG_RUN) {
            // Short uniform run: include it, continue scanning
            return run + _short_region_pairs_<PAIR_START + run, TOTAL_PAIRS, NUM_PACKED, FMT_PACKED>();
        } else {
            return 0;  // Hit a long run — stop
        }
    }
}

// Emit calls for all pairs starting at PAIR_START
template <size_t PAIR_START, size_t TOTAL_PAIRS, size_t NUM_PACKED, const std::array<uint32_t, NUM_PACKED>& FMT_PACKED>
FORCE_INLINE void _emit_runs_(volatile uint* cfg, uint32_t& address_a, uint32_t reg0_base, uint32_t reg2_base) {
    if constexpr (PAIR_START < TOTAL_PAIRS) {
        constexpr size_t run = _pair_run_length_<PAIR_START, TOTAL_PAIRS, NUM_PACKED, FMT_PACKED>();

        if constexpr (run >= MIN_LONG_RUN) {
            // Long uniform run: noinline function with loop (fast, compact)
            constexpr uint32_t fmt = _get_fmt_<PAIR_START * 2, NUM_PACKED, FMT_PACKED>();
            _run_fmt_<fmt>(cfg, address_a, reg0_base, reg2_base, run);
            _emit_runs_<PAIR_START + run, TOTAL_PAIRS, NUM_PACKED, FMT_PACKED>(cfg, address_a, reg0_base, reg2_base);
        } else {
            // Short run or mixed pair: inline all consecutive short runs as constexpr pairs
            constexpr size_t short_pairs = _short_region_pairs_<PAIR_START, TOTAL_PAIRS, NUM_PACKED, FMT_PACKED>();
            static_assert(short_pairs > 0, "short region must have at least 1 pair");
            _inline_pairs_<PAIR_START, short_pairs, NUM_PACKED, FMT_PACKED>(cfg, address_a, reg0_base, reg2_base);
            _emit_runs_<PAIR_START + short_pairs, TOTAL_PAIRS, NUM_PACKED, FMT_PACKED>(
                cfg, address_a, reg0_base, reg2_base);
        }
    }
}

template <uint32_t KT_DIM, uint32_t CT_DIM, size_t NUM_PACKED, const std::array<uint32_t, NUM_PACKED>& FMT_PACKED>
FORCE_INLINE void custom_mm_compressed_block_compact(
    uint32_t addr_in0, uint32_t addr_in1, uint32_t in0_face_r_dim, uint32_t dst_index) {
    UNPACK(({
        volatile uint* cfg = get_cfg_pointer();
        uint32_t reg0_base = cfg[THCON_SEC0_REG0_TileDescriptor_ADDR32] & ~0x0f;
        uint32_t reg2_base = cfg[THCON_SEC0_REG2_Out_data_format_ADDR32] & ~0x0f;

        wait_for_next_context(1);
        reset_config_context();

        cfg[THCON_SEC1_REG3_Base_address_ADDR32] = addr_in0;
        TT_MOP(0, (KT_DIM / 2) - 1, 0);

        uint32_t address_a = addr_in1;
        constexpr size_t total_pairs = KT_DIM * CT_DIM / 2;
        _emit_runs_<0, total_pairs, NUM_PACKED, FMT_PACKED>(cfg, address_a, reg0_base, reg2_base);
    }));
    MATH((_llk_math_custom_mm_<true>(in0_face_r_dim, dst_index, KT_DIM, CT_DIM)));
}

/**
 * @brief Constexpr custom MM block. Zero runtime format lookup.
 *
 * FMT_PACKED is a constexpr array of packed format words (from fill_cta_array).
 * Tiles packed row-major: (k=0,n=0), (k=0,n=1), ..., (k=1,n=0), ...
 * Total tiles = KT_DIM * CT_DIM, must be even.
 */
template <uint32_t KT_DIM, uint32_t CT_DIM, size_t NUM_PACKED, const std::array<uint32_t, NUM_PACKED>& FMT_PACKED>
FORCE_INLINE void custom_mm_compressed_block_constexpr(
    uint32_t addr_in0, uint32_t addr_in1, uint32_t in0_face_r_dim, uint32_t dst_index) {
    UNPACK(({
        volatile uint* cfg = get_cfg_pointer();
        uint32_t reg0_base = cfg[THCON_SEC0_REG0_TileDescriptor_ADDR32] & ~0x0f;
        uint32_t reg2_base = cfg[THCON_SEC0_REG2_Out_data_format_ADDR32] & ~0x0f;

        wait_for_next_context(1);
        reset_config_context();

        cfg[THCON_SEC1_REG3_Base_address_ADDR32] = addr_in0;
        TT_MOP(0, (KT_DIM / 2) - 1, 0);

        uint32_t address_a = addr_in1;
        _custom_mm_compressed_unroll_<KT_DIM, CT_DIM, NUM_PACKED, FMT_PACKED>(cfg, address_a, reg0_base, reg2_base);
    }));
    MATH((_llk_math_custom_mm_<true>(in0_face_r_dim, dst_index, KT_DIM, CT_DIM)));
}

// ---------------------------------------------------------------------------
// Constexpr array, runtime loop (compact code, fast array access)
//
// Format array is passed as constexpr CTAs (same as above), but the loop
// is a regular for-loop instead of template-unrolled. This gives compact
// code size (~1.5KB vs ~10KB) while avoiding slow volatile L1 pointer reads.
// ---------------------------------------------------------------------------

/**
 * @brief Extract format index from packed constexpr array at runtime index.
 */
template <size_t NUM_PACKED>
FORCE_INLINE uint32_t _get_packed_format_(const std::array<uint32_t, NUM_PACKED>& fmt_packed, uint32_t tile_idx) {
    return (fmt_packed[tile_idx / TILES_PER_UINT32] >> ((tile_idx % TILES_PER_UINT32) * ASSIGN_BITS)) & ASSIGN_MASK;
}

/**
 * @brief Custom MM block with constexpr format array but runtime loop.
 *
 * Same interface as custom_mm_compressed_block_constexpr but uses a for-loop
 * instead of template unrolling. Compact code, array lives in local memory.
 */
/**
 * @brief Runtime loop reading packed pairs from an L1 tensor.
 *
 * fmt_l1_addr: byte address of uint32 array in L1, one packed word per pair.
 * Each word: [sz1:8 | sz0:8 | fmt1:8 | fmt0:8]
 * No constexpr arrays, no stack usage — scales to any K dimension.
 */
template <uint32_t KT_DIM, uint32_t CT_DIM>
FORCE_INLINE void custom_mm_compressed_block_runtime_loop(
    uint32_t fmt_l1_addr, uint32_t addr_in0, uint32_t addr_in1, uint32_t in0_face_r_dim, uint32_t dst_index) {
    UNPACK(({
        volatile uint* cfg = get_cfg_pointer();
        uint32_t reg0_base = cfg[THCON_SEC0_REG0_TileDescriptor_ADDR32] & ~0x0f;

        wait_for_next_context(1);
        reset_config_context();

        cfg[THCON_SEC1_REG3_Base_address_ADDR32] = addr_in0;
        TT_MOP(0, (KT_DIM / 2) - 1, 0);

        uint32_t address_a = addr_in1;
        constexpr uint32_t num_pairs = KT_DIM * CT_DIM / 2;

        // Read packed pairs directly from L1 tensor
        const volatile uint32_t* fmt_ptr = reinterpret_cast<const volatile uint32_t*>(fmt_l1_addr);

        union PairInfo {
            uint32_t packed;
            struct {
                uint8_t fmt0, fmt1, sz0, sz1;
            };
        };

        for (uint32_t pair = 0; pair < num_pairs; pair++) {
            PairInfo p;
            p.packed = fmt_ptr[pair];  // direct L1 load

            wait_for_next_context(2);
            reconfig_custom_mm_srca_input_only(cfg, p.fmt0, reg0_base);
            cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address_a;
            address_a += p.sz0;
            semaphore_post(semaphore::UNPACK_SYNC);

            wait_for_next_context(2);
            reconfig_custom_mm_srca_input_only(cfg, p.fmt1, reg0_base);
            cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = address_a;
            address_a += p.sz1;
            semaphore_post(semaphore::UNPACK_SYNC);
        }
    }));
    MATH((_llk_math_custom_mm_<true>(in0_face_r_dim, dst_index, KT_DIM, CT_DIM)));
}

// ---------------------------------------------------------------------------
// Runtime version (for fallback / ct_dim>1)
// ---------------------------------------------------------------------------

/**
 * @brief Custom _run_ for compressed weights with per-tile format switching.
 *
 * Supports ct_dim=1 and even ct_dim>1.
 */
FORCE_INLINE void _custom_mm_compressed_run_(
    const volatile uint8_t* assign_ptr, uint32_t address_a, uint32_t address_b, uint32_t kt_dim, uint32_t ct_dim) {
    UNPACK(({
        volatile uint* cfg = get_cfg_pointer();

        uint32_t reg0_base = cfg[THCON_SEC0_REG0_TileDescriptor_ADDR32] & ~0x0f;
        uint32_t reg2_base = cfg[THCON_SEC0_REG2_Out_data_format_ADDR32] & ~0x0f;

        cfg[THCON_SEC1_REG3_Base_address_ADDR32] = address_b;
        TT_MOP(0, (kt_dim / 2) - 1, 0);

        if (ct_dim == 1) {
            for (uint32_t k = 0; k < kt_dim; k += 2) {
                uint32_t fmt0 = get_tile_format(assign_ptr, k);
                wait_for_next_context(2);
                reconfig_custom_mm_srca(cfg, fmt0, reg0_base, reg2_base);
                cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address_a;
                address_a += TILE_SIZES[fmt0] >> cb_addr_shift;
                semaphore_post(semaphore::UNPACK_SYNC);

                uint32_t fmt1 = get_tile_format(assign_ptr, k + 1);
                wait_for_next_context(2);
                reconfig_custom_mm_srca(cfg, fmt1, reg0_base, reg2_base);
                cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = address_a;
                address_a += TILE_SIZES[fmt1] >> cb_addr_shift;
                semaphore_post(semaphore::UNPACK_SYNC);
            }
        } else {
            uint32_t tile_idx = 0;
            for (uint32_t k = 0; k < kt_dim; k++) {
                uint32_t row_start = address_a;
                for (uint32_t ct = 0; ct < ct_dim; ct += 2) {
                    uint32_t fmt0 = get_tile_format(assign_ptr, tile_idx);
                    wait_for_next_context(2);
                    reconfig_custom_mm_srca(cfg, fmt0, reg0_base, reg2_base);
                    cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address_a;
                    address_a += TILE_SIZES[fmt0] >> cb_addr_shift;
                    tile_idx++;
                    semaphore_post(semaphore::UNPACK_SYNC);

                    uint32_t fmt1 = get_tile_format(assign_ptr, tile_idx);
                    wait_for_next_context(2);
                    reconfig_custom_mm_srca(cfg, fmt1, reg0_base, reg2_base);
                    cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = address_a;
                    address_a += TILE_SIZES[fmt1] >> cb_addr_shift;
                    tile_idx++;
                    semaphore_post(semaphore::UNPACK_SYNC);
                }
            }
        }
    }));
}

/**
 * @brief Custom MM block for compressed weights.
 */
FORCE_INLINE void custom_mm_compressed_block(
    const volatile uint8_t* assign_ptr,
    uint32_t addr_in0,
    uint32_t addr_in1,
    uint32_t in0_face_r_dim,
    uint32_t kt_dim,
    uint32_t ct_dim,
    uint32_t dst_index) {
    UNPACK(({
        wait_for_next_context(1);
        reset_config_context();
    }));
    _custom_mm_compressed_run_(assign_ptr, addr_in1, addr_in0, kt_dim, ct_dim);
    MATH((_llk_math_custom_mm_<true>(in0_face_r_dim, dst_index, kt_dim, ct_dim)));
}

}  // namespace compressed
