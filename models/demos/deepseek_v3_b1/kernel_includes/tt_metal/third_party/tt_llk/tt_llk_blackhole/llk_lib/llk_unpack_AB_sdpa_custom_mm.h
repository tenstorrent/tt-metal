// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cunpack_common.h"
#include "llk_unpack_AB_custom_mm.h"

using namespace ckernel;
using namespace ckernel::unpacker;

// #43563 candidate fix: order the reset_config_context() SETC16 (CfgContextOffset=0) BEFORE the
// subsequent cfg[] base-address writes / UNPACR that depend on the context. 1 = on.
#define FIX_CFG_ORDER_43563 0

// #43562/3 candidate fix (Welford #39225 / PR#43582 analog): the standard LLK paths prime the SrcB/SrcA
// data-valid state via _llk_unpack_set_srcb_dummy_valid_ (UNPACR_NOP SET_DVALID + UNP_ZEROSRC); the SDPA
// custom unpacker port omits it. Hypothesis: the QK's SrcB(K) valid/bank state is left in a stale,
// iteration/bank-parity-dependent configuration by the prior chunk's PV; priming it to a known
// zero+valid state before the real K unpack (each chunk) resets it. 1 = on (inlined body, per chunk).
#define FIX_SRCB_DUMMY_VALID_43563 0

// #43562/3 candidate fix: the sibling generic unpacker _llk_unpack_AB_custom_mm_ does a CLR_SRC on SrcB
// before unpacking (clear_src=true default, llk_unpack_AB_custom_mm.h:254-258) that "clears BOTH banks
// after waiting for both to be free" — SrcB here = Q (tiny 8-of-64-row tile), so 7/8 rows + the inactive
// ping-pong bank carry stale data. The SDPA unpacker calls _run_ directly and SKIPS this. Hypothesis:
// the missing both-banks SrcB clear leaves a bank-parity-dependent stale SrcB -> bank-parity mm1 (the
// modulo-2 alternation). This adds exactly the sibling's CLR_SRC. 1 = on.
#define FIX_CLR_SRCB_43563 0

template <bool read_transposed = false>
inline void _llk_unpack_AB_sdpa_custom_mm_(
    const std::uint32_t base_address_a,
    const std::uint32_t base_address_b,
    const std::uint32_t base_address_mask,
    const std::uint32_t tile_index_a,
    const std::uint32_t tile_index_b,
    const std::uint32_t tile_size_a,
    const std::uint32_t tile_size_b,
    const std::uint32_t kt_dim,
    const std::uint32_t ct_dim = 1,
    const bool mask_chunk = false) {
    volatile uint* cfg = get_cfg_pointer();
    const std::uint32_t block_increment = read_transposed ? kt_dim * tile_size_a : tile_size_a;
    const std::uint32_t inner_increment = read_transposed ? -(((ct_dim - 1) * kt_dim) - 1) * tile_size_a : tile_size_a;

    const std::uint32_t address_a = base_address_a + tile_size_a * tile_index_a;
    const std::uint32_t address_b = base_address_b + tile_size_b * tile_index_b;

    // Wait for all contexts to be free
    wait_for_next_context(1);
    reset_config_context();
#if defined(FIX_CLR_SRCB_43563) && FIX_CLR_SRCB_43563
    // #43562/3: clear SrcB (=Q) data in BOTH banks before unpacking — exactly the SrcB clear the sibling
    // generic unpacker does (clear_src=true) that the SDPA port omitted. Removes stale bank-parity SrcB.
    TTI_UNPACR_NOP(SrcB, 0, 0, 0, 0, 0, 1, 0, p_unpacr_nop::CLR_SRC);
#endif
#if defined(FIX_SRCB_DUMMY_VALID_43563) && FIX_SRCB_DUMMY_VALID_43563
    // #43562/3: inlined _llk_unpack_set_srcb_dummy_valid_ — prime SrcB/SrcA to a known zero+valid state
    // before the real K/Q unpack, resetting any stale iteration/bank-parity SrcB state left by the prior
    // chunk's PV. (Standard LLK paths do this; the SDPA custom port omitted it — Welford #39225 analog.)
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::UNPACK);
    TTI_UNPACR_NOP(SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, 0, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
    TTI_UNPACR_NOP(SrcA, 0, 0, p_unpacr_nop::SET_DVALID, 0, 0, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
#endif
#if defined(FIX_CFG_ORDER_43563) && FIX_CFG_ORDER_43563
    // #43563: ensure the reset_config_context() SETC16 (CfgContextOffset=0) COMMITS before the cfg[]
    // base-address writes below (mask base, and SrcA/SrcB base in _llk_unpack_AB_custom_mm_run_) which
    // depend on the active config context. Without it, on the no-mask path the SrcB(K) base write can
    // race the context reset -> UNPACR reads K from the wrong/stale context -> bank-dependent mm1.
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
#endif

    if (mask_chunk) {
        cfg[THCON_SEC1_REG3_Base_cntx1_address_ADDR32] = base_address_mask;
        TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
        TTI_UNPACR_COMMON_EXPLICIT_CONTEXT(SrcB, 0b00000000, 1, 1);
    }

    _llk_unpack_AB_custom_mm_run_(cfg, address_a, address_b, block_increment, inner_increment, kt_dim);
}
