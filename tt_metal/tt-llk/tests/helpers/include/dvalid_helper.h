// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef ARCH_QUASAR

#include <cstdint>

#include "ckernel.h"
#include "ckernel_instr_params.h"
#include "llk_defs.h"

#if defined(LLK_TRISC_UNPACK)
#include "llk_unpack_common.h"
#endif

#if defined(LLK_TRISC_MATH) || defined(LLK_TRISC_ISOLATE_SFPU) || defined(LLK_TRISC_PACK)
#include "llk_math_common.h"
// #include "llk_math_eltwise_sfpu_common.h"
#endif

#if defined(LLK_TRISC_PACK)
#include "llk_pack_common.h"
#endif

namespace test_utils::dvalid
{

using ckernel::dest_dvalid_client;
using ckernel::DstSync;
using ckernel::set_up_dest_dvalid_per_thread;

// ─────────────────────────────────────────────────────────────────────────────
// DestChain — all valid Dest dvalid chain configurations
// ─────────────────────────────────────────────────────────────────────────────

enum class DestChain : std::uint8_t
{
    FPU_PACK,
    UNPACK_PACK,
    FPU_SFPU_PACK,
    UNPACK_SFPU_PACK,
    UNPACK_FPU_PACK,
    UNPACK_FPU_SFPU_PACK,
    NONE,
};

// ─────────────────────────────────────────────────────────────────────────────
// setup_dest_dvalid — programs the dvalid control registers for the calling thread
// ─────────────────────────────────────────────────────────────────────────────

namespace detail
{

// Chain arrays — defined once, used by all threads
inline constexpr dest_dvalid_client chain_fpu_pack[] = {
    dest_dvalid_client::FPU,
    dest_dvalid_client::PACK,
};

inline constexpr dest_dvalid_client chain_unpack_pack[] = {
    dest_dvalid_client::UNPACK,
    dest_dvalid_client::PACK,
};

inline constexpr dest_dvalid_client chain_fpu_sfpu_pack[] = {
    dest_dvalid_client::FPU,
    dest_dvalid_client::SFPU,
    dest_dvalid_client::PACK,
};

inline constexpr dest_dvalid_client chain_unpack_sfpu_pack[] = {
    dest_dvalid_client::UNPACK,
    dest_dvalid_client::SFPU,
    dest_dvalid_client::PACK,
};

inline constexpr dest_dvalid_client chain_unpack_fpu_pack[] = {
    dest_dvalid_client::UNPACK,
    dest_dvalid_client::FPU,
    dest_dvalid_client::PACK,
};

inline constexpr dest_dvalid_client chain_unpack_fpu_sfpu_pack[] = {
    dest_dvalid_client::UNPACK,
    dest_dvalid_client::FPU,
    dest_dvalid_client::SFPU,
    dest_dvalid_client::PACK,
};

template <dest_dvalid_client CLIENT, DestChain CHAIN>
inline void setup_for_client()
{
    if constexpr (CHAIN == DestChain::FPU_PACK)
    {
        set_up_dest_dvalid_per_thread<CLIENT>(chain_fpu_pack);
    }
    else if constexpr (CHAIN == DestChain::UNPACK_PACK)
    {
        set_up_dest_dvalid_per_thread<CLIENT>(chain_unpack_pack);
    }
    else if constexpr (CHAIN == DestChain::FPU_SFPU_PACK)
    {
        set_up_dest_dvalid_per_thread<CLIENT>(chain_fpu_sfpu_pack);
    }
    else if constexpr (CHAIN == DestChain::UNPACK_SFPU_PACK)
    {
        set_up_dest_dvalid_per_thread<CLIENT>(chain_unpack_sfpu_pack);
    }
    else if constexpr (CHAIN == DestChain::UNPACK_FPU_PACK)
    {
        set_up_dest_dvalid_per_thread<CLIENT>(chain_unpack_fpu_pack);
    }
    else if constexpr (CHAIN == DestChain::UNPACK_FPU_SFPU_PACK)
    {
        set_up_dest_dvalid_per_thread<CLIENT>(chain_unpack_fpu_sfpu_pack);
    }
}

template <DestChain CHAIN>
inline constexpr bool chain_has_sfpu_v = CHAIN == DestChain::FPU_SFPU_PACK || CHAIN == DestChain::UNPACK_SFPU_PACK || CHAIN == DestChain::UNPACK_FPU_SFPU_PACK;

template <DestChain CHAIN>
inline constexpr bool chain_has_fpu_v =
    CHAIN == DestChain::FPU_PACK || CHAIN == DestChain::FPU_SFPU_PACK || CHAIN == DestChain::UNPACK_FPU_PACK || CHAIN == DestChain::UNPACK_FPU_SFPU_PACK;

} // namespace detail

template <DestChain CHAIN>
inline void setup_dest_dvalid()
{
    if constexpr (CHAIN == DestChain::NONE)
    {
        return;
    }

#if defined(LLK_TRISC_UNPACK)
    detail::setup_for_client<dest_dvalid_client::UNPACK, CHAIN>();
#endif

#if defined(LLK_TRISC_MATH)
    if constexpr (detail::chain_has_fpu_v<CHAIN>)
    {
        detail::setup_for_client<dest_dvalid_client::FPU, CHAIN>();
    }
    if constexpr (detail::chain_has_sfpu_v<CHAIN>)
    {
        detail::setup_for_client<dest_dvalid_client::SFPU, CHAIN>();
    }
#endif

#if defined(LLK_TRISC_ISOLATE_SFPU)
    if constexpr (detail::chain_has_sfpu_v<CHAIN>)
    {
        detail::setup_for_client<dest_dvalid_client::SFPU, CHAIN>();
    }
#endif

#if defined(LLK_TRISC_PACK)
    detail::setup_for_client<dest_dvalid_client::PACK, CHAIN>();
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// drain_pipeline — wait for this thread's engines to go idle
// ─────────────────────────────────────────────────────────────────────────────

inline void drain_pipeline()
{
#if defined(LLK_TRISC_UNPACK)
    wait_unpack_idle();
#endif

#if defined(LLK_TRISC_MATH)
    wait_fpu_idle();
    wait_sfpu_idle();
    wait_mop_idle();
#endif

#if defined(LLK_TRISC_ISOLATE_SFPU)
    wait_sfpu_idle();
#endif

#if defined(LLK_TRISC_PACK)
    wait_pack_idle();
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// reset_dest_dvalid — clear all dvalid state bits and bank IDs
// ─────────────────────────────────────────────────────────────────────────────

inline void reset_dest_dvalid()
{
    TTI_CLEARDVALID(0, 0, 0xf, 0xf, 0, 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// reprogram_dest_dvalid — drain, reset, and set up a new chain
// ─────────────────────────────────────────────────────────────────────────────

template <DestChain NEW_CHAIN>
inline void reprogram_dest_dvalid()
{
    drain_pipeline();
    reset_dest_dvalid();
    setup_dest_dvalid<NEW_CHAIN>();
}

// ─────────────────────────────────────────────────────────────────────────────
// Dest dvalid signal functions — producer completion
// ─────────────────────────────────────────────────────────────────────────────

#if defined(LLK_TRISC_UNPACK)

template <DstSync DST>
inline void signal_unpack_done()
{
    _llk_unpack_dest_dvalid_section_done_<DST>();
}

#endif // LLK_TRISC_UNPACK

#if defined(LLK_TRISC_MATH) || defined(LLK_TRISC_ISOLATE_SFPU) || defined(LLK_TRISC_PACK)

template <DstSync DST>
inline void signal_fpu_done()
{
    _llk_math_set_dvalid_<p_cleardvalid::FPU, DST>();
}

template <DstSync DST>
inline void signal_sfpu_done()
{
    _llk_math_set_dvalid_<p_cleardvalid::SFPU, DST>();
}

#endif // LLK_TRISC_MATH || LLK_TRISC_ISOLATE_SFPU || LLK_TRISC_PACK

// ─────────────────────────────────────────────────────────────────────────────
// Dest dvalid section done — consumer completion (pack)
// ─────────────────────────────────────────────────────────────────────────────

#if defined(LLK_TRISC_PACK)

template <DstSync DST, bool EN_32BIT_DEST>
inline void pack_section_done()
{
    _llk_pack_dest_dvalid_section_done_<DST, EN_32BIT_DEST>();
}

#endif // LLK_TRISC_PACK
}

#endif // ARCH_QUASAR
