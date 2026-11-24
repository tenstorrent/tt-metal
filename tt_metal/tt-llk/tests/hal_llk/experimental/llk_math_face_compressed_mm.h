// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// HAL port of tt_llk_blackhole/llk_lib/experimental/llk_math_face_compressed_mm.h. Same instruction sequence and access
// kinds as production, expressed through the HAL, except the HAL-GAPs marked below. The SFPLOAD-based ADDRMOD_ONLY
// (SFPU) stays as production has it.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "cmath_common.h"
#include "hal/cfg.h"
#include "hal/fpu.h"
#include "hal/mop.h"
#include "hal/move.h"
#include "hal/nop.h"
#include "hal/replay.h"
#include "hal/sync.h"
#include "hal/tensix_counters.h"
#include "llk_assert.h"
#include "llk_math_common.h"

using namespace ckernel;
using namespace ckernel::math;

// HAL port of tt_llk_blackhole/common/inc/ckernel_code_sequence.h, the code-sequence helper both face-compressed
// headers use. Same compile-time sequence and table semantics; the replay handles are encoded with
// hal::replay::get_operation and the sequence is recorded with hal::replay::record (record only, as
// ckernel::load_replay_buf<NoExec>). The block is identical in llk_unpack_AB_face_compressed_mm.h and
// llk_math_face_compressed_mm.h; the guard keeps a translation unit that includes both to one definition.
#ifndef HAL_PORT_CODE_SEQUENCE_DEFINED_
#define HAL_PORT_CODE_SEQUENCE_DEFINED_

namespace hal_port
{
namespace code_seq
{

/**
 * @brief A compile-time code sequence bound to its slot in the replay buffer.
 *
 * @tparam CodeLen: Number of instructions in the sequence, deduced from the literal by @ref make_sequence.
 * @tparam ReplayBase: Offset of the sequence within the replay buffer.
 */
template <std::size_t CodeLen, std::uint32_t ReplayBase>
class sequence
{
public:
    constexpr explicit sequence(const char (&codes)[CodeLen + 1]) : m_codes(codes)
    {
    }

    /**
     * @brief Resolve an instruction fragment to a replay handle over this sequence.
     *
     * @param needle: The fragment to replay, as characters of the sequence.
     * @note Returns 0 when unrepresentable; a REPLAY encoding always sets an opcode bit, so 0 is never a real handle.
     */
    constexpr std::uint32_t fragment(const char* needle) const
    {
        std::uint32_t needle_len = 0;
        while (needle[needle_len] != '\0')
        {
            ++needle_len;
        }
        if (needle_len == 0)
        {
            return 0; // a zero-length replay is not an instruction
        }
        for (std::uint32_t i = 0; i + needle_len <= CodeLen; ++i)
        {
            std::uint32_t k = 0;
            for (; k < needle_len; ++k)
            {
                if (m_codes[i + k] != needle[k])
                {
                    break;
                }
            }
            if (k == needle_len)
            {
                return hal::replay::get_operation(hal::replay::BufferRange {ReplayBase + i /*start*/, needle_len /*count*/});
            }
        }
        return 0; // not a substring of this sequence
    }

    /**
     * @brief Resolve the fragment a has-work mask selects out of a lower-case template.
     *
     * @param tmpl: All-lower-case template, one character per work item.
     * @param mask: Bit i set upper-cases character i, e.g. case_encode("ni", 0b10) resolves "nI".
     */
    template <std::size_t N>
    constexpr std::uint32_t case_encode(const char (&tmpl)[N], const std::uint32_t mask) const
    {
        std::array<char, N> encoded {};
        std::size_t n = 0;
        for (; tmpl[n] != '\0'; ++n)
        {
            encoded[n] = ((mask >> n) & 1u) ? static_cast<char>(tmpl[n] - ('a' - 'A')) : tmpl[n];
        }
        encoded[n] = '\0';
        return fragment(encoded.data());
    }

    /**
     * @brief Record the sequence into the replay buffer, one instruction per character.
     *
     * @param emit: Callable taking one code character and issuing exactly one instruction for it.
     */
    template <typename Emit>
    [[gnu::always_inline, gnu::flatten]] inline void load(Emit&& emit) const
    {
        const char* codes = m_codes;
        hal::replay::record<hal::replay::BufferRange {ReplayBase /*start*/, CodeLen /*count*/}>(
            [codes, &emit]
            {
                auto step = [&](auto self, auto idx) -> void
                {
                    constexpr std::size_t i = decltype(idx)::value;
                    if constexpr (i < CodeLen)
                    {
                        emit(codes[i]);
                        self(self, std::integral_constant<std::size_t, i + 1> {});
                    }
                };
                step(step, std::integral_constant<std::size_t, 0> {});
            });
    }

private:
    const char* m_codes;
};

namespace detail
{

template <std::size_t N>
constexpr bool all_resolved(const std::array<std::uint32_t, N>& table)
{
    for (const std::uint32_t entry : table)
    {
        if (entry == 0)
        {
            return false;
        }
    }
    return true;
}

template <const auto& Seq, std::size_t N, typename Entry>
constexpr std::array<std::uint32_t, N> build_table(Entry&& entry)
{
    const auto find        = [](const char* needle) { return Seq.fragment(needle); };
    const auto case_encode = [](const auto& tmpl, const std::uint32_t mask) { return Seq.case_encode(tmpl, mask); };

    std::array<std::uint32_t, N> table {};
    for (std::uint32_t i = 0; i < N; ++i)
    {
        if constexpr (std::is_invocable_v<Entry&, std::uint32_t, decltype(find), decltype(case_encode)>)
        {
            table[i] = entry(i, find, case_encode);
        }
        else if constexpr (std::is_invocable_v<Entry&, std::uint32_t, decltype(find)>)
        {
            table[i] = entry(i, find);
        }
        else
        {
            table[i] = entry(i);
        }
    }
    return table;
}

} // namespace detail

/**
 * @brief Bind a code-sequence literal to its slot in the replay buffer.
 *
 * @tparam ReplayBase: Offset of the sequence within the replay buffer.
 * @tparam MaxLen: Instructions the sequence may occupy.
 * @param codes: One character per instruction, NUL-terminated.
 */
template <std::uint32_t ReplayBase, std::size_t MaxLen, std::size_t N>
constexpr sequence<N - 1, ReplayBase> make_sequence(const char (&codes)[N])
{
    static_assert(N - 1 <= MaxLen, "code sequence is longer than its slot in the replay buffer");
    static_assert(ReplayBase + MaxLen <= ckernel::REPLAY_BUF_SIZE, "code sequence slot does not fit the replay buffer");
    return sequence<N - 1, ReplayBase> {codes};
}

/**
 * @brief Build a runtime-index -> instruction decode table over a code sequence.
 *
 * @tparam Seq: The @ref sequence to resolve against.
 * @tparam N: Table size.
 * @tparam Entry: Callable invoked as entry(index), entry(index, find) or entry(index, find, case_encode).
 */
template <const auto& Seq, std::size_t N, typename Entry>
constexpr std::array<std::uint32_t, N> make_table(Entry&& entry)
{
    constexpr auto table = detail::build_table<Seq, N>(entry);
    static_assert(detail::all_resolved(table), "code sequence: a table fragment is not a substring of the sequence");
    return table;
}

} // namespace code_seq
} // namespace hal_port

#endif // HAL_PORT_CODE_SEQUENCE_DEFINED_

namespace hal_port
{

// One ADDR_MOD slot in the units of ckernel::addr_mod_t (the fields this op sets; Dest increment already masked to
// its 10-bit two's-complement encoding, as addr_mod_dest_t::val() does).
struct FaceCompressedAddrMod
{
    std::uint32_t srcb_incr = 0;
    std::uint32_t srcb_clr  = 0;
    std::uint32_t dest_incr = 0;
    std::uint32_t dest_clr  = 0;
};

// addr_mod_t::set(n): three SETC16 words (SrcA|SrcB, Dest|Fidelity, Bias), each fully specified.
template <hal::cfg::Sec S, FaceCompressedAddrMod M>
inline void set_face_compressed_addr_mod()
{
    namespace cfg = hal::cfg;
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AddrMod[cfg::SrcA].Incr, S, 0>(),
        cfg::set<cfg::AddrMod[cfg::SrcA].CR, S, 0>(),
        cfg::set<cfg::AddrMod[cfg::SrcA].Clear, S, 0>(),
        cfg::set<cfg::AddrMod[cfg::SrcB].Incr, S, M.srcb_incr>(),
        cfg::set<cfg::AddrMod[cfg::SrcB].CR, S, 0>(),
        cfg::set<cfg::AddrMod[cfg::SrcB].Clear, S, M.srcb_clr>());
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AddrMod[cfg::Dest].Incr, S, M.dest_incr>(),
        cfg::set<cfg::AddrMod[cfg::Dest].CR, S, 0>(),
        cfg::set<cfg::AddrMod[cfg::Dest].Clear, S, M.dest_clr>(),
        cfg::set<cfg::AddrMod[cfg::Dest].CToCR, S, 0>(),
        cfg::set<cfg::AddrMod[cfg::Fidelity].Incr, S, 0>(),
        cfg::set<cfg::AddrMod[cfg::Fidelity].Clear, S, 0>());
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::set<cfg::AddrMod[cfg::Bias].Incr, S, 0>(), cfg::set<cfg::AddrMod[cfg::Bias].Clear, S, 0>());
}

// addr_mod_dest_t::val() masks a signed increment to the 10-bit field.
constexpr std::uint32_t face_compressed_dest_incr(const std::int32_t incr)
{
    return static_cast<std::uint32_t>(incr) & 0x3FF;
}

// TT_OP_MOP(p_mop::MASK_LOOP, count - 1, mask): a Template 0 run handle for the decode tables.
constexpr std::uint32_t math_face_compressed_mop(const std::uint32_t count, const std::uint32_t mask)
{
    return hal::mop::Runner<hal::mop::MopTemplate::Template0> {.count = count, .mask = mask}.operation();
}

// TT_OP_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD): clear SrcB and reset the SrcA, SrcB and Dest counters.
inline constexpr std::uint32_t face_compressed_clr_b =
    hal::counters::rwc::MathCounter {
        .src_a = hal::counters::rwc::Set(0),
        .src_b = hal::counters::rwc::Set(0),
        .dest  = hal::counters::rwc::Set(0),
        .flip  = hal::counters::rwc::Src::B,
    }
        .operation();

} // namespace hal_port

// Dest rows from a face's accumulation to its split partial: consecutive MVMULs land in different rows so
// the FPU does not stall on the read-after-write. A CT == 1 tile is two faces wide, so dest ends up holding
// accumulations at rows 0 and 16 and their partials at 8 and 24, and finalize steps SrcB by this same amount
// to merge each pair.
constexpr std::int16_t _llk_math_face_compressed_mm_split_acc_partial_rows_ = 8;

// The math thread owns the replay buffer from replay_buf_offset up; the SFPU owns everything below it.
constexpr std::uint32_t _llk_math_face_compressed_mm_replay_base_ = ckernel::math::replay_buf_offset;
constexpr std::uint32_t _llk_math_face_compressed_mm_replay_slot_ = ckernel::REPLAY_BUF_SIZE - _llk_math_face_compressed_mm_replay_base_;

// The math code sequences: one character is one instruction, recorded into the replay buffer at init (see
// instr_for_code in _llk_math_face_compressed_mm_mop_config_ for the encodings). The letter names the
// addrmod that runs with it, so it says what moves after the face is handled:
//
//   n / N   noB    dest advances one face, both sources hold
//   i / I   incB   SrcB advances one activation row block, dest restarts
//   c / C   clrB   end of a K block: the SrcB and dest counters reset
//   r / R   revB   like incB, but dest rewinds to the other split-accumulation partial (ct_dim == 1)
//
// Lower case skips a face with no data, advancing the counters without multiplying; upper case issues the
// MVMUL. Each string is an overlap-packing of exactly the fragments its tables ask for, sized to the
// instructions this thread owns:
//
//   multi_seq (ct_dim >= 2) needs n i c N I C singly, plus ni Ni nI NI for the two-face midIncB only an
//   odd width emits. They pack into the first eight characters at 0, 2, 4 and 6, leaving the clrB endings
//   in the tail.
//
//   one_seq (ct_dim == 1) needs the I and C endings, nr for the MOP operands, and the six three-face
//   midRevB fragments -- {n,N}{r,R}{n,N} less nrn and nrN, which the MOP issues directly. They pack as
//   Nrn at 2, nRn at 4, nRN at 6, NrN at 8, NRN at 10, NRn at 12, and nr at 14.
constexpr auto _llk_math_face_compressed_mm_multi_seq_ =
    hal_port::code_seq::make_sequence<_llk_math_face_compressed_mm_replay_base_, _llk_math_face_compressed_mm_replay_slot_>("niNinINIncNcnCNC");
constexpr auto _llk_math_face_compressed_mm_one_seq_ =
    hal_port::code_seq::make_sequence<_llk_math_face_compressed_mm_replay_base_, _llk_math_face_compressed_mm_replay_slot_>("ICNrnRnRNrNRNRnr");

constexpr auto _llk_math_face_compressed_mm_even_l1_table_ = hal_port::code_seq::make_table<_llk_math_face_compressed_mm_multi_seq_, 64>(
    [](const std::uint32_t m) -> std::uint32_t
    {
        const std::uint32_t faces123 = (m >> 2) & 0b111;
        return hal_port::math_face_compressed_mop(3, faces123); // nnn case encoded in the mop: TT_OP_MOP(MASK_LOOP, 2, faces123)
    });

constexpr auto _llk_math_face_compressed_mm_even_l2_table_ = hal_port::code_seq::make_table<_llk_math_face_compressed_mm_multi_seq_, 64>(
    [](const std::uint32_t m, auto /* find */, auto case_encode) -> std::uint32_t
    {
        const std::uint32_t hdr   = m & 0b11;
        const std::uint32_t face4 = (m >> 5) & 0b1;
        switch (hdr)
        {
            case 0b00:
                return case_encode("n", face4); // noB
            case 0b01:
                return case_encode("i", face4); // endIncB
            case 0b10:
                return case_encode("c", face4); // endClrB
            default:
                return hal::nop::thread_operation(); // invalid hdr
        }
    });

constexpr auto _llk_math_face_compressed_mm_odd_l1_table_ = hal_port::code_seq::make_table<_llk_math_face_compressed_mm_multi_seq_, 64>(
    [](const std::uint32_t m, auto /* find */, auto case_encode) -> std::uint32_t
    {
        const std::uint32_t hdr      = m & 0b11;
        const std::uint32_t faces12  = (m >> 2) & 0b011;
        const std::uint32_t faces123 = (m >> 2) & 0b111;
        switch (hdr)
        {
            case 0b00:
                return hal_port::math_face_compressed_mop(3, faces123); // noB
            case 0b01:
                return case_encode("ni", faces12); // midIncB only two faces in this case
            case 0b10:
                return hal_port::math_face_compressed_mop(3, faces123); // endIncB
            case 0b11:
                return hal_port::math_face_compressed_mop(3, faces123); // endClrB
            default:
                return 0;
        }
    });

constexpr auto _llk_math_face_compressed_mm_odd_l2_table_ = hal_port::code_seq::make_table<_llk_math_face_compressed_mm_multi_seq_, 64>(
    [](const std::uint32_t m, auto /* find */, auto case_encode) -> std::uint32_t
    {
        const std::uint32_t hdr     = m & 0b11;
        const std::uint32_t faces34 = (m >> 4) & 0b11;
        const std::uint32_t face4   = (m >> 5) & 0b01;
        switch (hdr)
        {
            case 0b00:
                return case_encode("n", face4); // noB
            case 0b01:
                return hal_port::math_face_compressed_mop(2, faces34); // midIncB the mop walks two faces: TT_OP_MOP(MASK_LOOP, 1, faces34)
            case 0b10:
                return case_encode("i", face4); // endIncB
            case 0b11:
                return case_encode("c", face4); // endClrB
            default:
                return 0;
        }
    });

constexpr auto _llk_math_face_compressed_mm_one_l1_table_ = hal_port::code_seq::make_table<_llk_math_face_compressed_mm_one_seq_, 64>(
    [](const std::uint32_t m, auto /* find */, auto case_encode) -> std::uint32_t
    {
        const std::uint32_t faces123 = (m >> 2) & 0b111;
        switch (faces123)
        {
            case 0b000:
                return hal_port::math_face_compressed_mop(1, 0); // "nrn" encoded in the mop: TT_OP_MOP(MASK_LOOP, 0, 0)
            case 0b100:
                return hal_port::math_face_compressed_mop(1, 1); // "nrN" encoded in the mop: TT_OP_MOP(MASK_LOOP, 0, 1)
            default:
                return case_encode("nrn", faces123); // always midRevB
        }
    });

constexpr auto _llk_math_face_compressed_mm_one_l2_table_ = hal_port::code_seq::make_table<_llk_math_face_compressed_mm_one_seq_, 64>(
    [](const std::uint32_t m, auto find) -> std::uint32_t
    {
        const std::uint32_t hdr   = m & 0b11;
        const std::uint32_t face4 = (m >> 5) & 0b1;
        switch ((hdr << 1) | face4)
        {
            case 0b000:
                return TT_OP_ADDRMOD_ONLY(ADDR_MOD_1); // endIncB zero face4 (i); SFPLOAD, kept as production
            case 0b001:
                return find("I"); // endIncB data face4 (I)
            case 0b010:
                return hal_port::face_compressed_clr_b; // endClrB zero face4 (c)
            case 0b011:
                return find("C"); // endClrB data face4 (C)
            default:
                return hal::nop::thread_operation(); // invalid hdr
        }
    });

/**
 * @brief Program the five address modifiers the code sequence moves the sources and dest with.
 *
 * Every instruction in the sequence names one of these, and it decides what moves afterwards: ADDR_MOD_0
 * holds both sources and advances dest one face, ADDR_MOD_1 and ADDR_MOD_3 step SrcB to the next activation
 * row block, ADDR_MOD_2 clears the SrcB and dest counters, and ADDR_MOD_4 walks SrcB onto the partial that
 * finalize merges.
 *
 * @param face_r_dim: Activation rows per face, 1 or 8. This is the SrcB increment, so ADDR_MOD_1 and
 *                    ADDR_MOD_3 are programmed from one of two compile-time arms.
 * @note Called from @ref _llk_math_face_compressed_mm_init_ rather than directly.
 */
inline void _llk_math_face_compressed_mm_addrmod_config_(const std::uint32_t face_r_dim)
{
    using hal::cfg::Sec;
    using hal_port::FaceCompressedAddrMod;
    using hal_port::set_face_compressed_addr_mod;
    constexpr std::uint32_t rev_dest = hal_port::face_compressed_dest_incr(-_llk_math_face_compressed_mm_split_acc_partial_rows_);

    // noB: hold both sources, advance dest by one face
    set_face_compressed_addr_mod<static_cast<Sec>(ADDR_MOD_0), FaceCompressedAddrMod {.dest_incr = 16}>();

    // clrB: reset the SrcB and dest counters
    set_face_compressed_addr_mod<static_cast<Sec>(ADDR_MOD_2), FaceCompressedAddrMod {.srcb_clr = 1, .dest_clr = 1}>();

    // One arm per M value: every addrmod field must be a compile-time constant for the immediate SETC16.
    if (face_r_dim == 1)
    {
        // incB: advance SrcB one activation row block, reset dest to the beginning
        set_face_compressed_addr_mod<static_cast<Sec>(ADDR_MOD_1), FaceCompressedAddrMod {.srcb_incr = 1, .dest_clr = 1}>();
        // revB: advance SrcB one activation row block, rewind dest to the other partial
        set_face_compressed_addr_mod<static_cast<Sec>(ADDR_MOD_3), FaceCompressedAddrMod {.srcb_incr = 1, .dest_incr = rev_dest}>();
    }
    else
    {
        // incB: advance SrcB one activation row block, reset dest to the beginning
        set_face_compressed_addr_mod<static_cast<Sec>(ADDR_MOD_1), FaceCompressedAddrMod {.srcb_incr = 8, .dest_clr = 1}>();
        // revB: advance SrcB one activation row block, rewind dest to the other partial
        set_face_compressed_addr_mod<static_cast<Sec>(ADDR_MOD_3), FaceCompressedAddrMod {.srcb_incr = 8, .dest_incr = rev_dest}>();
    }

    // finalize: advance SrcB onto the partial MOVD2B parked there, hold dest
    set_face_compressed_addr_mod<
        static_cast<Sec>(ADDR_MOD_4),
        FaceCompressedAddrMod {.srcb_incr = static_cast<std::uint32_t>(_llk_math_face_compressed_mm_split_acc_partial_rows_)}>();
}

/**
 * @brief Record the math code sequence into the replay buffer and program the MOP that replays it.
 *
 * One instruction per character: lower case skips a face, advancing the counters with SFPLOAD or SETRWC,
 * and upper case multiplies it in with MVMUL. ct_dim == 1 records the split-accumulation sequence, every
 * other width the multi-tile one.
 *
 * @tparam ct_dim: Output width in tiles, 1 to 16.
 * @param face_r_dim: Activation rows per face, 1 or 8. Only 1 takes MVMUL's single-row SrcB broadcast,
 *                    which drops the SrcB alignment requirement from 8 rows to 1.
 * @note Called from @ref _llk_math_face_compressed_mm_init_ rather than directly.
 * @ref _llk_math_face_compressed_mm_addrmod_config_ programs the addrmods these instructions name.
 */
template <std::uint32_t ct_dim>
inline void _llk_math_face_compressed_mm_mop_config_(const std::uint32_t face_r_dim)
{
    namespace fpu = hal::fpu;

    // MVMUL's instr_mod19 SrcB broadcast, using it for a single row reduces the SrcB alignment requirement from 8 to 1
    const fpu::SrcBRowBroadcast mvmul_single_row = (face_r_dim == 1) ? fpu::SrcBRowBroadcast::Row : fpu::SrcBRowBroadcast::None;

    // One instruction per code-sequence character. Lower case means no data for this face, so it is
    // skipped: the counters advance via SFPLOAD or SETRWC without multiplying. Upper case multiplies it
    // in. The ADDR_MOD then picks what advances afterwards -- see _llk_math_face_compressed_mm_addrmod_config_.
    // The MVMULs carry the runtime broadcast bit, so they go through the instruction buffer, as production's TT_MVMUL.
    auto instr_for_code = [mvmul_single_row](char code)
    {
        switch (code)
        {
            case 'n':
                TTI_ADDRMOD_ONLY(ADDR_MOD_0);
                break; // skip, noB
            case 'N':
                fpu::run(fpu::MatrixMultiply {
                    .broadcast = mvmul_single_row, .address_modifier = ADDR_MOD_0, .dest_row_offset = 0, .release = fpu::SourceRelease::SrcA});
                break; // mul, noB
            case 'i':
                TTI_ADDRMOD_ONLY(ADDR_MOD_1);
                break; // skip, incB
            case 'I':
                fpu::run(fpu::MatrixMultiply {
                    .broadcast = mvmul_single_row, .address_modifier = ADDR_MOD_1, .dest_row_offset = 0, .release = fpu::SourceRelease::SrcA});
                break; // mul, incB
            case 'c':
                TTI_INSN(hal_port::face_compressed_clr_b);
                break; // skip, clrB: TTI_SETRWC(CLR_B, 0, 0, 0, 0, SET_ABD)
            case 'C':
                fpu::run(fpu::MatrixMultiply {
                    .broadcast = mvmul_single_row, .address_modifier = ADDR_MOD_2, .dest_row_offset = 0, .release = fpu::SourceRelease::Both});
                break; // mul, clrB
            case 'r':
                TTI_ADDRMOD_ONLY(ADDR_MOD_3);
                break; // skip, revB
            case 'R':
                fpu::run(fpu::MatrixMultiply {
                    .broadcast = mvmul_single_row, .address_modifier = ADDR_MOD_3, .dest_row_offset = 0, .release = fpu::SourceRelease::SrcA});
                break; // mul, revB
            default:
                LLK_ASSERT(false, "Invalid code for math instruction");
                break;
        }
    };

    if constexpr (ct_dim == 1)
    {
        _llk_math_face_compressed_mm_one_seq_.load(instr_for_code);
    }
    else
    {
        _llk_math_face_compressed_mm_multi_seq_.load(instr_for_code);
    }

    constexpr std::uint32_t op0m = _llk_math_face_compressed_mm_multi_seq_.fragment("n");
    constexpr std::uint32_t op0M = _llk_math_face_compressed_mm_multi_seq_.fragment("N");
    constexpr std::uint32_t op0s = _llk_math_face_compressed_mm_one_seq_.fragment("nr");
    constexpr std::uint32_t op1s = _llk_math_face_compressed_mm_one_seq_.fragment("n");
    constexpr std::uint32_t op0S = _llk_math_face_compressed_mm_one_seq_.fragment("nr");
    constexpr std::uint32_t op1S = _llk_math_face_compressed_mm_one_seq_.fragment("N");

    // Every MOP operand must resolve to a real replay instruction; a 0 means its fragment
    // is not a substring of the code sequence (otherwise silent until it breaks at runtime).
    static_assert(
        op0m != 0 && op0M != 0 && op0s != 0 && op1s != 0 && op0S != 0 && op1S != 0,
        "face_compressed_mm (math): a MOP operand fragment is not a substring of the code sequence");

    // ckernel_unpack_template(unpackB = ct_dim == 1, unpackHalo = false, A, A1-3 = NOP, skipA, B = op1s, skipB = op1S).
    // The HAL leaves the words the expander does not read unwritten: InsnA1-3 always, InsnB/SkipB when ct_dim != 1.
    using namespace hal::mop;
    if constexpr (ct_dim == 1)
    {
        program(foreach_mask_bit(if_clear(op0s, op1s), otherwise(op0S, op1S)));
    }
    else
    {
        program(foreach_mask_bit(if_clear(op0m), otherwise(op0M)));
    }
}

/**
 * @brief Configure the math thread for a face-granular compressed matmul.
 *
 * @tparam ct_dim: Output width in tiles, 1 to 16.
 * @param face_r_dim: Activation rows per face, 1 or 8.
 * @note Call this before @ref _llk_math_face_compressed_mm_ with the same ct_dim. This thread has no
 *       uninit: it leaves behind only addrmods, the MOP and the replay buffer, which the next op's init
 *       reprograms anyway.
 * @note On the unpack thread, pair with @ref _llk_unpack_AB_face_compressed_mm_init_.
 */
template <std::uint32_t ct_dim = 1>
inline void _llk_math_face_compressed_mm_init_(const std::uint32_t face_r_dim)
{
    static_assert(ct_dim >= 1 && ct_dim <= 16, "face_compressed_mm (math): ct_dim must be in [1, 16]");
    LLK_ASSERT(face_r_dim == 1 || face_r_dim == 8, "face_compressed_mm (math): unsupported face_r_dim (expected 1 or 8)");

    _llk_math_face_compressed_mm_addrmod_config_(face_r_dim);
    _llk_math_face_compressed_mm_mop_config_<ct_dim>(face_r_dim);

    // math::reset_counters(p_setrwc::SET_ABD_F)
    hal_port::reset_counters_abd_f();
}

/**
 * @brief Select the decode table pair for one output width.
 *
 * @tparam ct_dim: Output width in tiles, 1 to 16.
 * @return The level-1 and level-2 tables, which expand a 6-bit meta into its two instructions.
 * @note Three regimes, because the header values a width can produce differ: an odd width can advance SrcB
 *       in the middle of a meta, which an even one never does, and ct_dim == 1 splits the accumulation on
 *       top of that.
 * @note Assumes ct_dim is in range. @ref _llk_math_face_compressed_mm_, its only caller, asserts it.
 */
template <std::uint32_t ct_dim>
inline constexpr std::array<const std::uint32_t*, 2> _llk_math_face_compressed_mm_tables_()
{
    if constexpr (ct_dim == 1)
    {
        return {{_llk_math_face_compressed_mm_one_l1_table_.data(), _llk_math_face_compressed_mm_one_l2_table_.data()}};
    }
    else if constexpr (ct_dim % 2 == 0)
    {
        return {{_llk_math_face_compressed_mm_even_l1_table_.data(), _llk_math_face_compressed_mm_even_l2_table_.data()}};
    }
    else
    {
        return {{_llk_math_face_compressed_mm_odd_l1_table_.data(), _llk_math_face_compressed_mm_odd_l2_table_.data()}};
    }
}

/**
 * @brief Multiply the unpacked activation by the compressed weight faces, accumulating into DST.
 *
 * Streams the math meta section, expanding each 6-bit meta through the two tables into the instructions
 * that multiply its four faces and step SrcB. For ct_dim == 1 the accumulation runs as two halves per face,
 * which finalize merges at the end.
 *
 * @tparam ct_dim: Output width in tiles, 1 to 16.
 * @tparam finalize: Merge the split-accumulation partials, values = <true/false>. Only ct_dim == 1 splits,
 *                   so it has no effect at any other width.
 * @param base_address_meta: L1 address of the meta buffer. The math metas are its first section, 6 bits
 *                           each at a 6-bit stride, five per word.
 * @param face_r_dim: Activation rows per face, 1 or 8.
 * @param dst_index: Tile index in DST that the result is written to.
 * @param kt_dim: Inner dimension in tiles, an even number in [2, 256].
 * @note Call @ref _llk_math_face_compressed_mm_init_ first, with the same ct_dim.
 * @note On the unpack thread, pair with @ref _llk_unpack_AB_face_compressed_mm_.
 */
template <std::uint32_t ct_dim = 1, bool finalize = true>
inline void _llk_math_face_compressed_mm_(
    const std::uint32_t base_address_meta, const std::uint32_t face_r_dim, const std::uint32_t dst_index, const std::uint32_t kt_dim)
{
    static_assert(ct_dim >= 1 && ct_dim <= 16, "face_compressed_mm (math): ct_dim must be in [1, 16]");
    LLK_ASSERT(kt_dim >= 2 && kt_dim <= 256 && kt_dim % 2 == 0, "face_compressed_mm (math): kt_dim must be an even number in [2, 256]");
    LLK_ASSERT(face_r_dim == 1 || face_r_dim == 8, "face_compressed_mm (math): unsupported face_r_dim (expected 1 or 8)");

    hal_port::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);

    // Geometry of the math meta section: one 6-bit meta per four faces, packed at its own width, so five
    // fit per word with two bits spare. The unpack index words share a bit between neighbours to fit six;
    // these do not overlap.
    constexpr std::uint32_t meta_index_bits  = 6;
    constexpr std::uint32_t meta_stride_bits = 6;
    constexpr std::uint32_t meta_index_mask  = (1u << meta_index_bits) - 1;
    constexpr std::uint32_t metas_per_word   = 5;
    static_assert((metas_per_word - 1) * meta_stride_bits + meta_index_bits <= 32, "five 6-bit metas at a 6-bit stride must fit one 32-bit word");

    const std::uint32_t iters      = kt_dim * ct_dim;
    const std::uint32_t full_iters = iters / metas_per_word; // whole meta words
    const std::uint32_t rem_iters  = iters % metas_per_word; // metas in the trailing word
    const std::uint32_t* meta_ptr  = reinterpret_cast<const std::uint32_t*>(base_address_meta);

    constexpr auto tables                   = _llk_math_face_compressed_mm_tables_<ct_dim>();
    constexpr const std::uint32_t* l1_table = tables[0];
    constexpr const std::uint32_t* l2_table = tables[1];

    // HAL-GAP: no public HAL call issues a runtime-selected, pre-encoded instruction word (the decode-table replay and
    // MOP handles); TT_INSN kept as production.
    for (std::uint32_t i = 0; i < full_iters; ++i)
    {
        std::uint32_t meta = meta_ptr[i];

        std::uint32_t idx0 = (meta >> (0 * meta_stride_bits)) & meta_index_mask;
        std::uint32_t idx1 = (meta >> (1 * meta_stride_bits)) & meta_index_mask;
        std::uint32_t idx2 = (meta >> (2 * meta_stride_bits)) & meta_index_mask;
        std::uint32_t idx3 = (meta >> (3 * meta_stride_bits)) & meta_index_mask;
        std::uint32_t idx4 = (meta >> (4 * meta_stride_bits)) & meta_index_mask;

        TT_INSN(l1_table[idx0]);
        TT_INSN(l2_table[idx0]);
        TT_INSN(l1_table[idx1]);
        TT_INSN(l2_table[idx1]);
        TT_INSN(l1_table[idx2]);
        TT_INSN(l2_table[idx2]);
        TT_INSN(l1_table[idx3]);
        TT_INSN(l2_table[idx3]);
        TT_INSN(l1_table[idx4]);
        TT_INSN(l2_table[idx4]);
    }
    std::uint32_t meta = meta_ptr[full_iters];
    for (std::uint32_t i = 0; i < rem_iters; ++i)
    {
        std::uint32_t idx0 = meta & meta_index_mask;
        TT_INSN(l1_table[idx0]);
        TT_INSN(l2_table[idx0]);
        meta >>= meta_stride_bits;
    }

    if constexpr (ct_dim == 1 && finalize)
    {
        namespace sync = hal::sync;
        namespace move = hal::move;
        namespace fpu  = hal::fpu;

        // Merge each face's split partial into its accumulation.
        constexpr std::uint32_t partial = static_cast<std::uint32_t>(_llk_math_face_compressed_mm_split_acc_partial_rows_);

        // wait for both operands
        sync::wait::stall<sync::StallTarget::Math, sync::StallCondition::MathIdle | sync::StallCondition::SrcAValid | sync::StallCondition::SrcBValid>();
        // Move both partials into SrcB, face 0 at row 0 and face 1 at row 8.
        // ADDR_MOD_0 advances dest by one face between the two, and ADDR_MOD_2 resets it afterwards.
        // TTI_MOVD2B(DEST_NORM, src = SrcB row, addrmod, MOV_4_ROWS, dst = Dest row)
        move::run<move::Transfer<move::Dst, move::SrcB> {
            .source_row = partial, .destination_row = 0, .address_mode = ADDR_MOD_0, .number_of_rows = move::rows::Four}>();
        move::run<move::Transfer<move::Dst, move::SrcB> {
            .source_row = partial, .destination_row = partial, .address_mode = ADDR_MOD_2, .number_of_rows = move::rows::Four}>();
        if (face_r_dim == 8)
        {
            // Rows 4-7 of each partial, walking the counters the same way as the pair above.
            move::run<move::Transfer<move::Dst, move::SrcB> {
                .source_row = partial + 4, .destination_row = 4, .address_mode = ADDR_MOD_0, .number_of_rows = move::rows::Four}>();
            move::run<move::Transfer<move::Dst, move::SrcB> {
                .source_row = partial + 4, .destination_row = partial + 4, .address_mode = ADDR_MOD_2, .number_of_rows = move::rows::Four}>();
        }
        // Add each partial into its accumulation, at dest 0 and 16, with ADDR_MOD_4 stepping SrcB from the
        // first to the second in between.
        fpu::run<fpu::Elementwise {
            .operation        = fpu::ElementwiseOperation::Add,
            .broadcast        = fpu::SrcBBroadcast::None,
            .dest_write       = fpu::DestWriteMode::Accumulate,
            .address_modifier = ADDR_MOD_4,
            .dest_row_offset  = 0,
            .release          = fpu::SourceRelease::None}>();
        fpu::run<fpu::Elementwise {
            .operation        = fpu::ElementwiseOperation::Add,
            .broadcast        = fpu::SrcBBroadcast::None,
            .dest_write       = fpu::DestWriteMode::Accumulate,
            .address_modifier = ADDR_MOD_2,
            .dest_row_offset  = 16,
            .release          = fpu::SourceRelease::Both}>();
    }
}
