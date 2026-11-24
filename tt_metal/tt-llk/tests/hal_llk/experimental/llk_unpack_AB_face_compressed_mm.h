// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// HAL port of tt_llk_blackhole/llk_lib/experimental/llk_unpack_AB_face_compressed_mm.h. Same instruction sequence and
// access kinds as production, expressed through the HAL, except the HAL-GAPs marked below.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "cunpack_common.h"
#include "hal/cfg.h"
#include "hal/mop.h"
#include "hal/replay.h"
#include "hal/sync.h"
#include "hal/tensix_counters.h"
#include "hal/unpack.h"
#include "llk_assert.h"
#include "llk_unpack_common.h"

using namespace ckernel;
using namespace ckernel::unpacker;

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

// p_setadc / mutex::THREAD2_ADC as a HAL mutex.
inline constexpr hal::sync::Mutex THREAD2_ADC_MUTEX = static_cast<hal::sync::Mutex>(ckernel::mutex::THREAD2_ADC);

// Unpacker 0 tile descriptor Z dim, low byte only: production's cfg_reg_rmw_tensix<TileDescriptor_ADDR32 + 1, 16, 0xFF0000>
// touches byte 2 alone (one RMWCIB), while the HAL's 16-bit TileDescriptor.ZDim would also rewrite byte 3.
inline constexpr hal::cfg::Field face_compressed_zdim_lo8 {hal::cfg::RegisterScope::State, 32, 64, 1, 16, 8, 2, 1536};
static_assert(face_compressed_zdim_lo8.addr32(hal::cfg::Sec::S0) == THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1, "Z-dim word");
static_assert(face_compressed_zdim_lo8.mask(hal::cfg::Sec::S0) == 0xFF0000, "Z-dim low byte");

// TT_OP_MOP(p_mop::MASK_LOOP, count - 1, mask): a Template 0 run handle for the decode tables.
constexpr std::uint32_t face_compressed_mop(const std::uint32_t count, const std::uint32_t mask)
{
    return hal::mop::Runner<hal::mop::MopTemplate::Template0> {.count = count, .mask = mask}.operation();
}

// TTI_UNPACR_COMMON_EXPLICIT_CONTEXT_AND_COUNTER(SrcA, 0b00'00'01'00 /*Ch0Y += 1*/, cfg_context, adc_context, 1)
constexpr std::uint32_t face_compressed_unpack_weight(const std::uint8_t cfg_context, const std::uint8_t adc_context)
{
    return hal::unpack::get_operation(hal::unpack::DataTransfer {
        .engine     = hal::unpack::Engine::Unpacker0,
        .increments = {.channel0 = {.y = 1}},
        .context    = hal::unpack::ContextSelection::explicit_context(cfg_context, adc_context),
        .handoff    = hal::unpack::SourceHandoff::FlipAndSetDataValid,
    });
}

// TTI_UNPACR_COMMON(SrcB, 0b00'00'10'10 /*Ch0Y += 2, Ch0Z += 2*/, 1)
inline constexpr std::uint32_t face_compressed_unpack_activation = hal::unpack::get_operation(hal::unpack::DataTransfer {
    .engine     = hal::unpack::Engine::Unpacker1,
    .increments = {.channel0 = {.y = 2, .z = 2}},
    .context    = hal::unpack::ContextSelection::explicit_context(0, 0),
    .handoff    = hal::unpack::SourceHandoff::FlipAndSetDataValid,
});

} // namespace hal_port

// The unpack code sequence: one character is one instruction, recorded into the replay buffer at init (see
// instr_for_code in _llk_unpack_AB_face_compressed_mm_mop_config_ for the encodings):
//
//   0   unpack the next weight face through the bfp2 context (UNPACR into SrcA, Ch0Y += 1)
//   1   the same through the bfp4 context
//   S   CLR_SRC, which has to precede any face whose format differs from the one before it
//   B   unpack the next activation block into SrcB (Ch0Y += 2, Ch0Z += 2)
//
// The header table asks for 0, 1, S0, S1, B0 and B1, issuing SB0 and SB1 through the MOP instead; the
// pairs table asks for the sixteen three-transition concatenations its four face bits can produce, from
// 000 up to S1S0S1. All twenty-three fragments are substrings of the thirty-two characters below, which is
// the entire replay buffer -- on this thread nothing else uses it, so it starts at offset 0.
constexpr auto _llk_unpack_AB_face_compressed_mm_seq_ =
    hal_port::code_seq::make_sequence<0 /* replay_base */, ckernel::REPLAY_BUF_SIZE>("SB0B1S0S1S0S111S00S1S000S11S0S11");

// Unpack meta layout, in the index-word section of the meta buffer.
//
// One 32-bit index word packs six 6-bit metas at a stride of five bits, so adjacent windows overlap by one
// bit: meta k is (word >> 5k) & 0b111111, and its bit 0 is the very bit that held meta k-1's last face.
// That shared bit is what buys the sixth meta per word. Within a window:
//
//   bit 0     format of the preceding face; in word 0 the producer replicates this meta's own first face,
//             so the first meta never opens with a format switch
//   bit 1     whether this meta starts an activation block, in which case it also loads the group's SrcB;
//             a meta holds either one first-in-block face or none, never two
//   bits 2-5  formats of this meta's four faces, 0 bfp2 and 1 bfp4
//
// The two tables below decode overlapping slices of it: the header table takes bits 0-2 and issues the
// transition into face 1 plus the optional B, the pairs table takes bits 2-5 as three sliding 2-bit windows
// and issues faces 2-4. Bit 5 therefore belongs to meta k+1 as much as to meta k, so masking the window
// with 0b11111 would silently unpack every fourth face as bfp2.
//
// The shared bit is a producer invariant nothing on this side can check: word w bit 0 must equal word w-1
// bit 30, the two spellings of that one face format. encode_meta in test_matmul_face_compressed.py upholds
// it; a hand-built meta that does not corrupts silently.

constexpr auto _llk_unpack_AB_face_compressed_mm_header_table_ = hal_port::code_seq::make_table<_llk_unpack_AB_face_compressed_mm_seq_, 64>(
    [](const std::uint32_t m, auto find) -> std::uint32_t
    {
        // A face's bit is its unpacker context: 0 bfp2, 1 bfp4. The previous face rides along so the
        // two can be compared -- when the format changes, an S (CLR_SRC) has to be issued first to
        // stall the unpacker before the context switches. The header bit adds the group's SrcB unpack.
        const std::uint32_t face0  = m & 0b1;        // previous face
        const std::uint32_t header = (m >> 1) & 0b1; // also unpack SrcB
        const std::uint32_t face1  = (m >> 2) & 0b1; // this face
        switch ((header << 2) | (face1 << 1) | face0)
        {
            case 0b000:
                return find("0"); // bfp2 -> bfp2
            case 0b001:
                return find("S0"); // bfp4 -> bfp2
            case 0b010:
                return find("S1"); // bfp2 -> bfp4
            case 0b011:
                return find("1"); // bfp4 -> bfp4
            case 0b100:
                return find("B0"); // bfp2 -> bfp2, with B
            case 0b101:
                return hal_port::face_compressed_mop(1, 0x0); // bfp4 -> bfp2, with B (SB0): TT_OP_MOP(MASK_LOOP, 0, 0x0)
            case 0b110:
                return hal_port::face_compressed_mop(1, 0x1); // bfp2 -> bfp4, with B (SB1): TT_OP_MOP(MASK_LOOP, 0, 0x1)
            case 0b111:
                return find("B1"); // bfp4 -> bfp4, with B
            default:
                return 0;
        }
    });

constexpr auto _llk_unpack_AB_face_compressed_mm_pairs_table_ = hal_port::code_seq::make_table<_llk_unpack_AB_face_compressed_mm_seq_, 64>(
    [](const std::uint32_t m, auto find) -> std::uint32_t
    {
        // faces1234 holds the group's four face formats, same encoding as the header table. That table
        // issued face1, so this one issues faces 2..4 and concatenates their fragments into a single
        // handle. Every transition needs its predecessor to decide the stall, so the 2-bit windows
        // overlap -- low bit the previous face, high bit this face -- and the shift is by one, not two.
        auto pair_frag = [](std::uint32_t transition) -> const char*
        {
            switch (transition)
            {
                case 0b00:
                    return "0"; // bfp2 -> bfp2
                case 0b01:
                    return "S0"; // bfp4 -> bfp2
                case 0b10:
                    return "S1"; // bfp2 -> bfp4
                case 0b11:
                    return "1"; // bfp4 -> bfp4
                default:
                    return "";
            }
        };
        std::uint32_t faces1234 = m >> 2;
        char needle[8]          = {};
        std::uint32_t len       = 0;
        for (std::uint32_t i = 0; i < 3; ++i)
        {
            for (const char* p = pair_frag(faces1234 & 0x3); *p != '\0'; ++p)
            {
                needle[len++] = *p;
            }
            faces1234 >>= 1;
        }
        return find(needle);
    });

/**
 * @brief Record the unpack code sequence into the replay buffer and program the MOP that replays it.
 *
 * One instruction per character: '0' and '1' unpack a bfp2 or bfp4 face into SrcA, 'S' is the CLR_SRC stall
 * a format switch needs first, and 'B' unpacks the activation block into SrcB. The decode tables hand the
 * execute loop replay handles naming runs of these.
 *
 * @note Called from @ref _llk_unpack_AB_face_compressed_mm_init_ rather than directly.
 */
inline void _llk_unpack_AB_face_compressed_mm_mop_config_()
{
    constexpr std::uint32_t unpack_bfp2 = hal_port::face_compressed_unpack_weight(0 /*cfg_context*/, 1 /*adc_context*/);
    constexpr std::uint32_t unpack_bfp4 = hal_port::face_compressed_unpack_weight(1 /*cfg_context*/, 2 /*adc_context*/);
    constexpr std::uint32_t unpack_b    = hal_port::face_compressed_unpack_activation;

    auto instr_for_code = [](char code)
    {
        switch (code)
        {
            case '0':
                TTI_INSN(unpack_bfp2);
                break; // Ch0Y += 1
            case '1':
                TTI_INSN(unpack_bfp4);
                break; // Ch0Y += 1
            case 'S':
                // HAL-GAP: hal::nop encodes only the pure-delay UNPACR_NOP; the clear-source flavor (CLR_SRC) has no HAL
                // descriptor, so the production encoding is kept.
                TTI_UNPACR_NOP(SrcA, 0, 0, 0, 0, 1, 0, 0, p_unpacr_nop::CLR_SRC);
                break;
            case 'B':
                TTI_INSN(unpack_b);
                break; // Ch0Y += 2 Ch0Z += 2
            default:
                LLK_ASSERT(false, "Invalid code for unpack instruction");
                break;
        }
    };

    _llk_unpack_AB_face_compressed_mm_seq_.load(instr_for_code);

    constexpr std::uint32_t SB  = _llk_unpack_AB_face_compressed_mm_seq_.fragment("SB");
    constexpr std::uint32_t op0 = _llk_unpack_AB_face_compressed_mm_seq_.fragment("0");
    constexpr std::uint32_t op1 = _llk_unpack_AB_face_compressed_mm_seq_.fragment("1");

    // ckernel_unpack_template(unpackB = true, unpackHalo = false, A = SB, A1-3 = NOP, skipA = SB, B = op0, skipB = op1).
    // The HAL leaves the unused InsnA1-3 words unwritten (the expander does not read them without unpackHalo).
    using namespace hal::mop;
    program(foreach_mask_bit(if_clear(SB, op0), otherwise(SB, op1)));
}

// The two BFP precisions need their own address counters so one instruction stream can alternate format
// face to face, and the unpacker has only one set of its own. It borrows other threads' ADC sets via the
// SETADC thread-id override: bfp2 counts in the math thread's set, bfp4 in the pack thread's. That is why
// every SETADC* site below holds mutex::THREAD2_ADC -- see ckernel_common_ops.h for the override macros.

/**
 * @brief Configure the unpack thread for a face-granular compressed matmul.
 *
 * @tparam transpose: Haloize the SrcA read, values = <true/false>
 * @param unpB_face_r_dim: Activation rows per face, 1 or 8. Sets unpacker 1's X end.
 * @note Call this before @ref _llk_unpack_AB_face_compressed_mm_, and
 *       @ref _llk_unpack_AB_face_compressed_mm_uninit_ after the last one, to put back the tile descriptor
 *       num_faces this forces to a single face.
 * @note On the math thread, pair with @ref _llk_math_face_compressed_mm_init_.
 */
template <bool transpose = false>
inline void _llk_unpack_AB_face_compressed_mm_init_(const std::uint32_t unpB_face_r_dim)
{
    // Two-context handshake op: an earlier op's init may have left Auto TTSync on, and tt-metal calls no uninit.
    hal_port::unpack_sync_mode<hal_port::UnpackSyncMode::Handshake>();
    LLK_ASSERT(unpB_face_r_dim == 1 || unpB_face_r_dim == 8, "face_compressed_mm (unpack): unsupported activation face_r_dim (expected 1 or 8)");

    namespace cfg = hal::cfg;
    namespace adc = hal::counters::adc;
    using cfg::Sec;

    constexpr auto R2 = cfg::Thcon[cfg::Reg2];
    constexpr auto R5 = cfg::Thcon[cfg::Reg5];
    constexpr auto R7 = cfg::Thcon[cfg::Reg7];

    constexpr std::uint32_t bfp2 = static_cast<std::uint32_t>(DataFormat::Bfp2_b);
    constexpr std::uint32_t bfp4 = static_cast<std::uint32_t>(DataFormat::Bfp4_b);

    // The config writes below need no preceding stall. The unpacker samples these fields when it issues an
    // UNPACR, not while the unpack runs, so they are free to change as soon as the previous UNPACR has been
    // issued -- which program order on this thread already guarantees.
    // Haloize_mode and Ovrd_data_format (read dataformat from per cntx registers) share one byte: one RMWCIB.
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::set<R2.Haloize_mode, Sec::S0, (transpose ? 1u : 0u)>(), cfg::set<R2.Ovrd_data_format, Sec::S0, 1>());

    // Input and output format of one context share one byte: one RMWCIB per context.
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::set<R7.Unpack_data_format_cntx0, Sec::S0, bfp2>(), cfg::set<R7.Unpack_out_data_format_cntx0, Sec::S0, bfp2>());
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::set<R7.Unpack_data_format_cntx1, Sec::S0, bfp4>(), cfg::set<R7.Unpack_out_data_format_cntx1, Sec::S0, bfp4>());
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::set<R7.Unpack_data_format_cntx2, Sec::S0, bfp2>(), cfg::set<R7.Unpack_out_data_format_cntx2, Sec::S0, bfp2>());
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::set<R7.Unpack_data_format_cntx3, Sec::S0, bfp4>(), cfg::set<R7.Unpack_out_data_format_cntx3, Sec::S0, bfp4>());

    // Replicate onto cntx2/3 what hw_configure programs for cntx0/1, the only contexts used before. Both
    // values come from there: the x_dim through the same canonical helper, the dest address as a literal.
    // Each word packs the cntx2 value low and the cntx3 value high (whole-word RMWCIB, as production).
    constexpr std::uint32_t canonical_x_dim_cntx = canonical_unpA_tile_x_dim_cntx(FACE_R_DIM);
    static_assert(R5.Tile_x_dim_cntx2.addr32(Sec::S0) == THCON_SEC0_REG5_Tile_x_dim_cntx2_ADDR32, "Tile_x_dim cntx2/3 word");
    static_assert(R5.Dest_cntx2_address.addr32(Sec::S0) == THCON_SEC0_REG5_Dest_cntx2_address_ADDR32, "Dest cntx2/3 word");
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<R5.Tile_x_dim_cntx2, Sec::S0, (canonical_x_dim_cntx & 0xffff)>(), cfg::set<R5.Tile_x_dim_cntx3, Sec::S0, (canonical_x_dim_cntx >> 16)>());
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::set<R5.Dest_cntx2_address, Sec::S0, 4 * 16>(), cfg::set<R5.Dest_cntx3_address, Sec::S0, 4 * 16>());

    // override z dim
    cfg::write<cfg::Access::TensixCfgUnit, hal_port::face_compressed_zdim_lo8, Sec::S0, 1>();

    _llk_unpack_AB_face_compressed_mm_mop_config_();

    const std::uint32_t unpB_x_end = 4 * unpB_face_r_dim * FACE_C_DIM - 1;

    hal::sync::mutex::acquire<hal_port::THREAD2_ADC_MUTEX>();
    // the thread override covers CH0 XY and CH1 X, CH0 ZW stays shared
    // reset the borrowed counters, bfp2's in the math set and bfp4's in the pack set
    // HAL-GAP: hal::counters::adc does not support the SETADC* thread-id override; production lines kept.
    TTI_SETADCXY_THREAD_OVERRIDE(p_setadc::UNP_AB, p_setadc::THREAD_OVRD_MATH, 0, 0, 0, 0, SETADC_CH01(p_setadc::XY));
    TTI_SETADCXY_THREAD_OVERRIDE(p_setadc::UNP_AB, p_setadc::THREAD_OVRD_PACK, 0, 0, 0, 0, SETADC_CH01(p_setadc::XY));
    // unp1's own set, unp0's rides along: TTI_SETADCXY(UNP_AB, 0, 0, 0, 0, 0b1111)
    adc::set<adc::AddrCounter {.channel0 = {.x = 0, .y = 0}, .channel1 = {.x = 0, .y = 0}}.for_clients<adc::Client::Unpacker0, adc::Client::Unpacker1>()>();
    // ZW has no override, both formats share it: TTI_SETADCZW(UNP_AB, 0, 0, 0, 0, 0b1111)
    adc::set<adc::AddrCounter {.channel0 = {.z = 0, .w = 0}, .channel1 = {.z = 0, .w = 0}}.for_clients<adc::Client::Unpacker0, adc::Client::Unpacker1>()>();
    // set CH1 X for both unpackers; bfp4 and bfp2 sit in different thread overrides, so both get set
    // HAL-GAP: hal::counters::adc does not support the SETADC thread-id override; production lines kept.
    constexpr std::uint32_t unpA_x_end = FACE_R_DIM * FACE_C_DIM - 1;
    TTI_SETADC_THREAD_OVERRIDE(p_setadc::UNP0, p_setadc::CH_1, p_setadc::SET_X, p_setadc::THREAD_OVRD_MATH, unpA_x_end);
    TTI_SETADC_THREAD_OVERRIDE(p_setadc::UNP0, p_setadc::CH_1, p_setadc::SET_X, p_setadc::THREAD_OVRD_PACK, unpA_x_end);
    // TT_SETADCXX(UNP_B, unpB_x_end, 0x0)
    adc::set(adc::AddrCounter {.channel0 = {.x = 0}, .channel1 = {.x = unpB_x_end}}.for_clients<adc::Client::Unpacker1>());
    hal::sync::mutex::release<hal_port::THREAD2_ADC_MUTEX>();
}

/**
 * @brief Restore the unpacker state the init changed.
 *
 * @param unpA_num_faces: Faces per tile to put back in unpacker 0's tile descriptor, which
 *                        @ref _llk_unpack_AB_face_compressed_mm_init_ forced to a single face.
 * @note Call after the last @ref _llk_unpack_AB_face_compressed_mm_.
 */
inline void _llk_unpack_AB_face_compressed_mm_uninit_(const std::uint32_t unpA_num_faces)
{
    namespace cfg = hal::cfg;
    namespace adc = hal::counters::adc;

    // No stall needed here either, same as the init: the last UNPACR sampled these fields when it issued.
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg2].Ovrd_data_format, cfg::Sec::S0, 0>();
    cfg::write<cfg::Access::TensixCfgUnit, hal_port::face_compressed_zdim_lo8, cfg::Sec::S0>(unpA_num_faces);

    hal::sync::mutex::acquire<hal_port::THREAD2_ADC_MUTEX>();
    // reset CH0 XY counters for both unpackers to leave them in a clean state: TTI_SETADCXY(UNP_AB, 0, 0, 0, 0, 0b1111)
    adc::set<adc::AddrCounter {.channel0 = {.x = 0, .y = 0}, .channel1 = {.x = 0, .y = 0}}.for_clients<adc::Client::Unpacker0, adc::Client::Unpacker1>()>();
    hal::sync::mutex::release<hal_port::THREAD2_ADC_MUTEX>();
}

/**
 * @brief Unpack the activation block into SrcB and the compressed weight faces into SrcA.
 *
 * Streams the meta buffer's index words, pushing one replay handle pair per meta, and reloads the
 * double-buffered unpacker contexts with the next chunk's weight base addresses as it goes.
 *
 * @tparam ct_dim: Output width in tiles, 1 to 16.
 * @tparam clear_src: Clear SrcB before the first unpack, values = <true/false>
 * @tparam finalize: For ct_dim == 1, leave both sources zeroed and valid so the math thread can merge its
 *                   split-accumulation partials, values = <true/false>
 * @param base_address_b: SrcB base address, which is the activation CB's read pointer.
 * @param base_address_meta: L1 address of the meta buffer. This thread reads the weight base addresses and
 *                           the index words that follow the math metas.
 * @param kt_dim: Inner dimension in tiles, an even number in [2, 256].
 * @note Call @ref _llk_unpack_AB_face_compressed_mm_init_ first.
 * @note On the math thread, pair with @ref _llk_math_face_compressed_mm_.
 */
template <std::uint32_t ct_dim = 1, bool clear_src = true, bool finalize = true>
inline void _llk_unpack_AB_face_compressed_mm_(const std::uint32_t base_address_b, const std::uint32_t base_address_meta, const std::uint32_t kt_dim)
{
    static_assert(ct_dim >= 1 && ct_dim <= 16, "face_compressed_mm (unpack): ct_dim must be in [1, 16]");
    LLK_ASSERT(kt_dim >= 2 && kt_dim <= 256 && kt_dim % 2 == 0, "face_compressed_mm (unpack): kt_dim must be an even number in [2, 256]");

    namespace hcfg = hal::cfg;
    namespace adc  = hal::counters::adc;
    namespace sync = hal::sync;

    // A meta address word packs two fields: the low 24 bits are a 16B-word base address, written to one
    // unpacker context's Base_address, and the top byte is a Y offset, applied as that context's SET_Y.
    constexpr std::uint32_t meta_addr_base_mask   = 0x00FFFFFF;
    constexpr std::uint32_t meta_addr_y_off_shift = 24;

    // Geometry of the index words, per the layout block above the decode tables. The stride is narrower
    // than the meta because consecutive metas share a bit, which is what fits six of them in one word.
    constexpr std::uint32_t meta_index_bits      = 6;
    constexpr std::uint32_t meta_stride_bits     = 5;
    constexpr std::uint32_t meta_index_mask      = (1u << meta_index_bits) - 1;
    constexpr std::uint32_t metas_per_index_word = 6;
    static_assert((metas_per_index_word - 1) * meta_stride_bits + meta_index_bits <= 32, "six 6-bit metas at a five-bit stride must fit one 32-bit index word");

    // Unpacker 0 base address words of contexts 0-3 and unpacker 1 context 0, written by whole-word MMIO stores into
    // the state-CFG bank, as production.
    constexpr std::uint32_t base_cntx0   = hcfg::Unpacker[0].Cntx[0].Base.addr32(hcfg::Sec::S0);
    constexpr std::uint32_t base_cntx1   = hcfg::Unpacker[0].Cntx[1].Base.addr32(hcfg::Sec::S0);
    constexpr std::uint32_t base_cntx2   = hcfg::Unpacker[0].Cntx[2].Base.addr32(hcfg::Sec::S0);
    constexpr std::uint32_t base_cntx3   = hcfg::Unpacker[0].Cntx[3].Base.addr32(hcfg::Sec::S0);
    constexpr std::uint32_t base_b_cntx0 = hcfg::Unpacker[1].Cntx[0].Base.addr32(hcfg::Sec::S0);
    static_assert(base_cntx0 == THCON_SEC0_REG3_Base_address_ADDR32 && base_cntx1 == THCON_SEC0_REG3_Base_cntx1_address_ADDR32, "cntx0/1 base words");
    static_assert(base_cntx2 == THCON_SEC0_REG3_Base_cntx2_address_ADDR32 && base_cntx3 == THCON_SEC0_REG3_Base_cntx3_address_ADDR32, "cntx2/3 base words");
    static_assert(base_b_cntx0 == THCON_SEC1_REG3_Base_address_ADDR32, "unpacker 1 base word");

    // Apply the Y offsets of a consecutive bfp2/bfp4 pair of meta address words to the two borrowed Y
    // counters, so each context reads from the right row of its base. bfp2 counts in the math thread's ADC
    // set and bfp4 in the pack thread's, so each needs its own write. Caller holds the mutex.
    auto set_y_off = [](const std::uint32_t* addr_pair)
    {
        const std::uint32_t bfp2_y = addr_pair[0] >> meta_addr_y_off_shift;
        const std::uint32_t bfp4_y = addr_pair[1] >> meta_addr_y_off_shift;
        // HAL-GAP: hal::counters::adc does not support the SETADC thread-id override; production lines kept.
        TT_SETADC_THREAD_OVERRIDE(p_setadc::UNP0, p_setadc::CH_0, p_setadc::SET_Y, p_setadc::THREAD_OVRD_MATH, bfp2_y);
        TT_SETADC_THREAD_OVERRIDE(p_setadc::UNP0, p_setadc::CH_0, p_setadc::SET_Y, p_setadc::THREAD_OVRD_PACK, bfp4_y);
    };

    // Switch the context pair the UNPACRs take their base address and format from:
    // TTI_SETC16(UNPACK_MISC_CFG_CfgContextOffset_0_ADDR32, offset), every other field of the word zero.
    auto set_cfg_context_offset_alt = []
    {
        hcfg::write<hcfg::Access::TensixCfgUnit>(
            hcfg::set<hcfg::UnpackMiscCfg::CfgContextOffset_0, hcfg::Sec::S0, 2 /*cntx2 / cntx3*/>(),
            hcfg::set<hcfg::UnpackMiscCfg::CfgContextOffset_1, hcfg::Sec::S0, 0>());
    };
    auto set_cfg_context_offset_base = []
    {
        hcfg::write<hcfg::Access::TensixCfgUnit>(
            hcfg::set<hcfg::UnpackMiscCfg::CfgContextOffset_0, hcfg::Sec::S0, 0 /*cntx0 / cntx1*/>(),
            hcfg::set<hcfg::UnpackMiscCfg::CfgContextOffset_1, hcfg::Sec::S0, 0>());
    };

    // HAL-GAP: no public HAL call issues a runtime-selected, pre-encoded instruction word (the decode-table replay and
    // MOP handles); TT_INSN kept as production.
    auto emit_word = [](std::uint32_t meta)
    {
        std::uint32_t idx0 = (meta >> (0 * meta_stride_bits)) & meta_index_mask;
        std::uint32_t idx1 = (meta >> (1 * meta_stride_bits)) & meta_index_mask;
        std::uint32_t idx2 = (meta >> (2 * meta_stride_bits)) & meta_index_mask;
        std::uint32_t idx3 = (meta >> (3 * meta_stride_bits)) & meta_index_mask;
        std::uint32_t idx4 = (meta >> (4 * meta_stride_bits)) & meta_index_mask;
        std::uint32_t idx5 = (meta >> (5 * meta_stride_bits)) & meta_index_mask;

        TT_INSN(_llk_unpack_AB_face_compressed_mm_header_table_[idx0]);
        TT_INSN(_llk_unpack_AB_face_compressed_mm_pairs_table_[idx0]);
        TT_INSN(_llk_unpack_AB_face_compressed_mm_header_table_[idx1]);
        TT_INSN(_llk_unpack_AB_face_compressed_mm_pairs_table_[idx1]);
        TT_INSN(_llk_unpack_AB_face_compressed_mm_header_table_[idx2]);
        TT_INSN(_llk_unpack_AB_face_compressed_mm_pairs_table_[idx2]);
        TT_INSN(_llk_unpack_AB_face_compressed_mm_header_table_[idx3]);
        TT_INSN(_llk_unpack_AB_face_compressed_mm_pairs_table_[idx3]);
        TT_INSN(_llk_unpack_AB_face_compressed_mm_header_table_[idx4]);
        TT_INSN(_llk_unpack_AB_face_compressed_mm_pairs_table_[idx4]);
        TT_INSN(_llk_unpack_AB_face_compressed_mm_header_table_[idx5]);
        TT_INSN(_llk_unpack_AB_face_compressed_mm_pairs_table_[idx5]);
    };

    volatile std::uint32_t tt_reg_ptr* cfg = hcfg::detail::state_cfg_bank();

    // per tile math meta is 6 bits, 5 entries fit into 32bits, round up
    const std::uint32_t math_meta_size = (kt_dim * ct_dim + 4) / 5;
    const std::uint32_t* pre_meta_ptr  = reinterpret_cast<std::uint32_t*>(base_address_meta) + math_meta_size;
    const std::uint32_t iters          = pre_meta_ptr[0];
    const std::uint32_t full_iters     = iters / metas_per_index_word; // whole index words
    const std::uint32_t rem_iters      = iters % metas_per_index_word; // metas in the trailing word
    const std::uint32_t full_blocks    = full_iters / 16;
    const bool odd_block               = (full_iters % 16) >= 8;
    const std::uint32_t* meta_ptr      = pre_meta_ptr + 1 + 2 * (full_iters / 8) + 2;

    hal_port::wait_for_next_context(1);
    hal_port::reset_config_context();

    if constexpr (clear_src)
    {
        // HAL-GAP: hal::nop encodes only the pure-delay UNPACR_NOP; the clear-source flavor (CLR_SRC) has no HAL
        // descriptor, so the production encoding is kept.
        TTI_UNPACR_NOP(SrcB, 0, 0, 0, 0, 0, 1, 0, p_unpacr_nop::CLR_SRC);
    }
    cfg[base_cntx0] = pre_meta_ptr[1] & meta_addr_base_mask;
    cfg[base_cntx1] = pre_meta_ptr[2] & meta_addr_base_mask;
    sync::mutex::acquire<hal_port::THREAD2_ADC_MUTEX>();
    set_y_off(pre_meta_ptr + 1);
    sync::mutex::release<hal_port::THREAD2_ADC_MUTEX>();
    cfg[base_b_cntx0] = base_address_b;

    // Trisc::SEMPOST for context acquire (MMIO)
    sync::semaphore::post<sync::Access::MMIO, hal_port::UNPACK_SYNC_SEM>();

    // Stall unpacker until pending CFG writes from Trisc have completed
    sync::wait::stall<sync::StallTarget::Unpack, sync::StallCondition::RiscvAccessProcessed>();

    std::uint32_t c = 0;

    for (std::uint32_t b = 0; b < full_blocks; ++b)
    {
        for (std::uint32_t i = 0; i < 4; ++i, ++c)
        {
            emit_word(meta_ptr[c]);
        }
        cfg[base_cntx2] = pre_meta_ptr[3 + 4 * b] & meta_addr_base_mask;
        cfg[base_cntx3] = pre_meta_ptr[4 + 4 * b] & meta_addr_base_mask;
        for (std::uint32_t i = 0; i < 4; ++i, ++c)
        {
            emit_word(meta_ptr[c]);
        }
        set_cfg_context_offset_alt();
        sync::mutex::acquire<hal_port::THREAD2_ADC_MUTEX>();
        set_y_off(pre_meta_ptr + 3 + 4 * b);
        sync::mutex::release<hal_port::THREAD2_ADC_MUTEX>();
        for (std::uint32_t i = 0; i < 4; ++i, ++c)
        {
            emit_word(meta_ptr[c]);
        }
        cfg[base_cntx0] = pre_meta_ptr[5 + 4 * b] & meta_addr_base_mask;
        cfg[base_cntx1] = pre_meta_ptr[6 + 4 * b] & meta_addr_base_mask;
        for (std::uint32_t i = 0; i < 4; ++i, ++c)
        {
            emit_word(meta_ptr[c]);
        }
        set_cfg_context_offset_base();
        sync::mutex::acquire<hal_port::THREAD2_ADC_MUTEX>();
        set_y_off(pre_meta_ptr + 5 + 4 * b);
        sync::mutex::release<hal_port::THREAD2_ADC_MUTEX>();
    }

    if (odd_block)
    {
        for (std::uint32_t i = 0; i < 4; ++i, ++c)
        {
            emit_word(meta_ptr[c]);
        }
        cfg[base_cntx2] = pre_meta_ptr[3 + 4 * full_blocks] & meta_addr_base_mask;
        cfg[base_cntx3] = pre_meta_ptr[4 + 4 * full_blocks] & meta_addr_base_mask;
        for (std::uint32_t i = 0; i < 4; ++i, ++c)
        {
            emit_word(meta_ptr[c]);
        }
        set_cfg_context_offset_alt();
        sync::mutex::acquire<hal_port::THREAD2_ADC_MUTEX>();
        set_y_off(pre_meta_ptr + 3 + 4 * full_blocks);
        sync::mutex::release<hal_port::THREAD2_ADC_MUTEX>();
    }

    for (; c < full_iters; ++c)
    {
        emit_word(meta_ptr[c]);
    }
    std::uint32_t meta = meta_ptr[full_iters];
    for (std::uint32_t j = 0; j < rem_iters; ++j)
    {
        std::uint32_t idx0 = meta & meta_index_mask;
        TT_INSN(_llk_unpack_AB_face_compressed_mm_header_table_[idx0]); // HAL-GAP: see emit_word
        TT_INSN(_llk_unpack_AB_face_compressed_mm_pairs_table_[idx0]);
        meta >>= meta_stride_bits;
    }

    if constexpr (ct_dim == 1 && finalize)
    {
        sync::wait::stall<sync::StallTarget::Unpack, sync::StallCondition::Unpacker0Idle | sync::StallCondition::Unpacker1Idle>();
        // HAL-GAP: hal::nop encodes only the pure-delay UNPACR_NOP; the zero-source + set-data-valid flavor
        // (UNP_ZEROSRC, SET_DVALID, Stall_Clr_Cntrl) has no HAL descriptor, so the production encoding is kept.
        TTI_UNPACR_NOP(SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, 1, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
        TTI_UNPACR_NOP(SrcA, 0, 0, p_unpacr_nop::SET_DVALID, 0, 1, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
    }

    // T6::SEMGET for context release
    sync::semaphore::get<hal_port::UNPACK_SYNC_SEM>();

    hal_port::wait_for_next_context(1);
    hal_port::reset_config_context();

    // Put the counters back the way init left them, ready for the next call: zero the Y counters the
    // per-face SET_Y writes moved and the shared ZW pair, but not X, so init's CH1 X ends survive.
    sync::mutex::acquire<hal_port::THREAD2_ADC_MUTEX>();
    // HAL-GAP: hal::counters::adc does not support the SETADC* thread-id override; production lines kept.
    TTI_SETADCXY_THREAD_OVERRIDE(p_setadc::UNP_AB, p_setadc::THREAD_OVRD_MATH, 0, 0, 0, 0, SETADC_CH01(p_setadc::Y));
    TTI_SETADCXY_THREAD_OVERRIDE(p_setadc::UNP_AB, p_setadc::THREAD_OVRD_PACK, 0, 0, 0, 0, SETADC_CH01(p_setadc::Y));
    // TTI_SETADCZW(UNP_AB, 0, 0, 0, 0, 0b1111)
    adc::set<adc::AddrCounter {.channel0 = {.z = 0, .w = 0}, .channel1 = {.z = 0, .w = 0}}.for_clients<adc::Client::Unpacker0, adc::Client::Unpacker1>()>();
    sync::mutex::release<hal_port::THREAD2_ADC_MUTEX>();
}
