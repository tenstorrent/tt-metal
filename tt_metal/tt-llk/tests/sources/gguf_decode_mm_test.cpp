// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// GGUF weight decode on the SFPU and the DEST-fed matmul (Blackhole, 32-bit DEST, full sync).

#include <cstdint>

#include "ckernel.h"
#include "experimental/ckernel_gguf_page.h"
#include "llk_defs.h"
#include "params.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

constexpr ckernel::gguf::PageLayout test_layout()
{
    if constexpr (GGUF_FMT == 0)
    {
        return ckernel::gguf::make_layout(1, 1, 4096, 0, 0); // a Float32 tile
    }
    else
    {
        return ckernel::gguf::page_layout<static_cast<ckernel::gguf::Format>(GGUF_FMT)>();
    }
}

constexpr ckernel::gguf::PageLayout LAYOUT = test_layout();

// MXFP4 and NVFP4 pages hold block-float codes, unpacked to the 16-bit DEST rows that the decoded tile's rows address
// (Dst16b rows 64 tile), so the decoded tile is not tile 0: tile 1 (codes in tile 0), with matmul tiles 3 and 5
// (codes in tiles 1 and 2); the page follows.
constexpr bool BFP                         = ckernel::gguf::bfp_codes<GGUF_FMT>;
constexpr std::uint32_t OUT                = BFP ? 1 : 0; // the decoded tile without matmul (GGUF_FMT == 0: the copied tile)
constexpr std::uint32_t PAGE_BYTES         = LAYOUT.rows * 64;
constexpr std::uint32_t PAGE_FACES         = LAYOUT.rows / 16;
constexpr std::uint32_t PAGES_PER_N_TILE   = GGUF_KT / LAYOUT.page_tiles;
constexpr std::uint32_t OUTPUT_TILES       = GGUF_MM ? GGUF_NT : GGUF_NT * GGUF_KT;

// The formats are the LLKs': pages unpacked to DEST as 32-bit words, in0 bf16, Float32 out.
constexpr std::uint32_t PAGE_FORMAT = ckernel::to_underlying(GGUF_FMT == 0 ? DataFormat::Float32 : DataFormat::UInt32);
constexpr std::uint32_t IN0_FORMAT  = ckernel::to_underlying(DataFormat::Float16_b);
constexpr std::uint32_t OUT_FORMAT  = ckernel::to_underlying(DataFormat::Float32);

static_assert(dest_sync == ckernel::DstSync::SyncFull && is_fp32_dest_acc_en, "gguf_decode_mm_test: full sync, 32-bit DEST");
static_assert(GGUF_PAGE_ROWS % 64 == 0 && (GGUF_PAGE_ROWS == 0 || GGUF_MM), "gguf_decode_mm_test: page rows offset with matmul only");
static_assert(!BFP || GGUF_PAGE_ROWS == 0, "gguf_decode_mm_test: no page rows offset with block-float codes");

// buffer_A: the weight pages, page p of N tile nt at p + nt * PAGES_PER_N_TILE, each complete (fp32 scale copies and
// level codes written by the host); GGUF_FMT == 0: each page is a Float32 [K][N] weight tile.
// buffer_B (GGUF_MM): the in0 tiles (bf16, GGUF_IN0_FACES faces), K tile kt at kt.
inline std::uint32_t page_address(const std::uint32_t pages_base, const std::uint32_t nt, const std::uint32_t kt)
{
    return pages_base + (nt * PAGES_PER_N_TILE + kt / LAYOUT.page_tiles) * PAGE_BYTES;
}

#ifdef LLK_TRISC_UNPACK

#include "experimental/llk_unpack_dest_srca_mm.h"
#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(PAGE_FORMAT, IN0_FORMAT, PAGE_FORMAT, IN0_FORMAT, FACE_R_DIM, FACE_R_DIM, PAGE_FACES, GGUF_IN0_FACES);
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, true>(
        0, 0, ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, PAGE_FACES), PAGE_FORMAT, PAGE_FORMAT);
    if constexpr (GGUF_MM)
    {
        _llk_unpack_dest_srca_mm_init_(GGUF_IN0_FACES);
    }

    const std::uint32_t pages_base = params.buffer_A[0];
    const auto unpack_page         = [&](const std::uint32_t nt, const std::uint32_t kt)
    {
        if constexpr (BFP)
        {
            _llk_unpack_dest_srca_mm_codes_page_(
                L1_ADDRESS(page_address(pages_base, nt, kt)),
                ckernel::gguf::BFP_CODES,
                ckernel::gguf::BFP_CODE_EXP,
                LAYOUT.scale_offset,
                ckernel::gguf::bfp_scale_words<GGUF_FMT>(),
                FACE_R_DIM * FACE_C_DIM - 1);
        }
        else
        {
            _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, true>(
                L1_ADDRESS(page_address(pages_base, nt, kt)), PAGE_FORMAT, PAGE_FORMAT);
        }
    };
    for (std::uint32_t nt = 0; nt < GGUF_NT; nt++)
    {
        if constexpr (GGUF_MM)
        {
            unpack_page(nt, 0);
            for (std::uint32_t kt = 0; kt < GGUF_KT; kt++)
            {
                _llk_unpack_dest_srca_mm_(L1_ADDRESS(params.buffer_B[kt]));
                if (kt + 1 < GGUF_KT)
                {
                    unpack_page(nt, kt + 1);
                }
            }
        }
        else
        {
            for (std::uint32_t kt = 0; kt < GGUF_KT; kt++)
            {
                unpack_page(nt, kt);
            }
        }
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "experimental/llk_math_dest_srca_mm.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "sfpu/experimental/ckernel_sfpu_gguf_decode.h"

using namespace ckernel;

// Only the matmul runs copy the next page beside a decode; the others use the decoders for no DEST writer beside.
constexpr bool BESIDE_UNPACK = GGUF_MM;

// group: the first tile of its group decoded at dst_index (sfpu_gguf_decode_group). GGUF_PAGE_ROWS (a multiple of
// 64): the same DEST rows, addressed GGUF_PAGE_ROWS / 64 tiles lower with that row offset. PARAMS, BODY: the phases
// to run (the group step goes with phase 1).
template <bool decode, bool PARAMS = true, bool BODY = true>
inline void decode_tile(const std::uint32_t dst_index, const std::uint32_t kt, const bool group)
{
    if constexpr (decode)
    {
        const std::uint32_t base = dst_index - GGUF_PAGE_ROWS / 64;
        if (PARAMS && group)
        {
            sfpu::sfpu_gguf_decode_group<GGUF_FMT>(base, GGUF_PAGE_ROWS);
        }
        sfpu::sfpu_gguf_decode_tile<GGUF_FMT, PARAMS, BODY, BESIDE_UNPACK>(base, kt % LAYOUT.group_tiles, GGUF_PAGE_ROWS);
    }
}

constexpr bool body_after_copy()
{
    if constexpr (GGUF_FMT == 0)
    {
        return false;
    }
    else
    {
        return sfpu::sfpu_gguf_decode_body_unarbitrated<GGUF_FMT, BESIDE_UNPACK>();
    }
}

// The unpack-to-DEST of the page into tile `page`, with its codes under the rows of tile `dec` when BFP.
inline void page_post(const std::uint32_t page, const std::uint32_t dec)
{
    if constexpr (BFP)
    {
        _llk_math_dest_srca_mm_codes_post_(page, (dec << 6) + 2 * get_dest_buffer_base());
    }
    else
    {
        _llk_math_dest_srca_mm_unpack_post_(page);
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
    _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(OUT_FORMAT, IN0_FORMAT);
    _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>();
    sfpu::sfpu_gguf_decode_init<GGUF_FMT, BESIDE_UNPACK>();
    _llk_math_dest_srca_mm_init_<MATH_FIDELITY, GGUF_IN0_ROWS>();

    for (std::uint32_t nt = 0; nt < GGUF_NT; nt++)
    {
        if constexpr (GGUF_MM)
        {
            // One Float32 output tile per N tile, D += in0[kt] * W[kt][nt] over the K tiles. The K tiles alternate
            // between two slot sets (decoded tile, page, decoder scratch), so the next page is unpacked while this one
            // decodes; a phase 2 whose stores are not arbitrated against the unpacker's DEST writes waits for the copy.
            _llk_math_wait_for_dest_available_<dest_sync>();
            sfpu::sfpu_gguf_zero_tile(0); // the accumulator
            // The decoded tile per set (its page is the next tile, scratch past it).
            const auto out_of = [](const std::uint32_t set) -> std::uint32_t { return BFP ? (set ? 5 : 3) : (set ? 4 : 1); };
            page_post(out_of(0) + 1, out_of(0));
            _llk_math_dest_srca_mm_unpack_wait_(out_of(0) + 1, PAGE_FACES);
            std::uint32_t set = 0;
            for (std::uint32_t kt = 0; kt < GGUF_KT; kt++)
            {
                const std::uint32_t out       = out_of(set);
                const std::uint32_t next_page = out_of(set ^ 1) + 1;
                const bool next               = kt + 1 < GGUF_KT;
                if (next)
                {
                    page_post(next_page, out_of(set ^ 1));
                }
                constexpr bool split = body_after_copy();
                decode_tile<GGUF_FMT != 0, true, !split>(out, kt, kt % LAYOUT.group_tiles < 2); // a group starts in both sets
                if (next)
                {
                    _llk_math_dest_srca_mm_unpack_wait_(next_page, PAGE_FACES);
                }
                if constexpr (split)
                {
                    decode_tile<GGUF_FMT != 0, false, true>(out, kt, false);
                }
                _llk_math_dest_srca_mm_<MATH_FIDELITY, GGUF_IN0_ROWS>(GGUF_FMT != 0 ? out : out + 1, 0);
                set ^= 1;
            }
            _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
        }
        else
        {
            // One decoded Float32 [K][N] tile out per (nt, kt).
            for (std::uint32_t kt = 0; kt < GGUF_KT; kt++)
            {
                _llk_math_wait_for_dest_available_<dest_sync>();
                page_post(OUT + 1, OUT);
                _llk_math_dest_srca_mm_unpack_wait_(OUT + 1, PAGE_FACES);
                decode_tile<GGUF_FMT != 0>(OUT, kt, true); // each tile is a DEST section of its own
                _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
            }
        }
    }
    _llk_math_dest_srca_mm_uninit_();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack.h"
#include "llk_pack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, ckernel::PackMode::Default>(OUT_FORMAT, OUT_FORMAT, FACE_R_DIM * FACE_C_DIM * 4);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(OUT_FORMAT, FACE_R_DIM, TILE_C_DIM, 4);
    _llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en>();
    for (std::uint32_t tile = 0; tile < OUTPUT_TILES; tile++)
    {
        _llk_packer_wait_for_math_done_();
        _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(GGUF_MM ? 0 : OUT, L1_ADDRESS(params.buffer_Res[tile]));
        _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
    }
}

#endif
