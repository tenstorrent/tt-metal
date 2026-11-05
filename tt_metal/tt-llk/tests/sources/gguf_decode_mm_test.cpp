// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// GGUF weight decode on the SFPU (sfpu/experimental/ckernel_sfpu_gguf_decode.h) and the DEST-fed matmul
// (experimental/llk_math_dest_srca_mm.h), Blackhole, 32-bit DEST, full sync.
//
// buffer_A: the weight pages, page p of N tile nt at p + nt * (GGUF_KT / page_tiles), each page_bytes long; with
//           GGUF_FMT == 0 each page is a Float32 [K][N] weight tile.
// buffer_B: the in0 tiles (bf16, GGUF_IN0_FACES faces), K tile kt at kt (GGUF_MM).
// buffer_C: the level table of a codebook format.
// Pack thread first: the RISC-side steps of every page (fp32 scale copies, grid lookups, experimental/
// ckernel_gguf_page.h), then it releases the unpack thread through its mailbox.
// GGUF_MM = false: one decoded Float32 [K][N] tile out per (nt, kt).
// GGUF_MM = true: one Float32 output tile per N tile, D += in0[kt] * W[kt][nt] over the K tiles. DEST tile 0 is the
// accumulator; the K tiles alternate between two slot sets (decoded tile, page, decoder scratch) 1..3 and 4..6, so the
// next page is unpacked while this one decodes, as the matmul kernel these LLKs come from does.

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
constexpr std::uint32_t PAGE_BYTES         = LAYOUT.rows * 64;
constexpr std::uint32_t PAGE_FACES         = LAYOUT.rows / 16;
constexpr std::uint32_t PAGES_PER_N_TILE   = GGUF_KT / LAYOUT.page_tiles;
constexpr bool PREPARE_PAGES               = GGUF_FMT != 0 && (LAYOUT.fp16_rows > 0 || LAYOUT.table_entries > 0);
constexpr std::uint32_t OUTPUT_TILES       = GGUF_MM ? GGUF_NT : GGUF_NT * GGUF_KT;

// The formats are the LLKs': pages unpacked to DEST as 32-bit words, in0 bf16, Float32 out.
constexpr std::uint32_t PAGE_FORMAT = ckernel::to_underlying(GGUF_FMT == 0 ? DataFormat::Float32 : DataFormat::UInt32);
constexpr std::uint32_t IN0_FORMAT  = ckernel::to_underlying(DataFormat::Float16_b);
constexpr std::uint32_t OUT_FORMAT  = ckernel::to_underlying(DataFormat::Float32);

static_assert(dest_sync == ckernel::DstSync::SyncFull && is_fp32_dest_acc_en, "gguf_decode_mm_test: full sync, 32-bit DEST");

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
    if constexpr (PREPARE_PAGES)
    {
        (void)mailbox_read(ThreadId::PackThreadId);
    }

    const std::uint32_t pages_base = params.buffer_A[0];
    const auto unpack_page         = [&](const std::uint32_t nt, const std::uint32_t kt)
    {
        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, true>(
            L1_ADDRESS(page_address(pages_base, nt, kt)), PAGE_FORMAT, PAGE_FORMAT);
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

// The accumulator: a Float32 DEST tile of zeros, written by the SFPU.
inline void zero_tile(const std::uint32_t dst_index)
{
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, (dst_index << 6) + get_dest_buffer_base());
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
#pragma GCC unroll 8
    for (int i = 0; i < 32; i++)
    {
        sfpi::dst_reg[i] = 0.0f;
    }
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
}

// group: the first tile of its group decoded at dst_index (sfpu_gguf_decode_group).
template <bool decode>
inline void decode_tile(const std::uint32_t dst_index, const std::uint32_t kt, const bool group)
{
    if constexpr (decode)
    {
        if (group)
        {
            sfpu::sfpu_gguf_decode_group<GGUF_FMT>(dst_index);
        }
        sfpu::sfpu_gguf_decode_tile<GGUF_FMT>(dst_index, kt % LAYOUT.group_tiles);
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
    _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(OUT_FORMAT, IN0_FORMAT);
    _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>();
    sfpu::sfpu_gguf_decode_init<GGUF_FMT>();
    _llk_math_dest_srca_mm_init_<MATH_FIDELITY, GGUF_IN0_ROWS>();

    for (std::uint32_t nt = 0; nt < GGUF_NT; nt++)
    {
        if constexpr (GGUF_MM)
        {
            _llk_math_wait_for_dest_available_<dest_sync>();
            zero_tile(0);
            _llk_math_dest_srca_mm_unpack_post_(2);
            _llk_math_dest_srca_mm_unpack_wait_(2, PAGE_FACES);
            std::uint32_t set = 0;
            for (std::uint32_t kt = 0; kt < GGUF_KT; kt++)
            {
                const std::uint32_t out       = set ? 4 : 1; // the decoded tile; its page is out + 1, scratch out + 2
                const std::uint32_t next_page = set ? 2 : 5;
                const bool next               = kt + 1 < GGUF_KT;
                if (next)
                {
                    _llk_math_dest_srca_mm_unpack_post_(next_page);
                }
                decode_tile<GGUF_FMT != 0>(out, kt, kt % LAYOUT.group_tiles < 2); // a group starts in both sets
                if (next)
                {
                    _llk_math_dest_srca_mm_unpack_wait_(next_page, PAGE_FACES);
                }
                _llk_math_dest_srca_mm_<MATH_FIDELITY, GGUF_IN0_ROWS>(GGUF_FMT != 0 ? out : out + 1, 0);
                set ^= 1;
            }
            _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
        }
        else
        {
            for (std::uint32_t kt = 0; kt < GGUF_KT; kt++)
            {
                _llk_math_wait_for_dest_available_<dest_sync>();
                _llk_math_dest_srca_mm_unpack_post_(1);
                _llk_math_dest_srca_mm_unpack_wait_(1, PAGE_FACES);
                decode_tile<GGUF_FMT != 0>(0, kt, true); // each tile is a DEST section of its own
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
    if constexpr (PREPARE_PAGES)
    {
        const auto* table = reinterpret_cast<const std::uint16_t*>(params.buffer_C[0]);
        for (std::uint32_t page = 0; page < GGUF_NT * PAGES_PER_N_TILE; page++)
        {
            ckernel::gguf::prepare_page<static_cast<ckernel::gguf::Format>(GGUF_FMT)>(params.buffer_A[0] + page * PAGE_BYTES, table);
        }
        asm volatile("fence" ::: "memory"); // the pages are in L1 before the unpacker reads them
        mailbox_write(ThreadId::UnpackThreadId, 1);
    }

    _llk_pack_hw_configure_<is_fp32_dest_acc_en, ckernel::PackMode::Default>(OUT_FORMAT, OUT_FORMAT, FACE_R_DIM * FACE_C_DIM * 4);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(OUT_FORMAT, FACE_R_DIM, TILE_C_DIM, 4);
    _llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en>();
    for (std::uint32_t tile = 0; tile < OUTPUT_TILES; tile++)
    {
        _llk_packer_wait_for_math_done_();
        _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(0, L1_ADDRESS(params.buffer_Res[tile]));
        _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
    }
}

#endif
