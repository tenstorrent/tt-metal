// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Sort-network library SIM probe (lane FG, X5 acceptance).
//
// Structure mirrors sfpu_crosslane_probe.cpp (lane FB's tracer, itself the
// lane-DS ladder-probe skeleton): UInt32 unpack-to-dest input, per-stage
// truncated runs of the GENERATED sfpi_sortnet.h networks, results stored
// to output tile 3.  The python side (test_crosslane_sortnet.py) compares
// every machine per stage against lane FB's oracle traces
// (crosslane_fixtures/bitonic_stages.json + bitonic_sort_kv_trace).
//
// Template parameters:
//   SORT_NET    0 bitonic_sort8   (element = register, machine = lane)
//               1 bitonic_sort32  (element = (row, register), machine =
//                                  column)
//               2 bitonic_sort16_kv (keys L0..L3, companions L4..L7,
//                                  machine = column; ENABLE_DEST_INDEX
//                                  window owned here around the call)
//             100/101/102 calibration (identity / rowtag / lanetag)
//   SORT_ORDER  0 ascending, 1 descending
//   SORT_STAGES stages to run (per-stage trace truncation)

#include <array>
#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static constexpr ckernel::DstSync DST_SYNC = ckernel::DstSync::SyncHalf;

#ifdef LLK_TRISC_UNPACK
#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, TILE_NUM_FACES, TILE_NUM_FACES);
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0, 0, ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, TILE_NUM_FACES), formats.unpack_A_src, formats.unpack_A_dst);
    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        L1_ADDRESS(params.buffer_A[0]), formats.unpack_A_src, formats.unpack_A_dst);
}
#endif

#ifdef LLK_TRISC_MATH
#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_math_welfords_sfpu_params.h"
using namespace ckernel;

// Raw-builtin arity (nullptr iptr), lane-DS pattern: drop the sfpi macros.
#ifdef __builtin_rvtt_sfpload
#undef __builtin_rvtt_sfpload
#endif
#ifdef __builtin_rvtt_sfpstore
#undef __builtin_rvtt_sfpstore
#endif
#ifdef __builtin_rvtt_sfpxloadi
#undef __builtin_rvtt_sfpxloadi
#endif

namespace
{

constexpr unsigned FMT_I32 = 4; // MOD0_FMT_INT32
constexpr unsigned NOINC   = 7;

#define LDROW(i)    __builtin_rvtt_sfpload(nullptr, 2 * (i), 0, 0, FMT_I32, NOINC)
#define STROW(v, i) __builtin_rvtt_sfpstore(nullptr, (v), 192 + 2 * (i), 0, 0, FMT_I32, NOINC)

constexpr sfpi::SortOrder ORDER
    = SORT_ORDER == 0 ? sfpi::SortOrder::Ascending : sfpi::SortOrder::Descending;

inline void body_identity()
{
#define IDENT(i) STROW(LDROW(i), i)
    IDENT(0); IDENT(1); IDENT(2); IDENT(3);
    IDENT(4); IDENT(5); IDENT(6); IDENT(7);
    IDENT(8); IDENT(9); IDENT(10); IDENT(11);
    IDENT(12); IDENT(13); IDENT(14); IDENT(15);
#undef IDENT
}

inline void body_rowtag()
{
#define ROWTAG(i) STROW(__builtin_rvtt_sfpxloadi(nullptr, 0x00A00000 + (i), 0, 0, 31), i)
    ROWTAG(0); ROWTAG(1); ROWTAG(2); ROWTAG(3);
    ROWTAG(4); ROWTAG(5); ROWTAG(6); ROWTAG(7);
    ROWTAG(8); ROWTAG(9); ROWTAG(10); ROWTAG(11);
    ROWTAG(12); ROWTAG(13); ROWTAG(14); ROWTAG(15);
#undef ROWTAG
}

inline void body_lanetag()
{
    auto v = __builtin_rvtt_sfpreadlreg(15); /* vConstTileId */
#define LANETAG(i) STROW(v, i)
    LANETAG(0); LANETAG(1); LANETAG(2); LANETAG(3);
    LANETAG(4); LANETAG(5); LANETAG(6); LANETAG(7);
    LANETAG(8); LANETAG(9); LANETAG(10); LANETAG(11);
    LANETAG(12); LANETAG(13); LANETAG(14); LANETAG(15);
#undef LANETAG
}

// --- the generated networks, truncated to SORT_STAGES ----------------------

inline void body_sort8()
{
    sfpi::vFloat v[8] = {
        sfpi::vFloat(LDROW(0)), sfpi::vFloat(LDROW(1)),
        sfpi::vFloat(LDROW(2)), sfpi::vFloat(LDROW(3)),
        sfpi::vFloat(LDROW(4)), sfpi::vFloat(LDROW(5)),
        sfpi::vFloat(LDROW(6)), sfpi::vFloat(LDROW(7)),
    };
    sfpi::bitonic_sort8<ORDER, (SORT_STAGES > 6 ? 6 : SORT_STAGES)>(v);
    STROW(v[0].get(), 0); STROW(v[1].get(), 1);
    STROW(v[2].get(), 2); STROW(v[3].get(), 3);
    STROW(v[4].get(), 4); STROW(v[5].get(), 5);
    STROW(v[6].get(), 6); STROW(v[7].get(), 7);
}

inline void body_sort32()
{
    sfpi::vFloat v[8] = {
        sfpi::vFloat(LDROW(0)), sfpi::vFloat(LDROW(1)),
        sfpi::vFloat(LDROW(2)), sfpi::vFloat(LDROW(3)),
        sfpi::vFloat(LDROW(4)), sfpi::vFloat(LDROW(5)),
        sfpi::vFloat(LDROW(6)), sfpi::vFloat(LDROW(7)),
    };
    sfpi::bitonic_sort32<ORDER, (SORT_STAGES > 15 ? 15 : SORT_STAGES)>(v);
    STROW(v[0].get(), 0); STROW(v[1].get(), 1);
    STROW(v[2].get(), 2); STROW(v[3].get(), 3);
    STROW(v[4].get(), 4); STROW(v[5].get(), 5);
    STROW(v[6].get(), 6); STROW(v[7].get(), 7);
}

inline void body_sort16_kv()
{
    sfpi::vFloat k[4] = {
        sfpi::vFloat(LDROW(0)), sfpi::vFloat(LDROW(1)),
        sfpi::vFloat(LDROW(2)), sfpi::vFloat(LDROW(3)),
    };
    sfpi::vUInt p[4] = {
        sfpi::vUInt(LDROW(4)), sfpi::vUInt(LDROW(5)),
        sfpi::vUInt(LDROW(6)), sfpi::vUInt(LDROW(7)),
    };
    sfpi::set_dest_index_window<true>();
    sfpi::bitonic_sort16_kv<ORDER, (SORT_STAGES > 10 ? 10 : SORT_STAGES)>(k, p);
    if constexpr (SORT_STAGES < 4)
    {
        // Data-identity SFPTRANSP involution pair: a truncation short
        // enough to contain no transp8 leaves an 8-live indexed-swap
        // graph with no exact-register anchor, which trips the KNOWN
        // IRA dual-bank coloring gap (lane EX repro-top16-ira; lreg
        // allocator territory) -- the pair anchors the eight values
        // without changing them (all-lanes context; TEN-2932-exempt).
        // Under -mtt-tensix-optimize-crosslane the involution pass
        // would delete this pair again; the probe compiles flag-off.
        sfpi::transp8(k[0], k[1], k[2], k[3], p[0], p[1], p[2], p[3]);
        sfpi::transp8(k[0], k[1], k[2], k[3], p[0], p[1], p[2], p[3]);
    }
    sfpi::set_dest_index_window<false>();
    STROW(k[0].get(), 0); STROW(k[1].get(), 1);
    STROW(k[2].get(), 2); STROW(k[3].get(), 3);
    STROW(p[0].get(), 4); STROW(p[1].get(), 5);
    STROW(p[2].get(), 6); STROW(p[3].get(), 7);
}

inline void probe_body()
{
    if constexpr (SORT_NET == 0)
        body_sort8();
    else if constexpr (SORT_NET == 1)
        body_sort32();
    else if constexpr (SORT_NET == 2)
        body_sort16_kv();
    else if constexpr (SORT_NET == 100)
        body_identity();
    else if constexpr (SORT_NET == 101)
        body_rowtag();
    else if constexpr (SORT_NET == 102)
        body_lanetag();
}

} // namespace

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, PackMode::Default>(
        TILE_NUM_FACES, formats.math);
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();
    _llk_math_wait_for_dest_available_<DST_SYNC>();
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
        0, formats.math, formats.math);
    _llk_math_eltwise_unary_sfpu_init_once_();
    math::reset_counters(p_setrwc::SET_ABD_F);
    _llk_math_welfords_sfpu_params_(+[]()
    {
        /* Raw loads/stores carry no predication; the network's lane-state
           contract is the all-lanes state.  */
        __builtin_rvtt_sfpencc_all_lanes();
        probe_body();
    }, 0);
    _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
}
#endif

#ifdef LLK_TRISC_PACK
#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"
void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
        formats.pack_src, formats.pack_dst, FACE_R_DIM * FACE_C_DIM * TILE_NUM_FACES);
    _llk_pack_init_wrapper_<PackMode::Default, false>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, TILE_NUM_FACES);
    _llk_pack_dest_init_<DST_SYNC, is_fp32_dest_acc_en>();
    _llk_packer_wait_for_math_done_();
    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(0, L1_ADDRESS(params.buffer_Res[0]));
    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(1, L1_ADDRESS(params.buffer_Res[1]));
    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(2, L1_ADDRESS(params.buffer_Res[2]));
    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(3, L1_ADDRESS(params.buffer_Res[3]));
    _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
}
#endif
