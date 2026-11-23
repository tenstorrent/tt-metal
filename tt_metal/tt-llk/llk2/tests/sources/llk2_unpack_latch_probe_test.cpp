// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Hardware probe, not an API test: when does an unpacker read its configuration? It measures what llk_lib/unpack.h's
// configure relies on (it writes the unpackers' words behind the previous op's UNPACRs, with a wait only in front
// of the words the hardware reads while data moves).
//
// Every round is one add of a and b (TILES tiles each, both unpackers streaming, the UNPACRs issued back to back).
// Behind the add's last UNPACR the unpack thread
//   1. issues round * NOP_STEP NOPs, so the write below passes the Wait Gate a different number of cycles behind the
//      last UNPACR in every round: right behind it in round 0, behind the end of all unpacks in the last rounds;
//   2. issues ONE write (POISON) that changes one field the unpackers use to a value that would give other data;
//   3. waits for both unpackers (a STALLWAIT that blocks everything), so the next round's configure, which writes
//      every word again, runs with idle unpackers and the write of step 2 is the only one that can land on an
//      unpack in flight.
// Each round packs to its own output tiles. A field the unpacker has taken before the write lands leaves every
// round exact; a field it reads while data moves corrupts the early rounds and leaves the late ones (the control:
// their write lands on idle unpackers) exact.
//
// MATH_SPIN > 0 is the widest window: the math thread spins on the RISC-V before every add, so its Src banks stay
// full, and the unpack of the third tile (TILES is 3 then) starts and then stands still in the unpacker, waiting
// for a bank, while the write lands. Every round's write then lands on an unpack in flight.
//
// POISON 0 writes nothing. The others: see poison() below and test_llk2_unpack_latch_probe.py. SrcAUnsigned and
// SrcBUnsigned are not among them: they do nothing to BF16 data (test_llk2_unpack_signedness.py has them).

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_ops.h"
#include "cunpack_common.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "params.h"

// ckernel_globals.h declares these extern; unp_cfg_context is the unpack context unpack.h flips.
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#define LLK2_DEST_FP32 is_fp32_dest_acc_en
#include "compute_api/dest.h"
#include "compute_api/eltwise_binary.h"
#include "compute_api/pack.h"

constexpr DataFormat in_format  = static_cast<DataFormat>(UNPACK_A_IN);
constexpr DataFormat out_format = static_cast<DataFormat>(PACK_OUT);
static_assert(in_format == DataFormat::Float16_b && static_cast<DataFormat>(UNPACK_B_IN) == DataFormat::Float16_b, "the poison values assume BF16 inputs");

inline constexpr auto& in_spec  = llk2::l1_spec<llk2::default_tile_dim, TILES * llk2::default_tile_dim, in_format>;
inline constexpr auto& out_spec = llk2::l1_spec<llk2::default_tile_dim, TILES * llk2::default_tile_dim, out_format>;
inline constexpr auto& acc_spec = llk2::dst_spec<llk2::default_tile_dim, TILES * llk2::default_tile_dim, llk2::kernel_dest_format>;

#ifdef LLK_TRISC_UNPACK
/// One write that would change what the unpackers deliver, were it read after it lands. BF16 in L1 and in Src, 32x32
/// tiles, x4 throttle, no transpose is what the add configured.
inline __attribute__((always_inline)) void poison()
{
    using namespace ckernel;
    constexpr std::uint32_t fp16         = static_cast<std::uint32_t>(DataFormat::Float16);
    constexpr std::uint32_t fp32         = static_cast<std::uint32_t>(DataFormat::Float32);
    constexpr std::uint32_t one_face_end = 16 * 16 - 1;
    switch (POISON)
    {
        case 0:
            break;
        // ---- unpacker 0 (SrcA) ----
        case 1: // register format
            cfg_reg_rmw_tensix<THCON_SEC0_REG2_Out_data_format_RMW>(fp16);
            break;
        case 2: // L1 format (tile descriptor, InDataFormat)
            cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32, 0, 0xf>(fp32);
            break;
        case 3: // transpose inside the face
            cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(1);
            break;
        case 4: // face size per context: 16 datums instead of 256
            cfg_reg_rmw_tensix<THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32, 0, 0xffffffff>(16 | (16 << 16));
            break;
        case 5: // faces per tile (tile descriptor, ZDim): 1 instead of 4
            cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1, 0, 0xffff0000>(1 << 16);
            break;
        case 6: // X end: one face instead of the tile
            TTI_SETADCXX(p_setadc::UNP_A, one_face_end, 0x0);
            break;
        case 7: // input Z counter: two faces on
            TTI_SETADCZW(p_setadc::UNP_A, 0, 0, 0, 2, 0b0001);
            break;
        case 8: // Src write base, both contexts: 16 rows further
            cfg_reg_rmw_tensix<THCON_SEC0_REG5_Dest_cntx0_address_ADDR32, 0, 0xffffffff>((4 + 16) * 16 | (((4 + 16) * 16) << 16));
            break;
        case 9: // throttle: x1
            cfg_reg_rmw_tensix<THCON_SEC0_REG2_Throttle_mode_RMW>(0);
            break;
        case 10: // write Dest instead of SrcA, both contexts
            cfg_reg_rmw_tensix<THCON_SEC0_REG2_Unpack_if_sel_cntx0_ADDR32, THCON_SEC0_REG2_Unpack_if_sel_cntx0_SHAMT, 0x30>(3);
            break;
        case 12: // SrcA bank select by address off: the word unpack.h does wait for
            TTI_SETC16(SRCA_SET_SetOvrdWithAddr_ADDR32, 0);
            break;
        case 13: // output Y, Z counters (the Src side): one step on
            TTI_SETADCXY(p_setadc::UNP_A, 1, 0, 0, 0, 0b1000);
            TTI_SETADCZW(p_setadc::UNP_A, 0, 1, 0, 0, 0b0100);
            break;
        // ---- unpacker 1 (SrcB) ----
        case 21:
            cfg_reg_rmw_tensix<THCON_SEC1_REG2_Out_data_format_RMW>(fp16);
            break;
        case 22:
            cfg_reg_rmw_tensix<THCON_SEC1_REG0_TileDescriptor_ADDR32, 0, 0xf>(fp32);
            break;
        case 23: // face size (tile descriptor, XDim): 16 datums instead of 256
            cfg_reg_rmw_tensix<THCON_SEC1_REG0_TileDescriptor_ADDR32, 0, 0xffff0000>(16 << 16);
            break;
        case 24:
            cfg_reg_rmw_tensix<THCON_SEC1_REG0_TileDescriptor_ADDR32 + 1, 0, 0xffff0000>(1 << 16);
            break;
        case 25:
            TTI_SETADCXX(p_setadc::UNP_B, one_face_end, 0x0);
            break;
        case 26:
            TTI_SETADCZW(p_setadc::UNP_B, 0, 0, 0, 2, 0b0001);
            break;
        case 27:
            cfg_reg_rmw_tensix<THCON_SEC1_REG2_Throttle_mode_RMW>(0);
            break;
        default:
            break;
    }
}
#endif

void run_kernel(RUNTIME_PARAMETERS params)
{
    using AccRing = llk2::DestRing<acc_spec>;
    const auto a  = llk2::Tensor<in_spec>(params.buffer_A[0]);
    const auto b  = llk2::Tensor<in_spec>(params.buffer_B[0]);
    for (std::uint32_t round = 0; round < ROUNDS; round++)
    {
#ifdef LLK_TRISC_MATH
        for (volatile std::uint32_t spin = 0; spin < MATH_SPIN; spin = spin + 1)
        {
        }
#endif
        const auto y = llk2::add(a, b, llk2::reserve_back<AccRing>());
#ifdef LLK_TRISC_UNPACK
        for (std::uint32_t nop = 0; nop < round * NOP_STEP; nop++)
        {
            TTI_NOP;
        }
        poison();
        TTI_STALLWAIT(ckernel::p_stall::STALL_THREAD, ckernel::p_stall::UNPACK0 | ckernel::p_stall::UNPACK1);
#endif
        llk2::pack(y, llk2::Tensor<out_spec>(params.buffer_Res[round * TILES]));
        llk2::release(y);
    }
}
