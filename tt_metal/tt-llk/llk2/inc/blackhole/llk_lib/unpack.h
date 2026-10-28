// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// llk-lib, unpack thread, Blackhole. Written only in HAL calls (tt_llk_blackhole/llk_lib/hal/).
//
// llk-lib is the layer under the Compute API: it splits a Compute API call into per-thread work and, on each
// thread, programs the hardware for it. This file is everything the unpack thread does, for every op:
//
//   unpack_params                 one struct for every unpack operation; configure and execute take the same one
//   unpack_params_from(Plan)      fills it from the Resolver's Plan, for any op
//   unpack_configure<P>()         programs both unpackers completely: formats, strides, descriptors, then the
//                                 address walk of P.op (for matmul: a replay + MOP that streams one operand)
//   unpack_execute<P>(a, b)       one unpack step of P.op from the given SrcA / SrcB L1 addresses
//   unpack<On, Plan>(in0, in1)    the per-thread call the Compute API makes: plans on every thread, configures and
//                                 executes only when this build is for thread On (the unpack thread)
//
// configure assumes nothing about what ran before; removing writes an earlier op already made is left to the
// compiler (config folding). There is no init, uninit or reconfig.
//
// The operands: unpacker 0 writes SrcA and is configured through THCON SEC0; unpacker 1 writes SrcB through SEC1.
// The L1 base address of each sits in a config register per context; the unpack thread alternates between two
// config contexts so that it can set up the next step's addresses while the unpackers still run the current one.
// UNPACK_SYNC (semaphore 5) counts the contexts in flight.
//
// Not HAL yet: the descriptor format helpers (masked_data_format, datum_size_in_bytes, canonical_unpA_y_stride
// from cunpack_common.h) and the software state ckernel keeps (unp_cfg_context).

#pragma once

#include <array>
#include <cstdint>

#include "ckernel.h"
#include "cunpack_common.h"
#include "hal/cfg.h"
#include "hal/mop.h"
#include "hal/replay.h"
#include "hal/sync.h"
#include "hal/tensix_counters.h"
#include "hal/unpack.h"
#include "llk_lib/thread.h"
#include "llk_memory_checks.h"
#include "resolver/plan.h"

// hal/scalar.h has an LLK_ASSERT inside a constexpr function, which C++17 flags under ENABLE_LLK_ASSERT.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wc++20-extensions"
#include "hal/scalar.h"
#pragma GCC diagnostic pop

namespace llk2::llk_lib
{

namespace cfg = hal::cfg;
using hal::sync::Semaphore;
using hal::sync::StallCondition;
using hal::sync::StallTarget;

/// What one unpacker reads from L1 and writes to its Src register.
struct unpack_operand
{
    DataFormat l1_format;     ///< format in L1 (the unpacker's InDataFormat)
    DataFormat reg_format;    ///< format written to Src (OutDataFormat)
    std::uint32_t face_r_dim; ///< rows per face
    std::uint32_t num_faces;  ///< faces per tile
    std::uint32_t tile_size;  ///< one tile in L1, in 16-byte address units
    bool transpose;           ///< transpose the faces on the way in (unpacker 0 only)
};

/// Parameters of one unpack operation, for any op. configure and execute take the same struct.
struct unpack_params
{
    MathOp op;            ///< the math op this unpack feeds: selects the address walk
    unpack_operand src_a; ///< unpacker 0 -> SrcA
    unpack_operand src_b; ///< unpacker 1 -> SrcB
    std::uint32_t rt_dim; ///< output tiles along y
    std::uint32_t ct_dim; ///< output tiles along x
    std::uint32_t kt_dim; ///< K tiles (matmul)
};

constexpr unpack_operand operand_from(const UnpackPlan& u)
{
    return {u.reads, u.writes, u.face_r_dim, u.num_faces, u.tile_bytes / 16, u.transpose_faces};
}

/// The Plan of any op as unpack parameters: each operand goes to the unpacker its Plan names (Where::SrcA / SrcB).
constexpr unpack_params unpack_params_from(const Plan& p)
{
    const UnpackPlan& a = p.in0.to == Where::SrcA ? p.in0 : p.in1;
    const UnpackPlan& b = p.in0.to == Where::SrcB ? p.in0 : p.in1;
    return {p.op, operand_from(a), operand_from(b), p.work.rt_dim, p.work.ct_dim, p.work.kt_dim};
}

inline constexpr Semaphore unpack_sync = Semaphore::S5; // ckernel::semaphore::UNPACK_SYNC: contexts in flight

/// Where the unpackers start writing in Src, for both contexts. The same value configure_unpack_AB uses.
inline constexpr std::uint32_t src_write_base = 4 * 16;

constexpr std::uint32_t fmt(DataFormat f)
{
    return static_cast<std::uint32_t>(f);
}

/// The last datum index of a tile along X, for the unpacker's X counter: every face, every row, 16 datums per row.
constexpr std::uint32_t x_end(const unpack_operand& o)
{
    return o.num_faces * o.face_r_dim * ckernel::FACE_C_DIM - 1;
}

/// Bytes one face takes in Src: how far the unpacker's Src (ch1) address moves per Z step.
constexpr std::uint32_t src_face_stride(const unpack_operand& o)
{
    return ckernel::FACE_C_DIM * ckernel::FACE_R_DIM * ckernel::datum_size_in_bytes(ckernel::masked_data_format(fmt(o.reg_format)));
}

/// Selects config context 0 or 1 for both unpackers.
inline __attribute__((always_inline)) void set_unpack_context(std::uint32_t context)
{
    if (context == 0)
    {
        cfg::write<cfg::Access::TensixCfgUnit>(
            cfg::set<cfg::UnpackMiscCfg::CfgContextOffset_0, cfg::Sec::S0, 0>(), cfg::set<cfg::UnpackMiscCfg::CfgContextOffset_1, cfg::Sec::S0, 0>());
    }
    else
    {
        cfg::write<cfg::Access::TensixCfgUnit>(
            cfg::set<cfg::UnpackMiscCfg::CfgContextOffset_0, cfg::Sec::S0, 1>(), cfg::set<cfg::UnpackMiscCfg::CfgContextOffset_1, cfg::Sec::S0, 1>());
    }
}

/// One unpacker's tile descriptor (input format and tile geometry) and config (output format, transpose), each
/// in one write. S is the unpacker's THCON section: S0 for unpacker 0, S1 for unpacker 1.
template <cfg::Sec S, unpack_operand O, std::uint32_t XDim>
inline __attribute__((always_inline)) void configure_unpacker()
{
    using ckernel::masked_data_format;
    constexpr auto& td = cfg::Thcon[cfg::Reg0].TileDescriptor;
    cfg::write<cfg::Access::MMIO>(
        cfg::set<td.InDataFormat, S, masked_data_format(fmt(O.l1_format))>(),
        cfg::set<td.Uncompressed, S, 1>(),
        cfg::set<td.BlobsPerXyPlane, S, 0>(),
        cfg::set<td.XDim, S, XDim>(),
        cfg::set<td.YDim, S, 1>(),
        cfg::set<td.ZDim, S, O.num_faces>());

    using r2 = cfg::ThconReg2Fields;
    cfg::write<cfg::Access::MMIO>(
        cfg::set<r2::Out_data_format, S, masked_data_format(fmt(O.reg_format))>(),
        cfg::set<r2::Throttle_mode, S, 2>(),
        cfg::set<r2::Context_count, S, 0>(),
        cfg::set<r2::Haloize_mode, S, O.transpose ? 1u : 0u>(),
        cfg::set<r2::Tileize_mode, S, 0>(),
        cfg::set<r2::Unpack_Src_Reg_Set_Upd, S, 0>(),
        cfg::set<r2::Unpack_If_Sel, S, 0>(),
        cfg::set<r2::Upsample_rate, S, 0>(),
        cfg::set<r2::Ovrd_data_format, S, 0>(),
        cfg::set<r2::Upsample_and_interleave, S, 0>(),
        cfg::set<r2::Shift_amount_cntx0, S, 0>(),
        cfg::set<r2::Shift_amount_cntx1, S, 0>(),
        cfg::set<r2::Shift_amount_cntx2, S, 0>(),
        cfg::set<r2::Shift_amount_cntx3, S, 0>(),
        cfg::set<r2::Disable_zero_compress_cntx0, S, 1>(),
        cfg::set<r2::Disable_zero_compress_cntx1, S, 1>(),
        cfg::set<r2::Disable_zero_compress_cntx2, S, 1>(),
        cfg::set<r2::Disable_zero_compress_cntx3, S, 1>(),
        cfg::set<r2::Disable_zero_compress_cntx4, S, 1>(),
        cfg::set<r2::Disable_zero_compress_cntx5, S, 1>(),
        cfg::set<r2::Disable_zero_compress_cntx6, S, 1>(),
        cfg::set<r2::Disable_zero_compress_cntx7, S, 1>(),
        cfg::set<r2::Force_shared_exp, S, 0>());
}

// ---- matmul's address walk ---------------------------------------------------------------------------------------
//
// A matmul K step multiplies in0's tile column k (rt_dim tiles, SrcB) by in1's tile row k (ct_dim tiles, SrcA).
// The larger output dimension is the one to reuse along: when ct_dim >= rt_dim, each in0 tile stays in SrcB while
// ct_dim in1 tiles stream through SrcA; otherwise each in1 tile stays in SrcA while rt_dim in0 tiles stream through
// SrcB. One "outer" step unpacks the reused tile once, then runs the MOP, which unpacks the streamed operand once
// per output tile and moves its L1 base address on by one tile (in1: the next column) or one tile row (in0: K tiles).

namespace matmul_walk
{

/// True when in0 (SrcB) is the operand that stays: the reuse_a case of tt-llk.
constexpr bool reuse_src_b(const unpack_params& p)
{
    return p.ct_dim >= p.rt_dim;
}

/// Records the replay the MOP runs per streamed tile: unpack, then base address += step, in both config contexts.
/// Engine and Sec are the streamed unpacker; StepGpr holds the step in 16-byte units.
template <hal::unpack::Engine Engine, cfg::Sec Sec, std::uint32_t StepGpr>
inline __attribute__((always_inline)) void record_streamed_unpack()
{
    using ckernel::p_gpr_unpack;
    constexpr hal::unpack::DataTransfer unpack_tile {.engine = Engine, .handoff = hal::unpack::SourceHandoff::FlipAndSetDataValid};
    constexpr auto& base_cntx0 = cfg::Thcon[cfg::Reg3].Base_address;
    constexpr auto& base_cntx1 = cfg::Thcon[cfg::Reg3].Base_cntx1_address;
    hal::replay::record<hal::replay::BufferRange {0, 12}>(
        []
        {
            // context 0: replay slots 0..5
            hal::unpack::run<unpack_tile>();
            cfg::read<cfg::Access::TensixCfgUnit, base_cntx0, Sec>(hal::gpr<p_gpr_unpack::TMP0>());
            hal::scalar::add(hal::gpr<p_gpr_unpack::TMP0>(), hal::gpr<StepGpr>(), hal::gpr<p_gpr_unpack::TMP0>());
            hal::sync::wait::stall<StallTarget::Config, StallCondition::ScalarIdle>();
            cfg::write<cfg::Access::TensixCfgUnit, base_cntx0, Sec, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Wait>(hal::gpr<p_gpr_unpack::TMP0>());
            // context 1: replay slots 6..11
            hal::unpack::run<unpack_tile>();
            cfg::read<cfg::Access::TensixCfgUnit, base_cntx1, Sec>(hal::gpr<p_gpr_unpack::TMP0>());
            hal::scalar::add(hal::gpr<p_gpr_unpack::TMP0>(), hal::gpr<StepGpr>(), hal::gpr<p_gpr_unpack::TMP0>());
            hal::sync::wait::stall<StallTarget::Config, StallCondition::ScalarIdle>();
            cfg::write<cfg::Access::TensixCfgUnit, base_cntx1, Sec, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Wait>(hal::gpr<p_gpr_unpack::TMP0>());
        });
}

/// matmul's part of configure: X ends, GPRs for the address steps, the replay and the MOP.
template <unpack_params P>
inline __attribute__((always_inline)) void configure()
{
    using ckernel::p_gpr_unpack;
    using namespace hal::counters::adc;
    hal::counters::set<AddrCounter {.channel0 = {.x = 0}, .channel1 = {.x = x_end(P.src_a)}}.for_clients<Client::Unpacker0>()>();
    hal::counters::set<AddrCounter {.channel0 = {.x = 0}, .channel1 = {.x = x_end(P.src_b)}}.for_clients<Client::Unpacker1>()>();

    // GPRs: tile sizes (the per-tile step of each operand) and K (in0's step is a row of K tiles).
    hal::scalar::set_low(hal::gpr<p_gpr_unpack::TILE_SIZE_A>(), P.src_a.tile_size & 0xffff);
    hal::scalar::set_low(hal::gpr<p_gpr_unpack::TILE_SIZE_B>(), P.src_b.tile_size & 0xffff);
    hal::scalar::set_low(hal::gpr<p_gpr_unpack::KT_DIM>(), P.kt_dim & 0xffff);

    if constexpr (reuse_src_b(P))
    {
        record_streamed_unpack<hal::unpack::Engine::Unpacker0, cfg::Sec::S0, p_gpr_unpack::TILE_SIZE_A>(); // stream in1: one tile
    }
    else
    {
        record_streamed_unpack<hal::unpack::Engine::Unpacker1, cfg::Sec::S1, p_gpr_unpack::TMP_LO>(); // stream in0: K tiles
    }
    // MOP template 0: each iteration runs the replay of the current context.
    hal::mop::program<hal::mop::MopConfig<hal::mop::MopTemplate::Template0> {
        .start_op        = hal::replay::get_operation<hal::replay::BufferRange {0, 6}>(),
        .start_op_shadow = hal::replay::get_operation<hal::replay::BufferRange {6, 6}>()}>();
}

/// One K step from in1's tile (k, 0) at src_a_address and in0's tile (0, k) at src_b_address (byte addresses).
template <unpack_params P>
inline __attribute__((always_inline)) void execute(std::uint32_t src_a_address, std::uint32_t src_b_address)
{
    using namespace ckernel;
    constexpr bool reuse_b                   = reuse_src_b(P);
    constexpr std::uint32_t reused_tiles     = reuse_b ? P.rt_dim : P.ct_dim; // outer steps: one reused tile each
    constexpr std::uint32_t streamed_tiles   = reuse_b ? P.ct_dim : P.rt_dim; // MOP iterations per outer step
    constexpr std::uint32_t context_mask_all = 0xff;
    constexpr hal::unpack::DataTransfer unpack_reused {
        .engine = reuse_b ? hal::unpack::Engine::Unpacker1 : hal::unpack::Engine::Unpacker0, .handoff = hal::unpack::SourceHandoff::FlipAndSetDataValid};

    if constexpr (!reuse_b)
    {
        // in0 streams a row at a time: TMP_LO = its tile size * K.
        hal::scalar::multiply_low_u16(hal::gpr<p_gpr_unpack::TMP_LO>(), hal::gpr<p_gpr_unpack::KT_DIM>(), hal::gpr<p_gpr_unpack::TILE_SIZE_B>());
    }

    const std::uint32_t src_a_base = L1_ADDRESS(src_a_address);
    const std::uint32_t src_b_base = L1_ADDRESS(src_b_address);
    for (std::uint32_t reused_tile = 0; reused_tile < reused_tiles; reused_tile++)
    {
        // The reused operand moves on per outer step: in0 by a tile row (K tiles), in1 by one tile.
        const std::uint32_t src_b_tile_address = src_b_base + P.src_b.tile_size * (reuse_b ? reused_tile * P.kt_dim : 0);
        const std::uint32_t src_a_tile_address = src_a_base + P.src_a.tile_size * (reuse_b ? 0 : reused_tile);
        LLK_ASSERT(is_valid_L1_address(src_a_tile_address) && is_valid_L1_address(src_b_tile_address), "matmul operand address is outside L1");

        // Take a free config context (at most two in flight) and point both unpackers at this step's tiles.
        while (hal::sync::semaphore::read<unpack_sync>() >= 2)
        {
        }
        if (unp_cfg_context == 0)
        {
            cfg::write<cfg::Access::MMIO, cfg::Thcon[cfg::Reg3].Base_address, cfg::Sec::S0>(src_a_tile_address);
            cfg::write<cfg::Access::MMIO, cfg::Thcon[cfg::Reg3].Base_address, cfg::Sec::S1>(src_b_tile_address);
        }
        else
        {
            cfg::write<cfg::Access::MMIO, cfg::Thcon[cfg::Reg3].Base_cntx1_address, cfg::Sec::S0>(src_a_tile_address);
            cfg::write<cfg::Access::MMIO, cfg::Thcon[cfg::Reg3].Base_cntx1_address, cfg::Sec::S1>(src_b_tile_address);
        }
        hal::sync::semaphore::post<hal::sync::Access::MMIO, unpack_sync>();
        hal::sync::wait::stall<StallTarget::Unpack, StallCondition::RiscvAccessProcessed>();

        hal::unpack::run<unpack_reused>();
        hal::mop::Runner<hal::mop::MopTemplate::Template0>::run(streamed_tiles, unp_cfg_context == 0 ? 0 : context_mask_all);

        // Release the context once the unpackers are done with it, and switch to the other one.
        hal::sync::semaphore::get<unpack_sync>();
        unp_cfg_context = 1 - unp_cfg_context;
        set_unpack_context(unp_cfg_context);
    }
}

} // namespace matmul_walk

// ---- every unpack op ---------------------------------------------------------------------------------------------

/// Programs both unpackers for P: the part every op shares, then P.op's address walk.
template <unpack_params P>
inline __attribute__((always_inline)) void unpack_configure()
{
    using namespace ckernel;
    using namespace ckernel::unpacker;
    using namespace hal::counters::adc;
    static_assert(P.src_a.num_faces == 4 && P.src_b.num_faces == 4 && P.src_a.face_r_dim == FACE_R_DIM && P.src_b.face_r_dim == FACE_R_DIM,
                  "32x32 tiles only, for now: partial faces and tiny tiles are not ported");

    // Wait until math has taken every context, then reset the unpackers' address counters.
    while (hal::sync::semaphore::read<unpack_sync>() > 0)
    {
    }
    hal::counters::set<AddrCounter {.channel0 = {.x = 0, .y = 0}, .channel1 = {.y = 0}}.for_clients<Client::Unpacker0, Client::Unpacker1>()>();
    hal::counters::set<AddrCounter {.channel0 = {.z = 0, .w = 0}, .channel1 = {.z = 0, .w = 0}}.for_clients<Client::Unpacker0, Client::Unpacker1>()>();

    // Src (ch1) strides: one face per Z step; SrcA's Y stride for its register format.
    using cfg::UnpackerReg;
    cfg::write<cfg::Access::MMIO>(
        cfg::set<cfg::Unpacker[0].AddrCtrl[UnpackerReg::Reg1].Zstride, cfg::Sec::S0, src_face_stride(P.src_a)>(),
        cfg::set<cfg::Unpacker[0].AddrCtrl[UnpackerReg::Reg1].Wstride, cfg::Sec::S0, 0>());
    cfg::write<cfg::Access::MMIO>(
        cfg::set<cfg::Unpacker[1].AddrCtrl[UnpackerReg::Reg1].Zstride, cfg::Sec::S0, src_face_stride(P.src_b)>(),
        cfg::set<cfg::Unpacker[1].AddrCtrl[UnpackerReg::Reg1].Wstride, cfg::Sec::S0, 0>());
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Unpacker[0].AddrCtrl[UnpackerReg::Reg1].Ystride, cfg::Sec::S0, canonical_unpA_y_stride(masked_data_format(fmt(P.src_a.reg_format)))>();

    // ALU: unsigned Src for UINT8, no stochastic rounding; FP8 E4M3 per unpacker. Shared with other threads, so
    // under the RMW mutex.
    hal::sync::mutex::acquire<hal::sync::Mutex::M0>();
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AluRoundingMode::Fpu_srnd_en, cfg::Sec::S0, 0>(),
        cfg::set<cfg::AluRoundingMode::Gasket_srnd_en, cfg::Sec::S0, 0>(),
        cfg::set<cfg::AluRoundingMode::Packer_srnd_en, cfg::Sec::S0, 0>(),
        cfg::set<cfg::AluFormatSpecReg0::SrcAUnsigned, cfg::Sec::S0, P.src_a.l1_format == DataFormat::UInt8 ? 1u : 0u>(),
        cfg::set<cfg::AluFormatSpecReg0::SrcBUnsigned, cfg::Sec::S0, P.src_b.l1_format == DataFormat::UInt8 ? 1u : 0u>());
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg1].Unp_LF8_4b_exp, cfg::Sec::S0, P.src_a.l1_format == DataFormat::Fp8_e4m3 ? 1u : 0u>();
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg1].Unp_LF8_4b_exp, cfg::Sec::S1, P.src_b.l1_format == DataFormat::Fp8_e4m3 ? 1u : 0u>();
    hal::sync::mutex::release<hal::sync::Mutex::M0>();

    // Descriptors and configs. Unpacker 0 takes its X dim from the per-context Tile_x_dim below, so 0 here.
    constexpr std::uint32_t src_a_face_datums = P.src_a.face_r_dim * FACE_C_DIM;
    constexpr std::uint32_t src_b_face_datums = P.src_b.face_r_dim * FACE_C_DIM;
    configure_unpacker<cfg::Sec::S0, P.src_a, 0>();
    configure_unpacker<cfg::Sec::S1, P.src_b, src_b_face_datums>();

    // Src write base for both contexts, and unpacker 0's face size per context.
    cfg::write<cfg::Access::MMIO>(
        cfg::set<cfg::Thcon[cfg::Reg5].Dest_cntx0_address, cfg::Sec::S0, src_write_base>(),
        cfg::set<cfg::Thcon[cfg::Reg5].Dest_cntx1_address, cfg::Sec::S0, src_write_base>());
    cfg::write<cfg::Access::MMIO>(
        cfg::set<cfg::Thcon[cfg::Reg5].Tile_x_dim_cntx0, cfg::Sec::S0, src_a_face_datums>(),
        cfg::set<cfg::Thcon[cfg::Reg5].Tile_x_dim_cntx1, cfg::Sec::S0, src_a_face_datums>());

    // SrcA bank select by address, and unpacker 0's Src address counter added to the write base.
    cfg::write<cfg::Access::TensixCfgUnit, cfg::SrcASet::SetOvrdWithAddr, cfg::Sec::S0, 1>();
    cfg::write<cfg::Access::MMIO, cfg::Unpacker[0].AddDestAddrCntr, cfg::Sec::S0, 1>(std::array<std::uint32_t, 1> {1u << 8});

    unp_cfg_context = 0;
    set_unpack_context(0);

    if constexpr (P.op == MathOp::Matmul)
    {
        matmul_walk::configure<P>();
    }
    else
    {
        static_assert(P.op == MathOp::Matmul, "unpack: op not ported yet");
    }
}

/// One unpack step of P.op. Addresses are byte addresses in L1 of what goes to SrcA and SrcB.
template <unpack_params P>
inline __attribute__((always_inline)) void unpack_execute(std::uint32_t src_a_address, std::uint32_t src_b_address)
{
    if constexpr (P.op == MathOp::Matmul)
    {
        matmul_walk::execute<P>(src_a_address, src_b_address);
    }
}

/// The unpack thread's part of a Compute API call. in0_address and in1_address are the byte addresses of the
/// call's input tensors; Pl says which unpacker each goes to and what to do with them.
template <Thread On, Plan Pl>
inline __attribute__((always_inline)) void unpack(std::uint32_t in0_address, std::uint32_t in1_address)
{
    static_assert(On == Thread::Unpack, "the unpackers are driven from the unpack thread");
    constexpr unpack_params p = unpack_params_from(Pl);
    if constexpr (runs_on<On>)
    {
        const std::uint32_t src_a_address = Pl.in0.to == Where::SrcA ? in0_address : in1_address;
        const std::uint32_t src_b_address = Pl.in0.to == Where::SrcB ? in0_address : in1_address;
        unpack_configure<p>();
        if constexpr (p.op == MathOp::Matmul)
        {
            // K steps: in1's tile row k (SrcA) starts k * ct_dim tiles in, in0's tile column k (SrcB) k tiles in.
            for (std::uint32_t k = 0; k < p.kt_dim; k++)
            {
                unpack_execute<p>(src_a_address + k * p.ct_dim * p.src_a.tile_size * 16, src_b_address + k * p.src_b.tile_size * 16);
            }
        }
    }
}

} // namespace llk2::llk_lib
