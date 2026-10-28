// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// LLK API (merged with Lib): matmul on Blackhole, written in HAL calls.
//
//   matmul_params              the op's parameter struct, filled from the Resolver's Plan; configure and execute
//                              take the same struct
//   unpack_configure/execute   unpacker 0 (SrcA) <- in1, unpacker 1 (SrcB) <- in0, as tt-llk does
//   math_configure/execute     the FPU: rt_dim x ct_dim output tiles over kt_dim K steps
//   unpack_matmul, math_matmul the per-thread calls: plan on every thread, configure + execute on their own
//
// configure programs everything its unit needs and assumes nothing about what ran before. Removing the writes an
// earlier op already made is the compiler's job (config folding).
//
// Reproduces _llk_unpack_hw_configure_ + _llk_unpack_AB_matmul_init_ / _llk_unpack_AB_matmul_ and
// _llk_math_hw_configure_ + _llk_math_matmul_init_ / _llk_math_matmul_ for 32x32 tiles, no partial faces,
// THROTTLE_LEVEL 0, no kernel broadcast. Not HAL yet: the unpack descriptor unions and format helpers
// (cunpack_common.h), the Src zero-flag rule (cmath_common.h), and the software state ckernel keeps
// (unp_cfg_context, the Dest buffer base).

#pragma once

#include <array>
#include <cstdint>

#include "ckernel.h"
#include "cmath_common.h"
#include "cunpack_common.h"
#include "hal/cfg.h"
#include "hal/fpu.h"
#include "hal/mop.h"
#include "hal/replay.h"
#include "hal/sync.h"
#include "hal/tensix_counters.h"
#include "hal/unpack.h"
#include "llk_api/thread.h"
#include "llk_memory_checks.h"
#include "resolver/plan.h"

// hal/scalar.h has an LLK_ASSERT inside a constexpr function, which C++17 flags under ENABLE_LLK_ASSERT.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wc++20-extensions"
#include "hal/scalar.h"
#pragma GCC diagnostic pop

namespace llk2::blackhole
{

struct matmul_params
{
    // in0 [M, K] -> SrcB through unpacker 1; in1 [K, N] -> SrcA through unpacker 0
    DataFormat in0_l1, in0_reg, in1_l1, in1_reg;
    std::uint32_t in0_face_r_dim, in0_num_faces, in1_face_r_dim, in1_num_faces;
    std::uint32_t in0_tile_size, in1_tile_size; // in 16-byte L1 address units
    bool transpose;                             // transpose in1's faces
    std::uint32_t rt_dim, ct_dim, kt_dim;
    std::uint32_t math_fidelity; // ckernel::MathFidelity value: 0 LoFi, 2..4 HiFi2..4
    bool dest_fp32;
    DataFormat math_format;
};

constexpr matmul_params matmul_params_from(const Plan& p)
{
    return {
        p.in0.reads,
        p.in0.writes,
        p.in1.reads,
        p.in1.writes,
        p.in0.face_r_dim,
        p.in0.num_faces,
        p.in1.face_r_dim,
        p.in1.num_faces,
        p.in0.tile_bytes / 16,
        p.in1.tile_bytes / 16,
        p.in1.transpose_faces,
        p.work.rt_dim,
        p.work.ct_dim,
        p.work.kt_dim,
        p.math.math_fidelity,
        p.math.dest_fp32,
        p.math.format,
    };
}

constexpr bool reuse_in0(const matmul_params& p) // reuse_a in tt-llk: in0's tile stays while in1 streams
{
    return p.ct_dim >= p.rt_dim;
}

constexpr std::uint32_t fmt(DataFormat f)
{
    return static_cast<std::uint32_t>(f);
}

namespace cfg = hal::cfg;
using hal::sync::Semaphore;
using hal::sync::StallCondition;
using hal::sync::StallTarget;

inline constexpr Semaphore unpack_sync = Semaphore::S5; // semaphore::UNPACK_SYNC

// ---- unpack ------------------------------------------------------------------------------------------------------

inline __attribute__((always_inline)) void set_unpack_context_offset(std::uint32_t context)
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

// The replay body that streams one operand: unpack it, then move its base address on by one tile (or a row of
// tiles), in both config contexts.
template <hal::unpack::Engine E, const cfg::Field& Base, const cfg::Field& BaseCntx1, cfg::Sec S, std::uint32_t StepGpr>
inline __attribute__((always_inline)) void record_stream_body()
{
    using ckernel::p_gpr_unpack;
    constexpr hal::unpack::DataTransfer unpack {.engine = E, .handoff = hal::unpack::SourceHandoff::FlipAndSetDataValid};
    hal::replay::record<hal::replay::BufferRange {0, 12}>(
        []
        {
            hal::unpack::run<unpack>();
            cfg::read<cfg::Access::TensixCfgUnit, Base, S>(hal::gpr<p_gpr_unpack::TMP0>());
            hal::scalar::add(hal::gpr<p_gpr_unpack::TMP0>(), hal::gpr<StepGpr>(), hal::gpr<p_gpr_unpack::TMP0>());
            hal::sync::wait::stall<StallTarget::Config, StallCondition::ScalarIdle>();
            cfg::write<cfg::Access::TensixCfgUnit, Base, S, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Wait>(hal::gpr<p_gpr_unpack::TMP0>());
            hal::unpack::run<unpack>();
            cfg::read<cfg::Access::TensixCfgUnit, BaseCntx1, S>(hal::gpr<p_gpr_unpack::TMP0>());
            hal::scalar::add(hal::gpr<p_gpr_unpack::TMP0>(), hal::gpr<StepGpr>(), hal::gpr<p_gpr_unpack::TMP0>());
            hal::sync::wait::stall<StallTarget::Config, StallCondition::ScalarIdle>();
            cfg::write<cfg::Access::TensixCfgUnit, BaseCntx1, S, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Wait>(hal::gpr<p_gpr_unpack::TMP0>());
        });
}

template <matmul_params P>
inline __attribute__((always_inline)) void unpack_configure()
{
    using namespace ckernel;
    using namespace ckernel::unpacker;
    using namespace hal::counters::adc;
    static_assert(P.in0_num_faces == 4 && P.in1_num_faces == 4 && P.in0_face_r_dim == 16 && P.in1_face_r_dim == 16, "32x32 tiles only, for now");

    // SrcA (unpacker 0, THCON SEC0) carries in1; SrcB (unpacker 1, THCON SEC1) carries in0.
    constexpr std::uint32_t a_src = masked_data_format(fmt(P.in1_l1)), a_dst = masked_data_format(fmt(P.in1_reg));
    constexpr std::uint32_t b_src = masked_data_format(fmt(P.in0_l1)), b_dst = masked_data_format(fmt(P.in0_reg));

    // Wait until math has taken every context, then reset the address counters.
    while (hal::sync::semaphore::read<unpack_sync>() > 0)
    {
    }
    hal::counters::set<AddrCounter {.channel0 = {.x = 0, .y = 0}, .channel1 = {.y = 0}}.for_clients<Client::Unpacker0, Client::Unpacker1>()>();
    hal::counters::set<AddrCounter {.channel0 = {.z = 0, .w = 0}, .channel1 = {.z = 0, .w = 0}}.for_clients<Client::Unpacker0, Client::Unpacker1>()>();

    // ch1 (Src) strides: one face of the register format per Z step.
    constexpr std::uint32_t a_z_stride = FACE_C_DIM * FACE_R_DIM * datum_size_in_bytes(a_dst);
    constexpr std::uint32_t b_z_stride = FACE_C_DIM * FACE_R_DIM * datum_size_in_bytes(b_dst);
    using cfg::UnpackerReg;
    cfg::write<cfg::Access::MMIO>(
        cfg::set<cfg::Unpacker[0].AddrCtrl[UnpackerReg::Reg1].Zstride, cfg::Sec::S0, a_z_stride>(),
        cfg::set<cfg::Unpacker[0].AddrCtrl[UnpackerReg::Reg1].Wstride, cfg::Sec::S0, 0>());
    cfg::write<cfg::Access::MMIO>(
        cfg::set<cfg::Unpacker[1].AddrCtrl[UnpackerReg::Reg1].Zstride, cfg::Sec::S0, b_z_stride>(),
        cfg::set<cfg::Unpacker[1].AddrCtrl[UnpackerReg::Reg1].Wstride, cfg::Sec::S0, 0>());
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Unpacker[0].AddrCtrl[UnpackerReg::Reg1].Ystride, cfg::Sec::S0, canonical_unpA_y_stride(a_dst)>();

    // ALU unsigned-format and rounding bits, and the FP8 E4M3 bit, under the RMW mutex.
    hal::sync::mutex::acquire<hal::sync::Mutex::M0>();
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AluRoundingMode::Fpu_srnd_en, cfg::Sec::S0, 0>(),
        cfg::set<cfg::AluRoundingMode::Gasket_srnd_en, cfg::Sec::S0, 0>(),
        cfg::set<cfg::AluRoundingMode::Packer_srnd_en, cfg::Sec::S0, 0>(),
        cfg::set<cfg::AluFormatSpecReg0::SrcAUnsigned, cfg::Sec::S0, P.in1_l1 == DataFormat::UInt8 ? 1u : 0u>(),
        cfg::set<cfg::AluFormatSpecReg0::SrcBUnsigned, cfg::Sec::S0, P.in0_l1 == DataFormat::UInt8 ? 1u : 0u>());
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg1].Unp_LF8_4b_exp, cfg::Sec::S0, (a_src & 0x1F) == fmt(DataFormat::Fp8_e4m3) ? 1u : 0u>();
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg1].Unp_LF8_4b_exp, cfg::Sec::S1, (b_src & 0x1F) == fmt(DataFormat::Fp8_e4m3) ? 1u : 0u>();
    hal::sync::mutex::release<hal::sync::Mutex::M0>();

    // Tile descriptors and unpacker configs, two words each.
    unpack_tile_descriptor_u td {};
    td.f.in_data_format = a_src;
    td.f.uncompressed   = 1;
    td.f.x_dim          = 0; // unpacker 0 takes x_dim from the per-context Tile_x_dim below
    td.f.y_dim          = 1;
    td.f.z_dim          = P.in1_num_faces;
    cfg::write<cfg::Access::MMIO, cfg::Thcon[cfg::Reg0].TileDescriptor, cfg::Sec::S0, 2>(std::array<std::uint32_t, 2> {td.val[0], td.val[1]});
    td.f.in_data_format = b_src;
    td.f.x_dim          = P.in0_face_r_dim * FACE_C_DIM;
    td.f.z_dim          = P.in0_num_faces;
    cfg::write<cfg::Access::MMIO, cfg::Thcon[cfg::Reg0].TileDescriptor, cfg::Sec::S1, 2>(std::array<std::uint32_t, 2> {td.val[0], td.val[1]});

    unpack_config_u uc {};
    uc.f.out_data_format    = a_dst;
    uc.f.throttle_mode      = 2;
    uc.f.haloize_mode       = P.transpose ? 1 : 0;
    uc.f.uncompress_cntx0_3 = 0xf;
    uc.f.uncompress_cntx4_7 = 0xf;
    cfg::write<cfg::Access::MMIO, cfg::Thcon[cfg::Reg2].Out_data_format, cfg::Sec::S0, 2>(std::array<std::uint32_t, 2> {uc.val[0], uc.val[1]});
    uc.f.out_data_format = b_dst;
    uc.f.haloize_mode    = 0;
    cfg::write<cfg::Access::MMIO, cfg::Thcon[cfg::Reg2].Out_data_format, cfg::Sec::S1, 2>(std::array<std::uint32_t, 2> {uc.val[0], uc.val[1]});

    // Src write base for both contexts, and unpacker 0's face size per context.
    cfg::write<cfg::Access::MMIO>(
        cfg::set<cfg::Thcon[cfg::Reg5].Dest_cntx0_address, cfg::Sec::S0, 4 * 16>(), cfg::set<cfg::Thcon[cfg::Reg5].Dest_cntx1_address, cfg::Sec::S0, 4 * 16>());
    constexpr std::uint32_t face_dim = P.in1_face_r_dim * FACE_C_DIM;
    cfg::write<cfg::Access::MMIO>(
        cfg::set<cfg::Thcon[cfg::Reg5].Tile_x_dim_cntx0, cfg::Sec::S0, face_dim>(), cfg::set<cfg::Thcon[cfg::Reg5].Tile_x_dim_cntx1, cfg::Sec::S0, face_dim>());

    cfg::write<cfg::Access::TensixCfgUnit, cfg::SrcASet::SetOvrdWithAddr, cfg::Sec::S0, 1>();
    cfg::write<cfg::Access::MMIO, cfg::Unpacker[0].AddDestAddrCntr, cfg::Sec::S0, 1>(std::array<std::uint32_t, 1> {1u << 8});

    // Config context 0.
    unp_cfg_context = 0;
    set_unpack_context_offset(0);

    // Tile sizes and K, in GPRs, for the MOP's address steps.
    hal::scalar::set_low(hal::gpr<p_gpr_unpack::TILE_SIZE_A>(), P.in1_tile_size & 0xffff);
    hal::scalar::set_low(hal::gpr<p_gpr_unpack::TILE_SIZE_B>(), P.in0_tile_size & 0xffff);
    hal::scalar::set_low(hal::gpr<p_gpr_unpack::KT_DIM>(), P.kt_dim & 0xffff);

    // Matmul init: transpose, Z counters, X end of each unpacker.
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg2].Haloize_mode, cfg::Sec::S0, P.transpose ? 1u : 0u>();
    hal::counters::set<AddrCounter {.channel0 = {.z = 0, .w = 0}, .channel1 = {.z = 0, .w = 0}}.for_clients<Client::Unpacker0, Client::Unpacker1>()>();
    hal::counters::set<AddrCounter {.channel0 = {.x = 0}, .channel1 = {.x = P.in1_num_faces * P.in1_face_r_dim * FACE_C_DIM - 1}}.for_clients<Client::Unpacker0>()>();
    hal::counters::set<AddrCounter {.channel0 = {.x = 0}, .channel1 = {.x = P.in0_num_faces * P.in0_face_r_dim * FACE_C_DIM - 1}}.for_clients<Client::Unpacker1>()>();

    // The MOP streams the operand that is not reused: in1 (SrcA, step one tile) when in0 is reused, else in0
    // (SrcB, step a row of K tiles).
    if constexpr (reuse_in0(P))
    {
        record_stream_body<hal::unpack::Engine::Unpacker0, cfg::Thcon[cfg::Reg3].Base_address, cfg::Thcon[cfg::Reg3].Base_cntx1_address, cfg::Sec::S0, p_gpr_unpack::TILE_SIZE_A>();
    }
    else
    {
        record_stream_body<hal::unpack::Engine::Unpacker1, cfg::Thcon[cfg::Reg3].Base_address, cfg::Thcon[cfg::Reg3].Base_cntx1_address, cfg::Sec::S1, p_gpr_unpack::TMP_LO>();
    }
    hal::mop::program<hal::mop::MopConfig<hal::mop::MopTemplate::Template0> {
        .start_op        = hal::replay::get_operation<hal::replay::BufferRange {0, 6}>(),
        .start_op_shadow = hal::replay::get_operation<hal::replay::BufferRange {6, 6}>()}>();
}

// One K step. in0_address: byte address of in0's tile (0, k); in1_address: of in1's tile (k, 0).
template <matmul_params P>
inline __attribute__((always_inline)) void unpack_execute(std::uint32_t in0_address, std::uint32_t in1_address)
{
    using namespace ckernel;
    constexpr bool reuse            = reuse_in0(P);
    constexpr std::uint32_t t_dim   = reuse ? P.rt_dim : P.ct_dim;
    constexpr std::uint32_t mop_len = reuse ? P.ct_dim : P.rt_dim;
    constexpr hal::unpack::DataTransfer reused {
        .engine = reuse ? hal::unpack::Engine::Unpacker1 : hal::unpack::Engine::Unpacker0, .handoff = hal::unpack::SourceHandoff::FlipAndSetDataValid};

    if constexpr (!reuse)
    {
        hal::scalar::multiply_low_u16(hal::gpr<p_gpr_unpack::TMP_LO>(), hal::gpr<p_gpr_unpack::KT_DIM>(), hal::gpr<p_gpr_unpack::TILE_SIZE_B>());
    }

    const std::uint32_t base0 = L1_ADDRESS(in0_address);
    const std::uint32_t base1 = L1_ADDRESS(in1_address);
    for (std::uint32_t t = 0; t < t_dim; t++)
    {
        const std::uint32_t address0 = base0 + P.in0_tile_size * (reuse ? t * P.kt_dim : 0);
        const std::uint32_t address1 = base1 + P.in1_tile_size * (reuse ? 0 : t);
        LLK_ASSERT(is_valid_L1_address(address0) && is_valid_L1_address(address1), "matmul operand address is outside L1");

        while (hal::sync::semaphore::read<unpack_sync>() >= 2)
        {
        }
        if (unp_cfg_context == 0)
        {
            cfg::write<cfg::Access::MMIO, cfg::Thcon[cfg::Reg3].Base_address, cfg::Sec::S0>(address1);
            cfg::write<cfg::Access::MMIO, cfg::Thcon[cfg::Reg3].Base_address, cfg::Sec::S1>(address0);
        }
        else
        {
            cfg::write<cfg::Access::MMIO, cfg::Thcon[cfg::Reg3].Base_cntx1_address, cfg::Sec::S0>(address1);
            cfg::write<cfg::Access::MMIO, cfg::Thcon[cfg::Reg3].Base_cntx1_address, cfg::Sec::S1>(address0);
        }
        hal::sync::semaphore::post<hal::sync::Access::MMIO, unpack_sync>();
        hal::sync::wait::stall<StallTarget::Unpack, StallCondition::RiscvAccessProcessed>();

        hal::unpack::run<reused>();
        hal::mop::Runner<hal::mop::MopTemplate::Template0>::run(mop_len, unp_cfg_context == 0 ? 0 : 0xff);

        hal::sync::semaphore::get<unpack_sync>();
        unp_cfg_context = 1 - unp_cfg_context;
        set_unpack_context_offset(unp_cfg_context);
    }
}

// ---- math --------------------------------------------------------------------------------------------------------

template <cfg::Sec S, std::uint32_t AI, std::uint32_t ACl, std::uint32_t ACr, std::uint32_t BI, std::uint32_t BCl, std::uint32_t BCr, std::uint32_t DI,
          std::uint32_t DCl, std::uint32_t DCr, std::uint32_t FI = 0>
inline __attribute__((always_inline)) void set_addr_mod()
{
    using cfg::AddrMod;
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<AddrMod[cfg::SrcA].Incr, S, AI>(),
        cfg::set<AddrMod[cfg::SrcA].Clear, S, ACl>(),
        cfg::set<AddrMod[cfg::SrcA].CR, S, ACr>(),
        cfg::set<AddrMod[cfg::SrcB].Incr, S, BI>(),
        cfg::set<AddrMod[cfg::SrcB].Clear, S, BCl>(),
        cfg::set<AddrMod[cfg::SrcB].CR, S, BCr>());
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<AddrMod[cfg::Dest].Incr, S, DI>(),
        cfg::set<AddrMod[cfg::Dest].Clear, S, DCl>(),
        cfg::set<AddrMod[cfg::Dest].CR, S, DCr>(),
        cfg::set<AddrMod[cfg::Fidelity].Incr, S, FI>(),
        cfg::set<AddrMod[cfg::Fidelity].Clear, S, 0>());
    cfg::write<cfg::Access::TensixCfgUnit, AddrMod[cfg::Bias].Incr, S, 0>();
}

template <std::uint8_t AddrMod, hal::fpu::SourceRelease R = hal::fpu::SourceRelease::None>
inline __attribute__((always_inline)) void mvmul()
{
    hal::fpu::run<hal::fpu::MatrixMultiply {.address_modifier = AddrMod, .release = R}>();
}

template <matmul_params P>
inline __attribute__((always_inline)) void math_configure()
{
    using namespace ckernel;
    using namespace hal::counters::rwc;
    constexpr bool hifi = P.math_fidelity != 0;
    constexpr bool reuse = reuse_in0(P);
    static_assert(!P.transpose, "transposed matmul is not ported yet");

    // hw_configure: Dest mode and ALU format baseline.
    hal::sync::wait::stall<StallTarget::Config, StallCondition::MathIdle | StallCondition::SfpuIdle>();
    cfg::write<cfg::Access::TensixCfgUnit, cfg::DestAccessCfg::zeroacc_absolute_tile_mode, cfg::Sec::S0, 0>();
    constexpr std::uint32_t int8 = is_int8_or_int32_format(fmt(P.math_format)) ? 1u : 0u;
    cfg::write<cfg::Access::TensixCfgUnit, cfg::AluAccCtrl::INT8_math_enabled, cfg::Sec::S0, int8>();
    cfg::write<cfg::Access::TensixCfgUnit, cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S0, P.dest_fp32 ? 1u : 0u>();
    cfg::write<cfg::Access::TensixCfgUnit, cfg::AluAccCtrl::SFPU_Fp32_enabled, cfg::Sec::S0, P.dest_fp32 ? 1u : 0u>();
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AluFormatSpecReg::SrcA_val, cfg::Sec::S0, 0>(),
        cfg::set<cfg::AluFormatSpecReg::SrcA_override, cfg::Sec::S0, 0>(),
        cfg::set<cfg::AluFormatSpecReg::SrcB_val, cfg::Sec::S0, 0>(),
        cfg::set<cfg::AluFormatSpecReg::SrcB_override, cfg::Sec::S0, 0>());

    // Address modifiers for a 32x32 x 32x32 tile matmul (matmul_configure_addrmod).
    set_addr_mod<cfg::Sec::S0, 0, 0, 0, 8, 0, 0, 8, 0, 0>();
    set_addr_mod<cfg::Sec::S1, 16, 0, 0, 0, 0, 1, 8, 0, 0>();
    set_addr_mod<cfg::Sec::S2, 0, 0, 1, 32, 0, 1, 8, 0, 0>();
    set_addr_mod<cfg::Sec::S4, 32, 0, 1, 48, 0, 1, 0, 0, 1>();
    set_addr_mod<cfg::Sec::S5, 0, 1, 1, 0, 1, 1, 0, 1, 1, hifi ? 1u : 0u>();

    // One tile: 16 MVMULs in a replay. LoFi releases the streamed source on the last one; HiFi releases it in the
    // MOP's end op, after the last fidelity phase.
    constexpr auto last_release = hifi ? hal::fpu::SourceRelease::None : (reuse ? hal::fpu::SourceRelease::SrcA : hal::fpu::SourceRelease::SrcB);
    hal::replay::record<hal::replay::BufferRange {16, 16}>(
        []
        {
            mvmul<0>();
            mvmul<1>();
            mvmul<0>();
            mvmul<2>();
            mvmul<0>();
            mvmul<1>();
            mvmul<0>();
            mvmul<4>();
            mvmul<0>();
            mvmul<1>();
            mvmul<0>();
            mvmul<2>();
            mvmul<0>();
            mvmul<1>();
            mvmul<0>();
            mvmul<5, last_release>();
        });

    constexpr std::uint32_t replay_op = hal::replay::get_operation<hal::replay::BufferRange {16, 16}>();
    if constexpr (hifi)
    {
        constexpr std::uint32_t release_streamed =
            MathCounter {.src_a = Set(0), .src_b = Set(0), .dest = Set(0), .flip = reuse ? Src::A : Src::B, .clear_fidelity = true}.operation();
        hal::mop::program<hal::mop::MopConfig<hal::mop::MopTemplate::Template1> {
            .outer_loop = {.count = 1, .end_op = release_streamed}, .inner_loop = {.count = P.math_fidelity, .body_op = replay_op}}>();
    }
    else
    {
        hal::mop::program<hal::mop::MopConfig<hal::mop::MopTemplate::Template1> {.outer_loop = {.count = 1}, .inner_loop = {.count = 1, .body_op = replay_op}}>();
    }
    set<MathCounter {.src_a = Set(0), .src_b = Set(0), .dest = Set(0), .clear_fidelity = true}>();

    // Src zero-flag baseline for these formats; the ckernel caches stay in step for the LLK asserts.
    math::src_zero_flag_srca_fmt = fmt(P.math_format);
    math::src_zero_flag_srcb_fmt = fmt(P.math_format);
    const std::uint32_t zero_flag = requires_disabled_src_zero_flag(fmt(P.math_format), fmt(P.math_format)) ? 1u : 0u;
    cfg::write<cfg::Access::TensixCfgUnit, cfg::AluAccCtrl::Zero_Flag_disabled_src, cfg::Sec::S0>(zero_flag);
    math::src_zero_flag_hw = zero_flag;
}

// One K step: accumulates in0's tile column k times in1's tile row k into the rt_dim x ct_dim output tiles.
// dest_row: the Dest row of the output's first tile, relative to the current Dest section.
template <matmul_params P>
inline __attribute__((always_inline)) void math_execute(std::uint32_t dest_row)
{
    using namespace ckernel;
    using namespace hal::counters::rwc;
    constexpr bool reuse            = reuse_in0(P);
    constexpr std::uint32_t t_dim   = reuse ? P.rt_dim : P.ct_dim;
    constexpr std::uint32_t rut_dim = reuse ? P.ct_dim : P.rt_dim;
    constexpr MathCounter release_reused {.src_a = Set(0), .src_b = Set(0), .dest = Set(0), .flip = reuse ? Src::B : Src::A, .clear_fidelity = true};

    for (std::uint32_t t = 0; t < t_dim; t++)
    {
        for (std::uint32_t rut = 0; rut < rut_dim; rut++)
        {
            const std::uint32_t tile = reuse ? P.ct_dim * t + rut : t + rut * P.ct_dim;
            cfg::write<cfg::Access::TensixCfgUnit, cfg::DestTargetRegCfgMath::Offset, cfg::Sec::S0>(dest_row + (tile << 6) + get_dest_buffer_base());
            hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();
            if (rut == rut_dim - 1)
            {
                set<release_reused>();
            }
        }
    }
}

// ---- per-thread calls --------------------------------------------------------------------------------------------

template <Thread On, Plan Pl>
inline __attribute__((always_inline)) void unpack_matmul(std::uint32_t in0_address, std::uint32_t in1_address)
{
    static_assert(On == Thread::Unpack, "the unpacker is driven from the unpack thread");
    constexpr matmul_params p = matmul_params_from(Pl);
    if constexpr (runs_on<On>)
    {
        unpack_configure<p>();
        for (std::uint32_t k = 0; k < p.kt_dim; k++)
        {
            // K step k: column k of in0's tile rows, row k of in1's tile columns.
            unpack_execute<p>(in0_address + k * p.in0_tile_size * 16, in1_address + k * p.ct_dim * p.in1_tile_size * 16);
        }
    }
}

template <Thread On, Plan Pl>
inline __attribute__((always_inline)) void math_matmul(std::uint32_t dest_row)
{
    static_assert(On == Thread::Math, "the FPU is driven from the math thread");
    constexpr matmul_params p = matmul_params_from(Pl);
    if constexpr (runs_on<On>)
    {
        math_configure<p>();
        for (std::uint32_t k = 0; k < p.kt_dim; k++)
        {
            math_execute<p>(dest_row);
        }
    }
}

} // namespace llk2::blackhole

namespace llk2
{
using blackhole::math_matmul;
using blackhole::unpack_matmul;
} // namespace llk2
