// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Hardware property test: when does the packer take Pack_L1_Acc (THCON SEC0 REG1)? llk_lib/pack.h switches the bit
// between two PACRs of a sum pack behind a wait for the packer, and a compiler that tracks Tensix state deletes that
// wait (and pack_configure's) on the model that a PACR takes its config when it is issued. This test issues the
// switch with NO wait for the packer, SWITCH_AT PACRs into a tile, and the Python side checks exactly which datums
// of the tile were written with which value of the bit:
//   taken per PACR at issue   the first 64 * SWITCH_AT datums see ACC_FROM, the rest ACC_TO
//   read later / live         the boundary moves below 64 * SWITCH_AT
// SWITCH_AT = 16 is the case the deleted wait leaves: the write right behind the tile's last PACR (USE_MOP: behind the
// MOP that issues the tile's PACRs back to back, as fast as the packer can take them). SWITCH_AT = 15 is its
// positive control: one PACR earlier, and the last 4 rows must change.
//
// The kernel: a matmul's result Y (1 x TILES tiles) is packed once the usual way, so L1 holds Y. Then PASSES times:
// the matmul again, and every tile t of it packed to L1 tile t by latch_pack_tile below: configure as pack does
// (llk_lib::pack_configure, one tile), then the experiment's own execute in HAL calls. Datums packed with the bit
// off end as Y, datums packed with it on as (PASSES + 1) * Y.
//
// Not the shape of an op: the execute below is this test's, on purpose (it is the sequence pack must not rely on
// without this test passing).

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "params.h"

// ckernel_globals.h declares these extern; unp_cfg_context is the unpack context unpack.h flips.
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#define LLK2_DEST_FP32     is_fp32_dest_acc_en
#define LLK2_MATH_FIDELITY static_cast<std::uint32_t>(MATH_FIDELITY)
#include "compute_api/dest.h"
#include "compute_api/matmul.h"
#include "compute_api/pack.h"

constexpr DataFormat in0_format = static_cast<DataFormat>(UNPACK_A_IN);
constexpr DataFormat in1_format = static_cast<DataFormat>(UNPACK_B_IN);
constexpr DataFormat out_format = static_cast<DataFormat>(PACK_OUT);

inline constexpr auto& in0_spec  = llk2::l1_spec<llk2::default_tile_dim, llk2::default_tile_dim, in0_format>;
inline constexpr auto& in1_spec  = llk2::l1_spec<llk2::default_tile_dim, TILES * llk2::default_tile_dim, in1_format>;
inline constexpr auto& out_spec  = llk2::l1_spec<llk2::default_tile_dim, TILES * llk2::default_tile_dim, out_format>;
inline constexpr auto& tile_spec = llk2::l1_spec<llk2::default_tile_dim, llk2::default_tile_dim, out_format>;
inline constexpr auto& acc_spec  = llk2::dst_spec<llk2::default_tile_dim, TILES * llk2::default_tile_dim, llk2::kernel_dest_format>;

static_assert(SWITCH_AT <= 16, "a tile is 16 PACRs");
static_assert(!USE_MOP || SWITCH_AT == 16, "the MOP issues the whole tile: the switch can only follow it");

namespace llk2::llk_lib
{

/// One Dest tile to one L1 tile, with Pack_L1_Acc switched from From to To behind PACR number SwitchAt - 1 and no
/// wait for the packer in front of the switch. P is configured (pack_configure<P>, one tile, block form: its MOP is
/// the tile's 16 PACRs, the last one with Last and Y back to 0).
template <pack_params P, std::uint32_t From, std::uint32_t To, std::uint32_t SwitchAt, bool UseMop>
inline __attribute__((always_inline)) void latch_pack_tile(std::uint32_t dest_row, std::uint32_t l1_address)
{
    using namespace hal::counters::adc;
    using ckernel::p_gpr_pack;
    static_assert(pack_in_blocks(P) && P.tiles == 1 && pack_pacrs_per_tile(P) == 16);

    hal::scalar::set(hal::gpr<p_gpr_pack::TMP0>(), dest_row);
    hal::scalar::set(hal::gpr<p_gpr_pack::OUTPUT_ADDR>(), pack_packer0_addr_flag | L1_ADDRESS(l1_address));
    // The packer is idle here: the tile starts with Pack_L1_Acc = From.
    hal::sync::wait::stall<StallTarget::Config, StallCondition::PackerIdle | StallCondition::ScalarIdle>();
    cfg::write<cfg::Access::TensixCfgUnit, cfg::ThconReg1Fields::Pack_L1_Acc, cfg::Sec::S0, From>();
    cfg::write<cfg::Access::TensixCfgUnit, cfg::DestTargetRegCfgPack::Offset, cfg::Sec::S0>(hal::gpr<p_gpr_pack::TMP0>());
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg1].L1_Dest_addr, cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Wait>(
        hal::gpr<p_gpr_pack::OUTPUT_ADDR>());
    hal::counters::set<AddrCounter {.channel0 = {.w = 0}}.for_clients<Client::Packers>()>();
    hal::sync::wait::stall<StallTarget::Pack, StallCondition::ConfigUnitIdle>();

    if constexpr (UseMop)
    {
        hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();
    }
    else
    {
#pragma GCC unroll 16
        for (std::uint32_t i = 0; i < 16; i++)
        {
            if (i == SwitchAt)
            {
                // The switch, behind PACR i - 1 with no wait for the packer; the PACRs behind it wait for the write
                // to land, as pack_execute's do.
                cfg::write<cfg::Access::TensixCfgUnit, cfg::ThconReg1Fields::Pack_L1_Acc, cfg::Sec::S0, To>();
                hal::sync::wait::stall<StallTarget::Pack, StallCondition::ConfigUnitIdle>();
            }
            if (i + 1 < 16)
            {
                hal::pack::run<hal::pack::DataTransfer {
                    .address_modifier = pack_next_rows, .interfaces = pack_all_read_interfaces, .boundary = hal::pack::TileBoundary::NotLast}>();
            }
            else
            {
                hal::pack::run<hal::pack::DataTransfer {
                    .address_modifier = pack_run_end, .interfaces = pack_all_read_interfaces, .boundary = hal::pack::TileBoundary::Last}>();
            }
        }
    }
    if constexpr (SwitchAt == 16)
    {
        // Right behind the tile's last PACR: nothing between them, the packer still has the tile.
        cfg::write<cfg::Access::TensixCfgUnit, cfg::ThconReg1Fields::Pack_L1_Acc, cfg::Sec::S0, To>();
    }
}

} // namespace llk2::llk_lib

/// The experiment's pack of every tile of src to the same tile of dst: the hand-off and the configure of a pack, then
/// latch_pack_tile per tile. src_tile and dst_tile only carry the specs of one tile (a plain one-tile pack's Plan).
template <
    const llk2::TensorSpec<llk2::tensor_rank>& D,
    class RD,
    const llk2::TensorSpec<llk2::tensor_rank>& O,
    class RO,
    const llk2::TensorSpec<llk2::tensor_rank>& SrcTile,
    class RSrcTile,
    const llk2::TensorSpec<llk2::tensor_rank>& DstTile,
    class RDstTile>
inline __attribute__((always_inline)) void latch_pack(
    const llk2::Tensor<D, RD>& src,
    const llk2::Tensor<O, RO>& dst,
    const llk2::Tensor<SrcTile, RSrcTile>& /*src_tile*/,
    const llk2::Tensor<DstTile, RDstTile>& /*dst_tile*/)
{
    using namespace llk2;
    constexpr const Resolved& r      = resolved<pack_op, SrcTile, DstTile, pack_modifiers {}>;
    constexpr llk_lib::pack_params params = llk_lib::pack_params_from(r.plan);
    llk_lib::acquire<Thread::Pack, Exu::Packer, D>();
    llk_lib::acquire<Thread::Pack, Exu::Packer, O>();
    if constexpr (runs_on<Thread::Pack>)
    {
        constexpr std::uint32_t rows_per_tile = params.num_faces * params.face_r_dim;
        llk_lib::pack_configure<params>();
        for (std::uint32_t tile = 0; tile < TILES; tile++)
        {
            llk_lib::latch_pack_tile<params, ACC_FROM, ACC_TO, SWITCH_AT, USE_MOP>(
                src.address + tile * rows_per_tile, dst.address + tile * params.tile_words * llk2::l1_word_bytes);
        }
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
    const auto x   = llk2::Tensor<in0_spec>(params.buffer_A[0]);
    const auto w   = llk2::Tensor<in1_spec>(params.buffer_B[0]);
    const auto out = llk2::Tensor<out_spec>(params.buffer_Res[0]);

    using AccRing = llk2::DestRing<acc_spec>;
    {
        const auto y = llk2::matmul(x, w, llk2::reserve_back<AccRing>());
        llk2::pack(y, out); // L1 holds Y
        llk2::release(y);
    }
    for (std::uint32_t pass = 0; pass < PASSES; pass++)
    {
        const auto y = llk2::matmul(x, w, llk2::reserve_back<AccRing>());
        latch_pack(y, out, llk2::dst_tensor<llk2::default_tile_dim, llk2::default_tile_dim, llk2::kernel_dest_format>(0), llk2::Tensor<tile_spec>(0));
        llk2::release(y);
    }
}
