// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL port of tt_llk_blackhole/llk_lib/llk_pack_common.h and the pack helpers of common/inc/cpack_common.h.
// Same Tensix instruction sequence and access kinds as production; every instruction and register access goes
// through the HAL except the listed HAL-GAPs (tensix_sync, RISC regfile[] stores).

#pragma once

#include <array>
#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "cpack_common.h" // types, constants and pure helpers only; no production instruction path is called
#include "hal/cfg.h"
#include "hal/counters/adc.h"
#include "hal/dst.h"
#include "hal/nop.h"
#include "hal/scalar.h"
#include "hal/sync.h"
#include "llk_assert.h"
#include "llk_defs.h"
#include "tensor_shape.h"

using namespace ckernel;
using namespace ckernel::packer;

namespace hal_port
{

namespace cfg  = hal::cfg;
namespace sync = hal::sync;
namespace adc  = hal::counters::adc;

// Place a value into field F of its CFG word, like the (v << SHAMT) & MASK of the raw defines.
template <const cfg::Field& F, cfg::Sec S = cfg::Sec::S0>
constexpr std::uint32_t place(const std::uint32_t value)
{
    return (value << F.shamt(S)) & F.mask(S);
}

// SETADC* sites of a thread that borrows the pack ADCs are serialized with mutex::THREAD2_ADC (M3).
template <bool enable = true>
class [[nodiscard]] AdcMutexGuard final
{
public:
    static_assert(ckernel::mutex::THREAD2_ADC == 3, "THREAD2_ADC must map to hal::sync::Mutex::M3");

    AdcMutexGuard() noexcept
    {
        if constexpr (enable)
        {
            sync::mutex::acquire<sync::Mutex::M3>();
        }
    }

    ~AdcMutexGuard()
    {
        if constexpr (enable)
        {
            sync::mutex::release<sync::Mutex::M3>();
        }
    }

    AdcMutexGuard(const AdcMutexGuard&)            = delete;
    AdcMutexGuard& operator=(const AdcMutexGuard&) = delete;
    AdcMutexGuard(AdcMutexGuard&&)                 = delete;
    AdcMutexGuard& operator=(AdcMutexGuard&&)      = delete;
};

// Set unpacker offsets to 0, except for unpacker 0, channel 1, X, which is the tile X dimension
// SETADCXY(PAC, mask 0b1011) + SETADCZW(PAC, mask 0b1111)
inline void packer_addr_counter_init()
{
    adc::set<adc::AddrCounter {.channel0 = {.x = 0, .y = 0, .z = 0, .w = 0}, .channel1 = {.y = 0, .z = 0, .w = 0}}.for_clients<adc::Client::Packers>()>();
}

template <PackMode pack_mode = PackMode::Default>
inline void set_packer_strides(const std::uint32_t pack_src_format, const std::uint32_t tile_c_dim)
{
    constexpr auto Ch0 = cfg::Packer[0].AddrCtrl[cfg::PackerReg::Reg0];
    constexpr auto Ch1 = cfg::Packer[0].AddrCtrl[cfg::PackerReg::Reg1];

    std::uint32_t x_stride = datum_size_in_bytes(pack_src_format);
    std::uint32_t y_stride = FACE_C_DIM * x_stride;
    std::uint32_t w_stride = TILE_NUM_FACES * FACE_C_DIM * FACE_R_DIM * x_stride;

    // Untilize mode has 2 packer interfaces active, so z counter needs to jump by 2
    // faces, since z counter is only 1 bit (can't be programmed to inc by 2)
    const std::uint32_t z_stride = ((pack_mode != PackMode::Default) && (tile_c_dim == TILE_C_DIM)) ? 2 * FACE_R_DIM * y_stride : FACE_R_DIM * y_stride;

    // x-stride not used! The low half of the Y-stride word is a compile-time zero (compile-time form of the SETDMAREG).
    static_assert(Ch0.Ystride.shamt(cfg::Sec::S0) == 16, "Y stride must occupy the high half of its CFG word");
    hal::scalar::set_low<0 /*Value*/>(hal::gpr<p_gpr_pack::TMP0>());
    hal::scalar::set_high(hal::gpr<p_gpr_pack::TMP0>(), place<Ch0.Ystride>(y_stride) >> 16);
    hal::scalar::set(hal::gpr<p_gpr_pack::TMP1>(), place<Ch0.Zstride>(z_stride) | place<Ch0.Wstride>(w_stride));
    sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::ScalarIdle>();
    cfg::write<cfg::Access::TensixCfgUnit, Ch0.Xstride, cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Deferred>(
        hal::gpr<p_gpr_pack::TMP0>());
    cfg::write<cfg::Access::TensixCfgUnit, Ch0.Zstride, cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Deferred>(
        hal::gpr<p_gpr_pack::TMP1>());
    // Channel-1 (L1 output) strides are zero for every pack mode: output placement comes from
    // L1_Dest_addr and the packer's write pointer. Only the fast-untilize strided pack sets a
    // channel-1 (Y) stride, in its own init.
    cfg::write<cfg::Access::TensixCfgUnit, Ch1.Xstride, cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Deferred>(hal::gpr<p_gpr::ZERO>());
    cfg::write<cfg::Access::TensixCfgUnit, Ch1.Zstride, cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Deferred>(hal::gpr<p_gpr::ZERO>());
    hal::nop::thread();
    hal::nop::thread();
}

// Compile-time form of production's reconfigure_packer_l1_acc(pack_l1_acc); the only caller passes 0.
template <std::uint32_t pack_l1_acc>
inline void reconfigure_packer_l1_acc()
{
    constexpr auto R1 = cfg::Thcon[cfg::Reg1];

    // Stall to avoid clobbering current packer configuration, then enable/disable L1 accumulation
    sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::PackerIdle>();
    cfg::write<cfg::Access::TensixCfgUnit, R1.Pack_L1_Acc, cfg::Sec::S0, pack_l1_acc>();
    // Accumulating writes are bank atomics; with the fast tile end drain the lockstep packer ports can settle into a slow rate.
#if !defined(TT_METAL_TTSIM) // ttsim does not model PACK_GLOBAL_CFG_CTL and aborts on any access to it
    cfg::write<cfg::Access::TensixCfgUnit, cfg::PackGlobalCfgCtl::pack_disable_fast_tile_end_drain, cfg::Sec::S0, pack_l1_acc ? 1 : 0>();
#endif
}

/**
 * @brief Configure packer exponent thresholding for the current destination and output formats.
 *
 * We use packer exponent thresholding to zero out datums that are too small to be repesentable
 * This is specifically done in FP32 (Dest) -> BFPxA (L1) case.
 *
 * This implies that exponent thresholding should be reconfigured whenever packer formats change.
 *
 * @tparam is_fp32_dest_acc_en True when Dest register is FP32.
 * @param pack_dst_format Pack output data format.
 */
template <bool is_fp32_dest_acc_en>
inline void reconfigure_exp_threshold(const std::uint32_t pack_dst_format)
{
    constexpr auto R1 = cfg::Thcon[cfg::Reg1];

    if constexpr (is_fp32_dest_acc_en)
    {
        bool enable             = false;
        std::uint32_t threshold = 0;

        // Workaround for HW bug: tenstorrent/budabackend#1394
        if (IS_BFP_A_FORMAT(pack_dst_format))
        {
            // After early format conversion the exponent is EXP_B; packing to BFP-A uses EXP_A.
            // Zero out values too small to represent in EXP_A (EXP_B < 113).
            enable    = true;
            threshold = 113;
        }

        // Exp_threshold_en and Exp_threshold share one CFG word: one grouped RMW.
        cfg::write<cfg::Access::TensixCfgUnit>(
            cfg::set<R1.Exp_threshold_en, cfg::Sec::S0>(enable ? 1 : 0), cfg::set<R1.Exp_threshold, cfg::Sec::S0>(threshold));
    }
    else
    {
        // Without an FP32 Dest the enable and the threshold are compile-time zero: immediate form.
        cfg::write<cfg::Access::TensixCfgUnit>(cfg::set<R1.Exp_threshold_en, cfg::Sec::S0, 0>(), cfg::set<R1.Exp_threshold, cfg::Sec::S0, 0>());
    }
}

template <bool is_fp32_dest_acc_en>
inline void set_packer_config(
    const std::uint32_t pack_src_format, const std::uint32_t pack_dst_format, const std::uint32_t num_faces = 4, const bool partial_face = false)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    constexpr auto R1 = cfg::Thcon[cfg::Reg1];

    const std::uint32_t pack_output_src_format = masked_data_format(pack_src_format);
    const std::uint32_t pack_output_dst_format = masked_data_format(pack_dst_format);
    // Gasket converts Float16_b -> Float16 before the packer, so hardware in_data_format must be Float16 for Fp8 output.
    const std::uint32_t pack_hw_src_format =
        ((pack_dst_format & 0x1F) == to_underlying(DataFormat::Fp8_e4m3)) ? to_underlying(DataFormat::Float16) : pack_output_src_format;

    // Packer config words 0 and 2 (word 1 is L1_Dest_addr, set up on every pack)
    const std::uint32_t exp_section_size =
        ((pack_output_dst_format == to_underlying(DataFormat::Lf8)) || (pack_output_dst_format == to_underlying(DataFormat::Int8)))
            ? 0
            : (partial_face ? 1 : num_faces); // set to num_faces as exp section size is not used for non-bfp formats except for lf8/int8

    const std::uint32_t config_word0 = place<R1.Exp_section_size>(exp_section_size);
    const std::uint32_t config_word2 =
        place<R1.Disable_zero_compress>(1 /*uncompress*/) | place<R1.Out_data_format>(pack_output_dst_format) | place<R1.In_data_format>(pack_hw_src_format);

    reconfigure_exp_threshold<is_fp32_dest_acc_en>(pack_output_dst_format);

    cfg::write<cfg::Access::MMIO, R1.Row_start_section_size, cfg::Sec::S0, 1 /*Count*/>(std::array<std::uint32_t, 1> {config_word0});
    cfg::write<cfg::Access::MMIO, R1.Disable_zero_compress, cfg::Sec::S0, 1 /*Count*/>(std::array<std::uint32_t, 1> {config_word2});

    // Reset L1 accumulation flag
    reconfigure_packer_l1_acc<0>();

    const bool is_32b_format = pack_src_format == to_underlying(DataFormat::Int32) || pack_src_format == to_underlying(DataFormat::UInt32) ||
                               pack_src_format == to_underlying(DataFormat::Float32);
    const bool is_int8_format = pack_src_format == to_underlying(DataFormat::Int8) || pack_src_format == to_underlying(DataFormat::UInt8);

    const bool read_32b_data = is_32b_format || is_fp32_dest_acc_en;
    const bool read_int8     = !(is_fp32_dest_acc_en || is_32b_format) && is_int8_format;
    const bool read_unsigned = pack_dst_format == to_underlying(DataFormat::UInt8);

    // Round to 10 bit mantissa from fp32 dest
    const bool round_10b_mant = (is_fp32_dest_acc_en && (pack_src_format == to_underlying(DataFormat::Float16))) ||
                                (!IS_A_FORMAT(pack_src_format) && (pack_dst_format & 0x1F) == to_underlying(DataFormat::Fp8_e4m3));

    const std::uint32_t dest_rd_ctrl_word = place<cfg::PckDestRdCtrl::Read_32b_data>(read_32b_data) | place<cfg::PckDestRdCtrl::Read_unsigned>(read_unsigned) |
                                            place<cfg::PckDestRdCtrl::Read_int8>(read_int8) | place<cfg::PckDestRdCtrl::Round_10b_mant>(round_10b_mant);
    cfg::write<cfg::Access::MMIO, cfg::PckDestRdCtrl::Read_32b_data, cfg::Sec::S0, 1 /*Count*/>(std::array<std::uint32_t, 1> {dest_rd_ctrl_word});

    // Save to GPR for quick data format reconfig
    // HAL-GAP: RISC MMIO regfile[] store (and the sync read-back) has no HAL API.
    regfile[p_gpr_pack::EXP0_SEC_SIZE_BFP] = (partial_face ? 1 : num_faces) << THCON_SEC0_REG8_Exp_section_size_SHAMT;
    sync_regfile_write(p_gpr_pack::EXP0_SEC_SIZE_BFP);
}

template <bool is_fp32_dest_acc_en, PackMode pack_mode = PackMode::Default>
inline void configure_pack(
    const std::uint32_t pack_src_format,
    const std::uint32_t pack_dst_format,
    const std::uint32_t tile_size,
    [[maybe_unused]] const std::uint32_t face_r_dim = FACE_R_DIM,
    const std::uint32_t tile_c_dim                  = TILE_C_DIM,
    const std::uint32_t num_faces                   = 4,
    const bool partial_face                         = false,
    const std::uint32_t relu_config                 = 0)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    LLK_ASSERT(
        is_packer_to_L1_conversion_supported(static_cast<DataFormat>(pack_src_format & 0xF), static_cast<DataFormat>(pack_dst_format & 0xF)),
        "Unsupported packer to L1 conversion.");

    constexpr auto R1 = cfg::Thcon[cfg::Reg1];

    // Drain the packer before any of the configuration below is rewritten, whatever the formats. The raw
    // cfg[] stores in this function and in set_packer_config are RISC MMIO, which no STALLWAIT can hold
    // back: the STALLWAIT only keeps new packer instructions out, and tensix_sync() blocks this RISC until
    // the packer has gone idle, which is what orders those stores behind a pack still in flight. Without
    // it the pack samples the new format/width mid-tile.
    sync::wait::stall<sync::StallTarget::Pack, sync::StallCondition::PackerIdle>();
    ckernel::tensix_sync(); // HAL-GAP: blocking RISC store to pc_buf has no HAL API (HAL's own atomic.h calls ckernel::tensix_sync directly)

    const std::uint32_t pack_output_src_format = masked_data_format(pack_src_format);

    set_packer_strides<pack_mode>(pack_src_format, tile_c_dim);

    // NOTE: the packer X (datum) counter (SETADCXX PAC) is intentionally NOT programmed here. It is
    // owned by _llk_pack_init_, which runs after hw-configure and sets its own value per the
    // "inits own SETADCXX" contract.

    sync::mutex::acquire<sync::Mutex::M0>(); // mutex::REG_RMW

    // Set Fp8 E4M3 mode for packer
    const bool is_fp8_e4m3 = (pack_dst_format & 0x1F) == static_cast<std::uint32_t>(DataFormat::Fp8_e4m3);
    cfg::write<cfg::Access::TensixCfgUnit, R1.Pac_LF8_4b_exp, cfg::Sec::S0>(is_fp8_e4m3 ? 1 : 0);

    cfg::write<cfg::Access::TensixCfgUnit, cfg::AluFormatSpecReg2::Dstacc, cfg::Sec::S0>(
        is_fp8_e4m3 ? to_underlying(DataFormat::Float16) : pack_output_src_format);

    // Establish the no-override baseline for the Dstacc ALU format-select fields (ADDR32=0), consumed
    // by the packer: clear Dstacc_val/override so the base REG2_Dstacc format programmed above is
    // used. The SrcA/SrcB override fields in the same word are owned by the math thread
    // (_llk_math_hw_configure_); the two writers touch disjoint bits and rely on per-byte RMWCIB
    // atomicity, so this write does not depend on the surrounding REG_RMW mutex for cross-thread safety.
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AluFormatSpecReg::Dstacc_val, cfg::Sec::S0, 0>(), cfg::set<cfg::AluFormatSpecReg::Dstacc_override, cfg::Sec::S0, 0>());

    // Config RELU
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::StaccRelu::ApplyRelu, cfg::Sec::S0>(relu_config & 0xf),
        cfg::set<cfg::StaccRelu::ReluThreshold, cfg::Sec::S0>((relu_config >> 16) & 0xffff));

    sync::mutex::release<sync::Mutex::M0>(); // mutex::REG_RMW

    set_packer_config<is_fp32_dest_acc_en>(pack_src_format, pack_dst_format, num_faces, partial_face);

    // Number of reads per face used for resetting tile position generator for edge masks.
    // pack_counters word: only pack_reads_per_xy_plane = 1 (disable auto last generation)
    for (std::uint32_t i = 0; i < NUM_PACKERS; i++)
    {
        static_assert(NUM_PACKERS == 1, "single packer: one PACK_COUNTERS word");
        cfg::write<cfg::Access::MMIO, cfg::PackCounters::pack_per_xy_plane, cfg::Sec::S0, 1 /*Count*/>(
            std::array<std::uint32_t, 1> {place<cfg::PackCounters::pack_reads_per_xy_plane>(1)});
    }

    cfg::write<cfg::Access::MMIO, cfg::PckEdgeOffsetSec0::mask, cfg::Sec::S0, 1 /*Count*/>(
        std::array<std::uint32_t, 1> {place<cfg::PckEdgeOffsetSec0::mask>(0xffff)});
    cfg::write<cfg::Access::MMIO, cfg::TileRowSetMapping[0][0], cfg::Sec::S0, 1 /*Count*/>(
        std::array<std::uint32_t, 1> {0}); // All packers use row set mapping 0, edge offset 0 mask
    cfg::write<cfg::Access::MMIO, cfg::PckEdgeTileFaceSetSelect::select, cfg::Sec::S0, 1 /*Count*/>(std::array<std::uint32_t, 1> {0});

    // HAL-GAP: RISC MMIO regfile[] stores (and the sync read-back) have no HAL API.
    regfile[p_gpr_pack::TILE_HEADER]     = tile_size;
    regfile[p_gpr_pack::TILE_HEADER + 1] = 0;
    regfile[p_gpr_pack::TILE_HEADER + 2] = 0;
    regfile[p_gpr_pack::TILE_HEADER + 3] = 0;
    sync_regfile_write(p_gpr_pack::TILE_HEADER + 3);
}

inline std::uint8_t get_packer_dest_offset_index()
{
    return (dest_offset_id ? p_gpr_pack::DEST_OFFSET_HI : p_gpr_pack::DEST_OFFSET_LO);
}

inline std::uint32_t get_packer_dest_offset()
{
    return (dest_offset_id ? DEST_REGISTER_HALF_SIZE : 0x0);
}

inline void flip_packer_dest_offset_id()
{
    dest_offset_id = 1 - dest_offset_id;
}

inline void reset_dest_offset_id()
{
    dest_offset_id = 0;
}

// Flip packer dest register offset to 0 or DEST_REGISTER_HALF_SIZE
// flip-flopping between two halves
template <DstSync Dst>
inline void select_packer_dest_registers()
{
    if constexpr (Dst == DstSync::SyncFull)
    {
        cfg::write<cfg::Access::TensixCfgUnit, cfg::DestTargetRegCfgPack::Offset, cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Deferred>(
            hal::gpr<p_gpr_pack::DEST_OFFSET_LO>());
    }
    else
    {
        cfg::write<cfg::Access::TensixCfgUnit, cfg::DestTargetRegCfgPack::Offset, cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Deferred>(
            hal::gpr(get_packer_dest_offset_index()));
    }
    hal::nop::scalar();
    hal::nop::scalar();
}

// Program packer destination addresses from GPRs
inline void program_packer_destination(const std::uint32_t addr)
{
    LLK_ASSERT(is_valid_L1_address(addr), "L1 address must be in valid L1 memory region");
    // No packer-drain STALLWAIT is needed before reprogramming the destination: the pack thread issues its
    // instruction stream in order, so each tile's PACR has already started -- and latched L1_Dest_addr,
    // sampled at PACR start -- before the next call's WRCFG reprograms it, and the Last=1 PACR that ends each
    // pack MOP drains the packer and forces the next pack to re-sample L1_Dest_addr. The STALLWAIT below is
    // only the GPR-producer fence: it ensures the SETDMAREG write to OUTPUT_ADDR retires before WRCFG reads it.
    // The address halves are computed once: bit 31 is set in the high half for the WRCFG, then cleared again.
    constexpr auto output_addr  = hal::gpr<p_gpr_pack::OUTPUT_ADDR>();
    const std::uint32_t addr_hi = UPPER_HALFWORD(addr);
    hal::scalar::set_low(output_addr, LOWER_HALFWORD(addr));
    hal::scalar::set_high(output_addr, addr_hi | (1u << 15));

    sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::ScalarIdle>();
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg1].L1_Dest_addr, cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Deferred>(
        output_addr);

    hal::scalar::set_high(output_addr, addr_hi);
    hal::nop::scalar(); // One NOP should be enough for WRCFG due to SETDMAREG above.
}

/**
 * @brief Select which destination-register tile the packer reads from.
 *
 * Sets the packer CH0 W counter to tile_index, which addresses the tile within the destination
 * register that subsequent PACR instructions pack out.
 *
 * @tparam mutex_ADC: When true, serialize the SETADC issue against mutex::THREAD2_ADC. Needed only when
 *         another thread borrows the pack thread's ADCs; forwarded from @ref _llk_pack_.
 * @param tile_index: Index of the source tile in the destination register.
 */
template <bool mutex_ADC = false>
inline void set_dst_write_addr(const std::uint32_t tile_index)
{
    AdcMutexGuard<mutex_ADC> guard;
    adc::set(adc::AddrCounter {.channel0 = {.w = tile_index}}.for_clients<adc::Client::Packers>());
}

} // namespace hal_port

/**
 * @brief Stall the packer until the math thread has produced data to pack.
 *
 * Waits on the MATH_PACK semaphore so the packer does not run ahead of the math result.
 */
// wait until math is done and has produced something to pack
inline void _llk_packer_wait_for_math_done_()
{
    // The mask must cover STALL_SYNC as well as STALL_TDMA: the Wait Gate only blocks classes named in it, so with a
    // TDMA-only mask the Sync-class ATGETM of a mutexed _llk_pack_ slips past this unmet wait and takes
    // mutex::THREAD2_ADC, the SETADC behind it blocks head-of-line, and the ATRELM is stranded behind that. The packer
    // then waits on MATH_PACK holding the mutex the unpack thread needs every iteration: deadlock. Naming the Sync
    // class costs nothing when unmutexed.
    hal::sync::wait::semaphore<
        hal::sync::StallTarget::Tdma | hal::sync::StallTarget::Sync,
        hal::sync::SemaphoreCondition::WhileZero,
        static_cast<hal::sync::Semaphore>(semaphore::MATH_PACK)>();
}

/**
 * @brief Signal the MATH_PACK semaphore (decrement via SEMGET) to release the math thread.
 *
 * Tells math it may overwrite the destination register again now that the packer has consumed it.
 *
 * @tparam WaitRes: p_stall resource mask to stall on before signalling (e.g. p_stall::NONE, p_stall::PACK); default p_stall::NONE issues no stall.
 */
// Tell math that it can write again
template <std::uint32_t WaitRes = p_stall::NONE>
inline void _llk_packer_set_math_semaphore_()
{
    if constexpr (WaitRes != p_stall::NONE)
    {
        hal::sync::wait::stall<hal::sync::StallTarget::Sync, static_cast<hal::sync::StallCondition>(WaitRes)>();
    }
    hal::sync::semaphore::get<static_cast<hal::sync::Semaphore>(semaphore::MATH_PACK)>(); // Indicate that packer is done and header is written into L1
}

/**
 * @brief Finish a destination-register section: wait for pack, optionally clear dest, and release math.
 *
 * Waits until the pack completes, then signals the MATH_PACK semaphore. If clear_dest is true,
 * first zeroes the just-packed dest region (all of dest for SyncFull, the active half for SyncHalf).
 * For SyncHalf it also flips the dest-offset id and re-selects the packer dest registers so the
 * next half can be packed.
 *
 * @tparam Dst: Destination sync mode, values = <SyncHalf/SyncFull>
 * @tparam is_fp32_dest_acc_en: True if the destination register accumulates in FP32.
 * @tparam clear_dest: Clear on release (default). If false, MATH must clear after acquiring each section.
 */
// Wait for all writes to complete in L1 (header + data)
// Tell math it can write again
// Clear dest only when clear_dest is true
template <DstSync Dst, bool is_fp32_dest_acc_en, bool clear_dest = true>
inline void _llk_pack_dest_section_done_()
{
    namespace dst  = hal::dst;
    namespace sync = hal::sync;

    constexpr dst::DestWidth width = is_fp32_dest_acc_en ? dst::DestWidth::Bits32 : dst::DestWidth::Bits16;

    if constexpr (clear_dest)
    {
        sync::wait::stall<sync::StallTarget::Math, sync::StallCondition::PackerIdle>(); // wait for pack to finish

        if constexpr (Dst == DstSync::SyncFull)
        {
            // ADDR_MOD_1: the hardware applies ZEROACC's ADDR_MOD in every mode
            dst::run<dst::Zero {.scope = dst::ZeroScope::All, .width = width, .address_mode = ADDR_MOD_1}>();
        }
        else
        {
            static_assert(Dst == DstSync::SyncHalf);
            dst::run(dst::Zero {.scope = dst::ZeroScope::Half, .index = dest_offset_id % 2, .width = width, .address_mode = ADDR_MOD_1});
        }

        // Tell math that it can write again
        _llk_packer_set_math_semaphore_<p_stall::NONE>();
    }
    else
    {
        // Without ZEROACC, block SEMGET itself until PACK has finished reading DST.
        _llk_packer_set_math_semaphore_<p_stall::PACK>();
    }

    if constexpr (Dst == DstSync::SyncHalf)
    {
        hal_port::flip_packer_dest_offset_id();
        hal_port::select_packer_dest_registers<Dst>();
    }
}

/**
 * @brief Initialize the packer destination-offset GPRs and select the dest registers.
 *
 * Programs the low/high dest-offset GPRs for row-major order and selects the packer destination
 * registers for the chosen sync mode.
 *
 * @tparam Dst: Destination sync mode, values = <SyncHalf/SyncFull>
 */
template <DstSync Dst>
inline void _llk_init_packer_dest_offset_registers_()
{
    namespace sync = hal::sync;

    sync::wait::stall<sync::StallTarget::Tdma | sync::StallTarget::Thcon, sync::StallCondition::PackerIdle>(); // wait for pack to finish

    // RowMajor order
    hal::scalar::set_low<0 /*Value*/>(hal::gpr<p_gpr_pack::DEST_OFFSET_LO>());
    hal::scalar::set_low<DEST_REGISTER_HALF_SIZE>(hal::gpr<p_gpr_pack::DEST_OFFSET_HI>());

    sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::ScalarIdle>();
    hal_port::select_packer_dest_registers<Dst>();
}

/**
 * @brief Initialize packer destination state, at kernel start or as a per-block re-init.
 *
 * Resets the dest-offset id, programs the packer dest-offset registers, initializes the packer address
 * counter, and resets the tile dest pointer. Everything it issues is a Tensix instruction ordered behind
 * the packer by the in-stream wait in @ref _llk_init_packer_dest_offset_registers_.
 *
 * @note Does not drain this RISC (no tensix_sync). A RISC MMIO packer-config write that must follow a
 *       pack still in flight needs its own drain; @ref _llk_pack_hw_configure_ provides one.
 * @note Pair with @ref _llk_math_pack_sync_init_ on the math thread, which re-seeds the MATH_PACK handshake.
 *
 * @tparam Dst: Destination sync mode, values = <SyncHalf/SyncFull>
 * @tparam is_fp32_dest_acc_en: True if the destination register accumulates in FP32.
 * @ref _llk_init_packer_dest_offset_registers_ performs the dest-offset register setup.
 */
template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_pack_dest_init_()
{
    hal_port::reset_dest_offset_id();
    _llk_init_packer_dest_offset_registers_<Dst>();
    hal_port::packer_addr_counter_init();
    pack_sync_tile_dst_ptr = 0;
}
