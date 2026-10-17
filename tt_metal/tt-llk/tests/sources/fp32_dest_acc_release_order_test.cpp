// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Regression test for the mid-kernel FP32 dest-acc handshake (`_llk_set_fp32_dest_acc_`): when it
// returns on UNPACK and PACK, the dest-acc config MATH just programmed must already be in effect, and
// the PCK_DEST_RD_CTRL Read_32b_data / Read_int8 bits must be the ones MATH derives from the mode and
// the pack source format PACK reported (cycled over Int8, UInt8, Int32, UInt32, Float32, Float16_b).
//
// Built with LLK_ASSERT compiled out, as tt-metal builds kernels by default, so the handshake runs
// with its mailbox values unchecked, exactly as it does in production. The harness enables asserts
// by default, so this file switches them off itself rather than depending on how it is run.
#undef ENABLE_LLK_ASSERT

#include <cstdint>

#include "build.h"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_fp32_dest_acc.h"

using namespace ckernel;

// Globals the harness links against.
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

namespace
{
constexpr std::uint32_t CYCLES = 256;
// Sent by UNPACK and PACK once their last readback is done, so that MATH does not restore the dest-acc
// fields under the final snapshot.
constexpr std::uint32_t SNAPSHOTS_DONE = 0x534E4150; // 'SNAP'

// Pack source formats PACK reports, one per handshake pair, so that every Read_32b_data / Read_int8
// outcome of the MATH arm's derivation is read back: Int8 / UInt8 take the 8-bit integer read in a 16-bit
// Dest, Int32 / UInt32 / Float32 keep the 32-bit read in both modes, Float16_b follows the mode.
constexpr DataFormat PACK_FORMATS[] = {DataFormat::Int8, DataFormat::UInt8, DataFormat::Int32, DataFormat::UInt32, DataFormat::Float32, DataFormat::Float16_b};
constexpr std::uint32_t NUM_PACK_FORMATS = sizeof(PACK_FORMATS) / sizeof(PACK_FORMATS[0]);

// PCK_DEST_RD_CTRL Read_32b_data / Read_int8 the MATH arm must have programmed for `enable` and `format`.
inline bool expected_read_32b(bool enable, DataFormat format)
{
    return enable || format == DataFormat::Int32 || format == DataFormat::UInt32 || format == DataFormat::Float32;
}

inline bool expected_read_int8(bool enable, DataFormat format)
{
    return !expected_read_32b(enable, format) && (format == DataFormat::Int8 || format == DataFormat::UInt8);
}

inline bool dest_rd_ctrl_matches(volatile std::uint32_t tt_reg_ptr* cfg, bool enable, DataFormat format)
{
    const std::uint32_t word = cfg[PCK_DEST_RD_CTRL_Read_32b_data_ADDR32];
    const bool read_32b      = (word & PCK_DEST_RD_CTRL_Read_32b_data_MASK) != 0;
    const bool read_int8     = (word & PCK_DEST_RD_CTRL_Read_int8_MASK) != 0;
    return read_32b == expected_read_32b(enable, format) && read_int8 == expected_read_int8(enable, format);
}

template <ThreadId thread_id>
inline void handshake(DataFormat pack_format)
{
    if constexpr (thread_id == ThreadId::PackThreadId)
    {
        _llk_set_fp32_dest_acc_<thread_id>(pack_format);
    }
    else
    {
        _llk_set_fp32_dest_acc_<thread_id>();
    }
}

// Runs the released thread's side of every enable/disable cycle and counts the calls after which
// both Dest-read bits already hold what MATH programmed for that mode and PACK's reported format.
template <ThreadId thread_id>
inline void released_side(std::uint32_t& enabled_seen, std::uint32_t& disabled_seen)
{
    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer();
    for (std::uint32_t i = 0; i < CYCLES; i++)
    {
        const DataFormat pack_format = PACK_FORMATS[i % NUM_PACK_FORMATS];
        handshake<thread_id>(pack_format);
        enabled_seen += dest_rd_ctrl_matches(cfg, true /*enable*/, pack_format) ? 1 : 0;
        handshake<thread_id>(pack_format);
        disabled_seen += dest_rd_ctrl_matches(cfg, false /*enable*/, pack_format) ? 1 : 0;
    }
}
} // namespace

#ifdef LLK_TRISC_UNPACK

void run_kernel(RUNTIME_PARAMETERS params)
{
    std::uint32_t enabled_seen  = 0;
    std::uint32_t disabled_seen = 0;
    released_side<ThreadId::UnpackThreadId>(enabled_seen, disabled_seen);
    mailbox_write(ThreadId::MathThreadId, SNAPSHOTS_DONE);

    auto* res = reinterpret_cast<std::uint32_t*>(params.buffer_Res[0]);
    res[3]    = enabled_seen;
    res[4]    = disabled_seen;
}

#endif

#ifdef LLK_TRISC_MATH

void run_kernel(RUNTIME_PARAMETERS)
{
    for (std::uint32_t i = 0; i < CYCLES; i++)
    {
        _llk_set_fp32_dest_acc_<ThreadId::MathThreadId>(true /*enable*/);
        _llk_set_fp32_dest_acc_<ThreadId::MathThreadId>(false /*enable*/);
    }

    // Wait until both released threads have taken their last readback, then leave the dest-acc fields as
    // the variant was built.
    const std::uint32_t unpack_done = mailbox_read(ThreadId::UnpackThreadId);
    const std::uint32_t pack_done   = mailbox_read(ThreadId::PackThreadId);
    static_cast<void>(unpack_done);
    static_cast<void>(pack_done);
    cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(is_fp32_dest_acc_en);
    cfg_reg_rmw_tensix<ALU_ACC_CTRL_SFPU_Fp32_enabled_RMW>(is_fp32_dest_acc_en);
    cfg_reg_rmw_tensix<PCK_DEST_RD_CTRL_Read_32b_data_RMW>(is_fp32_dest_acc_en);
    tensix_sync();
}

#endif

#ifdef LLK_TRISC_PACK

void run_kernel(RUNTIME_PARAMETERS params)
{
    std::uint32_t enabled_seen  = 0;
    std::uint32_t disabled_seen = 0;
    released_side<ThreadId::PackThreadId>(enabled_seen, disabled_seen);
    mailbox_write(ThreadId::MathThreadId, SNAPSHOTS_DONE);

    auto* res = reinterpret_cast<std::uint32_t*>(params.buffer_Res[0]);
    res[0]    = CYCLES;
    res[1]    = enabled_seen;
    res[2]    = disabled_seen;
}

#endif
