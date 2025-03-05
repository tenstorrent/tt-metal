// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ckernel
{

enum VectorMode
{
    None      = 0,
    R         = 1,
    C         = 2,
    RC        = 4,
    RC_custom = 6,
    Invalid   = 0xFF,
};

enum ReduceDim
{
    REDUCE_ROW,
    REDUCE_COL,
    REDUCE_SCALAR,
};

enum TileDim
{
    R_IDX = 0,
    C_IDX = 1,
};

enum PoolType
{
    SUM,
    AVG,
    MAX,
};

enum DataCopyType
{
    A2D,
    B2D,
};

enum EltwiseBinaryType
{
    ELWMUL,
    ELWDIV,
    ELWADD,
    ELWSUB,
    ELWLESS,
};

enum class EltwiseBinaryReuseDestType
{
    NONE         = 0,
    DEST_TO_SRCA = 1,
    DEST_TO_SRCB = 2,
};

enum DstSync
{
    SyncHalf   = 0,
    SyncFull   = 1,
    SyncTile16 = 2,
    SyncTile2  = 3,
};

enum BroadcastType
{
    NONE   = 0x0, // A - None || B - None
    COL    = 0x1, // A - None || B - Col Broadcast
    ROW    = 0x2, // A - None || B - Row Broadcast
    SCALAR = 0x3, // A - None || B - Scalar Broadcast
};

enum src_op_id_e
{
    OP_SRC0 = 0,
    OP_SRC1 = 1,
    OP_SRC2 = 2,
    OP_SRC3 = 3,
    OP_SRC4 = 4,
};

enum local_op_id_e
{
    OP_LOCAL0 = 0,
    OP_LOCAL1 = 1,
    OP_LOCAL2 = 2,
    OP_LOCAL3 = 3,
    OP_LOCAL4 = 4,
};

enum out_op_id_e
{
    OUT_ID0 = 0,
    OUT_ID1 = 1,
    OUT_ID2 = 2,
    OUT_ID3 = 3,
    OUT_ID4 = 4,
};

enum ReluType
{
    NO_RELU,
    ZERO_RELU,
    MIN_THRESHOLD_RELU,
    MAX_THRESHOLD_RELU,
};

constexpr bool UnpackToDestEn  = true;
constexpr bool UnpackToDestDis = false;

/*
Stochastic rounding modes:
    None: No stochastic rounding enabled, default rounding is round to nearest even.
    Fpu: Enables stochastic rounding for every accumulation in the fpu
    Pack: Enables stochastic rounding in both gasket and packer. Gasket rounding is in
    data format conversion stage from dest format to pack_src_format. Packer rounding
    is in data format conversion stage from pack_src_format to pack_dst_format.
    All: Enables fpu, pack and gasket rounding.
*/
enum struct StochRndType
{
    None = 0,
    Fpu  = 1,
    Pack = 2,
    All  = 0xf,
};

// This is populated per Blackhole ISA for SFPLOAD/SFPSTORE instructions.
enum InstrModLoadStore
{
    DEFAULT       = 0,
    FP16A         = 1,
    FP16B         = 2,
    FP32          = 3,
    INT32         = 4,
    INT8          = 5,
    LO16          = 6,
    HI16          = 7,
    INT32_2S_COMP = 12,
    INT8_2S_COMP  = 13,
    LO16_ONLY     = 14,
    HI16_ONLY     = 15
};

// This is populated per Blackhole ISA for SFPCAST instruction.
enum InstrModCast
{
    INT32_TO_FP32_NEAREST_EVEN     = 0,
    INT32_TO_FP32_STOCHASTIC       = 1,
    INT32_2S_COMP_TO_INT_SIGN_MAGN = 2,
    INT_SIGN_MAGN_TO_INT32_2S_COMP = 3
};

} // namespace ckernel
