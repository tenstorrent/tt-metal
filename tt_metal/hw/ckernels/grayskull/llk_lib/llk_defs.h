// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ckernel {
enum ReduceDim {
    REDUCE_ROW,
    REDUCE_COL,
    REDUCE_SCALAR,
};

enum Dim {
  None      = 0,
  R         = 1,
  C         = 2,
  Z         = 3,
  RC        = 4,
  ZR        = 5,
  Invalid   = 0xFF,
};

enum PoolType {
    SUM,
    AVG,
    MAX,
};

enum DataCopyType {
    A2D,
    B2D,
};

enum EltwiseBinaryType {
    ELWMUL,
    ELWDIV,
    ELWADD,
    ELWSUB,
    ELWLESS,
};

enum class EltwiseBinaryReuseDestType {
    NONE = 0,
    DEST_TO_SRCA = 1,
    DEST_TO_SRCB = 2,
};

enum DstSync {
    SyncHalf = 0,
    SyncFull = 1,
    SyncTile16 = 2,
    SyncTile2 = 3,
};

enum BroadcastType {
    NONE = 0x0,    // A - None || B - None
    COL = 0x1,     // A - None || B - Col Broadcast
    ROW = 0x2,     // A - None || B - Row Broadcast
    SCALAR = 0x3,  // A - None || B - Scalar Broadcast
};

enum src_op_id_e {
    OP_SRC0 = 0,
    OP_SRC1 = 1,
    OP_SRC2 = 2,
    OP_SRC3 = 3,
    OP_SRC4 = 4,
};

enum local_op_id_e {
    OP_LOCAL0 = 0,
    OP_LOCAL1 = 1,
    OP_LOCAL2 = 2,
    OP_LOCAL3 = 3,
    OP_LOCAL4 = 4,
};

enum out_op_id_e {
    OUT_ID0 = 0,
    OUT_ID1 = 1,
    OUT_ID2 = 2,
    OUT_ID3 = 3,
    OUT_ID4 = 4,
};

enum ReluType {
    NO_RELU,
    ZERO_RELU,
    MIN_THRESHOLD_RELU,
    MAX_THRESHOLD_RELU,
};

/* Only used for WHB0*/
enum struct StochRndType {
    None    = 0,
    Fpu     = 1,
    Pack    = 2,
    All     = 0xf,
};

template <bool headerless = false>
constexpr static std::int32_t MUL_TILE_SIZE_AND_INDEX(uint format, uint index) {
    switch (format&0x1F) {
        case ((uint8_t)DataFormat::Float32): return ((index<<8)+(!headerless)*(index<<1));
        case ((uint8_t)DataFormat::Float16):
        case ((uint8_t)DataFormat::Float16_b): return ((index<<7)+(!headerless)*(index<<1));
        case ((uint8_t)DataFormat::Bfp8):
        case ((uint8_t)DataFormat::Bfp8_b): return ((index<<6)+(index<<2)+(!headerless)*(index<<1));
        case ((uint8_t)DataFormat::Bfp4):
        case ((uint8_t)DataFormat::Bfp4_b): return ((index<<5)+(index<<2)+(!headerless)*(index<<1));
        case ((uint8_t)DataFormat::Bfp2):
        case ((uint8_t)DataFormat::Bfp2_b): return ((index<<4)+(index<<2)+(!headerless)*(index<<1));
        //Keep default as Lf8?
        default: return ((index<<6)+(!headerless)*(index<<1));
    };
}

template <bool headerless = false>
constexpr static std::int32_t GET_L1_TILE_SIZE(uint format) {
    switch (format&0x1F) {
        case ((uint8_t)DataFormat::Float32): return ((4096>>4)+(!headerless)*(32>>4));
        case ((uint8_t)DataFormat::Float16):
        case ((uint8_t)DataFormat::Float16_b): return ((2048>>4)+(!headerless)*(32>>4));
        case ((uint8_t)DataFormat::Bfp8):
        case ((uint8_t)DataFormat::Bfp8_b): return ((1024>>4)+(64>>4)+(!headerless)*(32>>4));
        case ((uint8_t)DataFormat::Bfp4):
        case ((uint8_t)DataFormat::Bfp4_b): return ((512>>4)+(64>>4)+(!headerless)*(32>>4));
        case ((uint8_t)DataFormat::Bfp2):
        case ((uint8_t)DataFormat::Bfp2_b): return ((256>>4)+(64>>4)+(!headerless)*(32>>4));
        default: return ((1024>>4)+(!headerless)*(32>>4));
    };
}


enum SfpiTestType {
    test1,
    test2,
    test3,
    test4,
    test5,
    test6,
    test7,
    test8,
    test9,
    test10,
    test11,
    test12,
    test13,
    test14,
    unused_test,
};
}  // namespace ckernel
