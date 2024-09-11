// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <cstdint>
#include <limits>
#include "test_tiles.hpp"
#include "bfloat16.hpp"

using std::vector; // TODO(AP)
using std::uint32_t;
using std::uint16_t;

//////////////////////////////////////////////////////////////////////////////////////////
// Reference CPU implementation of reduce_H
//////////////////////////////////////////////////////////////////////////////////////////

// input shape.x is assumed to have the full number of elements in bfloat16
// src_vec is expected to be untilized
// result is also untilized
// TODO(AP) - move to gold header
inline std::vector<uint16_t> gold_transpose_hc(std::vector<uint16_t> src_vec, std::vector<uint32_t> shape) {
    std::vector<uint32_t> shapeT{shape[0], shape[2], shape[1], shape[3]};
    TensAddr addr(shape);
    TensAddr addrt(shapeT);

    std::vector<uint16_t> transposed(src_vec.size());
    for (int n = 0; n < shape[0]; n++)
    for (int c = 0; c < shape[1]; c++)
    for (int h = 0; h < shape[2]; h++)
    for (int w = 0; w < shape[3]; w++) {
        auto toffs = addrt.offs(n, h, c, w);
        auto offs = addr.offs(n, c, h, w);
        TT_FATAL(toffs < transposed.size() && offs < src_vec.size(), "Error");
        transposed[toffs] = src_vec[offs];
    }
    //log_info(tt::LogVerif, "Prior size = {}", transposed.size());
    return transposed;
};

struct BcastDim {
    enum Enum : uint32_t {
        W = 2, // broadcast an H-tensor over destination's W
        H = 1, // broadcast a W-tensor over destination's H
        HW = 4, // broadcast a 1-element tensor over destination's HW
    };
    // TODO(AP): fix the gap to match defines in llk_3c.h

    static const std::vector<Enum> all() { return { W, H, HW }; }
};

struct BcastOp {
    enum Enum : uint32_t {
        ADD = 0,
        SUB = 1,
        MUL = 2,
    };
    // These constants above map to ops in llk_3c.h:
    // add_tiles_bcast, sub_tiles_bcast, mul_tiles_bcast

    static const vector<Enum> all() { return { ADD, SUB, MUL }; }
};


// input shape.x is assumed to have the full number of elements in bfloat16
// src_vec is expected to be untilized
// result is also untilized
// bcast_vals for hw mode is expected to have size 1
// bcast_vals for h or w mode is supposed to have h or w elements
inline std::vector<uint16_t> gold_bcast_op(
    const std::vector<uint16_t>& src_vec,
    const std::vector<uint32_t>& shape,
    const std::vector<uint16_t>& bcast_vals,
    BcastDim::Enum bcast_dim,
    BcastOp::Enum bcast_op
) {
    uint32_t N = shape[0], C = shape[1], H = shape[2], W = shape[3];
    TT_FATAL(bcast_dim == BcastDim::W ? bcast_vals.size() == N*C*H : true, "Error");
    TT_FATAL(bcast_dim == BcastDim::H ? bcast_vals.size() == N*C*W : true, "Error");
    TT_FATAL(bcast_dim == BcastDim::HW ? bcast_vals.size() == N*C : true, "Error");

    std::vector<uint32_t> shape_dst{N, C, H, W};
    TensAddr addr(shape);
    std::vector<uint16_t> result(addr.numel());
    std::fill(result.begin(), result.end(), 0);
    for (int n = 0; n < N; n++)
    for (int c = 0; c < C; c++)
    for (int h = 0; h < H; h++)
    for (int w = 0; w < W; w++) {
        auto offs = addr.offs(n, c, h, w);
        int b_index = 0;
        switch (bcast_dim) {
            case BcastDim::H:  b_index = w + c*W + n*C*W; break; // bcast tensor is ncw
            case BcastDim::W:  b_index = h + c*H + n*C*H; break; // bcast tensor is nch
            case BcastDim::HW: b_index = c + n*C; break; // bcast tensor is nc
            default:
            TT_FATAL(false && "Unexpected broadcast mode in gold_bcast_op", "Error");
        }
        float bval = bfloat16(bcast_vals[b_index]).to_float();
        float result1 = 0.0f;
        switch (bcast_op) {
            case BcastOp::ADD: result1 = bfloat16(src_vec[offs]).to_float() + bval; break;
            case BcastOp::SUB: result1 = bfloat16(src_vec[offs]).to_float() - bval; break;
            case BcastOp::MUL: result1 = bfloat16(src_vec[offs]).to_float() * bval; break;
            default:
                TT_FATAL(false && "Unexpected bcast_op", "Error");
        }
        result[offs] = bfloat16(result1).to_uint16();
    }

    return result;
}


// Basic gold batch matmul implementation.
// Returns C=A*B, A and B are row-major untilized
// Accumulates in FP32
inline std::vector<uint16_t> gold_bmm(
    const std::vector<uint32_t> shapeA,
    const std::vector<uint16_t>& A,
    const std::vector<uint32_t>& shapeB,
    const vector<uint16_t>& B,
    bool acc16 = false
    )
{
    TT_FATAL(shapeB[0] == 1 && shapeA[0] == 1, "Error");
    uint32_t nb = shapeA[1]; TT_FATAL(shapeB[1] == nb, "Error");
    uint32_t M = shapeA[2];
    uint32_t K = shapeA[3]; TT_FATAL(shapeB[2] == K, "Error");
    uint32_t N = shapeB[3];

    vector<uint32_t> shapeC{1, nb, M, N};
    TensAddr addrC(shapeC);
    TensAddr addrA(shapeA);
    TensAddr addrB(shapeB);
    vector<uint16_t> result(addrC.numel());
    vector<float> resultf(addrC.numel());
    std::fill(resultf.begin(), resultf.end(), 0);

    for (int ib = 0; ib < nb; ib++)
    for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++)
    for (int k = 0; k < K; k++) {
        auto offsA = addrA.offs(0, ib, m, k);
        auto offsB = addrB.offs(0, ib, k, n);
        auto offsC = addrC.offs(0, ib, m, n);

        float aa = bfloat16(A[offsA]).to_float();
        float bb = bfloat16(B[offsB]).to_float();
        resultf[offsC] += aa * bb;
        if (acc16)
            resultf[offsC] = bfloat16(resultf[offsC]).to_float();
    }

    // write back to fp16 after we accumulated in fp32
    for (int ib = 0; ib < nb; ib++)
    for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++) {
        auto offsC = addrC.offs(0, ib, m, n);
        result[offsC] = bfloat16(resultf[offsC]).to_uint16();
    }

    return result;
}


typedef BcastOp EltwiseOp;
