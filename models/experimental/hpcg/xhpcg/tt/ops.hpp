// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The solver's custom ops (tt/custom_ops/sparse_rows/op.py's SparseRowOp, tt/custom_ops/cg_vector/op.py's
// CgVectorOp and DotFinishOp): each builds its program description once; calling it launches it (trace-capturable).
// The kernels are the Python package's, unchanged.
#pragma once
#include <optional>
#include <string>
#include <vector>

#include "device.hpp"
#include "plan.hpp"

namespace hpcg_tt {

struct Op {
    tt::tt_metal::ProgramDescriptor prog;
    std::vector<Tensor> io;
    std::vector<Tensor> keep;  // tensors the launch reads that io does not list
    void operator()() const;
};

struct SparseRowArgs {
    bool gs = true;  // mode gs (Gauss-Seidel phases, SpMV passes 8 + b) or spmv
    std::vector<int64_t> phases;
    std::optional<Tensor> b, out2;
    bool zerox = false, addb = false, restf = false, wsplit = false;
    int64_t tpw = TILE;
    std::optional<std::vector<int>> noc_flip;
    std::optional<std::vector<int64_t>> table_phases;
};
Op sparse_row_op(
    const Dev& D,
    Plan& plan,
    const Tensor& values,
    const Tensor& offsets,
    const Tensor& x,
    const Tensor& out,
    const SparseRowArgs& a,
    const Tensor& dbg);

enum class VecMode { dot = 0, axpy = 1, xr = 2, zero = 3 };
// dot: (a, b, part). axpy: b = a + (s1 / s2) b. xr: x += (s1 / s2) p, r -= (s1 / s2) Ap, partials of r.r.
// zero: a = 0, b = 0 (zeros written, nothing read)
struct VecOp : Op {
    int nc = 0;
};
VecOp cg_vector_op(
    const Dev& D,
    VecMode mode,
    const Tensor& a,
    const Tensor& b,
    const Tensor* c,
    const Tensor* d,
    const Tensor* s1,
    const Tensor* s2,
    const Tensor* part,
    int64_t P);
// adds the per-core slots in the first nparts pages of part into word 0 of out (keep_old: out's page copied to old
// first; sqrt: the square root)
Op dot_finish_op(
    const Dev& D, const Tensor& part, int nparts, const Tensor& out, const Tensor* old, bool keep_old, bool sqrt);

}  // namespace hpcg_tt
