// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Matrix generation on the card (tt/custom_ops/stencil_matrix/op.py): each core writes its rows' matrix values and
// window offsets (and b and zero vectors) into DRAM, one launch for a list of jobs.
#pragma once
#include <map>
#include <vector>

#include "ops.hpp"

namespace hpcg_tt {

struct Job {
    int kind = 0;  // 0 a level's 27-slot tables, 1 restriction, 2 prolongation
    Plan* plan = nullptr;
    int64_t ntlv = 0;
    Tensor values, offsets;
    std::optional<Tensor> b;
    std::vector<Tensor> zeros;
    std::vector<const Level*> levels;
    int hw = 0;                                // header words of the job's META segment
    std::map<int, std::vector<int64_t>> head;  // header words by offset
    std::vector<Tensor> tensors() const;
};

// a level's values / offsets tensors (they depend on its size only)
std::pair<Tensor, Tensor> allocate_tables(const Dev& D, const Level& L);
Job stencil_job(
    const Dev& D,
    const Level& L,
    Plan& plan,
    const std::optional<Tensor>& b,
    const std::vector<Tensor>& zeros,
    const std::optional<std::pair<Tensor, Tensor>>& tables);
Job transfer_job(const Dev& D, Plan& plan);

struct GenOp {
    Tensor meta;
    std::vector<Tensor> io;
    tt::tt_metal::ProgramDescriptor prog;
    // what program() needs to rebuild the descriptor with a new launch id
    CoreRangeSet cores;
    std::vector<uint32_t> ct, args;
    tt::tt_metal::KernelDescriptor::Defines defs;
    tt::tt_metal::ProgramDescriptor::CBDescriptors cbs;
    bool called = false;
    void build(const Dev& D, uint32_t launch);
    void operator()(const Dev& D);
};
// one launch running every job; ncores: cores of the launch (default the most any job's plan uses); job_slots:
// argument slots for this many jobs (default the job count)
GenOp gen_op(const Dev& D, const std::vector<Job>& jobs, int ncores = 0, int job_slots = -1);
// a launch with no jobs, shaped like a one-job launch on ncores cores: loads the kernel binaries and builds that
// program, so the fine level's launch reuses it
void warm_up(const Dev& D, int ncores);

}  // namespace hpcg_tt
