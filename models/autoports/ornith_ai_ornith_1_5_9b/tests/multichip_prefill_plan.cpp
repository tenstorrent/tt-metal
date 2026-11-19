// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Pure host planner enumeration. No device is opened or queried.
#include <iostream>
#include "ttnn/cpp/ttnn/operations/experimental/ccl/matmul_reduce_scatter_prefill/device/matmul_reduce_scatter_prefill_planner.hpp"
namespace mrs = ttnn::operations::experimental::ccl::matmul_reduce_scatter_prefill::detail;
int main() {
    std::cout << "budget_kib,M,K,resident,m_on_x,grid_m,grid_n,w_read_noc,kb,footprint,chunk_m,chunk_n\n";
    for (unsigned budget : {64, 96, 128, 192, 256, 384, 512, 768, 1024, 1280}) {
        unsigned bad = 0, missing = 0;
        for (unsigned mt = 1; mt <= 32; ++mt) {
            for (unsigned kt : {32, 96}) {
                mrs::PlanInputs in{
                    .G = 4,
                    .kind = mrs::ScheduleKind::Line,
                    .L = 1,
                    .Mt = mt,
                    .Kt = kt,
                    .Nt = 128,
                    .block_m = mt,
                    .block_n = 32,
                    .gx = 11,
                    .gy = 10,
                    .tA = 2048,
                    .tW = 1088,
                    .scatter_dim = -1,
                    .budget = budget * 1024};
                try {
                    const auto& p = mrs::solve_plan(in);
                    const bool mx = p.grid.orient == mrs::Orient::MOnX;
                    const bool thin = p.resident && !mx && p.grid.grid_m <= 4 && p.grid.grid_n >= 10;
                    const unsigned noc = p.resident && !thin;
                    bad += noc;
                    std::cout << budget << ',' << mt * 32 << ',' << kt * 32 << ',' << p.resident << ',' << mx << ','
                              << p.grid.grid_m << ',' << p.grid.grid_n << ',' << noc << ',' << p.kb << ','
                              << p.footprint << ',' << p.chunk_m << ',' << p.chunk_n << '\n';
                } catch (const std::exception& e) {
                    ++missing;
                }
            }
        }
        std::cerr << "BUDGET " << budget << " NOC1_PLANS " << bad << " MISSING_PLANS " << missing << '\n';
    }
}
