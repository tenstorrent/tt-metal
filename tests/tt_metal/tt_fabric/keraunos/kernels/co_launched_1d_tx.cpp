// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The stock tt_fabric_1d_tx on a hart beside its CCE's fabric router (co_launched_worker.hpp).

#define kernel_main stock_1d_tx_main
#include "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_1d_tx.cpp"
#undef kernel_main

#include "tests/tt_metal/tt_fabric/keraunos/kernels/co_launched_worker.hpp"

void kernel_main() { run_co_launched_worker(stock_1d_tx_main, test_results, test_results_size_bytes); }
