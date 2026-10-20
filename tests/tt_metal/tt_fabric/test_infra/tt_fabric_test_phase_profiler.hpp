// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

#include "tt_fabric_test_eth_readback.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/telemetry/fabric_phase_profile_types.hpp"

// Host side of the experimental router phase profiler (TT_FABRIC_PHASE_PROFILE, TT_FABRIC_TXQ_COUNTERS, see
// fabric_phase_profile_types.hpp). The router keeps running totals, so a test is measured as the difference
// between a snapshot taken before launch and one taken after the programs finish.
//
// Results are logged per (device, eth channel, ERISC, plus the router's DRISC with EXP_DRISC_OFFLOAD) and, when
// TT_FABRIC_PHASE_PROFILE_CSV names a file, appended to it as CSV (column erisc: 0, 1, or 2 for the DRISC).
class PhaseProfiler {
public:
    explicit PhaseProfiler(EthCoreBufferReadback& eth_readback);

    static bool is_enabled();

    void begin();
    void end_and_report(const std::string& test_name);

private:
    void report_microbench();

    EthCoreBufferReadback& eth_readback_;
    std::vector<EthCoreBufferResult> before_;
    inline static bool microbench_reported_ = false;
};
