// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <vector>

#include <tt-logger/tt-logger.hpp>
#include "tt_fabric_telemetry.hpp"
#include "tt_fabric_test_eth_readback.hpp"

// Manages fabric code profiling lifecycle (readback, clearing, reporting).
class CodeProfiler {
public:
    explicit CodeProfiler(EthCoreBufferReadback& eth_readback);

    void set_enabled(bool enabled);
    bool is_enabled() const;

    void clear_code_profiling_buffers();
    void read_code_profiling_results();
    void report_code_profiling_results() const;
    void reset();  // Clears entries and device buffers when profiling is enabled

    const std::vector<CodeProfilingEntry>& get_entries() const;

private:
    EthCoreBufferReadback& eth_readback_;
    std::vector<CodeProfilingEntry> entries_;
    bool enabled_ = false;
};
