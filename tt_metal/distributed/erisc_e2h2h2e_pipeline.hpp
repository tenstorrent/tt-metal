// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// E2H -> H2H -> H2E. The sending rank owns E2H, the receiving rank H2E; both own their H2H half.
// No hop blocks: each moves what it can, so a stalled leg cannot wedge the others.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include "tt_metal/distributed/erisc_e2h_leg.hpp"
#include "tt_metal/distributed/erisc_h2e_leg.hpp"
#include "tt_metal/distributed/erisc_h2h_socket.hpp"

namespace tt::tt_metal::experimental {

class E2H2H2EPipeline {
public:
    using Inspect = std::function<void(const BridgeDeliverTask&)>;
    struct Counters {
        uint64_t forwarded = 0;     // E2H frames queued to H2H
        uint64_t injected = 0;      // H2H frames published into the far router
        uint64_t landed = 0;        // of those, frames the far router has taken
        uint64_t out_of_order = 0;  // E2H frames that were not the next expected
    };

    // H2H always; E2H on the sending rank and H2E on the receiving rank, either may be null.
    E2H2H2EPipeline(
        std::unique_ptr<EriscH2HSocket> h2h,
        std::unique_ptr<E2HLeg> e2h,
        std::unique_ptr<H2ELeg> h2e,
        uint32_t max_per_poll);
    ~E2H2H2EPipeline();
    E2H2H2EPipeline(const E2H2H2EPipeline&) = delete;
    E2H2H2EPipeline& operator=(const E2H2H2EPipeline&) = delete;

    // One turn of every hop this rank owns; returns frames moved. `inspect` sees each frame H2E took.
    uint32_t poll(const Inspect& inspect = {});

    EriscH2HSocket& h2h() const { return *h2h_; }
    const Counters& counters() const { return counters_; }
    std::string first_error() const;

private:
    std::unique_ptr<EriscH2HSocket> h2h_;
    std::unique_ptr<E2HLeg> e2h_;
    std::unique_ptr<H2ELeg> h2e_;
    uint32_t max_per_poll_ = 0;
    Counters counters_{};
};

}  // namespace tt::tt_metal::experimental
