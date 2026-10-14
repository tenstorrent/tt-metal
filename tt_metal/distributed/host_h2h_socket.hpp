// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// The host-to-host hop. Forwards each [payload|trailer] frame verbatim into the peer's RX
// arena; the trailer doubles as the arrival flag, so no separate notice is sent.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "tt_metal/distributed/host_tasks.hpp"
#include "hostdevcommon/uva.h"

namespace tt::tt_metal::experimental {

class H2HSocket {
public:
    struct Config {
        HostTopology topo{};
        uint32_t chip = 0;
        uint32_t cores = 0;
        uint32_t page_bytes = 0;
        uint32_t ring_pages = 1;
        // Where the aliased H2D ring starts inside each RX arena -- H2DLeg::data_offset().
        // One value for every core AND for the peer: both hosts run an identical layout.
        uint32_t rx_data_offset = 0;
        uint8_t* region_base = nullptr;
        uint64_t region_bytes = 0;
        // 0 => cores * ring_pages. Caps frames in flight across the whole socket.
        uint32_t send_window = 0;
        // Stamps every put and closes it on the peer's credit. Off by default: the samples
        // are only wanted by a benchmark, and the vector grows one entry per frame.
        bool collect_timing = false;
    };

    // Collective: creates the window, so every rank must call it at the same point.
    static std::unique_ptr<H2HSocket> create(const Config& cfg, std::string& err);
    ~H2HSocket();

    H2HSocket(const H2HSocket&) = delete;
    H2HSocket& operator=(const H2HSocket&) = delete;

    // Queue a frame the D2H leg produced. False means the window is full; the caller
    // retries, which is what propagates backpressure to the device.
    bool submit(const SendTask& task);

    // Frees the D2H page behind a frame whose put has retired locally.
    using Retire = std::function<void(uint32_t core, uint32_t pages)>;
    // Hands an arrived frame to the H2D leg. False means it could not be taken this lap.
    using Deliver = std::function<bool(const DeliverTask&)>;

    // One non-blocking pass: start queued sends, retire completed ones, harvest arrivals.
    uint32_t poll(const Retire& retire, const Deliver& deliver);

    // Called by the H2D leg once the device has consumed a delivered frame. Posts the
    // credit that lets the origin reuse its slot.
    void consumed(uint32_t core, uint32_t pages);

    // Puts what consumed() noted: one op per array per peer, spanning every core touched.
    // Call after a drain loop, not per core, or the coalescing is lost. False means failed.
    bool publish_credits();

    // Frames of this core's the peers have consumed. One peer per core today, so the
    // sum is that peer's count.
    uint64_t credit_total(uint32_t core) const;

    // One entry per credited frame, this host's clock. The completion end is stamped once
    // per poll pass, so a sample carries that pass period: ~95 us at 8 cores, ~291 at 64.
    const std::vector<uint64_t>& put_to_credit_ns() const;

    // Why a pass costs what it does. The per-pass overhead is fixed, so it lands on however
    // many frames that pass posted: posts/passes sets throughput, not the window depth.
    struct PassStats {
        uint64_t passes = 0;          // poll() calls that reached the flush
        uint64_t starved = 0;         // of those, the ones that posted nothing
        // Why a pass posted nothing. These want opposite policies: no supply means batch
        // harder, no room means flush sooner, since our flush is what frees the peer's ring.
        uint64_t starved_credit = 0;  // a destination ring was full
        uint64_t starved_window = 0;  // in_flight hit window_cap: flush sooner, do not batch
        uint64_t starved_empty = 0;   // nothing was queued to send
        uint64_t posts = 0;           // FRAMES posted, not puts: coalescing adds `run` at once
        uint64_t payload_puts = 0;    // the operations those frames cost, one per run
        uint64_t trailer_puts = 0;    // the second phase, also one per run
        uint64_t credit_puts = 0;     // coalesced credit puts, NOT frames credited
        uint64_t done_puts = 0;       // the same for the done array; the two coalesce apart
        // A flush costs the same whatever it covers, so the bytes it covered are what say
        // whether it was worth issuing. pending_max bounds any batching threshold we pick.
        uint64_t flushes = 0;
        uint64_t flushes_tiny = 0;    // covered less than one page: paid in full for nothing
        uint64_t flushes_held = 0;    // withheld: below the watermark and more was coming
        uint64_t pending_sum = 0;     // bytes covered, summed over every flush
        uint64_t pending_max = 0;     // most bytes a single flush ever covered
        uint64_t flush_ns = 0;        // time inside flush_dirty() ONLY; 0 unless collect_timing
        // Occupancy, time-integrated: in_flight sampled once per pass. A level that is only
        // incremented and decremented cannot yield a mean, and L = lambda*W needs one.
        uint64_t in_flight_sum = 0;
        // How long the oldest bytes in a flush had waited. This is the batching window, and
        // it is what separates the cost of moving a page from its dwell in the pipeline.
        uint64_t accum_ns_sum = 0;
        uint64_t accum_flushes = 0;  // payload-bearing flushes, so the sum above has a mean
    };
    const PassStats& pass_stats() const;

    // Per-sample series for the ratios that were means only. A mean and a median over ONE
    // of these is a matched pair; a sum divided by a count admits no median at all.
    struct Series {
        std::vector<uint32_t> in_flight;      // frames outstanding, one per poll pass
        std::vector<uint64_t> accum_ns;       // batching window, one per payload flush
        std::vector<uint64_t> covered_bytes;  // bytes a flush covered, one per flush
        std::vector<double> us_per_frame;     // flush-cycle service time, one per cycle
    };
    const Series& series() const;
    // Zeroed at the warmup boundary: a count spanning the ramp cannot be divided by a
    // steady-state duration, and every amortized figure is exactly that division.
    void reset_stats();

    std::string barrier();

    // LOCAL ONLY: a per-frame failure drops the frame with nothing telling the peer, so poll
    // these before any device wait -- tt_uva_sync() parks on a count that will never advance.
    bool failed() const;
    std::string first_error() const;
    std::string describe() const;

private:
    H2HSocket();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace tt::tt_metal::experimental
