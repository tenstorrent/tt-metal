// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// D2H2H2DSocket's REPORTING SURFACE. Nothing here moves a byte: these three run on the
// reporting path -- once every five seconds while a caller waits, once at collect() -- and
// exist only to print. They were separated from d2h2h2d_socket.cpp so that file holds the
// send, receive and credit paths and nothing else.
//
// The counters they read are still written on the hot path, in the socket proper. Moving the
// readers does not move that cost; removing it would mean deleting the counters, which is a
// behaviour change and not this.

#include <tt-metalium/experimental/sockets/d2h2h2d_socket.hpp>

#include <sstream>
#include <string>

namespace tt::tt_metal::experimental {

#if defined(TT_METAL_HOST_BRIDGE)

void D2H2H2DSocket::append_transport_stats(RunStats& s) const {
    // The sender thread's samples ARE the host-to-host stages; without this they are simply
    // absent from the table.
    s.per_worker.push_back(sender_stats_);

    s.window = send_window_;
    s.sender_shape = send_blocking_ ? "blocking" : "windowed";

    // Two counters add, so folding peers needs no weighted mean and no special first case.
    // d.sum is now a MEASURED total rather than mean*n back-computed from it, so latency_us
    // for this hop is checkable against the count the same way every other hop's is.
    RetireStats rs{};
    for (Transport* t : peers_.all()) {
        const RetireStats one = t->retire_stats();
        rs.n += one.n;
        rs.sum_ns += one.sum_ns;
    }
    if (rs.n > 0) {
        WorkerStats w{};
        Dist& d = w.hop[kHopH2HRetire];
        d.n = rs.n;
        d.sum = rs.sum_ns;
        s.per_worker.push_back(w);
    }
}

void D2H2H2DSocket::dump_transport(std::string& into) const {
    std::ostringstream m;
    for (Transport* tp_i : peers_.all()) {
        const TransportDiag d = tp_i->diag();
        m << "    transport[host " << tp_i->peer().host_id << "]: posted=" << d.posted
          << " retired=" << d.retired << " injected=" << d.injected
          << " outstanding=" << d.outstanding
          << " unmatched=" << d.unmatched << " abandoned=" << d.abandoned;
        if (d.outstanding != 0) {
            m << " oldest_tag=" << d.oldest_tag;
        }
        m << "\n";
        if (!d.last_error.empty()) {
            m << "    transport[host " << tp_i->peer().host_id << "] last CQ error: " << d.last_error << "\n";
        }
    }
    into = m.str();
}

RunStats D2H2H2DSocket::collect() const {
    RunStats s = scanner_ ? scanner_->collect() : RunStats{};
    const uint64_t t0 = timed_start_ns_.load(std::memory_order_relaxed);
    const uint64_t t1 = timed_end_ns_.load(std::memory_order_relaxed);
    s.timed_ns = (t0 > 0 && t1 > t0) ? (t1 - t0) : 0;
    append_transport_stats(s);
    return s;
}

std::string D2H2H2DSocket::stall_dump(const char* where) const {
    std::ostringstream m;
    m << "\n  [STALL @ " << where << "]\n";
    m << "    tx_done=" << counters_.tx_done.load() << " delivered=" << counters_.delivered.load()
      << " routed_remote=" << counters_.routed_remote.load()
      << " errors=" << counters_.errors.load() << "\n";
    m << "    sender: " << sender_state_.load() << "  (0=idle 1=credit-wait 2=payload 3=notice)\n";
    m << "    credit_flushes=" << credit_flushes_.load(std::memory_order_relaxed) << "\n";
    m << "    rx_gate: deliveries=" << rx_deliveries_.load(std::memory_order_relaxed)
      << " with_remote_notice=" << rx_remote_notice_.load(std::memory_order_relaxed)
      << " credits_entered=" << rx_credits_posted_.load(std::memory_order_relaxed)
      << " credits_returned=" << rx_credits_done_.load(std::memory_order_relaxed)
      << " origin_host=" << static_cast<int64_t>(rx_origin_host_.load(std::memory_order_relaxed))
      << std::hex << " origin_sel=0x" << rx_origin_sel_.load(std::memory_order_relaxed)
      << " first_ctrl=0x" << rx_first_ctrl_.load(std::memory_order_relaxed) << std::dec << "\n";
    std::string t;
    dump_transport(t);
    m << t;
    m << "    scan_rx=yes  rejects:";
    for (uint32_t i = 0; i < 8; ++i) {
        const uint64_t n = counters_.rejects[i].load();
        if (n) {
            m << " " << ctrl_verdict_name(i) << "=" << n;
        }
    }
    m << "\n";
    const uint32_t show = cfg_.cores < 8 ? cfg_.cores : 8;
    for (uint32_t c = 0; c < show; ++c) {
        m << "    core " << c << ": notice_sent=" << notice_sent_[c].load()
          << " credit_in=" << credit_total(region_, c) << " tx_retired=" << tx_retired_[c].load()
          << " delivered=" << delivered_per_core_[c].load() << " ctrl_tx=0x" << std::hex
          << load_acquire(region_.ctrl_tx(c)) << " ctrl_rx=0x" << load_acquire(region_.ctrl_rx(c))
          << std::dec << "\n";
    }
    const std::string fe = first_error();
    if (!fe.empty()) {
        m << "    first error: " << fe << "\n";
    }
    return m.str();
}


#endif  // TT_METAL_HOST_BRIDGE

}  // namespace tt::tt_metal::experimental
