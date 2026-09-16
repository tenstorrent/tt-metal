// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/sockets/internal/host_stats.hpp>

#include <cmath>

#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <map>
#include <sstream>
#include <utility>
#include <vector>

#include <tt-metalium/experimental/sockets/internal/host_uva_layout.hpp>

namespace tt::tt_metal::experimental {

// CLOCK_REALTIME here, not the CLOCK_MONOTONIC_RAW used for every measurement. These two are
// labels, not durations: they exist so a row can be tied to a wall-clock moment and to the
// other rows from the same process, and a monotonic count is meaningless across reboots.
std::string utc_now_iso() {
    timespec ts{};
    clock_gettime(CLOCK_REALTIME, &ts);
    const time_t secs = static_cast<time_t>(ts.tv_sec);
    std::tm tm{};
    gmtime_r(&secs, &tm);
    char buf[40];
    std::snprintf(buf, sizeof(buf), "%04d-%02d-%02dT%02d:%02d:%02d.%03ldZ", tm.tm_year + 1900, tm.tm_mon + 1,
                  tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec, static_cast<long>(ts.tv_nsec / 1000000));
    return buf;
}

// timestamp (ms) plus pid
std::string make_run_id() {
    timespec ts{};
    clock_gettime(CLOCK_REALTIME, &ts);
    const uint64_t ms = static_cast<uint64_t>(ts.tv_sec) * 1000ull + static_cast<uint64_t>(ts.tv_nsec) / 1000000ull;
    char buf[48];
    std::snprintf(buf, sizeof(buf), "%011llx-%05x", static_cast<unsigned long long>(ms),
                  static_cast<unsigned>(getpid() & 0xfffff));
    return buf;
}

// `want_header` is passed in rather than assumed, so the stripped writer validates against
// its header instead of the wide one -- otherwise every append to a basic file would report a
// schema mismatch against columns it never claimed to have.
std::string csv_schema_error(const std::string& path, const std::string& want_header) {
    std::ifstream in(path);
    if (!in.good()) {
        return {};  // no file: the caller writes the header
    }
    std::string first;
    if (!std::getline(in, first) || first.empty()) {
        return {};  // empty file: same
    }
    std::string want = want_header;
    while (!want.empty() && (want.back() == '\n' || want.back() == '\r')) {
        want.pop_back();
    }
    while (!first.empty() && (first.back() == '\n' || first.back() == '\r')) {
        first.pop_back();
    }
    if (first == want) {
        return {};
    }
    return "csv schema mismatch: " + path +
           " was written by a build with different columns. Appending would produce a ragged file "
           "that readers accept and misread.";
}

uint64_t measure_clock_overhead_ns() {
    // Minimum of many back-to-back reads.
    constexpr int kTrials = 2000;
    uint64_t best = UINT64_MAX;
    for (int i = 0; i < kTrials; ++i) {
        const uint64_t a = now_ns();
        const uint64_t b = now_ns();
        if (b > a && (b - a) < best) {
            best = b - a;
        }
    }
    return best == UINT64_MAX ? 0 : best;
}

namespace {

std::string auto_unit(double v_ns) {
    char buf[64];
    if (v_ns >= 1000000.0) {
        std::snprintf(buf, sizeof(buf), "%.3f ms", v_ns / 1e6);
    } else if (v_ns >= 1000.0) {
        std::snprintf(buf, sizeof(buf), "%.3f us", v_ns / 1e3);
    } else {
        std::snprintf(buf, sizeof(buf), "%.0f ns", v_ns);
    }
    return buf;
}

}  // namespace

std::string format_table(const RunStats& s) {
    std::ostringstream o;
    char line[640];

    o << "\n=== stage latency (" << s.mode << ", " << s.payload_bytes << " B payload, " << s.provider
      << ") ===\n\n";
    o << "  stage                     count       mean\n";
    o << "  ----------------------- ------- ----------\n";

    auto emit_rows = [&](uint32_t from, uint32_t to, bool show_empty) {
        for (uint32_t h = from; h < to; ++h) {
            const Dist d = s.merged(h);
            if (d.n == 0) {
                if (show_empty) {
                    std::snprintf(line, sizeof(line), "  %-23s %7s %10s\n", hop_name(h), "-", "-");
                    o << line;
                }
                continue;
            }
            std::snprintf(line, sizeof(line), "  %-23s %7" PRIu64 " %10s\n", hop_name(h), d.n,
                          auto_unit(d.mean_ns()).c_str());
            o << line;
        }
    };

    emit_rows(0, kHopStageCount, true);

    {
        bool any = false;
        for (uint32_t h = kHopStageCount; h < kHopCount; ++h) {
            if (s.merged(h).n != 0) {
                any = true;
                break;
            }
        }
        if (any) {
            o << "\n  diagnostics (not stages -- they explain a stage that looks wrong)\n";
            emit_rows(kHopStageCount, kHopCount, false);
        }
    }

    o << "\n  clock domains crossed by the rows marked above:\n";
    if (s.device_clock_valid) {
        std::snprintf(line, sizeof(line), "    dev  : device<->host calibrated, +/- %" PRIu64 " ns\n",
                      s.device_clock_uncertainty_ns);
    } else {
        std::snprintf(line, sizeof(line), "    dev  : NOT CALIBRATED -- device-crossing rows are not trustworthy\n");
    }
    o << line;
    if (s.host_clock_valid) {
        std::snprintf(line, sizeof(line), "    host : host<->host calibrated, +/- %" PRIu64 " ns\n",
                      s.host_clock_uncertainty_ns);
    } else {
        std::snprintf(line, sizeof(line), "    host : same host (offset exactly 0) or no peer\n");
    }
    o << line;
    std::snprintf(line, sizeof(line), "    clock read overhead %" PRIu64 " ns -- a floor under every row\n",
                  s.clock_overhead_ns);
    o << line;

    auto bound_fn = [&s](auto const& h) -> uint64_t {
        uint64_t retval = 0;
        if (hop_crosses_device_clock(h)) {
            retval = s.device_clock_uncertainty_ns;
        } else if (hop_crosses_host_clock(h)) {
            retval = s.host_clock_uncertainty_ns;
        }
        return retval;
    };

    for (uint32_t h = 0; h < kHopStageCount; ++h) {
        const Dist d = s.merged(h);
        if (d.n == 0) {
            continue;
        }
        const uint64_t bound = bound_fn(h);

        const double mean_ns = d.mean_ns();
        if (bound > 0 && mean_ns < static_cast<double>(bound)) {
            std::snprintf(
                line,
                sizeof(line),
                "    WARNING: %s mean %.0f ns is below its own +/- %" PRIu64
                " ns bound -- report it as \"under the\n"
                "             resolution of the clock sync\", not as a value.\n",
                hop_name(h),
                mean_ns,
                bound);
            o << line;
        }
    }

    o << "\n=== work stealing ===\n\n";
    o << "  worker    scanned    found  delivered   stolen  donated   idle-spins\n";
    o << "  ------ ---------- -------- ---------- -------- -------- ------------\n";
    for (size_t i = 0; i < s.per_worker.size(); ++i) {
        const auto& w = s.per_worker[i];
        std::snprintf(
            line, sizeof(line), "  %6zu %10" PRIu64 " %8" PRIu64 " %10" PRIu64 " %8" PRIu64 " %8" PRIu64 " %12" PRIu64
                                "\n",
            i, w.scanned, w.found, w.delivered, w.stolen, w.donated, w.idle_spins);
        o << line;
    }

    uint64_t max_found = 0, min_found = UINT64_MAX;
    for (const auto& w : s.per_worker) {
        max_found = std::max(max_found, w.found);
        min_found = std::min(min_found, w.found);
    }
    if (!s.per_worker.empty() && max_found > 0) {
        const double mean_found = static_cast<double>(s.total_found()) / static_cast<double>(s.per_worker.size());
        std::snprintf(
            line, sizeof(line),
            "\n  serviced per worker: min %" PRIu64 ", mean %.1f, max %" PRIu64 " (imbalance %.2fx)\n"
            "  stolen %" PRIu64 " of %" PRIu64 " (%.1f%%)\n",
            min_found, mean_found, max_found, mean_found > 0 ? static_cast<double>(max_found) / mean_found : 0.0,
            s.total_stolen(), s.total_found(),
            s.total_found() ? 100.0 * static_cast<double>(s.total_stolen()) / static_cast<double>(s.total_found())
                            : 0.0);
        o << line;
        if (s.total_tx_credit_skips() > 0) {
            std::snprintf(line, sizeof(line),
                          "  tx credit skips %" PRIu64 " (%.1f per message serviced) -- armed TX banks\n"
                          "     passed over waiting on the peer's credit; high means CREDIT-bound,\n"
                          "     so send concurrency would not help\n",
                          s.total_tx_credit_skips(),
                          s.total_found() ? static_cast<double>(s.total_tx_credit_skips()) /
                                                static_cast<double>(s.total_found())
                                          : 0.0);
            o << line;
        }
    }

    uint64_t rej[8] = {};
    bool any = false;
    for (const auto& w : s.per_worker) {
        for (int i = 0; i < 8; ++i) {
            rej[i] += w.rejected[i];
            if (i != kCtrlOk && i != kCtrlIdle && w.rejected[i]) {
                any = true;
            }
        }
    }
    if (any) {
        o << "\n=== REJECTED CONTROL WORDS ===\n\n";
        for (int i = 0; i < 8; ++i) {
            if (i == kCtrlOk || i == kCtrlIdle || rej[i] == 0) {
                continue;
            }
            std::snprintf(line, sizeof(line), "  %-16s %" PRIu64 "\n", ctrl_verdict_name(i), rej[i]);
            o << line;
        }
    }

    if (s.wall_ns > 0 && s.total_bytes() > 0) {
        const double mb = static_cast<double>(s.total_bytes()) / 1e6;
        const double sec = static_cast<double>(s.wall_ns) / 1e9;
        std::snprintf(
            line, sizeof(line), "\n  %" PRIu64 " messages, %" PRIu64 " delivered, %.2f MB in %s => %.1f MB/s (wall)\n",
            s.total_found(), s.total_delivered(), mb, auto_unit(static_cast<double>(s.wall_ns)).c_str(), mb / sec);
        o << line;
    }

    if (s.timed_ns > 0 && s.total_timed_bytes() > 0) {
        const double usec = static_cast<double>(s.timed_ns) / 1e3;
        const uint64_t xfers = static_cast<uint64_t>(s.timed_iters) * s.cores;
        std::snprintf(line, sizeof(line),
                      "\n=== BANDWIDTH (completion-bounded interval) ===\n\n"
                      "  %-8s%-8s%-10s%10s %10s%13s%13s\n"
                      "  %-8u%-8u%-10.2f%10s%10.2f%13.2f%13.4f\n",
                      "bytes", "iters", "total_MB", "time", "MB/sec", "usec/xfer", "Mxfers/sec",
                      s.payload_bytes, s.timed_iters, static_cast<double>(s.total_timed_bytes()) / 1e6,
                      auto_unit(static_cast<double>(s.timed_ns)).c_str(), s.timed_mb_per_s(),
                      xfers > 0 ? usec / static_cast<double>(xfers) : 0.0,
                      xfers > 0 ? static_cast<double>(xfers) / usec : 0.0);
        o << line;

        const std::string window_str = s.window ? std::to_string(s.window) : std::string("per-core credit");
        std::snprintf(line, sizeof(line),
                      "  %u core%s, warmup %u discarded, window %s, sender %s\n"
                      "  interval opens after the warmup drain and closes on CONFIRMED ARRIVAL --\n"
                      "  not on the last post.\n",
                      s.cores, s.cores == 1 ? "" : "s", s.warmup,
                      window_str.c_str(), s.sender_shape.c_str());
        o << line;
    } else if (s.total_bytes() > 0) {
        o << "\n  NO BANDWIDTH NUMBER: the completion-bounded interval never closed (timed_ns=0).\n"
             "  The wall figure above spans setup and teardown too, so it is a floor, not a rate.\n";
    }

    o << pinning_warning(s);
    o << sample_count_warning(s);

    return o.str();
}

std::string sample_count_warning(const RunStats& s) {
    std::vector<std::pair<uint32_t, uint64_t>> pop;
    std::map<uint64_t, int> tally;
    for (uint32_t h = 0; h < kHopCount; ++h) {
        // diag:h2h-retire counts transport ops, two per message, so it sits at ~2x the message
	// population by construction on every run. Flagging it would teach the reader to skip
	// a warning that is meant to mean something.
        if (h == kHopH2HRetire) {
            continue;
        }
        const Dist d = s.merged(h);
        if (d.n == 0) {
            continue;
        }
        pop.emplace_back(h, d.n);
        ++tally[d.n];
    }
    if (pop.size() < 2) {
        return {};
    }
    uint64_t modal = 0;
    int best = 0;
    for (const auto& [n, count] : tally) {
        if (count > best || (count == best && n > modal)) {
            best = count;
            modal = n;
        }
    }

    // what the gate-flip race can account for. The gate is read at three instants -- per
    // serviced job, inside the scanner, and in the sender loop -- so a message in flight when it
    // opens is counted by whichever read landed after the flip. At most `window` can be in
    // flight, so a deficit up to that is the race; past it, samples are being dropped.
    const uint64_t in_flight = s.window != 0 ? s.window : s.cores;
    const uint64_t tolerance = in_flight != 0 ? in_flight : 4;

    std::ostringstream gross;
    int n_gross = 0;
    for (const auto& [h, n] : pop) {
        if (n == modal) {
            continue;
        }
        const int64_t delta = static_cast<int64_t>(n) - static_cast<int64_t>(modal);
        const uint64_t mag = static_cast<uint64_t>(delta < 0 ? -delta : delta);
        if (mag <= tolerance) {
            continue;  // the gate-flip race accounts for it
        }
        const double frac = static_cast<double>(mag) / static_cast<double>(modal);
        char line[160];
        std::snprintf(line, sizeof(line), "    %-22s %10" PRIu64 "  (%+" PRId64 ", %.3f%% of %" PRIu64 ")\n",
                      hop_name(h), n, delta, frac * 100.0, modal);
        gross << line;
        ++n_gross;
    }
    if (n_gross == 0) {
        return {};
    }

    std::ostringstream o;
    o << "\n  !! ROW POPULATION MISMATCH: " << n_gross << " row(s) are short or long by more than the\n"
         "     " << tolerance << " samples the gate-flip race can account for (the send window -- that is the\n"
         "     most messages that can be in flight when the recording gate opens). A row built from\n"
         "     a different population is not comparable to the rows beside it, and its mean is\n"
         "     whatever its surviving samples happened to be. Do not quote these:\n"
      << gross.str();
    return o.str();
}

std::string pinning_warning(const RunStats& s) {
    const uint32_t n = s.workers_unpinned();
    if (n == 0) {
        return {};
    }
    std::ostringstream o;
    o << "\n  !! " << n << " of " << s.per_worker.size()
      << " worker(s) NOT CPU-PINNED. The scheduler was free to migrate them mid-run, so\n"
         "     their hop timings carry migration noise and are not comparable with the\n"
         "     pinned workers'. The run continued and the numbers below are still real\n"
         "     work, but do not quote a spread that mixes these rows with the rest:\n";
    for (size_t i = 0; i < s.per_worker.size(); ++i) {
        if (!s.per_worker[i].pinned) {
            o << "       worker " << i << ": " << s.per_worker[i].pin_error << "\n";
        }
    }
    return o.str();
}

std::string basic_csv_header() {
    return "stage,samples,payload_bytes,window_ns,hop_window_ns,bandwidth_gb_per_s,"
           "messages_per_second,latency_us,total_ns,bytes_per_message,cores,run_id,host_ident\n";
}

std::string format_basic_csv(const RunStats& s, const std::string& tag) {
    (void)tag;
    static constexpr uint32_t kRows[] = {kHopT6ToHost,       kHopHostToRemoteHost, kHopRemoteHostToRemoteT6,
                                         kHopOneWayTotal,    kHopH2HPayloadAtPeer, kHopH2HCreditRaw,
                                         kHopH2HNet};
    std::ostringstream o;
    char line[512];
    for (const uint32_t h : kRows) {
        const Dist d = s.merged(h);
        const uint64_t bytes = s.merged_payload_bytes(h);

        char bw[32] = "";
        char msgs[32] = "";
        char lat[32] = "";
        const uint64_t hop_window = s.merged_hop_window_ns(h);
        const uint64_t denom = (h == kHopOneWayTotal) ? s.timed_ns : hop_window;
        if (bytes > 0 && denom > 0) {
            std::snprintf(bw, sizeof(bw), "%.6f",
                          static_cast<double>(bytes) / static_cast<double>(denom));
        }
        if (d.n > 0 && denom > 0) {
            std::snprintf(msgs, sizeof(msgs), "%.1f",
                          static_cast<double>(d.n) * 1e9 / static_cast<double>(denom));
        }
        if (d.n > 0) {
            std::snprintf(lat, sizeof(lat), "%.3f",
                          static_cast<double>(d.sum) / static_cast<double>(d.n) / 1000.0);
        }
        char hwin[32] = "";
        if (hop_window > 0) {
            std::snprintf(hwin, sizeof(hwin), "%" PRIu64, hop_window);
        }
        std::snprintf(line, sizeof(line),
                      "%s,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%s,%s,%s,%s,%" PRIu64 ",%u,%u,%s,%u\n",
                      h == kHopOneWayTotal ? "END_TO_END" : hop_name(h), d.n, bytes, s.timed_ns, hwin,
                      bw, msgs, lat, d.sum, s.payload_bytes, s.cores, s.run_id.c_str(), s.host_ident);
        o << line;
    }
    return o.str();
}

}  // namespace tt::tt_metal::experimental
