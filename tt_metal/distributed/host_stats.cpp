// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "host_stats.hpp"

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

#include "host_uva_layout.hpp"

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

// Millisecond timestamp plus pid. Not a UUID: it has to be short enough to read in a terminal
// and to type into a filter, and the pair is unique for any two processes that could be
// confused with each other -- two runs on one host differ in pid, two hosts write different
// files. The point is only to separate runs WITHIN a file.
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
// ITS header instead of the wide one -- otherwise every append to a basic file would report a
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

std::string rotate_csv(const std::string& path, std::string& error) {
    error.clear();

    struct stat st {};
    if (::stat(path.c_str(), &st) != 0) {
        return {};  // nothing there: the caller creates it and writes the header
    }

    // The EXISTING file's mtime, not now. The archived name should say when the data was
    // collected; a run archived three days later must not look like it was measured today.
    // UTC, matching the run_started_utc column, so the filename and the rows agree.
    std::tm tm {};
    const std::time_t mtime = st.st_mtime;
    if (::gmtime_r(&mtime, &tm) == nullptr) {
        error = "cannot convert mtime of " + path;
        return {};
    }
    char stamp[32];
    if (std::strftime(stamp, sizeof(stamp), "%Y%m%dT%H%M%SZ", &tm) == 0) {
        error = "cannot format mtime of " + path;
        return {};
    }

    // Insert before the extension, so a *.csv glob still finds the archive. Only a dot in the
    // final path component counts -- "./a.b/steady" has no extension.
    const std::size_t slash = path.find_last_of('/');
    const std::size_t dot = path.find_last_of('.');
    const bool has_ext = dot != std::string::npos && (slash == std::string::npos || dot > slash + 1);
    const std::string stem = has_ext ? path.substr(0, dot) : path;
    const std::string ext = has_ext ? path.substr(dot) : std::string {};

    for (int attempt = 0; attempt < 1000; ++attempt) {
        std::string target = stem + "." + stamp;
        if (attempt > 0) {
            target += "-" + std::to_string(attempt);
        }
        target += ext;

        // Two runs inside one second, or a re-rotation, must not clobber an archive.
        struct stat exists {};
        if (::stat(target.c_str(), &exists) == 0) {
            continue;
        }
        if (::rename(path.c_str(), target.c_str()) == 0) {
            return target;
        }
        error = "cannot rename " + path + " to " + target + ": " + std::strerror(errno);
        return {};
    }
    error = "cannot find a free archive name for " + path;
    return {};
}

uint64_t measure_clock_overhead_ns() {
    // Minimum of many back-to-back reads. The MINIMUM, not the mean: what is wanted is the
    // cost of the call with nothing in the way, and the mean folds in whatever else the
    // scheduler did during the loop.
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
    o << "  stage                     count        min       mean        max     rel.sd     MB/s\n";
    o << "  ----------------------- ------- ---------- ---------- ---------- ---------- --------\n";
    for (uint32_t h = 0; h < kHopStageCount; ++h) {
        const Dist d = s.merged(h);
        const char* domain = hop_crosses_device_clock(h) ? "dev" : (hop_crosses_host_clock(h) ? "host" : "-");
        if (d.n == 0) {
            // Printed, not skipped. A stage with no samples means that leg did not run in
            // this mode, and dropping the row makes a missing leg look like a leg that was
            // never part of the path.
            std::snprintf(
                line, sizeof(line), "  %-23s %7s %10s %10s %10s %10s %6s\n", hop_name(h), "-", "-", "-", "-", "-",
                domain);
            o << line;
            continue;
        }
        std::snprintf(
            line, sizeof(line), "  %-23s %7" PRIu64 " %10s %10s %10s %9.3f%% %8.1f\n", hop_name(h), d.n,
            auto_unit(static_cast<double>(d.min)).c_str(), auto_unit(d.mean).c_str(),
            auto_unit(static_cast<double>(d.max)).c_str(), d.rel_stddev() * 100.0,
            d.mean > 0 ? static_cast<double>(s.payload_bytes) * 1000.0 / d.mean : 0.0);
        (void)domain;
        o << line;
    }

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
            for (uint32_t h = kHopStageCount; h < kHopCount; ++h) {
                const Dist d = s.merged(h);
                if (d.n == 0) {
                    continue;  // unlike a stage, an absent diagnostic means "not applicable here"
                }
                std::snprintf(
                    line, sizeof(line), "  %-23s %7" PRIu64 " %10s %10s %10s %9.3f%%\n", hop_name(h), d.n,
                    auto_unit(static_cast<double>(d.min)).c_str(), auto_unit(d.mean).c_str(),
                    auto_unit(static_cast<double>(d.max)).c_str(), d.rel_stddev() * 100.0);
                o << line;
            }
        }
    }

    // The bounds, stated next to the rows they apply to rather than buried. A stage whose
    // duration is smaller than its own uncertainty is not a measurement.
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

    for (uint32_t h = 0; h < kHopStageCount; ++h) {
        const Dist d = s.merged(h);
        if (d.n == 0) {
            continue;
        }
        const uint64_t bound = hop_crosses_device_clock(h)   ? s.device_clock_uncertainty_ns
                               : hop_crosses_host_clock(h)   ? s.host_clock_uncertainty_ns
                                                             : 0;
        if (bound > 0 && d.mean < static_cast<double>(bound)) {
            std::snprintf(
                line, sizeof(line),
                "    WARNING: %s mean %.0f ns is below its own +/- %" PRIu64 " ns bound -- report it as \"under the\n"
                "             resolution of the clock sync\", not as a value.\n",
                hop_name(h), d.mean, bound);
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
        const uint64_t xfers = static_cast<uint64_t>(s.timed_iters) * s.cores * s.xfers_per_iter;
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
                      "  %u core%s x %u xfer%s/iter, warmup %u discarded, window %s, sender %s\n"
                      "  interval opens after the warmup drain and closes on CONFIRMED ARRIVAL --\n"
                      "  not on the last post. Compare: fi_rma_bw -p <prov> -e rdm -S %u -W 64\n",
                      s.cores, s.cores == 1 ? "" : "s", s.xfers_per_iter, s.xfers_per_iter == 1 ? "" : "s", s.warmup,
                      window_str.c_str(), s.sender_shape.c_str(), s.payload_bytes);
        o << line;
    } else if (s.total_bytes() > 0) {
        o << "\n  NO BANDWIDTH NUMBER: the completion-bounded interval never closed (timed_ns=0).\n"
             "  The wall figure above spans setup and teardown too, so it is a floor, not a rate.\n";
    }

    o << pinning_warning(s);
    o << sample_count_warning(s);

    return o.str();
}

std::string csv_header() {
    return "tag,mode,provider,payload_bytes,cores,iters,workers,stage,stage_index,is_stage,clock_domain,"
           "count,"
           "min_ns,mean_ns,max_ns,"
           "min_us,mean_us,max_us,"
           "min_ms,mean_ms,max_ms,"
           "rel_sd,uncertainty_ns,below_uncertainty,"
           "mb_per_s_mean,mb_per_s_best,wall_mb_per_s,"
           "messages,delivered,total_bytes,wall_ns,stolen,clock_overhead_ns,"
           "run_id,run_started_utc,role,host_ident,symmetric,tx_side,"
           "warmup,warmup_applied,ns_per_cycle,device_clock_ghz,"
           "rate_is_bandwidth,timed_ns,timed_bytes,timed_mb_per_s,timed_iters,xfers_per_iter,window,"
           "samples_warmup_gated,"
           "wire_bytes,wire_mb_per_s,"
           "h2d,"
           "sender_shape\n";
}

std::string sample_count_warning(const RunStats& s) {
    std::vector<std::pair<uint32_t, uint64_t>> pop;
    std::map<uint64_t, int> tally;
    for (uint32_t h = 0; h < kHopCount; ++h) {
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

    std::ostringstream gross, minor;
    int n_gross = 0, n_minor = 0;
    for (const auto& [h, n] : pop) {
        if (n == modal) {
            continue;
        }
        const int64_t delta = static_cast<int64_t>(n) - static_cast<int64_t>(modal);
        const double frac = static_cast<double>(delta < 0 ? -delta : delta) / static_cast<double>(modal);
        char line[160];
        std::snprintf(line, sizeof(line), "    %-22s %10" PRIu64 "  (%+" PRId64 ", %.3f%% of %" PRIu64 ")\n",
                      hop_name(h), n, delta, frac * 100.0, modal);
        if (frac > 0.01) {
            gross << line;
            ++n_gross;
        } else if (delta > 4 || delta < -4) {
            minor << line;
            ++n_minor;
        }
    }
    if (n_gross == 0 && n_minor == 0) {
        return {};
    }

    std::ostringstream o;
    if (n_gross > 0) {
        o << "\n  !! ROW POPULATION MISMATCH: " << n_gross << " row(s) do not share this run's sample count.\n"
             "     A row built from a different population is not comparable to the rows beside it,\n"
             "     and its mean is whatever its surviving samples happened to be. Do not quote these:\n"
          << gross.str();
    }
    if (n_minor > 0) {
        o << "\n  note: " << n_minor << " row(s) off by a handful of samples (gate-flip race, ANALYSIS.md B.1):\n"
          << minor.str();
    }
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

std::string format_trace_csv(const RunStats& s, const std::string& tag) {
    const std::vector<TraceBucket> t = s.merged_trace();
    const uint64_t width_ns = 1ull << s.trace_shift;

    uint32_t last = 0;
    for (uint32_t i = 0; i < kTraceBuckets; ++i) {
        if (t[i].n > 0) {
            last = i;
        }
    }

    std::ostringstream o;
    o << "tag,provider,payload_bytes,cores,run_id,role,tx_side,bucket_ns,clamped,"
         "bucket,t_ns,bytes,cum_bytes,inst_mb_per_s,n,mean_ns\n";
    uint64_t cum = 0;
    for (uint32_t i = 0; i <= last; ++i) {
        cum += t[i].bytes;
        o << tag << ',' << s.provider << ',' << s.payload_bytes << ',' << s.cores << ',' << s.run_id << ','
          << s.role << ',' << (s.tx_side ? 1 : 0) << ',' << width_ns << ',' << s.total_trace_clamped() << ','
          << i << ',' << (static_cast<uint64_t>(i) * width_ns) << ',' << t[i].bytes << ',' << cum << ',';
        char rate[32] = "";
        if (t[i].n > 0) {
            std::snprintf(rate, sizeof(rate), "%.3f", static_cast<double>(t[i].bytes) * 1000.0 / static_cast<double>(width_ns));
        }
        o << rate << ',' << t[i].n << ',';
        if (t[i].n > 0) {
            o << (t[i].ns_sum / t[i].n);
        }
        o << '\n';
    }
    return o.str();
}

// ===========================================================================
// THE STRIPPED CSV
//
// Four rows, thirteen columns, and every derived number reproducible from the raw columns in
// the SAME row with a calculator:
//
//   bandwidth_gb_per_s   == payload_bytes / hop_window_ns  <- all three LEG rows
//   bandwidth_gb_per_s   == payload_bytes / window_ns      <- the END_TO_END row only
//   messages_per_second  == samples * 1e9 / <that row's denominator>
//   latency_us           == total_ns / samples / 1000
//   payload_bytes        == samples * bytes_per_message
//
//   payload_bytes / total_ns        the PER-CORE PUSH RATE -- bytes over the time cores spent
//                                   inside the leg. This is what the t6->host bandwidth cell
//                                   printed until 2026-09-07: 29.207 GB/s at 512 K, 93% of
//                                   Gen5 x8. It is a real number and it is NOT a link rate; it
//                                   does not become one by multiplying by `cores`.
//   total_ns / hop_window_ns        MESSAGES IN FLIGHT (T/W) -- the concurrency, and exactly the
//                                   factor by which S/T understates S/W.
//
// so S/W == (S/T) x (T/W) is checkable on the row, which is the whole reason both columns stay.
//
// GB/s needs no scale factor: 1 byte/ns is 1e9 B/s is 1 GB/s decimal. The division is the
// answer as written.
//
// ===========================================================================
std::string basic_csv_header() {
    return "stage,samples,payload_bytes,window_ns,hop_window_ns,bandwidth_gb_per_s,"
           "messages_per_second,latency_us,total_ns,bytes_per_message,cores,run_id,host_ident\n";
}

std::string format_basic_csv(const RunStats& s, const std::string& tag) {
    (void)tag;
    static constexpr uint32_t kRows[] = {kHopT6ToHost, kHopHostToRemoteHost, kHopRemoteHostToRemoteT6,
                                         kHopOneWayTotal};
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

std::string format_csv(const RunStats& s, const std::string& tag) {
    std::ostringstream o;
    char line[1024];
    for (uint32_t h = 0; h < kHopCount; ++h) {
        const Dist d = s.merged(h);
        if (d.n == 0) {
            continue;
        }
        const int is_stage = h < kHopStageCount ? 1 : 0;
        const uint64_t bound = hop_crosses_device_clock(h)   ? s.device_clock_uncertainty_ns
                               : hop_crosses_host_clock(h)   ? s.host_clock_uncertainty_ns
                                                             : 0;
        const char* domain = hop_crosses_device_clock(h) ? "device" : (hop_crosses_host_clock(h) ? "host" : "none");
        const double mn = static_cast<double>(d.min), mx = static_cast<double>(d.max);

        char rate_mean[32] = "";
        char rate_best[32] = "";
        if (hop_rate_is_bandwidth(h)) {
            if (d.mean > 0) {
                std::snprintf(rate_mean, sizeof(rate_mean), "%.3f", static_cast<double>(s.payload_bytes) * 1000.0 / d.mean);
            }
            if (d.min > 0) {
                std::snprintf(rate_best, sizeof(rate_best), "%.3f",
                              static_cast<double>(s.payload_bytes) * 1000.0 / static_cast<double>(d.min));
            }
        }
        std::snprintf(
            line, sizeof(line),
            "%s,%s,%s,%u,%u,%u,%zu,%s,%u,%d,%s,%" PRIu64 ","
            "%" PRIu64 ",%.1f,%" PRIu64 ","
            "%.4f,%.4f,%.4f,"
            "%.7f,%.7f,%.7f,"
            "%.6f,%" PRIu64 ",%d,"
            "%s,%s,%.3f,"
            "%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64,
            tag.c_str(), s.mode.c_str(), s.provider.c_str(), s.payload_bytes, s.cores, s.iters, s.per_worker.size(),
            hop_name(h), h, is_stage, domain, d.n,
            d.min, d.mean, d.max,
            mn / 1e3, d.mean / 1e3, mx / 1e3,
            mn / 1e6, d.mean / 1e6, mx / 1e6,
            d.rel_stddev(), bound, (bound > 0 && d.mean < static_cast<double>(bound)) ? 1 : 0,
            rate_mean, rate_best,
            s.wall_ns > 0 ? static_cast<double>(s.total_bytes()) * 1000.0 / static_cast<double>(s.wall_ns) : 0.0,
            s.total_found(), s.total_delivered(), s.total_bytes(), s.wall_ns, s.total_stolen(),
            s.clock_overhead_ns);

        o << line << ',' << s.run_id << ',' << s.run_started_utc << ',' << s.role << ',' << s.host_ident << ','
          << (s.symmetric ? 1 : 0) << ',' << (s.tx_side ? 1 : 0) << ',' << s.warmup << ','
          << (s.warmup_applied ? 1 : 0) << ',' << s.ns_per_cycle << ','
          << (s.ns_per_cycle > 0.0 ? 1.0 / s.ns_per_cycle : 0.0) << ','
          << (hop_rate_is_bandwidth(h) ? 1 : 0) << ',';

        if (s.timed_ns > 0) {
            o << s.timed_ns << ',' << s.total_timed_bytes() << ',' << s.timed_mb_per_s() << ',';
        } else {
            o << ",,,";
        }
        o << s.timed_iters << ',' << s.xfers_per_iter << ',' << s.window << ','
          << (hop_samples_warmup_gated(h) ? 1 : 0) << ',';

        const uint64_t wire = s.merged_wire_bytes(h);
        if (wire > 0) {
            o << wire << ',';
            if (d.mean > 0 && d.n > 0) {
                const double per_sample = static_cast<double>(wire) / static_cast<double>(d.n);
                std::snprintf(line, sizeof(line), "%.3f", per_sample * 1000.0 / d.mean);
                o << line;
            }
            o << ',' << s.h2d << ',' << s.sender_shape << '\n';
        } else {
            o << ",," << s.h2d << ',' << s.sender_shape << '\n';
        }
    }
    return o.str();
}


std::string ladder_csv_header() {
    return "tag,run_id,provider,cores,workers,chunk_bytes,checkpoint,nominal_bytes,actual_bytes,"
           "stage,stage_index,is_stage,rate_is_bandwidth,quiesced,quiesce_clean,quiesce_degraded,discarded_bytes,"
           "win_count,win_min_ns,win_mean_ns,win_max_ns,win_rel_sd,win_mb_per_s,"
           "cum_count,cum_min_ns,cum_mean_ns,cum_max_ns,cum_rel_sd,cum_mb_per_s\n";
}

std::string ladder_csv_rows(const RunStats& s, const std::string& tag) {
    std::ostringstream o;
    const size_t points = s.ladder_points();
    const uint32_t workers = static_cast<uint32_t>(s.per_worker.size());
    for (size_t i = 0; i < points; ++i) {
        const uint64_t nominal = i < s.ladder.marks.size() ? s.ladder.marks[i] : 0;
        const uint64_t actual = s.ladder_bytes_at(i);
        for (uint32_t h = 0; h < kHopCount; ++h) {
            const Dist w = s.ladder_window(i, h);
            const Dist c = s.ladder_cumulative(i, h);
            if (w.n == 0 && c.n == 0) {
                continue;
            }
            auto rate = [&](const Dist& d) -> std::string {
                if (!hop_rate_is_bandwidth(h) || d.n == 0 || d.mean <= 0.0) {
                    return "";
                }
                char buf[64];
                std::snprintf(buf, sizeof(buf), "%.3f",
                              static_cast<double>(s.ladder.chunk_bytes) * 1000.0 / d.mean);
                return buf;
            };
            auto rsd = [](const Dist& d) -> std::string {
                if (d.n < 2 || d.mean <= 0.0) {
                    return "";
                }
                char buf[64];
                std::snprintf(buf, sizeof(buf), "%.3f",
                              100.0 * std::sqrt(d.m2 / static_cast<double>(d.n - 1)) / d.mean);
                return buf;
            };
            auto emit = [&](const Dist& d) {
                o << d.n << ',' << (d.n ? d.min : 0) << ',' << static_cast<uint64_t>(d.mean) << ','
                  << d.max << ',' << rsd(d) << ',' << rate(d);
            };
            o << tag << ',' << s.run_id << ',' << s.provider << ',' << s.cores << ',' << workers << ','
              << s.ladder.chunk_bytes << ',' << i << ',' << nominal << ',' << actual << ','
              << hop_name(h) << ',' << h << ',' << (h < kHopStageCount ? 1 : 0) << ','
              << (hop_rate_is_bandwidth(h) ? 1 : 0) << ',' << (s.ladder.quiesced ? 1 : 0) << ','
              << s.ladder.quiesce_clean << ',' << s.ladder.quiesce_degraded << ','
              << s.ladder.discarded_bytes << ',';
            emit(w);
            o << ',';
            emit(c);
            o << '\n';
        }
    }
    return o.str();
}

}  // namespace tt::tt_metal::experimental
