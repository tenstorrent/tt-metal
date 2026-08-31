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

std::string csv_schema_error(const std::string& path) {
    std::ifstream in(path);
    if (!in.good()) {
        return {};  // no file: the caller writes the header
    }
    std::string first;
    if (!std::getline(in, first) || first.empty()) {
        return {};  // empty file: same
    }
    std::string want = csv_header();
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

    // THE DIAGNOSTIC ROWS, WHICH USED TO BE COMPUTED AND THEN DROPPED ON THE FLOOR.
    //
    // This loop runs from kHopStageCount to kHopCount and did not exist. Every diag:* hop was
    // accumulated, written to the CSV, and never printed -- so `run_point`'s `ret=` column in
    // run_sweep_{server,peer}.sh, which greps stdout for "^  diag:h2h-retire", matched nothing
    // and printed "-" on EVERY point of every campaign these scripts have ever run. The
    // extraction was written against a table that does not print what it is looking for.
    //
    // diag:h2h-retire is the one that matters and is the reason this is worth fixing: on verbs
    // `host->remote_host` brackets the POST ONLY (a descriptor handoff, flat in payload size),
    // so the transfer itself is invisible unless this row is readable. Reading the post as a
    // transfer is what produced the 240 GB/s figure the CSV now writes empty.
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
        // CREDIT-BOUND OR POST-BOUND? Printed next to the stealing figures because it is the
        // same question one level down: stealing says whether workers are balanced, this says
        // whether the send path can proceed at all. Ratio against `found`, since one skip per
        // message is normal pacing and many skips per message is the credit ladder throttling.
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

    // THE BANDWIDTH LINE, laid out like fabtests' show_perf() so the two can be read side by
    // side against `fi_rma_bw` on the same NIC pair. Same columns, same MB = 10^6 bytes.
    if (s.timed_ns > 0 && s.total_timed_bytes() > 0) {
        const double usec = static_cast<double>(s.timed_ns) / 1e3;
        const uint64_t xfers = static_cast<uint64_t>(s.timed_iters) * s.cores * s.xfers_per_iter;
        std::snprintf(line, sizeof(line),
                      "\n=== BANDWIDTH (completion-bounded interval) ===\n\n"
                      "  %-8s%-8s%-10s%10s %10s%13s%13s\n"
                      "  %-8u%-8u%-10.2f%10s%10.2f%13.2f%13.4f\n",
                      "bytes", "iters", "total_MB", "time", "MB/sec", "usec/xfer", "Mxfers/sec",
                      s.payload_bytes, s.timed_iters, static_cast<double>(s.total_timed_bytes()) / 1e6,
                      // auto_unit rather than fabtests' fixed "%8.2fs": their runs are seconds
                      // long by construction (size_to_count picks 200-20,000 iterations), and a
                      // short run here would print 0.00s for a perfectly good measurement.
                      auto_unit(static_cast<double>(s.timed_ns)).c_str(), s.timed_mb_per_s(),
                      xfers > 0 ? usec / static_cast<double>(xfers) : 0.0,
                      xfers > 0 ? static_cast<double>(xfers) / usec : 0.0);
        o << line;
        // A named local rather than a conditional on `.c_str()`: the temporary string would
        // survive the call, but only just, and it is the kind of expression that stops being
        // correct the moment someone hoists it.
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

    // LAST, so it is the final thing on stdout rather than something to scroll back for. A row
    // built from the wrong population invalidates every comparison made against it, so it
    // belongs after the numbers it disqualifies, not above them.
    o << sample_count_warning(s);

    return o.str();
}

// EVERY DURATION IN ALL THREE UNITS, as asked. Derived columns rather than a unit flag:
// a spreadsheet that has to multiply by 1000 is a spreadsheet where someone eventually
// multiplies the wrong column, and these files are meant to be compared across runs and
// machines.
std::string csv_header() {
    return "tag,mode,provider,payload_bytes,cores,iters,workers,stage,stage_index,is_stage,clock_domain,"
           "count,"
           "min_ns,mean_ns,max_ns,"
           "min_us,mean_us,max_us,"
           "min_ms,mean_ms,max_ms,"
           "rel_sd,uncertainty_ns,below_uncertainty,"
           "mb_per_s_mean,mb_per_s_best,wall_mb_per_s,"
           "messages,delivered,total_bytes,wall_ns,stolen,clock_overhead_ns,"
           // APPENDED, NOT INSERTED. Anything already pivoting these files by column
           // position keeps working; only readers that want the new fields need to change.
           //
           // run_id/run_started_utc: rows APPEND across invocations, and without an id the
           //   only way to separate runs is position in the file. See ANALYSIS.md B.2.1 --
           //   that is how a 1.945x bimodality in stage 1 hid inside a provider comparison.
           // role/host_ident/symmetric/tx_side: stages 1-2 are recorded by the sender and
           //   stage 3 plus the total by the receiver, in different processes and files.
           // warmup/warmup_applied: two different facts, and they used to disagree.
           // ns_per_cycle/device_clock_ghz: the scalar that converted stage 1 out of cycles.
           "run_id,run_started_utc,role,host_ident,symmetric,tx_side,"
           "warmup,warmup_applied,ns_per_cycle,device_clock_ghz,"
           // THE COMPLETION-BOUNDED BRACKET. `timed_mb_per_s` is the only bandwidth in this
           // file; every other rate column is per-hop and now EMPTY wherever the hop's
           // interval does not contain the bytes (rate_is_bandwidth=0). See
           // hop_rate_is_bandwidth() and MEASURING-BANDWIDTH.md.
           //
           // These are per-RUN values repeated on every row, like wall_ns above: the file is
           // one row per hop, and a reader grouping by run_id gets the same constant from
           // whichever row it happens to hold.
           "rate_is_bandwidth,timed_ns,timed_bytes,timed_mb_per_s,timed_iters,xfers_per_iter,window,"
           // PER-ROW, unlike `warmup`/`warmup_applied` beside it, which are run-level and say
           // only that a warmup was configured. Three rows used to ignore the gate while
           // carrying warmup_applied=1, so the file asserted something false about them. See
           // hop_samples_warmup_gated().
           "samples_warmup_gated,"
           // BYTES THIS HOP PUT ON THE WIRE, and the rate that follows from them. Written only
           // for hops that call add_sample_with_size(); EMPTY, not 0, everywhere else, so a
           // reader cannot mistake "this hop does not track it" for "this hop moved nothing".
           //
           // wire_mb_per_s divides these bytes by THIS HOP's own duration, so it is subject to
           // the same caveat as every other per-hop rate -- it is a rate only where the bytes
           // actually crossed inside the interval (rate_is_bandwidth). It is here because the
           // gap between it and a payload-derived rate is the per-message protocol overhead,
           // which is 2x at a 32 B payload and nothing at 1 MiB, and no column showed it.
           "wire_bytes,wire_mb_per_s,"
           // WHICH H2D IMPLEMENTATION. Appended like everything above it. See RunStats::h2d
           // for why the filename was not a good enough record of this.
           "h2d,"
           // WHICH SENDER SHAPE. Appended, not inserted, for the same reason as everything
           // above it. Pairs with `window`: the two are independent knobs and a file that
           // records only one of them cannot distinguish "one in flight, spinning" from
           // "one in flight, parked" -- which is the whole question the shapes exist to
           // answer. See RunStats::sender_shape.
           "sender_shape\n";
}

// EVERY ROW OF A RUN HOLDS ONE POPULATION, AND NOTHING CHECKED THAT UNTIL NOW.
//
// hop_samples_warmup_gated() is a hand-maintained table that returns true for everything, so
// it records an intention; this function tests the consequence. The two are not the same
// thing and the difference has cost real numbers twice:
//
//   * Three rows spanned the whole run while the stages beside them spanned the timed window
//     (diag:decode, diag:h2h-retire, and half of diag:steal-wait). Every one of them was
//     found by comparing sample counts -- none of them could fail a run, and the CSV said
//     warmup_applied=1 on all of them alike.
//   * diag:h2h-retire at 256 B on verbs reports 25.8 us built from ONE sample where every
//     sibling row holds 100. It reproduces across three campaigns, it carries
//     samples_warmup_gated=1, and the point PASSes. 2.3x its neighbours off 1% of the
//     population, and nothing anywhere said so.
//
// MODAL RATHER THAN DERIVED, deliberately. The obvious check is against
// timed_iters * cores * xfers_per_iter, but that number is right only for the modes and roles
// where it is right, and a check that is wrong in a corner cries wolf until it gets deleted.
// The mode of the observed counts needs no such knowledge: whatever the population should
// have been, the rows should agree on it, and the outlier is the row that does not.
//
// TWO TIERS, because the tolerances differ by three orders of magnitude. A few samples of
// disagreement is the documented gate-flip race -- the scanner reads the flag just before the
// service path sets it (ANALYSIS.md B.1) -- and is not worth shouting about. A percent or
// more is a row measuring a different event over a different window, which is the defect
// class above.
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
    // Ties break toward the LARGER count, because the failure that actually happens is a row
    // losing samples, not a row inventing them. With two rows at 100 and 1 there is no mode,
    // and calling 1 the population would name 100 as the outlier -- backwards.
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

std::string format_trace_csv(const RunStats& s, const std::string& tag) {
    const std::vector<TraceBucket> t = s.merged_trace();
    const uint64_t width_ns = 1ull << s.trace_shift;

    // Find the last bucket that holds anything, so trailing empties are not written. A run
    // shorter than the span would otherwise emit thousands of zero rows and a plot of it would
    // show a long flat tail that is absence, not a stall.
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
        // Bytes over the BUCKET WIDTH, which is the derivative of the cumulative curve and the
        // instantaneous rate. Empty rather than 0 for a bucket with no samples, so a gap reads
        // as a gap.
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

std::string format_csv(const RunStats& s, const std::string& tag) {
    std::ostringstream o;
    char line[1024];
    // EVERY HOP, NOT JUST THE EIGHT STAGES.
    //
    // The diagnostic hops -- l1-write and doorbell (the two halves of the H->D leg), decode,
    // notice, steal-wait -- were printed to the console and dropped from the file, so "this leg
    // may have sub-legs" was answerable only from scrollback. They are in the file now, with
    // `is_stage` distinguishing them: filter is_stage=1 for the six legs plus two totals,
    // is_stage=0 for the breakdown inside them.
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

        // PER-HOP RATE, WRITTEN ONLY WHERE IT IS ONE. An empty field parses as NaN in pandas
        // and is skipped by awk's arithmetic, so a reader that averages the column no longer
        // silently folds in 240 GB/s from a hop that never carried a byte. Zero would NOT do
        // that: zero is a number and it would drag the mean the other way.
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
            // Bandwidth per stage: this payload divided by the time THIS stage took -- and
            // EMPTY for any stage whose interval does not contain the payload, which is what
            // hop_rate_is_bandwidth() decides above. The two host-to-host stages are the ones
            // this removes, and they are the ones that read 240 GB/s.
            rate_mean, rate_best,
            // THROUGHPUT AT THE WALL CLOCK. Honest arithmetic over the wrong span: `wall_ns`
            // runs from BankScanner::start() to join(), so it contains kernel JIT, first-touch
            // faults, the warmup and teardown, while `total_bytes` is ungated and includes the
            // warmup payloads. It cannot exceed the hardware -- it is a real elapsed time --
            // but it is biased LOW by however much setup the run paid.
            //
            // `timed_mb_per_s` below is the same quantity over the completion-bounded bracket
            // and is the number to quote. This column stays because a large gap between the two
            // is itself the diagnostic: it says setup dominated the run.
            s.wall_ns > 0 ? static_cast<double>(s.total_bytes()) * 1000.0 / static_cast<double>(s.wall_ns) : 0.0,
            s.total_found(), s.total_delivered(), s.total_bytes(), s.wall_ns, s.total_stolen(),
            s.clock_overhead_ns);
        // The identity columns go through the stream rather than the snprintf: they include
        // three strings of caller-controlled length, and silently truncating a row at 1024
        // bytes is exactly the class of corruption csv_schema_error() exists to prevent.
        o << line << ',' << s.run_id << ',' << s.run_started_utc << ',' << s.role << ',' << s.host_ident << ','
          << (s.symmetric ? 1 : 0) << ',' << (s.tx_side ? 1 : 0) << ',' << s.warmup << ','
          << (s.warmup_applied ? 1 : 0) << ',' << s.ns_per_cycle << ','
          << (s.ns_per_cycle > 0.0 ? 1.0 / s.ns_per_cycle : 0.0) << ','
          << (hop_rate_is_bandwidth(h) ? 1 : 0) << ',';
        // EMPTY, NOT ZERO, when the bracket never closed. A run that aborted before the stop
        // stamp has no bandwidth measurement; writing 0.0 would put it in the same cell as a
        // run that genuinely moved nothing, and a sweep averaging the column would then be
        // pulled toward zero by its own failures.
        if (s.timed_ns > 0) {
            o << s.timed_ns << ',' << s.total_timed_bytes() << ',' << s.timed_mb_per_s() << ',';
        } else {
            o << ",,,";
        }
        o << s.timed_iters << ',' << s.xfers_per_iter << ',' << s.window << ','
          << (hop_samples_warmup_gated(h) ? 1 : 0) << ',';
        // EMPTY for a hop that does not track wire bytes, so "not measured" and "measured
        // zero" stay distinguishable -- the same reasoning as the per-hop rate cells above.
        const uint64_t wire = s.merged_wire_bytes(h);
        if (wire > 0) {
            o << wire << ',';
            if (d.mean > 0 && d.n > 0) {
                // Bytes per sample over the mean duration of a sample: the same arithmetic as
                // mb_per_s_mean, on wire bytes instead of payload.
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
           // WINDOW: only the messages between the previous checkpoint and this one. This is
           // the column that shows drift -- warmup, thermal, credit steady-state -- because
           // each row is an independent sample rather than an average diluted by everything
           // before it.
           "win_count,win_min_ns,win_mean_ns,win_max_ns,win_rel_sd,win_mb_per_s,"
           // CUMULATIVE: every message from the start. The last row of a run equals the
           // matching row in the main CSV, which is the check that the two agree.
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
            // A hop with no samples in EITHER view contributed nothing at this checkpoint and
            // is omitted -- writing a zero row would put "this hop does not run in this mode"
            // in the same cell as "this hop ran and measured nothing".
            if (w.n == 0 && c.n == 0) {
                continue;
            }
            auto rate = [&](const Dist& d) -> std::string {
                // Bytes over the hop's OWN mean duration, and only where that interval
                // actually contains the bytes -- the same rule the main CSV applies via
                // hop_rate_is_bandwidth(). Empty, not zero, everywhere else.
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
              // RUN-LEVEL, repeated per row like wall_ns: how many checkpoints held every
              // worker inside the budget, and how many gave up and proceeded. quiesced=1 with
              // a non-zero degraded count is a ladder that ASKED for exact boundaries and did
              // not get them everywhere -- which is a different thing from not asking.
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
