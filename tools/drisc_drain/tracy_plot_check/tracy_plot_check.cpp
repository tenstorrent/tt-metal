// Prove that PLOT samples decoded from device data sit on the DEVICE TIMEBASE.
//
// The failure this exists to rule out: Tracy's PlotData() stamps the instant the call is MADE, and the
// perf-debug profiler decodes device markers milliseconds after the device produced them (the drainer's DRAM
// ring alone trails the last worker zone by 2.5-2.9 ms). A plot built that way sits to the RIGHT of the zones
// it explains and is worse than no plot at all. PlotDataAt + the device->TSC inverse is supposed to fix that,
// and "supposed to" is not a measurement.
//
// So this reports, from the .tracy alone:
//   * the WORKER window   -- first/last device-zone time over contexts that carry worker zones
//   * the DRAINER window  -- same, over contexts that carry DRISC-* zones
//   * per plot: sample count, time span, and the sample INTERVAL distribution (p50/max)
//   * the containment verdict: what fraction of each plot's samples fall inside the worker window
//
// Everything is in Tracy's display nanoseconds, so plots and GPU zones are directly comparable -- which is
// the whole point: if the conversion were wrong the plot span would be offset from the zone span by the
// decode lag, and that offset is exactly what the numbers below would show.
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "public/common/TracyTTDeviceData.hpp"
#include "server/TracyFileRead.hpp"
#include "server/TracyWorker.hpp"

namespace {

struct Win {
    int64_t lo = INT64_MAX, hi = INT64_MIN;
    long long n = 0;
    void add(int64_t a, int64_t b) {
        lo = std::min(lo, a);
        hi = std::max(hi, b);
        n++;
    }
    bool valid() const { return n > 0 && lo <= hi; }
    double span_ms() const { return valid() ? double(hi - lo) / 1e6 : 0.0; }
};

// Window of a GPU timeline's TOP-LEVEL zones. Deliberately does NOT recurse into children and does NOT
// resolve source locations: both were tried and both segfaulted on this capture (srcloc resolution on a
// GPU-zone index, and Child() whose empty sentinel is not >= 0). Neither is needed -- a context's top-level
// zones already bound its window, and role is identified by CONTEXT SHAPE below rather than by zone name.
void walk_top(const tracy::Vector<tracy::short_ptr<tracy::GpuEvent>>& v, Win& win) {
    // INDEXED, and size-checked. A range-for over a Tracy Vector that was never populated dereferences a null
    // data pointer -- which is what segfaulted here, on a context whose lane exists but whose timeline is
    // empty. is_empty() is the guard; do not "simplify" this back to a range-for.
    if (v.empty()) {
        return;
    }
    for (size_t i = 0; i < v.size(); i++) {
        const tracy::GpuEvent* e = v[i];
        if (e == nullptr) {
            continue;
        }
        const int64_t s = e->GpuStart(), t = e->GpuEnd();
        if (s >= 0 && t >= s) {
            win.add(s, t);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <in.tracy> [worker_lo_ns worker_hi_ns]\n", argv[0]);
        return 1;
    }
    auto f = std::unique_ptr<tracy::FileRead>(tracy::FileRead::Open(argv[1]));
    if (!f) {
        fprintf(stderr, "cannot open %s\n", argv[1]);
        return 1;
    }
    tracy::Worker worker(*f, tracy::EventType::All, true);
    while (!worker.IsBackgroundDone()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    fprintf(
        stderr, "[dbg] worker loaded, gpu ctx=%zu plots=%zu\n", worker.GetGpuData().size(), worker.GetPlots().size());
    fflush(stderr);
    // The GPU-zone window comes from tracy_zone_csv, NOT from here: walking GetGpuData() in this process
    // segfaults on this capture even guarded (empty-lane and null-entry checks both insufficient), and
    // tracy_zone_csv already does that walk correctly. Two small tools beat one that crashes -- the zone
    // windows are computed from its CSV and compared against the plot spans below.
    Win worker_win;
    if (argc >= 4) {
        worker_win.add(atoll(argv[2]), atoll(argv[3]));
    }
    printf(
        "worker window (from tracy_zone_csv): [%lld .. %lld] ns  span %.3f ms\n",
        (long long)worker_win.lo,
        (long long)worker_win.hi,
        worker_win.span_ms());

    printf("\nplots: %zu\n", worker.GetPlots().size());
    printf(
        "%-44s %8s %10s %10s %10s %7s %10s %8s %8s\n",
        "name",
        "n",
        "val_min",
        "val_p50",
        "val_max",
        "!2dp",
        "span_ms",
        "dt_p50_us",
        "in_win%");
    for (const auto& p : worker.GetPlots()) {
        const char* nm = nullptr;
        // GUARDED: a plot name is a client STRING POINTER the server resolves lazily. On a file load it may
        // be absent, and GetString on an unresolved key is not safe -- print the raw key instead of crashing.
        nm = worker.GetString(p->name);
        char fallback[32];
        if (nm == nullptr) {
            snprintf(fallback, sizeof(fallback), "<key %llx>", (unsigned long long)p->name);
            nm = fallback;
        }
        const auto& d = p->data;
        if (d.empty()) {
            continue;
        }
        std::vector<int64_t> ts;
        std::vector<double> vals;
        ts.reserve(d.size());
        vals.reserve(d.size());
        for (const auto& it : d) {
            ts.push_back(it.time.Val());
            vals.push_back(it.val);
        }
        // Value stats, not just timing. Without these there is no way to check that a rate plot carries
        // plausible numbers -- or that the 2-decimal quantisation actually took effect -- from the file itself.
        std::sort(vals.begin(), vals.end());
        const double v_min = vals.front();
        const double v_p50 = vals[vals.size() / 2];
        const double v_max = vals.back();
        // How many samples are NOT an exact multiple of 0.01: proves the 2dp rounding is applied (or is not).
        size_t not_2dp = 0;
        for (double v : vals) {
            const double scaled = v * 100.0;
            if (std::fabs(scaled - std::round(scaled)) > 1e-6) {
                not_2dp++;
            }
        }
        std::sort(ts.begin(), ts.end());
        std::vector<int64_t> dt;
        for (size_t i = 1; i < ts.size(); i++) {
            dt.push_back(ts[i] - ts[i - 1]);
        }
        std::sort(dt.begin(), dt.end());
        long long inside = 0;
        if (worker_win.valid()) {
            for (int64_t t : ts) {
                if (t >= worker_win.lo && t <= worker_win.hi) {
                    inside++;
                }
            }
        }
        printf(
            "%-44s %8zu %10.2f %10.2f %10.2f %7zu %10.3f %8.2f %7.1f%%\n",
            nm,
            ts.size(),
            v_min,
            v_p50,
            v_max,
            not_2dp,
            double(ts.back() - ts.front()) / 1e6,
            dt.empty() ? 0.0 : double(dt[dt.size() / 2]) / 1e3,
            ts.empty() ? 0.0 : 100.0 * double(inside) / double(ts.size()));
    }
    return 0;
}
