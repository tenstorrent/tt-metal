// SPDX-License-Identifier: Apache-2.0
//
// GUI-free Tracy trace inspector: dumps every GPU (tt_device) context in a .tracy
// file with its zone count, thread count, calibration flag and name. Use it to
// verify device-context creation without opening the Tracy GUI — e.g. to confirm
// the RT profiler produced one context per (chip,core) and how many zones landed
// in each. Build with build.sh (see that script for the fiddly capstone/ppqsort
// flag details).
#include <algorithm>
#include <chrono>
#include <vector>
#include <cstdio>
#include <cstring>
#include <memory>
#include <thread>
#include "server/TracyFileRead.hpp"
#include "server/TracyWorker.hpp"

// Max GPU-zone nesting depth under `vec`. A correctly-paired workload lane is depth<=2 (a real zone plus
// an occasional PRODUCER-STALL nested inside it); a lost-END / mis-paired lane grows a deep staircase.
static constexpr int kDepthCap = 256;  // stop descending here; a bugged lane can nest thousands deep
// A Tracy GPU zone vector is either "magic" (GpuEvent stored by value) or pointer-stored (short_ptr).
// The top-level timeline is magic; children may be either. Deref the wrong way = segfault, so branch on
// is_magic() exactly like the Tracy server does.
static int gpu_max_depth(
    const tracy::Worker& w, const tracy::Vector<tracy::short_ptr<tracy::GpuEvent>>& vec, int depth) {
    if (depth >= kDepthCap) {
        return depth;  // bound recursion so a pathological nest can't overflow the stack
    }
    int md = depth;
    auto descend = [&](const tracy::GpuEvent& e) {
        if (e.Child() >= 0) {
            int d = gpu_max_depth(w, w.GetGpuChildren(e.Child()), depth + 1);
            if (d > md) {
                md = d;
            }
        }
    };
    if (vec.is_magic()) {
        auto& mv = *reinterpret_cast<const tracy::Vector<tracy::GpuEvent>*>(&vec);
        for (auto& e : mv) {
            descend(e);
        }
    } else {
        for (auto& p : vec) {
            descend(*p);
        }
    }
    return md;
}

// Top-level time window of `vec`: earliest GpuStart and latest GpuEnd. Used for the cross-device
// ALIGNMENT check -- every device runs the same MeshWorkload, launched together, so once each device's
// anchor is applied their zone windows must overlap. An unanchored (or wrongly anchored) device sits
// billions of ns away, since the raw inter-device clock offset here is ~3.7 s.
static void ctx_window(const tracy::Vector<tracy::short_ptr<tracy::GpuEvent>>& vec, long long& lo, long long& hi) {
    auto take = [&](const tracy::GpuEvent& e) {
        const long long st = (long long)e.GpuStart();
        const long long en = (long long)e.GpuEnd();
        if (st > 0 && st < lo) { lo = st; }
        if (en > hi) { hi = en; }
    };
    if (vec.is_magic()) {
        auto& mv = *reinterpret_cast<const tracy::Vector<tracy::GpuEvent>*>(&vec);
        for (auto& e : mv) { take(e); }
    } else {
        for (auto& q : vec) { take(*q); }
    }
}

// ---- ETH SYNC CAUSALITY CHECK ----
// The sender's ETH_SYNC_RTT zone spans [t0,t2] on ITS clock; the peer's ETH_SYNC_ECHO marker is t1 on the
// PEER's clock. Each is rendered through its own device's anchor, so once both land on the common timeline
// the peer's t1 MUST fall inside the sender's [t0,t2] -- the message cannot be received before it was sent
// nor after the reply came back. This is the only check here that can FALSIFY the alignment: every other
// indicator is computed from the anchors and so agrees with them by construction.
struct EthIv {
    long long lo, hi;
};
static std::vector<EthIv> g_rtt;
static std::vector<long long> g_echo;

static void collect_rtt(const tracy::Worker& w, const tracy::Vector<tracy::short_ptr<tracy::GpuEvent>>& vec) {
    auto take = [&](const tracy::GpuEvent& e) {
        const char* nm = w.GetZoneName(e);
        if (nm != nullptr && strcmp(nm, "ETH_SYNC_RTT") == 0) {
            g_rtt.push_back(EthIv{(long long)e.GpuStart(), (long long)e.GpuEnd()});
        }
    };
    if (vec.is_magic()) {
        auto& mv = *reinterpret_cast<const tracy::Vector<tracy::GpuEvent>*>(&vec);
        for (auto& e : mv) { take(e); }
    } else {
        for (auto& q : vec) { take(*q); }
    }
}

// Print the zones along the DEEPEST nesting path (one per level): name + gpu start/end. Reveals whether a
// pathological nest is real zones ("T*_Zone*") or PRODUCER-STALL, and whether ENDs are missing (start==end / a
// parent that never closes before its children).
static void dump_deep_path(
    const tracy::Worker& w, const tracy::Vector<tracy::short_ptr<tracy::GpuEvent>>& vec, int depth) {
    if (depth >= kDepthCap) {
        return;
    }
    const tracy::GpuEvent* best = nullptr;
    int best_sub = -1;
    auto consider = [&](const tracy::GpuEvent& e) {
        int sub = (e.Child() >= 0) ? gpu_max_depth(w, w.GetGpuChildren(e.Child()), 0) : 0;
        if (sub > best_sub) {
            best_sub = sub;
            best = &e;
        }
    };
    if (vec.is_magic()) {
        auto& mv = *reinterpret_cast<const tracy::Vector<tracy::GpuEvent>*>(&vec);
        for (auto& e : mv) {
            consider(e);
        }
    } else {
        for (auto& p : vec) {
            consider(*p);
        }
    }
    if (!best) {
        return;
    }
    const char* nm = w.GetZoneName(*best);
    printf(
        "    d=%-3d %-24s gpuStart=%lld gpuEnd=%lld dur=%lld\n",
        depth,
        nm ? nm : "(null)",
        (long long)best->GpuStart(),
        (long long)best->GpuEnd(),
        (long long)(best->GpuEnd() - best->GpuStart()));
    if (best->Child() >= 0) {
        dump_deep_path(w, w.GetGpuChildren(best->Child()), depth + 1);
    }
}

// Print the first N top-level zones' name + duration (in device ticks/cycles) — to eyeball that the
// per-zone spin durations landed (Zone0..Zone9). GUI ns = ticks * context.period.
static void sample_top_zones(
    const tracy::Worker& w, const tracy::Vector<tracy::short_ptr<tracy::GpuEvent>>& vec, int n) {
    int printed = 0;
    auto show = [&](const tracy::GpuEvent& e) {
        if (printed >= n) {
            return;
        }
        const char* nm = w.GetZoneName(e);
        printf(
            "      %-16s start=%lld end=%lld dur=%lld ticks\n",
            nm ? nm : "(null)",
            (long long)e.GpuStart(),
            (long long)e.GpuEnd(),
            (long long)(e.GpuEnd() - e.GpuStart()));
        printed++;
    };
    if (vec.is_magic()) {
        auto& mv = *reinterpret_cast<const tracy::Vector<tracy::GpuEvent>*>(&vec);
        for (auto& e : mv) {
            if (printed >= n) {
                break;
            }
            show(e);
        }
    } else {
        for (auto& p : vec) {
            if (printed >= n) {
                break;
            }
            show(*p);
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <trace.tracy>\n", argv[0]);
        return 1;
    }
    setvbuf(stdout, nullptr, _IONBF, 0);  // unbuffered: partial output survives if the walk aborts
    auto f = std::unique_ptr<tracy::FileRead>(tracy::FileRead::Open(argv[1]));
    if (!f) {
        fprintf(stderr, "cannot open %s\n", argv[1]);
        return 1;
    }
    try {
        // bgTasks=TRUE + wait for completion: the nested-zone children tree (GetGpuChildren) is built by
        // the background pass, so walking zone depth REQUIRES it (bgTasks=false leaves children unbuilt ->
        // GetGpuChildren returns garbage -> segfault). Context counts alone would be fine with bgTasks=false.
        tracy::Worker worker(*f, tracy::EventType::All, true, false);
        while (!worker.IsBackgroundDone()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        const auto& gpu = worker.GetGpuData();
        printf("=== GPU contexts: %zu ===\n", (size_t)gpu.size());
        size_t idx = 0, total_zones = 0, total_markers = 0;
        long long dev_lo[64], dev_hi[64];
        bool dev_seen[64] = {false};
        // ETH rows tracked SEPARATELY from worker rows. The causality check compares two eth cores against
        // each other, so a shift common to both (eth domain vs Tensix domain) slides straight past it -- it
        // did, by 25 minutes. Comparing the eth window against the WORKLOAD window is what catches that.
        long long eth_lo = 0x7fffffffffffffffLL, eth_hi = -1;
        long long wrk_lo = 0x7fffffffffffffffLL, wrk_hi = -1;
        for (int i = 0; i < 64; ++i) { dev_lo[i] = 0x7fffffffffffffffLL; dev_hi[i] = -1; }
        for (auto* c : gpu) {
            const char* nm = c->name.Active() ? worker.GetString(c->name) : "(unnamed)";
            // Per-thread (RISC) max nesting depth; flag any thread deeper than 3 (staircase bug).
            int ctx_max_depth = 0;
            char depth_note[256] = {0};
            for (const auto& td : c->threadData) {
                int d = gpu_max_depth(worker, td.second.timeline, 0);
                if (d > ctx_max_depth) {
                    ctx_max_depth = d;
                }
                if (d > 3) {
                    char frag[48];
                    snprintf(frag, sizeof(frag), " tid=%llu:d=%d", (unsigned long long)td.first, d);
                    strncat(depth_note, frag, sizeof(depth_note) - strlen(depth_note) - 1);
                }
            }
            long long win_lo = 0x7fffffffffffffffLL, win_hi = -1;
            for (const auto& td : c->threadData) {
                ctx_window(td.second.timeline, win_lo, win_hi);
                collect_rtt(worker, td.second.timeline);
                for (const auto& m : td.second.markers) {
                    const auto& sl = worker.GetSourceLocation(m->srcloc);
                    const char* mn = worker.GetString(sl.name);
                    if (mn != nullptr && strcmp(mn, "ETH_SYNC_ECHO") == 0) {
                        g_echo.push_back((long long)m->gpuTime);
                    }
                }
            }
            if (win_hi > 0) {
                const bool is_eth = strstr(nm, " ETH ") != nullptr;
                long long& lo = is_eth ? eth_lo : wrk_lo;
                long long& hi = is_eth ? eth_hi : wrk_hi;
                if (win_lo < lo) { lo = win_lo; }
                if (win_hi > hi) { hi = win_hi; }
            }
            int devno = -1;
            if (sscanf(nm, "Device: %d", &devno) == 1 && devno >= 0 && devno < 64 && win_hi > 0) {
                dev_seen[devno] = true;
                if (win_lo < dev_lo[devno]) { dev_lo[devno] = win_lo; }
                if (win_hi > dev_hi[devno]) { dev_hi[devno] = win_hi; }
            }
            printf(
                "[%3zu] count=%-8llu threads=%-3zu hasCal=%d maxdepth=%-3d period=%.3f name=%s%s\n",
                idx++,
                (unsigned long long)c->count,
                (size_t)c->threadData.size(),
                (int)c->hasCalibration,
                ctx_max_depth,
                c->period,
                nm,
                depth_note[0] ? depth_note : "");
            if (ctx_max_depth > 3) {
                for (const auto& td : c->threadData) {
                    if (gpu_max_depth(worker, td.second.timeline, 0) > 3) {
                        printf("  --- deepest path, tid=%llu ---\n", (unsigned long long)td.first);
                        dump_deep_path(worker, td.second.timeline, 0);
                    }
                }
            }
            // --all-threads: per-tid top-level zone list (name + gpu start/end), bounded. For
            // diagnosing cross-RISC timeline skew: prints the actual begin/end ticks per thread.
            if (getenv("CTX_ALL_THREADS") != nullptr) {
                for (const auto& td : c->threadData) {
                    printf("  --- top-level zones, tid=%llu ---\n", (unsigned long long)td.first);
                    sample_top_zones(worker, td.second.timeline, 16);
                }
            }
            static bool sampled = false;
            if (!sampled && c->threadData.size() >= 5) {
                sampled = true;
                printf(
                    "  --- sample top-level zone durations, tid=%llu ---\n",
                    (unsigned long long)c->threadData.begin()->first);
                sample_top_zones(worker, c->threadData.begin()->second.timeline, 12);
            }
            // Device markers (point-in-time events: TS_EVENT / TS_DATA) live per lane alongside the zones.
            size_t ctx_markers = 0;
            for (const auto& td : c->threadData) {
                ctx_markers += td.second.markers.size();
            }
            if (ctx_markers != 0) {
                printf("  markers: %zu\n", ctx_markers);
                int shown = 0;
                for (const auto& td : c->threadData) {
                    for (const auto& m : td.second.markers) {
                        if (shown++ >= 4) {
                            break;
                        }
                        const auto& sl = worker.GetSourceLocation(m->srcloc);
                        printf(
                            "    tid=%llu t=%lld type=%u name=%s meta=[%s]\n",
                            (unsigned long long)td.first,
                            (long long)m->gpuTime,
                            (unsigned)m->markerType,
                            worker.GetString(sl.name),
                            m->meta.Active() ? worker.GetString(m->meta) : "");
                    }
                    if (shown >= 4) {
                        break;
                    }
                }
            }
            total_markers += ctx_markers;
            total_zones += c->count;
        }
        // ---- ETH DOMAIN vs WORKLOAD ----
        if (eth_hi > 0 && wrk_hi > 0) {
            // The sync runs BEFORE the workload, so a correctly anchored eth window ends before it. The
            // gap is legitimately SECONDS -- device bring-up sits between them (measured 7.3 s) -- so only
            // a negative gap or a wildly large one indicates a wrong origin. The 25-minute miss this check
            // exists to catch was 3 orders of magnitude past bring-up.
            const long long gap = wrk_lo - eth_hi;
            printf("=== eth vs workload (ns) ===\n");
            printf("  eth rows      [%lld .. %lld]\n", eth_lo, eth_hi);
            printf("  workload rows [%lld .. %lld]\n", wrk_lo, wrk_hi);
            printf("  gap eth_end -> workload_start: %lld ns (%.3f ms)%s\n", gap, (double)gap / 1e6,
                   (gap < -1000000LL || gap > 120000000000LL)
                       ? "   [!! eth rows are NOT on the workload's timeline -- wrong anchor origin]"
                       : "   [plausible: sync runs just before the workload]");
        }

        // ---- ETH SYNC CAUSALITY ----
        if (!g_echo.empty() && !g_rtt.empty()) {
            std::sort(g_rtt.begin(), g_rtt.end(), [](const EthIv& a, const EthIv& b) { return a.lo < b.lo; });
            size_t inside = 0, outside = 0;
            long long worst_out = 0;      // furthest an echo sits outside any round trip
            long long min_slack = 0x7fffffffffffffffLL;  // tightest margin of an echo inside its round trip
            for (const long long t : g_echo) {
                // First interval starting after t, then scan back: intervals are short and nearly disjoint.
                auto it = std::upper_bound(
                    g_rtt.begin(), g_rtt.end(), t, [](long long v, const EthIv& iv) { return v < iv.lo; });
                bool ok = false;
                long long best_dist = 0x7fffffffffffffffLL;
                for (auto j = g_rtt.begin() == it ? it : std::prev(it); ; ) {
                    if (t >= j->lo && t <= j->hi) {
                        ok = true;
                        const long long slack = std::min(t - j->lo, j->hi - t);
                        if (slack < min_slack) { min_slack = slack; }
                        break;
                    }
                    const long long d = (t < j->lo) ? (j->lo - t) : (t - j->hi);
                    if (d < best_dist) { best_dist = d; }
                    if (j == g_rtt.begin()) { break; }
                    --j;
                    if (t - j->hi > 1000000) { break; }  // 1 ms back is far beyond any 0.64 us round trip
                }
                if (ok) {
                    ++inside;
                } else {
                    ++outside;
                    if (best_dist > worst_out) { worst_out = best_dist; }
                }
            }
            printf("=== eth sync causality: %zu echo(es) vs %zu round trip(s) ===\n", g_echo.size(), g_rtt.size());
            printf("  INSIDE  %zu   OUTSIDE %zu%s\n", inside, outside,
                   outside == 0 ? "   [causality holds -- alignment is consistent with the raw samples]"
                                : "   [!! an echo outside its round trip DISPROVES the alignment]");
            if (inside != 0) {
                printf("  tightest margin inside: %lld ns\n", min_slack);
            }
            if (outside != 0) {
                printf("  worst excursion outside: %lld ns\n", worst_out);
            }
        }

        // ---- CROSS-DEVICE ALIGNMENT ----
        // Every device ran the same MeshWorkload, dispatched together, so their zone windows must
        // overlap on the common timeline. Report each device's window relative to the earliest one.
        // Scale: the RAW inter-device clock offset is ~3.7e9 ns, so a device whose anchor was not
        // applied would show a delta in the billions; real dispatch skew is sub-millisecond.
        {
            long long base = 0x7fffffffffffffffLL;
            for (int d = 0; d < 64; ++d) {
                if (dev_seen[d] && dev_lo[d] < base) { base = dev_lo[d]; }
            }
            if (base != 0x7fffffffffffffffLL) {
                printf("=== cross-device alignment (ns, relative to earliest device) ===\n");
                long long worst = 0;
                for (int d = 0; d < 64; ++d) {
                    if (!dev_seen[d]) { continue; }
                    const long long off = dev_lo[d] - base;
                    if (off > worst) { worst = off; }
                    printf("  device %d: start %+12lld ns   span %12lld ns\n",
                           d, off, dev_hi[d] - dev_lo[d]);
                }
                printf("  WORST START SKEW: %lld ns (%.3f ms)\n", worst, (double)worst / 1e6);
            }
        }
        printf("=== total gpu zones across contexts: %zu ===\n", total_zones);
        printf(
            "=== total device markers across contexts: %zu (worker: %llu) ===\n",
            total_markers,
            (unsigned long long)worker.GetGpuMarkerCount());
    } catch (const std::exception& e) {
        fprintf(stderr, "EXCEPTION: %s\n", e.what());
        return 2;
    } catch (...) {
        fprintf(stderr, "UNKNOWN EXCEPTION (likely trace version/format mismatch)\n");
        return 3;
    }
    return 0;
}
