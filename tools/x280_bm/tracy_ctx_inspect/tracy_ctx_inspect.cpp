// SPDX-License-Identifier: Apache-2.0
//
// GUI-free Tracy trace inspector: dumps every GPU (tt_device) context in a .tracy
// file with its zone count, thread count, calibration flag and name. Use it to
// verify device-context creation without opening the Tracy GUI — e.g. to confirm
// the RT profiler produced one context per (chip,core) and how many zones landed
// in each. Build with build.sh (see that script for the fiddly capstone/ppqsort
// flag details).
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <thread>
#include "server/TracyFileRead.hpp"
#include "server/TracyWorker.hpp"

// Max GPU-zone nesting depth under `vec`. A correctly-paired workload lane is depth<=2 (a real zone plus
// an occasional X280-STALL nested inside it); a lost-END / mis-paired lane grows a deep staircase.
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

// Print the zones along the DEEPEST nesting path (one per level): name + gpu start/end. Reveals whether a
// pathological nest is real zones ("T*_Zone*") or X280-STALL, and whether ENDs are missing (start==end / a
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
        size_t idx = 0, total_zones = 0;
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
            total_zones += c->count;
        }
        printf("=== total gpu zones across contexts: %zu ===\n", total_zones);
    } catch (const std::exception& e) {
        fprintf(stderr, "EXCEPTION: %s\n", e.what());
        return 2;
    } catch (...) {
        fprintf(stderr, "UNKNOWN EXCEPTION (likely trace version/format mismatch)\n");
        return 3;
    }
    return 0;
}
