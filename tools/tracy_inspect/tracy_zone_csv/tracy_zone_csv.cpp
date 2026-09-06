// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// GUI-free inspector for the device (GPU) zones in a .tracy capture. Prints one line per device context (zones,
// threads, markers, deepest nesting, calibration, name) so a test can check what the Tracy sink produced without the
// GUI; given an output path it also writes every zone in pre-order with its depth, so the nesting tree can be rebuilt
// offline. Columns: ctx,tid,risc,seq,depth,name,start,end; a row's parent is the nearest preceding row on the same
// (ctx,tid) with depth-1. A correctly paired lane is depth <= 2 (a zone plus a PROFILER-STALL nested in it); a lost
// end grows a deep staircase.
//
//   usage: tracy_zone_csv <in.tracy> [out.csv]
#include <chrono>
#include <cstdio>
#include <memory>
#include <thread>

#include "public/common/TracyTTDeviceData.hpp"
#include "server/TracyFileRead.hpp"
#include "server/TracyWorker.hpp"

namespace {

constexpr int kDepthCap = 256;

struct Cursor {
    FILE* out = nullptr;
    unsigned long long ctx = 0, tid = 0;
    int risc = 0;
    long long seq = 0, rows = 0;
};

// Walks one thread's zone tree in pre-order, emitting rows when `out` is set; returns the deepest nesting seen.
// A Tracy GPU zone vector is either "magic" (events by value) or pointer-stored; deref the wrong way and it faults.
int walk(const tracy::Worker& w, const tracy::Vector<tracy::short_ptr<tracy::GpuEvent>>& vec, int depth, Cursor& c) {
    if (depth >= kDepthCap) {
        return depth;
    }
    int deepest = depth;
    auto visit = [&](const tracy::GpuEvent& e) {
        if (c.out != nullptr) {
            const char* nm = w.GetZoneName(e);
            fprintf(
                c.out,
                "%llu,%llu,%d,%lld,%d,%s,%lld,%lld\n",
                c.ctx,
                c.tid,
                c.risc,
                c.seq++,
                depth,
                nm ? nm : "?",
                (long long)e.GpuStart(),
                (long long)e.GpuEnd());
            c.rows++;
        }
        int d = depth + 1;
        if (e.Child() >= 0) {
            d = walk(w, w.GetGpuChildren(e.Child()), depth + 1, c);
        }
        deepest = d > deepest ? d : deepest;
    };
    if (vec.is_magic()) {
        for (auto& e : *reinterpret_cast<const tracy::Vector<tracy::GpuEvent>*>(&vec)) {
            visit(e);
        }
    } else {
        for (auto& p : vec) {
            visit(*p);
        }
    }
    return deepest;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <in.tracy> [out.csv]\n", argv[0]);
        return 1;
    }
    auto f = std::unique_ptr<tracy::FileRead>(tracy::FileRead::Open(argv[1]));
    if (!f) {
        fprintf(stderr, "cannot open %s\n", argv[1]);
        return 1;
    }
    // The zone children tree is built by the worker's background pass; walking depth before it is done faults.
    tracy::Worker worker(*f, tracy::EventType::All, true);
    while (!worker.IsBackgroundDone()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    Cursor c;
    if (argc > 2) {
        c.out = fopen(argv[2], "w");
        if (c.out == nullptr) {
            fprintf(stderr, "cannot write %s\n", argv[2]);
            return 1;
        }
        fprintf(c.out, "ctx,tid,risc,seq,depth,name,start,end\n");
    }
    const auto& gpu = worker.GetGpuData();
    printf("contexts %zu\n", (size_t)gpu.size());
    unsigned long long total_zones = 0, total_markers = 0;
    for (auto* ctx : gpu) {
        int deepest = 0;
        size_t markers = 0;
        for (const auto& td : ctx->threadData) {
            c.tid = td.first;
            c.seq = 0;
            c.risc = (int)(td.first & ((1u << tracy::TTDeviceMarker::RISC_BIT_COUNT) - 1));
            const int d = walk(worker, td.second.timeline, 0, c);
            deepest = d > deepest ? d : deepest;
            markers += td.second.markers.size();
        }
        printf(
            "ctx %llu zones=%llu threads=%zu markers=%zu maxdepth=%d calibrated=%d name=%s\n",
            c.ctx,
            (unsigned long long)ctx->count,
            (size_t)ctx->threadData.size(),
            markers,
            deepest,
            (int)ctx->hasCalibration,
            ctx->name.Active() ? worker.GetString(ctx->name) : "(unnamed)");
        total_zones += ctx->count;
        total_markers += markers;
        c.ctx++;
    }
    printf("total zones=%llu markers=%llu\n", total_zones, total_markers);
    if (c.out != nullptr) {
        fclose(c.out);
        fprintf(stderr, "wrote %lld zone rows to %s\n", c.rows, argv[2]);
    }
    return 0;
}
