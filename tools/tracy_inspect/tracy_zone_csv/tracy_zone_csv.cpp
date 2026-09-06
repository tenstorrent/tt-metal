// Dump every device (GPU) zone in a .tracy to CSV, in pre-order, so the nesting TREE can be rebuilt
// off-line. Columns: ctx,tid,risc,seq,depth,name,start,end
// Pre-order + depth is enough to recover parent/child: a row's parent is the nearest preceding row on the
// same (ctx,tid) with depth-1.
#include <cstdio>
#include <cstring>
#include <chrono>
#include <memory>
#include <thread>
#include "public/common/TracyTTDeviceData.hpp"
#include "server/TracyFileRead.hpp"
#include "server/TracyWorker.hpp"

static constexpr int kDepthCap = 256;
static FILE* g_out = nullptr;
static unsigned long long g_ctx = 0, g_tid = 0;
static int g_risc = 0;
static long long g_seq = 0;
static long long g_rows = 0;

static void walk(const tracy::Worker& w, const tracy::Vector<tracy::short_ptr<tracy::GpuEvent>>& vec, int depth) {
    if (depth >= kDepthCap) {
        return;
    }
    auto emit = [&](const tracy::GpuEvent& e) {
        const char* nm = w.GetZoneName(e);
        fprintf(
            g_out,
            "%llu,%llu,%d,%lld,%d,%s,%lld,%lld\n",
            g_ctx,
            g_tid,
            g_risc,
            g_seq++,
            depth,
            nm ? nm : "?",
            (long long)e.GpuStart(),
            (long long)e.GpuEnd());
        ++g_rows;
        if (e.Child() >= 0) {
            walk(w, w.GetGpuChildren(e.Child()), depth + 1);
        }
    };
    if (vec.is_magic()) {
        auto& mv = *reinterpret_cast<const tracy::Vector<tracy::GpuEvent>*>(&vec);
        for (auto& e : mv) {
            emit(e);
        }
    } else {
        for (auto& p : vec) {
            emit(*p);
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <in.tracy> <out.csv>\n", argv[0]);
        return 1;
    }
    auto f = std::unique_ptr<tracy::FileRead>(tracy::FileRead::Open(argv[1]));
    if (!f) {
        fprintf(stderr, "cannot open %s\n", argv[1]);
        return 1;
    }
    tracy::Worker worker(*f, tracy::EventType::All, true);
    while (!worker.AreSourceLocationZonesReady() || worker.GetGpuData().empty()) {
        if (!worker.IsBackgroundDone()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        } else {
            break;
        }
    }
    while (!worker.IsBackgroundDone()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    g_out = fopen(argv[2], "w");
    fprintf(g_out, "ctx,tid,risc,seq,depth,name,start,end\n");
    unsigned long long ci = 0;
    for (auto* c : worker.GetGpuData()) {
        for (const auto& td : c->threadData) {
            g_ctx = ci;
            g_tid = td.first;
            g_seq = 0;
            g_risc = (int)(td.first & ((1u << tracy::TTDeviceMarker::RISC_BIT_COUNT) - 1));
            walk(worker, td.second.timeline, 0);
        }
        ++ci;
    }
    fclose(g_out);
    fprintf(stderr, "wrote %lld zone rows from %zu contexts\n", g_rows, worker.GetGpuData().size());
    return 0;
}
