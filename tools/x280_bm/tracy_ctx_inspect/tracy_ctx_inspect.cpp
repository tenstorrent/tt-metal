// SPDX-License-Identifier: Apache-2.0
//
// GUI-free Tracy trace inspector: dumps every GPU (tt_device) context in a .tracy
// file with its zone count, thread count, calibration flag and name. Use it to
// verify device-context creation without opening the Tracy GUI — e.g. to confirm
// the RT profiler produced one context per (chip,core) and how many zones landed
// in each. Build with build.sh (see that script for the fiddly capstone/ppqsort
// flag details).
#include <cstdio>
#include <memory>
#include "server/TracyFileRead.hpp"
#include "server/TracyWorker.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <trace.tracy>\n", argv[0]);
        return 1;
    }
    auto f = std::unique_ptr<tracy::FileRead>(tracy::FileRead::Open(argv[1]));
    if (!f) {
        fprintf(stderr, "cannot open %s\n", argv[1]);
        return 1;
    }
    try {
        // bgTasks=false: we only need gpuData, which is populated synchronously in the
        // constructor. Skipping background tasks avoids a needless worker thread.
        tracy::Worker worker(*f, tracy::EventType::All, false, false);
        const auto& gpu = worker.GetGpuData();
        printf("=== GPU contexts: %zu ===\n", (size_t)gpu.size());
        size_t idx = 0, total_zones = 0;
        for (auto* c : gpu) {
            const char* nm = c->name.Active() ? worker.GetString(c->name) : "(unnamed)";
            printf(
                "[%3zu] count=%-8llu threads=%-3zu hasCal=%d timeDiff=%lld period=%.3f name=%s\n",
                idx++,
                (unsigned long long)c->count,
                (size_t)c->threadData.size(),
                (int)c->hasCalibration,
                (long long)c->timeDiff,
                c->period,
                nm);
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
