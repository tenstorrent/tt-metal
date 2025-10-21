// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef _UTIL_PROFILER_
#define _UTIL_PROFILER_

#include <chrono>
#include <unordered_map>
#include <string>
#include <cassert>

class Profiler {
public:
    Profiler() = default;
    ~Profiler() = default;

    void start(const std::string& name) { timings[name] = std::chrono::high_resolution_clock::now(); }

    void stop(const std::string& name) {
        assert(timings.find(name) != timings.end());
        auto& start_time = timings[name];
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double>(end_time - start_time).count();
        if (results.find(name) != results.end()) {
            results[name] += duration;
        } else {
            results[name] = duration;
        }
    }

    double get(const std::string& name) const {
        auto it = results.find(name);
        if (it != results.end()) {
            return it->second;
        }
        return 0;
    }

private:
    std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> timings;
    std::unordered_map<std::string, double> results;
};

#endif  // _UTIL_PROFILER_
