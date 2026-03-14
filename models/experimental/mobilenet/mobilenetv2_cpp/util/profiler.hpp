// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef MOBILENETV2_CPP_UTIL_PROFILER_HPP
#define MOBILENETV2_CPP_UTIL_PROFILER_HPP

#include <cassert>
#include <chrono>
#include <stdexcept>
#include <string>
#include <unordered_map>

class Profiler {
public:
    Profiler() = default;
    ~Profiler() = default;

    void start(const std::string& name) { timings[name] = std::chrono::high_resolution_clock::now(); }

    void stop(const std::string& name) {
        if (timings.find(name) == timings.end()) {
            throw std::runtime_error("Profiler::stop() called for " + name + " without corresponding start()");
        }
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

#endif  // MOBILENETV2_CPP_UTIL_PROFILER_HPP
