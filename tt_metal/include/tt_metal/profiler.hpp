#pragma once

#include <string>
#include <unordered_map>
#include <chrono>
#include <vector>
#include <mutex>

namespace tt::metal {

struct ProfileEvent {
    std::string name;
    double duration_ms;
    uint64_t timestamp_ns;
};

class Profiler {
public:
    static Profiler& get_instance() {
        static Profiler instance;
        return instance;
    }

    void start_event(const std::string& name) {
        std::lock_guard<std::mutex> lock(mutex_);
        events_.push_back({name, 0.0, get_now_ns()});
    }

    void stop_event(const std::string& name) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto now = get_now_ns();
        for (auto it = events_.rbegin(); it != events_.rend(); ++it) {
            if (it->name == name && it->duration_ms == 0.0) {
                it->duration_ms = static_cast<double>(now - it->timestamp_ns) / 1e6;
                break;
            }
        }
    }

    void track_accumulation_op(const std::string& op_name, double duration_ms) {
        std::lock_guard<std::mutex> lock(mutex_);
        accumulation_stats_[op_name] += duration_ms;
        accumulation_counts_[op_name]++;
    }

    void print_report() {
        std::lock_guard<std::mutex> lock(mutex_);
        printf("\n--- Accumulation Ops Tracker Report ---\n");
        for (auto const& [op, total_time] : accumulation_stats_) {
            double avg = total_time / accumulation_counts_[op];
            printf("Op: %s | Total Time: %.4f ms | Avg Time: %.4f ms | Calls: %llu\n",
                   op.c_str(), total_time, avg, accumulation_counts_[op]);
        }
        printf("---------------------------------------\n");
    }

private:
    Profiler() = default;
    uint64_t get_now_ns() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    }

    std::vector<ProfileEvent> events_;
    std::unordered_map<std::string, double> accumulation_stats_;
    std::unordered_map<std::string, uint64_t> accumulation_counts_;
    std::mutex mutex_;
};

} // namespace tt::metal
