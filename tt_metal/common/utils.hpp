// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <sstream>
#include <thread>
#include <mutex>
#include <vector>
#include <map>

using std::string;

namespace tt {
namespace utils {
bool run_command(const string& cmd, const string& log_file, const bool verbose);
void create_file(const string& file_path_str);
const std::string& get_reports_dir();

// Ripped out of boost for std::size_t so as to not pull in bulky boost dependencies
template <typename T>
void hash_combine(std::size_t& seed, const T& value) {
    std::hash<T> hasher;
    seed ^= hasher(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

struct DefinesHash {
    DefinesHash() {}
    size_t operator()(const std::map<std::string, std::string>& c_defines) const;
};

inline std::vector<std::string> strsplit(std::string input, char delimiter) {
    std::vector<std::string> result = {};
    std::stringstream ss(input);

    while (ss.good()) {
        std::string substr;
        getline(ss, substr, delimiter);
        result.push_back(substr);
    }
    return result;
}

// A simple thread manager that joins all threads and rethrows the first caught exception
// instead of letting the program terminate.
class ThreadManager {
public:
    template <typename Func, typename... Args>
    void start(Func&& func, Args&&... args) {
        threads.emplace_back(std::thread([=]() {
            try {
                func(args...);
            } catch (...) {
                std::lock_guard<std::mutex> lock(exceptionMutex);
                exceptions.push_back(std::current_exception());
            }
        }));
    }

    void join_and_rethrow() {
        for (auto& thread : threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }

        if (!exceptions.empty()) {
            std::rethrow_exception(exceptions.front());
        }
    }

private:
    std::vector<std::thread> threads;
    std::vector<std::exception_ptr> exceptions;
    std::mutex exceptionMutex;
};

template <typename E, std::enable_if_t<std::is_enum<E>::value, bool> = true>
auto underlying_type(const E& e) {
    return static_cast<typename std::underlying_type<E>::type>(e);
}
}  // namespace utils
}  // namespace tt
