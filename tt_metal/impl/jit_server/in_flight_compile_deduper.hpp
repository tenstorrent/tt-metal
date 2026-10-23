// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

namespace tt::tt_metal::jit_server {

template <typename TValue>
class InFlightCompileDeduper {
public:
    template <typename TCallback>
    TValue run(const std::string& dedup_key, TCallback&& callback) {
        std::shared_future<TValue> shared_future;
        std::shared_ptr<std::promise<TValue>> owner_promise;

        {
            std::lock_guard<std::mutex> lock(in_flight_mutex_);
            auto it = in_flight_.find(dedup_key);
            if (it == in_flight_.end()) {
                owner_promise = std::make_shared<std::promise<TValue>>();
                shared_future = owner_promise->get_future().share();
                in_flight_[dedup_key] = shared_future;
            } else {
                shared_future = it->second;
            }
        }

        if (owner_promise != nullptr) {
            try {
                owner_promise->set_value(std::forward<TCallback>(callback)());
            } catch (...) {
                owner_promise->set_exception(std::current_exception());
            }

            std::lock_guard<std::mutex> lock(in_flight_mutex_);
            in_flight_.erase(dedup_key);
        }

        return shared_future.get();
    }

private:
    std::mutex in_flight_mutex_;
    std::unordered_map<std::string, std::shared_future<TValue>> in_flight_;
};

}  // namespace tt::tt_metal::jit_server
