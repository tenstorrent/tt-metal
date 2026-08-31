// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <any>
#include <exception>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <tt-metalium/graph_tracking.hpp>

namespace tt::tt_metal::internal {

// Internal, unstable API - read the stability/usage conditions in tt_metal/api/internal/README.md
// before depending on anything here.
//
// Transport between ScopedTrackedFunction (and unwind_open_functions) and TTNN's GraphProcessor.
// Abort rides the existing IGraphProcessor::track_function_end(const std::any&) slot so the
// stable processor vtable is not extended. Not part of the stable tt-metalium contract.

struct GraphFunctionAbort {
    std::string reason;
    bool unwind_all = false;
};

// RAII pairing of GraphTracker::track_function_start with its end.
//
// The two calls used to be written as plain statements around the tracked body, so an exception
// thrown in between skipped the end entirely. The processor was then left holding the dead scope:
// the next end closed the wrong function, and every event after it was recorded one level too
// deep, nested under an operation that never finished. That corrupts the whole remainder of a
// capture which outlives the failure, such as the per-test capture behind the TTNN Visualizer
// report.
//
// The destructor closes the scope on both paths and tells them apart, so a failing operation is
// reported as aborted instead of silently unbalancing the trace. Call end() on the success path to
// report the operation's output; anything left unclosed is finished by the destructor.
class [[nodiscard]] ScopedTrackedFunction {
public:
    template <class... Args>
    explicit ScopedTrackedFunction(std::string_view function_name, Args&&... args) :
        entry_uncaught_exceptions_(std::uncaught_exceptions()),
        // Snapshot the processors that will receive the start. Holding the shared_ptrs
        // keeps that set alive if it is popped before this guard ends.
        started_processors_(GraphTracker::instance().get_processors()) {
        GraphTracker::instance().track_function_start(function_name, std::forward<Args>(args)...);
    }

    template <class ReturnType>
    void end(ReturnType& output_tensors) {
        if (std::exchange(ended_, true) || started_processors_.empty()) {
            return;
        }
        for (auto& processor : started_processors_) {
            processor->track_function_end(std::ref(output_tensors));
        }
    }

    void end() {
        if (std::exchange(ended_, true) || started_processors_.empty()) {
            return;
        }
        for (auto& processor : started_processors_) {
            processor->track_function_end();
        }
    }

    // Closes the scope as failed. Call from a catch block, where the message is in reach; the
    // destructor cannot supply one (see below).
    void abort(std::string_view reason) {
        if (std::exchange(ended_, true) || started_processors_.empty()) {
            return;
        }
        for (auto& processor : started_processors_) {
            processor->track_function_end(std::any(GraphFunctionAbort{std::string(reason), false}));
        }
    }

    ~ScopedTrackedFunction() {
        if (ended_ || started_processors_.empty()) {
            return;
        }
        // Destructors must not let an exception escape, least of all while one is already unwinding.
        try {
            if (std::uncaught_exceptions() > entry_uncaught_exceptions_) {
                // No message: std::current_exception() only reports an exception that a handler has
                // begun handling, and stack unwinding runs destructors before any handler is
                // entered, so there is nothing to read here. Callers that want the text must use
                // abort() from a catch block.
                for (auto& processor : started_processors_) {
                    processor->track_function_end(std::any(GraphFunctionAbort{{}, false}));
                }
            } else {
                for (auto& processor : started_processors_) {
                    processor->track_function_end();
                }
            }
        } catch (...) {  // NOLINT(bugprone-empty-catch)
        }
    }

    ScopedTrackedFunction(const ScopedTrackedFunction&) = delete;
    ScopedTrackedFunction(ScopedTrackedFunction&&) = delete;
    ScopedTrackedFunction& operator=(const ScopedTrackedFunction&) = delete;
    ScopedTrackedFunction& operator=(ScopedTrackedFunction&&) = delete;

private:
    int entry_uncaught_exceptions_;
    std::vector<std::shared_ptr<IGraphProcessor>> started_processors_;
    bool ended_ = false;
};

// Close every scope the processors of this thread still hold open, marking each aborted.
void unwind_open_functions(std::string_view reason);

}  // namespace tt::tt_metal::internal
