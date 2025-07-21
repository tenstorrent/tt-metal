// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nlohmann/json.hpp>
#include <stdint.h>
#include <any>
#include <array>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/core_coord.hpp>

namespace tt {
namespace tt_metal {
class Buffer;
class IDevice;
}  // namespace tt_metal
}  // namespace tt

namespace tt::tt_metal {

class Program;

class IGraphProcessor {
public:
    enum class RunMode {
        NORMAL,      // running everything as is
        NO_DISPATCH  // don't do memory allocations and program runs.
    };

    IGraphProcessor() = default;

    virtual void track_allocate(const tt::tt_metal::Buffer* buffer) {};

    virtual void track_deallocate(tt::tt_metal::Buffer* buffer) {};

    virtual void track_allocate_cb(
        const CoreRangeSet& core_range_set,
        uint64_t addr,
        uint64_t size,
        bool is_globally_allocated,
        const IDevice* device) {};

    virtual void track_deallocate_cb(const IDevice* device) {};

    virtual void track_program(tt::tt_metal::Program* program, const IDevice* device) {};

    virtual void track_function_start(std::string_view function_name, std::span<std::any> input_parameters) {};

    virtual void track_function_end() {};
    virtual void track_function_end(const std::any& output_tensors) {};

    virtual void begin_capture(RunMode mode) {};

    virtual nlohmann::json end_capture() { return nullptr; };

    virtual ~IGraphProcessor() = default;
};

class IGraphHooks {
public:
    IGraphHooks() = default;
    virtual bool hook_allocate(const tt::tt_metal::Buffer* buffer) = 0;

    virtual bool hook_deallocate(tt::tt_metal::Buffer* buffer) = 0;

    virtual bool hook_program(Program* program) = 0;

    virtual ~IGraphHooks() = default;
};

class GraphTracker {
public:
    GraphTracker(const GraphTracker&) = delete;
    GraphTracker(GraphTracker&&) = delete;

    static GraphTracker& instance() {
        static GraphTracker tracker;
        return tracker;
    }

    bool is_enabled() const;

    void push_processor(const std::shared_ptr<IGraphProcessor>& processor);
    void pop_processor();

    bool add_hook(const std::shared_ptr<IGraphHooks>& hook);

    void track_allocate(const Buffer* buffer);

    void track_deallocate(Buffer* buffer);

    void track_allocate_cb(
        const CoreRangeSet& core_range_set,
        uint64_t addr,
        uint64_t size,
        bool is_globally_allocated,
        const IDevice* device);

    void track_deallocate_cb(const IDevice* device);

    void track_program(Program* program, const IDevice* device);

    template <class... Args>
    void track_function_start(std::string_view function_name, Args&&... args) {
        if (processors.empty()) {
            return;
        }
        std::array<std::any, sizeof...(Args)> params{std::any(std::ref(args))...};
        for (auto& it : processors) {
            it->track_function_start(function_name, params);
        }
    }

    // Track op that doesn't return anything
    void track_function_end() {
        if (processors.empty()) {
            return;
        }
        for (auto& it : processors) {
            it->track_function_end();
        }
    }

    template <class ReturnType>
    void track_function_end(ReturnType&& output_tensors) {
        if (processors.empty()) {
            return;
        }
        for (auto& it : processors) {
            it->track_function_end(std::ref(output_tensors));
        }
    }

    bool hook_allocate(const Buffer* buffer);

    bool hook_deallocate(Buffer* buffer);

    bool hook_program(tt::tt_metal::Program* program);

    const std::vector<std::shared_ptr<IGraphProcessor>>& get_processors() const;

    const std::shared_ptr<IGraphHooks>& get_hook() const;

    void clear();

    void clear_hook();

private:
    GraphTracker() = default;
    ~GraphTracker() = default;

    std::vector<std::shared_ptr<IGraphProcessor>> processors;

    std::shared_ptr<IGraphHooks> hook;

    std::mutex hooked_buffers_mutex;
    std::unordered_set<const Buffer*> hooked_buffers;
};
}  // namespace tt::tt_metal
