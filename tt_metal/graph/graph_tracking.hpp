// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <any>
#include <span>
#include <string_view>

#include "tt_metal/common/core_coord.hpp"
#include "tt_metal/impl/buffers/buffer.hpp"

namespace tt::tt_metal {
inline namespace v0 {

    class Program;

}  // namespace v0

    class IGraphProcessor{
    public:
        enum class RunMode {
            NORMAL, // running everything as is
            NO_DISPATCH // don't do memory allocations and program runs.
        };

        IGraphProcessor() = default;

        virtual void track_allocate(tt::tt_metal::Buffer* buffer, bool bottom_up) {};

        virtual void track_deallocate(tt::tt_metal::Buffer* buffer) {};

        virtual void track_allocate_cb(const CoreRangeSet &core_range_set, uint64_t addr, uint64_t size) {};

        virtual void track_deallocate_cb() {};

        virtual void track_program(tt::tt_metal::Program* program) {};

        virtual void track_function_start(std::string_view function_name, std::span<std::any> input_parameters) {};

        virtual void track_function_end() {};
        virtual void track_function_end(const std::any& output_tensors) {};

        virtual void begin_capture(RunMode mode) {};

        virtual nlohmann::json end_capture() {return nullptr;};

        virtual ~IGraphProcessor() = default;

    };

    class IGraphHooks {
    public:
        IGraphHooks() = default;
        virtual bool hook_allocate(tt::tt_metal::Buffer* buffer, bool bottom_up) = 0;

        virtual bool hook_deallocate(tt::tt_metal::Buffer* buffer) = 0;

        virtual bool hook_program(Program* program) = 0;

        virtual ~IGraphHooks() = default;
    };

    class GraphTracker {
    public:
        static GraphTracker& instance() {
            static GraphTracker tracker;
            return tracker;
        }

        bool is_enabled() const;

        void push_processor(const std::shared_ptr<IGraphProcessor>& processor);
        void pop_processor();

        bool add_hook(const std::shared_ptr<IGraphHooks>& hook);

        void track_allocate(Buffer* buffer, bool bottom_up);

        void track_deallocate(Buffer* buffer);

        void track_allocate_cb(const CoreRangeSet &core_range_set, uint64_t addr, uint64_t size);

        void track_deallocate_cb();

        void track_program(Program* program);

        template<class... Args>
        void track_function_start(std::string_view function_name, Args&&... args) {
            if (processors.empty()) {
                return;
            }
            std::array<std::any, sizeof...(Args)>  params{std::any(std::ref(args))...};
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

        template<class ReturnType>
        void track_function_end(ReturnType&& output_tensors) {
            if (processors.empty()) {
                return;
            }
            for (auto& it : processors) {
                it->track_function_end(std::ref(output_tensors));
            }
        }

        bool hook_allocate(Buffer* buffer, bool bottom_up);

        bool hook_deallocate(Buffer* buffer);

        bool hook_program(tt::tt_metal::Program* program);

        const std::vector<std::shared_ptr<IGraphProcessor>>& get_processors() const;

        const std::shared_ptr<IGraphHooks>& get_hook() const;

        void clear();

        void clear_hook();

       private:
        GraphTracker() = default;
        ~GraphTracker() = default;
        GraphTracker(const GraphTracker&) = delete;
        GraphTracker(GraphTracker&&) = delete;

        std::vector<std::shared_ptr<IGraphProcessor>> processors;

        std::shared_ptr<IGraphHooks> hook;

    };
}
