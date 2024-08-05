// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <any>
#include <span>
#include <string_view>

#include "tt_metal/common/core_coord.h"
#include "tt_metal/impl/buffers/buffer.hpp"

namespace tt::tt_metal {

    class IGraphProcessor{
    public:
        IGraphProcessor() = default;

        virtual void track_allocate(tt::tt_metal::Buffer* buffer, bool bottom_up) {};

        virtual void track_deallocate(tt::tt_metal::Buffer* buffer) {};

        virtual void track_allocate_cb(const CoreRange &core_range, uint64_t addr, uint64_t size) {};

        virtual void track_deallocate_cb() {};

        virtual void track_begin_op(std::string_view function_name, std::span<std::any> input_parameters) {};

        virtual void track_end_op(const std::any& output_tensors) {};


        virtual void begin_capture() {};
        virtual std::string end_capture() {return "";};

        virtual ~IGraphProcessor() = default;

    };

    class IGraphHooks {
    public:
        IGraphHooks() = default;
        virtual bool hook_allocate(tt::tt_metal::Buffer* buffer, bool bottom_up) = 0;

        virtual bool hook_deallocate(tt::tt_metal::Buffer* buffer) = 0;

        virtual bool block_run_program() = 0;

        virtual ~IGraphHooks() = default;
    };

    class GraphTracker {
    private:
        std::vector<std::shared_ptr<IGraphProcessor>> processors;

        std::shared_ptr<IGraphHooks> hook;
    public:
        static GraphTracker& instance() {
            static GraphTracker tracker;
            return tracker;
        }

        size_t add_processor(const std::shared_ptr<IGraphProcessor>& processor);

        bool add_hook(const std::shared_ptr<IGraphHooks>& hook);

        void track_allocate(Buffer* buffer, bool bottom_up);

        void track_deallocate(Buffer* buffer);

        void track_allocate_cb(const CoreRange &core_range, uint64_t addr, uint64_t size);

        void track_deallocate_cb();

        template<class... Args>
        void track_begin_op(std::string_view function_name, Args&&... args) {
            if (processors.empty()) {
                return;
            }
            // we don't knpw anything in metal about ttnn classes so std::any is the only option here
            std::array<std::any, sizeof...(Args)>  params{std::any(std::ref(args))...};
            for (auto& it : processors) {
                it->track_begin_op(function_name, params);
            }
        }
        template<class ReturnType>
        void track_end_op(ReturnType&& output_tensors) {
            if (processors.empty()) {
                return;
            }
            for (auto& it : processors) {
                it->track_end_op(std::ref(output_tensors));
            }
        }

        bool hook_allocate(Buffer* buffer, bool bottom_up);

        bool hook_deallocate(Buffer* buffer);

        bool block_run_program();

        const std::vector<std::shared_ptr<IGraphProcessor>>& get_processors() const;

        const std::shared_ptr<IGraphHooks>& get_hooks() const;

        void clean();

    private:
        GraphTracker() = default;
        ~GraphTracker() = default;
    };
}
