// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <string_view>

#include "tt_metal/common/core_coord.h"
#include "tt_metal/impl/buffers/buffer.hpp"

/*
namespace {
template <typename T>
struct is_gatherable { static constexpr bool value = false; };

template <>
struct is_gatherable<ttnn::Tensor> {
  static constexpr bool value = true;
};

template <>
struct is_gatherable<std::vector<ttnn::Tensor>> {
  static constexpr bool value = true;
};

template <>
struct is_gatherable<std::vector<std::optional<ttnn::Tensor>>> {
  static constexpr bool value = true;
};

template <typename... Args>
std::vector<std::optional<ttnn::Tensor>> gather_tensors(Args&&... args) {
  std::vector<std::optional<ttnn::Tensor>> result;
  //(static_cast<void>(is_gatherable(args...) && (result.push_back(&args...), true))..., 0);
  return result;
}
}
*/
namespace tt::tt_metal {

    //class
    class GraphTracker {
    private:
        int depth = 0;
    public:
        static GraphTracker& instance() {
            static GraphTracker tracker;
            return tracker;
        }
        void track_allocate(Buffer* buffer, bool bottom_up) {
            auto alloc_id = reinterpret_cast<std::uintptr_t>(buffer);
            tt::log_info("{}Called Allocate id: {}, size: {}, bottom_up: {}", std::string(depth,'-'), alloc_id, buffer->size(), bottom_up);
        }

        void track_deallocate(Buffer* buffer) {
            auto alloc_id = reinterpret_cast<std::uintptr_t>(buffer);
            tt::log_info("{}Called Deallocate id: {}", std::string(depth,'-'), alloc_id);
        }

        void track_allocate_cb(const CoreRange &core_range, uint64_t addr, uint64_t size) {
            tt::log_info( "{}Called allocate circular buffer rangeX: {}:{}, rangeY: {}:{} , addr: {}, size: {}", std::string(depth,'-'), core_range.start.x, core_range.end.x, core_range.start.y, core_range.end.y, addr, size);
        }

        void track_deallocate_cb() {
            tt::log_debug("{}Called deallocate circular buffers", std::string(depth,'-'));
        }

        template<class... Args>
        void track_begin_op(std::string_view function_name, Args&&... args) {
            tt::log_info( "{}Called Begin Op:{}", std::string(depth,'-'), function_name);
            //auto all_tensors = gather_tensors(args...);
            depth++;
        }
        template<class ReturnType>
        void track_end_op(ReturnType&& output_tensors) {
            tt::log_info( "Called End Op");
            depth--;
        }

        bool hook_allocate(Buffer* buffer, bool bottom_up) {
            return false;
        }

        bool hook_deallocate(Buffer* buffer) {
            return false;
        }

        bool block_run_program() {
            return true;
        }
    private:
        GraphTracker() = default;
        ~GraphTracker() = default;
    };
}
