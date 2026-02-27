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
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/mesh_buffer.hpp>

namespace tt::tt_metal {
class Buffer;
class IDevice;
}  // namespace tt::tt_metal

namespace tt::tt_metal {

class Program;

struct TrackedArgument {
    std::any value;
    std::string (*to_string_fn)(const std::any&);
};

// Serialization helpers for graph argument tracking.
// The templates below rely on symbols from tt_stl/reflection.hpp (ttsl::reflection,
// ttsl::is_specialization_v, ttsl::concepts::Reflectable, reflect::*).  These are
// intentionally NOT included here to avoid pulling heavyweight headers into every
// translation unit that includes graph_tracking.hpp.  The symbols are resolved at
// template instantiation time via transitive includes in the calling TUs.
namespace graph_detail {

template <typename MemberT>
void serialize_member(std::ostringstream& oss, const MemberT& member) {
    if constexpr (std::is_enum_v<MemberT>) {
        ttsl::reflection::operator<<(oss, member);
    } else if constexpr (ttsl::is_specialization_v<MemberT, std::vector>) {
        oss << "{";
        for (size_t i = 0; i < member.size(); ++i) {
            if (i > 0) {
                oss << ", ";
            }
            serialize_member(oss, member[i]);
        }
        oss << "}";
    } else if constexpr (ttsl::is_specialization_v<MemberT, std::optional>) {
        if (member.has_value()) {
            serialize_member(oss, member.value());
        } else {
            oss << "std::nullopt";
        }
    } else if constexpr (ttsl::is_specialization_v<MemberT, std::reference_wrapper>) {
        serialize_member(oss, member.get());
    } else if constexpr (ttsl::is_specialization_v<MemberT, std::pair>) {
        oss << "{";
        serialize_member(oss, member.first);
        oss << ", ";
        serialize_member(oss, member.second);
        oss << "}";
    } else if constexpr (requires { oss << member; }) {
        oss << member;
    } else if constexpr (ttsl::concepts::Reflectable<MemberT>) {
        oss << reflect::type_name<MemberT>() << "(";
        reflect::for_each(
            [&oss, &member](auto I) {
                if constexpr (I > 0) {
                    oss << ", ";
                }
                serialize_member(oss, reflect::get<I>(member));
            },
            member);
        oss << ")";
    } else {
        oss << "<" << reflect::type_name<MemberT>() << ">";
    }
}

}  // namespace graph_detail

template <typename T>
std::string serialize_tracked_arg(const std::any& a) {
    std::ostringstream oss;
    const auto& ref = std::any_cast<const std::reference_wrapper<T>&>(a);
    const auto& val = ref.get();
    using CleanT = std::remove_cv_t<T>;

    // Guard against incomplete types (e.g. forward-declared MeshCommandQueue)
    // so that type traits below are never evaluated on them.
    if constexpr (!requires { sizeof(CleanT); }) {
        oss << "<incomplete type>";
    } else if constexpr (ttsl::is_specialization_v<CleanT, std::vector>) {
        ttsl::reflection::operator<<(oss, val);
    } else if constexpr (requires { oss << val; }) {
        oss << val;
    } else if constexpr (ttsl::concepts::Reflectable<CleanT>) {
        oss << reflect::type_name<CleanT>() << "(";
        reflect::for_each(
            [&oss, &val](auto I) {
                if constexpr (I > 0) {
                    oss << ", ";
                }
                graph_detail::serialize_member(oss, reflect::get<I>(val));
            },
            val);
        oss << ")";
    } else {
        oss << "<" << reflect::type_name<T>() << ">";
    }
    return oss.str();
}

class IGraphProcessor {
public:
    enum class RunMode {
        NORMAL,      // running everything as is
        NO_DISPATCH  // don't do memory allocations and program runs.
    };

    IGraphProcessor() = default;

    virtual void track_allocate(const tt::tt_metal::Buffer* /*buffer*/) {};

    virtual void track_deallocate(tt::tt_metal::Buffer* /*buffer*/) {};

    virtual void track_allocate_cb(
        const CoreRangeSet& /*core_range_set*/,
        uint64_t /*addr*/,
        uint64_t /*size*/,
        bool /*is_globally_allocated*/,
        const IDevice* /*device*/) {};

    virtual void track_deallocate_cb(const IDevice* /*device*/) {};

    virtual void track_program(tt::tt_metal::Program* /*program*/, const IDevice* /*device*/) {};

    virtual void track_function_start(
        std::string_view /*function_name*/, std::span<TrackedArgument> /*input_parameters*/){};

    virtual void track_function_end() {};
    virtual void track_function_end(const std::any& /*output_tensors*/) {};

    virtual void begin_capture(RunMode /*mode*/){};

    virtual nlohmann::json end_capture() { return nullptr; };

    virtual ~IGraphProcessor() = default;
};

class IGraphHooks {
public:
    IGraphHooks() = default;
    virtual bool hook_allocate(const tt::tt_metal::Buffer* buffer) = 0;

    virtual bool hook_deallocate(tt::tt_metal::Buffer* buffer) = 0;

    virtual bool hook_program(Program* program) = 0;

    virtual bool hook_write_to_device(const tt::tt_metal::Buffer* buffer) = 0;

    virtual bool hook_read_from_device(tt::tt_metal::Buffer* buffer) = 0;

    virtual bool hook_read_from_device(const tt::tt_metal::distributed::MeshBuffer* mesh_buffer) = 0;

    virtual bool hook_write_to_device(const tt::tt_metal::distributed::MeshBuffer* mesh_buffer) = 0;

    virtual ~IGraphHooks() = default;
};

class GraphTracker {
public:
    GraphTracker(const GraphTracker&) = delete;
    GraphTracker(GraphTracker&&) = delete;

    static GraphTracker& instance();

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

    // NOLINTBEGIN(cppcoreguidelines-missing-std-forward)
    template <class... Args>
    void track_function_start(std::string_view function_name, Args&&... args) {
        if (processors.empty()) {
            return;
        }
        std::array<TrackedArgument, sizeof...(Args)> params{
            TrackedArgument{std::any(std::ref(args)), &serialize_tracked_arg<std::remove_reference_t<Args>>}...};
        for (auto& it : processors) {
            it->track_function_start(function_name, params);
        }
    }
    // NOLINTEND(cppcoreguidelines-missing-std-forward)

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
    void track_function_end(ReturnType& output_tensors) {
        if (processors.empty()) {
            return;
        }
        for (auto& it : processors) {
            it->track_function_end(std::ref(output_tensors));
        }
    }

    bool hook_allocate(const Buffer* buffer);

    bool hook_deallocate(Buffer* buffer);

    bool hook_write_to_device(const Buffer* buffer);

    bool hook_write_to_device(const distributed::MeshBuffer* mesh_buffer);

    bool hook_read_from_device(Buffer* buffer);

    bool hook_read_from_device(const distributed::MeshBuffer* mesh_buffer);

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
