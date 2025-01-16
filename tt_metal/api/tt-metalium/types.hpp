// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "host_api.hpp"

namespace tt::tt_metal {

namespace v1 {

using ProgramHandle = v0::Program;

class CommandQueueHandle {
private:
    explicit constexpr CommandQueueHandle(IDevice* device, std::uint8_t id = 0) : device{device}, id{id} {}

    IDevice* device;
    std::uint8_t id;

    friend CommandQueueHandle GetCommandQueue(IDevice* device, std::uint8_t id);
    friend IDevice* GetDevice(CommandQueueHandle cq);
    friend std::uint8_t GetId(CommandQueueHandle cq);
};

class TraceHandle {
public:
    explicit constexpr operator std::uint32_t() const noexcept { return id; }

private:
    explicit constexpr TraceHandle(CommandQueueHandle cq, std::uint32_t id) noexcept : cq{cq}, id{id} {}

    CommandQueueHandle cq;
    std::uint32_t id;

    friend TraceHandle BeginTraceCapture(CommandQueueHandle cq);
    friend CommandQueueHandle GetCommandQueue(TraceHandle trace);
};

class KernelHandle {
public:
    explicit constexpr KernelHandle(tt_metal::KernelHandle id) noexcept : id{id} {}

    explicit constexpr operator tt_metal::KernelHandle() const noexcept { return id; }

private:
    tt_metal::KernelHandle id;
};

class CircularBufferHandle {
public:
    explicit constexpr CircularBufferHandle(v0::CBHandle cb_id) noexcept : cb_id{cb_id} {}

    explicit constexpr operator v0::CBHandle() const noexcept { return cb_id; }

private:
    v0::CBHandle cb_id;
};

class BufferHandle {
public:
    explicit BufferHandle(const std::shared_ptr<v0::Buffer>& buffer_ptr) noexcept : buffer_ptr{buffer_ptr} {}
    explicit BufferHandle(std::shared_ptr<v0::Buffer>&& buffer_ptr) noexcept :
        buffer_ptr{static_cast<std::shared_ptr<v0::Buffer>&&>(buffer_ptr)} {}

    explicit operator const std::shared_ptr<v0::Buffer>&() const noexcept { return buffer_ptr; }

    v0::Buffer& operator*() const noexcept { return *buffer_ptr.get(); }
    v0::Buffer* operator->() const noexcept { return buffer_ptr.get(); }

private:
    std::shared_ptr<v0::Buffer> buffer_ptr;
};

class EventHandle {
public:
    explicit EventHandle();
    explicit EventHandle(const std::shared_ptr<v0::Event>& event_ptr) noexcept : event_ptr{event_ptr} {}
    explicit EventHandle(std::shared_ptr<v0::Event>&& event_ptr) noexcept :
        event_ptr{static_cast<std::shared_ptr<v0::Event>&&>(event_ptr)} {}

    explicit operator const std::shared_ptr<v0::Event>&() const noexcept { return event_ptr; }

    v0::Event& operator*() const noexcept { return *event_ptr.get(); }
    v0::Event* operator->() const noexcept { return event_ptr.get(); }

private:
    std::shared_ptr<v0::Event> event_ptr;
};

using RuntimeArgs = tt::stl::Span<const std::uint32_t>;

}  // namespace v1
}  // namespace tt::tt_metal
