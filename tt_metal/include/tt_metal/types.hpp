// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/tt_cluster_descriptor_types.h"
#include "hostdevcommon/common_values.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/device/device_handle.hpp"
#include "tt_metal/impl/dispatch/dispatch_core_manager.hpp"

namespace tt::tt_metal {
namespace v1 {

using ProgramHandle = v0::Program;
class DeviceHandle;
class CommandQueue;
class Trace;

class KernelHandle {
   public:
    explicit constexpr KernelHandle(tt_metal::KernelHandle kernel_id) noexcept : kernel_id{kernel_id} {}

    explicit constexpr operator tt_metal::KernelHandle() const noexcept { return kernel_id; }

   private:
    tt_metal::KernelHandle kernel_id;
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
    explicit BufferHandle(const std::shared_ptr<v0::Buffer> &buffer_ptr) noexcept : buffer_ptr{buffer_ptr} {}
    explicit BufferHandle(std::shared_ptr<v0::Buffer> &&buffer_ptr) noexcept :
        buffer_ptr{static_cast<std::shared_ptr<v0::Buffer> &&>(buffer_ptr)} {}

    explicit operator const std::shared_ptr<v0::Buffer> &() const noexcept { return buffer_ptr; }

    v0::Buffer &operator*() const noexcept { return *buffer_ptr.get(); }
    v0::Buffer *operator->() const noexcept { return buffer_ptr.get(); }

   private:
    std::shared_ptr<v0::Buffer> buffer_ptr;
};

// Not likely going to be opaque, but pending review of
// completion of the prototype of the runtime args.
class Event;
class RuntimeArgs;
class RuntimeArgsData;

}  // namespace v1
}  // namespace tt::tt_metal
