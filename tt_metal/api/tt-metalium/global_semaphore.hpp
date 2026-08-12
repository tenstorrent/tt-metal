// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <tuple>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal_types.hpp>
#include <ostream>

namespace tt::tt_metal {

class IDevice;
class GlobalSemaphoreImpl;

class GlobalSemaphore {
public:
    explicit GlobalSemaphore(GlobalSemaphoreImpl impl);

    GlobalSemaphore(const GlobalSemaphore& other);
    GlobalSemaphore& operator=(const GlobalSemaphore& other);

    GlobalSemaphore(GlobalSemaphore&& other) noexcept;
    GlobalSemaphore& operator=(GlobalSemaphore&& other) noexcept;

    ~GlobalSemaphore();

    IDevice* device() const;

    DeviceAddr address() const;

    void reset_semaphore_value(uint32_t reset_value) const;

    static constexpr auto attribute_names = std::forward_as_tuple("cores", "buffer_type");
    std::tuple<CoreRangeSet, BufferType> attribute_values() const;

    GlobalSemaphoreImpl& impl();
    const GlobalSemaphoreImpl& impl() const;

private:
    std::unique_ptr<GlobalSemaphoreImpl> impl_;
};

}  // namespace tt::tt_metal

std::ostream& operator<<(std::ostream& os, const tt::tt_metal::GlobalSemaphore& global_semaphore);

namespace std {

template <>
struct hash<tt::tt_metal::GlobalSemaphore> {
    std::size_t operator()(const tt::tt_metal::GlobalSemaphore& global_semaphore) const;
};

}  // namespace std
