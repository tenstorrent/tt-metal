// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <tuple>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal_types.hpp>
#include <ostream>

namespace tt::tt_metal {

class IDevice;
class GlobalSemaphoreImpl;
namespace distributed {
class MeshDevice;
}  // namespace distributed
}  // namespace tt::tt_metal

namespace tt::tt_metal {

class GlobalSemaphore {
public:
    /**
     * @brief Allocates a global semaphore in L1 on the mesh device.
     *
     * @param device Mesh device to create the semaphore on.
     * @param cores Range of Tensix coordinates using the semaphore.
     * @param initial_value Initial value of the semaphore.
     * @param buffer_type Buffer type to store the semaphore. Can only be an L1 buffer type.
     * @param cq_id Command queue used for the initial-value write with fast dispatch (cq 0 if not provided). Only
     * looked up with fast dispatch; ignored on slow dispatch / simulator.
     */
    GlobalSemaphore(
        distributed::MeshDevice& device,
        CoreRangeSet cores,
        uint32_t initial_value,
        BufferType buffer_type = BufferType::L1,
        std::optional<uint8_t> cq_id = std::nullopt);

    explicit GlobalSemaphore(GlobalSemaphoreImpl impl);
    GlobalSemaphore(const GlobalSemaphore& other);
    GlobalSemaphore& operator=(const GlobalSemaphore& other);

    GlobalSemaphore(GlobalSemaphore&& other) noexcept;
    GlobalSemaphore& operator=(GlobalSemaphore&& other) noexcept;

    ~GlobalSemaphore();

    IDevice* device() const;

    DeviceAddr address() const;

    /**
     * @brief Resets the semaphore on every local device to `reset_value` (blocking).
     *
     * With fast dispatch the write is issued on command queue `cq_id` (cq 0 if not provided). The queue is only
     * looked up with fast dispatch; it is ignored on slow dispatch / simulator.
     */
    void reset_semaphore_value(uint32_t reset_value, std::optional<uint8_t> cq_id = std::nullopt) const;

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
