// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <ostream>
#include <tuple>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>

namespace tt::tt_metal {

using DeviceAddr = std::uint64_t;

class IDevice;
class GlobalSemaphoreImpl;
namespace distributed {
class MeshCommandQueue;
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
     *
     * The initial value is written through command queue 0 with fast dispatch.
     */
    GlobalSemaphore(
        distributed::MeshDevice& device,
        CoreRangeSet cores,
        uint32_t initial_value,
        BufferType buffer_type = BufferType::L1);

    /**
     * @brief Allocates a global semaphore in L1 on the mesh device, writing the initial value through `mesh_cq`.
     *
     * @param device Mesh device to create the semaphore on.
     * @param mesh_cq Command queue used for the initial-value write with fast dispatch. Must belong to `device`.
     * It is not used on slow dispatch / simulator, where no command queue is involved.
     * @param cores Range of Tensix coordinates using the semaphore.
     * @param initial_value Initial value of the semaphore.
     * @param buffer_type Buffer type to store the semaphore. Can only be an L1 buffer type.
     */
    GlobalSemaphore(
        distributed::MeshDevice& device,
        distributed::MeshCommandQueue& mesh_cq,
        CoreRangeSet cores,
        uint32_t initial_value,
        BufferType buffer_type = BufferType::L1);

    explicit GlobalSemaphore(GlobalSemaphoreImpl impl);
    GlobalSemaphore(const GlobalSemaphore& other);
    GlobalSemaphore& operator=(const GlobalSemaphore& other);

    GlobalSemaphore(GlobalSemaphore&& other) noexcept;
    GlobalSemaphore& operator=(GlobalSemaphore&& other) noexcept;

    ~GlobalSemaphore();

    IDevice* device() const;

    DeviceAddr address() const;

    /**
     * @brief Resets the semaphore on every local device to `reset_value` (blocking), through command queue 0 with
     * fast dispatch.
     */
    void reset_semaphore_value(uint32_t reset_value) const;

    /**
     * @brief Resets the semaphore on every local device to `reset_value` (blocking).
     *
     * @param reset_value Value to write.
     * @param mesh_cq Command queue used for the write with fast dispatch. Must belong to the semaphore's device. It
     * is not used on slow dispatch / simulator.
     */
    void reset_semaphore_value(uint32_t reset_value, distributed::MeshCommandQueue& mesh_cq) const;

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
