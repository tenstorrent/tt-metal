// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>

#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>

// Experimental and subject to change: this header carries no API-stability guarantee.

namespace tt::tt_metal::experimental {

using distributed::MeshCommandQueue;
using distributed::MeshDevice;
using distributed::MeshWorkload;

class CommandList;
namespace detail {
class CommandListBuilderImpl;
}

/**
 * @brief Records MeshWorkloads for later replay.
 * Only one active builder may exist per MeshDevice.
 */
class CommandListBuilder {
public:
    /**
     * @brief Captures the active sub-device manager.
     * add() and build() require that manager to be active.
     *
     * @param device Mesh device to record workloads on.
     */
    explicit CommandListBuilder(MeshDevice& device);

    CommandListBuilder(const CommandListBuilder&) = delete;
    CommandListBuilder& operator=(const CommandListBuilder&) = delete;
    CommandListBuilder(CommandListBuilder&&) noexcept;
    CommandListBuilder& operator=(CommandListBuilder&&) noexcept;
    ~CommandListBuilder();

    /**
     * @brief Records one workload without launching it.
     * Compiles the workload and uploads any required kernel binaries through the current thread's command queue.
     *
     * @param workload Workload to record.
     */
    void add(MeshWorkload& workload);

    /**
     * @brief Builds an independent command list bound to @p cq.
     * User allocations are not retained. Replay uses the raw device addresses encoded in the recorded commands.
     *
     * @param cq Command queue the resulting list is bound to.
     * @return Move-only handle to the recorded command list.
     */
    CommandList build(MeshCommandQueue& cq) const;

    /**
     * @brief Mesh device this builder records on.
     */
    MeshDevice& device() const;

    /**
     * @brief Clears all recorded workloads.
     */
    void clear();

    /**
     * @brief Releases the active-builder reservation and invalidates the builder.
     * Repeated calls have no effect.
     */
    void deallocate();

private:
    std::unique_ptr<detail::CommandListBuilderImpl> impl_;
};

/**
 * @brief Move-only handle to a replayable command list stored in device DRAM.
 * Destruction releases its device resources.
 */
class CommandList {
public:
    CommandList(const CommandList&) = delete;
    CommandList& operator=(const CommandList&) = delete;
    CommandList(CommandList&&) noexcept;
    CommandList& operator=(CommandList&&) noexcept;
    ~CommandList();

    /**
     * @brief Mesh device this command list was built for.
     */
    MeshDevice& device() const;

    /**
     * @brief Command-queue id this list is bound to.
     */
    uint8_t cq_id() const;

    /**
     * @brief Releases device resources and invalidates the handle.
     * Repeated calls have no effect.
     */
    void deallocate();

private:
    friend class detail::CommandListBuilderImpl;
    friend void EnqueueCommandList(MeshCommandQueue& cq, CommandList& command_list, bool blocking);

    /**
     * @brief Replays on the command queue used by build().
     * Use EnqueueCommandList() to replay a command list.
     * The recorded sub-device manager must be active.
     *
     * @param blocking If true, waits for completion.
     */
    void replay(bool blocking) const;

    class Impl;
    explicit CommandList(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief Replays @p command_list on @p cq.
 * @p cq must match the device and queue used to build the list.
 *
 * @param cq Command queue to replay on.
 * @param command_list Command list to replay.
 * @param blocking If true, waits for completion.
 */
void EnqueueCommandList(MeshCommandQueue& cq, CommandList& command_list, bool blocking);

}  // namespace tt::tt_metal::experimental
