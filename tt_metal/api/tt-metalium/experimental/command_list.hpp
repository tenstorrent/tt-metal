// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/node_coord.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>
#include <tt-metalium/experimental/metal2_host_api/utility/table.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt_stl/strong_type.hpp>

// Experimental and subject to change: this header carries no API-stability guarantee.

namespace tt::tt_metal::experimental {

using distributed::MeshCommandQueue;
using distributed::MeshDevice;
using distributed::MeshWorkload;

class CommandList;
namespace detail {
class CommandListBuilderImpl;
}

// Command list parameter names are distinct types from Program parameter names so one cannot be passed for the other.
using CmdListTensorArgName = ttsl::StrongType<std::string, struct CmdListTensorArgNameTag>;
using CmdListRuntimeArgName = ttsl::StrongType<std::string, struct CmdListRuntimeArgNameTag>;
using CmdListCommonRuntimeArgName = ttsl::StrongType<std::string, struct CmdListCommonRuntimeArgNameTag>;

/**
 * @brief Links a command list tensor parameter to a TensorParameter of a recorded program.
 */
struct CmdListTensorArgInfo {
    // Program in the workload passed to CommandListBuilder::add(). Only needs to outlive that call.
    std::reference_wrapper<const Program> program;
    // TensorParameter of program whose bindings are patched.
    TensorParamName param_name;
};

/**
 * @brief Links a command list runtime parameter to a named runtime argument on a set of nodes.
 */
struct CmdListRuntimeArgInfo {
    // Program in the workload passed to CommandListBuilder::add(). Only needs to outlive that call.
    std::reference_wrapper<const Program> program;
    KernelSpecName kernel_name;
    std::string arg_name;
    // Nodes whose copy of the argument is patched. Must not be empty.
    std::vector<NodeCoord> nodes;
};

/**
 * @brief Links a command list common runtime parameter to a named common runtime argument.
 */
struct CmdListCommonRuntimeArgInfo {
    // Program in the workload passed to CommandListBuilder::add(). Only needs to outlive that call.
    std::reference_wrapper<const Program> program;
    KernelSpecName kernel_name;
    std::string arg_name;
};

/**
 * @brief Maps command list parameter names to fields of the programs in one recorded workload.
 * A name may map to several fields, which are all patched with the same value.
 */
struct CmdListParameters {
    Table<CmdListTensorArgName, std::vector<CmdListTensorArgInfo>> tensor_parameters;
    Table<CmdListRuntimeArgName, std::vector<CmdListRuntimeArgInfo>> runtime_parameters;
    Table<CmdListCommonRuntimeArgName, std::vector<CmdListCommonRuntimeArgInfo>> common_runtime_parameters;
};

/**
 * @brief New values for command list parameters, applied by CommandList::update_args().
 */
struct CmdListArgPatch {
    Table<CmdListTensorArgName, ProgramRunArgs::TensorArgument> tensor_args;
    Table<CmdListRuntimeArgName, uint32_t> runtime_args;
    Table<CmdListCommonRuntimeArgName, uint32_t> common_runtime_args;
};

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
     * @param device    Mesh device to record workloads on.
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
     * @param workload      Workload to record.
     * @param parameters    Fields of @p workload that built command lists can patch with CommandList::update_args().
     */
    void add(MeshWorkload& workload, const CmdListParameters& parameters = {});

    /**
     * @brief Builds an independent command list bound to @p cq.
     * User allocations are not retained. Replay uses the raw device addresses encoded in the recorded commands.
     * Parameters start with the values recorded by add().
     *
     * @param cq    Command queue the resulting list is bound to.
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
     * @brief Patches parameters for all later replays.
     * The patch is ordered on the bound command queue: replays enqueued before this call use the old values,
     * and replays enqueued after this call use the new ones. Every name and tensor is validated before anything
     * is written. Tensor arguments record raw device addresses; the tensors are not retained.
     *
     * @param patch     New parameter values.
     * @param blocking  If true, waits until the patch has been written.
     */
    void update_args(const CmdListArgPatch& patch, bool blocking = false);

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
     * @param blocking  If true, waits for completion.
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
 * @param cq            Command queue to replay on.
 * @param command_list  Command list to replay.
 * @param blocking      If true, waits for completion.
 */
void EnqueueCommandList(MeshCommandQueue& cq, CommandList& command_list, bool blocking);

}  // namespace tt::tt_metal::experimental
