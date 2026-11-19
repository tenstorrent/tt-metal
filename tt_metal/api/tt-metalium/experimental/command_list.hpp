// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Experimental extensions of the command list API (tt-metalium/command_list.hpp).

#pragma once

#include <functional>
#include <string>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/advanced_options.hpp>
#include <tt-metalium/experimental/metal2_host_api/prefetcher_pipe_parameter.hpp>
#include <tt-metalium/experimental/tensor_prefetcher.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/workload/utils/table.hpp>
#include <tt_stl/optional_reference.hpp>
#include <tt_stl/strong_type.hpp>

namespace tt::tt_metal {

class CommandListBuilder;

namespace experimental {

using CmdListPrefetcherPipeArgName = ttsl::StrongType<std::string, struct CmdListPrefetcherPipeArgNameTag>;

/**
 * @brief Links a command list PrefetcherPipe parameter to a PrefetcherPipeParameter of a recorded program.
 * The parameter must be bound by SetProgramRunArgs() before the workload is recorded.
 */
struct CmdListPrefetcherPipeArgInfo {
    // Program in the workload passed to CommandListBuilder::add(). Only needs to outlive that call.
    std::reference_wrapper<const Program> program;
    // PrefetcherPipeParameter of program whose pipe addresses are patched.
    PrefetcherPipeParamName param_name;
};

/**
 * @brief Experimental fields of CmdListParameters.
 */
struct CmdListAdvancedParameters {
    Table<CmdListPrefetcherPipeArgName, std::vector<CmdListPrefetcherPipeArgInfo>> prefetcher_pipe_parameters;
};

/**
 * @brief Experimental fields of CmdListArgPatch.
 *
 * CommandList::update_args() moves a PrefetcherPipe parameter, and the recorded Tensor prefetcher requests on its
 * pipe, to a new pipe:
 * - It must have the bound pipe's geometry.
 * - Pipes sharing addresses (one space or relay DFB) must all be parameters, patched together.
 */
struct CmdListAdvancedArgPatch {
    Table<CmdListPrefetcherPipeArgName, PrefetcherPipeArgument> prefetcher_pipe_args;
};

// Record one prefetch request into `builder` without sending it. Takes the same arguments and has the same
// preconditions as QueueTensorPrefetcherRequest, and serializes the request now. Every EnqueueCommandList of a
// built list re-queues the recorded requests, in the order they were recorded, before it enqueues the list's
// workloads.
//
// A list replays only these requests. WaitForCqOnTensorPrefetcher cannot be recorded: it takes effect when
// called, so data the requests read, including a group selector's mask, must be written and fenced before the
// replay.
//
// Built lists can be replayed only while the prefetcher that was active here is still running. Keep the
// tensors alive until it stops, and the pipes alive until it stops or until CommandList::update_args() replaces
// them through a PrefetcherPipe parameter.
void AddTensorPrefetcherRequest(
    CommandListBuilder& builder,
    const std::vector<std::reference_wrapper<const PrefetcherPipe>>& prefetcher_pipes,
    ttsl::optional_reference<const distributed::MeshCoordinateRangeSet> device_subset,
    const std::vector<TensorPrefetcherInput>& input_tensors);

}  // namespace experimental

}  // namespace tt::tt_metal
