// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// API
#include <tt-metalium/assert.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/bfloat4.hpp>
#include <tt-metalium/bfloat8.hpp>
#include <tt-metalium/blockfloat_common.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_page_mapping.hpp>
#include <tt-metalium/command_queue.hpp>
#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/device_pool.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/fabric.hpp>
#include <tt-metalium/graph_tracking.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel.hpp>
#include <tt-metalium/lightmetal_replay.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/mesh_graph.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_cache.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/profiler_types.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tile.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/tt_metal_profiler.hpp>
#include <tt-metalium/allocator.hpp>

// FlatBuffers
#include <flatbuffers/flatbuffers.h>

// range-v3
// It looks like we are not using range-v3 in this project

// TT::Metalium::HostDevCommon
#include <hostdevcommon/common_values.hpp>
#include <hostdevcommon/dprint_common.h>
#include <hostdevcommon/fabric_common.h>
#include <hostdevcommon/kernel_structs.h>
#include <hostdevcommon/profiler_common.h>

// Metalium::Metal::Hardware
#include <tt_metal/hw/inc/dev_msgs.h>
#include <tt_metal/hw/inc/debug/ring_buffer.h>

// Metalium::Metal::LLRT
#include <tt_metal/llrt/public_pch.hpp>
#include <tt_metal/llrt/get_platform_architecture.hpp>
#include <tt_metal/llrt/llrt.hpp>
#include <tt_metal/llrt/tt_cluster.hpp>
#include <tt_metal/llrt/hal.hpp>
#include <tt_metal/llrt/rtoptions.hpp>
