// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// API
#include <tt_metal/api/tt-metalium/assert.hpp>
#include <tt_metal/api/tt-metalium/control_plane.hpp>
#include <tt_metal/api/tt-metalium/device_pool.hpp>
#include <tt_metal/api/tt-metalium/distributed_context.hpp>
#include <tt_metal/api/tt-metalium/fabric_types.hpp>

// Metalium::Metal::Impl
#include <tt_metal/impl/public_pch.hpp>
#include <tt_metal/impl/allocator/l1_banking_allocator.hpp>
#include <tt_metal/impl/buffers/dispatch.hpp>
#include <tt_metal/impl/context/metal_context.hpp>
#include <tt_metal/impl/debug/inspector.hpp>
#include <tt_metal/impl/device/dispatch.hpp>
#include <tt_metal/impl/dispatch/device_command.hpp>
#include <tt_metal/impl/dispatch/dispatch_query_manager.hpp>
#include <tt_metal/impl/dispatch/ringbuffer_cache.hpp>
#include <tt_metal/impl/program/dispatch.hpp>
#include <tt_metal/impl/program/program_command_sequence.hpp>
#include <tt_metal/impl/sub_device/sub_device_manager.hpp>
#include <tt_metal/impl/trace/dispatch.hpp>
#include <tt_metal/impl/program/program_impl.hpp>

// Metalium::Metal::LLRT
#include <tt_metal/llrt/public_pch.hpp>
#include <tt_metal/llrt/tt_cluster.hpp>
#include <tt_metal/llrt/hal.hpp>

// TT::Metalium::HostDevCommon
#include <hostdevcommon/common_values.hpp>
#include <hostdevcommon/kernel_structs.h>

// FlatBuffers::FlatBuffers
#include <flatbuffers/flatbuffers.h>
