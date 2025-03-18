// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "flatbuffer/base_types_to_flatbuffer.hpp"
#include "lightmetal/host_api_capture_helpers.hpp"
#include "program_types_generated.h"
#include <core_coord.hpp>
#include <kernel_types.hpp>
#include <sub_device_types.hpp>
#include <buffer.hpp>

namespace tt::tt_metal {

using FlatbufferCoreCoordVector = flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffer::CoreCoord>>>;
using FlatbufferUInt32VecOfVec =
    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffer::UInt32Vector>>>;

flatbuffers::Offset<flatbuffer::CoreCoord> to_flatbuffer(
    flatbuffers::FlatBufferBuilder& builder, const CoreCoord& coord);
flatbuffers::Offset<flatbuffer::CoreRange> to_flatbuffer(
    flatbuffers::FlatBufferBuilder& builder, const CoreRange& range);
flatbuffers::Offset<flatbuffer::CoreRangeSet> to_flatbuffer(
    flatbuffers::FlatBufferBuilder& builder, const CoreRangeSet& range_set);

std::pair<flatbuffer::CoreSpec, ::flatbuffers::Offset<void>> to_flatbuffer(
    flatbuffers::FlatBufferBuilder& builder, const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec);

FlatbufferCoreCoordVector to_flatbuffer(
    flatbuffers::FlatBufferBuilder& builder, const std::vector<CoreCoord>& core_spec);

FlatbufferUInt32VecOfVec to_flatbuffer(
    flatbuffers::FlatBufferBuilder& builder, const std::vector<std::vector<uint32_t>>& vec_of_vec);

std::pair<flatbuffer::KernelConfig, flatbuffers::Offset<void>> to_flatbuffer(
    flatbuffers::FlatBufferBuilder& builder, const DataMovementConfig& config);

std::pair<flatbuffer::KernelConfig, flatbuffers::Offset<void>> to_flatbuffer(
    flatbuffers::FlatBufferBuilder& builder, const ComputeConfig& config);

std::pair<flatbuffer::KernelConfig, flatbuffers::Offset<void>> to_flatbuffer(
    flatbuffers::FlatBufferBuilder& builder, const EthernetConfig& config);

std::pair<flatbuffer::KernelConfig, flatbuffers::Offset<void>> to_flatbuffer(
    flatbuffers::FlatBufferBuilder& builder,
    const std::variant<DataMovementConfig, ComputeConfig, EthernetConfig>& config);

std::pair<flatbuffer::KernelConfig, flatbuffers::Offset<void>> to_flatbuffer(
    flatbuffers::FlatBufferBuilder& builder, const ReaderDataMovementConfig& config);

std::pair<flatbuffer::KernelConfig, flatbuffers::Offset<void>> to_flatbuffer(
    flatbuffers::FlatBufferBuilder& builder, const WriterDataMovementConfig& config);

flatbuffers::Offset<flatbuffer::RuntimeArg> create_runtime_arg(
    flatbuffers::FlatBufferBuilder& builder, const std::variant<Buffer*, uint32_t>& arg);

flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffer::RuntimeArg>>> to_flatbuffer(
    flatbuffers::FlatBufferBuilder& builder, const std::shared_ptr<RuntimeArgs>& runtime_args);

flatbuffers::Offset<flatbuffers::Vector<uint8_t>> to_flatbuffer(
    flatbuffers::FlatBufferBuilder& builder, tt::stl::Span<const SubDeviceId> sub_device_ids);

}  // namespace tt::tt_metal
