// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/assert.hpp>
#include "lightmetal/lightmetal_capture.hpp"
#include "flatbuffers/flatbuffers.h"
#include "command_generated.h"
#include "light_metal_binary_generated.h"
#include "trace/trace_buffer.hpp"
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/kernel.hpp>

namespace tt::tt_metal {

LightMetalCaptureContext::LightMetalCaptureContext() : is_tracing_(false), builder_() {}

// Singleton instance accessor
LightMetalCaptureContext& LightMetalCaptureContext::get() {
    static LightMetalCaptureContext instance;
    return instance;
}

bool LightMetalCaptureContext::is_tracing() const { return is_tracing_; }

void LightMetalCaptureContext::set_tracing(bool is_tracing) { is_tracing_ = is_tracing; }

flatbuffers::FlatBufferBuilder& LightMetalCaptureContext::get_builder() { return builder_; }

std::vector<flatbuffers::Offset<tt::tt_metal::flatbuffer::Command>>& LightMetalCaptureContext::get_cmds_vector() {
    return cmds_vec_;
}

void LightMetalCaptureContext::capture_trace_descriptor(const TraceDescriptor& trace_desc, const uint32_t tid) {
    trace_descs_vec_.push_back(to_flatbuffer(builder_, trace_desc, tid));
}

// Create final flatbuffer binary from the built up data and return to caller as blob.
// If light_metal_binary itself (flatbuffer object) is of interest, could return it instead.
LightMetalBinary LightMetalCaptureContext::create_light_metal_binary() {
    auto cmds_vec_fb = builder_.CreateVector(cmds_vec_);
    auto sorted_trace_descs = builder_.CreateVectorOfSortedTables(&trace_descs_vec_);
    auto light_metal_binary =
        tt::tt_metal::flatbuffer::CreateLightMetalBinary(builder_, cmds_vec_fb, sorted_trace_descs);
    builder_.Finish(light_metal_binary);

    const uint8_t* buffer_ptr = builder_.GetBufferPointer();
    size_t buffer_size = builder_.GetSize();

    std::vector<uint8_t> binary_data(buffer_ptr, buffer_ptr + buffer_size);
    return LightMetalBinary(std::move(binary_data));
}

// Reset some internal state, and ensure tracing isn't active. Should only be called at start of tracing.
void LightMetalCaptureContext::reset() {
    TT_ASSERT(!is_tracing_, "Cannot reset light metal capture context while tracing is enabled.");
    builder_.Clear();
    next_global_id_ = 0;
    cmds_vec_.clear();
    trace_descs_vec_.clear();
    buffer_id_to_global_id_map_.clear();
    program_id_to_global_id_map_.clear();
    kernel_to_global_id_map_.clear();
    cb_handle_to_global_id_map_.clear();
}

////////////////////////////////////////////
// Object Map Public Accessors            //
////////////////////////////////////////////

bool LightMetalCaptureContext::is_in_map(const Buffer* obj) {
    return buffer_id_to_global_id_map_.find(obj->unique_id()) != buffer_id_to_global_id_map_.end();
}

uint32_t LightMetalCaptureContext::add_to_map(const Buffer* obj) {
    if (is_in_map(obj)) {
        TT_THROW("Buffer id: {} already exists in global_id map.", obj->unique_id());
    }
    uint32_t global_id = next_global_id_++;
    buffer_id_to_global_id_map_[obj->unique_id()] = global_id;
    return global_id;
}

void LightMetalCaptureContext::remove_from_map(const Buffer* obj) {
    if (!is_in_map(obj)) {
        log_warning(tt::LogMetalTrace, "Buffer not found in global_id map.");
    }
    buffer_id_to_global_id_map_.erase(obj->unique_id());
}

uint32_t LightMetalCaptureContext::get_global_id(const Buffer* obj) {
    auto it = buffer_id_to_global_id_map_.find(obj->unique_id());
    if (it != buffer_id_to_global_id_map_.end()) {
        return it->second;
    } else {
        TT_THROW("Buffer id: {} not found in global_id map", obj->unique_id());
    }
}

bool LightMetalCaptureContext::is_in_map(const Program* obj) {
    return program_id_to_global_id_map_.find(obj->get_id()) != program_id_to_global_id_map_.end();
}

uint32_t LightMetalCaptureContext::add_to_map(const Program* obj) {
    if (is_in_map(obj)) {
        log_warning(tt::LogMetalTrace, "Program id: {} already exists in global_id map.", obj->get_id());
    }
    uint32_t global_id = next_global_id_++;
    program_id_to_global_id_map_[obj->get_id()] = global_id;
    return global_id;
}

void LightMetalCaptureContext::remove_from_map(const Program* obj) {
    if (!is_in_map(obj)) {
        log_warning(tt::LogMetalTrace, "Program id: {} not found in global_id map.", obj->get_id());
    }
    program_id_to_global_id_map_.erase(obj->get_id());
}

uint32_t LightMetalCaptureContext::get_global_id(const Program* obj) {
    auto it = program_id_to_global_id_map_.find(obj->get_id());
    if (it != program_id_to_global_id_map_.end()) {
        return it->second;
    } else {
        TT_THROW("Program id: {} not found in global_id map.", obj->get_id());
    }
}

bool LightMetalCaptureContext::is_in_map(const Kernel* obj) {
    return kernel_to_global_id_map_.find(obj) != kernel_to_global_id_map_.end();
}

uint32_t LightMetalCaptureContext::add_to_map(const Kernel* obj) {
    if (is_in_map(obj)) {
        log_warning(tt::LogMetalTrace, "Kernel already exists in global_id map.");
    }
    uint32_t global_id = next_global_id_++;
    kernel_to_global_id_map_[obj] = global_id;
    return global_id;
}

void LightMetalCaptureContext::remove_from_map(const Kernel* obj) {
    if (!is_in_map(obj)) {
        log_warning(tt::LogMetalTrace, "Kernel not found in global_id map.");
    }
    kernel_to_global_id_map_.erase(obj);
}

uint32_t LightMetalCaptureContext::get_global_id(const Kernel* obj) {
    auto it = kernel_to_global_id_map_.find(obj);
    if (it != kernel_to_global_id_map_.end()) {
        return it->second;
    } else {
        TT_THROW("Kernel not found in global_id map.");
    }
}

bool LightMetalCaptureContext::is_in_map(const CBHandle handle) {
    return cb_handle_to_global_id_map_.find(handle) != cb_handle_to_global_id_map_.end();
}

uint32_t LightMetalCaptureContext::add_to_map(const CBHandle handle) {
    if (is_in_map(handle)) {
        log_warning(tt::LogMetalTrace, "CBHandle already exists in global_id map.");
    }
    uint32_t global_id = next_global_id_++;
    cb_handle_to_global_id_map_[handle] = global_id;
    return global_id;
}

void LightMetalCaptureContext::remove_from_map(const CBHandle handle) {
    if (!is_in_map(handle)) {
        log_warning(tt::LogMetalTrace, "CBHandle not found in global_id map.");
    }
    cb_handle_to_global_id_map_.erase(handle);
}

uint32_t LightMetalCaptureContext::get_global_id(const CBHandle handle) {
    auto it = cb_handle_to_global_id_map_.find(handle);
    if (it != cb_handle_to_global_id_map_.end()) {
        return it->second;
    } else {
        TT_THROW("CBHandle not found in global_id map.");
    }
}

////////////////////////////////////////////
// Non-Class Helper Functions             //
////////////////////////////////////////////

// Serialize tt-metal traceDescriptor and trace_id to flatbuffer format.
TraceDescriptorByTraceIdOffset to_flatbuffer(
    flatbuffers::FlatBufferBuilder& builder, const TraceDescriptor& trace_desc, const uint32_t trace_id) {
    // Serialize the trace_data vector
    auto trace_data_offset = builder.CreateVector(trace_desc.data);

    // Serialize the sub_device_descriptors (map)
    std::vector<flatbuffers::Offset<tt::tt_metal::flatbuffer::SubDeviceDescriptorMapping>>
        sub_device_descriptor_offsets;
    for (const auto& [sub_device_id, descriptor] : trace_desc.descriptors) {
        auto descriptor_offset = tt::tt_metal::flatbuffer::CreateTraceWorkerDescriptor(
            builder,
            descriptor.num_completion_worker_cores,
            descriptor.num_traced_programs_needing_go_signal_multicast,
            descriptor.num_traced_programs_needing_go_signal_unicast);
        auto mapping_offset = tt::tt_metal::flatbuffer::CreateSubDeviceDescriptorMapping(
            builder,
            *sub_device_id,  // No need for static_cast; directly use uint8_t
            descriptor_offset);
        sub_device_descriptor_offsets.push_back(mapping_offset);
    }
    auto sub_device_descriptors_offset = builder.CreateVector(sub_device_descriptor_offsets);

    // Serialize the sub_device_ids vector
    std::vector<uint8_t> sub_device_ids_converted;
    sub_device_ids_converted.reserve(trace_desc.sub_device_ids.size());
    for (const auto& sub_device_id : trace_desc.sub_device_ids) {
        sub_device_ids_converted.push_back(*sub_device_id);
    }
    auto sub_device_ids_offset = builder.CreateVector(sub_device_ids_converted);

    // Create the TraceDescriptor
    auto trace_descriptor_offset = tt::tt_metal::flatbuffer::CreateTraceDescriptor(
        builder, trace_data_offset, sub_device_descriptors_offset, sub_device_ids_offset);

    // Create the TraceDescriptorByTraceId
    return tt::tt_metal::flatbuffer::CreateTraceDescriptorByTraceId(builder, trace_id, trace_descriptor_offset);
}

}  // namespace tt::tt_metal
