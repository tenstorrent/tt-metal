// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "lightmetal_replay.hpp"
#include <iostream>
#include "binary_generated.h"
#include "command_generated.h"
#include "tt_metal/impl/trace/trace_buffer.hpp"
#include "tt_metal/common/logger.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/impl/device/device.hpp"

namespace tt::tt_metal {
inline namespace v0 {

//////////////////////////////////////
// Helper Functions                 //
//////////////////////////////////////

// A convenience function - Read arbitrary binary blob from file.
void readBinaryBlobFromFile(const std::string& filename, std::vector<uint8_t>& blob) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    std::streamsize size = file.tellg();
    if (size <= 0) {
        throw std::runtime_error("File is empty or invalid: " + filename);
    }

    blob.resize(static_cast<size_t>(size));

    file.seekg(0, std::ios::beg);
    if (!file.read(reinterpret_cast<char*>(blob.data()), size)) {
        throw std::runtime_error("Failed to read file: " + filename);
    }
}

detail::TraceDescriptor fromFlatBuffer(const tt::target::lightmetal::TraceDescriptor* fb_desc) {
    if (!fb_desc) {
        std::cerr << "TraceDescriptor is null." << std::endl;
        return {};
    }

    detail::TraceDescriptor traceDesc;
    if (auto trace_data_fb = fb_desc->trace_data()) {
        traceDesc.data.assign(trace_data_fb->begin(), trace_data_fb->end());
    }
    traceDesc.num_completion_worker_cores = fb_desc->num_completion_worker_cores();
    traceDesc.num_traced_programs_needing_go_signal_multicast = fb_desc->num_traced_programs_needing_go_signal_multicast();
    traceDesc.num_traced_programs_needing_go_signal_unicast = fb_desc->num_traced_programs_needing_go_signal_unicast();

    return traceDesc;
}

inline BufferType fromFlatbuffer(tt::target::BufferType type) {
    switch (type) {
        case tt::target::BufferType::DRAM: return BufferType::DRAM;
        case tt::target::BufferType::L1: return BufferType::L1;
        case tt::target::BufferType::SystemMemory: return BufferType::SYSTEM_MEMORY;
        case tt::target::BufferType::L1Small: return BufferType::L1_SMALL;
        case tt::target::BufferType::Trace: return BufferType::TRACE;
        default: throw std::invalid_argument("Unknown BufferType value in fromFlatbuffer()");
    }
}

inline tt::tt_metal::DataMovementProcessor fromFlatbuffer(tt::target::DataMovementProcessor in) {
    switch(in) {
        case tt::target::DataMovementProcessor::RISCV_0: return tt::tt_metal::DataMovementProcessor::RISCV_0;
        case tt::target::DataMovementProcessor::RISCV_1: return tt::tt_metal::DataMovementProcessor::RISCV_1;
        default: throw std::invalid_argument("Unknown DataMovementProcessor value in fromFlatbuffer()");
    }
}

inline tt::tt_metal::NOC fromFlatbuffer(tt::target::NOC in) {
    switch (in) {
        case tt::target::NOC::NOC_0: return tt::tt_metal::NOC::NOC_0;
        case tt::target::NOC::NOC_1: return tt::tt_metal::NOC::NOC_1;
        default: throw std::invalid_argument("Invalid NOC value passed to fromFlatbuffer");
    }
}

inline tt::tt_metal::NOC_MODE fromFlatbuffer(tt::target::NOC_MODE in) {
    switch(in) {
        case tt::target::NOC_MODE::DM_DEDICATED_NOC: return tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC;
        case tt::target::NOC_MODE::DM_DYNAMIC_NOC: return tt::tt_metal::NOC_MODE::DM_DYNAMIC_NOC;
        default: throw std::invalid_argument("Unknown NOC_MODE value in fromFlatbuffer()");
    }
}

inline tt::tt_metal::Eth fromFlatbuffer(tt::target::Eth in) {
    switch(in) {
        case tt::target::Eth::SENDER: return tt::tt_metal::Eth::SENDER;
        case tt::target::Eth::RECEIVER: return tt::tt_metal::Eth::RECEIVER;
        case tt::target::Eth::IDLE: return tt::tt_metal::Eth::IDLE;
        default: throw std::invalid_argument("Unknown Eth value in fromFlatbuffer()");
    }
}

inline MathFidelity fromFlatbuffer(tt::target::MathFidelity input) {
    switch (input) {
        case tt::target::MathFidelity::LoFi: return MathFidelity::LoFi;
        case tt::target::MathFidelity::HiFi2: return MathFidelity::HiFi2;
        case tt::target::MathFidelity::HiFi3: return MathFidelity::HiFi3;
        case tt::target::MathFidelity::HiFi4: return MathFidelity::HiFi4;
        case tt::target::MathFidelity::Invalid: return MathFidelity::Invalid;
        default: throw std::invalid_argument("Unknown MathFidelity value in fromFlatbuffer()");
    }
}

inline UnpackToDestMode fromFlatbuffer(tt::target::UnpackToDestMode input) {
    switch (input) {
        case tt::target::UnpackToDestMode::UnpackToDestFp32: return UnpackToDestMode::UnpackToDestFp32;
        case tt::target::UnpackToDestMode::Default: return UnpackToDestMode::Default;
        default: throw std::invalid_argument("Invalid UnpackToDestMode value passed to fromFlatbuffer");
    }
}

inline std::variant<CoreCoord, CoreRange, CoreRangeSet> fromFlatbuffer(
    const tt::target::CoreSpec core_spec, const void *flatbuffer_union) {

    switch (core_spec) {
        case tt::target::CoreSpec::CoreCoord: {
            auto core_coord = static_cast<const tt::target::CoreCoord *>(flatbuffer_union);
            if (!core_coord) throw std::runtime_error("Invalid CoreCoord data");
            return CoreCoord{core_coord->x(), core_coord->y()};
        }
        case tt::target::CoreSpec::CoreRange: {
            auto core_range = static_cast<const tt::target::CoreRange *>(flatbuffer_union);
            if (!core_range) throw std::runtime_error("Invalid CoreRange data");
            return CoreRange{
                {core_range->start()->x(), core_range->start()->y()},
                {core_range->end()->x(), core_range->end()->y()}
            };
        }
        case tt::target::CoreSpec::CoreRangeSet: {
            auto core_range_set = static_cast<const tt::target::CoreRangeSet *>(flatbuffer_union);
            if (!core_range_set) throw std::runtime_error("Invalid CoreRangeSet data");
            std::vector<CoreRange> ranges;
            for (const auto range : *core_range_set->ranges()) {
                ranges.emplace_back(
                    CoreCoord{range->start()->x(), range->start()->y()},
                    CoreCoord{range->end()->x(), range->end()->y()}
                );
            }
            return CoreRangeSet{ranges};
        }
        default:
            throw std::runtime_error("Unhandled CoreSpec type in fromFlatbuffer");
    }
}

inline DataMovementConfig fromFlatbuffer(const tt::target::DataMovementConfig *fb_config) {
    DataMovementConfig config;

    // Extract processor, noc, and noc_mode
    config.processor = fromFlatbuffer(fb_config->processor());
    config.noc = fromFlatbuffer(fb_config->noc());
    config.noc_mode = fromFlatbuffer(fb_config->noc_mode());

    // Extract compile_args
    auto fb_compile_args = fb_config->compile_args();
    config.compile_args.assign(fb_compile_args->begin(), fb_compile_args->end());

    // Extract defines
    auto fb_defines = fb_config->defines();
    for (auto fb_define : *fb_defines) {
        config.defines.emplace(fb_define->key()->str(), fb_define->value()->str());
    }

    return config;
}

inline ComputeConfig fromFlatbuffer(const tt::target::ComputeConfig *fb_config) {
    ComputeConfig config;

    // Extract math_fidelity and boolean flags
    config.math_fidelity = fromFlatbuffer(fb_config->math_fidelity());
    config.fp32_dest_acc_en = fb_config->fp32_dest_acc_en();
    config.dst_full_sync_en = fb_config->dst_full_sync_en();
    config.bfp8_pack_precise = fb_config->bfp8_pack_precise();
    config.math_approx_mode = fb_config->math_approx_mode();

    // Extract unpack_to_dest_mode
    auto fb_unpack_modes = fb_config->unpack_to_dest_mode();
    config.unpack_to_dest_mode.reserve(fb_unpack_modes->size());
    for (auto fb_mode : *fb_unpack_modes) {
        config.unpack_to_dest_mode.push_back(fromFlatbuffer(fb_mode));
    }

    // Extract compile_args
    auto fb_compile_args = fb_config->compile_args();
    config.compile_args.assign(fb_compile_args->begin(), fb_compile_args->end());

    // Extract defines
    auto fb_defines = fb_config->defines();
    for (auto fb_define : *fb_defines) {
        config.defines.emplace(fb_define->key()->str(), fb_define->value()->str());
    }

    return config;
}

inline EthernetConfig fromFlatbuffer(const tt::target::EthernetConfig *fb_config) {
    EthernetConfig config;

    // Extract eth_mode, noc, and processor
    config.eth_mode = fromFlatbuffer(fb_config->eth_mode());
    config.noc = fromFlatbuffer(fb_config->noc());
    config.processor = fromFlatbuffer(fb_config->processor());

    // Extract compile_args
    auto fb_compile_args = fb_config->compile_args();
    config.compile_args.assign(fb_compile_args->begin(), fb_compile_args->end());

    // Extract defines
    auto fb_defines = fb_config->defines();
    for (auto fb_define : *fb_defines) {
        config.defines.emplace(fb_define->key()->str(), fb_define->value()->str());
    }

    return config;
}

inline std::variant<DataMovementConfig, ComputeConfig, EthernetConfig> fromFlatbuffer(
    const tt::target::KernelConfig config_type, const void *flatbuffer_union) {

    switch (config_type) {
        case tt::target::KernelConfig::DataMovementConfig:
            return fromFlatbuffer(static_cast<const tt::target::DataMovementConfig *>(flatbuffer_union));
        case tt::target::KernelConfig::ComputeConfig:
            return fromFlatbuffer(static_cast<const tt::target::ComputeConfig *>(flatbuffer_union));
        case tt::target::KernelConfig::EthernetConfig:
            return fromFlatbuffer(static_cast<const tt::target::EthernetConfig *>(flatbuffer_union));
        default:
            throw std::runtime_error("Unhandled KernelConfig type in fromFlatbuffer.");
    }
}

inline Tile fromFlatBuffer(const tt::target::Tile *tile_fb) {
    if (!tile_fb) {
        throw std::runtime_error("Invalid Tile FlatBuffer object");
    }

    // Convert FlatBuffer vectors to std::array
    std::array<uint32_t, 2> tile_shape = {tile_fb->tile_shape()->Get(0), tile_fb->tile_shape()->Get(1)};
    std::array<uint32_t, 2> face_shape = {tile_fb->face_shape()->Get(0), tile_fb->face_shape()->Get(1)};

    // Create and return the Tile object, explicitly initializing the members
    Tile tile;
    tile.tile_shape = tile_shape;
    tile.face_shape = face_shape;
    tile.tile_hw = tile_fb->tile_hw();
    tile.face_hw = tile_fb->face_hw();
    tile.num_faces = tile_fb->num_faces();
    tile.partial_face = tile_fb->partial_face();
    tile.narrow_tile = tile_fb->narrow_tile();
    tile.transpose_within_face = tile_fb->transpose_within_face();
    tile.transpose_of_faces = tile_fb->transpose_of_faces();

    return tile;
}


inline std::array<std::optional<Tile>, NUM_CIRCULAR_BUFFERS> fromFlatBuffer(
    const flatbuffers::Vector<flatbuffers::Offset<tt::target::Tile>> *tiles_fb) {

    std::array<std::optional<Tile>, NUM_CIRCULAR_BUFFERS> tiles = {};
    if (tiles_fb) {
        for (size_t i = 0; i < tiles_fb->size() && i < NUM_CIRCULAR_BUFFERS; ++i) {
            tiles[i] = fromFlatBuffer(tiles_fb->Get(i));
        }
    }

    return tiles;
}

inline CircularBufferConfig fromFlatBuffer(const tt::target::CircularBufferConfig *config_fb) {
    if (!config_fb) {
        throw std::runtime_error("Invalid CircularBufferConfig FlatBuffer object");
    }

    // Create a CircularBufferConfig. Constructor doesn't matter much, since we serialized all
    // members, will deserialize them here to get fully formed object.
    CircularBufferConfig config(0, {});
    config.total_size_ = config_fb->total_size();

    // Note: std::optional is not supported by FlatBuffers, so nullopt was serialized as value 0 in FlatBuffer.
    config.globally_allocated_address_ = config_fb->globally_allocated_address() == 0 ? std::nullopt : std::optional<uint32_t>(config_fb->globally_allocated_address());

    if (config_fb->data_formats()) {
        for (auto entry : *config_fb->data_formats()) {
            log_info(tt::LogMetalTrace, "KCM DF index: {}, Format: {}", entry->index(), entry->format());
            config.data_formats_[entry->index()] = static_cast<tt::DataFormat>(entry->format());
        }
    }

    if (config_fb->page_sizes()) {
        for (auto entry : *config_fb->page_sizes()) {
            log_info(tt::LogMetalTrace, "KCM Buffer index: {}, Page size: {}", entry->index(), entry->size());
            config.page_sizes_[entry->index()] = entry->size();
        }
    }

    config.tiles_ = fromFlatBuffer(config_fb->tiles());

    if (config_fb->buffer_indices()) {
        config.buffer_indices_.insert(config_fb->buffer_indices()->begin(), config_fb->buffer_indices()->end());
    }

    config.dynamic_cb_ = config_fb->dynamic_cb();
    config.max_size_ = config_fb->max_size();

    return config;
}

//////////////////////////////////////
// LightMetalReplay Class           //
//////////////////////////////////////

LightMetalReplay::LightMetalReplay(std::vector<uint8_t> blob)
    : blob_(std::move(blob)), lm_binary_(nullptr) {
    lm_binary_ = parseFlatBufferBinary();  // Parse and store the FlatBuffer binary
    if (!lm_binary_) {
        throw std::runtime_error("Failed to parse FlatBuffer binary during initialization.");
    }
}

const target::lightmetal::LightMetalBinary* LightMetalReplay::parseFlatBufferBinary() {
    try {
        const uint8_t* data = blob_.data();
        size_t size = blob_.size();

        // Verify the FlatBuffer data.
        flatbuffers::Verifier verifier(data, size);
        if (!target::lightmetal::VerifyLightMetalBinaryBuffer(verifier)) {
            std::cerr << "Failed to verify FlatBuffer data." << std::endl;
            return nullptr;
        }

        // Parse and return the FlatBuffer object.
        return target::lightmetal::GetLightMetalBinary(data);
    } catch (const std::exception& e) {
        std::cerr << "Exception while parsing FlatBuffer binary: " << e.what() << std::endl;
        return nullptr;
    }
}

// Return a TraceDescriptor for a given trace_id from the FlatBuffer binary.
std::optional<detail::TraceDescriptor> LightMetalReplay::getTraceByTraceId(uint32_t target_trace_id) {
    if (const auto* trace_descriptors = lm_binary_ ? lm_binary_->trace_descriptors() : nullptr) {
        if (const auto* fb_trace_desc_by_id = trace_descriptors->LookupByKey(target_trace_id)) {
            if (const auto* fb_desc = fb_trace_desc_by_id->desc()) {
                return fromFlatBuffer(fb_desc);
            }
        }
    }

    std::cerr << "Failed to find trace_id: " << target_trace_id << " in binary." << std::endl;
    return std::nullopt;
}


// Object maps public accessors
void LightMetalReplay::addBufferToMap(uint32_t global_id, std::shared_ptr<::tt::tt_metal::Buffer> buffer) {
    if (bufferMap_.find(global_id) != bufferMap_.end()) {
        log_warning(tt::LogMetalTrace, "Buffer with global_id: {} already exists in map.", global_id);
    }
    bufferMap_[global_id] = buffer; // Shared ownership
}

std::shared_ptr<::tt::tt_metal::Buffer> LightMetalReplay::getBufferFromMap(uint32_t global_id) const {
    auto it = bufferMap_.find(global_id);
    if (it != bufferMap_.end()) {
        return it->second; // Return shared_ptr
    }
    return nullptr; // If not found
}

void LightMetalReplay::removeBufferFromMap(uint32_t global_id) {
    bufferMap_.erase(global_id);
}

void LightMetalReplay::addProgramToMap(uint32_t global_id, std::shared_ptr<::tt::tt_metal::Program> program) {
    if (programMap_.find(global_id) != programMap_.end()) {
        log_warning(tt::LogMetalTrace, "Program with global_id: {} already exists in map.", global_id);
    }
    programMap_[global_id] = program; // Shared ownership
}

std::shared_ptr<::tt::tt_metal::Program> LightMetalReplay::getProgramFromMap(uint32_t global_id) const {
    auto it = programMap_.find(global_id);
    if (it != programMap_.end()) {
        return it->second; // Return shared_ptr
    }
    return nullptr; // If not found
}

void LightMetalReplay::removeProgramFromMap(uint32_t global_id) {
    programMap_.erase(global_id);
}

void LightMetalReplay::addKernelHandleToMap(uint32_t global_id, ::tt::tt_metal::KernelHandle kernel_id) {
    if (kernelHandleMap_.find(global_id) != kernelHandleMap_.end()) {
        log_warning(tt::LogMetalTrace, "KernelHandle with global_id: {} already exists in map.", global_id);
    }
    kernelHandleMap_[global_id] = kernel_id; // Shared ownership
}

::tt::tt_metal::KernelHandle LightMetalReplay::getKernelHandleFromMap(uint32_t global_id) const {
    if (auto it = kernelHandleMap_.find(global_id); it != kernelHandleMap_.end()) {
        return it->second; // Return KernelHandle.
    }
    throw std::runtime_error(fmt::format("KernelHandle with global_id: {} used but doesn't exist.", global_id));
}

void LightMetalReplay::removeKernelHandleFromMap(uint32_t global_id) {
    kernelHandleMap_.erase(global_id);
}


void LightMetalReplay::addCBHandleToMap(uint32_t global_id, ::tt::tt_metal::CBHandle cb_handle) {
    if (cbHandleMap_.find(global_id) != cbHandleMap_.end()) {
        log_warning(tt::LogMetalTrace, "CBHandle with global_id: {} already exists in map.", global_id);
    }
    cbHandleMap_[global_id] = cb_handle; // Shared ownership
}

::tt::tt_metal::CBHandle LightMetalReplay::getCBHandleFromMap(uint32_t global_id) const {
    if (auto it = cbHandleMap_.find(global_id); it != cbHandleMap_.end()) {
        return it->second; // Return CBHandle.
    }
    throw std::runtime_error(fmt::format("CBHandle with global_id: {} used but doesn't exist.", global_id));
}

void LightMetalReplay::removeCBHandleFromMap(uint32_t global_id) {
    cbHandleMap_.erase(global_id);
}

void LightMetalReplay::setupDevices() {
    log_info(tt::LogMetalTrace, "Setting up system now...");

    // FIXME - Get these from lm_binary_ systemdesc once available. For now hardcode.
    const size_t buffer_size = 2048;
    this->arch_ = tt::ARCH::WORMHOLE_B0;
    const int device_id = 0;
    const auto dispatch_core_type = tt_metal::DispatchCoreType::WORKER;
    const chip_id_t mmio_device_id = 0;
    auto devices_map = tt::tt_metal::detail::CreateDevices({mmio_device_id}, 1, DEFAULT_L1_SMALL_SIZE, buffer_size, dispatch_core_type);
    this->device_ = devices_map.at(mmio_device_id);
}

//////////////////////////////////////
// Executor                         //
//////////////////////////////////////

// Some open questions...
// 1. How to pass Device* to replay functions? Can use a global variable for now.
// 2. How to pass other things like input tensors?
// 3. Can we fully encapsulate each host API command here.


// Execute a command by dispatching to appropriate handler based on type.
void LightMetalReplay::execute(tt::target::Command const *command) {
  switch (command->cmd_type()) {
  case ::tt::target::CommandType::EnqueueTraceCommand: {
    execute(command->cmd_as_EnqueueTraceCommand());
    break;
  }
  case ::tt::target::CommandType::ReplayTraceCommand: {
    execute(command->cmd_as_ReplayTraceCommand());
    break;
  }
  case ::tt::target::CommandType::LoadTraceCommand: {
    execute(command->cmd_as_LoadTraceCommand());
    break;
  }
  case ::tt::target::CommandType::ReleaseTraceCommand: {
    execute(command->cmd_as_ReleaseTraceCommand());
    break;
  }
  case ::tt::target::CommandType::CreateBufferCommand: {
    execute(command->cmd_as_CreateBufferCommand());
    break;
  }
  case ::tt::target::CommandType::DeallocateBufferCommand: {
    execute(command->cmd_as_DeallocateBufferCommand());
    break;
  }
  case ::tt::target::CommandType::EnqueueWriteBufferCommand: {
    execute(command->cmd_as_EnqueueWriteBufferCommand());
    break;
  }
  case ::tt::target::CommandType::EnqueueReadBufferCommand: {
    execute(command->cmd_as_EnqueueReadBufferCommand());
    break;
  }
  case ::tt::target::CommandType::FinishCommand: {
    execute(command->cmd_as_FinishCommand());
    break;
  }
  case ::tt::target::CommandType::CreateProgramCommand: {
    execute(command->cmd_as_CreateProgramCommand());
    break;
  }
  case ::tt::target::CommandType::EnqueueProgramCommand: {
    execute(command->cmd_as_EnqueueProgramCommand());
    break;
  }
  case ::tt::target::CommandType::CreateKernelCommand: {
    execute(command->cmd_as_CreateKernelCommand());
    break;
  }
  case ::tt::target::CommandType::SetRuntimeArgsCommand: {
    execute(command->cmd_as_SetRuntimeArgsCommand());
    break;
  }
  case ::tt::target::CommandType::CreateCircularBufferCommand: {
    execute(command->cmd_as_CreateCircularBufferCommand());
    break;
  }
  default:
    throw std::runtime_error("Unsupported type: " + std::string(EnumNameCommandType(command->cmd_type())));
    break;
  }
}

// Per API command handlers.
void LightMetalReplay::execute(tt::target::EnqueueTraceCommand const *cmd) {
    log_info(tt::LogMetalTrace, "LightMetalReplay EnqueueTrace(). cq_id: {} tid: {} blocking: {}", cmd->cq_id(), cmd->tid(), cmd->blocking());
    // FIXME - Needs some tweaking, since API takes CQ should binarize cq_id and device_id.
    CommandQueue &cq = this->device_->command_queue(cmd->cq_id());
    EnqueueTrace(cq, cmd->tid(), cmd->blocking());
}

void LightMetalReplay::execute(tt::target::ReplayTraceCommand const *cmd) {
    log_info(tt::LogMetalTrace, "LightMetalReplay ReplayTrace(). cq_id: {} tid: {} blocking: {}", cmd->cq_id(), cmd->tid(), cmd->blocking());
    ReplayTrace(this->device_, cmd->cq_id(), cmd->tid(), cmd->blocking());
}

void LightMetalReplay::execute(tt::target::LoadTraceCommand const *cmd) {
    log_info(tt::LogMetalTrace, "LightMetalReplay LoadTrace(). cq_id: {} tid: {}", cmd->cq_id(), cmd->tid());
    // Get the trace descriptor from flatbuffer and load it to device.
    auto trace_desc = getTraceByTraceId(cmd->tid());
    LoadTrace(this->device_, cmd->cq_id(), cmd->tid(), trace_desc.value());
}

void LightMetalReplay::execute(tt::target::ReleaseTraceCommand const *cmd) {
    log_info(tt::LogMetalTrace, "LightMetalReplay ReleaseTrace(). tid: {}", cmd->tid());
    ReleaseTrace(this->device_, cmd->tid());
}

void LightMetalReplay::execute(tt::target::CreateBufferCommand const *cmd) {
    log_info(tt::LogMetalTrace, "LightMetalReplay CreateBufferCommand(). global_id: {} size: {} page_size: {} layout: {} buffer_type: {}",
        cmd->global_id(), cmd->config()->size(), cmd->config()->page_size(),
        EnumNameTensorMemoryLayout(cmd->config()->buffer_layout()), EnumNameBufferType(cmd->config()->buffer_type()));

    switch (cmd->config()->buffer_layout()) {
    case tt::target::TensorMemoryLayout::Interleaved: {
        tt::tt_metal::InterleavedBufferConfig config{
            .device = this->device_,
            .size = cmd->config()->size(),
            .page_size = cmd->config()->page_size(),
            .buffer_type = fromFlatbuffer(cmd->config()->buffer_type())};

        auto buffer = CreateBuffer(config);
        addBufferToMap(cmd->global_id(), buffer);
        break;
    }
    default:
        throw std::runtime_error("Unsupported buffer_layout: " + std::string(EnumNameTensorMemoryLayout(cmd->config()->buffer_layout())));
    }
}

void LightMetalReplay::execute(tt::target::DeallocateBufferCommand const *cmd) {
    auto buffer = getBufferFromMap(cmd->global_id());
    if (!buffer) {
        throw std::runtime_error("Buffer w/ global_id: " + std::to_string(cmd->global_id()) + " not previously created");
    }
    DeallocateBuffer(*buffer); // Buffer& expected.
    removeBufferFromMap(cmd->global_id());
}

void LightMetalReplay::execute(tt::target::EnqueueWriteBufferCommand const *cmd) {
    auto buffer = getBufferFromMap(cmd->buffer_global_id());
    if (!buffer) {
        throw std::runtime_error("Buffer w/ global_id: " + std::to_string(cmd->buffer_global_id()) + " not previously created");
    }

    log_info(tt::LogMetalTrace, "LightMetalReplay EnqueueWriteBufferCommand(). cq_global_id: {} buffer_global_id: {} addr: 0x{:x}",
        cmd->cq_global_id(), cmd->buffer_global_id(), buffer->address());

    // FIXME - get cq object from global CQ map instead.
    CommandQueue &cq = this->device_->command_queue(cmd->cq_global_id());
    EnqueueWriteBuffer(cq, buffer, cmd->src()->data(), cmd->blocking());
}

void LightMetalReplay::execute(tt::target::EnqueueReadBufferCommand const *cmd) {
    auto buffer = getBufferFromMap(cmd->buffer_global_id());
    if (!buffer) {
        throw std::runtime_error("Buffer w/ global_id: " + std::to_string(cmd->buffer_global_id()) + " not previously created");
    }

    log_info(tt::LogMetalTrace, "LightMetalReplay EnqueueReadBufferCommand(). cq_global_id: {} buffer_global_id: {} addr: 0x{:x} buf_size: {}",
        cmd->cq_global_id(), cmd->buffer_global_id(), buffer->address(), buffer->size());

    // FIXME - get cq object from global CQ map instead.
    CommandQueue &cq = this->device_->command_queue(cmd->cq_global_id());
    std::vector<uint32_t> readback_data(buffer->size() / sizeof(uint32_t), 0);
    EnqueueReadBuffer(cq, buffer, readback_data.data(), cmd->blocking());

    // FIXME - What should we do with readback data? For not just print.
    // One idea is to store in map by global_read_id that caller can access.
    bool show_reads = std::getenv("SHOW_READS");
    if (show_reads) {
        for (size_t i = 0; i < readback_data.size(); i++) {
            log_info(tt::LogMetalTrace, " rd_data i: {:3d} => data: {}", i, readback_data[i]);
        }
    }
}

void LightMetalReplay::execute(tt::target::FinishCommand const *cmd) {
    log_info(tt::LogMetalTrace, "LightMetalReplay FinishCommand(). cq_global_id: {}", cmd->cq_global_id());
    CommandQueue &cq = this->device_->command_queue(cmd->cq_global_id());
    Finish(cq);
}

void LightMetalReplay::execute(tt::target::CreateProgramCommand const *cmd) {
    log_info(tt::LogMetalTrace, "LightMetalReplay CreateProgramCommand(). global_id: {} ", cmd->global_id());
    auto program = CreateProgram();
    addProgramToMap(cmd->global_id(), std::make_shared<Program>(std::move(program)));
}

void LightMetalReplay::execute(tt::target::EnqueueProgramCommand const *cmd) {
    auto program = getProgramFromMap(cmd->program_global_id());
    if (!program) {
        throw std::runtime_error("Program with global_id: " + std::to_string(cmd->program_global_id()) + " not previously created");
    }

    log_info(tt::LogMetalTrace, "LightMetalReplay EnqueueProgramCommand(). program_global_id: {} cq_global_id: {}", cmd->program_global_id(), cmd->cq_global_id());

    // FIXME - get cq object from global CQ map instead.
    CommandQueue &cq = this->device_->command_queue(cmd->cq_global_id());
    EnqueueProgram(cq, *program, cmd->blocking());
}

void LightMetalReplay::execute(tt::target::CreateKernelCommand const *cmd) {
    log_info(tt::LogMetalTrace, "LightMetalReplay CreateKernelCommand(). global_id: {} program_global_id: {}", cmd->global_id(), cmd->program_global_id());
    auto program = getProgramFromMap(cmd->program_global_id());
    if (!program) {
        throw std::runtime_error("Program with global_id: " + std::to_string(cmd->program_global_id()) + " not previously created");
    }

    auto core_spec = fromFlatbuffer(cmd->core_spec_type(), cmd->core_spec());
    auto kernel_config = fromFlatbuffer(cmd->config_type(), cmd->config());
    auto kernel_id = CreateKernel(*program, cmd->file_name()->c_str(), core_spec, kernel_config);
    addKernelHandleToMap(cmd->global_id(), kernel_id);
}

void LightMetalReplay::execute(tt::target::SetRuntimeArgsCommand const *cmd) {
    log_info(tt::LogMetalTrace, "LightMetalReplay SetRuntimeArgsCommand(). program_global_id: {} kernel_global_id: {}", cmd->program_global_id(), cmd->kernel_global_id());
    auto program = getProgramFromMap(cmd->program_global_id());
    auto kernel_id = getKernelHandleFromMap(cmd->kernel_global_id());

    if (!program) {
        throw std::runtime_error("Program with global_id: " + std::to_string(cmd->program_global_id()) + " not previously created");
    }

    // API expects a span so create from flatbuffer vector.
    stl::Span<const uint32_t> args_span(cmd->args()->data(), cmd->args()->size());
    auto core_spec = fromFlatbuffer(cmd->core_spec_type(), cmd->core_spec());
    SetRuntimeArgs(*program, kernel_id, core_spec, args_span);
}

void LightMetalReplay::execute(tt::target::CreateCircularBufferCommand const *cmd) {
    log_info(tt::LogMetalTrace, "LightMetalReplay CreateCircularBufferCommand(). global_id: {} program_global_id: {}", cmd->global_id(), cmd->program_global_id());
    auto program = getProgramFromMap(cmd->program_global_id());
    if (!program) {
        throw std::runtime_error("Program with global_id: " + std::to_string(cmd->program_global_id()) + " not previously created");
    }

    auto core_spec = fromFlatbuffer(cmd->core_spec_type(), cmd->core_spec());
    auto config = fromFlatBuffer(cmd->config());
    auto cb_handle = CreateCircularBuffer(*program, core_spec, config);
    addCBHandleToMap(cmd->global_id(), cb_handle);
}


// Main entry point to execute a light metal binary blob, return true if pass.
bool LightMetalReplay::executeLightMetalBinary() {

    if (!lm_binary_) {
        std::cerr << "FlatBuffer binary not initialized." << std::endl;
        return false;
    }

    try {
        const auto* trace_descriptors = lm_binary_->trace_descriptors();
        const auto* commands = lm_binary_->commands();
        if (!commands) {
            std::cerr << "Nothing to run, no commands in binary." << std::endl;
            return false;
        }

        setupDevices();
        log_info(tt::LogMetalTrace, "Executing Binary w/ cmds: {} traces: {}", commands->size(), trace_descriptors->size());

        // Just loop over all commands, and execute. This is purposely kept simple for prototyping v0,
        // should expand to cover multiple program, devices, cqs, etc. FIXME
        uint32_t cmd_idx = 1; // Debug
        for (const auto* cmd : *commands) {
            log_info(tt::LogMetalTrace, "Executing Binary CMD {}/{} (Type: {})", cmd_idx++, commands->size(), std::string(EnumNameCommandType(cmd->cmd_type())));
            execute(cmd);
        }

        return true;
    } catch (const std::exception& e) {
        log_fatal(e.what());
        return false;
    }
}


}  // namespace v0
}  // namespace tt::tt_metal
