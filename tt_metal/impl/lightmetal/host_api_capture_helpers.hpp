#pragma once

#include <cstdint>
#include "lightmetal_capture_context.hpp"
#include "command_generated.h"
#include "tt_metal/common/logger.hpp"

// KCM - Temporary hack for bringup.
#define ENABLE_TRACING 1

#ifdef ENABLE_TRACING
    #define TRACE_FUNCTION_CALL(capture_func, ...) \
        do { \
            if (LightMetalCaptureContext::getInstance().isTracing()) { \
                capture_func(__VA_ARGS__); \
            } \
        } while (0)
#else
    #define TRACE_FUNCTION_CALL(capture_func, ...) do { } while (0)
#endif

//////////////////////////////////////////////////////////////
// Debug Code                                               //
//////////////////////////////////////////////////////////////

inline void PrintHostDataType(const HostDataType& data) {
    std::visit([](const auto& value) {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, const std::shared_ptr<std::vector<uint8_t>>>) {
            log_info(tt::LogMetalTrace, "HostDataType contains: std::shared_ptr<std::vector<uint8_t>>");
        } else if constexpr (std::is_same_v<T, const std::shared_ptr<std::vector<uint16_t>>>) {
            log_info(tt::LogMetalTrace, "HostDataType contains: std::shared_ptr<std::vector<uint16_t>>");
        } else if constexpr (std::is_same_v<T, const std::shared_ptr<std::vector<int32_t>>>) {
            log_info(tt::LogMetalTrace, "HostDataType contains: std::shared_ptr<std::vector<int32_t>>");
        } else if constexpr (std::is_same_v<T, const std::shared_ptr<std::vector<uint32_t>>>) {
            log_info(tt::LogMetalTrace, "HostDataType contains: std::shared_ptr<std::vector<uint32_t>>");
        } else if constexpr (std::is_same_v<T, const std::shared_ptr<std::vector<float>>>) {
            log_info(tt::LogMetalTrace, "HostDataType contains: std::shared_ptr<std::vector<float>>");
        } else if constexpr (std::is_same_v<T, const std::shared_ptr<std::vector<bfloat16>>>) {
            log_info(tt::LogMetalTrace, "HostDataType contains: std::shared_ptr<std::vector<bfloat16>>");
        } else if constexpr (std::is_same_v<T, const void*>) {
            log_info(tt::LogMetalTrace, "HostDataType contains: const void*");
        } else {
            log_info(tt::LogMetalTrace, "HostDataType contains: Unknown type");
        }
    }, data);
}

//////////////////////////////////////////////////////////////
// To-flatbuffer helper functions                           //
//////////////////////////////////////////////////////////////

// Original types defined in buffer_constants.hpp
inline tt::target::BufferType toFlatbuffer(BufferType type) {
    switch(type) {
        case BufferType::DRAM: return tt::target::BufferType::DRAM;
        case BufferType::L1: return tt::target::BufferType::L1;
        case BufferType::SYSTEM_MEMORY: return tt::target::BufferType::SystemMemory;
        case BufferType::L1_SMALL: return tt::target::BufferType::L1Small;
        case BufferType::TRACE: return tt::target::BufferType::Trace;
    }
}

// Original types defined in buffer_constants.hpp
inline tt::target::TensorMemoryLayout toFlatbuffer(TensorMemoryLayout layout) {
    switch(layout) {
        case TensorMemoryLayout::INTERLEAVED: return tt::target::TensorMemoryLayout::Interleaved;
        case TensorMemoryLayout::SINGLE_BANK: return tt::target::TensorMemoryLayout::SingleBank;
        case TensorMemoryLayout::HEIGHT_SHARDED: return tt::target::TensorMemoryLayout::HeightSharded;
        case TensorMemoryLayout::WIDTH_SHARDED: return tt::target::TensorMemoryLayout::WidthSharded;
        case TensorMemoryLayout::BLOCK_SHARDED: return tt::target::TensorMemoryLayout::BlockSharded;
    }
}

//////////////////////////////////////////////////////////////
// Host API tracing helper functions                        //
//////////////////////////////////////////////////////////////

// Generic helper to build command and add to vector of cmds (CQ)
inline void captureCommand(tt::target::CommandType cmd_type, ::flatbuffers::Offset<void> fb_offset) {
    auto& ctx = LightMetalCaptureContext::getInstance();
    // FIXME - Handle device_id.
    ctx.getCmdsVector().push_back(tt::target::CreateCommand(ctx.getBuilder(), cmd_type ,fb_offset));
}

inline void captureReplayTrace(Device *device, uint8_t cq_id, uint32_t tid, bool blocking) {
    auto& ctx = LightMetalCaptureContext::getInstance();
    if (!ctx.isTracing()) return;
    log_info(tt::LogMetalTrace, "captureReplayTrace: cq_id: {}, tid: {}, blocking: {}", cq_id, tid, blocking);
    auto cmd_variant = tt::target::CreateReplayTraceCommand(ctx.getBuilder(), cq_id, tid, blocking);
    captureCommand(tt::target::CommandType::ReplayTraceCommand, cmd_variant.Union());
}

inline void captureEnqueueTrace(CommandQueue& cq, uint32_t trace_id, bool blocking) {
    auto& ctx = LightMetalCaptureContext::getInstance();
    if (!ctx.isTracing()) return;
    log_info(tt::LogMetalTrace, "captureEnqueueTrace: cq_id: {}, trace_id: {}, blocking: {}", cq.id(), trace_id, blocking);
    auto cmd_variant = tt::target::CreateEnqueueTraceCommand(ctx.getBuilder(), cq.id(), trace_id, blocking);
    captureCommand(tt::target::CommandType::EnqueueTraceCommand, cmd_variant.Union());
}

inline void captureLoadTrace(Device *device, const uint8_t cq_id, const uint32_t tid) {
    auto& ctx = LightMetalCaptureContext::getInstance();
    if (!ctx.isTracing()) return;
    log_info(tt::LogMetalTrace, "{}: cq_id: {}, tid: {}", __FUNCTION__, cq_id, tid);
    auto cmd_variant = tt::target::CreateLoadTraceCommand(ctx.getBuilder(), tid, cq_id);
    captureCommand(tt::target::CommandType::LoadTraceCommand, cmd_variant.Union());
}

inline void captureReleaseTrace(Device *device, uint32_t tid) {
    auto& ctx = LightMetalCaptureContext::getInstance();
    if (!ctx.isTracing()) return;
    log_info(tt::LogMetalTrace, "captureReleaseTrace: tid: {}", tid);
    captureCommand(tt::target::CommandType::ReleaseTraceCommand, tt::target::CreateReleaseTraceCommand(ctx.getBuilder(), tid).Union());
}

// FIXME - Seems better idea to pass Buffer* to capture functions intead so it's clear we don't extend lifetime of buffer?
inline void captureCreateBuffer(std::shared_ptr<Buffer> buffer, const InterleavedBufferConfig &config) {
    auto& ctx = LightMetalCaptureContext::getInstance();
    if (!ctx.isTracing()) return;

    uint32_t buffer_global_id = ctx.addToMap(buffer.get());
    log_info(tt::LogMetalTrace, "{}: size: {} page_size: {} buffer_type: {} buffer_layout: {} buffer_global_id: {}",
        __FUNCTION__, config.size, config.page_size, config.buffer_type, config.buffer_layout, buffer_global_id);

    assert (config.device->id() == 0 && "multichip not supported yet");
    auto buffer_config_offset = tt::target::CreateInterleavedBufferConfig(ctx.getBuilder(), config.device->id(), config.size, config.page_size, toFlatbuffer(config.buffer_type), toFlatbuffer(config.buffer_layout));
    auto cmd_variant = tt::target::CreateCreateBufferCommand(ctx.getBuilder(), buffer_global_id, buffer_config_offset);
    captureCommand(tt::target::CommandType::CreateBufferCommand, cmd_variant.Union());
}

inline void captureDeallocateBuffer(Buffer *buffer) {
    auto& ctx = LightMetalCaptureContext::getInstance();
    if (!ctx.isTracing()) return;
    auto buffer_global_id = ctx.getGlobalId(buffer);
    log_info(tt::LogMetalTrace, "{} : buffer_global_id: {} size: {} address: {}", __FUNCTION__, buffer_global_id, buffer->size(), buffer->address());
    captureCommand(tt::target::CommandType::DeallocateBufferCommand,
        tt::target::CreateDeallocateBufferCommand(ctx.getBuilder(), buffer_global_id).Union());
}


inline void captureEnqueueWriteBuffer(CommandQueue& cq, std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>> buffer, HostDataType src, bool blocking) {
    auto& ctx = LightMetalCaptureContext::getInstance();
    if (!ctx.isTracing()) return;

    // We don't want to use shared_ptr to extend lifetime of buffer when adding to global_id map.
    Buffer* buffer_ptr = std::holds_alternative<std::shared_ptr<Buffer>>(buffer)
                             ? std::get<std::shared_ptr<Buffer>>(buffer).get()
                             : &std::get<std::reference_wrapper<Buffer>>(buffer).get();

    uint32_t cq_global_id = cq.id(); // FIXME - Maybe not correct, probably should handle same way as Buffers.
    uint32_t buffer_global_id = ctx.getGlobalId(buffer_ptr);

    log_info(tt::LogMetalTrace, "{} for cq_global_id: {} buffer_global_id: {}", __FUNCTION__, cq_global_id, buffer_global_id);
    // PrintHostDataType(src);

    // FIXME - Currently support limited data formats. Long term we might not store data in flatbuffer,
    // but have it provided at runtime so just do what's easiest here and support few types for now.
    ::flatbuffers::Offset<::flatbuffers::Vector<uint32_t>> src_vector;
    if (auto* uint32_vec = std::get_if<const std::shared_ptr<std::vector<uint32_t>>>(&src)) {
        src_vector = ctx.getBuilder().CreateVector(**uint32_vec);
    } else if (auto* uint16_vec = std::get_if<const std::shared_ptr<std::vector<uint16_t>>>(&src)) {
        // Convert uint16_t to uint32_t before creating the FlatBuffers vector
        std::vector<uint32_t> converted(uint16_vec->get()->begin(), uint16_vec->get()->end());
        src_vector = ctx.getBuilder().CreateVector(converted);
    } else if (auto* void_ptr = std::get_if<const void*>(&src)) {
        // Assuming the void* points to a buffer of uint32_t values. Infer size, cast to uint32_t.
        size_t num_elements = buffer_ptr->size() / sizeof(uint32_t);
        auto uint32_data = static_cast<const uint32_t*>(*void_ptr);
        src_vector = ctx.getBuilder().CreateVector(uint32_data, num_elements);
    } else {
        throw std::runtime_error("Unsupported HostDataType for captureEnqueueWriteBuffer()");
    }

    auto cmd_variant = tt::target::CreateEnqueueWriteBufferCommand(ctx.getBuilder(), cq_global_id, buffer_global_id, src_vector, blocking);
    captureCommand(tt::target::CommandType::EnqueueWriteBufferCommand, cmd_variant.Union());
}

inline void captureEnqueueReadBuffer(CommandQueue& cq, std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>> buffer, void* dst, bool blocking) {
    auto& ctx = LightMetalCaptureContext::getInstance();
    if (!ctx.isTracing()) return;

    // We don't want to use shared_ptr to extend lifetime of buffer when adding to global_id map.
    Buffer* buffer_ptr = std::holds_alternative<std::shared_ptr<Buffer>>(buffer)
                             ? std::get<std::shared_ptr<Buffer>>(buffer).get()
                             : &std::get<std::reference_wrapper<Buffer>>(buffer).get();

    uint32_t cq_global_id = cq.id(); // FIXME - Maybe not correct, probably should handle same way as Buffers.
    uint32_t buffer_global_id = ctx.getGlobalId(buffer_ptr);

    log_info(tt::LogMetalTrace, "{} for cq_global_id: {} buffer_global_id: {}", __FUNCTION__, cq_global_id, buffer_global_id);

    // Idea store a read_global_id to keep track of read results.
    auto cmd_variant = tt::target::CreateEnqueueReadBufferCommand(ctx.getBuilder(), cq_global_id, buffer_global_id, blocking);
    captureCommand(tt::target::CommandType::EnqueueReadBufferCommand, cmd_variant.Union());
}

inline void captureFinish(CommandQueue& cq) {
    auto& ctx = LightMetalCaptureContext::getInstance();
    if (!ctx.isTracing()) return;
    uint32_t cq_global_id = cq.id(); // FIXME - Maybe not correct, probably should handle same way as Buffers.
    log_info(tt::LogMetalTrace, "{} for cq_global_id: {}", __FUNCTION__, cq_global_id);
    auto cmd_variant = tt::target::CreateFinishCommand(ctx.getBuilder(), cq_global_id);
    captureCommand(tt::target::CommandType::FinishCommand, cmd_variant.Union());
}

inline void captureCreateProgram(Program& program) {
    auto& ctx = LightMetalCaptureContext::getInstance();
    if (!ctx.isTracing()) return;
    uint32_t program_global_id = ctx.addToMap(&program);
    log_info(tt::LogMetalTrace, "captureCreateProgram: program_global_id: {}", program_global_id);

    auto cmd_variant = tt::target::CreateCreateProgramCommand(ctx.getBuilder(), program_global_id);
    captureCommand(tt::target::CommandType::CreateProgramCommand, cmd_variant.Union());
}
