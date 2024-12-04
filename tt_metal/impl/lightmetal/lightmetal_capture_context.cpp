#include "tt_metal/common/logger.hpp"
#include "lightmetal_capture_context.hpp"
#include "flatbuffers/flatbuffers.h"
#include "command_generated.h"
#include "binary_generated.h"
#include "tt_metal/impl/buffers/buffer.hpp"

#include <iostream>
#include <fstream>

namespace tt::tt_metal {
inline namespace v0 {

LightMetalCaptureContext::LightMetalCaptureContext()
    : tracing_(false), builder_() {}

LightMetalCaptureContext& LightMetalCaptureContext::getInstance() {
    static LightMetalCaptureContext instance;
    return instance;
}

bool LightMetalCaptureContext::isTracing() const {
    return tracing_;
}

void LightMetalCaptureContext::setTracing(bool tracing) {
    tracing_ = tracing;
}

flatbuffers::FlatBufferBuilder& LightMetalCaptureContext::getBuilder() {
    return builder_;
}

std::vector<flatbuffers::Offset<tt::target::Command>>& LightMetalCaptureContext::getCmdsVector() {
    return cmdsVector_;
}

std::vector<flatbuffers::Offset<tt::target::lightmetal::TraceDescriptorByTraceId>>& LightMetalCaptureContext::getTraceDescsVector() {
    return traceDescsVector_;
}

// Create final flatbuffer binary from the built up data and return to caller as blob.
// If light_metal_binary itself (flatbuffer object) is of interest, could return it instead.
std::vector<uint8_t> LightMetalCaptureContext::createLightMetalBinary() {
    auto commands = builder_.CreateVector(cmdsVector_);
    auto sorted_trace_descs = builder_.CreateVectorOfSortedTables(&traceDescsVector_);
    auto light_metal_binary = CreateLightMetalBinary(builder_, commands, sorted_trace_descs);
    builder_.Finish(light_metal_binary);

    const uint8_t* buffer_ptr = builder_.GetBufferPointer();
    size_t buffer_size = builder_.GetSize();
    return {buffer_ptr, buffer_ptr + buffer_size};
}

void LightMetalCaptureContext::reset() {
    builder_.Clear();
    cmdsVector_.clear();
}

uint32_t LightMetalCaptureContext::getBufferGlobalId(Buffer* buffer) {
    auto it = bufferToGlobalIdMap_.find(buffer);
    if (it != bufferToGlobalIdMap_.end()) {
        return it->second;
    }

    uint32_t global_id = nextGlobalId_++;
    bufferToGlobalIdMap_[buffer] = global_id;
    return global_id;
}

////////////////////////////////////////////
// Non-Class Helper Functions             //
////////////////////////////////////////////

bool writeBinaryBlobToFile(const std::string& filename, const std::vector<uint8_t>& blob) {
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile) {
        std::cerr << "Unable to open file: " << filename << " for writing." << std::endl;
        return false;
    }

    if (!outFile.write(reinterpret_cast<const char*>(blob.data()), blob.size())) {
        std::cerr << "Failed to write binary data to file: " << filename << std::endl;
        return false;
    }

    return true;
}


}  // namespace v0
}  // namespace tt::tt_metal
