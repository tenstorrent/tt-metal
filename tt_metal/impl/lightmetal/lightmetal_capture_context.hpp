#pragma once

#include <vector>
#include <memory>
#include <flatbuffers/flatbuffers.h>

// Forward decl for command_generated.h
namespace tt::target {
    class Command;
}

// Forward decl for binary_generated.h
namespace tt::target::lightmetal {
    struct TraceDescriptorByTraceId;
}

namespace tt::tt_metal { // KCM Consider adding lightmetal namespace.
inline namespace v0 {

class Buffer;

class LightMetalCaptureContext {
public:
    static LightMetalCaptureContext& getInstance();

    bool isTracing() const;
    void setTracing(bool tracing);

    flatbuffers::FlatBufferBuilder& getBuilder();
    std::vector<flatbuffers::Offset<tt::target::Command>>& getCmdsVector();
    std::vector<flatbuffers::Offset<tt::target::lightmetal::TraceDescriptorByTraceId>>& getTraceDescsVector();
    std::vector<uint8_t> createLightMetalBinary();

    // Object maps public accessors
    uint32_t getBufferGlobalId(Buffer* buffer);

    void reset();

private:
    LightMetalCaptureContext(); // Private constructor

    bool tracing_;
    flatbuffers::FlatBufferBuilder builder_;
    std::vector<flatbuffers::Offset<tt::target::Command>> cmdsVector_;
    std::vector<flatbuffers::Offset<tt::target::lightmetal::TraceDescriptorByTraceId>> traceDescsVector_;

    // Object maps for associating each object with a global_id
    uint32_t nextGlobalId_ = 0; // Shared across all object types.
    std::unordered_map<Buffer*, uint32_t> bufferToGlobalIdMap_;
    // FIXME - Add one for CommandQueue object.

    // Delete copy constructor and assignment operator
    LightMetalCaptureContext(const LightMetalCaptureContext&) = delete;
    LightMetalCaptureContext& operator=(const LightMetalCaptureContext&) = delete;
};


bool writeBinaryBlobToFile(const std::string& filename, const std::vector<uint8_t>& blob);

}  // namespace v0
}  // namespace tt::tt_metal
