#include "tensor/types.hpp"

#include "third_party/magic_enum/magic_enum.hpp"

#include <boost/core/demangle.hpp>

namespace tt {

namespace tt_metal {

tt::DataFormat datatype_to_dataformat_converter(tt::tt_metal::DataType datatype) {
    switch (datatype) {
        case tt::tt_metal::DataType::BFLOAT16: return tt::DataFormat::Float16_b;
        case tt::tt_metal::DataType::BFLOAT8_B: return tt::DataFormat::Bfp8_b;
        case tt::tt_metal::DataType::FLOAT32: return tt::DataFormat::Float32;
        case tt::tt_metal::DataType::UINT32: return tt::DataFormat::UInt32;
        default:
            TT_ASSERT(false, "Unsupported DataType");
            return tt::DataFormat::Float16_b;
    }
}

tt::stl::reflection::Attributes MemoryConfig::attributes() const {
    return {
        {"interleaved", this->interleaved},
        {"buffer_type", this->buffer_type}
    };
}


tt::stl::reflection::Attributes HostStorage::attributes() const {
    return {};
}


tt::stl::reflection::Attributes DeviceStorage::attributes() const {
    return {
        {"memory_config", this->memory_config}
    };
}

}  // namespace tt_metal

}  // namespace tt
