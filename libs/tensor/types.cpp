#include "tensor/types.hpp"

#include "third_party/magic_enum/magic_enum.hpp"

#include <boost/core/demangle.hpp>

namespace tt {

namespace tt_metal {

tt::stl::reflection::Attributes MemoryConfig::attributes() const {
    return {
        {"interleaved", fmt::format("{}", this->interleaved)},
        {"buffer_type", fmt::format("{}", this->buffer_type)}
    };
}


tt::stl::reflection::Attributes HostStorage::attributes() const {
    return {
    };
}


tt::stl::reflection::Attributes DeviceStorage::attributes() const {
    return {
        {"memory_config", fmt::format("{}", this->memory_config)}
    };
}

}  // namespace tt_metal

}  // namespace tt
