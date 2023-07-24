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

Shape::Shape(const std::initializer_list<uint32_t> data) : rank_(data.size()) {
    std::copy(std::begin(data), std::end(data), std::begin(this->data_));
}
Shape::Shape(const std::array<uint32_t, 4>& data) : rank_(data.size()) {
    std::copy(std::begin(data), std::end(data), std::begin(this->data_));
}
Shape::Shape(const std::vector<uint32_t>& data) : rank_(data.size()) {
    std::copy(std::begin(data), std::end(data), std::begin(this->data_));
}

uint32_t Shape::rank() const { return this->rank_; }

uint32_t& Shape::operator[](const std::size_t index) { return this->data_[index]; }
const uint32_t& Shape::operator[](const std::size_t index) const { return this->data_[index]; }

uint32_t& Shape::back() { return this->data_[this->rank_ - 1]; }
const uint32_t& Shape::back() const { return this->data_[this->rank_ - 1]; }

const uint32_t* Shape::begin() const { return this->data_.data(); }
const uint32_t* Shape::end() const { return this->data_.data() + this->rank_; }


bool operator==(const Shape& shape_a, const Shape& shape_b) {
    if (shape_a.rank() != shape_b.rank()) {
        return false;
    }
    for (auto index = 0; index < shape_a.rank(); index++) {
        if ( shape_a[index] != shape_b[index]) {
            return false;
        }
    }
    return true;
}

bool operator!=(const Shape& shape_a, const Shape& shape_b) {
    return not (shape_a == shape_b);
}

std::ostream& operator<<(std::ostream& os, const Shape& shape) {
    os << "{";
    for (auto index = 0; index < shape.rank(); index++) {
        os << shape[index];
        if (index != shape.rank() - 1) {
            os << ", ";
        }
    }
    os << "}";
    return os;
}



tt::stl::reflection::Attributes MemoryConfig::attributes() const {
    return {
        {"interleaved", this->interleaved},
        {"buffer_type", this->buffer_type},
    };
}


tt::stl::reflection::Attributes OwnedStorage::attributes() const {
    return {};
}


tt::stl::reflection::Attributes DeviceStorage::attributes() const {
    return {
        {"memory_config", this->memory_config},
    };
}


tt::stl::reflection::Attributes BorrowedStorage::attributes() const {
    return {};
}

}  // namespace tt_metal

}  // namespace tt
