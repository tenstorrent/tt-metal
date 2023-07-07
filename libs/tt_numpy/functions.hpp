#include <tensor/tensor.hpp>
#include <tensor/tensor_utils.hpp>
#include <tensor/host_buffer.hpp>

#include <optional>
#include <random>

namespace tt {

namespace numpy {

using Shape = std::array<uint32_t, 4>;

using tt_metal::Tensor;
using tt_metal::DataType;
using tt_metal::Layout;

namespace detail
{

template<typename T>
constexpr static DataType get_data_type() {
        if constexpr (std::is_same_v<T, uint32_t>) {
        return DataType::UINT32;
    }
    else if constexpr (std::is_same_v<T, float>) {
        return DataType::FLOAT32;
    }
    else if constexpr (std::is_same_v<T, bfloat16>) {
        return DataType::BFLOAT16;
    }
    else {
        TT_THROW("Unsupported DataType!");
    }
}

template<typename T>
static Tensor full(const Shape& shape, T value) {
    constexpr DataType data_type = detail::get_data_type<T>();
    auto host_buffer  = host_buffer::create<T>(tt_metal::volume(shape));
    auto host_view = host_buffer::view_as<T>(host_buffer);
    std::fill(std::begin(host_view), std::end(host_view), value);
    return Tensor(HostStorage{host_buffer}, shape, data_type, Layout::ROW_MAJOR);
}

} // namespace detail

template<typename T>
static Tensor full(const Shape& shape, const T value, const DataType data_type) {
    switch (data_type) {
        case DataType::UINT32: {
            return detail::full<uint32_t>(shape, value);
        }
        case DataType::FLOAT32: {
            return detail::full<float>(shape, value);
        }
        case DataType::BFLOAT16: {
            return detail::full<bfloat16>(shape, bfloat16(value));
        }
        default:
            TT_THROW("Unsupported DataType!");
    }
}

static Tensor zeros(const Shape& shape, const DataType data_type = DataType::BFLOAT16) {
    return full(shape, 0, data_type);
}

static Tensor ones(const Shape& shape, const DataType data_type = DataType::BFLOAT16) {
    return full(shape, 1, data_type);
}

static Tensor zeros_like(const Tensor& input_tensor, std::optional<DataType> data_type = std::nullopt) {
    DataType data_type_to_use = input_tensor.dtype();
    if (data_type.has_value()) {
        data_type_to_use = data_type.value();
    }
    auto output_tensor = zeros(input_tensor.shape(), data_type_to_use);
    output_tensor = output_tensor.to(input_tensor.layout());
    if (input_tensor.device() != nullptr) {
        output_tensor = output_tensor.to(input_tensor.device());
    }
    return output_tensor;
}

template<typename T>
static Tensor arange(int64_t start, int64_t stop, int64_t step) {
    constexpr DataType data_type = detail::get_data_type<T>();
    // Current implementation restrictions
    TT_ASSERT(step > 0, "Step must be greater than 0");
    TT_ASSERT(start < stop, "Start must be less than step");
    auto size = divup((stop - start), step);
    auto host_buffer  = host_buffer::create<T>(size);
    auto host_view = host_buffer::view_as<T>(host_buffer);

    auto index = 0;
    for (auto value = start; value < stop; value += step) {
        if constexpr (std::is_same_v<T, bfloat16>) {
         host_view[index++] = static_cast<T>(static_cast<float>(value));
        } else {
         host_view[index++] = static_cast<T>(value);
        }
    }
    return Tensor(HostStorage{host_buffer}, {1, 1, 1, static_cast<uint32_t>(size)}, data_type, Layout::ROW_MAJOR);
}

namespace random {

inline auto RANDOM_GENERATOR = std::mt19937(0);

static void seed(std::size_t seed) {
    RANDOM_GENERATOR = std::mt19937(seed);
}

template<typename T>
static Tensor uniform(T low, T high, const Shape& shape) {
    constexpr DataType data_type = detail::get_data_type<T>();

    auto output_buffer = host_buffer::create<T>(tt_metal::volume(shape));
    auto output_view = host_buffer::view_as<T>(output_buffer);

    if constexpr (std::is_same_v<T, uint32_t> ) {
        auto rand_value = std::bind(std::uniform_int_distribution<T>(low, high), RANDOM_GENERATOR);
        for (auto index = 0; index < output_view.size(); index++) {
            output_view[index] = rand_value();
        }
    } else if constexpr (std::is_same_v<T, float>) {
        auto rand_value = std::bind(std::uniform_real_distribution<T>(low, high), RANDOM_GENERATOR);
        for (auto index = 0; index < output_view.size(); index++) {
            output_view[index] = rand_value();
        }
    } else if constexpr (std::is_same_v<T, bfloat16>) {
        auto rand_value = std::bind(std::uniform_real_distribution<float>(low.to_float(), high.to_float()), RANDOM_GENERATOR);
        for (auto index = 0; index < output_view.size(); index++) {
            output_view[index] = bfloat16(rand_value());
        }
    }

    return Tensor(HostStorage{output_buffer}, shape, data_type, Layout::ROW_MAJOR);
}

static Tensor random(const Shape& shape, const DataType data_type = DataType::BFLOAT16) {
    switch (data_type)
    {
        case DataType::UINT32:
            return uniform(0u, 1u, shape);
        case DataType::FLOAT32:
            return uniform(0.0f, 1.0f, shape);
        case DataType::BFLOAT16:
            return uniform(bfloat16(0.0f), bfloat16(1.0f), shape);
        default:
            TT_THROW("Unsupported DataType!");
    };
}

}

namespace detail {
static bool nearly_equal(float a, float b, float epsilon = 1e-5f, float abs_threshold = 1e-5f) {
  auto diff = std::abs(a-b);
  auto norm = std::min((std::abs(a) + std::abs(b)), std::numeric_limits<float>::max());
  auto result = diff < std::max(abs_threshold, epsilon * norm);
  if (not result) {
    tt::log_error(tt::LogTest, "{} != {}", a, b);
  }
  return result;
}

template<typename ... Args>
static bool nearly_equal(bfloat16 a, bfloat16 b, Args ...  args) {
  return nearly_equal(a.to_float(), b.to_float(), args...);
}
}

template<typename DataType, typename ... Args>
static bool allclose(const Tensor& tensor_a, const Tensor& tensor_b, Args ...  args) {

    if (tensor_a.shape() != tensor_b.shape()) {
        return false;
    }

    if (tensor_a.dtype() != tensor_b.dtype()) {
        return false;
    }

    auto tensor_a_data = host_buffer::view_as<DataType>(tensor_a);
    auto tensor_b_data = host_buffer::view_as<DataType>(tensor_b);

    for (int index = 0; index < tensor_a_data.size(); index++) {
        if (not detail::nearly_equal(tensor_a_data[index], tensor_b_data[index], args...)) {
            return false;
        }
    }
    return true;
}


}
}
