#include <tensor/tensor.hpp>
#include <tensor/host_buffer.hpp>

#include <random>

namespace tt {

namespace numpy {

using Shape = std::array<uint32_t, 4>;

using tt_metal::Tensor;
using tt_metal::DataType;
using tt_metal::Initialize;
using tt_metal::Layout;

static Tensor zeros(const Shape& shape, const DataType data_type = DataType::BFLOAT16) {
    return Tensor(shape, Initialize::ZEROS, data_type, Layout::ROW_MAJOR);
}

static Tensor zeros_like(const Tensor& input_tensor, std::optional<DataType> data_type = std::nullopt) {
    DataType data_type_to_use = input_tensor.dtype();
    if (data_type.has_value()) {
        data_type_to_use = data_type.value();
    }
    auto output_tensor = zeros(input_tensor.shape(), data_type_to_use).to(input_tensor.layout());
    if (input_tensor.device() != nullptr) {
        output_tensor = output_tensor.to(input_tensor.device());
    }
    return output_tensor;
}

namespace random {

inline auto RANDOM_GENERATOR = std::mt19937(0);

static void seed(std::size_t seed) {
    RANDOM_GENERATOR = std::mt19937(seed);
}

static Tensor uniform(bfloat16 low, bfloat16 high, const Shape& shape) {

    auto rand_float = std::bind(std::uniform_real_distribution<float>(low.to_float(), high.to_float()), RANDOM_GENERATOR);

    auto volume = shape[0] * shape[1] * shape[2] * shape[3];

    auto output_buffer = host_buffer::create<bfloat16>(volume);
    auto output_view = host_buffer::view_as<bfloat16>(output_buffer);
    for (auto index = 0; index < output_view.size(); index++) {
        output_view[index] = bfloat16(rand_float());
    }

    return Tensor(output_buffer, shape, DataType::BFLOAT16, Layout::ROW_MAJOR);
}

static Tensor random(const Shape& shape, const DataType data_type) {
    TT_ASSERT(data_type ==  DataType::BFLOAT16);
    return uniform(bfloat16(0.0f), bfloat16(1.0f), shape);
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
