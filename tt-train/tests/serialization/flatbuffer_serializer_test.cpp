// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <bit>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <ostream>
#include <random>
#include <string>
#include <tt-metalium/bfloat16.hpp>
#include <ttnn/tensor/serialization.hpp>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/random.hpp"
#include "core/tt_tensor_utils.hpp"
#include "core/ttnn_all_includes.hpp"
#include "serialization/serialization.hpp"

namespace {
// Concept for trivially copyable types
template <typename T>
concept TriviallyCopyable = std::is_trivially_copyable_v<T>;

// Helper function to convert shape to bytes (similar to serialization.cpp)
template <TriviallyCopyable T>
std::span<const uint8_t> to_bytes(const T& value) {
    auto ptr = reinterpret_cast<const uint8_t*>(&value);
    return std::span<const uint8_t>(ptr, sizeof(T));
}

// Specialization for ttnn::Shape (not trivially copyable, handled specially)
inline std::span<const uint8_t> to_bytes(const ttnn::Shape& value) {
    auto ptr = reinterpret_cast<const uint8_t*>(value.view().data());
    return std::span<const uint8_t>(ptr, sizeof(value[0]) * value.rank());
}

std::string generate_unique_temp_dir_name() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 15);

    constexpr int name_length = 16;
    std::string name = "flatbuffer_test_";

    for (int i = 0; i < name_length; ++i) {
        name += "0123456789abcdef"[dis(gen)];
    }

    return name;
}

std::filesystem::path create_unique_temp_dir() {
    std::filesystem::path base_dir = std::filesystem::temp_directory_path();

    size_t max_attempts = 1024;
    while (--max_attempts > 0) {
        std::string random_name = generate_unique_temp_dir_name();
        std::filesystem::path temp_dir = base_dir / random_name;

        if (!std::filesystem::exists(temp_dir)) {
            std::filesystem::create_directories(temp_dir);
            return temp_dir;
        }
    }

    throw std::runtime_error("Failed to create unique temporary directory after maximum attempts");
}
}  // namespace

class FlatBufferFileTest : public ::testing::Test {
protected:
    void SetUp() override {
        temp_dir = create_unique_temp_dir();
        test_filename = temp_dir.string();  // Use directory path
    }

    void TearDown() override {
        if (std::filesystem::exists(temp_dir)) {
            std::filesystem::remove_all(temp_dir);
        }
    }

    std::filesystem::path temp_dir;
    std::string test_filename;
};

TEST_F(FlatBufferFileTest, SerializeDeserializePrimitives) {
    ttml::serialization::FlatBufferFile serializer;

    // Put primitive data
    serializer.put("int_key", 42);
    serializer.put("float_key", 3.14F);
    serializer.put("double_key", 2.71828);
    serializer.put("uint_key", static_cast<uint32_t>(123456789));
    serializer.put("string_key", "Hello, World!");

    // Serialize to directory
    ASSERT_NO_THROW(serializer.serialize(test_filename));

    // Verify metadata file exists
    std::filesystem::path metadata_file = std::filesystem::path(test_filename) / "metadata.flatbuffer";
    ASSERT_TRUE(std::filesystem::exists(metadata_file)) << "Metadata file should exist: " << metadata_file;

    // Deserialize from directory
    ttml::serialization::FlatBufferFile deserializer;
    ASSERT_NO_THROW(deserializer.deserialize(test_filename));

    // Get and check data
    int int_value = 0;
    EXPECT_NO_THROW(int_value = deserializer.get_int("int_key"));
    EXPECT_EQ(int_value, 42);

    float float_value = 0;
    EXPECT_NO_THROW(float_value = deserializer.get_float("float_key"));
    EXPECT_FLOAT_EQ(float_value, 3.14f);

    double double_value = 0;
    EXPECT_NO_THROW(double_value = deserializer.get_double("double_key"));
    EXPECT_DOUBLE_EQ(double_value, 2.71828);

    uint32_t uint_value = 0;
    EXPECT_NO_THROW(uint_value = deserializer.get_uint32("uint_key"));
    EXPECT_EQ(uint_value, 123456789U);

    std::string string_value;
    EXPECT_NO_THROW(string_value = deserializer.get_string("string_key"));
    EXPECT_EQ(string_value, "Hello, World!");
}

TEST_F(FlatBufferFileTest, MissingKeyThrows) {
    ttml::serialization::FlatBufferFile serializer;
    serializer.put("int_key", 42);
    ASSERT_NO_THROW(serializer.serialize(test_filename));
    ttml::serialization::FlatBufferFile deserializer;
    ASSERT_NO_THROW(deserializer.deserialize(test_filename));

    [[maybe_unused]] int unused = 0;
    EXPECT_ANY_THROW(unused = deserializer.get_int("nonexistent_key"));
}

TEST_F(FlatBufferFileTest, TypeMismatchThrows) {
    ttml::serialization::FlatBufferFile serializer;
    serializer.put("int_key", 42);
    serializer.serialize(test_filename);

    ttml::serialization::FlatBufferFile deserializer;
    deserializer.deserialize(test_filename);

    [[maybe_unused]] float unused = 0.F;
    EXPECT_ANY_THROW(unused = deserializer.get_float("int_key"));
}

TEST_F(FlatBufferFileTest, OverwriteExistingKey) {
    ttml::serialization::FlatBufferFile serializer;
    serializer.put("key", 42);
    serializer.put("key", "Overwritten");

    serializer.serialize(test_filename);

    ttml::serialization::FlatBufferFile deserializer;
    deserializer.deserialize(test_filename);

    std::string string_value;
    EXPECT_NO_THROW(string_value = deserializer.get_string("key"));
    EXPECT_EQ(string_value, "Overwritten");

    [[maybe_unused]] int unused = 0;
    EXPECT_ANY_THROW(unused = deserializer.get_int("key"));
}

TEST_F(FlatBufferFileTest, EmptySerializerSerialization) {
    ttml::serialization::FlatBufferFile serializer;
    ASSERT_NO_THROW(serializer.serialize(test_filename));

    ttml::serialization::FlatBufferFile deserializer;
    ASSERT_NO_THROW(deserializer.deserialize(test_filename));

    [[maybe_unused]] int unused = 0;
    EXPECT_ANY_THROW(unused = deserializer.get_int("any_key"));
}

TEST_F(FlatBufferFileTest, NonExistentFileDeserialization) {
    std::filesystem::create_directories(test_filename);
    ttml::serialization::FlatBufferFile deserializer;
    // Try to deserialize from non-existent directory
    EXPECT_THROW(deserializer.deserialize(test_filename + "/nonexistent"), std::runtime_error);
}

TEST_F(FlatBufferFileTest, InvalidDataDeserialization) {
    // Create directory and write invalid data to metadata file
    std::filesystem::create_directories(test_filename);
    std::filesystem::path metadata_file = std::filesystem::path(test_filename) / "metadata.flatbuffer";
    std::ofstream ofs(metadata_file, std::ios::binary);
    ofs << "Invalid Data";
    ofs.close();

    ttml::serialization::FlatBufferFile deserializer;
    EXPECT_ANY_THROW(deserializer.deserialize(test_filename));
}

TEST_F(FlatBufferFileTest, MultipleDataTypesSerialization) {
    ttml::serialization::FlatBufferFile serializer;

    serializer.put("int_key", 100);
    serializer.put("float_key", 1.23F);
    serializer.put("double_key", 4.56);
    serializer.put("string_key", "test string");

    std::vector<int> int_vec = {10, 20, 30};
    serializer.put("int_vector_key", std::span<const int>(int_vec));

    serializer.serialize(test_filename);

    ttml::serialization::FlatBufferFile deserializer;
    deserializer.deserialize(test_filename);

    int int_value = 0;
    EXPECT_NO_THROW(int_value = deserializer.get_int("int_key"));
    EXPECT_EQ(int_value, 100);

    float float_value = 0.F;
    EXPECT_NO_THROW(float_value = deserializer.get_float("float_key"));
    EXPECT_FLOAT_EQ(float_value, 1.23F);

    double double_value = 0.0;
    EXPECT_NO_THROW(double_value = deserializer.get_double("double_key"));
    EXPECT_DOUBLE_EQ(double_value, 4.56);

    std::string string_value;
    EXPECT_NO_THROW(string_value = deserializer.get_string("string_key"));
    EXPECT_EQ(string_value, "test string");
}

TEST_F(FlatBufferFileTest, BoolAndCharTypes) {
    ttml::serialization::FlatBufferFile serializer;

    serializer.put("bool_key", true);
    serializer.put("char_key", 'A');

    serializer.serialize(test_filename);

    ttml::serialization::FlatBufferFile deserializer;
    deserializer.deserialize(test_filename);

    bool bool_value = false;
    EXPECT_NO_THROW(bool_value = deserializer.get_bool("bool_key"));
    EXPECT_EQ(bool_value, true);

    char char_value = '\0';
    EXPECT_NO_THROW(char_value = deserializer.get_char("char_key"));
    EXPECT_EQ(char_value, 'A');
}

TEST_F(FlatBufferFileTest, SizeTType) {
    ttml::serialization::FlatBufferFile serializer;

    size_t size_value = 123456789012345ULL;
    serializer.put("size_key", size_value);

    serializer.serialize(test_filename);

    ttml::serialization::FlatBufferFile deserializer;
    deserializer.deserialize(test_filename);

    size_t result = 0;
    EXPECT_NO_THROW(result = deserializer.get_size_t("size_key"));
    EXPECT_EQ(result, size_value);
}

TEST_F(FlatBufferFileTest, UInt8Vector) {
    ttml::serialization::FlatBufferFile serializer;

    std::vector<uint8_t> uint8_vec = {0, 1, 2, 255, 128};
    serializer.put("uint8_vector_key", std::span<const uint8_t>(uint8_vec));

    serializer.serialize(test_filename);

    ttml::serialization::FlatBufferFile deserializer;
    deserializer.deserialize(test_filename);

    std::vector<uint8_t> result;
    EXPECT_NO_THROW(result = deserializer.get_vector_uint8("uint8_vector_key"));
    EXPECT_EQ(result, uint8_vec);
}

TEST_F(FlatBufferFileTest, CharVector) {
    ttml::serialization::FlatBufferFile serializer;

    std::vector<char> char_vec = {'a', 'b', 'c', 'z'};
    serializer.put("char_vector_key", std::span<const char>(char_vec));

    serializer.serialize(test_filename);
}

TEST_F(FlatBufferFileTest, BFloat16ScalarSerialization) {
    ttml::serialization::FlatBufferFile serializer;

    bfloat16 bf16_value(3.14159F);
    serializer.put("bf16_key", bf16_value);

    serializer.serialize(test_filename);

    ttml::serialization::FlatBufferFile deserializer;
    deserializer.deserialize(test_filename);

    bfloat16 result = deserializer.get_bfloat16("bf16_key");
    EXPECT_EQ(std::bit_cast<uint16_t>(result), std::bit_cast<uint16_t>(bf16_value));
}

TEST_F(FlatBufferFileTest, BFloat16BitExactPreservation) {
    ttml::serialization::FlatBufferFile serializer;

    // Test various bfloat16 values including edge cases
    std::vector<bfloat16> test_values = {
        bfloat16(0.0F),
        bfloat16(1.0F),
        bfloat16(-1.0F),
        bfloat16(3.14159F),
        bfloat16(1e10F),
        bfloat16(-1e10F),
        bfloat16(std::numeric_limits<float>::infinity()),
        bfloat16(-std::numeric_limits<float>::infinity()),
    };

    for (size_t i = 0; i < test_values.size(); ++i) {
        serializer.put("bf16_" + std::to_string(i), test_values[i]);
    }

    serializer.serialize(test_filename);

    ttml::serialization::FlatBufferFile deserializer;
    deserializer.deserialize(test_filename);

    for (size_t i = 0; i < test_values.size(); ++i) {
        bfloat16 result = deserializer.get_bfloat16("bf16_" + std::to_string(i));
        EXPECT_EQ(std::bit_cast<uint16_t>(result), std::bit_cast<uint16_t>(test_values[i]))
            << "Bit-exact preservation failed for value " << i;
    }
}

TEST_F(FlatBufferFileTest, BFloat16TypeMismatchThrows) {
    ttml::serialization::FlatBufferFile serializer;

    bfloat16 bf16_value(42.0F);
    serializer.put("bf16_key", bf16_value);
    serializer.serialize(test_filename);

    ttml::serialization::FlatBufferFile deserializer;
    deserializer.deserialize(test_filename);

    [[maybe_unused]] float unused = 0.F;
    EXPECT_ANY_THROW(unused = deserializer.get_float("bf16_key"));
}

namespace {
template <typename T>
std::vector<T> generate_random_vector(size_t size, uint32_t seed) {
    std::vector<T> data(size);

    if constexpr (std::is_floating_point_v<T>) {
        ttml::core::parallel_generate(
            std::span{data.data(), data.size()}, []() { return std::uniform_real_distribution<T>(-10.0, 10.0); }, seed);
    } else if constexpr (std::is_same_v<T, bfloat16>) {
        ttml::core::parallel_generate(
            std::span{data.data(), data.size()},
            []() { return std::uniform_real_distribution<float>(-10.0f, 10.0f); },
            seed);
    } else if constexpr (std::is_integral_v<T>) {
        ttml::core::parallel_generate(
            std::span{data.data(), data.size()},
            []() {
                return std::uniform_int_distribution<T>(
                    std::numeric_limits<T>::min() / 2, std::numeric_limits<T>::max() / 2);
            },
            seed);
    }

    return data;
}

// Create a random tensor with specified dtype, layout, and storage type
// shape: Tensor shape
// dtype: Data type (BFLOAT16, FLOAT32, UINT32, INT32, BFLOAT8_B, BFLOAT4_B)
// layout: Layout (ROW_MAJOR or TILE)
// storage_type: Storage type (HOST or DEVICE)
// seed: Random seed for reproducibility
// device: Target device for tensor creation
tt::tt_metal::Tensor create_random_tensor(
    const ttnn::Shape& shape,
    tt::tt_metal::DataType dtype,
    ttnn::Layout layout,
    tt::tt_metal::StorageType storage_type,
    uint32_t seed,
    ttnn::distributed::MeshDevice* device) {
    tt::tt_metal::Tensor tensor;

    switch (dtype) {
        case tt::tt_metal::DataType::BFLOAT16: {
            auto data = generate_random_vector<bfloat16>(shape.volume(), seed);
            tensor = ttml::core::from_vector<bfloat16, tt::tt_metal::DataType::BFLOAT16>(data, shape, device, layout);
            break;
        }
        case tt::tt_metal::DataType::FLOAT32: {
            auto data = generate_random_vector<float>(shape.volume(), seed);
            tensor = ttml::core::from_vector<float, tt::tt_metal::DataType::FLOAT32>(data, shape, device, layout);
            break;
        }
        case tt::tt_metal::DataType::UINT32: {
            auto data = generate_random_vector<uint32_t>(shape.volume(), seed);
            tensor = ttml::core::from_vector<uint32_t, tt::tt_metal::DataType::UINT32>(data, shape, device, layout);
            break;
        }
        case tt::tt_metal::DataType::INT32: {
            auto data = generate_random_vector<int32_t>(shape.volume(), seed);
            tensor = ttml::core::from_vector<int32_t, tt::tt_metal::DataType::INT32>(data, shape, device, layout);
            break;
        }
        case tt::tt_metal::DataType::BFLOAT8_B:
        case tt::tt_metal::DataType::BFLOAT4_B: {
            auto float_data = generate_random_vector<float>(shape.volume(), seed);
            auto float_tensor = ttml::core::from_vector<float, tt::tt_metal::DataType::FLOAT32>(
                float_data, shape, device, ttnn::Layout::TILE);
            auto cpu_float_tensor = float_tensor.cpu();
            auto converted_tensor = ttnn::to_dtype(cpu_float_tensor, dtype);
            tensor = converted_tensor.to_device(device);
            break;
        }
        default: throw std::runtime_error("Unsupported dtype for random tensor generation");
    }

    if (storage_type == tt::tt_metal::StorageType::HOST && tensor.storage_type() != tt::tt_metal::StorageType::HOST) {
        tensor = tensor.cpu();
    } else if (
        storage_type == tt::tt_metal::StorageType::DEVICE &&
        tensor.storage_type() != tt::tt_metal::StorageType::DEVICE) {
        tensor = tensor.to_device(device);
    }

    return tensor;
}

struct TensorTestCase {
    tt::tt_metal::DataType dtype;
    ttnn::Layout layout;
    tt::tt_metal::StorageType storage_type;
    std::string name;
};

using TestParam = TensorTestCase;

// Pretty printer for TensorTestCase - returns string
std::string to_string(const TensorTestCase& test_case) {
    std::string result;

    // Print dtype
    switch (test_case.dtype) {
        case tt::tt_metal::DataType::BFLOAT16: result = "BFLOAT16"; break;
        case tt::tt_metal::DataType::FLOAT32: result = "FLOAT32"; break;
        case tt::tt_metal::DataType::UINT32: result = "UINT32"; break;
        case tt::tt_metal::DataType::INT32: result = "INT32"; break;
        case tt::tt_metal::DataType::BFLOAT8_B: result = "BFLOAT8_B"; break;
        case tt::tt_metal::DataType::BFLOAT4_B: result = "BFLOAT4_B"; break;
        default: result = "Unknown"; break;
    }

    result += "_";

    // Print layout
    result += (test_case.layout == ttnn::Layout::ROW_MAJOR ? "ROW_MAJOR" : "TILE");

    result += "_";

    // Print storage type
    result += (test_case.storage_type == tt::tt_metal::StorageType::DEVICE ? "DEVICE" : "HOST");

    return result;
}

// Ostream operators for Google Test parameter printing
std::ostream& operator<<(std::ostream& os, const TensorTestCase& test_case) {
    return os << to_string(test_case);
}

}  // namespace

class FlatBufferFileSerializationTest : public FlatBufferFileTest, public ::testing::WithParamInterface<TestParam> {};

TEST_P(FlatBufferFileSerializationTest, ScopedTempDirWriteReadRoundTrip) {
    // get_device() will automatically open the device if it's not already open
    auto* device = &ttml::autograd::ctx().get_device();

    const TestParam& param = GetParam();
    const TensorTestCase& test_case = param;

    // Skip FLOAT32 ROW_MAJOR tests (both DEVICE and HOST) - they fail with typecast errors
    if (test_case.dtype == tt::tt_metal::DataType::FLOAT32 && test_case.layout == tt::tt_metal::Layout::ROW_MAJOR) {
        GTEST_SKIP() << "Skipping FLOAT32 ROW_MAJOR tensor serialization (API limitation: dump_tensor_flatbuffer "
                        "requires TILE layout for FLOAT32)";
    }

    const ttnn::Shape test_shape({1, 1, 32, 64});

    ttml::serialization::FlatBufferFile serializer;

    serializer.put("int_key", 42);
    serializer.put("float_key", 3.14159F);
    serializer.put("double_key", 2.71828);
    serializer.put("string_key", "Hello, World!");
    serializer.put("bool_key", true);
    serializer.put("char_key", 'Z');
    serializer.put("uint_key", static_cast<uint32_t>(987654321));
    serializer.put("size_key", static_cast<size_t>(123456789));
    bfloat16 bf16_scalar(42.5F);
    serializer.put("bf16_scalar", bf16_scalar);

    std::vector<int> int_vec = {10, 20, 30, 40, 50};
    std::vector<float> float_vec = {1.1F, 2.2F, 3.3F, 4.4F};
    std::vector<bfloat16> bf16_vec = {bfloat16(1.0F), bfloat16(2.0F), bfloat16(3.0F)};
    serializer.put("int_vector", std::span<const int>(int_vec));
    serializer.put("float_vector", std::span<const float>(float_vec));
    serializer.put("bf16_vector", std::span<const bfloat16>(bf16_vec));

    // Create and serialize the single tensor test case
    auto tensor =
        create_random_tensor(test_shape, test_case.dtype, test_case.layout, test_case.storage_type, 42, device);

    // Use tt-metal's flatbuffer methods directly
    std::string tensor_filename = (temp_dir / (test_case.name + ".tensorbin")).string();

    // Use tt-metal's dump_tensor_flatbuffer to write tensor to file
    tt::tt_metal::dump_tensor_flatbuffer(tensor_filename, tensor);

    // Store metadata in FlatBufferFile for reference
    serializer.put(test_case.name + "/tensor_file", std::string_view(tensor_filename));
    serializer.put(test_case.name + "/shape", to_bytes(tensor.logical_shape()));
    serializer.put(test_case.name + "/data_type", static_cast<int>(tensor.dtype()));
    serializer.put(test_case.name + "/layout", static_cast<int>(tensor.layout()));
    serializer.put(test_case.name + "/storage_type", static_cast<int>(tensor.storage_type()));

    std::filesystem::path output_dir = temp_dir / "test_data";
    ASSERT_NO_THROW(serializer.serialize(output_dir.string()));

    // Check that metadata file exists
    std::filesystem::path metadata_file = output_dir / "metadata.flatbuffer";
    ASSERT_TRUE(std::filesystem::exists(metadata_file)) << "Metadata file should exist: " << metadata_file;
    EXPECT_GT(std::filesystem::file_size(metadata_file), 0);

    // Check that tensor files exist (if any tensors were written)
    // Note: This test might not have tensors, so we just check metadata

    ttml::serialization::FlatBufferFile deserializer;
    // Deserialize from directory
    ASSERT_NO_THROW(deserializer.deserialize(output_dir.string()));

    int int_value = 0;
    EXPECT_NO_THROW(int_value = deserializer.get_int("int_key"));
    EXPECT_EQ(int_value, 42);

    float float_value = 0.F;
    EXPECT_NO_THROW(float_value = deserializer.get_float("float_key"));
    EXPECT_FLOAT_EQ(float_value, 3.14159F);

    double double_value = 0.0;
    EXPECT_NO_THROW(double_value = deserializer.get_double("double_key"));
    EXPECT_DOUBLE_EQ(double_value, 2.71828);

    std::string string_value;
    EXPECT_NO_THROW(string_value = deserializer.get_string("string_key"));
    EXPECT_EQ(string_value, "Hello, World!");

    bool bool_value = false;
    EXPECT_NO_THROW(bool_value = deserializer.get_bool("bool_key"));
    EXPECT_EQ(bool_value, true);

    char char_value = '\0';
    EXPECT_NO_THROW(char_value = deserializer.get_char("char_key"));
    EXPECT_EQ(char_value, 'Z');

    uint32_t uint_value = 0;
    EXPECT_NO_THROW(uint_value = deserializer.get_uint32("uint_key"));
    EXPECT_EQ(uint_value, 987654321U);

    size_t size_value = 0;
    EXPECT_NO_THROW(size_value = deserializer.get_size_t("size_key"));
    EXPECT_EQ(size_value, 123456789ULL);

    bfloat16 bf16_result = deserializer.get_bfloat16("bf16_scalar");
    EXPECT_EQ(std::bit_cast<uint16_t>(bf16_result), std::bit_cast<uint16_t>(bf16_scalar));

    // Deserialize and verify the single tensor test case
    std::string tensor_filename_read = tensor_filename;
    ASSERT_TRUE(std::filesystem::exists(tensor_filename_read)) << "Tensor file should exist: " << tensor_filename_read;

    tt::tt_metal::Tensor read_tensor;
    // For HOST tensors, pass nullptr to load as CPU tensor
    // For DEVICE tensors, pass device to load and then move to device
    auto* load_device = (test_case.storage_type == tt::tt_metal::StorageType::DEVICE) ? device : nullptr;
    ASSERT_NO_THROW(read_tensor = tt::tt_metal::load_tensor_flatbuffer(tensor_filename_read, load_device));

    // Restore original layout if needed
    tt::tt_metal::Layout original_layout = test_case.layout;
    if (read_tensor.layout() != original_layout) {
        read_tensor = read_tensor.to_layout(original_layout);
    }

    // Restore storage type if needed
    // load_tensor_flatbuffer loads tensors as HOST by default, so we need to restore device storage type
    if (test_case.storage_type == tt::tt_metal::StorageType::DEVICE &&
        read_tensor.storage_type() != tt::tt_metal::StorageType::DEVICE) {
        read_tensor = read_tensor.to_device(device);
    } else if (
        test_case.storage_type == tt::tt_metal::StorageType::HOST &&
        read_tensor.storage_type() != tt::tt_metal::StorageType::HOST) {
        // Ensure HOST tensors are on CPU
        read_tensor = read_tensor.cpu();
    }

    EXPECT_EQ(read_tensor.dtype(), test_case.dtype) << "Dtype mismatch for " << test_case.name;
    EXPECT_EQ(read_tensor.layout(), test_case.layout) << "Layout mismatch for " << test_case.name;
    EXPECT_EQ(read_tensor.storage_type(), test_case.storage_type) << "StorageType mismatch for " << test_case.name;

    // Call to_vector with the correct template parameter based on dtype
    std::vector<float> original_vec, read_vec;
    if (test_case.dtype == tt::tt_metal::DataType::UINT32) {
        auto orig_u32 = ttml::core::to_vector<uint32_t>(tensor);
        auto read_u32 = ttml::core::to_vector<uint32_t>(read_tensor);
        original_vec.assign(orig_u32.begin(), orig_u32.end());
        read_vec.assign(read_u32.begin(), read_u32.end());
    } else if (test_case.dtype == tt::tt_metal::DataType::INT32) {
        auto orig_i32 = ttml::core::to_vector<int32_t>(tensor);
        auto read_i32 = ttml::core::to_vector<int32_t>(read_tensor);
        original_vec.assign(orig_i32.begin(), orig_i32.end());
        read_vec.assign(read_i32.begin(), read_i32.end());
    } else {
        original_vec = ttml::core::to_vector<float>(tensor);
        read_vec = ttml::core::to_vector<float>(read_tensor);
    }

    EXPECT_EQ(original_vec.size(), read_vec.size()) << "Size mismatch for " << test_case.name;

    for (size_t i = 0; i < original_vec.size(); ++i) {
        if (test_case.dtype == tt::tt_metal::DataType::BFLOAT16) {
            EXPECT_NEAR(static_cast<float>(original_vec[i]), static_cast<float>(read_vec[i]), 1e-2)
                << "Value mismatch at index " << i << " for " << test_case.name;
        } else if (test_case.dtype == tt::tt_metal::DataType::FLOAT32) {
            EXPECT_FLOAT_EQ(original_vec[i], read_vec[i])
                << "Value mismatch at index " << i << " for " << test_case.name;
        } else if (
            test_case.dtype == tt::tt_metal::DataType::BFLOAT8_B ||
            test_case.dtype == tt::tt_metal::DataType::BFLOAT4_B) {
            EXPECT_NEAR(original_vec[i], read_vec[i], 1e-1)
                << "Value mismatch at index " << i << " for " << test_case.name;
        } else {
            EXPECT_EQ(original_vec[i], read_vec[i]) << "Value mismatch at index " << i << " for " << test_case.name;
        }
    }

    ttml::serialization::FlatBufferFile deserializer2;
    std::string deserialize_filename2 = (temp_dir / "test_data").string();
    ASSERT_NO_THROW(deserializer2.deserialize(deserialize_filename2));
    int int_value2 = 0;
    EXPECT_NO_THROW(int_value2 = deserializer2.get_int("int_key"));
    EXPECT_EQ(int_value2, 42);

    ttml::autograd::ctx().close_device();
}

INSTANTIATE_TEST_SUITE_P(
    SerializationConfigs,
    FlatBufferFileSerializationTest,
    ::testing::Values(
        TensorTestCase{
            tt::tt_metal::DataType::BFLOAT16,
            ttnn::Layout::ROW_MAJOR,
            tt::tt_metal::StorageType::DEVICE,
            "bf16_row_device"},
        TensorTestCase{
            tt::tt_metal::DataType::BFLOAT16,
            ttnn::Layout::TILE,
            tt::tt_metal::StorageType::DEVICE,
            "bf16_tile_device"},
        TensorTestCase{
            tt::tt_metal::DataType::BFLOAT16,
            ttnn::Layout::ROW_MAJOR,
            tt::tt_metal::StorageType::HOST,
            "bf16_row_host"},
        TensorTestCase{
            tt::tt_metal::DataType::BFLOAT16, ttnn::Layout::TILE, tt::tt_metal::StorageType::HOST, "bf16_tile_host"},
        TensorTestCase{
            tt::tt_metal::DataType::FLOAT32,
            ttnn::Layout::ROW_MAJOR,
            tt::tt_metal::StorageType::DEVICE,
            "f32_row_device"},
        TensorTestCase{
            tt::tt_metal::DataType::FLOAT32, ttnn::Layout::TILE, tt::tt_metal::StorageType::DEVICE, "f32_tile_device"},
        TensorTestCase{
            tt::tt_metal::DataType::FLOAT32, ttnn::Layout::ROW_MAJOR, tt::tt_metal::StorageType::HOST, "f32_row_host"},
        TensorTestCase{
            tt::tt_metal::DataType::FLOAT32, ttnn::Layout::TILE, tt::tt_metal::StorageType::HOST, "f32_tile_host"},
        TensorTestCase{
            tt::tt_metal::DataType::UINT32,
            ttnn::Layout::ROW_MAJOR,
            tt::tt_metal::StorageType::DEVICE,
            "u32_row_device"},
        TensorTestCase{
            tt::tt_metal::DataType::UINT32, ttnn::Layout::TILE, tt::tt_metal::StorageType::DEVICE, "u32_tile_device"},
        TensorTestCase{
            tt::tt_metal::DataType::UINT32, ttnn::Layout::ROW_MAJOR, tt::tt_metal::StorageType::HOST, "u32_row_host"},
        TensorTestCase{
            tt::tt_metal::DataType::UINT32, ttnn::Layout::TILE, tt::tt_metal::StorageType::HOST, "u32_tile_host"},
        TensorTestCase{
            tt::tt_metal::DataType::INT32,
            ttnn::Layout::ROW_MAJOR,
            tt::tt_metal::StorageType::DEVICE,
            "i32_row_device"},
        TensorTestCase{
            tt::tt_metal::DataType::INT32, ttnn::Layout::TILE, tt::tt_metal::StorageType::DEVICE, "i32_tile_device"},
        TensorTestCase{
            tt::tt_metal::DataType::INT32, ttnn::Layout::ROW_MAJOR, tt::tt_metal::StorageType::HOST, "i32_row_host"},
        TensorTestCase{
            tt::tt_metal::DataType::INT32, ttnn::Layout::TILE, tt::tt_metal::StorageType::HOST, "i32_tile_host"},
        TensorTestCase{
            tt::tt_metal::DataType::BFLOAT8_B,
            ttnn::Layout::TILE,
            tt::tt_metal::StorageType::DEVICE,
            "bf8_tile_device"},
        TensorTestCase{
            tt::tt_metal::DataType::BFLOAT8_B, ttnn::Layout::TILE, tt::tt_metal::StorageType::HOST, "bf8_tile_host"},
        TensorTestCase{
            tt::tt_metal::DataType::BFLOAT4_B,
            ttnn::Layout::TILE,
            tt::tt_metal::StorageType::DEVICE,
            "bf4_tile_device"},
        TensorTestCase{
            tt::tt_metal::DataType::BFLOAT4_B, ttnn::Layout::TILE, tt::tt_metal::StorageType::HOST, "bf4_tile_host"}),
    [](const ::testing::TestParamInfo<TestParam>& info) {
        // Use pretty printer to generate test name from all parameters
        std::string name = to_string(info.param);

        // Sanitize name: replace any invalid characters for test names
        // Google Test test names can only contain alphanumeric characters and underscores
        std::string sanitized_name;
        bool last_was_underscore = false;
        for (char c : name) {
            if (std::isalnum(static_cast<unsigned char>(c))) {
                sanitized_name += c;
                last_was_underscore = false;
            } else if (c == '_') {
                // Keep single underscores, but don't add multiple consecutive ones
                if (!last_was_underscore) {
                    sanitized_name += '_';
                    last_was_underscore = true;
                }
            } else {
                // Replace other characters with underscore, but collapse consecutive ones
                if (!last_was_underscore) {
                    sanitized_name += '_';
                    last_was_underscore = true;
                }
            }
        }

        // Remove trailing underscore if present
        if (!sanitized_name.empty() && sanitized_name.back() == '_') {
            sanitized_name.pop_back();
        }

        return sanitized_name;
    });
