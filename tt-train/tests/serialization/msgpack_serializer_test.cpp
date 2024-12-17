// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "serialization/msgpack_file.hpp"

class MsgPackFileTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Remove test file if it exists
        if (std::filesystem::exists(test_filename)) {
            std::filesystem::remove(test_filename);
        }
    }

    void TearDown() override {
        // Clean up test file after each test
        if (std::filesystem::exists(test_filename)) {
            std::filesystem::remove(test_filename);
        }
    }

    const std::string test_filename = "/tmp/test_data.msgpack";
};

TEST_F(MsgPackFileTest, SerializeDeserializePrimitives) {
    ttml::serialization::MsgPackFile serializer;

    // Put primitive data
    serializer.put("int_key", 42);
    serializer.put("float_key", 3.14F);
    serializer.put("double_key", 2.71828);
    serializer.put("uint_key", static_cast<uint32_t>(123456789));
    serializer.put("string_key", "Hello, World!");

    // Serialize to file
    ASSERT_NO_THROW(serializer.serialize(test_filename));

    // Deserialize from file
    ttml::serialization::MsgPackFile deserializer;
    ASSERT_NO_THROW(deserializer.deserialize(test_filename));

    // Get and check data
    int int_value = 0;
    EXPECT_NO_THROW(deserializer.get("int_key", int_value));
    EXPECT_EQ(int_value, 42);

    float float_value = 0;
    EXPECT_NO_THROW(deserializer.get("float_key", float_value));
    EXPECT_FLOAT_EQ(float_value, 3.14f);

    double double_value = 0;
    EXPECT_NO_THROW(deserializer.get("double_key", double_value));
    EXPECT_DOUBLE_EQ(double_value, 2.71828);

    uint32_t uint_value = 0;
    EXPECT_NO_THROW(deserializer.get("uint_key", uint_value));
    EXPECT_EQ(uint_value, 123456789U);

    std::string string_value;
    EXPECT_NO_THROW(deserializer.get("string_key", string_value));
    EXPECT_EQ(string_value, "Hello, World!");
}

TEST_F(MsgPackFileTest, SerializeDeserializeVectors) {
    ttml::serialization::MsgPackFile serializer;

    // Prepare data
    std::vector<int> int_vec = {1, 2, 3, 4, 5};
    std::vector<float> float_vec = {1.1F, 2.2F, 3.3F};
    std::vector<double> double_vec = {0.1, 0.01, 0.001};
    std::vector<uint32_t> uint_vec = {100, 200, 300};
    std::vector<std::string> string_vec = {"apple", "banana", "cherry"};

    // Put vector data
    serializer.put("int_vector_key", std::span<const int>(int_vec));
    serializer.put("float_vector_key", std::span<const float>(float_vec));
    serializer.put("double_vector_key", std::span<const double>(double_vec));
    serializer.put("uint_vector_key", std::span<const uint32_t>(uint_vec));
    serializer.put("string_vector_key", std::span<const std::string>(string_vec));

    // Serialize to file
    ASSERT_NO_THROW(serializer.serialize(test_filename));

    // Deserialize from file
    ttml::serialization::MsgPackFile deserializer;
    ASSERT_NO_THROW(deserializer.deserialize(test_filename));

    // Get and check data
    std::vector<int> int_vec_result;
    EXPECT_NO_THROW(deserializer.get("int_vector_key", int_vec_result));
    EXPECT_EQ(int_vec_result, int_vec);

    std::vector<float> float_vec_result;
    EXPECT_NO_THROW(deserializer.get("float_vector_key", float_vec_result));
    EXPECT_EQ(float_vec_result, float_vec);

    std::vector<double> double_vec_result;
    EXPECT_NO_THROW(deserializer.get("double_vector_key", double_vec_result));
    EXPECT_EQ(double_vec_result, double_vec);

    std::vector<uint32_t> uint_vec_result;
    EXPECT_NO_THROW(deserializer.get("uint_vector_key", uint_vec_result));
    EXPECT_EQ(uint_vec_result, uint_vec);

    std::vector<std::string> string_vec_result;
    EXPECT_NO_THROW(deserializer.get("string_vector_key", string_vec_result));
    EXPECT_EQ(string_vec_result, string_vec);
}

TEST_F(MsgPackFileTest, MissingKeyThrows) {
    ttml::serialization::MsgPackFile serializer;
    serializer.put("int_key", 42);
    ASSERT_NO_THROW(serializer.serialize(test_filename));
    ttml::serialization::MsgPackFile deserializer;
    ASSERT_NO_THROW(deserializer.deserialize(test_filename));

    int int_value = 0;
    EXPECT_ANY_THROW(deserializer.get("nonexistent_key", int_value));
}

TEST_F(MsgPackFileTest, TypeMismatchThrows) {
    ttml::serialization::MsgPackFile serializer;
    serializer.put("int_key", 42);
    serializer.serialize(test_filename);

    ttml::serialization::MsgPackFile deserializer;
    deserializer.deserialize(test_filename);

    float float_value = 0.F;
    EXPECT_ANY_THROW(deserializer.get("int_key", float_value));
}

TEST_F(MsgPackFileTest, OverwriteExistingKey) {
    ttml::serialization::MsgPackFile serializer;
    serializer.put("key", 42);
    serializer.put("key", "Overwritten");

    serializer.serialize(test_filename);

    ttml::serialization::MsgPackFile deserializer;
    deserializer.deserialize(test_filename);

    std::string string_value;
    EXPECT_NO_THROW(deserializer.get("key", string_value));
    EXPECT_EQ(string_value, "Overwritten");

    int int_value = 0;
    EXPECT_ANY_THROW(deserializer.get("key", int_value));
}

TEST_F(MsgPackFileTest, EmptySerializerSerialization) {
    ttml::serialization::MsgPackFile serializer;
    ASSERT_NO_THROW(serializer.serialize(test_filename));

    ttml::serialization::MsgPackFile deserializer;
    ASSERT_NO_THROW(deserializer.deserialize(test_filename));

    int int_value = 0;
    EXPECT_ANY_THROW(deserializer.get("any_key", int_value));
}

TEST_F(MsgPackFileTest, LargeDataSerialization) {
    ttml::serialization::MsgPackFile serializer;

    // Generate large data
    std::vector<int> large_int_vec(10000, 42);
    serializer.put("large_int_vector", std::span<const int>(large_int_vec));

    // Serialize to file
    ASSERT_NO_THROW(serializer.serialize(test_filename));

    // Deserialize from file
    ttml::serialization::MsgPackFile deserializer;
    ASSERT_NO_THROW(deserializer.deserialize(test_filename));

    // Get and check data
    std::vector<int> int_vec_result;
    EXPECT_NO_THROW(deserializer.get("large_int_vector", int_vec_result));
    EXPECT_EQ(int_vec_result.size(), large_int_vec.size());
    EXPECT_EQ(int_vec_result, large_int_vec);
}

TEST_F(MsgPackFileTest, NonExistentFileDeserialization) {
    ttml::serialization::MsgPackFile deserializer;
    EXPECT_THROW(deserializer.deserialize("nonexistent_file.msgpack"), std::runtime_error);
}

TEST_F(MsgPackFileTest, InvalidDataDeserialization) {
    // Write invalid data to file
    std::ofstream ofs(test_filename, std::ios::binary);
    ofs << "Invalid Data";
    ofs.close();

    ttml::serialization::MsgPackFile deserializer;
    EXPECT_ANY_THROW(deserializer.deserialize(test_filename));
}

TEST_F(MsgPackFileTest, MultipleDataTypesSerialization) {
    ttml::serialization::MsgPackFile serializer;

    serializer.put("int_key", 100);
    serializer.put("float_key", 1.23F);
    serializer.put("double_key", 4.56);
    serializer.put("string_key", "test string");

    std::vector<int> int_vec = {10, 20, 30};
    serializer.put("int_vector_key", std::span<const int>(int_vec));

    serializer.serialize(test_filename);

    ttml::serialization::MsgPackFile deserializer;
    deserializer.deserialize(test_filename);

    int int_value = 0;
    EXPECT_NO_THROW(deserializer.get("int_key", int_value));
    EXPECT_EQ(int_value, 100);

    float float_value = 0.F;
    EXPECT_NO_THROW(deserializer.get("float_key", float_value));
    EXPECT_FLOAT_EQ(float_value, 1.23F);

    double double_value = 0.0;
    EXPECT_NO_THROW(deserializer.get("double_key", double_value));
    EXPECT_DOUBLE_EQ(double_value, 4.56);

    std::string string_value;
    EXPECT_NO_THROW(deserializer.get("string_key", string_value));
    EXPECT_EQ(string_value, "test string");

    std::vector<int> int_vec_result;
    EXPECT_NO_THROW(deserializer.get("int_vector_key", int_vec_result));
    EXPECT_EQ(int_vec_result, int_vec);
}
