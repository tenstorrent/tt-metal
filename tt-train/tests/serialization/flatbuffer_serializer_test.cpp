// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "serialization/flatbuffer_file.hpp"
#include "serialization/tar_reader.hpp"
#include "serialization/tar_writer.hpp"

class FlatBufferFileTest : public ::testing::Test {
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

    const std::string test_filename = "/tmp/test_data.flatbuffer";
};

TEST_F(FlatBufferFileTest, SerializeDeserializePrimitives) {
    ttml::serialization::FlatBufferFile serializer;

    // Put primitive data
    serializer.put("int_key", 42);
    serializer.put("float_key", 3.14F);
    serializer.put("double_key", 2.71828);
    serializer.put("uint_key", static_cast<uint32_t>(123456789));
    serializer.put("string_key", "Hello, World!");

    // Serialize to file
    ASSERT_NO_THROW(serializer.serialize(test_filename));

    // Deserialize from file
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

TEST_F(FlatBufferFileTest, SerializeDeserializeVectors) {
    ttml::serialization::FlatBufferFile serializer;

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
    ttml::serialization::FlatBufferFile deserializer;
    ASSERT_NO_THROW(deserializer.deserialize(test_filename));

    // Get and check data
    std::vector<int> int_vec_result;
    EXPECT_NO_THROW(int_vec_result = deserializer.get_vector_int("int_vector_key"));
    EXPECT_EQ(int_vec_result, int_vec);

    std::vector<float> float_vec_result;
    EXPECT_NO_THROW(float_vec_result = deserializer.get_vector_float("float_vector_key"));
    EXPECT_EQ(float_vec_result, float_vec);

    std::vector<double> double_vec_result;
    EXPECT_NO_THROW(double_vec_result = deserializer.get_vector_double("double_vector_key"));
    EXPECT_EQ(double_vec_result, double_vec);

    std::vector<uint32_t> uint_vec_result;
    EXPECT_NO_THROW(uint_vec_result = deserializer.get_vector_uint32("uint_vector_key"));
    EXPECT_EQ(uint_vec_result, uint_vec);

    std::vector<std::string> string_vec_result;
    EXPECT_NO_THROW(string_vec_result = deserializer.get_vector_string("string_vector_key"));
    EXPECT_EQ(string_vec_result, string_vec);
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

TEST_F(FlatBufferFileTest, LargeDataSerialization) {
    ttml::serialization::FlatBufferFile serializer;

    // Generate large data
    std::vector<int> large_int_vec(10000, 42);
    serializer.put("large_int_vector", std::span<const int>(large_int_vec));

    // Serialize to file
    ASSERT_NO_THROW(serializer.serialize(test_filename));

    // Deserialize from file
    ttml::serialization::FlatBufferFile deserializer;
    ASSERT_NO_THROW(deserializer.deserialize(test_filename));

    // Get and check data
    std::vector<int> int_vec_result;
    EXPECT_NO_THROW(int_vec_result = deserializer.get_vector_int("large_int_vector"));
    EXPECT_EQ(int_vec_result.size(), large_int_vec.size());
    EXPECT_EQ(int_vec_result, large_int_vec);
}

TEST_F(FlatBufferFileTest, NonExistentFileDeserialization) {
    ttml::serialization::FlatBufferFile deserializer;
    EXPECT_THROW(deserializer.deserialize("nonexistent_file.flatbuffer"), std::runtime_error);
}

TEST_F(FlatBufferFileTest, InvalidDataDeserialization) {
    // Write invalid data to file (not a valid tarball)
    std::ofstream ofs(test_filename, std::ios::binary);
    ofs << "Invalid Data";
    ofs.close();

    ttml::serialization::FlatBufferFile deserializer;
    EXPECT_ANY_THROW(deserializer.deserialize(test_filename));
}

TEST_F(FlatBufferFileTest, InvalidTarballMissingFlatbufferFiles) {
    // Create a tarball without any .flatbuffer files using TarWriter
    ttml::serialization::TarWriter tar_writer;
    std::vector<uint8_t> test_data{'t', 'e', 's', 't'};
    tar_writer.add_file("other_file.txt", std::move(test_data));
    tar_writer.write_to_file(test_filename);

    // Try to deserialize - should throw because no .flatbuffer files found
    ttml::serialization::FlatBufferFile deserializer;
    EXPECT_THROW(deserializer.deserialize(test_filename), std::runtime_error);
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

    std::vector<int> int_vec_result;
    EXPECT_NO_THROW(int_vec_result = deserializer.get_vector_int("int_vector_key"));
    EXPECT_EQ(int_vec_result, int_vec);
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

    ttml::serialization::FlatBufferFile deserializer;
    deserializer.deserialize(test_filename);

    std::vector<char> result;
    EXPECT_NO_THROW(result = deserializer.get_vector_char("char_vector_key"));
    EXPECT_EQ(result, char_vec);
}

TEST_F(FlatBufferFileTest, TarballStructure) {
    ttml::serialization::FlatBufferFile serializer;

    serializer.put("test_key", 42);
    serializer.put("string_key", "test value");

    // Serialize to tarball
    serializer.serialize(test_filename);

    // Verify tarball structure using TarReader
    ttml::serialization::TarReader tar_reader;
    ASSERT_NO_THROW(tar_reader.read_from_file(test_filename));

    // Verify tarball contains data.flatbuffer (default prefix for keys without '/')
    EXPECT_TRUE(tar_reader.has_file("data.flatbuffer"));

    // Verify file list
    auto files = tar_reader.list_files();
    EXPECT_EQ(files.size(), 1);
    EXPECT_EQ(files[0], "data.flatbuffer");

    // Verify we can extract the flatbuffer data
    auto flatbuffer_data = tar_reader.get_file("data.flatbuffer");
    EXPECT_FALSE(flatbuffer_data.empty());
    EXPECT_GT(flatbuffer_data.size(), 0);
}

TEST_F(FlatBufferFileTest, MultipleFilesInTarball) {
    ttml::serialization::FlatBufferFile serializer;

    // Add data with different prefixes
    serializer.put("model/weight1", 1.0F);
    serializer.put("model/weight2", 2.0F);
    serializer.put("optimizer/lr", 0.001);
    serializer.put("optimizer/momentum", 0.9);
    serializer.put("scheduler/step", 100);

    // Serialize to tarball
    serializer.serialize(test_filename);

    // Verify tarball structure using TarReader
    ttml::serialization::TarReader tar_reader;
    tar_reader.read_from_file(test_filename);

    // Verify tarball contains multiple files
    EXPECT_TRUE(tar_reader.has_file("model.flatbuffer"));
    EXPECT_TRUE(tar_reader.has_file("optimizer.flatbuffer"));
    EXPECT_TRUE(tar_reader.has_file("scheduler.flatbuffer"));

    // Verify file list
    auto files = tar_reader.list_files();
    EXPECT_EQ(files.size(), 3);

    // Verify we can extract each flatbuffer data
    auto model_data = tar_reader.get_file("model.flatbuffer");
    EXPECT_FALSE(model_data.empty());
    EXPECT_GT(model_data.size(), 0);

    auto optimizer_data = tar_reader.get_file("optimizer.flatbuffer");
    EXPECT_FALSE(optimizer_data.empty());
    EXPECT_GT(optimizer_data.size(), 0);

    auto scheduler_data = tar_reader.get_file("scheduler.flatbuffer");
    EXPECT_FALSE(scheduler_data.empty());
    EXPECT_GT(scheduler_data.size(), 0);

    // Verify deserialization works correctly
    ttml::serialization::FlatBufferFile deserializer;
    deserializer.deserialize(test_filename);

    float weight1 = 0.F;
    EXPECT_NO_THROW(weight1 = deserializer.get_float("model/weight1"));
    EXPECT_FLOAT_EQ(weight1, 1.0F);

    float weight2 = 0.F;
    EXPECT_NO_THROW(weight2 = deserializer.get_float("model/weight2"));
    EXPECT_FLOAT_EQ(weight2, 2.0F);

    double lr = 0.0;
    EXPECT_NO_THROW(lr = deserializer.get_double("optimizer/lr"));
    EXPECT_DOUBLE_EQ(lr, 0.001);

    double momentum = 0.0;
    EXPECT_NO_THROW(momentum = deserializer.get_double("optimizer/momentum"));
    EXPECT_DOUBLE_EQ(momentum, 0.9);

    int step = 0;
    EXPECT_NO_THROW(step = deserializer.get_int("scheduler/step"));
    EXPECT_EQ(step, 100);
}

TEST_F(FlatBufferFileTest, TarballRoundTrip) {
    ttml::serialization::FlatBufferFile serializer;

    serializer.put("int_key", 123);
    serializer.put("float_key", 3.14F);
    std::vector<int> vec = {1, 2, 3};
    serializer.put("vec_key", std::span<const int>(vec));

    // Serialize to tarball
    serializer.serialize(test_filename);

    // Read tarball and verify structure
    ttml::serialization::TarReader tar_reader;
    tar_reader.read_from_file(test_filename);

    EXPECT_TRUE(tar_reader.has_file("data.flatbuffer"));
    auto flatbuffer_data = tar_reader.get_file("data.flatbuffer");

    // Deserialize using FlatBufferFile (which internally uses TarReader)
    ttml::serialization::FlatBufferFile deserializer;
    deserializer.deserialize(test_filename);

    // Verify data integrity
    int int_value = 0;
    EXPECT_NO_THROW(int_value = deserializer.get_int("int_key"));
    EXPECT_EQ(int_value, 123);

    float float_value = 0.F;
    EXPECT_NO_THROW(float_value = deserializer.get_float("float_key"));
    EXPECT_FLOAT_EQ(float_value, 3.14F);

    std::vector<int> vec_result;
    EXPECT_NO_THROW(vec_result = deserializer.get_vector_int("vec_key"));
    EXPECT_EQ(vec_result, vec);
}
