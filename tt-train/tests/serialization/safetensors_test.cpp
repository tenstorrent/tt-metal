// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "serialization/safetensors.hpp"

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/parameter_dtype.hpp"
#include "core/tt_tensor_utils.hpp"
#include "models/gpt2.hpp"
#include "models/llama.hpp"
#include "serialization/serialization.hpp"

std::string get_test_data_dir() {
    const char* env_var = std::getenv("TEST_LOCAL_DATA_DIR");
    return (env_var) ? std::string(env_var) : std::string(TEST_DATA_DIR);
}

inline std::vector<float> bytes_to_floats_copy(std::span<const std::byte> bytes) {
    if (bytes.size_bytes() % sizeof(float) != 0) {
        throw std::runtime_error("bytes_to_floats_copy: size not multiple of sizeof(float)");
    }
    const std::size_t n = bytes.size_bytes() / sizeof(float);
    std::vector<float> out(n);
    if (n) {
        std::memcpy(out.data(), bytes.data(), n * sizeof(float));
    }
    return out;
}
// Disabled test because it requires a specific test file to be present.
// It doesn't work in CI for now :(
TEST(SafeTensorsTest, DISABLED_LoadSimpleMlp) {
    const std::unordered_map<std::string, std::vector<float>> test_params = {
        // net.0.weight: shape (4, 2), row-major
        {"net.0.weight", {0.3930f, 0.8285f, 0.8702f, 0.8824f, 0.1990f, -0.8696f, 0.0920f, -0.6256f}},
        // net.0.bias: shape (4,)
        {"net.0.bias", {0.0f, 0.0f, 0.0f, 0.0f}},
        // net.2.weight: shape (4, 4), row-major
        {"net.2.weight",
         {-0.8071f,
          0.7695f,
          0.6585f,
          -0.8639f,
          0.1621f,
          -0.1459f,
          -0.1425f,
          -0.3964f,
          0.3330f,
          -0.5129f,
          0.3175f,
          0.4380f,
          0.6200f,
          0.3238f,
          -0.8571f,
          -0.5618f}},
        // net.2.bias: shape (4,)
        {"net.2.bias", {0.0f, 0.0f, 0.0f, 0.0f}},
        // net.4.weight: shape (8, 4), row-major
        {"net.4.weight",
         {0.3531f,  0.1480f,  -0.5516f, -0.4072f, 0.6652f,  0.4765f,  -0.3083f, -0.1780f, -0.6736f, -0.0127f, -0.5325f,
          -0.5454f, -0.0390f, 0.1062f,  -0.2896f, 0.4196f,  -0.4303f, 0.6416f,  0.4846f,  -0.5963f, -0.1760f, 0.0319f,
          0.1032f,  0.1677f,  0.2775f,  0.0424f,  -0.3450f, 0.3346f,  -0.6783f, -0.4191f, -0.1770f, -0.3444f}},
        // net.4.bias: shape (8,)
        {"net.4.bias", {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}},
    };

    std::filesystem::path path = get_test_data_dir() + "/tiny_mlp.safetensors";

    std::vector<ttml::serialization::SafetensorSerialization::TensorInfo> tensors;
    std::vector<std::vector<float>> tensors_data;

    ttml::serialization::SafetensorSerialization::visit_safetensors_file(
        path,
        [&tensors, &tensors_data](
            const ttml::serialization::SafetensorSerialization::TensorInfo& info, std::span<const std::byte> bytes) {
            tensors.emplace_back(info);
            auto floats = bytes_to_floats_copy(bytes);
            tensors_data.emplace_back(std::move(floats));

            return true;  // Continue visiting
        }

    );
    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto& info = tensors[i];
        const auto& tensor_data = tensors_data[i];
        fmt::print("Tensor: {}, dtype: {}, shape: {}\n", info.name, info.dtype, info.shape);
        fmt::print("  {} floats\n", tensor_data);
        EXPECT_EQ(tensor_data.size(), info.shape.volume());
        EXPECT_NEAR(tensor_data.size(), test_params.at(info.name).size(), 1e-6);
    }

    ASSERT_EQ(tensors.size(), 6);  // Assuming the MLP has 3 tensors
    EXPECT_EQ(tensors[0].name, "net.0.bias");
    EXPECT_EQ(tensors[1].name, "net.0.weight");
    EXPECT_EQ(tensors[2].name, "net.2.bias");
    EXPECT_EQ(tensors[3].name, "net.2.weight");
    EXPECT_EQ(tensors[4].name, "net.4.bias");
    EXPECT_EQ(tensors[5].name, "net.4.weight");
}

TEST(SafeTensorsTest, F16DecodeSubnormals) {
    // uint16 F16 bit patterns, little-endian, decoded via bytes_to_float_vec.
    const std::vector<uint16_t> halves = {
        0x0000,  // +0
        0x0001,  // smallest subnormal, 2^-24
        0x0200,  // mid subnormal, 2^-15
        0x03FF,  // largest subnormal
        0x0400,  // smallest normal, 2^-14
        0x3C00,  // 1.0
    };
    const std::vector<float> expected = {
        0.0f,
        5.9604645e-08f,
        3.0517578e-05f,
        6.0975552e-05f,
        6.1035156e-05f,
        1.0f,
    };

    std::vector<std::byte> bytes(halves.size() * sizeof(uint16_t));
    std::memcpy(bytes.data(), halves.data(), bytes.size());

    auto out = ttml::serialization::SafetensorSerialization::bytes_to_float_vec(
        std::span<const std::byte>(bytes.data(), bytes.size()), "F16");

    ASSERT_EQ(out.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(out[i], expected[i]);
    }
}

// The C++ safetensors loaders install each value in the dtype its parameter is stored in: fp32 parameters, converted
// before the load as precision.master_dtype: float32 does, get the file's fp32 values exactly; bf16 parameters get the
// same bits the loaders installed before they kept the stored dtype (the host rounding of core::from_vector()).
class SafetensorsLoaderTest : public ::testing::Test {
public:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
    }
    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }
};

namespace {

struct FileTensor {
    std::string name;
    std::vector<int64_t> shape;
    std::vector<float> values;
};

// Writes `tensors` as an F32 safetensors file named model.safetensors in a fresh directory and returns the directory.
std::filesystem::path write_f32_safetensors(const std::string& dir_name, const std::vector<FileTensor>& tensors) {
    const auto dir = std::filesystem::temp_directory_path() / dir_name;
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    std::string header = "{";
    uint64_t offset = 0;
    for (const auto& tensor : tensors) {
        const uint64_t size = tensor.values.size() * sizeof(float);
        std::string shape;
        for (const auto dim : tensor.shape) {
            shape += fmt::format("{}{}", shape.empty() ? "" : ",", dim);
        }
        header += fmt::format(
            R"({}"{}":{{"dtype":"F32","shape":[{}],"data_offsets":[{},{}]}})",
            header.size() > 1 ? "," : "",
            tensor.name,
            shape,
            offset,
            offset + size);
        offset += size;
    }
    header += "}";
    header.resize((header.size() + 7U) / 8U * 8U, ' ');
    std::ofstream out(dir / "model.safetensors", std::ios::binary);
    const uint64_t header_size = header.size();
    for (int byte = 0; byte < 8; ++byte) {
        out.put(static_cast<char>((header_size >> (8 * byte)) & 0xFFU));
    }
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    for (const auto& tensor : tensors) {
        out.write(
            reinterpret_cast<const char*>(tensor.values.data()),
            static_cast<std::streamsize>(tensor.values.size() * sizeof(float)));
    }
    return dir;
}

// Values in [-1, 1) with more fraction bits than bf16 keeps.
std::vector<float> random_values(size_t size, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    std::vector<float> values(size);
    for (auto& value : values) {
        value = dist(gen);
    }
    return values;
}

std::vector<float> transposed(const std::vector<float>& values, size_t rows, size_t cols) {
    std::vector<float> out(values.size());
    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < cols; ++col) {
            out[col * rows + row] = values[row * cols + col];
        }
    }
    return out;
}

// Checks that `name` holds `expected` in `dtype`: exactly for fp32, as core::from_vector() rounds it for bf16.
void expect_loaded(
    const ttml::serialization::NamedParameters& parameters,
    const std::string& name,
    const std::vector<float>& expected,
    ttnn::DataType dtype) {
    SCOPED_TRACE(name);
    const auto& stored = parameters.at(name)->get_value(ttml::autograd::PreferredPrecision::NATIVE);
    ASSERT_EQ(stored.dtype(), dtype);
    const auto loaded = ttml::core::to_vector<float>(stored);
    if (dtype == ttnn::DataType::FLOAT32) {
        EXPECT_EQ(loaded, expected);
        // The file's values aren't bf16 values, so a load through bf16 would have changed them.
        const auto through_bf16 =
            ttml::core::to_vector<float>(ttml::core::from_vector(expected, stored.logical_shape(), stored.device()));
        EXPECT_NE(loaded, through_bf16);
    } else {
        EXPECT_EQ(
            loaded,
            ttml::core::to_vector<float>(ttml::core::from_vector(expected, stored.logical_shape(), stored.device())));
    }
}

ttml::models::gpt2::TransformerConfig tiny_gpt2_config() {
    ttml::models::gpt2::TransformerConfig config;
    config.num_heads = 2;
    config.embedding_dim = 64;
    config.dropout_prob = 0.0F;
    config.num_blocks = 1;
    config.vocab_size = 64;
    config.max_sequence_length = 32;
    return config;
}

// GPT-2's three kinds of entries: one copied as is (ln_f), one transposed (c_attn, stored in x out in the file), and
// the embedding, whose rows are padded with zeros to the model's vocabulary.
void check_gpt2_load(ttnn::DataType dtype) {
    auto model = ttml::models::gpt2::create(tiny_gpt2_config());
    auto parameters = model->parameters();
    if (dtype == ttnn::DataType::FLOAT32) {
        ttml::core::store_trainable_parameters_as(parameters, ttnn::DataType::FLOAT32);
    }
    const auto ln_f = random_values(64, 1U);
    const auto c_attn = random_values(64 * 192, 2U);
    const auto wte = random_values(50 * 64, 3U);
    const auto dir = write_f32_safetensors(
        fmt::format("ttml_gpt2_safetensors_{}", static_cast<int>(dtype)),
        {{"ln_f.weight", {64}, ln_f}, {"h.0.attn.c_attn.weight", {64, 192}, c_attn}, {"wte.weight", {50, 64}, wte}});
    model->load_from_safetensors(dir);
    std::filesystem::remove_all(dir);

    auto padded_wte = wte;
    padded_wte.resize(64 * 64, 0.0F);
    expect_loaded(parameters, "transformer/ln_fc/gamma", ln_f, dtype);
    expect_loaded(
        parameters, "transformer/gpt_block_0/attention/qkv_linear/weight", transposed(c_attn, 64, 192), dtype);
    expect_loaded(parameters, "transformer/fc/weight", padded_wte, dtype);
}

ttml::models::llama::LlamaConfig tiny_llama_config() {
    ttml::models::llama::LlamaConfig config;
    config.num_heads = 2;
    config.num_groups = 1;
    config.embedding_dim = 64;
    config.intermediate_dim = 128;
    config.num_blocks = 1;
    config.vocab_size = 64;
    config.max_sequence_length = 32;
    return config;
}

// Llama's paths: the final norm copied as is, the (tied) embedding, the down projection, and k_proj + v_proj
// concatenated into kv_linear. k_proj's rows within a head are reordered on load; every row of the one key head is
// the same here, so the expected kv_linear is plain k followed by v.
void check_llama_load(ttnn::DataType dtype) {
    auto model = ttml::models::llama::create(tiny_llama_config());
    auto parameters = model->parameters();
    if (dtype == ttnn::DataType::FLOAT32) {
        ttml::core::store_trainable_parameters_as(parameters, ttnn::DataType::FLOAT32);
    }
    const auto norm = random_values(64, 4U);
    const auto embed = random_values(64 * 64, 5U);
    const auto down = random_values(64 * 128, 6U);
    const auto key_row = random_values(64, 7U);
    std::vector<float> key;
    for (int row = 0; row < 32; ++row) {
        key.insert(key.end(), key_row.begin(), key_row.end());
    }
    const auto value = random_values(32 * 64, 8U);
    const auto dir = write_f32_safetensors(
        fmt::format("ttml_llama_safetensors_{}", static_cast<int>(dtype)),
        {{"model.norm.weight", {64}, norm},
         {"model.embed_tokens.weight", {64, 64}, embed},
         {"model.layers.0.mlp.down_proj.weight", {64, 128}, down},
         {"model.layers.0.self_attn.k_proj.weight", {32, 64}, key},
         {"model.layers.0.self_attn.v_proj.weight", {32, 64}, value}});
    model->load_from_safetensors(dir);
    std::filesystem::remove_all(dir);

    auto kv = key;
    kv.insert(kv.end(), value.begin(), value.end());
    expect_loaded(parameters, "llama/ln_fc/gamma", norm, dtype);
    expect_loaded(parameters, "llama/fc/weight", embed, dtype);
    expect_loaded(parameters, "llama/llama_block_0/mlp/w2/weight", down, dtype);
    expect_loaded(parameters, "llama/llama_block_0/attention/kv_linear/weight", kv, dtype);
}

}  // namespace

TEST_F(SafetensorsLoaderTest, Gpt2LoadsFp32ValuesExactlyIntoFp32Parameters) {
    check_gpt2_load(ttnn::DataType::FLOAT32);
}

TEST_F(SafetensorsLoaderTest, Gpt2LoadsBf16ParametersAsBefore) {
    check_gpt2_load(ttnn::DataType::BFLOAT16);
}

TEST_F(SafetensorsLoaderTest, LlamaLoadsFp32ValuesExactlyIntoFp32Parameters) {
    check_llama_load(ttnn::DataType::FLOAT32);
}

TEST_F(SafetensorsLoaderTest, LlamaLoadsBf16ParametersAsBefore) {
    check_llama_load(ttnn::DataType::BFLOAT16);
}

TEST_F(SafetensorsLoaderTest, AssignFromFloatsRejectsNonFloatTensors) {
    auto* device = &ttml::autograd::ctx().get_device();
    auto tensor = ttml::autograd::create_tensor(ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        std::vector<uint32_t>(32 * 32, 1U), ttnn::Shape({1, 1, 32, 32}), device));
    EXPECT_THROW(
        ttml::serialization::assign_from_floats(tensor, std::vector<float>(32 * 32, 0.5F)), std::invalid_argument);
}
