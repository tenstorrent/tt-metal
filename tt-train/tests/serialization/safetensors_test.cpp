// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "serialization/safetensors.hpp"

#include <fmt/format.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/parameter_dtype.hpp"
#include "core/tt_tensor_utils.hpp"
#include "models/gpt2.hpp"
#include "models/llama.hpp"
#include "serialization/serialization.hpp"
#include "test_utils/program_cache.hpp"
#include "test_utils/random_data.hpp"

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
// before the load as precision.master_dtype: float32 does, get the file's fp32 values exactly; bf16 parameters get
// each value rounded to bf16 on the host, as core::from_vector() rounds it.
class SafetensorsLoaderTest : public ttml::test_utils::DeviceSuiteTest {};

namespace {

struct FileTensor {
    std::string name;
    std::vector<int64_t> shape;
    std::vector<float> values;
};

// A directory under gtest's temp dir named for the running test and this process, so ttml_tests processes sharing a
// host don't write into each other's files.
std::filesystem::path test_temp_dir() {
    const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
    auto name = fmt::format("ttml_{}_{}_{}", info->test_suite_name(), info->name(), getpid());
    std::replace(name.begin(), name.end(), '/', '_');
    return std::filesystem::path(::testing::TempDir()) / name;
}

// Writes `tensors` as an F32 safetensors file named model.safetensors in a fresh test_temp_dir() and returns it.
std::filesystem::path write_f32_safetensors(const std::vector<FileTensor>& tensors) {
    const auto dir = test_temp_dir();
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
    return ttml::test_utils::make_uniform_vector<float>(size, -1.0F, 1.0F, seed);
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
    const auto through_bf16 =
        ttml::core::to_vector<float>(ttml::core::from_vector(expected, stored.logical_shape(), stored.device()));
    if (dtype == ttnn::DataType::FLOAT32) {
        EXPECT_EQ(loaded, expected);
        // The file's values aren't bf16 values, so a load through bf16 would have changed them.
        EXPECT_NE(loaded, through_bf16);
    } else {
        EXPECT_EQ(loaded, through_bf16);
    }
}

// Builds a model, converts it to fp32 for an fp32 check (as precision.master_dtype: float32 does before the load),
// loads `tensors` from a temporary F32 file and returns the model's parameters.
template <typename CreateModel>
ttml::serialization::NamedParameters load_into_new_model(
    CreateModel create_model, ttnn::DataType dtype, const std::vector<FileTensor>& tensors) {
    auto model = create_model();
    auto parameters = model->parameters();
    if (dtype == ttnn::DataType::FLOAT32) {
        ttml::core::store_trainable_parameters_as(parameters, ttnn::DataType::FLOAT32);
    }
    const auto dir = write_f32_safetensors(tensors);
    model->load_from_safetensors(dir);
    std::filesystem::remove_all(dir);
    return parameters;
}

// The file's tensors of one load, each with what it should leave in the model.
struct LoadCase {
    std::vector<FileTensor> file;
    std::vector<std::pair<std::string, std::vector<float>>> expected;
    uint32_t seed = 1U;

    // A tensor the loader installs in `parameter` as `installed`.
    void add(
        const std::string& file_name,
        const std::vector<int64_t>& shape,
        const std::vector<float>& values,
        const std::string& parameter,
        const std::vector<float>& installed) {
        file.push_back({file_name, shape, values});
        expected.emplace_back(parameter, installed);
    }

    // A tensor of random values that the loader copies as it is.
    void copied(const std::string& file_name, const std::vector<int64_t>& shape, const std::string& parameter) {
        size_t size = 1;
        for (const auto dim : shape) {
            size *= static_cast<size_t>(dim);
        }
        const auto values = random_values(size, seed++);
        add(file_name, shape, values, parameter, values);
    }
};

// Loads the case's file into a new model and checks every expected parameter. The model must have no parameter the
// file leaves out, so each place the loader installs a tensor is checked.
template <typename CreateModel>
void expect_every_parameter_loaded(CreateModel create_model, ttnn::DataType dtype, const LoadCase& load_case) {
    const auto parameters = load_into_new_model(create_model, dtype, load_case.file);
    std::set<std::string> checked;
    for (const auto& [name, values] : load_case.expected) {
        ASSERT_TRUE(parameters.contains(name)) << name;
        expect_loaded(parameters, name, values, dtype);
        checked.insert(name);
    }
    for (const auto& [name, tensor] : parameters) {
        EXPECT_TRUE(checked.contains(name)) << "the test's file has nothing for " << name;
    }
}

// Weight tying on, as in configs/model_configs/gpt2s.yaml.
ttml::models::gpt2::TransformerConfig tiny_gpt2_config() {
    ttml::models::gpt2::TransformerConfig config;
    config.num_heads = 2;
    config.embedding_dim = 64;
    config.dropout_prob = 0.0F;
    config.num_blocks = 1;
    config.vocab_size = 64;
    config.max_sequence_length = 32;
    config.weight_tying = ttml::models::gpt2::WeightTyingType::Enabled;
    return config;
}

// Every tensor the GPT-2 loader maps. The embedding's rows are padded with zeros to the model's vocabulary and land
// in fc/weight, the one weight the output layer and the token embedding share; the four Conv1D weights are stored
// in x out in the file and transposed on load; the rest is copied as it is.
void check_gpt2_load(ttnn::DataType dtype) {
    LoadCase load_case;
    const auto wte = random_values(50 * 64, load_case.seed++);
    auto padded_wte = wte;
    padded_wte.resize(64 * 64, 0.0F);
    load_case.add("wte.weight", {50, 64}, wte, "transformer/fc/weight", padded_wte);
    load_case.copied("wpe.weight", {32, 64}, "transformer/pos_emb/weight");
    load_case.copied("ln_f.weight", {64}, "transformer/ln_fc/gamma");
    load_case.copied("ln_f.bias", {64}, "transformer/ln_fc/beta");

    const std::string block = "transformer/gpt_block_0/";
    const auto conv1d = [&](const std::string& file_name, int64_t in, int64_t out, const std::string& parameter) {
        const auto values = random_values(static_cast<size_t>(in * out), load_case.seed++);
        load_case.add(
            file_name,
            {in, out},
            values,
            block + parameter,
            transposed(values, static_cast<size_t>(in), static_cast<size_t>(out)));
    };
    load_case.copied("h.0.ln_1.weight", {64}, block + "ln1/gamma");
    load_case.copied("h.0.ln_1.bias", {64}, block + "ln1/beta");
    conv1d("h.0.attn.c_attn.weight", 64, 192, "attention/qkv_linear/weight");
    load_case.copied("h.0.attn.c_attn.bias", {192}, block + "attention/qkv_linear/bias");
    conv1d("h.0.attn.c_proj.weight", 64, 64, "attention/out_linear/weight");
    load_case.copied("h.0.attn.c_proj.bias", {64}, block + "attention/out_linear/bias");
    load_case.copied("h.0.ln_2.weight", {64}, block + "ln2/gamma");
    load_case.copied("h.0.ln_2.bias", {64}, block + "ln2/beta");
    conv1d("h.0.mlp.c_fc.weight", 64, 256, "mlp/fc1/weight");
    load_case.copied("h.0.mlp.c_fc.bias", {256}, block + "mlp/fc1/bias");
    conv1d("h.0.mlp.c_proj.weight", 256, 64, "mlp/fc2/weight");
    load_case.copied("h.0.mlp.c_proj.bias", {64}, block + "mlp/fc2/bias");

    expect_every_parameter_loaded([] { return ttml::models::gpt2::create(tiny_gpt2_config()); }, dtype, load_case);
}

// Weight tying off, as in configs/model_configs/llama8b.yaml: the embedding and the output layer load separately.
ttml::models::llama::LlamaConfig tiny_llama_config() {
    ttml::models::llama::LlamaConfig config;
    config.num_heads = 2;
    config.num_groups = 1;
    config.embedding_dim = 64;
    config.intermediate_dim = 128;
    config.num_blocks = 1;
    config.vocab_size = 64;
    config.max_sequence_length = 32;
    config.weight_tying = ttml::models::llama::WeightTyingType::Disabled;
    return config;
}

// `heads` x `rows_per_head` rows, the same row throughout a head. The Llama loader reorders the rows of q_proj and
// k_proj within a head, which leaves such a matrix as it is.
std::vector<float> one_row_per_head(size_t heads, size_t rows_per_head, size_t cols, uint32_t seed) {
    std::vector<float> out;
    for (size_t head = 0; head < heads; ++head) {
        const auto row = random_values(cols, seed + static_cast<uint32_t>(head));
        for (size_t repeat = 0; repeat < rows_per_head; ++repeat) {
            out.insert(out.end(), row.begin(), row.end());
        }
    }
    return out;
}

// Every tensor the Llama loader maps: the embedding, the output layer, the three norms, q_proj, k_proj and v_proj
// concatenated into kv_linear, o_proj, and the three MLP projections.
void check_llama_load(ttnn::DataType dtype) {
    LoadCase load_case;
    load_case.copied("model.embed_tokens.weight", {64, 64}, "llama/tok_emb/weight");
    load_case.copied("lm_head.weight", {64, 64}, "llama/fc/weight");
    load_case.copied("model.norm.weight", {64}, "llama/ln_fc/gamma");

    const std::string layer = "model.layers.0.";
    const std::string block = "llama/llama_block_0/";
    load_case.copied(layer + "input_layernorm.weight", {64}, block + "attention_norm/gamma");
    load_case.copied(layer + "post_attention_layernorm.weight", {64}, block + "mlp_norm/gamma");

    const auto query = one_row_per_head(2, 32, 64, 100U);
    load_case.add(layer + "self_attn.q_proj.weight", {64, 64}, query, block + "attention/q_linear/weight", query);
    const auto key = one_row_per_head(1, 32, 64, 200U);
    const auto value = random_values(32 * 64, 300U);
    auto key_value = key;
    key_value.insert(key_value.end(), value.begin(), value.end());
    load_case.file.push_back({layer + "self_attn.k_proj.weight", {32, 64}, key});
    load_case.add(layer + "self_attn.v_proj.weight", {32, 64}, value, block + "attention/kv_linear/weight", key_value);
    load_case.copied(layer + "self_attn.o_proj.weight", {64, 64}, block + "attention/out_linear/weight");

    load_case.copied(layer + "mlp.gate_proj.weight", {128, 64}, block + "mlp/w1/weight");
    load_case.copied(layer + "mlp.up_proj.weight", {128, 64}, block + "mlp/w3/weight");
    load_case.copied(layer + "mlp.down_proj.weight", {64, 128}, block + "mlp/w2/weight");

    expect_every_parameter_loaded([] { return ttml::models::llama::create(tiny_llama_config()); }, dtype, load_case);
}

}  // namespace

// FLOAT32: the file's values exactly. BFLOAT16: each value rounded to bf16 as core::from_vector() rounds it.
class SafetensorsLoaderDtypeTest : public SafetensorsLoaderTest,
                                   public ::testing::WithParamInterface<ttnn::DataType> {};

TEST_P(SafetensorsLoaderDtypeTest, Gpt2LoadsEachValueInTheStoredDtype) {
    check_gpt2_load(GetParam());
}

TEST_P(SafetensorsLoaderDtypeTest, LlamaLoadsEachValueInTheStoredDtype) {
    check_llama_load(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    StoredDtypes,
    SafetensorsLoaderDtypeTest,
    ::testing::Values(ttnn::DataType::FLOAT32, ttnn::DataType::BFLOAT16),
    [](const ::testing::TestParamInfo<ttnn::DataType>& info) {
        return info.param == ttnn::DataType::FLOAT32 ? "Fp32" : "Bf16";
    });

TEST_F(SafetensorsLoaderTest, AssignFromFloatsRejectsNonFloatTensors) {
    auto* device = &ttml::autograd::ctx().get_device();
    auto tensor = ttml::autograd::create_tensor(ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        std::vector<uint32_t>(32 * 32, 1U), ttnn::Shape({1, 1, 32, 32}), device));
    EXPECT_THAT(
        [&] { ttml::serialization::assign_from_floats(tensor, std::vector<float>(32 * 32, 0.5F)); },
        ::testing::ThrowsMessage<std::invalid_argument>(
            ::testing::HasSubstr("assign_from_floats loads into bf16 or fp32 tensors")));
}
