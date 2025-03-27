// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "models/llama.hpp"

#include <fcntl.h>
#include <gtest/gtest.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <cerrno>
#include <core/ttnn_all_includes.hpp>
#include <filesystem>
#include <ttnn/types.hpp>
#include <xtensor/xnpy.hpp>

#include "autograd/auto_context.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/tt_tensor_utils.hpp"
#include "core/xtensor_utils.hpp"
#include "gtest/gtest.h"
#include "modules/embedding_module.hpp"
#include "ops/multi_head_utils.hpp"
#include "ops/scaled_dot_product_attention.hpp"
#include "serialization/serialization.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

class LlamaTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

TEST_F(LlamaTest, RopeE2E) {
    using namespace ttml;
    auto seq_len = 32;
    auto head_dim = 64;
    auto theta = 10000.0F;
    auto rope_params = ttml::ops::build_rope_params(seq_len, head_dim, theta);

    fmt::println("testing overall rope");
    xt::xarray<float> rope_input_xt = xt::load_npy<float>("/home/j/intermediate_results/query_states_before_rope.npy");
    rope_input_xt = rope_input_xt.reshape({1, 32, 32, head_dim});

    auto interleave_halves = [&](const xt::xarray<float>& x, int dim = -1) -> xt::xarray<float> {
        // Normalize dim to positive
        if (dim < 0) {
            dim += x.dimension();
        }

        size_t d = x.shape()[dim];
        assert(d % 2 == 0 && "hidden dim must be even");

        // Split the array along the specified dimension
        auto a = xt::view(x, xt::all(), xt::all(), xt::all(), xt::range(0, d / 2));
        auto b = xt::view(x, xt::all(), xt::all(), xt::all(), xt::range(d / 2, d));

        // Stack and reshape to get interleaved result
        auto stacked = xt::stack(xtuple(a, b), dim + 1);
        auto result_shape = x.shape();
        xt::xarray<float> reshaped = xt::reshape_view(stacked, result_shape);
        return reshaped;
    };

    rope_input_xt = interleave_halves(rope_input_xt);

    auto rope_input =
        autograd::create_tensor(core::from_xtensor(rope_input_xt, &autograd::ctx().get_device(), ttnn::TILE_LAYOUT));
    auto rope_res = ttml::ops::rope(rope_input, rope_params);
    auto rope_res_xt = core::to_xtensor(rope_res->get_value());
    fmt::println("rope res shape: {}", rope_res_xt.shape());
    rope_res_xt = rope_res_xt.reshape({1, 32, 32, head_dim});
    auto expected_rope_res = xt::load_npy<float>("/home/j/intermediate_results/query_states_after_rope.npy");

    expected_rope_res = interleave_halves(expected_rope_res);
    auto max_diff_rope = xt::amax(xt::abs(expected_rope_res - rope_res_xt))();
    fmt::println("max diff for rope: {}", max_diff_rope);

    EXPECT_TRUE(max_diff_rope < .5F);
}

template <class E1, class E2>
std::pair<float, float> suggest_atol_rtol(
    std::string const& label, const E1& expected, const E2& actual, size_t first_n = 32) {
    auto abs_diffs = xt::abs(expected - actual);

    float max_abs_diff = xt::amax(abs_diffs)();
    float atol = max_abs_diff * 1.2f;  // 20 % safety margin

    constexpr float eps = 1e-5f;
    auto denom = xt::clip(xt::abs(expected), eps, std::numeric_limits<float>::max());
    auto rel_diffs = abs_diffs / denom;

    // ignore tiny expected values when taking the max
    auto valid_mask = xt::abs(expected) > eps;
    auto rel_pruned = xt::where(valid_mask, rel_diffs, 0.0f);  // zeros don't affect max
    float max_rel = xt::amax(rel_pruned)();
    float rtol = 1.2f * max_rel;

    fmt::println("[{}] suggested atol: {}", label, atol);
    fmt::println("[{}] suggested rtol: {}", label, rtol);

    auto expected_flat = xt::flatten(expected);
    auto actual_flat = xt::flatten(actual);
    size_t n = first_n == 0 ? expected_flat.size() : std::min(first_n, expected_flat.size());
    xt::xarray<float> expected_prefix = xt::view(expected_flat, xt::range(0, n));
    xt::xarray<float> actual_prefix = xt::view(actual_flat, xt::range(0, n));

    xt::xarray<float> zipped = xt::stack(xtuple(expected_prefix, actual_prefix), 1);
    fmt::println("[{}] expected vs actual (first {}): {}", label, n, zipped);

    fmt::println("[{}] expected shape: {}", label, expected.shape());
    fmt::println("[{}] actual shape: {}", label, actual.shape());
    return std::make_pair(atol, rtol);
}

ttml::models::llama::Llama init_llama_cached(uint32_t seq_len = 32) {
    namespace fs = std::filesystem;
    using namespace ttml;

    auto yaml_config = YAML::LoadFile("configs/training_shakespeare_tinyllama.yaml");
    auto training_config = yaml_config["training_config"];
    auto llama_config = training_config["transformer_config"];
    auto config = models::llama::read_config(llama_config);
    config.max_sequence_length = seq_len;

    auto llama_model = models::llama::Llama(config);
    llama_model.eval();

    auto home_dir = fs::path(std::getenv("HOME"));
    auto cache_dir = home_dir / ".cache/ttml/llama";
    if (!fs::exists(cache_dir)) {
        fs::create_directories(cache_dir);
    }

    auto cache_file = cache_dir / "tinyllama.msgpack.mmap";
    bool cache_exists = fs::exists(cache_file);
    bool cache_is_old = false;
    if (cache_exists) {
        auto cache_time = fs::last_write_time(cache_file);
        auto msgpack_time = fs::last_write_time("/home/j/load_llama/tinyllama.msgpack");
        cache_is_old = cache_time < msgpack_time;
    }

    ttml::serialization::MsgPackFile msgpack_file;
    if (cache_exists && !cache_is_old) {
        fmt::println("loading tinyllama.msgpack from cache");
        {  // Scope for the input archive and file stream
            std::ifstream ifs(cache_file, std::ios::binary);
            if (!ifs) {
                // Handle error: file couldn't be opened
                // For now, let's re-throw or handle appropriately.
                // Since this is a cache, maybe falling back to non-cached path is better?
                // However, the original code structure implies we proceed if cache exists.
                // Let's throw an exception for clarity in the test.
                throw std::runtime_error("Failed to open cache file for reading: " + cache_file.string());
            }
            boost::archive::binary_iarchive ia(ifs);
            // Deserialize the MsgPackFile object from the archive
            ia >> msgpack_file;
        }  // ifs and ia are destroyed here, closing the file
    } else {
        fmt::println("loading tinyllama.msgpack");
        msgpack_file.deserialize("/home/j/load_llama/tinyllama.msgpack");
        fmt::println("deserialized tinyllama.msgpack");

        // save to cache file
        {
            // compute the size of the cache file based on the msgpack_file
            // Compute the size of the cache file based on the msgpack_file
            // First, serialize to a temporary buffer to determine size
            std::ostringstream temp_stream;
            boost::archive::binary_oarchive temp_archive(temp_stream);
            temp_archive << msgpack_file;

            // Get the size from the temporary stream
            std::string temp_data = temp_stream.str();
            size_t file_size = temp_data.size();

            // Ensure the file has the computed size
            fmt::println("Creating cache file with size: {} bytes", file_size);
            int fd = open(cache_file.string().c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
            if (fd == -1) {
                throw std::runtime_error("Failed to open cache file for writing: " + cache_file.string());
            }
            // Ensure the file is the right size before mapping
            if (ftruncate(fd, file_size) == -1) {
                close(fd);
                throw std::runtime_error(
                    "Failed to resize cache file: " + cache_file.string() + ", error: " + strerror(errno));
            }

            // mmap it
            void* mapped_memory = mmap(NULL, file_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
            if (mapped_memory == MAP_FAILED) {
                close(fd);
                throw std::runtime_error(
                    "Failed to mmap cache file: " + cache_file.string() + ", error: " + strerror(errno));
            }

            // Write the serialized data to the mapped memory
            std::memcpy(mapped_memory, temp_data.data(), file_size);
            fmt::println("Successfully wrote {} bytes to cache file", file_size);

            // locking the pages
            if (mlock(mapped_memory, file_size) == -1) {
                throw std::runtime_error("Failed to mlock cache file: " + cache_file.string());
            }
            close(fd);
        }
    }

    fmt::println("loading weights");
    ttml::serialization::read_module(msgpack_file, /*name=*/"llama", &llama_model);
    fmt::println("loaded weights");
    return llama_model;
}

ttml::models::llama::Llama init_llama(uint32_t seq_len = 32) {
    using namespace ttml;
    auto yaml_config = YAML::LoadFile("configs/training_shakespeare_tinyllama.yaml");
    auto training_config = yaml_config["training_config"];
    auto llama_config = training_config["transformer_config"];
    auto config = models::llama::read_config(llama_config);
    config.max_sequence_length = seq_len;

    auto llama_model = models::llama::Llama(config);
    llama_model.eval();

    fmt::println("loading tinyllama.msgpack");
    ttml::serialization::MsgPackFile tinyllama_msgpack{};
    tinyllama_msgpack.deserialize("data/tinyllama_exported.msgpack");
    fmt::println("deserialized tinyllama.msgpack");

    fmt::println("loading weights");
    ttml::serialization::read_module(tinyllama_msgpack, /*name=*/"llama", &llama_model);
    fmt::println("loaded weights");
    return llama_model;
}

TEST_F(LlamaTest, TokenEmbedding) {
    using namespace ttml;
    // Load the float version first
    xt::xarray<float> input_ids_float_xt = xt::load_npy<float>("/home/j/intermediate_results/test_input_tokens.npy");
    // Convert the float xtensor to an int xtensor
    xt::xarray<uint32_t> input_ids_xt = xt::cast<uint32_t>(input_ids_float_xt.reshape({1U, 1U, 1U, 32U}));
    fmt::println("input_ids_xt: {}", input_ids_xt);
    xt::xarray<float> attention_mask_xt = xt::ones<float>({1, 1, 32, 32});
    for (int i = 0; i < 32; ++i) {
        for (int j = 0; j < 32; ++j) {
            if (j > i) {  // Mask out attention to future positions
                attention_mask_xt(0, 0, i, j) = 0.0F;
            }
        }
    }
    ttnn::Tensor input_ids_tt = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        input_ids_xt, &autograd::ctx().get_device(), /*layout=*/ttnn::Layout::ROW_MAJOR);
    auto x = autograd::create_tensor(input_ids_tt);
    auto mask = autograd::create_tensor(
        core::from_xtensor(attention_mask_xt, &autograd::ctx().get_device(), ttnn::TILE_LAYOUT));

    auto llama_model = init_llama(32);
    auto tok_emb = llama_model.tok_emb;
    // Cast the ModuleBase pointer to an Embedding pointer using dynamic_pointer_cast
    auto tok_emb_casted = std::dynamic_pointer_cast<ttml::modules::Embedding>(tok_emb);

    // Check if the cast was successful before accessing the member
    if (!tok_emb_casted) {
        throw std::runtime_error("Failed to cast tok_emb to Embedding type.");
    }
    auto tok_emb_res = (*tok_emb)(x);
    fmt::println("checking tok_emb_res");
    xt::xarray<float> actual_tok_emb_res = core::to_xtensor(tok_emb_res->get_value());
    xt::xarray<float> expected_tok_emb_res =
        xt::load_npy<float>("/home/j/intermediate_results/test_embedded_tokens_new.npy");
    expected_tok_emb_res = expected_tok_emb_res.reshape({1U, 1U, 32U, 2048U});
    float atol_emb = suggest_atol_rtol("tok_emb", expected_tok_emb_res, actual_tok_emb_res).first;
    EXPECT_TRUE(atol_emb < .25F);
}

TEST_F(LlamaTest, MLP) {
    using namespace ttml;
    auto llama_model = init_llama(32);
    fmt::println("checking the mlp");
    std::shared_ptr<ttml::modules::LlamaBlock> first_block_ptr =
        std::dynamic_pointer_cast<ttml::modules::LlamaBlock>(llama_model.blocks[0]);
    auto mlp = first_block_ptr->m_mlp;

    xt::xarray<float> mlp_input_xt = xt::load_npy<float>("/home/j/intermediate_results/mlp_test_input.npy");
    auto mlp_input = autograd::create_tensor(core::from_xtensor(mlp_input_xt, &autograd::ctx().get_device()));
    auto mlp_res = (*mlp)(mlp_input);
    xt::xarray<float> mlp_res_xt = core::to_xtensor(mlp_res->get_value());
    xt::xarray<float> expected_mlp_res = xt::load_npy<float>("/home/j/intermediate_results/mlp_test_output.npy");
    expected_mlp_res = expected_mlp_res.reshape({2048});
    mlp_res_xt = mlp_res_xt.reshape({2048});
    float atol_mlp = suggest_atol_rtol("mlp", expected_mlp_res, mlp_res_xt).first;
    EXPECT_TRUE(atol_mlp < .25F);
}

TEST_F(LlamaTest, QProjTest) {
    using namespace ttml;
    xt::xarray<float> q_input = xt::load_npy<float>("/home/j/intermediate_results/first_q_test_input.npy");
    xt::xarray<float> q_expected = xt::load_npy<float>("/home/j/intermediate_results/expected_first_q_result.npy");
    auto llama_model = init_llama(32);
    std::shared_ptr<ttml::modules::LlamaBlock> first_block_ptr =
        std::dynamic_pointer_cast<ttml::modules::LlamaBlock>(llama_model.blocks[0]);
    auto q_proj = first_block_ptr->m_attention->m_q_linear;

    auto q_input_tt = core::from_xtensor(q_input, &autograd::ctx().get_device());
    auto q_input_ag = autograd::create_tensor(q_input_tt);
    auto q_res = (*q_proj)(q_input_ag);
    auto q_res_xt = core::to_xtensor(q_res->get_value());

    float atol_q_proj = suggest_atol_rtol("q_proj", q_expected, q_res_xt, 0).first;
    EXPECT_TRUE(atol_q_proj < .25F);
}

TEST_F(LlamaTest, KVProjTest) {
    using namespace ttml;
    xt::xarray<float> kv_input = xt::load_npy<float>("/home/j/intermediate_results/first_q_test_input.npy");
    xt::xarray<float> kv_expected = xt::load_npy<float>("/home/j/intermediate_results/expected_first_kv_result.npy");
    auto llama_model = init_llama(32);
    std::shared_ptr<ttml::modules::LlamaBlock> first_block_ptr =
        std::dynamic_pointer_cast<ttml::modules::LlamaBlock>(llama_model.blocks[0]);
    auto kv_proj = first_block_ptr->m_attention->m_kv_linear;
    auto q_proj = first_block_ptr->m_attention->m_q_linear;
    auto kv_input_tt = core::from_xtensor(kv_input, &autograd::ctx().get_device());
    auto kv_input_ag = autograd::create_tensor(kv_input_tt);
    auto kv_res = (*kv_proj)(kv_input_ag);
    auto kv_res_xt = core::to_xtensor(kv_res->get_value());

    float atol_kv_proj = suggest_atol_rtol("kv_proj", kv_expected, kv_res_xt, 0).first;
    EXPECT_TRUE(atol_kv_proj < .25F);

    xt::xarray<float> expected_k = xt::load_npy<float>("/home/j/intermediate_results/expected_first_k_result.npy");
    xt::xarray<float> expected_v = xt::load_npy<float>("/home/j/intermediate_results/expected_first_v_result.npy");
    auto q_res = (*q_proj)(kv_input_ag);
    auto [query_with_heads, key_with_heads, value_with_heads] = ops::grouped_heads_creation(q_res, kv_res, 32, 4);

    fmt::println("query_with_heads shape: {}", query_with_heads->get_value().logical_shape());
    fmt::println("key_with_heads shape: {}", key_with_heads->get_value().logical_shape());
    fmt::println("value_with_heads shape: {}", value_with_heads->get_value().logical_shape());
    xt::xarray<float> actual_k = core::to_xtensor(key_with_heads->get_value());
    xt::xarray<float> actual_v = core::to_xtensor(value_with_heads->get_value());
    float atol_k_proj = suggest_atol_rtol("k_proj", expected_k, actual_k, 0).first;
    float atol_v_proj = suggest_atol_rtol("v_proj", expected_v, actual_v, 0).first;
    EXPECT_TRUE(atol_k_proj < .25F);
    EXPECT_TRUE(atol_v_proj < .25F);
}

TEST_F(LlamaTest, AttnNormTest) {
    using namespace ttml;
    xt::xarray<float> attention_norm_input =
        xt::load_npy<float>("/home/j/intermediate_results/test_embedded_tokens_new.npy");
    xt::xarray<float> expected_attention_norm_res =
        xt::load_npy<float>("/home/j/intermediate_results/expected_first_attn_norm_result.npy");
    attention_norm_input = attention_norm_input.reshape({1U, 1U, 32U, 2048U});
    auto llama_model = init_llama(32);
    std::shared_ptr<ttml::modules::LlamaBlock> first_block_ptr =
        std::dynamic_pointer_cast<ttml::modules::LlamaBlock>(llama_model.blocks[0]);
    auto attention_norm = first_block_ptr->m_attention_norm;
    auto attention_norm_input_tt = core::from_xtensor(attention_norm_input, &autograd::ctx().get_device());
    auto attention_norm_input_ag = autograd::create_tensor(attention_norm_input_tt);
    auto attention_norm_res = (*attention_norm)(attention_norm_input_ag);
    xt::xarray<float> attention_norm_res_xt = core::to_xtensor(attention_norm_res->get_value());
    float atol_attention_norm =
        suggest_atol_rtol("attention_norm", expected_attention_norm_res, attention_norm_res_xt, 0).first;
    EXPECT_TRUE(atol_attention_norm < .25F);
}

TEST_F(LlamaTest, E2E_AttnTest) {
    using namespace ttml;
    xt::xarray<float> attn_input = xt::load_npy<float>("/home/j/intermediate_results/expected_first_attn_input.npy");
    xt::xarray<float> expected_attention_res =
        xt::load_npy<float>("/home/j/intermediate_results/expected_first_attn_output.npy");

    int B = attn_input.shape()[0];
    int S = attn_input.shape()[1];
    int E = attn_input.shape()[2];

    assert(S == 32);
    xt::xarray<float> attn_mask = xt::ones<float>({B, 1, S, S});
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < 32; ++i) {
            for (int j = 0; j < 32; ++j) {
                if (j > i) {  // Mask out attention to future positions
                    attn_mask(b, 0, i, j) = 0.0F;
                }
            }
        }
    }
    attn_input = attn_input.reshape({B, 1, S, E});
    attn_mask = attn_mask.reshape({B, 1, S, S});
    expected_attention_res = expected_attention_res.reshape({B, 1, S, E});

    auto llama_model = init_llama(32);
    std::shared_ptr<ttml::modules::LlamaBlock> first_block_ptr =
        std::dynamic_pointer_cast<ttml::modules::LlamaBlock>(llama_model.blocks[0]);
    auto attention = first_block_ptr->m_attention;
    auto attention_input_tt = core::from_xtensor(attn_input, &autograd::ctx().get_device(), ttnn::TILE_LAYOUT);
    auto attention_input_ag = autograd::create_tensor(attention_input_tt);

    auto attention_mask_tt = core::from_xtensor(attn_mask, &autograd::ctx().get_device(), ttnn::TILE_LAYOUT);
    auto attention_mask_ag = autograd::create_tensor(attention_mask_tt);
    auto attention_res = (*attention)(attention_input_ag, attention_mask_ag);
    xt::xarray<float> attention_res_xt = core::to_xtensor(attention_res->get_value());

    // We need to preserve the same shape as the expected result to compare correctly
    attention_res_xt = attention_res_xt.reshape({B, 1, S, E});
    float atol_attention = suggest_atol_rtol("attention", expected_attention_res, attention_res_xt, 0).first;
    // attention results are very small, so we need to be stricter with the atol.
    EXPECT_TRUE(atol_attention < .1F);
}

TEST_F(LlamaTest, GQA_SDPA_Test) {
    using namespace ttml;
    auto B = 1;
    auto S = 32;
    auto H = 32;
    auto G = 4;
    auto E = 2048;
    auto D = E / H;

    xt::xarray<float> test_q = xt::load_npy<float>("/home/j/intermediate_results/random_sdpa_q.npy");
    test_q.reshape({B, H, S, D});
    xt::xarray<float> test_k = xt::load_npy<float>("/home/j/intermediate_results/random_sdpa_k.npy");
    test_k.reshape({B, G, S, D});
    xt::xarray<float> test_v = xt::load_npy<float>("/home/j/intermediate_results/random_sdpa_v.npy");
    test_v.reshape({B, G, S, D});
    xt::xarray<float> test_mask = xt::load_npy<float>("/home/j/intermediate_results/random_sdpa_mask.npy");
    test_mask.reshape({B, 1, S, S});
    xt::xarray<float> expected_sdpa_res = xt::load_npy<float>("/home/j/intermediate_results/random_sdpa_res.npy");
    expected_sdpa_res.reshape({B, S, H, D});

    auto q_tt = core::from_xtensor(test_q, &autograd::ctx().get_device());
    auto k_tt = core::from_xtensor(test_k, &autograd::ctx().get_device());
    auto v_tt = core::from_xtensor(test_v, &autograd::ctx().get_device());
    auto q_ag = autograd::create_tensor(q_tt);
    auto k_ag = autograd::create_tensor(k_tt);
    auto v_ag = autograd::create_tensor(v_tt);
    auto mask_tt = core::from_xtensor(test_mask, &autograd::ctx().get_device());
    auto mask_ag = autograd::create_tensor(mask_tt);

    // note: I'm not applying rope here for this test for simplicity, but I'll
    // be mirroring this choice in the Python test gen code so it is fine.
    auto attention = ttml::ops::scaled_dot_product_attention(q_ag, k_ag, v_ag, mask_ag, /*is_hf_mode=*/true);
    xt::xarray<float> sdpa_res = core::to_xtensor(attention->get_value());
    sdpa_res = sdpa_res.reshape({B, S, H, D});
    float atol_sdpa = suggest_atol_rtol("sdpa", expected_sdpa_res, sdpa_res, 0).first;
    EXPECT_TRUE(atol_sdpa < .1F);
}

TEST_F(LlamaTest, Torch_GQA_SDPA_Test) {
    using namespace ttml;
    auto B = 1;
    auto S = 32;
    auto H = 32;
    auto G = 4;
    auto E = 2048;
    auto D = E / H;

    xt::xarray<float> test_q = xt::load_npy<float>("/home/j/intermediate_results/torch_sdpa_q.npy");
    test_q.reshape({B, H, S, D});
    xt::xarray<float> test_k = xt::load_npy<float>("/home/j/intermediate_results/torch_sdpa_k.npy");
    test_k.reshape({B, G, S, D});
    xt::xarray<float> test_v = xt::load_npy<float>("/home/j/intermediate_results/torch_sdpa_v.npy");
    test_v.reshape({B, G, S, D});
    xt::xarray<float> test_mask = xt::load_npy<float>("/home/j/intermediate_results/torch_sdpa_mask.npy");
    test_mask.reshape({B, 1, S, S});
    // Generate a causal non-additive mask instead of loading from file
    test_mask = xt::ones<float>({B, 1, S, S});

    for (int i = 0; i < S; ++i) {
        for (int j = 0; j < S; ++j) {
            if (j > i) {  // Mask out attention to future positions
                test_mask(0, 0, i, j) = 0.0F;
            }
        }
    }
    xt::xarray<float> expected_sdpa_res = xt::load_npy<float>("/home/j/intermediate_results/torch_sdpa_res.npy");
    expected_sdpa_res.reshape({B, S, H, D});

    auto q_tt = core::from_xtensor(test_q, &autograd::ctx().get_device());
    auto k_tt = core::from_xtensor(test_k, &autograd::ctx().get_device());
    auto v_tt = core::from_xtensor(test_v, &autograd::ctx().get_device());
    auto q_ag = autograd::create_tensor(q_tt);
    auto k_ag = autograd::create_tensor(k_tt);
    auto v_ag = autograd::create_tensor(v_tt);
    auto mask_tt = core::from_xtensor(test_mask, &autograd::ctx().get_device());
    auto mask_ag = autograd::create_tensor(mask_tt);

    // note: I'm not applying rope here for this test for simplicity, but I'll
    // be mirroring this choice in the Python test gen code so it is fine.
    auto attention = ttml::ops::scaled_dot_product_attention(q_ag, k_ag, v_ag, mask_ag, /*is_hf_mode=*/false);
    xt::xarray<float> sdpa_res = core::to_xtensor(attention->get_value());
    sdpa_res = sdpa_res.reshape({B, S, H, D});
    float atol_sdpa = suggest_atol_rtol("torch_sdpa", expected_sdpa_res, sdpa_res, 0).first;
    EXPECT_TRUE(atol_sdpa < .1F);
}

TEST_F(LlamaTest, GroupSharedMatmulTest) {
    using namespace ttml;
    const std::string data_path = "/home/j/intermediate_results/";

    // --- Test Case 1: MHA (heads == groups) ---
    {
        auto B = 2;
        auto H = 4;
        auto G = 4;
        auto S_q = 16;
        auto S_kv = 16;
        auto D = 32;

        // Load inputs
        xt::xarray<float> mha_q = xt::load_npy<float>(data_path + "mha_q.npy");
        mha_q.reshape({B, H, S_q, D});
        xt::xarray<float> mha_k = xt::load_npy<float>(data_path + "mha_k.npy");
        mha_k.reshape({B, G, S_kv, D});
        xt::xarray<float> mha_v = xt::load_npy<float>(data_path + "mha_v.npy");
        mha_v.reshape({B, G, S_kv, D});

        // Load expected outputs
        xt::xarray<float> expected_mha_scores = xt::load_npy<float>(data_path + "mha_scores_qkt.npy");
        expected_mha_scores.reshape({B, H, S_q, S_kv});
        xt::xarray<float> expected_mha_result = xt::load_npy<float>(data_path + "mha_result_scoresv.npy");
        expected_mha_result.reshape({B, H, S_q, D});

        // Convert to ttnn tensors
        auto q_tt = core::from_xtensor(mha_q, &autograd::ctx().get_device());
        auto k_tt = core::from_xtensor(mha_k, &autograd::ctx().get_device());
        auto v_tt = core::from_xtensor(mha_v, &autograd::ctx().get_device());

        // Test 1a: Q @ K^T (transpose_a=False, transpose_b=True)
        auto mha_scores = ttml::ops::group_shared_matmul(q_tt, k_tt, false, true);
        xt::xarray<float> mha_scores_res = core::to_xtensor(mha_scores);
        mha_scores_res.reshape({B, H, S_q, S_kv});
        float atol_mha_scores = suggest_atol_rtol("mha_scores", expected_mha_scores, mha_scores_res, 0).first;
        EXPECT_TRUE(atol_mha_scores < 1e-5F);

        // Test 1b: Scores @ V (transpose_a=False, transpose_b=False)
        auto mha_result = ttml::ops::group_shared_matmul(mha_scores, v_tt, false, false);
        xt::xarray<float> mha_result_res = core::to_xtensor(mha_result);
        mha_result_res.reshape({B, H, S_q, D});
        float atol_mha_result = suggest_atol_rtol("mha_result", expected_mha_result, mha_result_res, 0).first;
        EXPECT_TRUE(atol_mha_result < 1e-5F);
    }

    // --- Test Case 2: GQA (heads > groups) ---
    {
        auto B = 2;
        auto H = 8;
        auto G = 4;
        auto S_q = 16;
        auto S_kv = 16;
        auto D = 32;

        // Load inputs
        xt::xarray<float> gqa_q = xt::load_npy<float>(data_path + "gqa_q.npy");
        gqa_q.reshape({B, H, S_q, D});
        xt::xarray<float> gqa_k = xt::load_npy<float>(data_path + "gqa_k.npy");
        gqa_k.reshape({B, G, S_kv, D});
        xt::xarray<float> gqa_v = xt::load_npy<float>(data_path + "gqa_v.npy");
        gqa_v.reshape({B, G, S_kv, D});

        // Load expected outputs
        xt::xarray<float> expected_gqa_scores = xt::load_npy<float>(data_path + "gqa_scores_qkt.npy");
        expected_gqa_scores.reshape({B, H, S_q, S_kv});
        xt::xarray<float> expected_gqa_result = xt::load_npy<float>(data_path + "gqa_result_scoresv.npy");
        expected_gqa_result.reshape({B, H, S_q, D});

        // Convert to ttnn tensors
        auto q_tt = core::from_xtensor(gqa_q, &autograd::ctx().get_device());
        auto k_tt = core::from_xtensor(gqa_k, &autograd::ctx().get_device());
        auto v_tt = core::from_xtensor(gqa_v, &autograd::ctx().get_device());

        // Test 2a: Q @ K^T (transpose_a=False, transpose_b=True)
        auto gqa_scores = ttml::ops::group_shared_matmul(q_tt, k_tt, false, true);
        xt::xarray<float> gqa_scores_res = core::to_xtensor(gqa_scores);
        gqa_scores_res.reshape({B, H, S_q, S_kv});
        float atol_gqa_scores = suggest_atol_rtol("gqa_scores", expected_gqa_scores, gqa_scores_res, 0).first;
        EXPECT_TRUE(atol_gqa_scores < 1e-5F);

        // Test 2b: Scores @ V (transpose_a=False, transpose_b=False)
        auto gqa_result = ttml::ops::group_shared_matmul(gqa_scores, v_tt, false, false);
        xt::xarray<float> gqa_result_res = core::to_xtensor(gqa_result);
        gqa_result_res.reshape({B, H, S_q, D});
        float atol_gqa_result = suggest_atol_rtol("gqa_result", expected_gqa_result, gqa_result_res, 0).first;
        EXPECT_TRUE(atol_gqa_result < 1e-5F);
    }
}

TEST_F(LlamaTest, SDPA_Intermediates_MHATest) {
    using namespace ttml;
    const std::string data_path = "/home/j/intermediate_results/";

    auto B = 1;
    auto H = 4;
    auto G = 4; // MHA: H == G
    auto S = 16;
    auto D = 32;
    const float scale = 1.0F / std::sqrtf(static_cast<float>(D));

    // Load inputs
    xt::xarray<float> q_xt = xt::load_npy<float>(data_path + "sdpa_interm_mha_q.npy");
    q_xt.reshape({B, H, S, D});
    xt::xarray<float> k_xt = xt::load_npy<float>(data_path + "sdpa_interm_mha_k.npy");
    k_xt.reshape({B, G, S, D});
    xt::xarray<float> v_xt = xt::load_npy<float>(data_path + "sdpa_interm_mha_v.npy");
    v_xt.reshape({B, G, S, D});
    xt::xarray<float> mask_xt = xt::load_npy<float>(data_path + "sdpa_interm_mha_mask.npy");
    mask_xt.reshape({B, 1, S, S}); // Additive mask

    // Load expected intermediates
    xt::xarray<float> expected_q_scaled = xt::load_npy<float>(data_path + "sdpa_interm_mha_q_scaled.npy");
    expected_q_scaled.reshape({B, H, S, D});
    xt::xarray<float> expected_qk_masked = xt::load_npy<float>(data_path + "sdpa_interm_mha_qk_masked.npy");
    expected_qk_masked.reshape({B, H, S, S});
    xt::xarray<float> expected_attn_weights = xt::load_npy<float>(data_path + "sdpa_interm_mha_attn_weights.npy");
    expected_attn_weights.reshape({B, H, S, S});
    xt::xarray<float> expected_attn_qkv = xt::load_npy<float>(data_path + "sdpa_interm_mha_attn_qkv.npy");
    expected_attn_qkv.reshape({B, H, S, D});

    // Convert inputs to ttnn tensors
    auto q_tt = core::from_xtensor(q_xt, &autograd::ctx().get_device());
    auto k_tt = core::from_xtensor(k_xt, &autograd::ctx().get_device());
    auto v_tt = core::from_xtensor(v_xt, &autograd::ctx().get_device());
    auto mask_tt = core::from_xtensor(mask_xt, &autograd::ctx().get_device());

    // --- Intermediate Calculation 1: q_scaled ---
    auto q_scaled = ttnn::experimental::mul(q_tt, scale);
    xt::xarray<float> actual_q_scaled = core::to_xtensor(q_scaled);
    actual_q_scaled.reshape({B, H, S, D});
    float atol_q_scaled = suggest_atol_rtol("mha_q_scaled", expected_q_scaled, actual_q_scaled, 0).first;
    EXPECT_TRUE(atol_q_scaled < 1e-6F);

    // --- Intermediate Calculation 2: qk_masked ---
    // QK Matmul
    ttnn::Tensor qk = ops::group_shared_matmul(q_scaled, k_tt, /*transpose_a=*/false, /*transpose_b=*/true);
    // Apply additive mask
    ttnn::Tensor qk_masked = ttnn::add(qk, mask_tt);
    xt::xarray<float> actual_qk_masked = core::to_xtensor(qk_masked);
    actual_qk_masked.reshape({B, H, S, S});
    float atol_qk_masked = suggest_atol_rtol("mha_qk_masked", expected_qk_masked, actual_qk_masked, 0).first;
    EXPECT_TRUE(atol_qk_masked < 1e-6F); // Matmul might introduce small errors

    // --- Intermediate Calculation 3: attention_weights ---
    auto attention_weights = ttnn_fixed::softmax(qk_masked, /* axis */ 3);
    xt::xarray<float> actual_attn_weights = core::to_xtensor(attention_weights);
    actual_attn_weights.reshape({B, H, S, S});
    float atol_attn_weights = suggest_atol_rtol("mha_attn_weights", expected_attn_weights, actual_attn_weights, 0).first;
    EXPECT_TRUE(atol_attn_weights < 1e-6F); // Softmax is generally stable

    // --- Intermediate Calculation 4: attention_qkv ---
    ttnn::Tensor attention_qkv = ops::group_shared_matmul(attention_weights, v_tt, /*transpose_a=*/false, /*transpose_b=*/false);
    xt::xarray<float> actual_attn_qkv = core::to_xtensor(attention_qkv);
    actual_attn_qkv.reshape({B, H, S, D});
    float atol_attn_qkv = suggest_atol_rtol("mha_attn_qkv", expected_attn_qkv, actual_attn_qkv, 0).first;
    EXPECT_TRUE(atol_attn_qkv < 1e-5F); // Second matmul
}


TEST_F(LlamaTest, SDPA_Intermediates_GQATest) {
    float allowed_atol = .3F;
    using namespace ttml;
    const std::string data_path = "/home/j/intermediate_results/";

    auto B = 1;
    auto H = 8;
    auto G = 2; // GQA: H > G
    auto S = 16;
    auto D = 32;
    const float scale = 1.0F / std::sqrtf(static_cast<float>(D));

    // Load inputs
    xt::xarray<float> q_xt = xt::load_npy<float>(data_path + "sdpa_interm_gqa_q.npy");
    q_xt.reshape({B, H, S, D});
    xt::xarray<float> k_xt = xt::load_npy<float>(data_path + "sdpa_interm_gqa_k.npy");
    k_xt.reshape({B, G, S, D});
    xt::xarray<float> v_xt = xt::load_npy<float>(data_path + "sdpa_interm_gqa_v.npy");
    v_xt.reshape({B, G, S, D});
    xt::xarray<float> mask_xt = xt::load_npy<float>(data_path + "sdpa_interm_gqa_mask.npy");
    mask_xt.reshape({B, 1, S, S}); // Additive mask

    // Load expected intermediates
    xt::xarray<float> expected_q_scaled = xt::load_npy<float>(data_path + "sdpa_interm_gqa_q_scaled.npy");
    expected_q_scaled.reshape({B, H, S, D});
    xt::xarray<float> expected_qk_masked = xt::load_npy<float>(data_path + "sdpa_interm_gqa_qk_masked.npy");
    expected_qk_masked.reshape({B, H, S, S});
    xt::xarray<float> expected_attn_weights = xt::load_npy<float>(data_path + "sdpa_interm_gqa_attn_weights.npy");
    expected_attn_weights.reshape({B, H, S, S});
    xt::xarray<float> expected_attn_qkv = xt::load_npy<float>(data_path + "sdpa_interm_gqa_attn_qkv.npy");
    expected_attn_qkv.reshape({B, H, S, D});

    // Convert inputs to ttnn tensors
    auto q_tt = core::from_xtensor(q_xt, &autograd::ctx().get_device());
    auto k_tt = core::from_xtensor(k_xt, &autograd::ctx().get_device());
    auto v_tt = core::from_xtensor(v_xt, &autograd::ctx().get_device());
    auto mask_tt = core::from_xtensor(mask_xt, &autograd::ctx().get_device());

    // --- Intermediate Calculation 1: q_scaled ---
    auto q_scaled = ttnn::experimental::mul(q_tt, scale);
    xt::xarray<float> actual_q_scaled = core::to_xtensor(q_scaled);
    actual_q_scaled.reshape({B, H, S, D});
    float atol_q_scaled = suggest_atol_rtol("gqa_q_scaled", expected_q_scaled, actual_q_scaled, 0).first;
    bool q_scaled_ok = atol_q_scaled < allowed_atol;
    EXPECT_TRUE(q_scaled_ok);

    // --- Intermediate Calculation 2: qk_masked ---
    // QK Matmul (uses group_shared_matmul for GQA)
    ttnn::Tensor qk = ops::group_shared_matmul(q_scaled, k_tt, /*transpose_a=*/false, /*transpose_b=*/true);
    // Apply additive mask
    ttnn::Tensor qk_masked = ttnn::add(qk, mask_tt);
    xt::xarray<float> actual_qk_masked = core::to_xtensor(qk_masked);
    actual_qk_masked.reshape({B, H, S, S});
    float atol_qk_masked = suggest_atol_rtol("gqa_qk_masked", expected_qk_masked, actual_qk_masked, 0).first;
    bool qk_masked_ok = atol_qk_masked < allowed_atol;
    EXPECT_TRUE(qk_masked_ok);

    // --- Intermediate Calculation 3: attention_weights ---
    auto attention_weights = ttnn_fixed::softmax(qk_masked, /* axis */ 3);
    xt::xarray<float> actual_attn_weights = core::to_xtensor(attention_weights);
    actual_attn_weights.reshape({B, H, S, S});
    float atol_attn_weights = suggest_atol_rtol("gqa_attn_weights", expected_attn_weights, actual_attn_weights, 0).first;
    bool attn_weights_ok = atol_attn_weights < allowed_atol;
    EXPECT_TRUE(attn_weights_ok);

    // --- Intermediate Calculation 4: attention_qkv ---
    // Final Matmul (uses group_shared_matmul for GQA)
    ttnn::Tensor attention_qkv = ops::group_shared_matmul(attention_weights, v_tt, /*transpose_a=*/false, /*transpose_b=*/false);
    xt::xarray<float> actual_attn_qkv = core::to_xtensor(attention_qkv);
    actual_attn_qkv.reshape({B, H, S, D});
    float atol_attn_qkv = suggest_atol_rtol("gqa_attn_qkv", expected_attn_qkv, actual_attn_qkv, 0).first;
    bool attn_qkv_ok = atol_attn_qkv < allowed_atol;
    EXPECT_TRUE(attn_qkv_ok);

    fmt::println("q_scaled_ok: {}", q_scaled_ok);
    fmt::println("qk_masked_ok: {}", qk_masked_ok);
    fmt::println("attn_weights_ok: {}", attn_weights_ok);
    fmt::println("attn_qkv_ok: {}", attn_qkv_ok);
}

TEST_F(LlamaTest, FirstBlock) {
    using namespace ttml;
    auto llama_model = init_llama();
    auto device = &autograd::ctx().get_device();
    xt::xarray<float> attention_mask_xt = xt::ones<float>({1, 1, 32, 32});
    for (int i = 0; i < 32; ++i) {
        for (int j = 0; j < 32; ++j) {
            if (j > i) {  // Mask out attention to future positions
                attention_mask_xt(0, 0, i, j) = 0.0F;
            }
        }
    }
    auto attention_mask_ag = autograd::create_tensor(core::from_xtensor(attention_mask_xt, device));
    xt::xarray<float> first_block_input =
        xt::load_npy<float>("/home/j/intermediate_results/first_block_input_embs.npy");
    first_block_input.reshape({1, 1, 32, 2048});
    xt::xarray<float> expected_first_block_res =
        xt::load_npy<float>("/home/j/intermediate_results/expected_first_block_output.npy");
    auto first_block = llama_model.blocks[0];
    auto input_tt = core::from_xtensor(first_block_input, device);
    auto tok_emb_ag = autograd::create_tensor(input_tt);
    auto actual_first_block_res_tensor = (*first_block)(tok_emb_ag, attention_mask_ag);
    xt::xarray<float> actual_first_block_res = core::to_xtensor(actual_first_block_res_tensor->get_value());
    actual_first_block_res = actual_first_block_res.reshape({1U, 32U, 2048U});
    float atol_first_block = suggest_atol_rtol("first_block", expected_first_block_res, actual_first_block_res).first;
    EXPECT_TRUE(atol_first_block < .25F);
}
