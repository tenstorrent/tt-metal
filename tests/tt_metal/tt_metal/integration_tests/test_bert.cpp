#include "constants.hpp"
#include "tensor/tensor.hpp"
#include "tensor/owned_buffer.hpp"
#include "tt_metal/host_api.hpp"

#include "tt_numpy/functions.hpp"

#include "tt_dnn/op_library/operation.hpp"
#include "tt_dnn/op_library/program_cache.hpp"

#include "tt_dnn/op_library/bmm/bmm_op.hpp"
#include "tt_dnn/op_library/bcast/bcast_op.hpp"
#include "tt_dnn/op_library/bert_large_tms/bert_large_tms.hpp"
#include "tt_dnn/op_library/layernorm/layernorm_op.hpp"
#include "tt_dnn/op_library/softmax/softmax_op.hpp"

#include <chrono>

using Parameters = std::map<std::string, Tensor>;

constexpr auto l1_memory_config = tt::tt_metal::MemoryConfig{.interleaved=true,.buffer_type=tt::tt_metal::BufferType::L1};
constexpr auto dram_memory_config = tt::tt_metal::MemoryConfig{.interleaved=true,.buffer_type=tt::tt_metal::BufferType::DRAM};

Tensor encoder(Tensor&& hidden_states, const Tensor& attention_mask, const Parameters& parameters, std::size_t encoder_index, const std::uint32_t head_size) {

    auto bert_large_fused_qkv_matmul_output = bert_large_fused_qkv_matmul(
        hidden_states,
        parameters.at(fmt::format("fused_qkv_weight_{}", encoder_index)),
        parameters.at(fmt::format("fused_qkv_bias_{}", encoder_index)),
        l1_memory_config
    );


    auto bert_large_create_qkv_heads_output = bert_large_create_qkv_heads(bert_large_fused_qkv_matmul_output, l1_memory_config);
    auto query = bert_large_create_qkv_heads_output[0];
    auto key = bert_large_create_qkv_heads_output[1];
    auto value = bert_large_create_qkv_heads_output[2];
    bert_large_fused_qkv_matmul_output.deallocate();
    bert_large_create_qkv_heads_output.clear();


    auto bert_large_pre_softmax_bmm_output = bert_large_pre_softmax_bmm(query, key, dram_memory_config);
    query.deallocate();
    key.deallocate();


    bert_large_pre_softmax_bmm_output = scale_mask_softmax_in_place(1.0f / std::sqrt(head_size), attention_mask, bert_large_pre_softmax_bmm_output);


    auto bert_large_post_softmax_bmm_output = bert_large_post_softmax_bmm(bert_large_pre_softmax_bmm_output, value, l1_memory_config);
    bert_large_pre_softmax_bmm_output.deallocate();
    value.deallocate();


    auto bert_large_concat_heads_output = bert_large_concat_heads(bert_large_post_softmax_bmm_output, l1_memory_config);
    bert_large_post_softmax_bmm_output.deallocate();


    auto bert_large_selfout_bmm_output = bert_large_selfout_matmul(
        bert_large_concat_heads_output,
        parameters.at(fmt::format("selfout_weight_{}", encoder_index)),
        parameters.at(fmt::format("selfout_bias_{}", encoder_index)),
        l1_memory_config
    );
    bert_large_concat_heads_output.deallocate();


    auto attention_layernorm_output = bert_large_add_layernorm(
        hidden_states,
        bert_large_selfout_bmm_output,
        1e-12,
        parameters.at(fmt::format("attention_layernorm_weight_{}", encoder_index)),
        parameters.at(fmt::format("attention_layernorm_bias_{}", encoder_index)),
        l1_memory_config
    );
    hidden_states.deallocate();
    bert_large_selfout_bmm_output.deallocate();


    auto bert_large_ff1_matmul_output = bert_large_ff1_matmul(
        attention_layernorm_output,
        parameters.at(fmt::format("ff1_weight_{}", encoder_index)),
        parameters.at(fmt::format("ff1_bias_{}", encoder_index)),
        true,
        dram_memory_config
    );


    auto bert_large_ff2_matmul_output = bert_large_ff2_matmul(
        bert_large_ff1_matmul_output,
        parameters.at(fmt::format("ff2_weight_{}", encoder_index)),
        parameters.at(fmt::format("ff2_bias_{}", encoder_index)),
        l1_memory_config
    );
    bert_large_ff1_matmul_output.deallocate();


    auto feedforward_layernorm_output = bert_large_add_layernorm(
        attention_layernorm_output,
        bert_large_ff2_matmul_output,
        1e-12,
        parameters.at(fmt::format("feedforward_layernorm_weight_{}", encoder_index)),
        parameters.at(fmt::format("feedforward_layernorm_bias_{}", encoder_index)),
        l1_memory_config
    );
    attention_layernorm_output.deallocate();
    bert_large_ff2_matmul_output.deallocate();


    return feedforward_layernorm_output;
}

Tensor qa_head(Tensor&& hidden_states, const Parameters& parameters) {

    auto output = matmul(hidden_states, parameters.at("qa_head_weight"));
    hidden_states.deallocate();


    return bcast(output, parameters.at("qa_head_bias"), tt::tt_metal::BcastOpMath::Enum::ADD, tt::tt_metal::BcastOpDim::Enum::H, l1_memory_config);
}


void test_bert() {
    using tt::constants::TILE_HEIGHT;
    using tt::constants::TILE_WIDTH;
    using tt::tt_metal::DataType;
    using tt::tt_metal::Device;
    using tt::tt_metal::Host;
    using tt::tt_metal::Layout;
    using tt::tt_metal::Tensor;

    tt::log_info(tt::LogTest, "Running {}", __func__);

    int pci_express_slot = 0;
    auto device = tt::tt_metal::CreateDevice(tt::ARCH::GRAYSKULL, pci_express_slot);
    auto host = tt::tt_metal::GetHost();

    TT_ASSERT(tt::tt_metal::InitializeDevice(device));

    std::size_t num_iterations = 2;
    std::size_t num_encoders = 24;
    std::uint32_t batch_size = 9;
    std::uint32_t sequence_size = 384;
    std::uint32_t num_heads = 16;
    std::uint32_t head_size = 64;
    std::uint32_t hidden_size = num_heads * head_size;
    std::uint32_t intermediate_size = hidden_size * 4;

    auto attention_mask = tt::numpy::random::uniform(bfloat16(-1.0f), bfloat16(1.0f), {1, 1, TILE_HEIGHT, sequence_size}, Layout::TILE).to(device, l1_memory_config);

    auto parameters = Parameters{};
    for (auto encoder_index = 0; encoder_index < num_encoders; encoder_index++) {
        parameters.emplace(fmt::format("fused_qkv_weight_{}", encoder_index), tt::numpy::random::uniform(bfloat16(-1.0f), bfloat16(1.0f), {1, 1, hidden_size, hidden_size * 3}, Layout::TILE).to(device, dram_memory_config));
        parameters.emplace(fmt::format("fused_qkv_bias_{}", encoder_index), tt::numpy::random::uniform(bfloat16(-1.0f), bfloat16(1.0f), {1, 1, TILE_HEIGHT, hidden_size * 3}, Layout::TILE).to(device, dram_memory_config));
        parameters.emplace(fmt::format("selfout_weight_{}", encoder_index), tt::numpy::random::uniform(bfloat16(-1.0f), bfloat16(1.0f), {1, 1, hidden_size, hidden_size}, Layout::TILE).to(device, dram_memory_config));
        parameters.emplace(fmt::format("selfout_bias_{}", encoder_index), tt::numpy::random::uniform(bfloat16(-1.0f), bfloat16(1.0f), {1, 1, TILE_HEIGHT, hidden_size}, Layout::TILE).to(device, dram_memory_config));
        parameters.emplace(fmt::format("attention_layernorm_weight_{}", encoder_index), tt::numpy::random::uniform(bfloat16(-1.0f), bfloat16(1.0f), {1, 1, TILE_HEIGHT, TILE_WIDTH}, Layout::ROW_MAJOR).to(device, dram_memory_config));
        parameters.emplace(fmt::format("attention_layernorm_bias_{}", encoder_index), tt::numpy::random::uniform(bfloat16(-1.0f), bfloat16(1.0f), {1, 1, TILE_HEIGHT, TILE_WIDTH}, Layout::ROW_MAJOR).to(device, dram_memory_config));
        parameters.emplace(fmt::format("ff1_weight_{}", encoder_index), tt::numpy::random::uniform(bfloat16(-1.0f), bfloat16(1.0f), {1, 1, hidden_size, intermediate_size}, Layout::TILE).to(device, dram_memory_config));
        parameters.emplace(fmt::format("ff1_bias_{}", encoder_index), tt::numpy::random::uniform(bfloat16(-1.0f), bfloat16(1.0f), {1, 1, TILE_HEIGHT, intermediate_size}, Layout::TILE).to(device, dram_memory_config));
        parameters.emplace(fmt::format("ff2_weight_{}", encoder_index), tt::numpy::random::uniform(bfloat16(-1.0f), bfloat16(1.0f), {1, 1, intermediate_size, hidden_size}, Layout::TILE).to(device, dram_memory_config));
        parameters.emplace(fmt::format("ff2_bias_{}", encoder_index), tt::numpy::random::uniform(bfloat16(-1.0f), bfloat16(1.0f), {1, 1, TILE_HEIGHT, hidden_size}, Layout::TILE).to(device, dram_memory_config));
        parameters.emplace(fmt::format("feedforward_layernorm_weight_{}", encoder_index), tt::numpy::random::uniform(bfloat16(-1.0f), bfloat16(1.0f), {1, 1, TILE_HEIGHT, TILE_WIDTH}, Layout::ROW_MAJOR).to(device, dram_memory_config));
        parameters.emplace(fmt::format("feedforward_layernorm_bias_{}", encoder_index), tt::numpy::random::uniform(bfloat16(-1.0f), bfloat16(1.0f), {1, 1, TILE_HEIGHT, TILE_WIDTH}, Layout::ROW_MAJOR).to(device, dram_memory_config));
    };
    parameters.emplace("qa_head_weight", tt::numpy::random::uniform(bfloat16(-1.0f), bfloat16(1.0f), {1, 1, hidden_size, TILE_WIDTH}, Layout::TILE).to(device, dram_memory_config));
    parameters.emplace("qa_head_bias", tt::numpy::random::uniform(bfloat16(-1.0f), bfloat16(1.0f), {1, 1, TILE_HEIGHT, TILE_WIDTH}, Layout::TILE).to(device, dram_memory_config));

    auto run_bert = [&]() {
        tt::log_info(tt::LogTest, "run_bert started");
        auto begin = std::chrono::steady_clock::now();
        auto hidden_states = tt::numpy::random::uniform(bfloat16(-1.0f), bfloat16(1.0f), {batch_size, 1, sequence_size, hidden_size}, Layout::TILE).to(device, l1_memory_config);
        for (auto encoder_index = 0; encoder_index < num_encoders; encoder_index++) {
            hidden_states = encoder(std::move(hidden_states), attention_mask, parameters, encoder_index, head_size);
        }
        auto output = qa_head(std::move(hidden_states), parameters).to(host);
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
        tt::log_info(tt::LogTest, "run_bert finished in {} microseconds", duration);
        return duration;
    };

    auto run_loop = [&]() {
        auto total_duration = 0;
        for (int iteration = 0; iteration < num_iterations; iteration++) {
            total_duration += run_bert();
        }
        auto average_duration = total_duration / num_iterations;
        auto num_samples_per_second = 1e6 / average_duration * batch_size;
        tt::log_info(tt::LogTest, "total duration: {} microseconds", total_duration);
        tt::log_info(tt::LogTest, "average duration: {} average_duration", total_duration);
        tt::log_info(tt::LogTest, "samples per second: {}", num_samples_per_second);
    };

    tt::tt_metal::program_cache::enable();
    run_bert();
    run_loop();
    tt::tt_metal::program_cache::disable_and_clear();

    TT_ASSERT(tt::tt_metal::CloseDevice(device));
}

int main(int argc, char** argv) {
    test_bert();
    return 0;
}
