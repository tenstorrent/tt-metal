// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/bfloat16.hpp"
#include "tensor/host_buffer/functions.hpp"
#include "tensor/host_buffer/types.hpp"
#include "tensor/types.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_eager/tensor/tensor.hpp"
#include "tt_numpy/functions.hpp"
#include "tt_dnn/op_library//sliding_window_op_infra/sliding_window.hpp"

#include "tensor/tensor.hpp"

#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <numeric>
#include <tuple>
#include <algorithm>

using tt::tt_metal::Tensor;
using tt::tt_metal::DataType;
using tt::tt_metal::Layout;
using tt::tt_metal::Shape;

using namespace std;

/* convolution for single channel*/
owned_buffer::Buffer<bfloat16> ref_conv_op(
    Tensor &input_padded_tensor,
    Shape input_nchw_shape,
    Tensor& filter_pyt_tensor,
    Shape& filter_pyt_tensor_shape,
    Shape &out_golden_pyt_tensor_shape
) {
    auto input_padded_tensor_buf = owned_buffer::get_as<bfloat16>(input_padded_tensor);
    auto filter_pyt_tensor_buf = owned_buffer::get_as<bfloat16>(filter_pyt_tensor);
    uint32_t output_n, output_c, output_h, output_w;
    std::tie(output_n, output_c, output_h, output_w) = std::tie(out_golden_pyt_tensor_shape[0],
                                                                out_golden_pyt_tensor_shape[1],
                                                                out_golden_pyt_tensor_shape[2],
                                                                out_golden_pyt_tensor_shape[3]);
    uint32_t filter_n, filter_c, filter_h, filter_w;
    std::tie(filter_n, filter_c, filter_h, filter_w) = std::tie(filter_pyt_tensor_shape[0],
                                                                filter_pyt_tensor_shape[1],
                                                                filter_pyt_tensor_shape[2],
                                                                filter_pyt_tensor_shape[3]);
    uint32_t input_n, input_c, input_h, input_w;
    std::tie(input_n, input_c, input_h, input_w) = std::tie(input_nchw_shape[0],
                                                            input_nchw_shape[1],
                                                            input_nchw_shape[2],
                                                            input_nchw_shape[3]);
    auto out_golden_pyt_tensor = owned_buffer::create<bfloat16>(output_n * output_c * output_h * output_w);
    /* Create filter vector */
    vector<vector<float>> filter_vector(filter_n, vector<float>());
    for(auto k = 0; k < filter_n; k++) {
        for(auto c = 0; c < filter_c; c++) {
            for(auto h = 0; h < filter_h; h++) {
                for(auto w = 0; w < filter_w; w++) {
                    filter_vector[k].push_back(filter_pyt_tensor_buf[c * filter_h * filter_w + h * filter_w + w].to_float());
                }
            }
        }
    }

    auto out_idx = 0;
    vector<float> input_window;
    for(int fn = 0; fn < filter_n; fn++){
        for(int i = 0; i < output_n; i++) {
            for(int j = 0; j < output_c; j++) {
                for (int k = 0; k < output_h; k++) {
                    for(int l = 0; l < output_w; l++) {
                        /* Get input vector of filter size to calculate convolution.*/
                        for(int m = 0; m < filter_h; m++) {
                            for(int n = 0; n < filter_w; n++) {
                                input_window.push_back(input_padded_tensor_buf[i * (input_c * input_h * input_w) + j * (input_h * input_w) + k * (input_w) + l + m * input_w + n].to_float());
                            }
                        }
                        out_golden_pyt_tensor[out_idx] = bfloat16(static_cast<float>(inner_product(input_window.begin(), input_window.end(), filter_vector[fn].begin(), 0.0)));
                        out_idx++;
                        input_window.clear();
                    }
                }
            }
        }
    }

    return out_golden_pyt_tensor;
}

uint32_t compare_out_with_golden(owned_buffer::Buffer<bfloat16> &out_golden_tensor_buf, owned_buffer::Buffer<bfloat16> &conv_tensor_buf) {
    uint32_t diff = 0;
    if(out_golden_tensor_buf != conv_tensor_buf) {
        assert(out_golden_tensor_buf.size() == conv_tensor_buf.size());
        for(uint32_t i = 0; i < out_golden_tensor_buf.size(); i++) {
            if(out_golden_tensor_buf[i] != conv_tensor_buf[i]) {
                std::cout << "Error at i=" << i << ", golden=" << out_golden_tensor_buf[i].to_float()  << ", result=" << conv_tensor_buf[i].to_float() << std::endl;
                diff++;
            }
        }
    }
    return diff;
}

// Validate generate op trace metadata
owned_buffer::Buffer<bfloat16> conv_op_trace_metadata(
    owned_buffer::Buffer<bfloat16> &input_padded_tensor_buf,
    vector<float> &filter_vector,
    vector<uint32_t> &op_trace_metadata,
    uint32_t filter_h,
    uint32_t filter_w,
    uint32_t padded_input_w,
    uint32_t out_tensor_size
) {
    auto conv_tensor_buf = owned_buffer::create<bfloat16>(out_tensor_size);
    vector<float> inputs;
    uint32_t out_idx = 0;
    for(auto anchor:op_trace_metadata) {
        for(uint32_t h = 0; h < filter_h; h++) {
            for(uint32_t w = 0; w < filter_w; w++) {
                inputs.push_back(input_padded_tensor_buf[anchor + h * padded_input_w + w].to_float());
            }
        }
        conv_tensor_buf[out_idx] = bfloat16(static_cast<float>(inner_product(inputs.begin(), inputs.end(), filter_vector.begin(), 0.0)));
        out_idx++;
        inputs.clear();
    }
    return conv_tensor_buf;
}

owned_buffer::Buffer<bfloat16> conv_shard_boundries(
    owned_buffer::Buffer<bfloat16> &input_padded_tensor_buf,
    vector<float> &filter_vector,
    vector<std::pair<uint32_pair_t, uint32_pair_t>> shard_boundaries,
    uint32_t input_h,
    uint32_t input_w,
    uint32_t padded_input_h,
    uint32_t padded_input_w,
    uint32_t filter_h,
    uint32_t filter_w,
    uint32_t out_tensor_size
) {
    auto conv_tensor_buf = owned_buffer::create<bfloat16>(out_tensor_size);
    vector<float> inputs;

    uint32_t input_hw = input_h * input_w;
    uint32_t padded_input_hw = padded_input_h * padded_input_w;
    for(auto shard_boundry:shard_boundaries) {
        auto [output_shard_start, output_shard_end] = shard_boundry.first;
        auto [input_index_start, input_index_end] = shard_boundry.second;
        for(auto i = output_shard_start; i <= output_shard_end; i++) {
            for(auto fh = 0; fh < filter_h; fh++) {
                for(auto fw = 0; fw < filter_w; fw++) {
                    auto input_idx_strt = (i / input_hw) * padded_input_hw + ((i % input_hw) / input_h) * padded_input_w + i % input_w;
                    auto idx = input_idx_strt + fh * padded_input_w + fw;
                    assert(idx >= input_index_start && idx <= input_index_end);
                    inputs.push_back(input_padded_tensor_buf[idx].to_float());
                }
            }
            conv_tensor_buf[i] = bfloat16(static_cast<float>(inner_product(inputs.begin(), inputs.end(), filter_vector.begin(), 0.0)));
            inputs.clear();
        }
    }
    return conv_tensor_buf;
}


owned_buffer::Buffer<bfloat16> conv_tensor_metadata(
    owned_buffer::Buffer<bfloat16> &input_padded_tensor_buf,
    vector<float> &filter_vector,
    vector<std::pair<uint32_pair_t, uint32_pair_t>> &shard_boundaries,
    vector<std::pair<bool, uint32_pair_t>> &tensor_metadata,
    uint32_t input_h,
    uint32_t input_w,
    uint32_t padded_input_h,
    uint32_t padded_input_w,
    uint32_t filter_h,
    uint32_t filter_w,
    uint32_t out_tensor_size
) {
    auto conv_tensor_buf = owned_buffer::create<bfloat16>(out_tensor_size);
    uint32_t out_shard_end_idx = 0;
    /* Manually calculate end idx of each shard*/
    for(auto metadata:tensor_metadata) {
        if(metadata.second.first != 0)
            break;
        out_shard_end_idx = max(out_shard_end_idx, metadata.second.second);
    }
    vector<float> inputs;
    uint32_t input_hw = input_h * input_w;
    uint32_t padded_input_hw = padded_input_h * padded_input_w;
    for(auto metadata:tensor_metadata) {
        auto is_pad_stick = metadata.first;
        auto [core_id, input_reshard_local_idx] = metadata.second;
        if(is_pad_stick) continue;
        auto out_idx = core_id * (out_shard_end_idx + 1) + input_reshard_local_idx;
        for(auto fh = 0; fh < filter_h; fh++) {
            for(auto fw = 0; fw < filter_w; fw++) {
                auto input_idx_strt = ((out_idx / input_hw) * padded_input_hw) + ((out_idx % input_hw) / input_h) * padded_input_w + out_idx % input_w;
                auto idx =  input_idx_strt + fh * padded_input_w + fw;
                inputs.push_back(input_padded_tensor_buf[idx].to_float());
            }
        }
        conv_tensor_buf[out_idx] = bfloat16(static_cast<float>(inner_product(inputs.begin(), inputs.end(), filter_vector.begin(), 0.0)));
        inputs.clear();
    }
    return conv_tensor_buf;
}

owned_buffer::Buffer<bfloat16> conv_sliding_window_op_config(
    owned_buffer::Buffer<bfloat16> &input_padded_tensor_buf,
    vector<float> &filter_vector,
    vector<uint32_t> &op_trace_metadata,
    vector<std::pair<uint32_pair_t, uint32_pair_t>> &shard_boundaries,
    vector<std::vector<uint16_t>> sharded_input_top_left_indices,
    uint32_t input_h,
    uint32_t input_w,
    uint32_t padded_input_h,
    uint32_t padded_input_w,
    uint32_t filter_h,
    uint32_t filter_w,
    uint32_t out_tensor_size
) {
    auto conv_tensor_buf = owned_buffer::create<bfloat16>(out_tensor_size);

    vector<float> input_window;
    uint32_t out_idx = 0;

    for(auto j = 0; j < sharded_input_top_left_indices.size(); j++) {
        auto shard = sharded_input_top_left_indices[j];
        auto [output_shard_start, output_shard_end] = shard_boundaries[j].first;
        for(auto idx:shard) {
            for(auto fh = 0; fh < filter_h; fh++) {
                for(auto fw = 0; fw < filter_w; fw++) {
                    input_window.push_back(input_padded_tensor_buf[op_trace_metadata[output_shard_start] + idx + fh * padded_input_w + fw].to_float());
                }
            }
            conv_tensor_buf[out_idx] = bfloat16(static_cast<float>(inner_product(input_window.begin(), input_window.end(), filter_vector.begin(), 0.0)));
            out_idx++;
            input_window.clear();
        }
    }
    return conv_tensor_buf;
}

uint32_t validate_generate_functions(
    SlidingWindowConfig &config,
    owned_buffer::Buffer<bfloat16> &input_padded_tensor_buf,
    owned_buffer::Buffer<bfloat16> &filter_tensor_buf,
    owned_buffer::Buffer<bfloat16> &out_golden_tensor_buf,
    uint32_t reshard_num_cores_nhw=0,
    bool pad_tile = false,
    bool pad_last_core = false
) {
    owned_buffer::Buffer<bfloat16> conv_tensor_buf;
    uint32_t diff;
    uint32_t failed_tests = 0;
    auto op_trace_metadata = sliding_window::generate_op_trace_metadata(config);
    auto shard_boundaries = sliding_window::generate_shard_boundaries(config, op_trace_metadata);
    auto sharded_input_top_left_indices = sliding_window::generate_sliding_window_op_config(op_trace_metadata, shard_boundaries, pad_tile, pad_last_core);
    auto pad_metadata = sliding_window::generate_pad_metadata(config);
    auto tensor_metadata = sliding_window::generate_tensor_metadata(pad_metadata, config, reshard_num_cores_nhw);

    auto [filter_h, filter_w] = config.window_hw_;
    auto [input_h, input_w] = config.input_hw_;
    auto [stride_h, stride_w] = config.stride_hw_;

    uint32_t padded_input_h = input_h + 2 * config.pad_hw_.first;
    uint32_t padded_input_w = input_w + 2 * config.pad_hw_.second;

    vector<float> filter_vector;
    for(auto h = 0; h < filter_h; h++) {
        for(auto w = 0; w < filter_w; w++) {
            filter_vector.push_back(filter_tensor_buf[ h * filter_w + w].to_float());
        }
    }

    conv_tensor_buf = conv_op_trace_metadata(
        input_padded_tensor_buf,
        filter_vector,
        op_trace_metadata,
        filter_h,
        filter_w,
        padded_input_w,
        out_golden_tensor_buf.size());
    diff = compare_out_with_golden(out_golden_tensor_buf, conv_tensor_buf);
    if(diff) {
        log_info(tt::LogTest, "Failed to validate generate_tensor_metadata, convolution calculated with op_trace_metadata differs at locations = ", diff);
        failed_tests++;
        diff = 0;
    }

    conv_tensor_buf = conv_shard_boundries(
        input_padded_tensor_buf,
        filter_vector,
        shard_boundaries,
        input_h,
        input_w,
        padded_input_h,
        padded_input_w,
        filter_h,
        filter_w,
        out_golden_tensor_buf.size());
    diff = compare_out_with_golden(out_golden_tensor_buf, conv_tensor_buf);
    if(diff) {
        log_info(tt::LogTest, "Failed to validate generate_tensor_metadata, convolution calculated with op_trace_metadata differs at locations = ", diff);
        failed_tests++;
        diff = 0;
    }

    conv_tensor_buf = conv_tensor_metadata(
        input_padded_tensor_buf,
        filter_vector,
        shard_boundaries,
        tensor_metadata,
        input_h,
        input_w,
        padded_input_h,
        padded_input_w,
        filter_h,
        filter_w,
        out_golden_tensor_buf.size());
    diff = compare_out_with_golden(out_golden_tensor_buf, conv_tensor_buf);
    if(diff) {
        log_info(tt::LogTest, "Failed to validate generate_tensor_metadata convolution calculated with op_trace_metadata differs at locations = ", diff);
        failed_tests++;
        diff = 0;
    }

    conv_tensor_buf = conv_sliding_window_op_config(
        input_padded_tensor_buf,
        filter_vector,
        op_trace_metadata,
        shard_boundaries,
        sharded_input_top_left_indices,
        input_h,
        input_w,
        padded_input_h,
        padded_input_w,
        filter_h,
        filter_w,
        out_golden_tensor_buf.size());
    diff = compare_out_with_golden(out_golden_tensor_buf, conv_tensor_buf);
    if(diff) {
        log_info(tt::LogTest, "Failed to validate generate_tensor_metadata convolution calculated with op_trace_metadata differs at locations = ", diff);
        failed_tests++;
        diff = 0;
    }
    return failed_tests;
}

vector<vector<int>> configs = {
    {1, 5, 5, 3, 3, 1, 1, 1, 1, 1, 1, 1},
    {1, 10, 10, 7, 7, 1, 1, 3, 3, 1, 1, 3},
    {7, 5, 5, 3, 3, 1, 1, 1, 1, 1, 1, 5},
};
int main () {
    int device_id = 0;
    auto device = tt::tt_metal::CreateDevice(device_id);

    for(auto c:configs) {
        SlidingWindowConfig config = SlidingWindowConfig(c[0], c[1], c[2],c[3], c[4], c[5],c[6], c[7], c[8],c[9], c[10], c[11]);
        Shape input_tensor_shape = {config.batch_size_, 1, config.input_hw_.first + 2*config.pad_hw_.first, config.input_hw_.second + 2*config.pad_hw_.second};
        Shape output_tensor_shape = {config.batch_size_, 1, config.input_hw_.first, config.input_hw_.second};
        Shape filter_tensor_shape = {1, 1, config.window_hw_.first, config.window_hw_.second};

        auto input_padded_tensor = tt::numpy::random::random(input_tensor_shape, DataType::BFLOAT16).to(Layout::ROW_MAJOR).cpu();
        auto filter_tensor = tt::numpy::random::random(filter_tensor_shape, DataType::BFLOAT16).to(Layout::ROW_MAJOR).cpu();
        auto input_padded_tensor_buf = owned_buffer::get_as<bfloat16>(input_padded_tensor);
        auto filter_tensor_buf = owned_buffer::get_as<bfloat16>(filter_tensor);

        auto out_golden_tensor_buf = ref_conv_op(input_padded_tensor, input_tensor_shape, filter_tensor, filter_tensor_shape, output_tensor_shape);

        auto failed_tests = validate_generate_functions(config, input_padded_tensor_buf, filter_tensor_buf, out_golden_tensor_buf, false, false, 2);
        if (failed_tests) {
            TT_THROW("Tests Falied = ", failed_tests);
        } else {
            log_info(tt::LogTest, "Test Passed");
        }
    }
    TT_FATAL(tt::tt_metal::CloseDevice(device));
    return 0;
}
