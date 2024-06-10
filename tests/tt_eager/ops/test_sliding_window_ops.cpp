// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <sys/wait.h>
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
#include <numeric>
#include <tuple>

using tt::tt_metal::Tensor;
using tt::tt_metal::Shape;


vector<float> create_filter_vector(owned_buffer::Buffer<bfloat16> &filter_tensor_buf, uint32_t filter_h, uint32_t filter_w) {
    vector<float> filter_vector;
    for(auto h = 0; h < filter_h; h++) {
        for(auto w = 0; w < filter_w; w++) {
            filter_vector.push_back(filter_tensor_buf[ h * filter_w + w].to_float());
        }
    }
    return filter_vector;
}

owned_buffer::Buffer<bfloat16> ref_conv_op(
    Tensor &input_padded_tensor,
    Shape input_nchw_shape,
    uint32_t stride_h,
    uint32_t stride_w,
    vector<float> &filter_vector,
    Shape& filter_pyt_tensor_shape,
    Shape &out_golden_pyt_tensor_shape
) {
    auto input_padded_tensor_buf = owned_buffer::get_as<bfloat16>(input_padded_tensor);
    uint32_t output_n, output_h, output_w;
    std::tie(output_n, output_h, output_w) = std::tie(out_golden_pyt_tensor_shape[0],
                                                      out_golden_pyt_tensor_shape[1],
                                                      out_golden_pyt_tensor_shape[2]);
    uint32_t filter_h, filter_w;
    std::tie(filter_h, filter_w) = std::tie(filter_pyt_tensor_shape[0],
                                            filter_pyt_tensor_shape[1]);
    uint32_t input_n, input_h, input_w;
    std::tie(input_n, input_h, input_w) = std::tie(input_nchw_shape[0],
                                                   input_nchw_shape[1],
                                                   input_nchw_shape[2]);

    auto out_golden_pyt_tensor = owned_buffer::create<bfloat16>(output_n * output_h * output_w);

    auto out_idx = 0;
    vector<float> input_window;
    for(int i = 0; i < output_n; i++) {
        for (int k = 0; k < output_h; k++) {
            for(int l = 0; l < output_w; l++) {
                /* Get input vector of filter size to calculate convolution.*/
                for(int m = 0; m < filter_h; m++) {
                    for(int n = 0; n < filter_w; n++) {
                        auto anchor = i * (input_h * input_w) + k * stride_h * input_w + l * stride_w;
                        auto idx = anchor + m * input_w * stride_h + n * stride_w;
                        input_window.push_back(input_padded_tensor_buf[idx].to_float());
                    }
                }
                out_golden_pyt_tensor[out_idx] = bfloat16(static_cast<float>(inner_product(input_window.begin(), input_window.end(), filter_vector.begin(), 0.0)));
                out_idx++;
                input_window.clear();
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
                log_info(tt::LogTest, "Error at i = {}, Golden = {}, Calculated = {}", i, out_golden_tensor_buf[i].to_float(), conv_tensor_buf[i].to_float());
                diff++;
            }
        }
    }
    return diff;
}

owned_buffer::Buffer<bfloat16> conv_using_op_trace_metadata(
    owned_buffer::Buffer<bfloat16> &input_padded_tensor_buf,
    vector<float> &filter_vector,
    vector<uint32_t> &op_trace_metadata,
    uint32_t stride_h,
    uint32_t stride_w,
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
                auto idx = anchor + h * stride_h * padded_input_w + w * stride_w;
                inputs.push_back(input_padded_tensor_buf[idx].to_float());
            }
        }
        conv_tensor_buf[out_idx] = bfloat16(static_cast<float>(inner_product(inputs.begin(), inputs.end(), filter_vector.begin(), 0.0)));
        out_idx++;
        inputs.clear();
    }
    return conv_tensor_buf;
}

owned_buffer::Buffer<bfloat16> conv_using_shard_boundaries(
    owned_buffer::Buffer<bfloat16> &input_padded_tensor_buf,
    vector<float> &filter_vector,
    vector<std::pair<uint32_pair_t, uint32_pair_t>> shard_boundaries,
    uint32_t input_h,
    uint32_t input_w,
    uint32_t stride_h,
    uint32_t stride_w,
    uint32_t padded_input_h,
    uint32_t padded_input_w,
    uint32_t filter_h,
    uint32_t filter_w,
    uint32_t output_h,
    uint32_t output_w,
    uint32_t out_tensor_size
) {
    auto conv_tensor_buf = owned_buffer::create<bfloat16>(out_tensor_size);
    vector<float> inputs;

    uint32_t input_hw = input_h * input_w;
    uint32_t output_hw = output_h * output_w;
    uint32_t padded_input_hw = padded_input_h * padded_input_w;
    uint32_t input_idx_strt, input_idx;
    for(auto shard_boundry:shard_boundaries) {
        auto [output_shard_start, output_shard_end] = shard_boundry.first;
        auto [input_index_start, input_index_end] = shard_boundry.second;
        for(auto i = output_shard_start; i <= output_shard_end; i++) {
            for(auto fh = 0; fh < filter_h; fh++) {
                for(auto fw = 0; fw < filter_w; fw++) {
                    input_idx_strt = (i / output_hw) * padded_input_hw + ((i % output_hw) / output_h) * padded_input_w * stride_h + (i % output_w) * stride_w;
                    input_idx = input_idx_strt + fh * padded_input_w * stride_h + fw * stride_w;
                    inputs.push_back(input_padded_tensor_buf[input_idx].to_float());
                }
            }
            conv_tensor_buf[i] = bfloat16(static_cast<float>(inner_product(inputs.begin(), inputs.end(), filter_vector.begin(), 0.0)));
            inputs.clear();
        }
    }
    return conv_tensor_buf;
}


vector<bool> pad_metadata_using_tensor_metadata(
    vector<std::pair<bool, uint32_pair_t>> &tensor_metadata
) {
    vector<bool> ref_pad_metadata;
    for(auto i = 0; i < tensor_metadata.size(); i++) {
        auto is_pad_stick = tensor_metadata[i].first;
        if(is_pad_stick) {
            ref_pad_metadata.push_back(true);
            continue;
        }
        ref_pad_metadata.push_back(false);
    }
    return ref_pad_metadata;
}

owned_buffer::Buffer<bfloat16> conv_using_sliding_window_op_config(
    owned_buffer::Buffer<bfloat16> &input_padded_tensor_buf,
    vector<float> &filter_vector,
    vector<uint32_t> &op_trace_metadata,
    vector<std::pair<uint32_pair_t, uint32_pair_t>> &shard_boundaries,
    vector<std::vector<uint16_t>> sharded_input_top_left_indices,
    uint32_t input_h,
    uint32_t input_w,
    uint32_t stride_h,
    uint32_t stride_w,
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
                    input_window.push_back(input_padded_tensor_buf[op_trace_metadata[output_shard_start] + idx + fh * padded_input_w * stride_h + fw * stride_w].to_float());
                }
            }
            conv_tensor_buf[out_idx] = bfloat16(static_cast<float>(inner_product(input_window.begin(), input_window.end(), filter_vector.begin(), 0.0)));
            out_idx++;
            input_window.clear();
        }
    }
    return conv_tensor_buf;
}

uint32_t validate_generate_halo_kernel_config(
    tt::tt_metal::Device *device,
    vector<std::pair<uint32_pair_t, uint32_pair_t>> &shard_boundaries,
    tuple<vector<vector<uint16_t>>, vector<vector<uint16_t>>, vector<vector<uint16_t>>, uint32_t> & halo_kernel_config,
    vector<bool> &pad_metadata,
    bool remote_read = false,
    bool is_block_sharded = false,
    bool transpose_mcast = false
) {
    auto [flattened_pad_config, flattened_local_config, flattened_remote_config, max_out_n_sticks_per_core] = halo_kernel_config;

    uint32_t padded_input_tensor_buf_idx = 0;
    uint32_t invalid_pads = 0, invalid_idx = 0;
    uint32_t failed_tests = 0;

    auto find_invalids = [&](int idx, int length, bool count_padding)->int {
        // Safe to use pad_metadata to validate since already validated for same input
        auto invalids = 0;
        for(auto i = idx; i < (idx + length); i++){
            if((count_padding == true) && (pad_metadata[i] != true)) {
                invalids++;
                log_info(tt::LogTest, "Error at i = {}, Calculated = {}", i, pad_metadata[i]);
            } else if((count_padding == false) && (pad_metadata[i] == true)){
                invalids++;
                log_info(tt::LogTest, "Error at i = {}, Calculated = {}", i, pad_metadata[i]);
            }
        }
        return invalids;
    };
    for(auto i = 0; i < shard_boundaries.size(); i++) {
        auto input_boundry = shard_boundaries[i].second;
        auto input_start = input_boundry.first;

        auto pad_config = flattened_pad_config[i];
        padded_input_tensor_buf_idx = input_start;
        for(auto j = 0; j < pad_config.size(); j+=2) {
            auto local_idx = pad_config[j];
            auto length = pad_config[j + 1];
            invalid_pads += find_invalids(padded_input_tensor_buf_idx + local_idx, length, true);
        }
    }

    if(invalid_pads != 0) {
        log_error(tt::LogTest, "Failed to validate flattened_pad_config of halo_kernel_config, invalid pads = {}", invalid_pads);
        failed_tests++;
    }

    for(auto i = 0; i < shard_boundaries.size(); i++) {
        auto input_boundry = shard_boundaries[i].second;
        auto input_start = input_boundry.first;

        auto local_config = flattened_local_config[i];
        padded_input_tensor_buf_idx = input_start;
        auto sz = local_config[2];
        for(auto j = 3; j < (sz + 3); j+=3) {
            auto local_idx = local_config[j+1];
            auto length = local_config[j + 2];
            invalid_idx += find_invalids(padded_input_tensor_buf_idx + local_idx, length, false);
        }
    }
    if(invalid_idx != 0) {
        log_error(tt::LogTest, "Failed to validate flattened_local_config of halo_kernel_config, invalid indexes = {}", invalid_idx);
        failed_tests++;
        invalid_idx = 0;
    }

    auto core_id_to_noc_coords = [is_block_sharded, transpose_mcast, device](uint32_t core_id) -> CoreCoord {
        auto num_cores_x = device->compute_with_storage_grid_size().x;
        auto core_coord = is_block_sharded ? (transpose_mcast ? CoreCoord(core_id, 0) : CoreCoord(0, core_id)) : CoreCoord(core_id % num_cores_x, core_id / num_cores_x);
        return device->worker_core_from_logical_core(core_coord);
    };

    unordered_map<CoreCoord, int> phy_to_log_core_map;
    for(uint32_t i = 0;i < shard_boundaries.size(); i++) {
        phy_to_log_core_map[core_id_to_noc_coords(i)] = i;
    }

    for(auto i = 0; i < shard_boundaries.size(); i++) {
        auto remote_config = flattened_remote_config[i];
        CoreCoord coord = {remote_config[0], remote_config[1]};
        auto core_id = phy_to_log_core_map[coord];
        padded_input_tensor_buf_idx = remote_read ? shard_boundaries[i].second.first:shard_boundaries[core_id].second.first;
        auto sz = remote_config[2];
        auto local_idx = 3;
        while(sz) {
            for(auto i = local_idx; i < local_idx + sz; i+=3) {
                auto local_idx = remote_config[i + 1];
                auto length = remote_config[i + 2];
                invalid_idx += find_invalids(padded_input_tensor_buf_idx + local_idx, length, false);
            }
            local_idx += sz;
            coord = {remote_config[local_idx], remote_config[local_idx+1]};
            core_id = phy_to_log_core_map[coord];
            padded_input_tensor_buf_idx = remote_read ? shard_boundaries[i].second.first:shard_boundaries[core_id].second.first;
            local_idx += 2;
            sz = remote_config[local_idx];
            local_idx++;
        }
    }
    if(invalid_idx != 0) {
        log_error(tt::LogTest, "Failed to validate flattened_remote_config of halo_kernel_config, invalid indexes = {}", invalid_idx);
        failed_tests++;
    }

    return failed_tests;
}

uint32_t validate_generate_functions(
    tt::tt_metal::Device *device,
    SlidingWindowConfig &config,
    owned_buffer::Buffer<bfloat16> &input_padded_tensor_buf,
    vector<float> &filter_vector,
    owned_buffer::Buffer<bfloat16> &out_golden_tensor_buf,
    uint32_t reshard_num_cores_nhw=0,
    bool remote_read = false
) {
    owned_buffer::Buffer<bfloat16> conv_tensor_buf;
    uint32_t diff;
    uint32_t failed_tests = 0;
    auto pad_metadata = sliding_window::generate_pad_metadata(config);
    auto tensor_metadata = sliding_window::generate_tensor_metadata(pad_metadata, config, reshard_num_cores_nhw);
    auto op_trace_metadata = sliding_window::generate_op_trace_metadata(config);
    auto shard_boundaries = sliding_window::generate_shard_boundaries(config, op_trace_metadata);
    auto sharded_input_top_left_indices = sliding_window::generate_sliding_window_op_config(op_trace_metadata, shard_boundaries, false, false);
    auto halo_kernel_config = sliding_window::generate_halo_kernel_config_tensors(tensor_metadata, shard_boundaries, false, false, remote_read, device);

    auto [filter_h, filter_w] = config.window_hw_;
    auto [input_h, input_w] = config.input_hw_;
    auto [stride_h, stride_w] = config.stride_hw_;
    auto output_shape = config.get_output_shape();
    uint32_t output_n, output_h, output_w;
    std::tie(output_n, output_h, output_w) = std::tie(output_shape[0], output_shape[1], output_shape[2]);

    uint32_t padded_input_h = input_h + 2 * config.pad_hw_.first;
    uint32_t padded_input_w = input_w + 2 * config.pad_hw_.second;

    auto ref_pad_metadata = pad_metadata_using_tensor_metadata(tensor_metadata);
    if(ref_pad_metadata != pad_metadata) {
        for(auto i = 0; i < ref_pad_metadata.size(); i++) {
            if(ref_pad_metadata[i] != pad_metadata[i])
                log_info(tt::LogTest, "Error at i = {}, Calculated = {}", i, ref_pad_metadata[i]);
        }
        log_error(tt::LogTest, "Failed to validate generate_tensor_metadata, convolution calculated with op_trace_metadata differs at locations = {}", diff);
        failed_tests++;
    }

    conv_tensor_buf = conv_using_op_trace_metadata(
        input_padded_tensor_buf,
        filter_vector,
        op_trace_metadata,
        stride_h,
        stride_w,
        filter_h,
        filter_w,
        padded_input_w,
        out_golden_tensor_buf.size());
    diff = compare_out_with_golden(out_golden_tensor_buf, conv_tensor_buf);
    if(diff) {
        log_error(tt::LogTest, "Failed to validate generate_tensor_metadata, convolution calculated with op_trace_metadata differs at locations = {}", diff);
        failed_tests++;
    }

    conv_tensor_buf = conv_using_shard_boundaries(
        input_padded_tensor_buf,
        filter_vector,
        shard_boundaries,
        input_h,
        input_w,
        stride_h,
        stride_w,
        padded_input_h,
        padded_input_w,
        filter_h,
        filter_w,
        output_h,
        output_w,
        out_golden_tensor_buf.size());
    diff = compare_out_with_golden(out_golden_tensor_buf, conv_tensor_buf);
    if(diff) {
        log_error(tt::LogTest, "Failed to validate generate_shard_boundaries, convolution calculated with op_trace_metadata differs at locations = {}", diff);
        failed_tests++;
    }

    conv_tensor_buf = conv_using_sliding_window_op_config(
        input_padded_tensor_buf,
        filter_vector,
        op_trace_metadata,
        shard_boundaries,
        sharded_input_top_left_indices,
        input_h,
        input_w,
        stride_h,
        stride_w,
        padded_input_w,
        filter_h,
        filter_w,
        out_golden_tensor_buf.size());
    diff = compare_out_with_golden(out_golden_tensor_buf, conv_tensor_buf);
    if(diff) {
        log_error(tt::LogTest, "Failed to validate generate_sliding_window_op_config, convolution calculated with op_trace_metadata differs at locations = {}", diff);
        failed_tests++;
    }
    failed_tests += validate_generate_halo_kernel_config(device, shard_boundaries, halo_kernel_config, pad_metadata, remote_read);
    return failed_tests;
}

// config = {batch_size_, input_h, input_w, filter_h, filter_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, has_parallel_config, reshard_num_cores_nhw }
struct testcase_config {
    uint32_t batch_size;
    uint32_t input_h, input_w;
    uint32_t filter_h, filter_w;
    uint32_t stride_h, stride_w;
    uint32_t pad_h, pad_w;
    uint32_t dilation_h, dilation_w;
    uint32_t num_cores_nhw;
    uint32_t reshard_num_cores_nhw;
    bool remote_read;
};

vector<struct testcase_config> configs = {
    {2, 5, 5, 3, 3, 1, 1, 1, 1, 1, 1, 2, 0, false},
    {2, 5, 5, 3, 3, 2, 2, 1, 1, 1, 1, 1, 4, true},
    {2, 10, 10, 7, 7, 4, 4, 3, 3, 1, 1, 4, 5, false},
    {7, 64, 64, 13, 13, 2, 2, 6, 6, 1, 1, 5, 4, true},
};

int main () {
    int device_id = 0;
    auto device = tt::tt_metal::CreateDevice(device_id);

    log_info(tt::LogTest, "Tests for Sliding window metadata calcations starts");
    for(auto tc:configs) {
        SlidingWindowConfig config = SlidingWindowConfig(
            tc.batch_size,
            tc.input_h,
            tc.input_w,
            tc.filter_h,
            tc.filter_w,
            tc.stride_h,
            tc.stride_w,
            tc.pad_h,
            tc.pad_w,
            tc.dilation_h,
            tc.dilation_w,
            tc.num_cores_nhw);
        Shape input_tensor_shape = {config.batch_size_, config.input_hw_.first + 2 * config.pad_hw_.first, config.input_hw_.second + 2 * config.pad_hw_.second};
        Shape output_tensor_shape = config.get_output_shape();
        Shape filter_tensor_shape = {config.window_hw_.first, config.window_hw_.second};

        Tensor input_padded_tensor = tt::numpy::random::random(input_tensor_shape, DataType::BFLOAT16).to(Layout::ROW_MAJOR).cpu();
        Tensor filter_tensor = tt::numpy::random::random(filter_tensor_shape, DataType::BFLOAT16).to(Layout::ROW_MAJOR).cpu();
        auto input_padded_tensor_buf = owned_buffer::get_as<bfloat16>(input_padded_tensor);
        auto filter_tensor_buf = owned_buffer::get_as<bfloat16>(filter_tensor);

        vector<float> filter_vector = create_filter_vector(filter_tensor_buf, tc.filter_h, tc.filter_w);
        owned_buffer::Buffer<bfloat16> out_golden_tensor_buf = ref_conv_op(input_padded_tensor, input_tensor_shape, tc.stride_h, tc.stride_w, filter_vector, filter_tensor_shape, output_tensor_shape);

        auto failed_tests = validate_generate_functions(device, config, input_padded_tensor_buf, filter_vector, out_golden_tensor_buf, tc.reshard_num_cores_nhw, tc.remote_read);
        if (failed_tests) {
            log_error(tt::LogTest, "Tests({}) failed for config ({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})",
                      failed_tests, tc.batch_size, tc.input_h, tc.input_w, tc.filter_h, tc.filter_w, tc.stride_h, tc.stride_w,
                      tc.pad_h, tc.pad_w, tc.dilation_h, tc.dilation_w, tc.num_cores_nhw, tc.reshard_num_cores_nhw);
            TT_THROW("Tests Falied");
        } else {
            log_info(tt::LogTest, "Tests Passed");
        }
    }
    log_info(tt::LogTest, "Tests for Sliding window metadata calcations ends");
    TT_FATAL(tt::tt_metal::CloseDevice(device));
    return 0;
}
