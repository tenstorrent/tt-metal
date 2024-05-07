// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <tuple>

#include "tensor/tensor.hpp"
#include "utils.hpp"

namespace tt::tt_metal {

    using uint32_pair_t = std::pair<uint32_t, uint32_t>;

    struct SlidingWindowConfig {

        // input tensor shape
        uint32_t batch_size_;
        uint32_pair_t input_hw_;

        // windowing parameters
        uint32_pair_t window_hw_;
        uint32_pair_t stride_hw_;
        uint32_pair_t pad_hw_;
        uint32_pair_t dilation_hw_;

        // parallel configuration
        bool has_parallel_config_;
        uint32_t num_cores_nhw_;        // num cores along collapsed height nhw
        CoreRangeSet core_range_set_;   // active cores


        SlidingWindowConfig(uint32_t batch_size, uint32_t input_h, uint32_t input_w, uint32_t window_h, uint32_t window_w, uint32_t stride_h, uint32_t stride_w, uint32_t pad_h, uint32_t pad_w, uint32_t dilation_h = 1, uint32_t dilation_w = 1, uint32_t num_cores_nhw = 0, CoreRangeSet core_range = {{}})
            : batch_size_(batch_size), input_hw_(input_h, input_w), window_hw_(window_h, window_w), stride_hw_(stride_h, stride_w), pad_hw_(pad_h, pad_w), dilation_hw_(dilation_h, dilation_w), has_parallel_config_(false), num_cores_nhw_(num_cores_nhw), core_range_set_(core_range) {
                has_parallel_config_ = num_cores_nhw_ > 0 && !core_range_set_.ranges().empty();
            }

        /**
         * Unique hash val for the sliding window configuration.
         */
        std::size_t get_hash() const {
            return std::hash<std::string>{}(std::to_string(batch_size_)
                                            + "_" + std::to_string(std::get<0>(input_hw_)) + "_" + std::to_string(std::get<1>(input_hw_))
                                            + "_" + std::to_string(std::get<0>(window_hw_)) + "_" + std::to_string(std::get<1>(window_hw_))
                                            + "_" + std::to_string(std::get<0>(stride_hw_)) + "_" + std::to_string(std::get<1>(stride_hw_))
                                            + "_" + std::to_string(std::get<0>(pad_hw_)) + "_" + std::to_string(std::get<1>(pad_hw_))
                                            + "_" + std::to_string(std::get<0>(dilation_hw_)) + "_" + std::to_string(std::get<1>(dilation_hw_))
                                            + "_" + std::to_string(num_cores_nhw_) + "_" + core_range_set_.str());
        }

        /**
         * Return the input shape (excluding depth)
         */
        Shape get_input_shape() const {
            return Shape({batch_size_, std::get<0>(input_hw_), std::get<1>(input_hw_)});
        }

        /**
         * Calculate the window op output shape, excludes the channel dimension since this config is independent of the depth.
         */
        Shape get_output_shape() const {
            uint32_t output_h = (std::get<0>(input_hw_) + 2 * std::get<0>(pad_hw_) - std::get<0>(dilation_hw_) * std::get<0>(window_hw_)) / std::get<0>(stride_hw_) + 1;
            uint32_t output_w = (std::get<1>(input_hw_) + 2 * std::get<1>(pad_hw_) - std::get<1>(dilation_hw_) * std::get<1>(window_hw_)) / std::get<1>(stride_hw_) + 1;
            return Shape({batch_size_, output_h, output_w});
        }

        /**
         * Calculate output tensor shard height
         */
        uint32_t get_output_shard_y(bool snap_to_tile = false) const {
            TT_ASSERT(has_parallel_config_, "Parallel config is not set in SlidingWindowConfig");
            Shape output_shape = get_output_shape();
            uint32_t output_nhw = output_shape[0] * output_shape[1] * output_shape[2];
            uint32_t output_nhw_padded = utils::nearest_y(output_nhw, num_cores_nhw_ * (snap_to_tile ? constants::TILE_HEIGHT : 1));
            return (output_nhw_padded / num_cores_nhw_);
        }
    }; // struct SlidingWindowConfig


    namespace sliding_window {

        std::vector<bool> generate_pad_metadata(const SlidingWindowConfig& config) {
            uint32_t padded_input_h = config.input_hw_.first + 2 * config.pad_hw_.first;
            uint32_t padded_input_w = config.input_hw_.second + 2 * config.pad_hw_.second;
            std::vector<bool> pad_metadata(config.batch_size_ * padded_input_h * padded_input_w, false);

            for (uint32_t b = 0; b < config.batch_size_; ++b) {
                for (uint32_t h = 0; h < padded_input_h; ++h) {
                    for (uint32_t w = 0; w < padded_input_w; ++w) {
                        if (h < config.pad_hw_.first || h >= config.pad_hw_.first + config.input_hw_.first ||
                            w < config.pad_hw_.second || w >= config.pad_hw_.second + config.input_hw_.second) {
                            pad_metadata[b * padded_input_h * padded_input_w + h * padded_input_w + w] = true;
                        }
                    }
                }
            }
            return pad_metadata;
        }

        std::vector<uint32_t> generate_op_trace_metadata(const SlidingWindowConfig& config) {
            Shape output_shape = config.get_output_shape();
            uint32_t output_nhw = output_shape[0] * output_shape[1] * output_shape[2];
            uint32_t padded_input_h = config.input_hw_.first + 2 * config.pad_hw_.first;
            uint32_t padded_input_w = config.input_hw_.second + 2 * config.pad_hw_.second;

            std::vector<uint32_t> op_trace_metadata(output_nhw, 0);
            for (uint32_t b = 0; b < output_shape[0]; ++b) {
                for (uint32_t h = 0; h < output_shape[1]; ++h) {
                    for (uint32_t w = 0; w < output_shape[2]; ++w) {
                        uint32_t input_index = b * padded_input_h * padded_input_w + h * config.stride_hw_.first * padded_input_w + w * config.stride_hw_.second;
                        op_trace_metadata[b * output_shape[1] * output_shape[2] + h * output_shape[2] + w] = input_index;
                    }
                }
            }
            return op_trace_metadata;
        }

        std::vector<std::pair<uint32_pair_t, uint32_pair_t>> generate_shard_boundaries(const SlidingWindowConfig& config, const std::vector<uint32_t>& op_trace_metadata) {
            std::vector<std::pair<uint32_pair_t, uint32_pair_t>> shard_boundaries;
            uint32_t num_cores = config.num_cores_nhw_;
            uint32_t output_shard_h = config.get_output_shard_y();
            uint32_t padded_input_w = config.input_hw_.second + 2 * config.pad_hw_.second;
            uint32_t max_index = op_trace_metadata.size();
            uint32_t halo_with_pad_len = (config.window_hw_.first - 1) * padded_input_w + config.window_hw_.second - 1;
            uint32_t output_index_start = 0;
            for (uint32_t core = 0; core < num_cores; ++ core) {
                uint32_t output_index_end = std::min(output_index_start + output_shard_h, max_index) - 1;
                uint32_t input_index_start = op_trace_metadata[output_index_start];
                uint32_t input_index_end = op_trace_metadata[output_index_end] + halo_with_pad_len;
                shard_boundaries.push_back({{output_index_start, output_index_end}, {input_index_start, input_index_end}});
                output_index_start += output_shard_h;
            }
            return shard_boundaries;
        }

        std::vector<std::pair<bool, uint32_pair_t>> generate_tensor_metadata(const std::vector<bool>& pad_metadata, const SlidingWindowConfig& config, uint32_t reshard_num_cores_nhw = 0) {
            Shape input_shape = config.get_input_shape();
            uint32_t input_nhw = input_shape[0] * input_shape[1] * input_shape[2];
            uint32_t input_shard_height = input_nhw / config.num_cores_nhw_;
            uint32_t input_reshard_height = reshard_num_cores_nhw == 0 ? input_shard_height : input_nhw / reshard_num_cores_nhw;

            auto remap = [input_shard_height, input_reshard_height](uint32_t core_id, uint32_t local_idx) -> std::pair<uint32_t, uint32_t> {
                if (input_shard_height == input_reshard_height) {
                    return std::make_pair(core_id, local_idx);
                } else {
                    uint32_t global_idx = core_id * input_shard_height + local_idx;
                    return std::make_pair(global_idx / input_reshard_height, global_idx % input_reshard_height);
                }
            };

            std::vector<std::pair<bool, uint32_pair_t>> tensor_metadata;
            uint32_t core_id = 0;
            uint32_t input_reshard_local_idx = 0;
            for (bool is_pad_stick : pad_metadata) {
                if (is_pad_stick) {
                    tensor_metadata.push_back(std::make_pair(is_pad_stick, std::make_pair(0, 0)));
                } else {
                    tensor_metadata.push_back(std::make_pair(is_pad_stick, remap(core_id, input_reshard_local_idx++)));
                    if (input_reshard_local_idx == input_shard_height) {
                        core_id++;
                        input_reshard_local_idx = 0;
                    }
                }
            }

            return tensor_metadata;
        }

        std::tuple<std::vector<std::vector<uint16_t>>, std::vector<std::vector<uint16_t>>, std::vector<std::vector<uint16_t>>, uint32_t> generate_halo_kernel_config_tensors(const std::vector<std::pair<bool, uint32_pair_t>>& tensor_metadata, const std::vector<std::pair<uint32_pair_t, uint32_pair_t>>& shard_boundaries, bool remote_read, Device* device) {
            bool is_block_sharding = false; // TODO: get this from config
            bool transpose_mcast = true;    // TODO: get this from config
            auto core_id_to_noc_coords = [is_block_sharding, transpose_mcast, device](uint32_t core_id) -> CoreCoord {
                auto num_cores_x = device->compute_with_storage_grid_size().x;
                auto core_coord = is_block_sharding ? (transpose_mcast ? CoreCoord(core_id, 0) : CoreCoord(0, core_id)) : CoreCoord(core_id % num_cores_x, core_id / num_cores_x);
                return device->worker_core_from_logical_core(core_coord);
            };

            const uint16_t pad_local = 0xFFFF;
            std::map<uint32_pair_t, std::vector<std::tuple<uint32_t, uint32_t, uint32_t>>> per_core_gather_data;

            uint32_t num_core_nhw = shard_boundaries.size();

            uint32_t core_id = 0;
            for (auto [output_boundary, input_boundary] : shard_boundaries) {
                auto [input_start, input_end] = input_boundary;
                for (uint32_t global_idx = input_start; global_idx <= input_end; ++global_idx) {
                    uint32_t dst_core_id = core_id;
                    uint32_t local_idx = global_idx - input_start;
                    auto [is_pad_stick, src_idx] = tensor_metadata[global_idx];
                    auto [src_core_id, src_local_idx] = src_idx;
                    TT_ASSERT(local_idx < pad_local && src_local_idx < pad_local, "Index overflow")
                    if (is_pad_stick) {
                        TT_ASSERT(src_local_idx == 0);
                        src_core_id = pad_local;
                    }
                    if (per_core_gather_data.find({src_core_id, dst_core_id}) != per_core_gather_data.end()) {
                        auto& [src_start, dst_start, length] = per_core_gather_data[{src_core_id, dst_core_id}].back();
                        // src idx is 0 if it is a pad
                        if ((src_local_idx == (src_start + length) || is_pad_stick) && local_idx == (dst_start + length)) {
                            ++ length;
                            continue;
                        }
                    }
                    // insert new tuple
                    per_core_gather_data[{src_core_id, dst_core_id}].push_back({src_local_idx, local_idx, 1});
                }
                ++ core_id;
            }

            // calculate max_out_nsticks_per_core
            uint32_t max_out_nsticks_per_core = 0;
            for (auto [_, in_shard] : shard_boundaries) {
                auto [in_start, in_end] = in_shard;
                max_out_nsticks_per_core = std::max(max_out_nsticks_per_core, in_end - in_start + 1);
            }

            // construct the config tensors
            /**
             * pad_config: length num_cores_nhw
             *     each element (for core i): [dst_start0, length0, dst_start1, length1, ...]
             * local_config: length num_cores_nhw
             *     each element (for core i): (nocx, nocy, len) -> [src_start0, dst_start0, length0, src_start1, dst_start1, length1, ...]
             * remote_config: length num_cores_nhw
             *     each element (for core i): { (nocx, nocy, len) -> [src_start0, dst_start0, length0, src_start1, dst_start1, length1, ...],
             *                                  (nocx, nocy, len) -> [src_start0, dst_start0, length0, src_start1, dst_start1, length1, ...], ...}
             */
            using uint32_triplet_t = std::tuple<uint32_t, uint32_t, uint32_t>;
            std::vector<std::vector<uint32_pair_t>> pad_config;
            std::vector<std::pair<uint32_triplet_t, std::vector<uint32_triplet_t>>> local_config;
            std::vector<std::vector<std::pair<uint32_triplet_t, std::vector<uint32_triplet_t>>>> remote_config;
            pad_config.reserve(num_core_nhw);
            local_config.reserve(num_core_nhw);
            remote_config.reserve(num_core_nhw);

            for (auto [src_dst, data] : per_core_gather_data) {
                auto [src_core_id, dst_core_id] = src_dst;
                bool is_pad = src_core_id == pad_local;
                bool is_local = src_core_id == dst_core_id;
                bool is_remote = !is_local && !is_pad;
                if (is_pad) {
                    for (auto [src_start, dst_start, length] : data) {
                        pad_config[dst_core_id].push_back({dst_start, length});
                    }
                } else if (is_local) {
                    CoreCoord noc_xy = core_id_to_noc_coords(dst_core_id);
                    local_config[src_core_id].first = {noc_xy.x, noc_xy.y, 3 * data.size()};
                    local_config[src_core_id].second = data;
                } else if (is_remote) {
                    if (remote_read) {
                        CoreCoord noc_xy = core_id_to_noc_coords(src_core_id);
                        remote_config[dst_core_id].push_back({{noc_xy.x, noc_xy.y, 3 * data.size()}, data});
                    } else {
                        CoreCoord noc_xy = core_id_to_noc_coords(dst_core_id);
                        remote_config[src_core_id].push_back({{noc_xy.x, noc_xy.y, 3 * data.size()}, data});
                    }
                }
            }

            // TODO: need null at end?

            // flatten and uniformize the lengths of each config list
            auto flatten_pad_config = [](auto& config) -> std::vector<std::vector<uint16_t>> {
                // find max length
                size_t max_len = 0;
                for (auto& data : config) {
                    max_len = std::max(max_len, 2 * data.size());   // each data is 2 * data.size()
                }
                std::vector<std::vector<uint16_t>> flattened_config;
                flattened_config.reserve(config.size());
                for (auto& data : config) {
                    std::vector<uint16_t> flat_data(max_len, 0);
                    uint32_t idx = 0;
                    for (auto data_elem : data) {
                        auto [dst_start, length] = data_elem;
                        flat_data[idx++] = dst_start;
                        flat_data[idx++] = length;
                    }
                }
                return flattened_config;
            };

            auto flatten_local_config = [](auto& config) -> std::vector<std::vector<uint16_t>> {
                // find max length
                size_t max_len = 0;
                for (auto& [_, data] : config) {
                    max_len = std::max(max_len, 3 * data.size());   // each key is 3, data is 3 * data.size()
                }
                max_len += 3;   // key tuple
                std::vector<std::vector<uint16_t>> flattened_config;
                flattened_config.reserve(config.size());
                for (auto& [key, data]: config) {
                    auto [nocx, nocy, len] = key;
                    std::vector<uint16_t> flat_data(max_len, 0);
                    flat_data[0] = nocx;
                    flat_data[1] = nocy;
                    flat_data[2] = len;
                    uint32_t idx = 3;
                    for (size_t i = 0; i < data.size(); ++i) {
                        auto [src_start, dst_start, length] = data[i];
                        flat_data[idx++] = src_start;
                        flat_data[idx++] = dst_start;
                        flat_data[idx++] = length;
                    }
                    flattened_config.push_back(flat_data);
                }
                return flattened_config;
            };

            auto flatten_remote_config = [](auto& config) -> std::vector<std::vector<uint16_t>> {
                // find max length
                size_t max_len = 0;
                for (auto& [_, data] : config) {
                    uint32_t curr_len = 3;  // each key is len 3
                    for (auto& [key, subdata] : data) {
                        curr_len += 3 * subdata.size();
                    }
                    max_len = std::max(max_len, curr_len);   // each key is 3, data is 3 * data.size()
                }
                std::vector<std::vector<uint16_t>> flattened_config;
                flattened_config.reserve(config.size());
                for (auto& core_config : config) {
                    std::vector<uint16_t> flat_data(max_len, 0);
                    uint32_t idx = 0;
                    for (auto& [key, data]: core_config) {
                        auto [nocx, nocy, len] = key;
                        flat_data[0] = nocx;
                        flat_data[1] = nocy;
                        flat_data[2] = len;
                        idx += 3;
                        for (size_t i = 0; i < data.size(); ++i) {
                            auto [src_start, dst_start, length] = data[i];
                            flat_data[idx++] = src_start;
                            flat_data[idx++] = dst_start;
                            flat_data[idx++] = length;
                        }
                    }
                    flattened_config.push_back(flat_data);
                }
                return flattened_config;
            };

            auto flattened_pad_config = flatten_pad_config(pad_config);
            auto flattened_local_config = flatten_local_config(local_config);
            auto flattened_remote_config = flatten_remote_config(remote_config);

            return std::make_tuple(flattened_pad_config,
                                   flattened_local_config,
                                   flattened_remote_config,
                                   max_out_nsticks_per_core);
        }

    } // namespace sliding_window
} // namespace tt::tt_metal
