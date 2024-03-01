// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dtx.hpp"
#include "dtx_passes.hpp"
#include "util.hpp"
#include "util_vector_of_ints.hpp"

using namespace std;

vector<uint32_t> generate_address_map(DataTransformations * dtx, bool in_bytes, uint32_t num_df_bytes) {
    assert(dtx->transformations.size() == 2);
    auto transformation_node = dtx->transformations.back();
    uint32_t data_size = 1;
    if(in_bytes) {
        data_size = num_df_bytes;
    }
    uint32_t num_groups = transformation_node->groups.size();
    vector<vector<uint32_t>>address_maps;
    for(uint32_t g = 0; g < num_groups; g++) {
        assert(transformation_node->groups[g]->transfers.size() > 0);
        // copy transfer addresses into a vector
        std::vector<uint32_t> address_map_this_group;
        // Generate address map
        for(auto transfer : transformation_node->groups[g]->transfers){
            address_map_this_group.push_back(transfer->src_address * data_size);
            address_map_this_group.push_back(transfer->dst_address * data_size);
            address_map_this_group.push_back(transfer->size * data_size);
            address_map_this_group.push_back(transfer->pad);
        }
        address_maps.push_back(address_map_this_group);
    }
    // combine address maps for all groups into one buffer with size information in the beginning of each group
    vector<uint32_t> address_map_full;
    address_map_full.push_back(num_groups);
    for(uint32_t g = 0; g < num_groups; g++) {
        address_map_full.push_back(address_maps[g].size());
        address_map_full.insert(address_map_full.end(), address_maps[g].begin(), address_maps[g].end());
    }
    return address_map_full;
}
vector<vector<float>> evaluate(vector<float> data, std::vector<uint32_t> address_map, vector<vector<int>> output_shape) {
    uint32_t address_map_index = 0;
    uint32_t num_groups = address_map[address_map_index];
    log_debug(tt::LogDTX, "num_groups = {}", num_groups);
    assert(output_shape.size() == num_groups);
    address_map_index += 1;
    vector<vector<float>> data_transformed_groups;
    for(uint32_t g = 0; g < num_groups; g++) {
        auto data_transformed_size = vector_product(output_shape[g]);
        vector<float> data_transformed(data_transformed_size, 0);
        uint32_t address_map_group_size = address_map[address_map_index];
        address_map_index += 1;
        for(uint32_t i = 0; i < address_map_group_size; i+=4) {
            auto src_address = address_map[address_map_index];
            auto dst_address = address_map[address_map_index+1];
            auto transfer_size = address_map[address_map_index+2];
            auto pad = address_map[address_map_index+3];
            address_map_index += 4;
            for (uint32_t s = 0; s < transfer_size; s++) {
                assert(dst_address+s < data_transformed.size());
                if (pad) {
                    data_transformed[dst_address+s] = 0;
                }
                else {
                    assert(src_address+s < data.size());
                    data_transformed[dst_address+s] = data[src_address+s];
                }
            }
        }
        data_transformed_groups.push_back(data_transformed);
    }
    return data_transformed_groups;
}
