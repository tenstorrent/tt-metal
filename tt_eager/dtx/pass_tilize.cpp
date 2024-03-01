// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dtx.hpp"
#include "util.hpp"
#include "util_vector_of_ints.hpp"

using namespace std;

vector<vector<int>> dim_order_counting(vector<int> shape, vector<int> dim_order) {
    bool DEBUG = false;
    tt::log_info(tt::LogDTX, "Helper function: dim order counting");

    int rank = shape.size();
    vector<int> shape_reordered;
    for (int d=0; d<rank; d++) {
        shape_reordered.push_back(shape[dim_order[d]]);
    }
    int count_size = vector_product(shape);
    vector<int> tile_shape = vector_pad_on_left(TILE_SHAPE, rank-2, 1);

    /*
    if (DEBUG) {
        cout << "shape           = " << v2s(shape) << endl;
        cout << "dim order       = " << v2s(dim_order) << endl;
        cout << "shape reordered = " << v2s(shape_reordered) << endl;
        cout << "count size      = " << count_size << endl;
        cout << "tile shape      = " << v2s(tile_shape) << endl;
    }*/

    vector<vector<int>> list_of_counted_dims;
    vector<int> counter = zeros(rank);
    for (int i=0; i<count_size; i++){
        vector<int> counter_reordered;

        for (int d=0; d<rank; d++) {
            counter_reordered.push_back(counter[dim_order[d]]);
        }

        vector<int> str = vector_multiplication(counter_reordered, tile_shape);
        vector<int> end = vector_addition(str, tile_shape);
        end[rank-1]--;
        end[rank-2]--;

        tt::log_debug(tt::LogDTX, "   counter = {} , reordered = {};   {} => {}", v2s(counter), v2s(counter_reordered), v2s(str), v2s(end));
        list_of_counted_dims.push_back(counter_reordered);

        counter.back()++;
        for (int d=rank-1; d>0; d--) {
            if (counter[d] == shape_reordered[d]) {
                counter[d-1]++;
                counter[d] = 0;
            }
        }
    }

    return list_of_counted_dims;
}


bool tilize_and_store(DataTransformations * dtx, vector<int> dim_order) {
    bool DEBUG = true;

    if (DEBUG) tt::log_info(tt::LogDTX, "PASS: Tilize and Store");

    // Identify producer TX & Consumer
    TransformationNode * producer = dtx->transformations.back();
    TransformationNode * consumer = new TransformationNode("tilize_and_store", producer->groups.size());  // TODO: generalize for groups>1
    dtx->transformations.push_back(consumer);

    for (int group_idx=0; group_idx<producer->groups.size(); group_idx++) {
        tt::log_debug(tt::LogDTX, "Group = {}", group_idx);

        TensorPairGroup * consumer_group = consumer->groups[group_idx];
        TensorPairGroup * producer_group = producer->groups[group_idx];
        inherit_group_attributes_from_producer(producer_group, consumer_group);

        vector<int> shape = producer_group->shape;
        int rank = producer_group->shape.size();
        vector<int> tile_shape = vector_pad_on_left(TILE_SHAPE, rank-2, 1);
        vector<int> shape_tiled = vector_division(shape, tile_shape);
        vector<vector<int>> list_of_counted_dims = dim_order_counting(shape_tiled,   dim_order);

        vector<int> consumer_str = zeros(rank);
        vector<int> consumer_end = vector_addition(consumer_str, tile_shape, -1);
        tt::log_debug(tt::LogDTX, "    tile_shape = {}", v2s(tile_shape));

        if (shape.size() != dim_order.size()) throw std::runtime_error("shape and dim_order dont have the same rank!");

        int shape_x = list_of_counted_dims.size() * 32;
        consumer_group->shape = {32, shape_x};

        tt::log_debug("    Tensor Pairs: {}", list_of_counted_dims.size());
        for (int i=0; i< list_of_counted_dims.size(); i++) {
            for(int j = 0; j < list_of_counted_dims[i].size(); j++) {
                tt::log_debug(tt::LogDTX, "dim {}", list_of_counted_dims[i][j]);
            }
            // Source Tensor: within the ND tensor from producer
            vector<int> str;
            vector<int> end;
            str = vector_multiplication(list_of_counted_dims[i], tile_shape);
            end = vector_addition(str, tile_shape, -1);

            TensorPair * tp = new TensorPair(new DTXTensor({str}, {end}),
                                            group_idx,
                                            new DTXTensor({consumer_str}, {consumer_end}));
            consumer_group->tensor_pairs.push_back(tp);

            tt::log_debug(tt::LogDTX, "      {}. {}", i, tp->get_string());

            // Prepare for the next itteration
            consumer_str.back() = (i+1) * 32;
            consumer_end = vector_addition(consumer_str, tile_shape, -1);
        }
    }
    return true;
}
