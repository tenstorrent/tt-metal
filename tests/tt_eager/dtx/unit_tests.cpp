// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <assert.h>
#include <stdio.h>
#include <math.h>

#include <iterator>
#include <iostream>
#include <fstream>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <queue>

#include "dtx/dtx.hpp"
#include "dtx/util_vector_of_ints.hpp"
#include "dtx/util.hpp"
#include "dtx/dtx_passes.hpp"

using namespace std;

bool test_GenerateAddresses() {
    bool pass = true;
    bool DEBUG = true;

    TransformationNode * node0 = new TransformationNode("producer", 1);
    TransformationNode * node1 = new TransformationNode("consumer", 1);

    // Producer node: 1 tensor.
    node0->groups[0]->shape = {40};
    node0->groups[0]->address = 0;
    node0->groups[0]->core = {2,3};

    // Transformation #1: vertical blocks
    node1->groups[0]->shape = {40};
    node1->groups[0]->address = 0;
    node1->groups[0]->core = {5,6};
    node1->groups[0]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({0},  {20}), 0,  new DTXTensor({20}, {40}))  );
    node1->groups[0]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({20}, {40}), 0,  new DTXTensor({0}, {20}))   );

    DataTransformations * dtx = new DataTransformations();
    dtx->transformations.push_back(node0);
    dtx->transformations.push_back(node1);

    if (DEBUG) dtx->print();

    pass = generate_transfer_addresses(dtx);

    return pass;
}

bool run_DTX_reverse_transformations_test_0(int DEBUG) {

    TransformationNode * node0 = new TransformationNode("producer", 2);
    TransformationNode * node1 = new TransformationNode("tx1", 2);

    // NODE 0:
    node0->groups[0]->shape = {40};
    node0->groups[1]->shape = {40};

    // NODE 1:
    node1->groups[0]->shape = {40};
    node1->groups[0]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({0},  {20}), 1, new DTXTensor({20}, {40}))  );
    node1->groups[0]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({20}, {40}), 1, new DTXTensor({0}, {20}))   );

    node1->groups[1]->shape = {40};
    node1->groups[1]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({0},  {20}), 0, new DTXTensor({20}, {40}))  );
    node1->groups[1]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({20}, {40}), 0,  new DTXTensor({0}, {20}))   );

    DataTransformations * dtx = new DataTransformations();
    dtx->transformations.push_back(node0);
    dtx->transformations.push_back(node1);

    DataTransformations * backwards = reverse_transformations(dtx);

    /*
    // Correctness checking
    TransformationNode * golden = new TransformationNode("golden", 1);
    golden->groups[0]->shape = {40,40};
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new DTXTensor( {20}, {30}), 1, new DTXTensor({10}, {20})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new DTXTensor( {0 }, {10}), 1, new DTXTensor({30}, {40})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new DTXTensor( {30}, {40}), 1, new DTXTensor({20}, {30})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new DTXTensor( {10}, {20}), 1, new DTXTensor({0 }, {10})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new DTXTensor( {20}, {30}), 0, new DTXTensor({10}, {20})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new DTXTensor( {0 }, {10}), 0, new DTXTensor({30}, {40})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new DTXTensor( {30}, {40}), 0, new DTXTensor({20}, {30})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new DTXTensor( {10}, {20}), 0, new DTXTensor({0 }, {10})));

    if (DEBUG) golden->print(0);

    pass = compare_two_groups(golden->groups[0], node2->groups[0]);
    if (DEBUG) cout << "PASS = " << pass << endl;
    */

   return true;
}

bool run_DTX_reverse_transformations_test_1(int DEBUG) {

    TransformationNode * node0 = new TransformationNode("producer", 2);
    TransformationNode * node1 = new TransformationNode("tx1", 3);


    // NODE 0:
    node0->groups[0]->shape = {100};
    node0->groups[1]->shape = {200};

    // NODE 1:
    node1->groups[0]->shape = {20};
    node1->groups[0]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({60}, {70}),  0, new DTXTensor({0},  {10}))  );
    node1->groups[0]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({70}, {80}),  0, new DTXTensor({10}, {20}))   );

    // NODE 1:
    node1->groups[1]->shape = {30};
    node1->groups[1]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({130}, {140}), 1, new DTXTensor({0},  {10}))  );
    node1->groups[1]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({140}, {150}), 1, new DTXTensor({10}, {20}))  );

    node1->groups[2]->shape = {40};
    node1->groups[2]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({80},  {90}),  0, new DTXTensor({0},  {10}))  );
    node1->groups[2]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({160}, {170}), 1, new DTXTensor({10}, {20}))  );

    DataTransformations * dtx = new DataTransformations();
    dtx->transformations.push_back(node0);
    dtx->transformations.push_back(node1);

    DataTransformations * backwards = reverse_transformations(dtx);


    /*
    // Correctness checking
    TransformationNode * golden = new TransformationNode("golden", 1);
    golden->groups[0]->shape = {40,40};
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new DTXTensor( {20}, {30}), 1, new DTXTensor({10}, {20})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new DTXTensor( {0 }, {10}), 1, new DTXTensor({30}, {40})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new DTXTensor( {30}, {40}), 1, new DTXTensor({20}, {30})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new DTXTensor( {10}, {20}), 1, new DTXTensor({0 }, {10})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new DTXTensor( {20}, {30}), 0, new DTXTensor({10}, {20})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new DTXTensor( {0 }, {10}), 0, new DTXTensor({30}, {40})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new DTXTensor( {30}, {40}), 0, new DTXTensor({20}, {30})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new DTXTensor( {10}, {20}), 0, new DTXTensor({0 }, {10})));

    if (DEBUG) golden->print(0);

    pass = compare_two_groups(golden->groups[0], node2->groups[0]);
    if (DEBUG) cout << "PASS = " << pass << endl;
    */

   return true;
}

bool run_DTX_reverse_transformations_test_2(int DEBUG) {
    bool pass = true;

    TransformationNode * node0 = new TransformationNode("producer", 1);
    TransformationNode * node1 = new TransformationNode("tx1", 1);
    TransformationNode * node2 = new TransformationNode("tx2", 1);

    // NODE 0:
    node0->groups[0]->shape = {40};

    // NODE 1:
    node1->groups[0]->shape = {40};
    node1->groups[0]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({10}, {20}),  0, new DTXTensor({0},  {10}))  );
    node1->groups[0]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({0},  {10}),  0, new DTXTensor({10}, {20}))   );
    node1->groups[0]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({30}, {40}),  0, new DTXTensor({20}, {30}))   );
    node1->groups[0]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({20}, {30}),  0, new DTXTensor({30}, {40}))   );

    // NODE 2:
    node2->groups[0]->shape = {40};
    node2->groups[0]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({20}, {30}),  0, new DTXTensor({0},  {10}))  );
    node2->groups[0]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({30}, {40}),  0, new DTXTensor({10}, {20}))   );
    node2->groups[0]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({0},  {10}),  0, new DTXTensor({20}, {30}))   );
    node2->groups[0]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({10}, {20}),  0, new DTXTensor({30}, {40}))   );

    DataTransformations * dtx = new DataTransformations();
    dtx->transformations.push_back(node0);
    dtx->transformations.push_back(node1);
    dtx->transformations.push_back(node2);

    dtx->print();
    pass &= collapse_transformations(dtx);
    dtx->print();

    DataTransformations * backwards = reverse_transformations(dtx);

    //backwards->print();
    pass &= collapse_transformations(backwards);
    backwards->print();
    pass &= generate_transfer_addresses(backwards);

    /*
    // Correctness checking
    TransformationNode * golden = new TransformationNode("golden", 1);
    golden->groups[0]->shape = {40,40};
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new DTXTensor( {20}, {30}), 1, new DTXTensor({10}, {20})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new DTXTensor( {0 }, {10}), 1, new DTXTensor({30}, {40})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new DTXTensor( {30}, {40}), 1, new DTXTensor({20}, {30})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new DTXTensor( {10}, {20}), 1, new DTXTensor({0 }, {10})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new DTXTensor( {20}, {30}), 0, new DTXTensor({10}, {20})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new DTXTensor( {0 }, {10}), 0, new DTXTensor({30}, {40})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new DTXTensor( {30}, {40}), 0, new DTXTensor({20}, {30})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new DTXTensor( {10}, {20}), 0, new DTXTensor({0 }, {10})));

    if (DEBUG) golden->print(0);

    pass = compare_two_groups(golden->groups[0], node2->groups[0]);
    if (DEBUG) cout << "PASS = " << pass << endl;
    */

   return true;
}

bool run_DTX_reverse_transformations_test_3(int DEBUG) {
    bool pass = true;

    DataTransformations * dtx = new DataTransformations();

    // Create producer node
    TransformationNode * node0 = new TransformationNode("producer", 1);
    node0->groups[0]->shape = {2,3,4};
    dtx->transformations.push_back(node0);

    pass &= transpose_yz(dtx);
    pass &= transpose_xy(dtx);
    dtx->print();

    DataTransformations * backwards = reverse_transformations(dtx);
    backwards->print();
    return pass;
}

bool test_DTX_reverse_transformations() {
    bool DEBUG = true;
    bool pass = true;

    pass &= run_DTX_reverse_transformations_test_0(DEBUG);
    pass &= run_DTX_reverse_transformations_test_1(DEBUG);
    pass &= run_DTX_reverse_transformations_test_2(DEBUG);
    pass &= run_DTX_reverse_transformations_test_3(DEBUG);
    return pass;
}

bool test_generate_sliced_ranges_helper_functions() {
    bool pass = true;

    // Part 1 - Test Generated sliced ranges
    vector<vector<vector<int>>> ranges = generate_sliced_ranges({10,10}, {2,2});

    // Part 2 - Test Generate list of cores based on range
    vector<vector<int>> list_of_cores;
    list_of_cores = generate_list_of_cores_based_on_range({0,0}, {0,0});
    list_of_cores = generate_list_of_cores_based_on_range({0,0}, {0,1});
    list_of_cores = generate_list_of_cores_based_on_range({0,0}, {3,3});
    list_of_cores = generate_list_of_cores_based_on_range({2,2}, {5,5});
    list_of_cores = generate_list_of_cores_based_on_range({3,2}, {3,5});
    list_of_cores = generate_list_of_cores_based_on_range({5,3}, {9,3});

    return pass;
}

bool test_pass_parallelize_generic_tensor_slice() {
    bool pass = true;

    // Part 3 - Test generic parallelization pass
    DataTransformations * dtx = new DataTransformations();
    TransformationNode * node0 = new TransformationNode("producer", 1);
    node0->groups[0]->shape = {20,20};
    dtx->transformations.push_back(node0);

    vector<int> slice_factors = {2,2};
    vector<int> cores_start   = {0,0};
    vector<int> cores_end     = {1,1};
    pass &= parallelize_generic_tensor_slice(dtx, slice_factors, cores_start, cores_end);

    return pass;
}


bool test_dim_order_counting_helper_function() {
    /*
    dim_order_counting({2,3},   {0,1});   // row major
    dim_order_counting({2,3},   {1,0});   // col major
    dim_order_counting({2,3,4}, {0,1,2}); // row major
    dim_order_counting({2,3,4}, {0,2,1}); // col major
    //dim_order_counting({2,3,4}, {1,2,0}); // sticks major, then row major after
    //dim_order_counting({2,3,4}, {2,1,0}); // sticks major, then col major after
    dim_order_counting({2,3,4,5}, {0,1,2,3}); // sticks major, then col major after
     */
    return true;
}

bool run_pass_tilize_and_store_test(vector<int> shape, vector<int> dim_order) {
    DataTransformations * dtx = new DataTransformations();
    TransformationNode * node0 = new TransformationNode("producer", 1);
    node0->groups[0]->shape = shape;
    dtx->transformations.push_back(node0);
    bool pass = tilize_and_store(dtx, dim_order);
    return pass;
}

bool run_pass_tilize_and_store_test_0() {
    DataTransformations * dtx = new DataTransformations();
    TransformationNode * node0 = new TransformationNode("producer", 2);
    node0->groups[0]->shape = {64,64};
    node0->groups[1]->shape = {128,128  };
    node0->groups[0]->core = {1,2};
    node0->groups[1]->core = {2,3};

    dtx->transformations.push_back(node0);
    bool pass = tilize_and_store(dtx, {0,1});
    dtx->print();
    return pass;
}


bool test_pass_tilize_and_store() {
    bool pass = true;

    // PART 1: The main helper function
    pass &= test_dim_order_counting_helper_function();

    // Test simple vector helper method
    /*
    cout << "dut: " << v2s(vector_pad_on_left({32,32}, 2, 0)) << endl;
    cout << "dut: " << v2s(vector_pad_on_left({32,32}, 2, 1)) << endl;
    cout << "dut: " << v2s(vector_pad_on_left({32,32}, 1, 0)) << endl;
    */

    // Test pass (producer groups == 1)
    //pass &= run_pass_tilize_and_store_test({64,64}, {1,0});
    //pass &= run_pass_tilize_and_store_test({128,128}, {1,0});
    //pass &= run_pass_tilize_and_store_test({2, 64, 64}, {2,1,0});

    // Test pass (producer groups > 1)
    pass &= run_pass_tilize_and_store_test_0();


    return pass;
}


bool test_transpose_xy() {
    TensorData * t = new TensorData({32,32});
    t->print();
    t->generate_csv("tensor1");

    return true;
}

bool test_tensor_evaluate() {
    TransformationNode * node0 = new TransformationNode("producer", 1);
    TransformationNode * node1 = new TransformationNode("tx1", 1);

    // NODE 0:
    node0->groups[0]->shape = {40};

    // NODE 1:
    node1->groups[0]->shape = {40};
    node1->groups[0]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({0},  {20}), 1, new DTXTensor({20}, {40}))  );
    node1->groups[0]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({20}, {40}), 1, new DTXTensor({0}, {20}))   );

    DataTransformations * dtx = new DataTransformations();
    dtx->transformations.push_back(node0);
    dtx->transformations.push_back(node1);

    TensorData * t = new TensorData({40});
    t->print();
    t->generate_csv("tensor1");

    //TensorData * t_out = dtx->evaluate(t_in);

    return true;
}

bool test_pass_transpose_xy() {
    bool pass = true;
    DataTransformations * dtx = new DataTransformations();
    TransformationNode * node0 = new TransformationNode("producer", 1);
    node0->groups[0]->shape = {40, 50};
    dtx->transformations.push_back(node0);
    pass &= transpose_xy(dtx);
    dtx->print();
    return pass;
}

bool run_transpose_yz(vector<int> shape) {
    bool pass = true;
    DataTransformations * dtx = new DataTransformations();
    TransformationNode * node0 = new TransformationNode("producer", 1);
    node0->groups[0]->shape = shape;
    dtx->transformations.push_back(node0);
    pass &= transpose_yz(dtx);
    dtx->print();
    return pass;
}

bool test_pass_transpose_yz() {
    bool pass = true;

    //pass &= run_transpose_yz({2,3,4});
    pass &= run_transpose_yz({90, 2, 2, 100});
    pass &= run_transpose_yz({20, 21, 22, 2, 2, 100});
    return pass;

}

bool test_pass_convert_tensor_layout_3d_conv_act_to_2Dmatrix() {
    bool pass = true;
    DataTransformations * dtx = new DataTransformations();
    TransformationNode * node0 = new TransformationNode("producer", 1);
    node0->groups[0]->shape = {10, 4, 4};
    dtx->transformations.push_back(node0);
    pass &= convert_tensor_layout_3d_conv_act_to_2Dmatrix(dtx, {3,3,1,1,0,0});
    dtx->print();
    return pass;
}

bool test_convert_abstract_tensor_to_channels_last_layout() {
    bool pass = true;
    DataTransformations * dtx = new DataTransformations();
    TransformationNode * node0 = new TransformationNode("producer", 1);
    node0->groups[0]->shape = {10, 4, 4};
    dtx->transformations.push_back(node0);
    pass &= convert_abstract_tensor_to_channels_last_layout(dtx);
    dtx->print();
    return pass;
}

bool test_channels_last_to_2D_matrix() {
    bool pass = true;

    /*
    // collapse transformation debug
    vector<int> shape = {5, 4,4};

    // Right side: AbstractTensor --> consumer conv/mm
    DataTransformations * dtx_right = new DataTransformations();
    TransformationNode * node0 = new TransformationNode("producer", 1);
    node0->groups[0]->shape = shape;
    dtx_right->transformations.push_back(node0);
    pass &= convert_tensor_layout_3d_conv_act_to_2Dmatrix(dtx_right);
    pass &= row_major_memory_store(dtx_right);
    pass &= collapse_transformations(dtx_right);
    */



    // FULL TEST

    vector<int> shape = {5, 4,4};

    // Right side: AbstractTensor --> consumer conv/mm
    DataTransformations * dtx_right = new DataTransformations();
    TransformationNode * node0 = new TransformationNode("producer", 1);
    node0->groups[0]->shape = shape;
    dtx_right->transformations.push_back(node0);
    pass &= convert_tensor_layout_3d_conv_act_to_2Dmatrix(dtx_right, {1,1,1,1,0,0});
    pass &= row_major_memory_store(dtx_right);

    tt::log_debug(tt::LogDTX, "DTX_RIGHT");
    dtx_right->print();


    // Left side: AbstractTensor --> producer memory buffer
    DataTransformations * dtx_left = new DataTransformations();
    TransformationNode * node1 = new TransformationNode("producer", 1);
    node1->groups[0]->shape = shape;
    dtx_left->transformations.push_back(node1);
    pass &= convert_abstract_tensor_to_channels_last_layout(dtx_left);

    log_debug(tt::LogDTX, "DTX_LEFT");
    dtx_left->print();

    DataTransformations * combined = reverse_and_combine_transformations(dtx_left, dtx_right);
    log_debug(tt::LogDTX, "DTX_COMBINED");
    combined->print();

    pass &= optimize_away_transpose(combined);
    log_debug(tt::LogDTX, "DTX_OPTIMIZED");
    combined->print();

    pass &= collapse_transformations(combined);
    log_debug(tt::LogDTX, "DTX_COLLAPSED");
    combined->print();
    pass &= generate_transfer_addresses(combined);
    combined->print();



    return pass;
}

bool test_channels_last_to_2D_matrix_conv1x1() {
    bool pass = true;


    // FULL TEST

    vector<int> shape = {5, 4,4};

    // Right side: AbstractTensor --> consumer conv/mm
    DataTransformations * dtx_right = new DataTransformations();
    TransformationNode * node0 = new TransformationNode("producer", 1);
    node0->groups[0]->shape = shape;
    dtx_right->transformations.push_back(node0);
    pass &= convert_tensor_layout_3d_conv_act_to_2Dmatrix(dtx_right, {1,1,1,1,0,0});
    pass &= row_major_memory_store(dtx_right);

    tt::log_debug(tt::LogDTX, "DTX_RIGHT");
    dtx_right->print();


    // Left side: AbstractTensor --> producer memory buffer
    DataTransformations * dtx_left = new DataTransformations();
    TransformationNode * node1 = new TransformationNode("producer", 1);
    node1->groups[0]->shape = shape;
    dtx_left->transformations.push_back(node1);
    pass &= convert_abstract_tensor_to_channels_last_layout(dtx_left);

    tt::log_debug(tt::LogDTX, "DTX_LEFT");
    dtx_left->print();

    DataTransformations * combined = reverse_and_combine_transformations(dtx_left, dtx_right);
    tt:log_debug(tt::LogDTX, "DTX_COMBINED");
    combined->print();

    pass &= optimize_away_transpose(combined);
    tt::log_debug(tt::LogDTX, "DTX_OPTIMIZED");
    combined->print();

    pass &= collapse_transformations(combined);
    tt::log_debug(tt::LogDTX, "DTX_COLLAPSED");
    combined->print();
    pass &= generate_transfer_addresses(combined);
    combined->print();



    return pass;
}

bool test_run_conv_transform_no_evaluate() {
    vector<int> act_shape = {32, 5, 5};
    vector<int> weight_shape = {32, 32, 1, 1};
    vector<int> conv_params = {1,1,1,1,0,0};
    uint32_t num_blocks_act_h = 1;
    uint32_t num_blocks_weight_w = 1;
    uint32_t act_block_h = 32;
    uint32_t act_block_w = 32;
    uint32_t weight_block_w = 32;
    auto address_map = conv_transform(act_shape, weight_shape, conv_params, act_block_h, act_block_w, weight_block_w, num_blocks_act_h, num_blocks_weight_w, 1, false);
    return true;
}

bool test_high_level_pass_and_evaluate() {
    vector<int> shape = {2, 2, 2};
    auto dtx = simple_high_level_pass(shape);
    vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8};
    vector<vector<float>> data_transformed = evaluate(data, generate_address_map(dtx), {shape});
    vector<vector<float>> golden_data = {{1, 2, 5, 6, 3, 4, 7, 8}};
    return data_transformed == golden_data;
}

bool test_padding_pass_(vector<int> shape, vector<int> pad_to_nearest, vector<float> input_data, vector<vector<float>> golden_data) {
    DataTransformations * dtx_left = new DataTransformations();
    TransformationNode * node1 = new TransformationNode("producer", 1);
    node1->groups[0]->shape = shape;
    dtx_left->transformations.push_back(node1);
    bool pass = row_major_memory_store(dtx_left);
    //dtx_left->print();
    DataTransformations * dtx_right = new DataTransformations();
    TransformationNode * node0 = new TransformationNode("producer", 1);
    node0->groups[0]->shape = shape;
    dtx_right->transformations.push_back(node0);
    pass &= pad_2d_matrix(dtx_right, pad_to_nearest);
    auto padded_shape = dtx_right->transformations.back()->groups[0]->shape;
    //dtx_right->print();
    //exit(1);
    pass &= row_major_memory_store(dtx_right);
    //dtx_right->print();
    //exit(1);
    //pass &= collapse_transformations(dtx_right);
    //dtx_right->print();
    //exit(1);
    DataTransformations * combined = reverse_and_combine_transformations(dtx_left, dtx_right);
    //cout << "\n\nDTX_COMBINED" << endl;
    //combined->print();
    pass &= collapse_transformations(combined);
    //cout << "\n\nDTX_COLLAPSED" << endl;
    //combined->print();
    pass &= generate_transfer_addresses(combined);

    vector<vector<float>> data_transformed = evaluate(input_data, generate_address_map(combined), {padded_shape});
    return data_transformed == golden_data;
}

bool test_padding_pass() {
    bool pass = true;
    vector<int> shape = {1, 2, 2};
    vector<float> input_data_2_2 = {1, 2, 3, 4};
    // list of tests - pad to nearest, golden data
    vector<tuple<vector<int>, vector<vector<float>>>> tests_2_2 = {
        { {4,4}, {{1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}} },
        { {3,4}, {{1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0}} },
        { {4,3}, {{1, 2, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0}} },
    };
    for (auto & t : tests_2_2) {
        auto pad_to_nearest = std::get<0>(t);
        auto golden_data = std::get<1>(t);
        pass &= test_padding_pass_(shape, pad_to_nearest, input_data_2_2, golden_data);
        if(pass) {
            tt::log_debug(tt::LogDTX, "Passed test with shape = {} , pad to nearest = {}", v2s(shape), v2s(pad_to_nearest));
        }
        else {
            tt::log_error(tt::LogDTX, "Failed test with shape = {} , pad to nearest = {}", v2s(shape), v2s(pad_to_nearest));
        }
        if(!pass) exit(1);
    }
    return pass;
}

bool test_block_2d_matrix_pass_(vector<int> shape, vector<int> block_shape, vector<int> dim_order,
                            vector<float> input_data, vector<vector<float>> golden_data) {
    DataTransformations * dtx_left = new DataTransformations();
    TransformationNode * node1 = new TransformationNode("producer", 1);
    node1->groups[0]->shape = shape;
    dtx_left->transformations.push_back(node1);
    bool pass = row_major_memory_store(dtx_left);
    //dtx_left->print();
    DataTransformations * dtx_right = new DataTransformations();
    TransformationNode * node0 = new TransformationNode("producer", 1);
    node0->groups[0]->shape = shape;
    dtx_right->transformations.push_back(node0);
    pass &= block_2d_matrix(dtx_right, dim_order, block_shape);
    pass &= row_major_memory_store(dtx_right);
    //dtx_right->print();
    DataTransformations * combined = reverse_and_combine_transformations(dtx_left, dtx_right);
    //cout << "\n\nDTX_COMBINED" << endl;
    //combined->print();
    pass &= collapse_transformations(combined);
    //cout << "\n\nDTX_COLLAPSED" << endl;
    //combined->print();
    pass &= generate_transfer_addresses(combined);
    //combined->print();
    vector<vector<float>> data_transformed = evaluate(input_data, generate_address_map(combined), {shape});
    return data_transformed == golden_data;
}

bool test_block_2d_matrix_pass() {
    bool pass = true;
    vector<int> shape = {1, 4, 4};
    vector<float> input_data_4_4 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    // list of tests - block shape, dim order, golden data
    vector<tuple<vector<int>, vector<int>, vector<vector<float>>>> tests_4_4 = {
        { {4,4}, {0,1,2}, {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}} },
        { {2,4}, {0,1,2}, {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}} },
        { {1,4}, {0,1,2}, {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}} },
        { {4,1}, {0,1,2}, {{1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16}} },
        { {4,2}, {0,1,2}, {{1, 2, 5, 6, 9, 10, 13, 14, 3, 4, 7, 8, 11, 12, 15, 16}} },
        { {2,2}, {0,1,2}, {{1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 13, 14, 11, 12, 15, 16}} },
        { {2,2}, {0,2,1}, {{1, 2, 5, 6, 9, 10, 13, 14, 3, 4, 7, 8, 11, 12, 15, 16}} },

    };
    for (auto & t : tests_4_4) {
        auto block_shape = std::get<0>(t);
        auto dim_order = std::get<1>(t);
        auto golden_data = std::get<2>(t);
        pass &= test_block_2d_matrix_pass_(shape, block_shape, dim_order, input_data_4_4, golden_data);
        if(pass) {
            tt::log_debug(tt::LogDTX, "Passed test with shape = {} , block shape = {} , dim_order = {}", v2s(shape), v2s(block_shape), v2s(dim_order));
        }
        else {
            tt::log_error(tt::LogDTX, "Failed test with shape = {} , block shape = {} , dim_order = {}", v2s(shape), v2s(block_shape), v2s(dim_order));
        }
    }
    // Testing shape with x != y
    shape = {1, 2, 6};
    vector<float> input_data_2_6 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    // list of tests - block shape, dim order, golden data
    vector<tuple<vector<int>, vector<int>, vector<vector<float>>>> tests_2_6 = {
        { {2,6}, {0,1,2}, {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}} },
        { {1,6}, {0,1,2}, {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}} },
        { {2,2}, {0,1,2}, {{1, 2, 7, 8, 3, 4, 9, 10, 5, 6, 11, 12}} },
        { {2,3}, {0,1,2}, {{1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12}} },
        { {1,3}, {0,1,2}, {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}} },
        { {1,3}, {0,2,1}, {{1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12}} },

    };
    for (auto & t : tests_2_6) {
        auto block_shape = std::get<0>(t);
        auto dim_order = std::get<1>(t);
        auto golden_data = std::get<2>(t);
        pass &= test_block_2d_matrix_pass_(shape, block_shape, dim_order, input_data_2_6, golden_data);
        if(pass) {
            tt::log_debug(tt::LogDTX, "Passed test with shape = {} , block shape = {} , dim order = {}", v2s(shape), v2s(block_shape), v2s(dim_order));
        }
        else {
            tt::log_error(tt::LogDTX, "Failed test with shape = {} , block shape = {} , dim order = {}", v2s(shape), v2s(block_shape), v2s(dim_order));
        }
    }
    return pass;

}

bool test_block_2d_matrix_group_pass_(vector<int> shape, vector<int> block_shape, vector<int> dim_order,
                            vector<float> input_data, vector<vector<float>> golden_data) {
    DataTransformations * dtx_left = new DataTransformations();
    TransformationNode * node1 = new TransformationNode("producer", 1);
    node1->groups[0]->shape = shape;
    dtx_left->transformations.push_back(node1);
    bool pass = row_major_memory_store(dtx_left);
    //dtx_left->print();
    DataTransformations * dtx_right = new DataTransformations();
    TransformationNode * node0 = new TransformationNode("producer", 1);
    node0->groups[0]->shape = shape;
    dtx_right->transformations.push_back(node0);
    pass &= block_2d_matrix(dtx_right, dim_order, block_shape);
    pass &= generate_groups_outermost_dim(dtx_right);
    //dtx_right->print();
    pass &= row_major_memory_store(dtx_right);
    //dtx_right->print();
    DataTransformations * combined = reverse_and_combine_transformations(dtx_left, dtx_right);
    //cout << "\n\nDTX_COMBINED" << endl;
    //combined->print();
    pass &= collapse_transformations(combined);
    //cout << "\n\nDTX_COLLAPSED" << endl;
    //combined->print();
    pass &= generate_transfer_addresses(combined);
    //combined->print();
    vector<vector<int>> output_shape_groups;
    uint32_t num_groups = combined->transformations.back()->groups.size();
    assert(golden_data.size() == num_groups);
    for(uint32_t g = 0; g < num_groups; g++) {
        output_shape_groups.push_back(combined->transformations.back()->groups[g]->shape);
    }
    vector<vector<float>> data_transformed = evaluate(input_data, generate_address_map(combined), output_shape_groups);
    assert(data_transformed.size() == golden_data.size());
    return data_transformed == golden_data;
}

bool test_block_2d_matrix_group_pass() {
    bool pass = true;
    vector<int> shape = {1, 4, 4};
    vector<float> input_data_4_4 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    // list of tests - block shape, dim order, golden data
    vector<tuple<vector<int>, vector<int>, vector<vector<float>>>> tests_4_4 = {
        { {4,4}, {0,1,2}, {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}} },
        { {2,4}, {0,1,2}, {{1, 2, 3, 4, 5, 6, 7, 8}, {9, 10, 11, 12, 13, 14, 15, 16}} },
        { {1,4}, {0,1,2}, {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}} },
        { {4,1}, {0,1,2}, {{1, 5, 9, 13}, {2, 6, 10, 14}, {3, 7, 11, 15}, {4, 8, 12, 16}} },
        { {4,2}, {0,1,2}, {{1, 2, 5, 6, 9, 10, 13, 14}, {3, 4, 7, 8, 11, 12, 15, 16}} },
        { {2,2}, {0,1,2}, {{1, 2, 5, 6}, {3, 4, 7, 8}, {9, 10, 13, 14}, {11, 12, 15, 16}} },
        { {2,2}, {0,2,1}, {{1, 2, 5, 6}, {9, 10, 13, 14}, {3, 4, 7, 8}, {11, 12, 15, 16}} },

    };
    for (auto & t : tests_4_4) {
        auto block_shape = std::get<0>(t);
        auto dim_order = std::get<1>(t);
        auto golden_data = std::get<2>(t);
        pass &= test_block_2d_matrix_group_pass_(shape, block_shape, dim_order, input_data_4_4, golden_data);
        if(pass) {
            tt::log_debug(tt::LogDTX, "Passed test with shape = {} , block shape = {} , dim order = {}", v2s(shape), v2s(block_shape), v2s(dim_order));
        }
        else {
            tt::log_error(tt::LogDTX, "Failed test with shape = {} , block shape = {} , dim order = {}", v2s(shape), v2s(block_shape), v2s(dim_order));
        }
    }
    return pass;

}

bool test_pad_and_block_passes_(vector<int> shape, vector<int> pad_to_nearest, vector<int> block_shape, vector<int> dim_order,
                            vector<float> input_data, vector<vector<float>> golden_data) {
    DataTransformations * dtx_left = new DataTransformations();
    TransformationNode * node1 = new TransformationNode("producer", 1);
    node1->groups[0]->shape = shape;
    dtx_left->transformations.push_back(node1);
    bool pass = row_major_memory_store(dtx_left);
    //dtx_left->print();
    DataTransformations * dtx_right = new DataTransformations();
    TransformationNode * node0 = new TransformationNode("producer", 1);
    node0->groups[0]->shape = shape;
    dtx_right->transformations.push_back(node0);
    pass &= pad_2d_matrix(dtx_right, pad_to_nearest);
    dtx_right->print();
    pass &= block_2d_matrix(dtx_right, dim_order, block_shape);
    dtx_right->print();

    pass &= row_major_memory_store(dtx_right);
    dtx_right->print();
    //exit(1);
    DataTransformations * combined = reverse_and_combine_transformations(dtx_left, dtx_right);
    //cout << "\n\nDTX_COMBINED" << endl;
    //combined->print();
    pass &= collapse_transformations(combined);
    //cout << "\n\nDTX_COLLAPSED" << endl;
    combined->print();
    pass &= generate_transfer_addresses(combined);
    auto output_shape = combined->transformations.back()->groups[0]->shape;
    vector<vector<float>> data_transformed = evaluate(input_data, generate_address_map(combined), {output_shape});
    return data_transformed == golden_data;
}
bool test_pad_and_block_passes() {
    bool pass = true;
    vector<int> shape = {1, 2, 2};
    vector<float> input_data_2_2 = {1, 2, 3, 4};
    // list of tests - pad to nearest, block shape, dim order, golden data
    vector<tuple<vector<int>, vector<int>, vector<int>, vector<vector<float>>>> tests_2_2 = {
        { {4,4}, {4,4}, {0,1,2}, {{1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}} },
        { {4,4}, {2,4}, {0,1,2}, {{1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}} },
        { {4,4}, {2,2}, {0,1,2}, {{1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}} },
        { {4,4}, {4,1}, {0,1,2}, {{1, 3, 0, 0, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}} },
    };
    for (auto & t : tests_2_2) {
        auto pad_to_nearest = std::get<0>(t);
        auto block_shape = std::get<1>(t);
        auto dim_order = std::get<2>(t);
        auto golden_data = std::get<3>(t);
        pass &= test_pad_and_block_passes_(shape, pad_to_nearest, block_shape, dim_order, input_data_2_2, golden_data);
        if(pass) {
            tt::log_debug(tt::LogDTX, "Passed test with shape = {} , pad to nearest = {} , block shape = {} , dim order = {}", v2s(shape), v2s(pad_to_nearest), v2s(block_shape), v2s(dim_order));
        }
        else {
            tt::log_error(tt::LogDTX, "Failed test with shape = {} , pad to nearest = {} , block shape = {} , dim order = {}", v2s(shape), v2s(pad_to_nearest), v2s(block_shape), v2s(dim_order));
        }
        if (!pass) exit(1);
    }
    return pass;
}

void run_dtx_tests() {
    bool pass = true;

    tt::log_info(tt::LogDTX, "==================================================================");
    tt::log_info(tt::LogDTX, "                         Starting DTX TESTs                       ");
    tt::log_info(tt::LogDTX, "==================================================================");

    // pass &= test_GenerateAddresses();
    // printf("test_GenerateAddresses - %d\n\n", pass);

    //pass &= test_DTX_reverse_transformations();
    //printf("test_DTX_reverse_transformations - %d\n\n", pass);

    // pass &= test_generate_sliced_ranges_helper_functions();
    // printf("test_generate_sliced_ranges_helper_functions - %d\n\n", pass);

    // pass &= test_pass_parallelize_generic_tensor_slice();
    // printf("test_pass_parallelize_generic_tensor_slice - %d\n\n", pass);

    // pass &= test_pass_tilize_and_store();
    // printf("test_pass_tilize_and_store - %d\n\n", pass);

    //pass &= test_transpose_xy();
    //printf("test_transpose_xy - %d\n\n", pass);

    //pass &= test_tensor_evaluate();
    //printf("test_tensor_evaluate - %d\n\n", pass);

    // pass &= test_pass_transpose_xy();
    // printf("test_pass_transpose_xy - %d\n\n", pass);

    // pass &= test_pass_transpose_yz();
    // printf("test_pass_transpose_yz - %d\n\n", pass);

    //pass &= test_pass_convert_tensor_layout_3d_conv_act_to_2Dmatrix();
    //printf("test_pass_transpose_xy - %d\n\n", pass);

    //pass &= test_convert_abstract_tensor_to_channels_last_layout();         // TO DO: generalize rank
    //printf("test_pass_convert_abstract_tensor_to_channels_last_layout - %d\n\n", pass);

    pass &= test_channels_last_to_2D_matrix();
    tt::log_info(tt::LogDTX, "test_channels_last_to_2D_matrix - {}", pass);

    pass &= test_channels_last_to_2D_matrix_conv1x1();
    tt::log_info(tt::LogDTX, "test_channels_last_to_2D_matrix_conv1x1 - {}", pass);

    pass &= test_high_level_pass_and_evaluate();
    tt::log_info(tt::LogDTX, "test_high_level_pass_and_evaluate - {}", pass);

    pass &= test_block_2d_matrix_pass();
    tt::log_info(tt::LogDTX, "test_block_2d_matrix_pass - {}", pass);

    pass &= test_block_2d_matrix_group_pass();
    tt::log_info(tt::LogDTX, "test_block_2d_matrix_group_pass - {}", pass);

    pass &= test_padding_pass();
    tt::log_info(tt::LogDTX, "test_pad_2d_matrix_pass - {}", pass);

    pass &= test_pad_and_block_passes();
    tt::log_info(tt::LogDTX, "test_pad_and_block_passes - {}", pass);

    pass &= test_run_conv_transform_no_evaluate();
    tt::log_info(tt::LogDTX, "test_run_conv_transform_no_evaluate - {}", pass);

    if (pass == true) tt::log_info(tt::LogDTX, "TESTS PASSED");
    else tt::log_error(tt::LogDTX, "TESTS FAILED");
}

// ===============================================================
// ===============================================================

int main(int argc, char** argv) {

    // Run all Data Transformation Tests
    run_dtx_tests();
}
