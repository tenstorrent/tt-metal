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

#include "ll_buda/impl/dtx/dtx.hpp"
#include "ll_buda/impl/dtx/util_vector_of_ints.hpp"
#include "ll_buda/impl/dtx/util.hpp"
#include "ll_buda/impl/dtx/dtx_passes.hpp"

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
    node1->groups[0]->tensor_pairs.push_back(  new TensorPair( new Tensor({0},  {20}), 0,  new Tensor({20}, {40}))  );
    node1->groups[0]->tensor_pairs.push_back(  new TensorPair( new Tensor({20}, {40}), 0,  new Tensor({0}, {20}))   );

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
    node1->groups[0]->tensor_pairs.push_back(  new TensorPair( new Tensor({0},  {20}), 1, new Tensor({20}, {40}))  );
    node1->groups[0]->tensor_pairs.push_back(  new TensorPair( new Tensor({20}, {40}), 1, new Tensor({0}, {20}))   );

    node1->groups[1]->shape = {40};
    node1->groups[1]->tensor_pairs.push_back(  new TensorPair( new Tensor({0},  {20}), 0, new Tensor({20}, {40}))  );
    node1->groups[1]->tensor_pairs.push_back(  new TensorPair( new Tensor({20}, {40}), 0,  new Tensor({0}, {20}))   );

    DataTransformations * dtx = new DataTransformations();
    dtx->transformations.push_back(node0);
    dtx->transformations.push_back(node1);

    DataTransformations * backwards = reverse_transformations(dtx);

    /*
    // Correctness checking
    TransformationNode * golden = new TransformationNode("golden", 1);
    golden->groups[0]->shape = {40,40};
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new Tensor( {20}, {30}), 1, new Tensor({10}, {20})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new Tensor( {0 }, {10}), 1, new Tensor({30}, {40})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new Tensor( {30}, {40}), 1, new Tensor({20}, {30})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new Tensor( {10}, {20}), 1, new Tensor({0 }, {10})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new Tensor( {20}, {30}), 0, new Tensor({10}, {20})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new Tensor( {0 }, {10}), 0, new Tensor({30}, {40})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new Tensor( {30}, {40}), 0, new Tensor({20}, {30})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new Tensor( {10}, {20}), 0, new Tensor({0 }, {10})));

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
    node1->groups[0]->tensor_pairs.push_back(  new TensorPair( new Tensor({60}, {70}),  0, new Tensor({0},  {10}))  );
    node1->groups[0]->tensor_pairs.push_back(  new TensorPair( new Tensor({70}, {80}),  0, new Tensor({10}, {20}))   );

    // NODE 1:
    node1->groups[1]->shape = {30};
    node1->groups[1]->tensor_pairs.push_back(  new TensorPair( new Tensor({130}, {140}), 1, new Tensor({0},  {10}))  );
    node1->groups[1]->tensor_pairs.push_back(  new TensorPair( new Tensor({140}, {150}), 1, new Tensor({10}, {20}))  );

    node1->groups[2]->shape = {40};
    node1->groups[2]->tensor_pairs.push_back(  new TensorPair( new Tensor({80},  {90}),  0, new Tensor({0},  {10}))  );
    node1->groups[2]->tensor_pairs.push_back(  new TensorPair( new Tensor({160}, {170}), 1, new Tensor({10}, {20}))  );

    DataTransformations * dtx = new DataTransformations();
    dtx->transformations.push_back(node0);
    dtx->transformations.push_back(node1);

    DataTransformations * backwards = reverse_transformations(dtx);


    /*
    // Correctness checking
    TransformationNode * golden = new TransformationNode("golden", 1);
    golden->groups[0]->shape = {40,40};
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new Tensor( {20}, {30}), 1, new Tensor({10}, {20})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new Tensor( {0 }, {10}), 1, new Tensor({30}, {40})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new Tensor( {30}, {40}), 1, new Tensor({20}, {30})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new Tensor( {10}, {20}), 1, new Tensor({0 }, {10})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new Tensor( {20}, {30}), 0, new Tensor({10}, {20})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new Tensor( {0 }, {10}), 0, new Tensor({30}, {40})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new Tensor( {30}, {40}), 0, new Tensor({20}, {30})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new Tensor( {10}, {20}), 0, new Tensor({0 }, {10})));

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
    node1->groups[0]->tensor_pairs.push_back(  new TensorPair( new Tensor({10}, {20}),  0, new Tensor({0},  {10}))  );
    node1->groups[0]->tensor_pairs.push_back(  new TensorPair( new Tensor({0},  {10}),  0, new Tensor({10}, {20}))   );
    node1->groups[0]->tensor_pairs.push_back(  new TensorPair( new Tensor({30}, {40}),  0, new Tensor({20}, {30}))   );
    node1->groups[0]->tensor_pairs.push_back(  new TensorPair( new Tensor({20}, {30}),  0, new Tensor({30}, {40}))   );

    // NODE 2:
    node2->groups[0]->shape = {40};
    node2->groups[0]->tensor_pairs.push_back(  new TensorPair( new Tensor({20}, {30}),  0, new Tensor({0},  {10}))  );
    node2->groups[0]->tensor_pairs.push_back(  new TensorPair( new Tensor({30}, {40}),  0, new Tensor({10}, {20}))   );
    node2->groups[0]->tensor_pairs.push_back(  new TensorPair( new Tensor({0},  {10}),  0, new Tensor({20}, {30}))   );
    node2->groups[0]->tensor_pairs.push_back(  new TensorPair( new Tensor({10}, {20}),  0, new Tensor({30}, {40}))   );

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
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new Tensor( {20}, {30}), 1, new Tensor({10}, {20})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new Tensor( {0 }, {10}), 1, new Tensor({30}, {40})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new Tensor( {30}, {40}), 1, new Tensor({20}, {30})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new Tensor( {10}, {20}), 1, new Tensor({0 }, {10})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new Tensor( {20}, {30}), 0, new Tensor({10}, {20})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new Tensor( {0 }, {10}), 0, new Tensor({30}, {40})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new Tensor( {30}, {40}), 0, new Tensor({20}, {30})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new Tensor( {10}, {20}), 0, new Tensor({0 }, {10})));

    if (DEBUG) golden->print(0);

    pass = compare_two_groups(golden->groups[0], node2->groups[0]);
    if (DEBUG) cout << "PASS = " << pass << endl;
    */

   return true;
}

bool test_DTX_reverse_transformations() {
    bool DEBUG = true;
    bool pass = true;

    pass &= run_DTX_reverse_transformations_test_0(DEBUG);
    pass &= run_DTX_reverse_transformations_test_1(DEBUG);
    pass &= run_DTX_reverse_transformations_test_2(DEBUG);
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

bool test_pass_convert_tensor_layout_CL1_to_2Dmatrix_conv3x3_s1() {
    bool pass = true;

    // Test #1
    DataTransformations * dtx = new DataTransformations();
    TransformationNode * node0 = new TransformationNode("producer", 1);

    int x = 5;
    int y = 5;
    int z = 128;


    node0->groups[0]->shape = {1, z*y*x};
    dtx->transformations.push_back(node0);

    pass &= convert_tensor_layout_CL1_to_2Dmatrix_conv3x3_s1(dtx);

    return pass;
}

void run_dtx_tests() {
    bool pass = true;

    cout << "==================================================================" << endl;
    cout << "                         Starting DTX TESTs                       " << endl;
    cout << "==================================================================" << endl;

    // pass &= test_GenerateAddresses();
    // printf("test_GenerateAddresses - %d\n\n", pass);

    // //pass &= test_DTX_reverse_transformations();
    // //printf("test_DTX_reverse_transformations - %d\n\n", pass);

    // pass &= test_generate_sliced_ranges_helper_functions();
    // printf("test_generate_sliced_ranges_helper_functions - %d\n\n", pass);

    // pass &= test_pass_parallelize_generic_tensor_slice();
    // printf("test_pass_parallelize_generic_tensor_slice - %d\n\n", pass);

    // pass &= test_pass_tilize_and_store();
    // printf("test_pass_tilize_and_store - %d\n\n", pass);

    // pass &= test_pass_convert_tensor_layout_CL1_to_2Dmatrix_conv3x3_s1();
    // printf("test_pass_convert_tensor_layout_CL1_to_2Dmatrix_conv3x3_s1 - %d\n\n", pass);


    if (pass == true) cout << "\nTESTS PASSED\n\n\n" << endl;
    else cout << "TESTS FAILED\n\n\n" << endl;
}

// ===============================================================
// ===============================================================

int main(int argc, char** argv) {

    // Run all Data Transformation Tests
    run_dtx_tests();
}
