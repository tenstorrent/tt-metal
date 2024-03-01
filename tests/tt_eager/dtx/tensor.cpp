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

bool test_util() {

    cout << v2s(increment({3,4}, {2,3}, {5,7})) << endl;
    cout << v2s(increment({3,6}, {2,3}, {5,7})) << endl;
    cout << v2s(increment({3,7}, {2,3}, {5,7})) << endl;

    cout << v2s(increment({2,2,5}, {2,2,2}, {5,5,5})) << endl;
    cout << v2s(increment({3,5,5}, {2,2,2}, {5,5,5})) << endl;
    return true;
}

bool test_Tensor_class() {

    DTXTensor * l1 = new DTXTensor( {0,0,0}, {1,2,3});
    l1->print();

    return 1;
}

bool test_TensorPair_class(){
    TensorPair * tp1 = new TensorPair( new DTXTensor({0,0,0},{10,10,10}), 3, new DTXTensor({2,2,2}, {8,8,8}));
    tp1->print_string();

    TensorPair * tp2 = new TensorPair( new DTXTensor({0,0,0},{10,10,10}), 2, new DTXTensor({2,2,2}, {8,8,8}));
    tp2->print_string();

    return 1;
}

bool test_golden_comparisons() {

    cout << "compare_two_vectors_of_ints" << endl;
    cout << "expecting 1, got " << compare_two_vectors_of_ints({1,2,3}, {1,2,3}) << endl;
    cout << "expecting 0, got " << compare_two_vectors_of_ints({1,2,3}, {1,2,3,4}) << endl;
    cout << "expecting 0, got " << compare_two_vectors_of_ints({1,2,3}, {1,2,4}) << endl;

    cout << "\ncompare_two_tensors" << endl;
    cout << "expecting 1, got " << compare_two_tensors(new DTXTensor({1}, {2}), new DTXTensor({1},{2})) << endl;
    cout << "expecting 0, got " << compare_two_tensors(new DTXTensor({1}, {2}), new DTXTensor({1},{3})) << endl;
    cout << "expecting 0, got " << compare_two_tensors(new DTXTensor({1}, {2}), new DTXTensor({1,1},{2,2})) << endl;

    cout << "\ncompare_two_tensor_pairs" << endl;
    cout << "expecting 1, got " << compare_two_tensor_pairs(new TensorPair(new DTXTensor({1},{2}), 0, new DTXTensor({1,1},{2,2})), new TensorPair(new DTXTensor({1},{2}), 0, new DTXTensor({1,1},{2,2}))           ) << endl;
    cout << "expecting 0, got " << compare_two_tensor_pairs(new TensorPair(new DTXTensor({1},{2}), 0, new DTXTensor({1,1},{2,2})), new TensorPair(new DTXTensor({1},{2}), 0, new DTXTensor({1,1},{9,2}))           ) << endl;
    cout << "expecting 0, got " << compare_two_tensor_pairs(new TensorPair(new DTXTensor({1},{2}), 0, new DTXTensor({1,1},{2,2})), new TensorPair(new DTXTensor({9},{2}), 0, new DTXTensor({1,1},{2,2}))           ) << endl;


    cout << "expecting 0, got " << compare_two_tensor_pairs(new TensorPair(new DTXTensor({1},{2}), 0, new DTXTensor({1,1},{9,2})), new TensorPair(new DTXTensor({1},{2}), 0, new DTXTensor({1,1},{2,2}))           ) << endl;
    cout << "expecting 0, got " << compare_two_tensor_pairs(new TensorPair(new DTXTensor({1},{2}), 0, new DTXTensor({1,1},{2,2})), new TensorPair(new DTXTensor({1},{2}), 3, new DTXTensor({1,1},{2,2}))           ) << endl;
    cout << "expecting 0, got " << compare_two_tensor_pairs(new TensorPair(new DTXTensor({2},{2}), 0, new DTXTensor({1,1},{2,2})), new TensorPair(new DTXTensor({1},{2}), 0, new DTXTensor({1,1},{2,2}))           ) << endl;

    return true;
}

void run_tensor_data_test(vector<int> shape, string filename){
    TensorData * t = new TensorData(shape);
    t->print();
    //t->generate_csv(filename);
}


bool test_tensor_data_class() {
    run_tensor_data_test({4},     "tensor1");
    run_tensor_data_test({4,4},   "tensor1");
    run_tensor_data_test({4,4,4}, "tensor1");
    return true;
}

int main(int argc, char** argv) {
    bool pass = true;
    /*
    pass &= test_util();
    printf("test_util = %d\n\n", pass);

    pass &= test_Tensor_class();
    printf("test_Tensor_class = %d\n\n", pass);

    pass &= test_TensorPair_class();
    printf("test_TensorPair_class = %d\n\n", pass);

    pass &= test_golden_comparisons();
    printf("test_DataTransformations - %d\n\n", pass);
    */
    pass &= test_tensor_data_class();
    tt::log_info(tt::LogDTX, "test_tensor_data_class - {}", pass);

    if (pass == true) tt::log_debug(tt::LogDTX, "TESTS PASSED");
    else tt::log_error(tt::LogDTX, "TESTS FAILED");

}
