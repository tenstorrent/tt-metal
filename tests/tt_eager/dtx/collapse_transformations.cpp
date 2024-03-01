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

bool run_DataTransformation_test_0(bool DEBUG) {
    bool pass = true;

    if (DEBUG) tt::log_debug(tt::LogDTX, "======================================\nrun_DataTransformation_test_0\n======================================");
    TransformationNode * node0 = new TransformationNode("producer", 1);
    TransformationNode * node1 = new TransformationNode("tx1", 1);
    TransformationNode * node2 = new TransformationNode("consumer", 1);

    // Producer node: 1 tensor.
    node0->groups[0]->shape = {20,20};

    // Transformation #1: vertical blocks
    node1->groups[0]->shape = {20,20};
    node1->groups[0]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({0,0},  {19,9}), 0,  new DTXTensor({0,0},  {19,9}))   );
    node1->groups[0]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({0,10}, {19,19}), 0,  new DTXTensor({0,10}, {19,19}))   );

    node2->groups[0]->shape = {20,20};
    node2->groups[0]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({0,0}, {9,19}),  0,  new DTXTensor({0,0}, {9,19}))    );
    node2->groups[0]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({10,0}, {19,19}), 0,  new DTXTensor({10,0}, {19,19}))   );

    DataTransformations * dtx = new DataTransformations();
    dtx->transformations.push_back(node0);
    dtx->transformations.push_back(node1);
    dtx->transformations.push_back(node2);

    dtx->print();
    pass &= collapse_transformations(dtx);
    dtx->print();

    // Correctness checking
    TransformationNode * golden = new TransformationNode("golden", 1);
    golden->groups[0]->shape = {20,20};
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new DTXTensor( {0,0},   {9,9}), 0, new DTXTensor({0,0},   {9,9})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new DTXTensor( {0,10},  {9,19}), 0, new DTXTensor({0,10},  {9,19})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new DTXTensor( {10,0},  {19,9}), 0, new DTXTensor({10,0},  {19,9})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new DTXTensor( {10,10}, {19,19}), 0, new DTXTensor({10,10}, {19,19})));

    if (DEBUG) golden->print(0);

    pass = compare_two_groups(golden->groups[0], node2->groups[0]);
    if (DEBUG) tt::log_debug(tt::LogDTX, "PASS = {}", pass);

    return pass;
}

bool run_DataTransformation_test_1(bool DEBUG) {
    bool pass = true;

    if (DEBUG) tt::log_debug(tt::LogDTX, "======================================\nrun_DataTransformation_test_1\n======================================");

    TransformationNode * node0 = new TransformationNode("producer", 1);
    TransformationNode * node1 = new TransformationNode("tx1", 1);
    TransformationNode * node2 = new TransformationNode("consumer", 1);

    // Producer node: 1 tensor.
    node0->groups[0]->shape = {40};

    // Transformation #1: vertical blocks
    node1->groups[0]->shape = {40};
    node1->groups[0]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({0},  {19}), 0,  new DTXTensor({20}, {39}))  );
    node1->groups[0]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({20}, {39}), 0,  new DTXTensor({0}, {19}))   );

    node2->groups[0]->shape = {40};
    node2->groups[0]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({0},  {9}), 0,  new DTXTensor({10}, {19}))   );
    node2->groups[0]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({10}, {29}), 0,  new DTXTensor({20}, {39}))   );
    node2->groups[0]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({30}, {39}), 0,  new DTXTensor({0},  {9}))    );

    DataTransformations * dtx = new DataTransformations();
    dtx->transformations.push_back(node0);
    dtx->transformations.push_back(node1);
    dtx->transformations.push_back(node2);

    if (DEBUG) dtx->print();
    pass &= collapse_transformations(dtx);
    if (DEBUG) dtx->print();

    // Correctness checking
    TransformationNode * golden = new TransformationNode("golden", 1);
    golden->groups[0]->shape = {40};
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new DTXTensor( {20}, {29}), 0, new DTXTensor({10}, {19})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new DTXTensor( {0}, {9}), 0, new DTXTensor({30}, {39})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new DTXTensor( {30}, {39}), 0, new DTXTensor({20}, {29})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new DTXTensor( {10}, {19}), 0, new DTXTensor({0}, {9})));

    if (DEBUG) golden->print(0);

    pass = compare_two_groups(golden->groups[0], node2->groups[0]);
    if (DEBUG) tt::log_debug(tt::LogDTX, "PASS = {}", pass);


    return pass;
}

bool run_DataTransformation_test_2(bool DEBUG) {
    bool pass = true;

    if (DEBUG) cout << "======================================\nrun_DataTransformation_test_2\n======================================" << endl;

    TransformationNode * node0 = new TransformationNode("producer", 2);
    TransformationNode * node1 = new TransformationNode("tx1", 2);
    TransformationNode * node2 = new TransformationNode("consumer", 1);

    // NODE 0:
    node0->groups[0]->shape = {40};
    node0->groups[1]->shape = {40};

    // NODE 1:
    node1->groups[0]->shape = {40};
    node1->groups[0]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({0},  {19}), 1, new DTXTensor({20}, {39}))  );
    node1->groups[0]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({20}, {39}), 1, new DTXTensor({0}, {19}))   );

    node1->groups[1]->shape = {40};
    node1->groups[1]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({0},  {19}), 0, new DTXTensor({20}, {39}))  );
    node1->groups[1]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({20}, {39}), 0,  new DTXTensor({0}, {19}))   );

    // NODE 2:
    node2->groups[0]->shape = {80};
    node2->groups[0]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({0},  {39}), 0,  new DTXTensor({0}, {39}))   );
    node2->groups[0]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({0}, {39}), 1,  new DTXTensor({40}, {79}))   );

    DataTransformations * dtx = new DataTransformations();
    dtx->transformations.push_back(node0);
    dtx->transformations.push_back(node1);
    dtx->transformations.push_back(node2);

    if (DEBUG) dtx->print();
    pass &= collapse_transformations(dtx);
    if (DEBUG) dtx->print();

    // Correctness checking
    TransformationNode * golden = new TransformationNode("golden", 1);
    golden->groups[0]->shape = {80};
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new DTXTensor( {0}, {19}), 1, new DTXTensor({20}, {39})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new DTXTensor( {20}, {39}), 1, new DTXTensor({0}, {19})));

    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new DTXTensor( {0}, {19}), 0, new DTXTensor({60}, {79})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new DTXTensor( {20}, {39}), 0, new DTXTensor({40}, {59})));

    if (DEBUG) golden->print(0);

    pass = compare_two_groups(golden->groups[0], node2->groups[0]);
    if (DEBUG) cout << "PASS = " << pass << endl;

    return pass;
}

bool run_DataTransformation_test_3(bool DEBUG) {
    bool pass = true;

    if (DEBUG) cout << "======================================\nrun_DataTransformation_test_3\n======================================" << endl;

    TransformationNode * node0 = new TransformationNode("producer", 1);
    TransformationNode * node1 = new TransformationNode("tx1", 1);
    TransformationNode * node2 = new TransformationNode("tx2", 1);
    TransformationNode * node3 = new TransformationNode("consumer", 1);

    // NODE 0:
    node0->groups[0]->shape = {100};

    // NODE 1:
    node1->groups[0]->shape = {100};
    node1->groups[0]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({0},  {20}), 0, new DTXTensor({20},  {40}))  );

    // NODE 2:
    node2->groups[0]->shape = {100};
    node2->groups[0]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({20},  {40}), 0,  new DTXTensor({40}, {60}))   );

    // NODE 3:
    node3->groups[0]->shape = {100};
    node3->groups[0]->tensor_pairs.push_back(  new TensorPair( new DTXTensor({40},  {60}), 0,  new DTXTensor({60}, {80}))   );

    DataTransformations * dtx = new DataTransformations();
    dtx->transformations.push_back(node0);
    dtx->transformations.push_back(node1);
    dtx->transformations.push_back(node2);
    dtx->transformations.push_back(node3);

    if (DEBUG) dtx->print();
    pass &= collapse_transformations(dtx);
    if (DEBUG) dtx->print();

    cout << "TO DO: Add golden check" << endl;
    if (DEBUG) cout << "PASS = " << pass << endl;

    return pass;
}

bool test_DataTransformations() {
    bool DEBUG = true;
    bool pass = true;

    pass &= run_DataTransformation_test_0(DEBUG);
    pass &= run_DataTransformation_test_1(DEBUG);
    //pass &= run_DataTransformation_test_2(DEBUG);
    //pass &= run_DataTransformation_test_3(DEBUG);   // TODO: missing golden comparison
    return pass;
}


int main(int argc, char** argv) {
    bool pass = true;

    pass &= test_DataTransformations();
    tt::log_info(tt::LogDTX, "test_DataTransformations - {}", pass);

    if (pass == true) tt::log_debug(tt::LogDTX, "TESTS PASSED");
    else tt::log_error(tt::LogDTX, "TESTS FAILED");

}
