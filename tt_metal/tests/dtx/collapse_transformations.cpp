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

#include "tt_metal/impl/dtx/dtx.hpp"
#include "tt_metal/impl/dtx/util_vector_of_ints.hpp"
#include "tt_metal/impl/dtx/util.hpp"
#include "tt_metal/impl/dtx/dtx_passes.hpp"

using namespace std;

bool run_DataTransformation_test_0(bool DEBUG) {
    bool pass = true;

    if (DEBUG) cout << "======================================\nrun_DataTransformation_test_0\n======================================" << endl;

    TransformationNode * node0 = new TransformationNode("producer", 1);
    TransformationNode * node1 = new TransformationNode("tx1", 1);
    TransformationNode * node2 = new TransformationNode("consumer", 1);

    // Producer node: 1 tensor.
    node0->groups[0]->shape = {20,20};

    // Transformation #1: vertical blocks
    node1->groups[0]->shape = {20,20};
    node1->groups[0]->tensor_pairs.push_back(  new TensorPair( new Tensor({0,0},  {20,10}), 0,  new Tensor({0,0},  {20,10}))   );
    node1->groups[0]->tensor_pairs.push_back(  new TensorPair( new Tensor({0,10}, {20,20}), 0,  new Tensor({0,10}, {20,20}))   );

    node2->groups[0]->shape = {20,20};
    node2->groups[0]->tensor_pairs.push_back(  new TensorPair( new Tensor({0,0}, {10,20}),  0,  new Tensor({0,0}, {10,20}))    );
    node2->groups[0]->tensor_pairs.push_back(  new TensorPair( new Tensor({10,0}, {20,20}), 0,  new Tensor({10,0}, {20,20}))   );

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
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new Tensor( {0,0},   {10,10}), 0, new Tensor({0,0},   {10,10})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new Tensor( {0,10},  {10,20}), 0, new Tensor({0,10},  {10,20})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new Tensor( {10,0},  {20,10}), 0, new Tensor({10,0},  {20,10})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new Tensor( {10,10}, {20,20}), 0, new Tensor({10,10}, {20,20})));

    if (DEBUG) golden->print(0);

    pass = compare_two_groups(golden->groups[0], node2->groups[0]);
    if (DEBUG) cout << "PASS = " << pass << endl;

    return pass;
}

bool run_DataTransformation_test_1(bool DEBUG) {
    bool pass = true;

    if (DEBUG) cout << "======================================\nrun_DataTransformation_test_1\n======================================" << endl;

    TransformationNode * node0 = new TransformationNode("producer", 1);
    TransformationNode * node1 = new TransformationNode("tx1", 1);
    TransformationNode * node2 = new TransformationNode("consumer", 1);

    // Producer node: 1 tensor.
    node0->groups[0]->shape = {40};

    // Transformation #1: vertical blocks
    node1->groups[0]->shape = {40};
    node1->groups[0]->tensor_pairs.push_back(  new TensorPair( new Tensor({0},  {20}), 0,  new Tensor({20}, {40}))  );
    node1->groups[0]->tensor_pairs.push_back(  new TensorPair( new Tensor({20}, {40}), 0,  new Tensor({0}, {20}))   );

    node2->groups[0]->shape = {40};
    node2->groups[0]->tensor_pairs.push_back(  new TensorPair( new Tensor({0},  {10}), 0,  new Tensor({10}, {20}))   );
    node2->groups[0]->tensor_pairs.push_back(  new TensorPair( new Tensor({10}, {30}), 0,  new Tensor({20}, {40}))   );
    node2->groups[0]->tensor_pairs.push_back(  new TensorPair( new Tensor({30}, {40}), 0,  new Tensor({0},  {0}))    );

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
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new Tensor( {20}, {30}), 0, new Tensor({10}, {20})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new Tensor( {0 }, {10}), 0, new Tensor({30}, {40})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new Tensor( {30}, {40}), 0, new Tensor({20}, {30})));
    golden->groups[0]->tensor_pairs.push_back( new TensorPair (new Tensor( {10}, {20}), 0, new Tensor({0 }, {10})));

    if (DEBUG) golden->print(0);

    pass = compare_two_groups(golden->groups[0], node2->groups[0]);
    if (DEBUG) cout << "PASS = " << pass << endl;


    return pass;
}

bool run_DataTransformation_test_2(bool DEBUG) {
    bool pass = true;

    if (DEBUG) cout << "======================================\nrun_DataTransformation_test_2\n======================================" << endl;

    TransformationNode * node0 = new TransformationNode("producer", 2);
    TransformationNode * node1 = new TransformationNode("tx1", 2);
    TransformationNode * node2 = new TransformationNode("consumer", 1);

    // NODE 0:
    node0->groups[0]->shape = {40,40};
    node0->groups[1]->shape = {40,40};

    // NODE 1:
    node1->groups[0]->shape = {40,40};
    node1->groups[0]->tensor_pairs.push_back(  new TensorPair( new Tensor({0},  {20}), 1, new Tensor({20}, {40}))  );
    node1->groups[0]->tensor_pairs.push_back(  new TensorPair( new Tensor({20}, {40}), 1, new Tensor({0}, {20}))   );

    node1->groups[1]->shape = {40,40};
    node1->groups[1]->tensor_pairs.push_back(  new TensorPair( new Tensor({0},  {20}), 0, new Tensor({20}, {40}))  );
    node1->groups[1]->tensor_pairs.push_back(  new TensorPair( new Tensor({20}, {40}), 0,  new Tensor({0}, {20}))   );

    // NODE 2:
    node2->groups[0]->shape = {40,40};
    node2->groups[0]->tensor_pairs.push_back(  new TensorPair( new Tensor({0},  {10}), 0,  new Tensor({10}, {20}))   );
    node2->groups[0]->tensor_pairs.push_back(  new TensorPair( new Tensor({10}, {30}), 1,  new Tensor({20}, {40}))   );
    node2->groups[0]->tensor_pairs.push_back(  new TensorPair( new Tensor({30}, {40}), 1,  new Tensor({0},  {0}))    );

    DataTransformations * dtx = new DataTransformations();
    dtx->transformations.push_back(node0);
    dtx->transformations.push_back(node1);
    dtx->transformations.push_back(node2);

    if (DEBUG) dtx->print();
    pass &= collapse_transformations(dtx);
    if (DEBUG) dtx->print();

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
    node1->groups[0]->tensor_pairs.push_back(  new TensorPair( new Tensor({0},  {20}), 0, new Tensor({20},  {40}))  );

    // NODE 2:
    node2->groups[0]->shape = {100};
    node2->groups[0]->tensor_pairs.push_back(  new TensorPair( new Tensor({20},  {40}), 0,  new Tensor({40}, {60}))   );

    // NODE 3:
    node3->groups[0]->shape = {100};
    node3->groups[0]->tensor_pairs.push_back(  new TensorPair( new Tensor({40},  {60}), 0,  new Tensor({60}, {80}))   );

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
    pass &= run_DataTransformation_test_2(DEBUG);
    pass &= run_DataTransformation_test_3(DEBUG);   // TODO: missing golden comparison
    return pass;
}


int main(int argc, char** argv) {
    bool pass = true;

    pass &= test_DataTransformations();
    printf("test_DataTransformations - %d\n\n", pass);

    if (pass == true) cout << "\nTESTS PASSED\n\n\n" << endl;
    else cout << "TESTS FAILED\n\n\n" << endl;

}
