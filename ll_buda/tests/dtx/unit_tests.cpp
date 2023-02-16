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

#include "dtx.hpp"
#include "util_vector_of_ints.hpp"
#include "util.hpp"
#include "dtx_passes.hpp"


#include "kb.hpp"

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

    Tensor * l1 = new Tensor( {0,0,0}, {1,2,3});
    l1->print();

    return 1;
}

bool test_TensorPair_class(){
    TensorPair * tp1 = new TensorPair( new Tensor({0,0,0},{10,10,10}), 3, new Tensor({2,2,2}, {8,8,8}));
    tp1->print_string();
    
    TensorPair * tp2 = new TensorPair( new Tensor({0,0,0},{10,10,10}), 2, new Tensor({2,2,2}, {8,8,8}));
    tp2->print_string();

    return 1;
}

bool compare_two_int_vectors(vector<int> first, vector<int> second){
    int rank = first.size();
    bool equal = true;
    for (int i=0; i<rank; i++){
        if (first[i] != second[i]) return false;
    }
    return true;
}

bool run_single_line_segment_overlap_test(vector<int> l1, vector<int> l2, vector<int> golden_overlap) {
    bool DEBUG = false;
    vector<int> overlap = calculate_line_segment_overlap_in_1d(l1[0], l1[1], l2[0], l2[1]);
    if (DEBUG) cout << "overlap:  calc=" << v2s(overlap) << ", golden=" << v2s(golden_overlap) << endl;
    return compare_two_int_vectors(overlap, golden_overlap);
}

bool test_calculate_line_segment_overlap_in_1d() {
    bool DEBUG = true;
    bool pass = true;

    // No overlap
    pass &= run_single_line_segment_overlap_test({0,10}, {20,30}, {-1,-1});
    pass &= run_single_line_segment_overlap_test({20,30}, {0,10}, {-1,-1});
    
    // Full overlap
    pass &= run_single_line_segment_overlap_test({0,10}, {2,8}, {2,8});
    pass &= run_single_line_segment_overlap_test({2,8}, {0,10}, {2,8});
    
    // Partial overlap
    pass &= run_single_line_segment_overlap_test({5,15}, {0,10}, {5,10});
    pass &= run_single_line_segment_overlap_test({0,10}, {5,15}, {5,10});    

    // TO DO
    // add a few more test cases:
    //  - lines have the same start/end point. to test the "=" in the overlap equations.      

    return pass;
}

bool compare_tensors(Tensor * t0, Tensor * t1) {
    bool is_equal = true;
    is_equal &= compare_two_int_vectors(t0->str, t1->str);
    is_equal &= compare_two_int_vectors(t0->end, t1->end);
    return is_equal;
}

bool run_single_tensor_overlap_test(Tensor * t0, Tensor * t1, Tensor * golden_overlap) {
    int DEBUG = false;
    bool pass = true;

    if (DEBUG) cout << "\nrun_single_tensor_overlap_test: " << t0->get_string() << " && " << t1->get_string() << endl;
    Tensor * overlap = calculate_tensor_overlap_in_nd(t0, t1);
    pass = compare_tensors(overlap, golden_overlap);
    if (DEBUG) cout << "comparing overlaps:  calculated: " << overlap->get_string() << " && golden: " << golden_overlap->get_string() << ", MATCH = " << pass << endl;
    return pass;
}

bool test_calculate_nd_tensor_overlap(){
    bool DEBUG = false;
    bool pass = true;

    //                                        INPUT TENSOR 1               INPUT TENSOR 2          GOLDEN OVERLAP
    pass &= run_single_tensor_overlap_test(new Tensor({0,1,2}, {30,31,32}), new Tensor({10,11,12}, {20,21,22}), new Tensor({10,11,12}, {20,21,22}));
    pass &= run_single_tensor_overlap_test(new Tensor({0,1}, {30,31}),      new Tensor({40,11}, {50,21}),       new Tensor({-1,11}, {-1,21}));
    pass &= run_single_tensor_overlap_test(new Tensor({0,1}, {30,31}),      new Tensor({10,40}, {20,50}),       new Tensor({10,-1}, {20,-1}));
    return pass;
}

bool run_DataTransformation_test_0(bool DEBUG) {
    bool pass = true;

    if (DEBUG) cout << "run_DataTransformation_test_0" << endl;

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

    if (DEBUG) cout << "run_DataTransformation_test_1" << endl;

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
    golden->groups[0]->shape = {40,40};
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

    if (DEBUG) cout << "run_DataTransformation_test_2" << endl;

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

bool test_golden_comparisons() {
    
    cout << "compare_two_vectors_of_ints" << endl;
    cout << "expecting 1, got " << compare_two_vectors_of_ints({1,2,3}, {1,2,3}) << endl;
    cout << "expecting 0, got " << compare_two_vectors_of_ints({1,2,3}, {1,2,3,4}) << endl;
    cout << "expecting 0, got " << compare_two_vectors_of_ints({1,2,3}, {1,2,4}) << endl;

    cout << "\ncompare_two_tensors" << endl;
    cout << "expecting 1, got " << compare_two_tensors(new Tensor({1}, {2}), new Tensor({1},{2})) << endl;
    cout << "expecting 0, got " << compare_two_tensors(new Tensor({1}, {2}), new Tensor({1},{3})) << endl;
    cout << "expecting 0, got " << compare_two_tensors(new Tensor({1}, {2}), new Tensor({1,1},{2,2})) << endl;

    cout << "\ncompare_two_tensor_pairs" << endl;
    cout << "expecting 1, got " << compare_two_tensor_pairs(new TensorPair(new Tensor({1},{2}), 0, new Tensor({1,1},{2,2})), new TensorPair(new Tensor({1},{2}), 0, new Tensor({1,1},{2,2}))           ) << endl;
    cout << "expecting 0, got " << compare_two_tensor_pairs(new TensorPair(new Tensor({1},{2}), 0, new Tensor({1,1},{2,2})), new TensorPair(new Tensor({1},{2}), 0, new Tensor({1,1},{9,2}))           ) << endl;
    cout << "expecting 0, got " << compare_two_tensor_pairs(new TensorPair(new Tensor({1},{2}), 0, new Tensor({1,1},{2,2})), new TensorPair(new Tensor({9},{2}), 0, new Tensor({1,1},{2,2}))           ) << endl;
    
    
    cout << "expecting 0, got " << compare_two_tensor_pairs(new TensorPair(new Tensor({1},{2}), 0, new Tensor({1,1},{9,2})), new TensorPair(new Tensor({1},{2}), 0, new Tensor({1,1},{2,2}))           ) << endl;
    cout << "expecting 0, got " << compare_two_tensor_pairs(new TensorPair(new Tensor({1},{2}), 0, new Tensor({1,1},{2,2})), new TensorPair(new Tensor({1},{2}), 3, new Tensor({1,1},{2,2}))           ) << endl;
    cout << "expecting 0, got " << compare_two_tensor_pairs(new TensorPair(new Tensor({2},{2}), 0, new Tensor({1,1},{2,2})), new TensorPair(new Tensor({1},{2}), 0, new Tensor({1,1},{2,2}))           ) << endl;
    
    return true;
}

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
    
    pass &= test_util();
    printf("test_util = %d\n\n", pass);
    
    /*
    pass &= test_Tensor_class();
    printf("test_Tensor_class = %d\n\n", pass);

    pass &= test_TensorPair_class();
    printf("test_TensorPair_class = %d\n\n", pass);
    
    pass &= test_calculate_line_segment_overlap_in_1d();
    printf("test_calculate_line_segment_overlap_in_1d - %d\n\n", pass);

    pass &= test_calculate_nd_tensor_overlap();
    printf("test_calculate_nd_tensor_overlap - %d\n\n", pass);

    pass &= test_golden_comparisons();
    printf("test_DataTransformations - %d\n\n", pass);

    pass &= test_DataTransformations();
    printf("test_DataTransformations - %d\n\n", pass);
    
    pass &= test_GenerateAddresses();
    printf("test_GenerateAddresses - %d\n\n", pass);
    
    //pass &= test_DTX_reverse_transformations();
    //printf("test_DTX_reverse_transformations - %d\n\n", pass);
    
    pass &= test_generate_sliced_ranges_helper_functions();
    printf("test_generate_sliced_ranges_helper_functions - %d\n\n", pass);

    pass &= test_pass_parallelize_generic_tensor_slice();
    printf("test_pass_parallelize_generic_tensor_slice - %d\n\n", pass);

    pass &= test_pass_tilize_and_store();
    printf("test_pass_tilize_and_store - %d\n\n", pass);

    pass &= test_pass_convert_tensor_layout_CL1_to_2Dmatrix_conv3x3_s1();
    printf("test_pass_convert_tensor_layout_CL1_to_2Dmatrix_conv3x3_s1 - %d\n\n", pass);
    */
    

    

    if (pass == true) cout << "\nTESTS PASSED\n\n\n" << endl;
    else cout << "TESTS FAILED\n\n\n" << endl;
}



void run_pass_memory_allocation(Graph * graph) {
}

// ===============================================================
//                  Kernel Buffer Graph Infra
// ===============================================================
void test_1() {

    Graph * g = new Graph();
    
    Node * buf_in = g->create_buffer_dram("dram_input_buffer", 0);
    Node * buf_out = g->create_buffer_dram("dram_output_buffer", 0);
    Node * buf_l1  = g->create_buffer_l1("l1_buffer", 1024, {0,0});
    Node * kernel = g->create_data_movement_kernel("dram_loopback", "/kernels/dram_loopback_va_buffer.cpp", {0,0});

    g->add_edge(buf_in, kernel);
    g->add_edge(kernel, buf_l1);
    g->add_edge(buf_l1, kernel);
    g->add_edge(kernel, buf_out);
    
    run_pass_memory_allocation(g);
}

bool run_kb_graph_tests() {
    bool pass = true;
    test_1();
    return pass;
}

// ===============================================================
// ===============================================================

int main(int argc, char** argv) {

    printf("\nHello World\n\n");

    // Run all Data Transformation Tests
    run_dtx_tests();

    //bool result = run_kb_graph_tests();
}
