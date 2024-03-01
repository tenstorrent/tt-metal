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

    // Partial overlap (single element) at edge
    pass &= run_single_line_segment_overlap_test({5,15}, {5,5}, {5,5});
    pass &= run_single_line_segment_overlap_test({15,15}, {5,15}, {15,15});
    pass &= run_single_line_segment_overlap_test({5,15}, {3,5}, {5,5});
    pass &= run_single_line_segment_overlap_test({5,15}, {15,16}, {15,15});

    // TO DO
    // add a few more test cases:
    //  - lines have the same start/end point. to test the "=" in the overlap equations.

    return pass;
}

bool compare_tensors(DTXTensor * t0, DTXTensor * t1) {
    bool is_equal = true;
    is_equal &= compare_two_int_vectors(t0->str, t1->str);
    is_equal &= compare_two_int_vectors(t0->end, t1->end);
    return is_equal;
}

bool run_single_tensor_overlap_test(DTXTensor * t0, DTXTensor * t1, DTXTensor * golden_overlap) {
    int DEBUG = false;
    bool pass = true;

    if (DEBUG) cout << "\nrun_single_tensor_overlap_test: " << t0->get_string() << " && " << t1->get_string() << endl;
    DTXTensor * overlap = calculate_tensor_overlap_in_nd(t0, t1);
    pass = compare_tensors(overlap, golden_overlap);
    if (DEBUG) cout << "comparing overlaps:  calculated: " << overlap->get_string() << " && golden: " << golden_overlap->get_string() << ", MATCH = " << pass << endl;
    return pass;
}

bool test_calculate_nd_tensor_overlap(){
    bool DEBUG = false;
    bool pass = true;

    //                                        INPUT TENSOR 1               INPUT TENSOR 2          GOLDEN OVERLAP
    pass &= run_single_tensor_overlap_test(new DTXTensor({0,1,2}, {30,31,32}), new DTXTensor({10,11,12}, {20,21,22}), new DTXTensor({10,11,12}, {20,21,22}));
    pass &= run_single_tensor_overlap_test(new DTXTensor({0,1}, {30,31}),      new DTXTensor({40,11}, {50,21}),       new DTXTensor({-1,11}, {-1,21}));
    pass &= run_single_tensor_overlap_test(new DTXTensor({0,1}, {30,31}),      new DTXTensor({10,40}, {20,50}),       new DTXTensor({10,-1}, {20,-1}));
    return pass;
}

int main(int argc, char** argv) {
    bool pass = true;

    pass &= test_calculate_line_segment_overlap_in_1d();
    tt::log_info(tt::LogDTX, "test_calculate_line_segment_overlap_in_1d - {}", pass);

    pass &= test_calculate_nd_tensor_overlap();
    tt::log_info(tt::LogDTX, "test_calculate_nd_tensor_overlap - {}", pass);

    if (pass == true) tt::log_debug(tt::LogDTX, "TESTS PASSED");
    else tt::log_error(tt::LogDTX, "TESTS FAILED");
}
