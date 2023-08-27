#include "dtx.hpp"
#include "dtx_passes.hpp"
#include "util.hpp"
#include "util_vector_of_ints.hpp"

using namespace std;

DataTransformations * simple_high_level_pass(vector<int> shape) {
    bool pass = true;
    assert(shape.size() == 3);
    DataTransformations * dtx_left = new DataTransformations();
    TransformationNode * node1 = new TransformationNode("producer", 1);
    node1->groups[0]->shape = shape;
    dtx_left->transformations.push_back(node1);
    pass &= row_major_memory_store(dtx_left);
    //dtx_left->print();
    DataTransformations * dtx_right = new DataTransformations();
    TransformationNode * node0 = new TransformationNode("producer", 1);
    node0->groups[0]->shape = shape;
    dtx_right->transformations.push_back(node0);
    pass &= transpose_yz(dtx_right);
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
    return combined;
}
