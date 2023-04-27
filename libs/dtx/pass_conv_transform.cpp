#include "dtx.hpp"
#include "dtx_passes.hpp"
#include "util.hpp"
#include "util_vector_of_ints.hpp"

DataTransformations * conv_transform(vector<int> shape, vector<int> conv_params, std::pair<vector<int>,vector<int>> block_info) {
    //assert(R == S && "Only square kernel window supported.");
    //assert((R == 1 || R == 3) && "Only 1x1 and 3x3 kernel window supported.");
    auto activation_C = shape[1];
    //assert(activation_C % 32 == 0 && "Channel depth of tensor needs to be divisible by 32");
    // Right side: AbstractTensor --> consumer conv/mm
    DataTransformations * dtx_right = new DataTransformations();
    TransformationNode * node0 = new TransformationNode("producer", 1);
    node0->groups[0]->shape = shape;
    dtx_right->transformations.push_back(node0);
    bool pass = true;
    pass &= convert_tensor_layout_3d_conv_act_to_2Dmatrix(dtx_right, conv_params);

    // Get the 2d matrix shape
    auto matrix_shape = dtx_right->transformations.back()->groups[0]->shape;
    assert(matrix_shape.size() == 3);
    assert(matrix_shape[0] == 1);
    uint32_t num_rows = (uint32_t) matrix_shape[1];
    uint32_t num_cols = (uint32_t) matrix_shape[2];
    if (true) {
        pass &= pad_2d_matrix(dtx_right, {32,32});
    }
    auto dim_order = block_info.first;
    auto block_shape = block_info.second;
    if(dim_order[0] != -1 && block_shape[0] != -1) {
        pass &= block_2d_matrix(dtx_right, dim_order, block_shape);
    }
    pass &= row_major_memory_store_blocks(dtx_right);

    //cout << "\n\nDTX_RIGHT" << endl;
    //dtx_right->print();


    // Left side: AbstractTensor --> producer memory buffer
    DataTransformations * dtx_left = new DataTransformations();
    TransformationNode * node1 = new TransformationNode("producer", 1);
    node1->groups[0]->shape = shape;
    dtx_left->transformations.push_back(node1);
    pass &= convert_abstract_tensor_to_channels_last_layout(dtx_left);

    //cout << "\n\nDTX_LEFT" << endl;
    //dtx_left->print();

    DataTransformations * combined = reverse_and_combine_transformations(dtx_left, dtx_right);
    //cout << "\n\nDTX_COMBINED" << endl;
    //combined->print();

    pass &= optimize_away_transpose(combined);
    //cout << "\n\nDTX_OPTIMIZED" << endl;
    //combined->print();

    pass &= collapse_transformations(combined);
    //cout << "\n\nDTX_COLLAPSED" << endl;
    //combined->print();
    pass &= generate_transfer_addresses(combined);
    //combined->print();
    //delete dtx_right;
    //delete dtx_left;
    return combined;
}
