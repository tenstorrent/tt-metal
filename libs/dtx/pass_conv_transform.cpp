#include "dtx.hpp"
#include "dtx_passes.hpp"
#include "util.hpp"
#include "util_vector_of_ints.hpp"

std::vector<uint32_t> conv_transform(vector<int> shape, vector<int> conv_params, std::pair<vector<int>,vector<int>> block_info, uint32_t num_bytes_of_df) {
    //assert(R == S && "Only square kernel window supported.");
    //assert((R == 1 || R == 3) && "Only 1x1 and 3x3 kernel window supported.");
    assert(shape.size() == 3);
    auto activation_C = shape[1];
    //assert(activation_C % 32 == 0 && "Channel depth of tensor needs to be divisible by 32");
    // Right side: AbstractTensor --> consumer conv/mm
    DataTransformations * dtx_right = new DataTransformations();
    TransformationNode * node0 = new TransformationNode("producer", 1);
    node0->groups[0]->shape = shape;
    dtx_right->transformations.push_back(node0);
    bool pass = true;
    pass &= convert_tensor_layout_3d_conv_act_to_2Dmatrix(dtx_right, conv_params);
    pass &= pad_2d_matrix(dtx_right, {32,32});
    // Get the 2d matrix shape
    auto matrix_shape = dtx_right->transformations.back()->groups[0]->shape;
    assert(matrix_shape.size() == 3);
    assert(matrix_shape[0] == 1);
    uint32_t num_rows = (uint32_t) matrix_shape[1];
    uint32_t num_cols = (uint32_t) matrix_shape[2];
    auto dim_order = block_info.first;
    auto block_shape = block_info.second;
    pass &= block_2d_matrix(dtx_right, dim_order, block_shape);
    auto blocked_matrix_shape = dtx_right->transformations.back()->groups[0]->shape;
    assert(blocked_matrix_shape.size() == 3);
    uint32_t num_blocks = blocked_matrix_shape[0];

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
    // TODO: Not deleting "dtx_right" causes memory leak. Fix it.
    // Cannot delete "dtx_right" because it contains raw pointers that are shared with the "combined" object.
    // Fix it by adding smart pointers and use move semantics to transfer ownership from "dtx_right" to "combined"
    //delete dtx_right;
    delete dtx_left;

    // Generate address map for reader kernel
    // copy transfer addresses into a vector
    std::vector<uint32_t> address_map;
    uint32_t t_bytes = 0;
    assert(combined->transformations.size() == 2 && "DTX transformations not collapsed.");
    assert(combined->transformations.back()->groups[0]->transfers.size() > 0 && "Addresses not generated in DTX.");
    uint32_t block_size_bytes = block_shape[0] * block_shape[1] * num_bytes_of_df;
    uint32_t b_bytes = 0;
    uint32_t n_blocks = 0;
    for(auto transfer : combined->transformations.back()->groups[0]->transfers){
        // Reads must be 32 byte aligned
        assert(transfer->size*num_bytes_of_df % 32 == 0);
        assert(transfer->src_address*num_bytes_of_df % 32 == 0);
        assert(transfer->dst_address*num_bytes_of_df % 32 == 0);
        address_map.push_back(transfer->src_address*num_bytes_of_df);
        address_map.push_back(transfer->dst_address*num_bytes_of_df);
        address_map.push_back(transfer->size*num_bytes_of_df);
        address_map.push_back(transfer->pad);

        t_bytes += transfer->size*num_bytes_of_df;
        b_bytes += transfer->size*num_bytes_of_df;
        if(b_bytes == block_size_bytes) {
            b_bytes = 0;
            n_blocks++;
        }
    }
    uint32_t total_bytes = num_rows * num_cols * num_bytes_of_df;
    assert(b_bytes == 0);
    assert(n_blocks == num_blocks);
    assert(total_bytes == t_bytes);
    assert(total_bytes % n_blocks == 0);
    uint32_t in0_block_size_bytes = total_bytes / n_blocks;
    assert(in0_block_size_bytes == block_size_bytes);
    delete combined;
    return address_map;
}
