#include "dtx.hpp"
#include "dtx_passes.hpp"
#include "util.hpp"
#include "util_vector_of_ints.hpp"
#include <fstream>

string get_dtx_output_file_name(vector<int> activation_shape, vector<int> conv_params, vector<int> block_shape_yx, uint32_t num_bytes_of_df) {
    assert(activation_shape.size() == 3);
    assert(conv_params.size() == 6);
    assert(block_shape_yx.size() == 2);
    int C = activation_shape[0];
    int H = activation_shape[1];
    int W = activation_shape[2];
    int kernel_window_h = conv_params[0];
    int kernel_window_w = conv_params[1];
    int stride_h = conv_params[2];
    int stride_w = conv_params[3];
    int pad_h = conv_params[4];
    int pad_w = conv_params[5];
    vector<int> weight_shape = {-1, C, H, W};
    string dtx_output_file_name = "actShape_1_" + to_string(C) + "_" + to_string(H) + "_" + to_string(W);
    dtx_output_file_name += "_weightsShape_X_" + to_string(C) + "_" + to_string(kernel_window_h) + "_" + to_string(kernel_window_w);
    dtx_output_file_name += "_kernelWindow_" + to_string(kernel_window_h) + "_" + to_string(kernel_window_w);
    dtx_output_file_name += "_stride_" + to_string(stride_h) + "_" + to_string(stride_w);
    dtx_output_file_name += "_padding_" + to_string(pad_h) + "_" + to_string(pad_w);
    dtx_output_file_name += "_blockSize_" + to_string(block_shape_yx[0]) + "_" + to_string(block_shape_yx[1]);
    dtx_output_file_name += "_numBytes_" + to_string(num_bytes_of_df);
    dtx_output_file_name += ".txt";
    return dtx_output_file_name;
}

std::vector<uint32_t> conv_transform(vector<int> shape, vector<int> conv_params, std::pair<vector<int>,vector<int>> block_info, uint32_t num_bytes_of_df) {
    assert(shape.size() == 3);
    assert(conv_params.size() == 6);
    auto dim_order = block_info.first;
    auto block_shape = block_info.second;
    assert(block_shape.size() == 2);
    // Generate dtx output file name for the given parameters and check if the file already exists
    string dtx_output_file_name = get_dtx_output_file_name(shape, conv_params, block_shape, num_bytes_of_df);
    assert(std::getenv("TT_METAL_HOME"));
    string dtx_file_path = ((string) std::getenv("TT_METAL_HOME")) + "/tt_metal/third_party/lfs/dtx_transform_outputs/" + dtx_output_file_name;
    std::ifstream file(dtx_file_path);
    if (file.is_open()) {
        std::cout << "DTX transform output file already exists! No need to run transform pass. Read address map." << std::endl;
        std::vector<uint32_t> address_map;
        std::string line;
        while (std::getline(file, line)) {
            address_map.push_back(static_cast<uint32_t>(std::stoul(line)));
        }
        file.close();
        return address_map;
    }
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
    // Save and cache dtx transform output to file
    std::ofstream outFile(dtx_file_path);
    assert(outFile.is_open());
    for (const auto &v : address_map) outFile << to_string(v) << "\n";
    return address_map;
}
