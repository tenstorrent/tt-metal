#include "tt_metal/host_api.hpp"
#include "dtx.hpp"
#include "dtx_passes.hpp"
#include "util.hpp"
#include "util_vector_of_ints.hpp"
#include <fstream>

string get_dtx_output_file_name_suffix(vector<int> activation_shape, vector<int> weight_shape, vector<int> conv_params, vector<int> act_block_shape_yx, vector<int> weight_block_shape_yx, uint32_t num_bytes_of_df) {
    assert(activation_shape.size() == 3);
    assert(weight_shape.size() == 4);
    assert(conv_params.size() == 6);
    assert(act_block_shape_yx.size() == 2);
    assert(weight_block_shape_yx.size() == 2);
    int K = weight_shape[0];
    int C = activation_shape[0];
    int H = activation_shape[1];
    int W = activation_shape[2];
    int kernel_window_h = conv_params[0];
    int kernel_window_w = conv_params[1];
    int stride_h = conv_params[2];
    int stride_w = conv_params[3];
    int pad_h = conv_params[4];
    int pad_w = conv_params[5];
    assert(weight_shape[1] == C);
    assert(weight_shape[2] == kernel_window_h);
    assert(weight_shape[3] == kernel_window_w);
    string dtx_output_file_name = "actShape_1_" + to_string(C) + "_" + to_string(H) + "_" + to_string(W);
    dtx_output_file_name += "_weightsShape_" + to_string(K) + "_" + to_string(C) + "_" + to_string(kernel_window_h) + "_" + to_string(kernel_window_w);
    dtx_output_file_name += "_kernelWindow_" + to_string(kernel_window_h) + "_" + to_string(kernel_window_w);
    dtx_output_file_name += "_stride_" + to_string(stride_h) + "_" + to_string(stride_w);
    dtx_output_file_name += "_padding_" + to_string(pad_h) + "_" + to_string(pad_w);
    dtx_output_file_name += "_actblockShape_" + to_string(act_block_shape_yx[0]) + "_" + to_string(act_block_shape_yx[1]);
    dtx_output_file_name += "_weightblockShape_" + to_string(weight_block_shape_yx[0]) + "_" + to_string(weight_block_shape_yx[1]);
    dtx_output_file_name += "_numBytes_" + to_string(num_bytes_of_df);
    dtx_output_file_name += ".txt";
    return dtx_output_file_name;
}

vector<uint32_t> read_address_map_from_file(std::ifstream& file) {
    assert(file.is_open());
    std::cout << "DTX transform output file already exists! No need to run transform pass. Read address map." << std::endl;
    std::vector<uint32_t> address_map;
    std::string line;
    while (std::getline(file, line)) {
        address_map.push_back(static_cast<uint32_t>(std::stoul(line)));
    }
    file.close();
    return address_map;
}

vector<uint32_t> conv_act_transform(vector<int> activation_shape, vector<int> conv_params, vector<int> act_block_shape_yx, int num_blocks_weight_w, uint32_t num_bytes_of_df) {
    assert(activation_shape.size() == 3);
    assert(conv_params.size() == 6);
    auto activation_C = activation_shape[1];

    // Right side: AbstractTensor --> consumer conv/mm
    DataTransformations * dtx_right = new DataTransformations();
    TransformationNode * node0 = new TransformationNode("producer", 1);
    node0->groups[0]->shape = activation_shape;
    dtx_right->transformations.push_back(node0);
    bool pass = true;
    pass &= convert_tensor_layout_3d_conv_act_to_2Dmatrix(dtx_right, conv_params);
    pass &= pad_2d_matrix(dtx_right, act_block_shape_yx);
    // Get the 2d matrix shape
    auto matrix_shape = dtx_right->transformations.back()->groups[0]->shape;
    assert(matrix_shape.size() == 3);
    assert(matrix_shape[0] == 1);
    uint32_t num_rows = (uint32_t) matrix_shape[1];
    uint32_t num_cols = (uint32_t) matrix_shape[2];
    auto dim_order = {0,1,2};
    pass &= block_2d_with_duplicate_blocks(dtx_right, dim_order, act_block_shape_yx, num_blocks_weight_w, 1);
    auto blocked_matrix_shape = dtx_right->transformations.back()->groups[0]->shape;
    assert(blocked_matrix_shape.size() == 3);
    uint32_t num_blocks = blocked_matrix_shape[0];
    pass &= generate_groups_outermost_dim(dtx_right);
    pass &= row_major_memory_store(dtx_right);

    //cout << "\n\nDTX_RIGHT" << endl;
    //dtx_right->print();

    // Left side: AbstractTensor --> producer memory buffer
    DataTransformations * dtx_left = new DataTransformations();
    TransformationNode * node1 = new TransformationNode("producer", 1);
    node1->groups[0]->shape = activation_shape;
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
    std::vector<uint32_t> address_map = generate_address_map(combined, true, num_bytes_of_df);

    // validate dram reads are 32 byte aligned (i.e. dram src address, and l1 dst address % 32 == 0)
    // validate padding boolean values in address map
    uint32_t address_map_index = 0;
    uint32_t num_groups = address_map[address_map_index];
    address_map_index++;
    assert(num_groups > 0);
    for(uint32_t g = 0; g < num_groups; g++) {
        assert(address_map.size() > address_map_index);
        uint32_t address_map_this_group_size = address_map[address_map_index];
        address_map_index++;
        assert(address_map.size() >= address_map_index+address_map_this_group_size);
        for(uint32_t i = 0; i < address_map_this_group_size; i+=4) {
            assert(address_map[address_map_index] % 32 == 0); // src address
            assert(address_map[address_map_index+1] % 32 == 0); // dst address
            assert(address_map[address_map_index+2] % 32 == 0); // size
            assert(address_map[address_map_index+3] == 0 || address_map[address_map_index+3] == 1); // pad
            address_map_index += 4;
        }
    }

    delete combined;
    return address_map;
}

vector<uint32_t> conv_weight_transform(vector<int> weight_shape, vector<int> conv_params, vector<int> weight_block_shape_yx, int num_blocks_act_h, uint32_t num_bytes_of_df) {
    // Validate weight shape and conv params
    assert(weight_shape.size() == 4);
    assert(conv_params.size() == 6);
    assert(conv_params[0] == weight_shape[2]);
    assert(conv_params[1] == weight_shape[3]);
    vector<int> conv_as_mm_weight_shape = {1,
                nearest_to(weight_shape[1] * weight_shape[2] * weight_shape[3], weight_block_shape_yx[0]),
                nearest_to(weight_shape[0], weight_block_shape_yx[1])};
    // Validate that the weight is padded to tile size (32x32)
    assert(conv_as_mm_weight_shape[1] % 32 == 0);
    assert(conv_as_mm_weight_shape[2] % 32 == 0);
    // Validate that weight block shape is divisible by tile size
    assert(weight_block_shape_yx[0] % 32 == 0);
    assert(weight_block_shape_yx[1] % 32 == 0);
    // Right side: AbstractTensor --> consumer conv/mm reader kernel output
    DataTransformations * dtx_right = new DataTransformations();
    TransformationNode * node0 = new TransformationNode("producer", 1);
    node0->groups[0]->shape = conv_as_mm_weight_shape;
    dtx_right->transformations.push_back(node0);
    bool pass = true;
    // Get the 2d matrix shape
    auto matrix_shape = dtx_right->transformations.back()->groups[0]->shape;
    assert(matrix_shape.size() == 3);
    assert(matrix_shape[0] == 1);
    uint32_t num_rows = (uint32_t) matrix_shape[1];
    uint32_t num_cols = (uint32_t) matrix_shape[2];
    auto dim_order = {0,2,1}; // blocks in col major order
    pass &= block_2d_with_duplicate_blocks(dtx_right, dim_order, weight_block_shape_yx, num_blocks_act_h, 0);
    auto blocked_matrix_shape = dtx_right->transformations.back()->groups[0]->shape;
    assert(blocked_matrix_shape.size() == 3);
    uint32_t num_blocks = blocked_matrix_shape[0];
    pass &= generate_groups_outermost_dim(dtx_right);
    // tilize the block
    vector<int> tilize_dim_order = {0,1,2}; // tiles in row major within block
    pass &= block_2d_matrix(dtx_right, tilize_dim_order, {32,32});

    //cout << "\n\nDTX_RIGHT" << endl;
    //dtx_right->print();

    // Left side: AbstractTensor --> conv weight in dram (producer memory buffer)
    DataTransformations * dtx_left = new DataTransformations();
    TransformationNode * node1 = new TransformationNode("producer", 1);
    node1->groups[0]->shape = conv_as_mm_weight_shape;
    dtx_left->transformations.push_back(node1);
    tilize_dim_order = {0,1,2}; // tiles in row major order
    pass &= block_2d_matrix(dtx_left, tilize_dim_order, {32,32});

    //cout << "\n\nDTX_LEFT" << endl;
    //dtx_left->print();

    DataTransformations * combined = reverse_and_combine_transformations(dtx_left, dtx_right);
    //cout << "\n\nDTX_COMBINED" << endl;
    //combined->print();
    pass &= collapse_transformations(combined);
    //cout << "\n\nDTX_COLLAPSED" << endl;
    //combined->print();
    pass &= generate_transfer_addresses_blocked_data(combined);
    //combined->print();
    // TODO Issue #673: Not deleting "dtx_right" causes memory leak. Fix it.
    // Cannot delete "dtx_right" because it contains raw pointers that are shared with the "combined" object.
    // Fix it by adding smart pointers and use move semantics to transfer ownership from "dtx_right" to "combined"
    //delete dtx_right;
    delete dtx_left;

    // Generate address map for reader kernel
    std::vector<uint32_t> address_map = generate_address_map(combined, true, num_bytes_of_df);

    // validate dram reads are 32 byte aligned (i.e. dram src address, and l1 dst address % 32 == 0)
    // validate padding boolean values in address map
    uint32_t address_map_index = 0;
    uint32_t num_groups = address_map[address_map_index];
    address_map_index++;
    assert(num_groups > 0);
    for(uint32_t g = 0; g < num_groups; g++) {
        assert(address_map.size() > address_map_index);
        uint32_t address_map_this_group_size = address_map[address_map_index];
        address_map_index++;
        assert(address_map.size() >= address_map_index+address_map_this_group_size);
        for(uint32_t i = 0; i < address_map_this_group_size; i+=4) {
            assert(address_map[address_map_index] % 32 == 0); // src address
            assert(address_map[address_map_index+1] % 32 == 0); // dst address
            assert(address_map[address_map_index+2] % 32 == 0); // size
            assert(address_map[address_map_index+3] == 0 || address_map[address_map_index+3] == 1); // pad
            address_map_index += 4;
        }
    }
    delete combined;
    return address_map;
}

std::pair<vector<uint32_t>, vector<uint32_t>> conv_transform(vector<int> activation_shape,
                                        vector<int> weight_shape,
                                        vector<int> conv_params,
                                        uint32_t act_block_h,
                                        uint32_t act_block_w,
                                        uint32_t weight_block_w,
                                        uint32_t num_blocks_act_h,
                                        uint32_t num_blocks_weight_w,
                                        uint32_t num_bytes_of_df) {
    assert(activation_shape.size() == 3);
    assert(weight_shape.size() == 4);
    assert(conv_params.size() == 6);
    bool enable_dtx_caching = false;
    std::pair<vector<uint32_t>, vector<uint32_t>> conv_act_and_weight_address_maps;
    vector<int> act_block_shape_yx = {(int)act_block_h, (int)act_block_w};
    vector<int> weight_block_shape_yx = {(int)act_block_w, (int)weight_block_w};
    // Generate dtx output file name for the given parameters and check if the file already exists
    string dtx_output_file_name_suffix = get_dtx_output_file_name_suffix(activation_shape, weight_shape, conv_params, act_block_shape_yx, weight_block_shape_yx, num_bytes_of_df);
    string dtx_conv_act_file_name = "Conv_activation_tranform_" + dtx_output_file_name_suffix;
    string dtx_conv_weight_file_name = "Conv_weight_transform_" + dtx_output_file_name_suffix;
    assert(std::getenv("TT_METAL_HOME"));
    string dtx_conv_act_file_path = ((string) std::getenv("TT_METAL_HOME")) + "/tt_metal/third_party/lfs/dtx_transform_outputs/" + dtx_conv_act_file_name;
    string dtx_conv_weight_file_path = ((string) std::getenv("TT_METAL_HOME")) + "/tt_metal/third_party/lfs/dtx_transform_outputs/" + dtx_conv_weight_file_name;
    std::ifstream dtx_conv_act_file(dtx_conv_act_file_path);
    std::ifstream dtx_conv_weight_file(dtx_conv_weight_file_path);
    if(enable_dtx_caching && dtx_conv_act_file.is_open() && dtx_conv_weight_file.is_open()) {
        conv_act_and_weight_address_maps.first = read_address_map_from_file(dtx_conv_act_file);
        conv_act_and_weight_address_maps.second = read_address_map_from_file(dtx_conv_weight_file);
        return conv_act_and_weight_address_maps;
    }
    conv_act_and_weight_address_maps.first = conv_act_transform(activation_shape, conv_params, act_block_shape_yx, num_blocks_weight_w, num_bytes_of_df);
    conv_act_and_weight_address_maps.second = conv_weight_transform(weight_shape, conv_params, weight_block_shape_yx, num_blocks_act_h, num_bytes_of_df);

    // Save and cache dtx transform outputs to file
    if(enable_dtx_caching) {
        std::ofstream conv_act_file(dtx_conv_act_file_path);
        assert(conv_act_file.is_open());
        for (const auto &v : conv_act_and_weight_address_maps.first) conv_act_file << to_string(v) << "\n";
        std::ofstream conv_weight_file(dtx_conv_weight_file_path);
        assert(conv_weight_file.is_open());
        for (const auto &v : conv_act_and_weight_address_maps.second) conv_weight_file << to_string(v) << "\n";
    }
    return conv_act_and_weight_address_maps;
}
