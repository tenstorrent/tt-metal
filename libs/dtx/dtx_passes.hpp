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

#include "dtx.hpp"
using namespace std;

// ========================================================
//                      Helper Functions
// ========================================================
vector<vector<vector<int>>> generate_sliced_ranges(vector<int> shape, vector<int> slice_factors);
vector<vector<int>> generate_list_of_cores_based_on_range(vector<int> cores_start, vector<int> cores_end);
void tilize_into_row_major_order(vector<int> shape);
vector<vector<int>> dim_order_counting(vector<int> shape, vector<int> dim_order);



// ========================================================
//                 PART 1: INFRA DTX PASSES
// ========================================================

// WORKS, WELL TESTED - Collapse all the transformations down to 1. Required step before generating transfer addresses
bool collapse_transformations(DataTransformations * dtx);               // TO DO: rename to "merge_transformations"

// WORKS, WELL TESTED, (missing golden check) - Reverse the transformations
DataTransformations * reverse_transformations(DataTransformations * forward);

DataTransformations * reverse_and_combine_transformations(DataTransformations * dtx_left, DataTransformations * dtx_right);

bool optimize_away_transpose(DataTransformations * dtx);

// Generates src/dst addresses for data transfers for 1D transformation
bool generate_transfer_addresses(DataTransformations * dtx);

// Generates src/dst addresses for data transfers for 2d blocked data (like tiled data)
bool generate_transfer_addresses_blocked_data(DataTransformations * dtx);

// OBSOLETE: Generates src/dst addresses for data transfers for tiled data transformations
bool generate_transfer_addresses_tiled_data(DataTransformations * dtx);

// ========================================================
//                 PART 2: TENSOR LAYOUTs
//
// Terminology:
//     RM - row major
//     Tile - Tiled, 32x32
//     CL - ChannelsLast
// ========================================================

bool row_major_memory_store(DataTransformations * dtx);

bool row_major_memory_store_blocks(DataTransformations * dtx);

// Slice into tiles (32x32) - WORKS?
bool tilize_and_store(DataTransformations * dtx, vector<int> dim_order);

bool pad_2d_matrix(DataTransformations * dtx, vector<int> pad_to_nearest);

bool block_2d_matrix(DataTransformations * dtx, vector<int> dim_order, vector<int> block_shape_yx);

bool block_2d_with_duplicate_blocks(DataTransformations * dtx, vector<int> dim_order, vector<int> block_shape_yx, int num_duplicates, int dim_to_duplicate);

// expects only 1 producer group and generates groups for the outermost dimension.
bool generate_groups_outermost_dim(DataTransformations * dtx);

// Slice into tiles and store into row-major, col-major, or any other dim order - IS THIS NOW OBSOLETE?
bool slice_into_tiles_and_store(DataTransformations * dtx, vector<int> dim_order);

bool convert_tensor_layout_3d_conv_act_to_2Dmatrix(DataTransformations * dtx, vector<int> conv_params);
bool convert_abstract_tensor_to_channels_last_layout(DataTransformations * dtx);

// Convert from a particular layout, stored in 1 place (ex, CPU), to the same layout, stored in 8 places (ex. device DRAM), with sharding
bool convert_tensor_layout_CL1_to_CL8(DataTransformations * dtx);       // need for CNN bring up
bool convert_tensor_layout_RM1_to_RM8(DataTransformations * dtx);
bool convert_tensor_layout_Tile1_to_Tile8(DataTransformations * dtx);

// ========================================================
//             PART 3: PARALLELIZATION & SLICING
// ========================================================

// Generic slicing of a tensor, based on a slice factor for each dim - WORKS?
bool parallelize_generic_tensor_slice(DataTransformations * dtx, vector<int> slice_factors, vector<int> cores_start, vector<int> cores_end);


// ========================================================
//             PART 4: TENSOR MANIPULATIONS (TMs)
// ========================================================

// For stress testing purposes (not a real OP)
bool random_tile_reshuffle(DataTransformations * dtx);

// Pytorch reshape op
bool reshape(DataTransformations * dtx, vector<int> reshaped_tensor);

// For full Pytorch transposeXY, this needs to be paired with a TransposeXY within the Math Engine
bool transpose_xy_of_tiles(DataTransformations * dtx);

// Pytorch permute op (with some limitations, can not permute with X-dim, this requires a decomposition into TransposeXY+TransposeY?)
bool permute(DataTransformations * dtx, vector<int> permute_dims);

// An abstract Transpose XY, which neds to be canceled out with other TransposeXY transformations,
// or explicitly executed on the device.
bool transpose_xy(DataTransformations * dtx);

bool transpose_yz(DataTransformations * dtx);

bool flatten_xy(DataTransformations * dtx);

// ========================================================
//             PART 5: CONVOLUTIONs
// ========================================================

// Stored in 1 place
bool convert_tensor_layout_rowmajor_2_channelslast(DataTransformations * dtx);



// Check if everything is in tiles

// Find the lowest common dominator (sub-tensor), what's the biggest sub-tensor that we can transfer, or deal with

// Coalless addresses


// ========================================================
//             PART 6: HIGH LEVEL PASSES
// ========================================================
vector<uint32_t> generate_address_map(DataTransformations * dtx, bool in_bytes=false, uint32_t num_df_bytes=0);
vector<vector<float>> evaluate(vector<float> data, vector<uint32_t> address_map, vector<vector<int>> output_shape);
DataTransformations * simple_high_level_pass(vector<int> shape);

std::pair<vector<uint32_t>, vector<uint32_t>> conv_transform(vector<int> activation_shape,
                                        vector<int> weight_shape,
                                        vector<int> conv_params,
                                        uint32_t act_block_h,
                                        uint32_t act_block_w,
                                        uint32_t weight_block_w,
                                        uint32_t num_blocks_act_h,
                                        uint32_t num_blocks_weight_w,
                                        uint32_t num_bytes_of_df);
