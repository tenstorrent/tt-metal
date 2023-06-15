#include "tt_dnn/op_library/conv/conv_op.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
// #include "test/tt_metal/llrt/test_libs/debug_mailbox.hpp"
#include "libs/dtx/dtx.hpp"
#include "libs/dtx/dtx_passes.hpp"
#include "llrt/tt_debug_print_server.hpp"

using namespace tt::constants;
namespace tt {

namespace tt_metal {

void create_CBs_for_fused_matmul_new_alloc(tt_metal::Program &program,
                                tt_metal::Device* device,
                                CoreRange core,
                                uint32_t act_block_size,
                                uint32_t weight_block_size,
                                uint32_t output_block_size,
                                uint32_t reblock_size,
                                uint32_t num_bytes_for_df,
                                bool untilize_out) {
    uint32_t in0_cb                                   = 0;
    uint32_t in1_cb                                   = 1;
    uint32_t tilize_mode_tilized_in0_cb               = 24;
    uint32_t matmul_partials_cb                       = 25;
    uint32_t untilize_mode_final_matmul_partials_cb   = 26;
    uint32_t untilize_mode_reblock_cb                 = 27;
    uint32_t out0_cb                                  = 16;

    uint32_t single_tile_size = num_bytes_for_df * 1024;

    uint32_t num_output_tiles = output_block_size;

    // Invariants
    uint32_t cb0_tiles = act_block_size;
    auto cb_in0 = tt_metal::CreateCircularBuffers(
        program,
        device,
        in0_cb,
        core,
        cb0_tiles,
        cb0_tiles * single_tile_size,
        tt::DataFormat::Float16_b
    );

    uint32_t cb1_tiles = weight_block_size;
    auto cb_in1 = tt_metal::CreateCircularBuffers(
        program,
        device,
        in1_cb,
        core,
        cb1_tiles,
        cb1_tiles * single_tile_size,
        tt::DataFormat::Float16_b
    );

    // Used for placing tilized activations
    auto cb_src0_tilized = tt_metal::CreateCircularBuffers(
        program,
        device,
        tilize_mode_tilized_in0_cb,
        core,
        cb0_tiles,
        cb0_tiles * single_tile_size,
        tt::DataFormat::Float16_b
    );
    if(untilize_out) {
        auto cb_matmul_partials = tt_metal::CreateCircularBuffers(
            program,
            device,
            matmul_partials_cb,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            tt::DataFormat::Float16_b
        );

        // Shares same address space as matmul partials
        auto cb_final_matmul_partials = tt_metal::CreateCircularBuffers(
            program,
            device,
            untilize_mode_final_matmul_partials_cb,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            tt::DataFormat::Float16_b
        );

        // Supposed to be a small CB only responsible for reorganizing
        // the output blocks to fill the whole "per core output block width"
        uint32_t reblock_cb_tiles = reblock_size; // Only space for one row
        auto cb_reblock = tt_metal::CreateCircularBuffers(
            program,
            device,
            untilize_mode_reblock_cb,
            core,
            reblock_cb_tiles,
            reblock_cb_tiles * single_tile_size,
            tt::DataFormat::Float16_b
        );

        auto cb_output = tt_metal::CreateCircularBuffers(
            program,
            device,
            out0_cb,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            tt::DataFormat::Float16_b
        );
    }
    else {

        auto cb_matmul_partials = tt_metal::CreateCircularBuffers(
            program,
            device,
            matmul_partials_cb,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            tt::DataFormat::Float16_b
        );

        auto cb_matmul_partials_addr = cb_matmul_partials->address();

        auto cb_output = tt_metal::CreateCircularBuffers(
            program,
            device,
            out0_cb,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            cb_matmul_partials_addr,
            tt::DataFormat::Float16_b
        );

    }
}

Tensor create_output_dram_buffer_(Device * device, DataType data_type, std::array<uint32_t, 4> cshape, bool untilize_out) {
    tt::tt_metal::Layout out_layout;
    tt_metal::Tensor output = tt_metal::Tensor(
        cshape,
        data_type,
        untilize_out ? tt::tt_metal::Layout::ROW_MAJOR : tt::tt_metal::Layout::TILE,
        device);

    return output;
}

std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> compute_conv_op_block_info(uint32_t M, uint32_t K, uint32_t N) {
    uint32_t single_tile_size_bytes = 2 * 1024;
    std::string report_string = "";

    // Constraint 1: in0 and in1 should fit in L1. If not, slice into blocks
    // Harcoded max sizes that can fit all CBs into 1MB - reserved KB L1 space
    // TODO: remove hardcoded values and calculate them.
    uint32_t max_in0_block_bytes = 50 * 1024;
    uint32_t max_in1_block_bytes = 50 * 1024;
    uint32_t max_in0_block_tiles = max_in0_block_bytes / single_tile_size_bytes;
    uint32_t max_in1_block_tiles = max_in1_block_bytes / single_tile_size_bytes;
    uint32_t num_blocks_in0_w = 1; // = number of blocks of in1 height
    uint32_t num_blocks_in0_h = 1;
    uint32_t in0_block_w = K; // = in1 block height
    uint32_t in0_block_h = M;
    if(M*K > max_in0_block_tiles) {
        // block width first
        while(in0_block_w*in0_block_h > max_in0_block_tiles && in0_block_w != 1) {
            num_blocks_in0_w += 1;
            while(K%num_blocks_in0_w != 0) {
                num_blocks_in0_w += 1;
            }
            assert(num_blocks_in0_w <= K);
            in0_block_w = K / num_blocks_in0_w;
        }
        // block in height dimension if still doesn't fit
        while(in0_block_w*in0_block_h > max_in0_block_tiles) {
            num_blocks_in0_h += 1;
            while(M%num_blocks_in0_h != 0) {
                num_blocks_in0_h += 1;
            }
            assert(num_blocks_in0_h <= M);
            in0_block_h = M / num_blocks_in0_h;
        }
    }
    // output block width = in1 block width
    // output block height = in0 block height
    // Constraint 2.1: output block width should fit in L1 CB (reblock for untilize)
    uint32_t max_n_reblock_bytes = 20 * 1024;
    uint32_t max_n_reblock_tiles = max_n_reblock_bytes / single_tile_size_bytes;
    log_debug(tt::LogOp, "max_out_reblock_tiles: {}", max_n_reblock_tiles);
    uint32_t in1_block_w = N;
    uint32_t num_blocks_in1_w = 1;
    if(in1_block_w > max_n_reblock_tiles) {
        // block the weight in width dimension
        while(in1_block_w > max_n_reblock_tiles) {
            num_blocks_in1_w += 1;
            while(N%num_blocks_in1_w != 0) {
                num_blocks_in1_w += 1;
            }
            assert(num_blocks_in1_w <= N);
            in1_block_w = N / num_blocks_in1_w;
        }
    }

    // output block width = in1 block width
    // output block height = in0 block height
    // Constraint 2.2: output block height * output width should fit in L1
    uint32_t max_out_block_bytes = 120 * 1024;
    uint32_t max_out_block_tiles = max_out_block_bytes / single_tile_size_bytes;
    log_debug(tt::LogOp, "max_out_block_tiles: {}", max_out_block_tiles);
    if(in0_block_h*in1_block_w > max_out_block_tiles) {
        // output block does not fit in L1
        // reduce the in0 block height
        while(in0_block_h*in1_block_w > max_out_block_tiles && in0_block_h != 1) {
            num_blocks_in0_h += 1;
            while(M%num_blocks_in0_h != 0) {
                num_blocks_in0_h += 1;
            }
            assert(num_blocks_in0_h <= M);
            in0_block_h = M / num_blocks_in0_h;
        }
        // if output block still doesn't fit after reducing in0 block height to 1,
        // reduce the in1 block width
        while(in0_block_h*in1_block_w > max_out_block_tiles) {
            num_blocks_in1_w += 1;
            while(N%num_blocks_in1_w != 0) {
                num_blocks_in1_w += 1;
            }
            assert(num_blocks_in1_w <= N);
            in1_block_w = N / num_blocks_in1_w;
        }
    }
    log_debug(tt::LogOp, "num_blocks_in0_h: {}", num_blocks_in0_h);
    log_debug(tt::LogOp, "num_blocks_in0_w: {}", num_blocks_in0_w);
    log_debug(tt::LogOp, "num_blocks_in1_w: {}", num_blocks_in1_w);

    // Constraint 3: output block (in0_block_h * in1_block_w) should should fit in half DST (8 tiles). If not, divide into output sublocks
    uint32_t out_subblock_h = in0_block_h;
    uint32_t out_subblock_w = in1_block_w;
    uint32_t num_out_subblocks_h = 1;
    uint32_t num_out_subblocks_w = 1;
    bool divide_h_next = true;
    while (out_subblock_h*out_subblock_w > 8) {
        if (divide_h_next) {
            if(num_out_subblocks_h < in0_block_h) {
                num_out_subblocks_h += 1;
                while(in0_block_h % num_out_subblocks_h != 0) {
                    num_out_subblocks_h += 1;
                }
            }
            out_subblock_h = in0_block_h / num_out_subblocks_h;
            divide_h_next = false;
        }
        else {
            if(num_out_subblocks_w < in1_block_w) {
                num_out_subblocks_w += 1;
                while(in1_block_w % num_out_subblocks_w != 0) {
                    num_out_subblocks_w += 1;
                }
            }
            out_subblock_w = in1_block_w / num_out_subblocks_w;
            divide_h_next = true;
        }
    }
    log_debug(tt::LogOp, "out_subblock_h: {}", out_subblock_h);
    log_debug(tt::LogOp, "out_subblock_w: {}", out_subblock_w);
    return std::make_tuple(num_blocks_in0_h, num_blocks_in0_w, num_blocks_in1_w, out_subblock_h, out_subblock_w);
}

vector<uint32_t> compute_conv_as_mm_shape(vector<int> shape, const std::vector<int>& conv_params, uint32_t in0_block_h, uint32_t in0_block_w) {
    int conv_input_x = shape[2];
    int conv_input_y = shape[1];
    int conv_input_z = shape[0];
    int R = conv_params[0];
    int S = conv_params[1];
    int U = conv_params[2];
    int V = conv_params[3];
    int Pad_H = conv_params[4];
    int Pad_W = conv_params[5];
    int conv_output_h = ((conv_input_x - R + (2 * Pad_H)) / U) + 1;
    int conv_output_w = ((conv_input_y - S + (2 * Pad_W)) / V) + 1;
    {
        log_debug(tt::LogOp, "conv_input_x: {}", conv_input_x);
        log_debug(tt::LogOp, "conv_input_y: {}", conv_input_y);
        log_debug(tt::LogOp, "conv_input_z: {}", conv_input_z);
        log_debug(tt::LogOp, "kernel_size_y: {}", R);
        log_debug(tt::LogOp, "kernel_size_x: {}", S);
        log_debug(tt::LogOp, "stride_y: {}", U);
        log_debug(tt::LogOp, "stride_x: {}", V);
        log_debug(tt::LogOp, "pad_y: {}", Pad_H);
        log_debug(tt::LogOp, "pad_x: {}", Pad_W);
        log_debug(tt::LogOp, "conv_output_x: {}", conv_output_w);
        log_debug(tt::LogOp, "conv_output_y: {}", conv_output_h);
    }
    // pad height
    uint32_t num_rows = (uint32_t) conv_output_h*conv_output_w;
    uint32_t in0_block_h_datums = in0_block_h * TILE_HEIGHT;
    uint32_t num_rows_padded = (uint32_t) (ceil((double) num_rows / (double) in0_block_h_datums ) * in0_block_h_datums);
    uint32_t num_cols = conv_input_z*R*S;
    uint32_t in0_block_w_datums = in0_block_w * TILE_WIDTH;
    uint32_t num_cols_padded = (uint32_t) (ceil((double) num_cols / (double) in0_block_w_datums ) * in0_block_w_datums);
    return {1,num_rows_padded, num_cols_padded};
}

std::pair<vector<uint32_t>, vector<uint32_t>> populate_address_map_vectors_for_reader_kernel(vector<uint32_t> address_map_raw) {
    // This function is called twice i.e., for activation and weight address maps
    // "address_map_raw" is the DTX address map vector returned from DTX "conv_transform" function.
    // "address_map_raw" contains metadata along with the address map data for all groups
    // To keep the reader kernel simple, the metadata is separated into a different buffer
    // So two buffers are created -
    // First buffer is in DRAM containing the address map for all groups
    //      This DRAM buffer is big and is streamed into L1 scratchpad
    // Second buffer contains the metadata and is copied to L1 from host
    // It contains number of groups in its first index, followed by group info for each group -
    //      1. dram read address offset of address map group in dram buffer (in bytes)
    //      2. size of address map group in dram buffer (in datums, not bytes)
    // TODO (nshanker), support for streaming the second buffer from dram if it does not fit in L1
    vector<uint32_t> address_map; // will be in dram
    vector<uint32_t> address_map_metadata; // will be in l1

    uint32_t num_address_map_fields_per_transfer = 4; // TODO (nshanker): remove hardcoded 4 and get this value from output of DTX
    uint32_t num_dtx_groups = address_map_raw[0];
    address_map_metadata.push_back(address_map_raw[0]);
    uint32_t address_map_raw_index = 1;
    uint32_t current_group_dram_address_offset = 0;
    for(uint32_t g = 0; g < num_dtx_groups; g++) {
        // insert group's dram read address (in bytes) in metadata buffer
        // Separate reads are issued for each "address map group"
        // DRAM reads should be 32B aligned
        assert(current_group_dram_address_offset%32 == 0);
        address_map_metadata.push_back(current_group_dram_address_offset);
        // insert group size (datums, not in bytes) into metadata buffer
        uint32_t current_group_size = address_map_raw[address_map_raw_index];
        address_map_metadata.push_back(current_group_size);
        address_map_raw_index += 1;
        // insert address map for this group into the address map buffer
        auto address_map_raw_current_group_start = address_map_raw.begin() + address_map_raw_index;
        address_map.insert(address_map.end(),
                                address_map_raw_current_group_start,
                                address_map_raw_current_group_start + current_group_size);
        address_map_raw_index += current_group_size;
        // Pad 0s in address map buffer to ensure each read address is 32B aligned (32/sizeof(uint32_t) == 8 elements)
        uint32_t current_group_size_padded = (uint32_t) (ceil((double) current_group_size / (double) 8) * 8);
        if(current_group_size_padded != current_group_size) {
            assert(current_group_size_padded > current_group_size);
            address_map.insert(address_map.end(), current_group_size_padded - current_group_size, 0);
        }
        // update next group's dram read address offset (in bytes)
        current_group_dram_address_offset += (current_group_size_padded*sizeof(uint32_t));
    }
    return make_pair(std::move(address_map), std::move(address_map_metadata));
}

Program conv_as_large_bmm_single_core_(const Tensor& a, const Tensor &b, vector<int> conv_params,
                                       uint32_t in0_block_h, uint32_t in0_block_w, uint32_t in1_block_w,
                                       uint32_t out_subblock_h, uint32_t out_subblock_w, bool untilize_out, Tensor &output) {
    bool pass = true;
    tt_metal::Device *device = a.device();
    TT_ASSERT(a.layout() == Layout::CHANNELS_LAST, "Conv activation should be in channels last layout");
    TT_ASSERT(a.shape()[0] == 1, "Only batch size 1 supported.");
    uint32_t num_bytes_of_df = 2; // 2 bytes for bfloat16
    uint32_t activation_C = a.shape()[1];
    //TT_ASSERT(activation_C % TILE_WIDTH == 0, "Channel depth must be divisible by tile width(32).");
    // Compute the 2d matrix shape
    vector<int> activation_shape = {(int)a.shape()[1], (int)a.shape()[2], (int)a.shape()[3]};    // TODO: Update types to use just one kind
    // Shape activation_shape_shape = {a.shape()[0], a.shape()[1], a.shape()[2], a.shape()[3]};
    auto matrix_shape = compute_conv_as_mm_shape(activation_shape, conv_params, in0_block_h, in0_block_w);
    assert(matrix_shape.size() == 3);
    assert(matrix_shape[0] == 1);
    uint32_t num_rows = (uint32_t) matrix_shape[1];
    uint32_t num_cols = (uint32_t) matrix_shape[2];

    // More Checks
    uint32_t Ba = 1;
    uint32_t Ca = 1;
    auto Ha = num_rows;
    auto Wa = num_cols;
    const auto [Bb, Cb, Hb, Wb] = b.shape();
    // Normal matrix shape checks
    TT_ASSERT(Ba == 1, "So far, large matmul op has only been tested for batch one.");
    TT_ASSERT(Ba == Bb, "Batch dimension needs to match");
    TT_ASSERT(Ca == Cb, "Channel dimension needs to match");
    TT_ASSERT(Wa == Hb, "The width of tensor a needs to match the height of tensor b");

    // Tile size divisibility checks
    TT_ASSERT(Ha % TILE_HEIGHT == 0, "Height of tensor a needs to be divisible by 32");
    TT_ASSERT(Wa % TILE_WIDTH == 0, "Width of tensor a needs to be divisible by 32");
    TT_ASSERT(Hb % TILE_HEIGHT == 0, "Height of tensor b needs to be divisible by 32");
    TT_ASSERT(Wb % TILE_WIDTH == 0, "Width of tensor b needs to be divisible by 32");

    // Device compatibility checks
    TT_ASSERT(not a.on_host() and not b.on_host(), "Operands to large matmul need to be on device!");
    TT_ASSERT(a.device() == b.device(), "Operands to large matmul need to be on the same device!");
    TT_ASSERT(a.buffer() != nullptr and b.buffer() != nullptr, "Operands to large matmul need to be allocated in buffers on device!");
    // Convert tensor dims to tile dims
    uint32_t B   = Ba;
    uint32_t Hat = Ha / TILE_HEIGHT;
    uint32_t Wat = Wa / TILE_WIDTH;
    uint32_t Wbt = Wb / TILE_WIDTH;
    log_debug(tt::LogOp, "Hat(MM Activation H in tiles): {}", Hat);
    log_debug(tt::LogOp, "Wat(MM Activation W (MM Weight H) in tiles): {}", Wat);
    log_debug(tt::LogOp, "Wbt(MM Weight W in tiles): {}", Wbt);
    // compute block size of input, weight and subblock size of the output
    //auto [num_blocks_in0_h, num_blocks_in0_w, num_blocks_in1_w, out_subblock_h, out_subblock_w] = compute_conv_op_block_info(Hat, Wat, Wbt);

    assert(Hat % in0_block_h == 0);
    assert(Wat % in0_block_w == 0);
    assert(Wbt % in1_block_w == 0);

    uint32_t num_blocks_in0_h = Hat / in0_block_h;
    uint32_t num_blocks_in0_w = Wat / in0_block_w;
    uint32_t num_blocks_in1_w = Wbt / in1_block_w;

    // in0 block info
    uint32_t in0_block_w_datums = Wa / num_blocks_in0_w;
    uint32_t in0_block_h_datums = Ha / num_blocks_in0_h;

    // in1 block info
    uint32_t in1_block_w_datums = Wb / num_blocks_in1_w;
    assert(in1_block_w % out_subblock_w == 0);
    uint32_t in1_num_subblocks = in1_block_w / out_subblock_w;
    uint32_t in1_block_h = in0_block_w;
    uint32_t in1_block_num_tiles = in1_block_w * in1_block_h;

    // DTX conv activation transform
    auto [act_address_map_raw, weight_address_map_raw]  = conv_transform(activation_shape, {(int) Wb, (int) activation_shape[0], (int) conv_params[0], (int) conv_params[1]}, conv_params,
                            in0_block_h_datums, in0_block_w_datums, in1_block_w_datums,
                            num_blocks_in0_h, num_blocks_in1_w, num_bytes_of_df);

    // sanity check
    uint32_t num_dtx_groups = act_address_map_raw[0];
    assert(weight_address_map_raw[0] == num_dtx_groups);

    auto [act_address_map, act_address_map_metadata] = populate_address_map_vectors_for_reader_kernel(act_address_map_raw);
    auto [weight_address_map, weight_address_map_metadata] = populate_address_map_vectors_for_reader_kernel(weight_address_map_raw);

    // debug prints
    int detailed_debug = 1;
    if(detailed_debug > 0) {
        log_debug(tt::LogOp, "Printing activation and weight address maps.");
        log_debug(tt::LogOp, "DTX groups: {}", num_dtx_groups);
        uint32_t act_metadata_index = 1;
        uint32_t weight_metadata_index = 1;
        uint32_t act_addr_map_index = 0;
        uint32_t weight_addr_map_index = 0;
        for(uint32_t g = 0; g < num_dtx_groups; g++) {
            log_debug(tt::LogOp, "  DTX group: {}", g);
            uint32_t act_current_group_address = act_address_map_metadata[act_metadata_index];
            act_metadata_index += 1;
            uint32_t act_current_group_size = act_address_map_metadata[act_metadata_index];
            act_metadata_index += 1;
            log_debug(tt::LogOp, "      act_current_group_address: {}", act_current_group_address);
            log_debug(tt::LogOp, "      act_current_group_size: {}", act_current_group_size);
            if(detailed_debug > 1) {
                uint32_t act_current_group_index = act_current_group_address/sizeof(uint32_t);
                for(uint32_t i = act_current_group_index; i < act_current_group_index + act_current_group_size; i+=4) {
                    log_debug(tt::LogOp, "          act_addr_map[0]: {}", act_address_map[i]);
                    log_debug(tt::LogOp, "          act_addr_map[1]: {}", act_address_map[i+1]);
                    log_debug(tt::LogOp, "          act_addr_map[2]: {}", act_address_map[i+2]);
                    log_debug(tt::LogOp, "          act_addr_map[3]: {}", act_address_map[i+3]);
                }
            }
            uint32_t weight_current_group_address = weight_address_map_metadata[weight_metadata_index];
            weight_metadata_index += 1;
            uint32_t weight_current_group_size = weight_address_map_metadata[weight_metadata_index];
            weight_metadata_index += 1;
            log_debug(tt::LogOp, "      weight_current_group_address: {}", weight_current_group_address);
            log_debug(tt::LogOp, "      weight_current_group_size: {}", weight_current_group_size);
            if(detailed_debug > 1) {
                uint32_t weight_current_group_index = weight_current_group_address/sizeof(uint32_t);
                for(uint32_t i = weight_current_group_index; i < weight_current_group_index + weight_current_group_size; i+=4) {
                    log_debug(tt::LogOp, "          weight_addr_map[0]: {}", weight_address_map[i]);
                    log_debug(tt::LogOp, "          weight_addr_map[1]: {}", weight_address_map[i+1]);
                    log_debug(tt::LogOp, "          weight_addr_map[2]: {}", weight_address_map[i+2]);
                    log_debug(tt::LogOp, "          weight_addr_map[3]: {}", weight_address_map[i+3]);
                }
            }
        }
    }

    uint32_t dram_bank_id = 0;
    auto act_address_map_buffer_size_in_dram = act_address_map.size() * sizeof(uint32_t);
    auto act_address_map_dram_buffer = tt_metal::Buffer(device, act_address_map_buffer_size_in_dram, dram_bank_id, act_address_map_buffer_size_in_dram, tt_metal::BufferType::DRAM);
    auto weight_address_map_buffer_size_in_dram = weight_address_map.size() * sizeof(uint32_t);
    auto weight_address_map_dram_buffer = tt_metal::Buffer(device, weight_address_map_buffer_size_in_dram, dram_bank_id, weight_address_map_buffer_size_in_dram, tt_metal::BufferType::DRAM);
    uint32_t act_address_map_dram_addr = act_address_map_dram_buffer.address();
    // DRAM to L1 writes should 32B aligned
    assert(act_address_map_dram_addr%32 == 0);
    auto act_dram_noc_xy = act_address_map_dram_buffer.noc_coordinates();
    uint32_t act_address_map_dram_noc_x = act_dram_noc_xy.x;
    uint32_t act_address_map_dram_noc_y = act_dram_noc_xy.y;
    uint32_t weight_address_map_dram_addr = weight_address_map_dram_buffer.address();
    // DRAM to L1 writes should 32B aligned
    assert(weight_address_map_dram_addr%32 == 0);
    auto weight_dram_noc_xy = weight_address_map_dram_buffer.noc_coordinates();
    uint32_t weight_address_map_dram_noc_x = weight_dram_noc_xy.x;
    uint32_t weight_address_map_dram_noc_y = weight_dram_noc_xy.y;

    // Write address maps to DRAM
    WriteToDeviceDRAMChannel(device, dram_bank_id, act_address_map_dram_addr, act_address_map);
    WriteToDeviceDRAMChannel(device, dram_bank_id, weight_address_map_dram_addr, weight_address_map);

    tt_metal::Program program = tt_metal::Program();
    CoreCoord core_coord = {0, 0};      // TODO: avoid another var here. Find a way to use core range instead.
    CoreRange core = {.start={0, 0}, .end={0, 0}};
    //tt_start_debug_print_server(a.device()->cluster(), {0}, {{1, 1}});

    uint32_t single_tile_size = num_bytes_of_df * TILE_HEIGHT * TILE_WIDTH;
    tt_metal::Buffer *src0_dram_buffer = a.buffer();
    tt_metal::Buffer *src1_dram_buffer = b.buffer();
    TT_ASSERT(src1_dram_buffer->size() % single_tile_size == 0, "Buffer size of tensor b must be divisible by single_tile_size (aka divisible by sizeof(df) * 1024)");

    std::array<uint32_t, 4> cshape{Ba, Ca, Ha, Wb};

    tt_metal::Buffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");

    // L1 buffers
    auto l1_bank_ids = device->bank_ids_from_logical_core(core_coord);
    TT_ASSERT(not l1_bank_ids.empty());
    auto l1_bank_id = l1_bank_ids.at(0);
    // Create scratchpad buffer in L1 to stream in dtx address map from dram
    // One scratchpad buffer is used for both activation and weight address maps
    uint32_t num_address_map_fields_per_transfer = 4; // TODO: (nshanker): remove hardcoded 4 and get this value from output of DTX
    // Scratchpad buffer size must be a multiple of 32B to ensure DRAM->L1 addresses align 32B
    auto scratch_pad_for_address_map_in_l1_b0_size_bytes = 32;
    // Scratchpad buffer size must also be a multiple of address map fields per transfer. We need all address map fields for a transfer in scratchpad.
    assert(scratch_pad_for_address_map_in_l1_b0_size_bytes % (num_address_map_fields_per_transfer*sizeof(uint32_t)) == 0);
    auto scratch_pad_for_address_map_l1_buffer = tt_metal::Buffer(device, scratch_pad_for_address_map_in_l1_b0_size_bytes, l1_bank_id, scratch_pad_for_address_map_in_l1_b0_size_bytes, tt_metal::BufferType::L1);
    uint32_t scratch_pad_for_address_map_l1_address = scratch_pad_for_address_map_l1_buffer.address();
    // DRAM to L1 writes should 32B aligned
    assert(scratch_pad_for_address_map_l1_address%32 == 0);
    // Create address map metadata buffers in L1
    // Metadata vectors are copied to L1 buffers from host before calling LaunchKernels
    auto act_address_map_metadata_l1_b0_size = act_address_map_metadata.size() * sizeof(uint32_t);
    auto act_address_map_metadata_l1_buffer = tt_metal::Buffer(device, act_address_map_metadata_l1_b0_size, l1_bank_id, act_address_map_metadata_l1_b0_size, tt_metal::BufferType::L1);
    uint32_t act_address_map_metadata_l1_address = act_address_map_metadata_l1_buffer.address();
    auto weight_address_map_metadata_l1_b0_size = weight_address_map_metadata.size() * sizeof(uint32_t);
    auto weight_address_map_metadata_l1_buffer = tt_metal::Buffer(device, weight_address_map_metadata_l1_b0_size, l1_bank_id, weight_address_map_metadata_l1_b0_size, tt_metal::BufferType::L1);
    uint32_t weight_address_map_metadata_l1_address = weight_address_map_metadata_l1_buffer.address();

    // out
    uint32_t out_dram_addr = dst_dram_buffer->address();
    uint32_t out_row_size = Wb * num_bytes_of_df;
    uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;

    TT_ASSERT(out_subblock_num_tiles <= 8, "Need to ensure that matmul partials fit in dst");

    // in0
    uint32_t in0_dram_addr = src0_dram_buffer->address();
    auto in0_dram_noc_xy = src0_dram_buffer->noc_coordinates();
    uint32_t in0_noc_x = in0_dram_noc_xy.x;
    uint32_t in0_noc_y = in0_dram_noc_xy.y;

    assert(Wat % in0_block_w == 0);
    assert(in0_block_h % out_subblock_h == 0);
    uint32_t in0_num_subblocks = in0_block_h / out_subblock_h;
    uint32_t in0_block_num_tiles = in0_block_h * in0_block_w;
    uint32_t in0_subblock_h = out_subblock_h;
    uint32_t in0_subblock_num_tiles = in0_subblock_h * in0_block_w;

    // in1
    uint32_t in1_dram_addr = src1_dram_buffer->address();
    auto in1_dram_noc_xy = src1_dram_buffer->noc_coordinates();
    uint32_t in1_noc_x = in1_dram_noc_xy.x;
    uint32_t in1_noc_y = in1_dram_noc_xy.y;

    // For debug
    {
        log_debug(tt::LogOp, "Hat (activation height in tiles): {}", Hat);
        log_debug(tt::LogOp, "Wat (activation width in tiles): {}", Wat);
        log_debug(tt::LogOp, "Wbt (weight width in tiles): {}", Wbt);
        log_debug(tt::LogOp, "num_blocks_in0_h: {}", num_blocks_in0_h);
        log_debug(tt::LogOp, "num_blocks_in0_w: {}", num_blocks_in0_w);
        log_debug(tt::LogOp, "num_blocks_in1_w: {}", num_blocks_in1_w);
        log_debug(tt::LogOp, "in0_dram_addr: {}", in0_dram_addr);
        log_debug(tt::LogOp, "in0_block_h: {}", in0_block_h);
        log_debug(tt::LogOp, "in0_block_h_datums: {}", in0_block_h_datums);
        log_debug(tt::LogOp, "in0_block_w: {}", in0_block_w);
        log_debug(tt::LogOp, "in0_block_w_datums: {}", in0_block_w_datums);
        log_debug(tt::LogOp, "in0_num_subblocks: {}", in0_num_subblocks);
        log_debug(tt::LogOp, "in0_block_num_tiles: {}", in0_block_num_tiles);
        log_debug(tt::LogOp, "act_address_map_dram_addr: {}", act_address_map_dram_addr);
        log_debug(tt::LogOp, "act_address_map_metadata_l1_address: {}", act_address_map_metadata_l1_address);
        log_debug(tt::LogOp, "in0_subblock_h: {}", in0_subblock_h);
        log_debug(tt::LogOp, "in0_subblock_num_tiles: {}", in0_subblock_num_tiles);
        log_debug(tt::LogOp, "in1_dram_addr: {}", in1_dram_addr);
        log_debug(tt::LogOp, "in1_num_subblocks: {}", in1_num_subblocks);
        log_debug(tt::LogOp, "in1_block_num_tiles: {}", in1_block_num_tiles);
        log_debug(tt::LogOp, "weight_address_map_dram_addr: {}", weight_address_map_dram_addr);
        log_debug(tt::LogOp, "weight_address_map_metadata_l1_address: {}", weight_address_map_metadata_l1_address);
        log_debug(tt::LogOp, "in1_block_w: {}", in1_block_w);
        log_debug(tt::LogOp, "in1_block_h: {}", in1_block_h);
        log_debug(tt::LogOp, "out_dram_addr: {}", out_dram_addr);
        log_debug(tt::LogOp, "out_row_size: {}", out_row_size);
        log_debug(tt::LogOp, "out_subblock_h: {}", out_subblock_h);
        log_debug(tt::LogOp, "out_subblock_w: {}", out_subblock_w);
        log_debug(tt::LogOp, "out_subblock_num_tiles: {}", out_subblock_num_tiles);
        log_debug(tt::LogOp, "num_dtx_groups: {}", num_dtx_groups);
        log_debug(tt::LogOp, "scratch_pad_for_address_map_l1_address: {}", scratch_pad_for_address_map_l1_address);
    }

    create_CBs_for_fused_matmul_new_alloc(
        program,
        a.device(),
        core,
        in0_block_h * in0_block_w,
        in1_block_h * in1_block_w,
        in0_block_h * in1_block_w,
        in1_block_w,
        num_bytes_of_df,
        untilize_out);

    string reader_kernel;
    vector<uint32_t> reader_rt_args;
    reader_kernel = "tt_metal/kernels/dataflow/reader_binary_dtx.cpp";
    reader_rt_args = {
        // arguments for in0
        in0_dram_addr,
        in0_noc_x,
        in0_noc_y,
        act_address_map_dram_addr,
        act_address_map_dram_noc_x,
        act_address_map_dram_noc_y,
        act_address_map_metadata_l1_address,
        in0_block_num_tiles,

        // arguments for in1
        in1_dram_addr,
        in1_noc_x,
        in1_noc_y,
        weight_address_map_dram_addr,
        weight_address_map_dram_noc_x,
        weight_address_map_dram_noc_y,
        weight_address_map_metadata_l1_address,
        in1_block_num_tiles,

        scratch_pad_for_address_map_l1_address,
    };

    string writer_kernel;
    vector<uint32_t> writer_rt_args;
    if (untilize_out) {
        writer_kernel = "tt_metal/kernels/dataflow/writer_unary_stick_layout_8bank_blocks.cpp";
        writer_rt_args = {
            out_dram_addr,
            in0_block_h_datums,
            in1_block_w*TILE_WIDTH*num_bytes_of_df,
            1,
            num_blocks_in0_h,
            num_blocks_in1_w,
            Wb*num_bytes_of_df
        };
    } else {
        assert(false && "Tiled output unsupported");
        writer_kernel = "tt_metal/kernels/dataflow/writer_matmul_tile_layout.cpp";
        writer_rt_args = {
            out_dram_addr,
            0,
            1,
            Wbt,
            out_subblock_w,
            out_subblock_h * Wbt,

            out_subblock_w,
            out_subblock_h,
            out_subblock_w * out_subblock_h,
            Wbt / out_subblock_w,
            Hat / out_subblock_h
        };
    }
    auto reader = tt_metal::CreateDataMovementKernel(
        program,
        reader_kernel,
        core, DataMovementProcessor::RISCV_1, NOC::RISCV_1_default);

    auto writer = tt_metal::CreateDataMovementKernel(
        program,
        writer_kernel,
        core, DataMovementProcessor::RISCV_0, NOC::RISCV_0_default);

    vector<uint32_t> compute_kernel_args = {
        in0_block_w,
        in0_num_subblocks,
        in0_block_num_tiles,
        in0_subblock_num_tiles,
        in0_subblock_h,

        in1_num_subblocks,
        in1_block_num_tiles,
        in1_block_w,

        num_blocks_in0_h,
        num_blocks_in0_w,
        num_blocks_in1_w,

        out_subblock_h,
        out_subblock_w,
        out_subblock_num_tiles,

        true,
        untilize_out
    };

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto eltwise_binary_kernel = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/bmm_tilize_untilize.cpp",
        core,
        compute_kernel_args,
        MathFidelity::HiFi4,
        fp32_dest_acc_en,
        math_approx_mode
    );

    tt_metal::WriteRuntimeArgsToDevice(
        device, reader, core,
        reader_rt_args
    );

    tt_metal::WriteRuntimeArgsToDevice(
        device, writer, core,
        writer_rt_args
    );

    pass &= tt_metal::CompileProgram(device, program, false);
    pass &= tt_metal::ConfigureDeviceWithProgram(device, program);
    tt_metal::WriteToDeviceL1(device, core_coord, act_address_map_metadata_l1_address, act_address_map_metadata);
    tt_metal::WriteToDeviceL1(device, core_coord, weight_address_map_metadata_l1_address, weight_address_map_metadata);
    pass &= tt_metal::LaunchKernels(device, program);

    TT_ASSERT(pass);
    return program;
}

Tensor conv(const Tensor& a, const Tensor &b, const vector<int> conv_params, uint32_t in0_block_h, uint32_t in0_block_w, uint32_t in1_block_w,
             uint32_t out_subblock_h, uint32_t out_subblock_w, bool untilize_out) {
    return operation::run_with_autopad(Conv(in0_block_h, in0_block_w, in1_block_w, out_subblock_h, out_subblock_w, conv_params, untilize_out), a, b);
}

Program conv_single_core(const Tensor& a, const Tensor &b, const vector<int> conv_params, uint32_t in0_block_h, uint32_t in0_block_w, uint32_t in1_block_w,
             uint32_t out_subblock_h, uint32_t out_subblock_w, bool untilize_out, Tensor &output) {
    return conv_as_large_bmm_single_core_(a, b, conv_params, in0_block_h, in0_block_w, in1_block_w, out_subblock_h, out_subblock_w, untilize_out, output);
}

void Conv::validate(const std::vector<std::reference_wrapper<const Tensor>>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0).get();
    const auto& input_tensor_b = input_tensors.at(1).get();
    // TODO: ...
}

std::vector<Shape> Conv::compute_output_shapes(const std::vector<std::reference_wrapper<const Tensor>>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0).get();
    const auto& input_tensor_b = input_tensors.at(1).get();
    vector<int> input_tensor_a_shape = { (int) input_tensor_a.shape()[1], (int) input_tensor_a.shape()[2], (int) input_tensor_a.shape()[3]};
    return {Shape(compute_conv_as_mm_shape(input_tensor_a_shape, conv_params, in0_block_h, in0_block_w))};
}

std::vector<Tensor> Conv::create_output_tensors(const std::vector<std::reference_wrapper<const Tensor>>& input_tensors) const {
    const auto output_shape = this->compute_output_shapes(input_tensors).at(0);
    const auto& input_tensor = input_tensors.at(0).get();
    Tensor output = create_output_dram_buffer_(input_tensor.device(), input_tensor.dtype(), output_shape, untilize_out);
    std::vector<Tensor> output_tensors;
    // TODO: check if anything else needs to be done here.
    output_tensors.emplace_back(output);
    return output_tensors;
}

Program Conv::create_program(const std::vector<std::reference_wrapper<const Tensor>>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0).get();
    const auto& input_tensor_b = input_tensors.at(1).get();
    auto& output_tensor = output_tensors.at(0);

    return conv_single_core(input_tensor_a, input_tensor_b, conv_params, in0_block_h, in0_block_w, in1_block_w, out_subblock_h, out_subblock_w, untilize_out, output_tensor);
}

}  // namespace tt_metal

}  // namespace tt
