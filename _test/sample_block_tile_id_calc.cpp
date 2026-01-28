// Sample code to compute:
//  - shape_blocks, src_block_id_gap, out_block_id_gap
//  - per-(logical)core block_coord, src_block_tile_id, out_block_tile_id
//
// Inputs:
//  - input/output padded shapes (rank=3)
//  - slice_start (element indices, rank=3)
//  - split_work_to_cores outputs (num cores, core group sizes, blocks-per-core)
//
// This mirrors the logic used in ttnn slice tiled interleaved code paths.
//
// Build as a standalone C++ file:
//   g++ -std=c++20 -O2 -Wall -Wextra -pedantic sample_block_tile_id_calc.cpp -o sample

#include <cstdint>
#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>

#define TEST_ROW_WISE 0
#define PRINT_ACCESS 0

static constexpr uint32_t TILE_WIDTH = 32;
static constexpr uint32_t TILE_HEIGHT = 32;

int g_cnt = 0;
int id_min = 100;
int id_max = -1;

using Shape = std::vector<uint32_t>;  // e.g. [D0, D1, D2]

struct SplitWorkResult {
    uint32_t num_cores;
    // In TT-Metal, these would be CoreRangeSet sizes:
    //   core_group.num_cores() and core_group_cliff.num_cores()
    uint32_t num_cores_in_core_group;
    uint32_t num_cores_in_core_group_cliff;
    uint32_t num_blocks_per_core;
    uint32_t num_blocks_per_core_cliff;
};

struct BlockIdResult {
    // Per the updated slice code:
    // - src_block_tile_id uses src_shape_blocks
    // - out_block_tile_id uses out_shape_blocks
    std::vector<uint32_t> src_shape_blocks;  // size rank-1
    std::vector<uint32_t> out_shape_blocks;  // size rank-1
    std::vector<uint32_t> src_block_id_gap;  // size rank-1
    std::vector<uint32_t> out_block_id_gap;  // size rank-1

    struct PerCore {
        uint32_t core_index;
        std::vector<uint32_t> block_coord;  // size rank-1
        uint32_t src_block_tile_id;
        uint32_t out_block_tile_id;
        uint32_t num_blocks_arg;
    };
    std::vector<PerCore> per_core;
};

// Convert a linear tile_id into a 3D tile coordinate [d0, tile_h, tile_w].
// Assumes a rank-3 tensor with tiles laid out in row-major order over the last two dims:
//   tile_id = ((d0 * tiles_h) + tile_h) * tiles_w + tile_w
static std::array<uint32_t, 3> tile_id_to_3d_tile_coord(uint32_t tile_id, const Shape& tensor_shape) {
    // tensor_shape is in elements: [D0, H, W]
    if (tensor_shape.size() != 3) {
        throw std::invalid_argument("tensor_shape must be of size 3");
        return {0, 0, 0};
    };
    const uint32_t tiles_h = tensor_shape.at(1) / TILE_HEIGHT;
    const uint32_t tiles_w = tensor_shape.at(2) / TILE_WIDTH;
    const uint32_t tiles_per_d0 = tiles_h * tiles_w;

    const uint32_t d0 = tile_id / tiles_per_d0;
    const uint32_t rem = tile_id % tiles_per_d0;
    const uint32_t tile_h = rem / tiles_w;
    const uint32_t tile_w = rem % tiles_w;
    return {d0, tile_h, tile_w};
}

// Simulate ONLY the src_block_id progression in the reader kernel:
//   ttnn/.../slice_reader_unary_tile_row_col_interleaved.cpp
//
// This ignores the actual read_block() I/O and just reproduces:
//   - src_block_id += block_id_stride
//   - block_coord++ with wrap against shape_blocks
//   - src_block_id += block_id_gap[j] when block_coord wraps
struct ReaderKernelSimResult {
    std::vector<uint32_t> src_block_ids_used;  // one entry per block iteration
    uint32_t final_src_block_id = 0;           // value after the loop finishes
    std::vector<uint32_t> final_block_coord;   // block_coord after the loop finishes
};

static ReaderKernelSimResult simulate_reader_kernel_src_block_id(
    uint32_t src_block_id_start,
    uint32_t num_blocks,
    uint32_t block_id_stride,
    uint32_t num_tiles,
    uint32_t tile_id_stride,
    const std::vector<uint32_t>& shape_blocks,       // runtime "shape_blocks" (output iteration space)
    const std::vector<uint32_t>& block_id_gap,       // runtime "block_id_gap" (src_block_id_gap)
    const std::vector<uint32_t>& block_coord_start,  // runtime "block_coord"
    std::vector<int>& src_access,
    const Shape& src_shape_blocks,
    bool print_debug) {
    ReaderKernelSimResult r;
    r.src_block_ids_used.reserve(num_blocks);

    uint32_t src_block_id = src_block_id_start;
    std::vector<uint32_t> block_coord = block_coord_start;
    const int32_t num_dims = static_cast<int32_t>(shape_blocks.size());

    if (print_debug) {
        std::cout << " --------------------------------------------------------------\n";
        std::cout << __FUNCTION__ << std::endl;
        std::cout << "  src_block_id_start: " << src_block_id_start << "\n";
        std::cout << "  num_blocks: " << num_blocks << "\n";
        std::cout << "  block_id_stride: " << block_id_stride << "\n";
        std::cout << "  shape_blocks: ";
        for (auto& sb : shape_blocks) {
            std::cout << "  " << sb << " ";
        }
        std::cout << "\n";
        std::cout << "  block_id_gap: ";
        for (auto& gap : block_id_gap) {
            std::cout << "  " << gap << " ";
        }
        std::cout << "\n";
        std::cout << "  block_coord_start: ";
        for (auto& bc : block_coord_start) {
            std::cout << "  " << bc << " ";
        }
        std::cout << "\n";
        std::cout << " --------------------------------------------------------------\n";
    }

    for (uint32_t i = 0; i < num_blocks; i++) {
        int32_t tile_id = src_block_id;
        for (uint32_t k = 0; k < num_tiles; k++) {
            g_cnt++;

            if (print_debug) {
                std::cout << "k = " << k << ", tile_id = " << tile_id << "\n";
            }

            if ((tile_id < 0) || (tile_id >= src_access.size())) {
                std::cout << "tile_id is out of range. tile_id =" << tile_id << "\n";
            } else {
                src_access[tile_id]++;
                if (id_min > tile_id) {
                    id_min = tile_id;
                }
                if (id_max < tile_id) {
                    id_max = tile_id;
                }
            }

            tile_id += tile_id_stride;
        }
#if 0
        if (print_debug) {
            // Print src_block_id and its corresponding 3d coordinate based on src_real_shape, not src_shape_blocks.
            if (src_shape_blocks.size() == 3) {
                auto coord = tile_id_to_3d_tile_coord(src_block_id, src_shape_blocks);
                std::cout << "src_block_id: " << src_block_id << " 3d_coord: [" << coord[0] << ", " << coord[1] << ", "
                          << coord[2] << "]\n";
            } else {
                std::cout << "src_block_id: " << src_block_id << "\n";
            }
        }
#endif
        r.src_block_ids_used.push_back(src_block_id);
        src_block_id += block_id_stride;

        // Kernel does: for j=num_dims-1..0: block_coord[j]++; if wrap -> add gap; else break
        for (int32_t j = num_dims - 1; j >= 0; --j) {
            block_coord[j]++;
            if (block_coord[j] == shape_blocks[j]) {
                block_coord[j] = 0;
                src_block_id += block_id_gap[j];
            } else {
                break;
            }
        }
    }

    r.final_src_block_id = src_block_id;
    r.final_block_coord = std::move(block_coord);
    return r;
}

// Simulate ONLY the out_block_id progression in the writer kernel:
//   ttnn/.../slice_writer_unary_tile_row_col_interleaved.cpp
//
// This ignores the actual write_block() I/O and just reproduces:
//   - out_block_id += block_id_stride
//   - block_coord++ with wrap against shape_blocks
//   - out_block_id += block_id_gap[j] when block_coord wraps
struct WriterKernelSimResult {
    std::vector<uint32_t> out_block_ids_used;  // one entry per block iteration
    uint32_t final_out_block_id = 0;           // value after the loop finishes
    std::vector<uint32_t> final_block_coord;   // block_coord after the loop finishes
};

static WriterKernelSimResult simulate_writer_kernel_out_block_id(
    uint32_t out_block_id_start,
    uint32_t num_blocks,
    uint32_t block_id_stride,
    uint32_t num_tiles,
    uint32_t tile_id_stride,
    const std::vector<uint32_t>& shape_blocks,       // runtime "shape_blocks" (output iteration space)
    const std::vector<uint32_t>& block_id_gap,       // runtime "block_id_gap" (out_block_id_gap)
    const std::vector<uint32_t>& block_coord_start,  // runtime "block_coord"
    std::vector<int>& out_access,
    const Shape& out_shape_blocks,
    bool print_debug) {
    WriterKernelSimResult r;
    r.out_block_ids_used.reserve(num_blocks);

    uint32_t out_block_id = out_block_id_start;
    std::vector<uint32_t> block_coord = block_coord_start;
    const int32_t num_dims = static_cast<int32_t>(shape_blocks.size());

    if (print_debug) {
        std::cout << " --------------------------------------------------------------\n";
        std::cout << __FUNCTION__ << std::endl;
        std::cout << "  out_block_id_start: " << out_block_id_start << "\n";
        std::cout << "  num_blocks: " << num_blocks << "\n";
        std::cout << "  block_id_stride: " << block_id_stride << "\n";
        std::cout << "  shape_blocks: ";
        for (auto& sb : shape_blocks) {
            std::cout << "  " << sb << " ";
        }
        std::cout << "\n";
        std::cout << "  block_id_gap: ";
        for (auto& gap : block_id_gap) {
            std::cout << "  " << gap << " ";
        }
        std::cout << "\n";
        std::cout << "  block_coord_start: ";
        for (auto& bc : block_coord_start) {
            std::cout << "  " << bc << " ";
        }
        std::cout << "\n";
        std::cout << " --------------------------------------------------------------\n";
    }

    for (uint32_t i = 0; i < num_blocks; i++) {
        int32_t tile_id = out_block_id;
        for (uint32_t k = 0; k < num_tiles; k++) {
            g_cnt++;

            if (print_debug) {
                std::cout << "k = " << k << ", tile_id = " << tile_id << "\n";
            }

            if ((tile_id < 0) || (tile_id >= out_access.size())) {
                std::cout << "tile_id is out of range. tile_id =" << tile_id << "\n";
            } else {
                out_access[tile_id]++;
                if (id_min > tile_id) {
                    id_min = tile_id;
                }
                if (id_max < tile_id) {
                    id_max = tile_id;
                }
            }

            tile_id += tile_id_stride;
        }
        r.out_block_ids_used.push_back(out_block_id);
        out_block_id += block_id_stride;

        // Kernel does: for j=num_dims-1..0: block_coord[j]++; if wrap -> add gap; else break
        for (int32_t j = num_dims - 1; j >= 0; --j) {
            block_coord[j]++;
            if (block_coord[j] == shape_blocks[j]) {
                block_coord[j] = 0;
                out_block_id += block_id_gap[j];
            } else {
                break;
            }
        }
    }

    r.final_out_block_id = out_block_id;
    r.final_block_coord = std::move(block_coord);
    return r;
}

// Simulate split_work_to_cores function:
//   tt_metal/common/work_split.cpp
//
// This simulates the work distribution logic used in slice_program_factory_tile_interleaved.cpp
// Returns a simplified SplitWorkResult matching the structure used in the sample code.
static SplitWorkResult simulate_split_work_to_cores(
    uint32_t num_cores_x, uint32_t num_cores_y, uint32_t units_to_divide, bool row_wise = false) {
    SplitWorkResult result;

    if (units_to_divide == 0) {
        result.num_cores = 0;
        result.num_cores_in_core_group = 0;
        result.num_cores_in_core_group_cliff = 0;
        result.num_blocks_per_core = 0;
        result.num_blocks_per_core_cliff = 0;
        return result;
    }

    uint32_t max_num_cores = num_cores_x * num_cores_y;
    uint32_t target_num_cores;

    // Determine target number of cores
    if (units_to_divide >= max_num_cores) {
        target_num_cores = max_num_cores;
    } else {
        target_num_cores = units_to_divide;
    }

    result.num_cores = target_num_cores;

    // Calculate work distribution
    uint32_t units_per_core_group_1 = units_to_divide / target_num_cores;
    uint32_t num_cores_with_more_work = units_to_divide % target_num_cores;

    // Evenly divided units to all target cores
    if (num_cores_with_more_work == 0) {
        result.num_cores_in_core_group = target_num_cores;
        result.num_cores_in_core_group_cliff = 0;
        result.num_blocks_per_core = units_per_core_group_1;
        result.num_blocks_per_core_cliff = 0;
    }
    // Uneven division of units across cores
    else {
        // Group of cores that do more work
        uint32_t num_core_group_1_cores = num_cores_with_more_work;
        uint32_t num_core_group_2_cores = target_num_cores - num_core_group_1_cores;

        result.num_cores_in_core_group = num_core_group_1_cores;
        result.num_cores_in_core_group_cliff = num_core_group_2_cores;
        result.num_blocks_per_core = units_per_core_group_1 + 1;    // Group 1 gets one more
        result.num_blocks_per_core_cliff = units_per_core_group_1;  // Group 2 gets base amount
    }

    return result;
}

// Specialization of the lambda from slice code for rank=3 (so coord has 2 dims).
static uint32_t calc_block_tile_id_rank3(
    const std::vector<uint32_t>& coord,       // [d0, inner]
    const std::vector<uint32_t>& shape,       // [D0, inner_extent]
    const std::vector<uint32_t>& zero_index,  // [start_d0, start_inner]
    bool row_wise,
    uint32_t sdim_size,
    uint32_t sdim_zero_index) {
    uint32_t tile_index = 0;
    uint32_t multiplier = 1;

    // inner dimension contribution
    if (row_wise) {
        tile_index += sdim_zero_index * multiplier;
        multiplier *= sdim_size;
        tile_index += (coord[1] + zero_index[1]) * multiplier;
        multiplier *= shape[1];
    } else {
        tile_index += (coord[1] + zero_index[1]) * multiplier;
        multiplier *= shape[1];
        tile_index += sdim_zero_index * multiplier;
        multiplier *= sdim_size;
    }

    // outer dimension contribution
    tile_index += (coord[0] + zero_index[0]) * multiplier;
    return tile_index;
}

static BlockIdResult compute_block_ids_rank3(
    const Shape& input_shape,   // padded shapes
    const Shape& output_shape,  // padded shapes
    const Shape& slice_start,   // element indices
    bool row_wise,
    const SplitWorkResult& split) {
    constexpr uint32_t rank = 3;
    BlockIdResult r;

    const uint32_t out_tiles_per_row = output_shape[2] / TILE_WIDTH;
    const uint32_t out_tiles_per_col = output_shape[1] / TILE_HEIGHT;
    const uint32_t src_tiles_per_row = input_shape[2] / TILE_WIDTH;
    const uint32_t src_tiles_per_col = input_shape[1] / TILE_HEIGHT;

    // --- src/out shape_blocks + gaps (matches updated slice code) ---
    r.src_shape_blocks.assign(rank - 1, 0);
    r.out_shape_blocks.assign(rank - 1, 0);
    r.src_block_id_gap.assign(rank - 1, 0);
    r.out_block_id_gap.assign(rank - 1, 0);

    int index = int(rank) - 2;  // = 1
    if (row_wise) {
        r.out_shape_blocks[index] = out_tiles_per_col;
        r.src_shape_blocks[index] = src_tiles_per_col;
        r.src_block_id_gap[index] = src_tiles_per_row * (src_tiles_per_col - out_tiles_per_col);
        r.out_block_id_gap[index] = 0;
        index--;
    } else {
        r.out_shape_blocks[index] = out_tiles_per_row;
        r.src_shape_blocks[index] = src_tiles_per_row;
        r.src_block_id_gap[index] = src_tiles_per_row * src_tiles_per_col - out_tiles_per_row;
        r.out_block_id_gap[index] = out_tiles_per_row * out_tiles_per_col - out_tiles_per_row;
        index--;
    }

    uint32_t size_acc_src = src_tiles_per_row * src_tiles_per_col;
    for (int32_t i = int32_t(rank) - 3; i >= 0; --i) {  // i=0 only
        int size_unpadded_dim = int(input_shape[i]) - int(output_shape[i]);
        r.out_shape_blocks[index] = output_shape[i];
        r.src_shape_blocks[index] = input_shape[i];
        r.src_block_id_gap[index] = uint32_t(size_unpadded_dim) * size_acc_src;
        r.out_block_id_gap[index] = 0;
        size_acc_src *= input_shape[i];
        index--;
    }

    // --- block_start and zero_coord ---
    // For rank=3, treat:
    //  - dim0 as "block" index directly (matches the original code shape_blocks[0]=output_shape[0])
    //  - dim1/dim2 starts converted to tile coords
    std::vector<uint32_t> block_start(rank - 1, 0);
    std::vector<uint32_t> zero_coord(rank - 1, 0);

    block_start[0] = slice_start[0];

    const uint32_t start_tile_w = slice_start[2] / TILE_WIDTH;
    const uint32_t start_tile_h = slice_start[1] / TILE_HEIGHT;

    // sdim (the "other" tiled dimension) start + size
    const uint32_t src_block_dim_start = row_wise ? start_tile_w : start_tile_h;
    const uint32_t src_block_dim_size = row_wise ? src_tiles_per_row : src_tiles_per_col;
    const uint32_t out_block_dim_size = row_wise ? out_tiles_per_row : out_tiles_per_col;

    // inner coordinate start (the iterated tiled dimension)
    block_start[1] = row_wise ? start_tile_h : start_tile_w;

    // --- iterate one block_coord per core (same pattern as the slice code) ---
    std::vector<uint32_t> block_coord(rank - 1, 0);
    r.per_core.reserve(split.num_cores);

    for (uint32_t core_i = 0; core_i < split.num_cores; ++core_i) {
        uint32_t num_blocks_arg = split.num_blocks_per_core;
        if (core_i >= split.num_cores_in_core_group) {
            num_blocks_arg = split.num_blocks_per_core_cliff;
        }

        const uint32_t src_block_tile_id = calc_block_tile_id_rank3(
            block_coord,
            r.src_shape_blocks,
            block_start,
            /*row_wise_for_calc=*/(row_wise ? 1 : 0),
            /*sdim_size=*/src_block_dim_size,
            /*sdim_zero_index=*/src_block_dim_start);

        const uint32_t out_block_tile_id = calc_block_tile_id_rank3(
            block_coord,
            r.out_shape_blocks,
            zero_coord,
            /*row_wise_for_calc=*/(row_wise ? 1 : 0),
            /*sdim_size=*/out_block_dim_size,
            /*sdim_zero_index=*/0);

        r.per_core.push_back(BlockIdResult::PerCore{
            .core_index = core_i,
            .block_coord = block_coord,
            .src_block_tile_id = src_block_tile_id,
            .out_block_tile_id = out_block_tile_id,
            .num_blocks_arg = num_blocks_arg,
        });

        // Advance block_coord the same way as the updated slice code:
        //  - jump by num_blocks_arg on the innermost dimension
        //  - then carry/warp across dims if we exceeded bounds
        block_coord[rank - 2] += num_blocks_arg;
        for (int j = int(rank) - 2; j >= 1; --j) {  // j=1 only for rank=3
            if (block_coord[j] >= r.out_shape_blocks[j]) {
                const uint32_t carry = block_coord[j] / r.out_shape_blocks[j];
                block_coord[j] = block_coord[j] % r.out_shape_blocks[j];
                block_coord[j - 1] += carry;
            } else {
                break;
            }
        }
    }

    return r;
}

int main(int argc, char* argv[]) {
    int rank = 3;
    Shape input, output;
#if TEST_ROW_WISE > 0
    Shape default_input = {768, 160, 768};
    Shape default_output = {686, 128, 686};
#else
    Shape default_input = {768, 768, 160};
    Shape default_output = {686, 686, 128};
#endif
    Shape start = {0, 0, 0};

    // Parse input and output shape from arguments if provided.
    // Usage: prog [input0 input1 input2 output0 output1 output2]
    if (argc == 7) {
        input = {
            static_cast<uint32_t>(std::stoi(argv[1])),
            static_cast<uint32_t>(std::stoi(argv[2])),
            static_cast<uint32_t>(std::stoi(argv[3]))};
        output = {
            static_cast<uint32_t>(std::stoi(argv[4])),
            static_cast<uint32_t>(std::stoi(argv[5])),
            static_cast<uint32_t>(std::stoi(argv[6]))};
    } else {
        input = default_input;
        output = default_output;
    }

    // Pad H/W and start H/W to multiples of tile, like typical tiled paths
    auto pad32 = [](uint32_t v) { return (v % 32) ? (v + (32 - v % 32)) : v; };
    input[1] = pad32(input[1]);
    input[2] = pad32(input[2]);
    output[1] = pad32(output[1]);
    output[2] = pad32(output[2]);
    start[1] = pad32(start[1]);
    start[2] = pad32(start[2]);

    std::cout << "input  = [" << input[0] << " " << input[1] << " " << input[2] << "]\n";
    std::cout << "output = [" << output[0] << " " << output[1] << " " << output[2] << "]\n";
    std::cout << "start  = [" << start[0] << " " << start[1] << " " << start[2] << "]\n";

    std::cout << "Tile Dimension" << "\n";
    std::cout << "input tiles = [" << input[0] << " " << input[1] / TILE_HEIGHT << " " << input[2] / TILE_WIDTH
              << "]\n";
    std::cout << "output tiles = [" << output[0] << " " << output[1] / TILE_HEIGHT << " " << output[2] / TILE_WIDTH
              << "]\n";
    std::cout << "start tiles  = [" << start[0] << " " << start[1] / TILE_HEIGHT << " " << start[2] / TILE_WIDTH
              << "]\n";

    //
    int input_size = input[0] * input[1] * input[2];
    std::vector<int> src_access(input_size / 32 / 32, 0);
    std::cout << "src_access.size = " << src_access.size() << std::endl;
    //
    int output_size = output[0] * output[1] * output[2];
    std::vector<int> out_access(output_size / 32 / 32, 0);
    std::cout << "out_access.size = " << out_access.size() << std::endl;

    // Example choice (your real code picks based on CB capacity).
    bool row_wise = (output[rank - 1] > output[rank - 2]);
    int num_blocks = output[0] * output[rank - 2] / TILE_HEIGHT * output[rank - 1] / TILE_WIDTH;
    if (row_wise) {
        num_blocks /= (output[rank - 1] / TILE_WIDTH);
    } else {
        num_blocks /= (output[rank - 2] / TILE_HEIGHT);
    }

    // Simulate split_work_to_cores (matching slice_program_factory_tile_interleaved.cpp logic)
    // For this sample, use a typical grid size (e.g., 8x8 = 64 cores)
    uint32_t num_cores_x = 8;
    uint32_t num_cores_y = 8;
    SplitWorkResult split = simulate_split_work_to_cores(num_cores_x, num_cores_y, num_blocks, row_wise);

    int proc_blocks = split.num_cores_in_core_group * split.num_blocks_per_core +
                      split.num_cores_in_core_group_cliff * split.num_blocks_per_core_cliff;
    if (num_blocks != proc_blocks) {
        std::cout << "num_blocks = " << num_blocks << " != proc_blocks = " << proc_blocks << "\n";
        return 1;
    }

    std::cout << "SplitWorkResult:\n";
    std::cout << "  num_cores = " << split.num_cores << "\n";
    std::cout << "  num_cores_in_core_group = " << split.num_cores_in_core_group << "\n";
    std::cout << "  num_cores_in_core_group_cliff = " << split.num_cores_in_core_group_cliff << "\n";
    std::cout << "  num_blocks_per_core = " << split.num_blocks_per_core << "\n";
    std::cout << "  num_blocks_per_core_cliff = " << split.num_blocks_per_core_cliff << "\n";
    std::cout << "row_wise = " << (row_wise ? "true" : "false") << "\n";

    auto r = compute_block_ids_rank3(input, output, start, row_wise, split);

    std::cout << "src_shape_blocks = [" << r.src_shape_blocks[0] << ", " << r.src_shape_blocks[1] << "]\n";
    std::cout << "out_shape_blocks = [" << r.out_shape_blocks[0] << ", " << r.out_shape_blocks[1] << "]\n";
    std::cout << "src_block_id_gap = [" << r.src_block_id_gap[0] << ", " << r.src_block_id_gap[1] << "]\n";
    std::cout << "out_block_id_gap = [" << r.out_block_id_gap[0] << ", " << r.out_block_id_gap[1] << "]\n";

    // Reader-kernel src_block_id progression verification (ignore actual read_block I/O)
    const uint32_t src_tiles_per_row = input[rank - 1] / TILE_WIDTH;
    const uint32_t src_tiles_per_col = input[rank - 2] / TILE_HEIGHT;
    const uint32_t out_tiles_per_row = output[rank - 1] / TILE_WIDTH;
    const uint32_t out_tiles_per_col = output[rank - 2] / TILE_HEIGHT;
    const uint32_t src_block_id_stride = row_wise ? src_tiles_per_row : 1;  // compile-time block_id_stride
    const uint32_t tile_id_stride = row_wise ? 1 : src_tiles_per_row;
    for (size_t core_i = 0; core_i < r.per_core.size(); ++core_i) {
        const auto& pc = r.per_core[core_i];
        std::array<uint32_t, 3> src_block_tile_pos, out_block_tile_pos;
        src_block_tile_pos = tile_id_to_3d_tile_coord(pc.src_block_tile_id, input);
        out_block_tile_pos = tile_id_to_3d_tile_coord(pc.out_block_tile_id, output);
        uint32_t num_tiles = (row_wise ? output[rank - 1] : output[rank - 2]) / TILE_WIDTH;

        bool print_debug = false;

        if (print_debug) {
            std::cout << "core " << pc.core_index << " num_blocks=" << pc.num_blocks_arg << " block_coord=["
                      << pc.block_coord[0] << "," << pc.block_coord[1] << "]"
                      << " src_block_tile_id=" << pc.src_block_tile_id << " src_block_tile_coord=["
                      << src_block_tile_pos[0] << "," << src_block_tile_pos[1] << "," << src_block_tile_pos[2] << "]"
                      << " out_block_tile_id=" << pc.out_block_tile_id << " out_block_tile_coord=["
                      << out_block_tile_pos[0] << "," << out_block_tile_pos[1] << "," << out_block_tile_pos[2] << "]"
                      << " num_tiles=" << num_tiles << "," << "tile_id_stride=" << tile_id_stride << std::endl;
        }

        print_debug = false;

        const auto sim = simulate_reader_kernel_src_block_id(
            /*src_block_id_start=*/pc.src_block_tile_id,
            /*num_blocks=*/pc.num_blocks_arg,
            /*block_id_stride=*/src_block_id_stride,
            /*num_tiles=*/num_tiles,
            /*tile_id_stride=*/tile_id_stride,
            /*shape_blocks=*/r.out_shape_blocks,
            /*block_id_gap=*/r.src_block_id_gap,
            /*block_coord_start=*/pc.block_coord,
            /*src_access=*/src_access,
            /*src_shape_blocks=*/input,
            /*print_debug=*/print_debug);

        // Writer-kernel out_block_id progression verification (ignore actual write_block I/O)
        const uint32_t out_block_id_stride =
            row_wise ? out_tiles_per_row : 1;  // compile-time block_id_stride for writer
        const uint32_t out_tile_id_stride = row_wise ? 1 : out_tiles_per_row;  // compile-time tile_id_stride for writer

        const auto writer_sim = simulate_writer_kernel_out_block_id(
            /*out_block_id_start=*/pc.out_block_tile_id,
            /*num_blocks=*/pc.num_blocks_arg,
            /*block_id_stride=*/out_block_id_stride,
            /*num_tiles=*/num_tiles,
            /*tile_id_stride=*/out_tile_id_stride,
            /*shape_blocks=*/r.out_shape_blocks,
            /*block_id_gap=*/r.out_block_id_gap,
            /*block_coord_start=*/pc.block_coord,
            /*out_access=*/out_access,
            /*out_shape_blocks=*/output,
            /*print_debug=*/print_debug);

        // Optional sanity check: the next core's start should match this core's "final" state
        // (because host advances block_coord by num_blocks_arg and re-computes src_block_tile_id).
        if (core_i + 1 < r.per_core.size()) {
            const auto& next = r.per_core[core_i + 1];
            if (sim.final_src_block_id != next.src_block_tile_id) {
                std::cout << "MISMATCH core " << core_i << ": sim.final_src_block_id=" << sim.final_src_block_id
                          << " next.src_block_tile_id=" << next.src_block_tile_id << "\n";
            }
            if (sim.final_block_coord != next.block_coord) {
                std::cout << "MISMATCH core " << core_i << ": sim.final_block_coord != next.block_coord\n";
            }
            // Writer sanity check
            if (writer_sim.final_out_block_id != next.out_block_tile_id) {
                std::cout << "MISMATCH core " << core_i
                          << ": writer_sim.final_out_block_id=" << writer_sim.final_out_block_id
                          << " next.out_block_tile_id=" << next.out_block_tile_id << "\n";
            }
            if (writer_sim.final_block_coord != next.block_coord) {
                std::cout << "MISMATCH core " << core_i << ": writer_sim.final_block_coord != next.block_coord\n";
            }
        }
    }

#if PRINT_ACCESS > 0
    std::cout << "g_count = " << g_cnt << ", min = " << id_min << ", max = " << id_max << "\n";
    if (g_cnt) {
        int k, m, n;
        k = m = n = 0;
        std::cout << "src_access:\n";
        std::cout << "[n " << n << "]\n";
        for (int i = 0; i < src_access.size(); i++) {
            // std::cout << "src_access[" << i << "] = " << src_access[i] << "\n";
            std::cout << src_access[i] << " ";
            k++;
            if (k == src_tiles_per_row) {
                std::cout << "\n";
                k = 0;
                m++;
                if (m == src_tiles_per_col) {
                    m = 0;
                    n++;
                    std::cout << "[n " << n << "]\n";
                }
            }
        }
        std::cout << "\n";

        // Print out_access
        k = m = n = 0;
        std::cout << "out_access:\n";
        std::cout << "[n " << n << "]\n";
        for (int i = 0; i < out_access.size(); i++) {
            // std::cout << "out_access[" << i << "] = " << out_access[i] << "\n";
            std::cout << out_access[i] << " ";
            k++;
            if (k == out_tiles_per_row) {
                std::cout << "\n";
                k = 0;
                m++;
                if (m == out_tiles_per_col) {
                    m = 0;
                    n++;
                    std::cout << "[n " << n << "]\n";
                }
            }
        }
    }
#endif
}
