// Sample model code for slice_program_factory_tile_interleaved.cpp
//
// This demonstrates how the SliceTileInterleavedProgramFactory works:
//  - Compile-time arguments for reader and writer kernels
//  - Runtime arguments calculation for each core
//  - Tile coordinate to tile ID conversion
//  - Work distribution across cores
//  - Reader kernel execution simulation (slice_reader_unary_tile_row_col_interleaved.cpp)
//  - Writer kernel execution simulation (slice_writer_unary_tile_row_col_interleaved.cpp)
//
// Build as a standalone C++ file:
//   g++ -std=c++20 -O2 -Wall -Wextra -pedantic sample_slice_tile_interleaved.cpp -o sample_slice_tile_interleaved
//
// Usage:
//   ./sample_slice_tile_interleaved
//   ./sample_slice_tile_interleaved [input0 input1 input2 output0 output1 output2 start0 start1 start2]
//   ./sample_slice_tile_interleaved --verbose
//   ./sample_slice_tile_interleaved --verbose [input0 input1 input2 output0 output1 output2 start0 start1 start2]

#include <cstdint>
#include <vector>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <algorithm>
#include <string>

static constexpr uint32_t TILE_WIDTH = 32;
static constexpr uint32_t TILE_HEIGHT = 32;
static constexpr uint32_t TILE_HW = TILE_WIDTH * TILE_HEIGHT;

using Shape = std::vector<uint32_t>;  // e.g. [D0, D1, D2, ...]

// Simulates the compile-time arguments for reader kernel
struct ReaderCompileTimeArgs {
    uint32_t cb_id;             // Circular buffer index (CBIndex::c_0 = 0)
    uint32_t src_tile_stride;   // Stride between tiles in source (num_cores)
    uint32_t rank;              // Tensor rank
    uint32_t single_tile_size;  // Size of a single tile in bytes
    // TensorAccessorArgs would follow (simplified here)
};

// Simulates the compile-time arguments for writer kernel
struct WriterCompileTimeArgs {
    uint32_t cb_id;             // Circular buffer index (CBIndex::c_0 = 0)
    uint32_t out_tile_stride;   // Stride between tiles in output (num_cores)
    uint32_t rank;              // Tensor rank
    uint32_t single_tile_size;  // Size of a single tile in bytes
    // TensorAccessorArgs would follow (simplified here)
};

// Runtime arguments for reader kernel per core
struct ReaderRuntimeArgs {
    uint32_t src_buffer_addr;               // Source buffer address
    uint32_t core_tile_id;                  // Starting tile ID for this core in source
    uint32_t num_tiles_arg;                 // Number of tiles to process
    std::vector<uint32_t> out_shape_tiles;  // Output shape in tiles (rank elements)
    std::vector<uint32_t> tile_coord;       // Starting tile coordinate (rank elements)
    std::vector<uint32_t> tile_id_acc;      // Source tile ID increments (rank elements)
    std::vector<uint32_t> coord_inc;        // Source tile coordinate increments (rank elements)
};

// Runtime arguments for writer kernel per core
struct WriterRuntimeArgs {
    uint32_t dst_buffer_addr;           // Destination buffer address
    uint32_t core_tile_id;              // Starting tile ID for this core in output
    uint32_t num_tiles_arg;             // Number of tiles to process
    std::vector<uint32_t> shape_tiles;  // Output shape in tiles (rank elements)
    std::vector<uint32_t> tile_coord;   // Starting tile coordinate (rank elements)
};

// Simulates split_work_to_cores result
struct SplitWorkResult {
    uint32_t num_cores;
    uint32_t num_tiles_per_core;
    uint32_t num_tiles_per_core_cliff;
    uint32_t num_cores_in_core_group;
    uint32_t num_cores_in_core_group_cliff;
};

// Convert tile coordinate to tile ID
// This matches the coord_to_tile_id lambda in slice_program_factory_tile_interleaved.cpp
static uint32_t coord_to_tile_id(
    const std::vector<uint32_t>& coord, const std::vector<uint32_t>& shape, const std::vector<uint32_t>& zero_index) {
    uint32_t tile_id = 0;
    uint32_t multiplier = 1;
    for (int i = static_cast<int>(coord.size()) - 1; i >= 0; i--) {
        tile_id += (coord[i] + zero_index[i]) * multiplier;
        multiplier *= shape[i];
    }
    return tile_id;
}

// Simulate split_work_to_cores
static SplitWorkResult simulate_split_work_to_cores(uint32_t num_tiles) {
    // Simplified: assume 8 cores for demonstration
    const uint32_t max_cores = 64;
    SplitWorkResult result;

    result.num_cores = std::min(num_tiles, max_cores);
    result.num_tiles_per_core = num_tiles / result.num_cores;
    uint32_t remainder = num_tiles % result.num_cores;

    if (remainder == 0) {
        result.num_cores_in_core_group = result.num_cores;
        result.num_cores_in_core_group_cliff = 0;
        result.num_tiles_per_core_cliff = 0;
    } else {
        result.num_cores_in_core_group = remainder;
        result.num_cores_in_core_group_cliff = result.num_cores - remainder;
        result.num_tiles_per_core_cliff = result.num_tiles_per_core;
        result.num_tiles_per_core += 1;
    }

    return result;
}

// Result structure for reader kernel simulation
struct ReaderKernelSimResult {
    std::vector<uint32_t> tile_ids_read;             // Tile IDs read in order
    std::vector<std::vector<uint32_t>> tile_coords;  // Tile coordinates at each step
    uint32_t final_tile_id;                          // Final tile ID after loop
    std::vector<uint32_t> final_tile_coord;          // Final tile coordinate
};

// Result structure for writer kernel simulation
struct WriterKernelSimResult {
    std::vector<uint32_t> tile_ids_written;          // Tile IDs written in order
    std::vector<std::vector<uint32_t>> tile_coords;  // Tile coordinates at each step
    uint32_t final_tile_id;                          // Final tile ID after loop
    std::vector<uint32_t> final_tile_coord;          // Final tile coordinate
};

// Simulate the reader kernel execution
// This models slice_reader_unary_tile_row_col_interleaved.cpp
static ReaderKernelSimResult simulate_reader_kernel(
    const ReaderCompileTimeArgs& ct_args,
    const ReaderRuntimeArgs& rt_args,
    std::vector<uint32_t>& input_tile_access,  // Debug array: mark accessed input tiles
    bool verbose = false) {
    ReaderKernelSimResult result;
    result.tile_ids_read.reserve(rt_args.num_tiles_arg);
    result.tile_coords.reserve(rt_args.num_tiles_arg);

    // Initialize state (matching kernel_main)
    uint32_t src_tile_id = rt_args.core_tile_id;
    std::vector<uint32_t> tile_coord = rt_args.tile_coord;
    const uint32_t tile_id_stride = ct_args.src_tile_stride;
    const uint32_t num_dims = ct_args.rank;
    const std::vector<uint32_t>& shape_tiles = rt_args.out_shape_tiles;
    const std::vector<uint32_t>& tile_id_acc = rt_args.tile_id_acc;
    const std::vector<uint32_t>& coord_inc = rt_args.coord_inc;

    if (verbose) {
        std::cout << "\n=== Reader Kernel Simulation ===\n";
        std::cout << "Compile-time args:\n";
        std::cout << "  cb_id: " << ct_args.cb_id << "\n";
        std::cout << "  tile_id_stride: " << tile_id_stride << "\n";
        std::cout << "  num_dims: " << num_dims << "\n";
        std::cout << "  size_tile: " << ct_args.single_tile_size << "\n";
        std::cout << "Runtime args:\n";
        std::cout << "  src_addr: 0x" << std::hex << rt_args.src_buffer_addr << std::dec << "\n";
        std::cout << "  src_tile_id_start: " << rt_args.core_tile_id << "\n";
        std::cout << "  num_tiles: " << rt_args.num_tiles_arg << "\n";
        std::cout << "  shape_tiles: [";
        for (size_t i = 0; i < shape_tiles.size(); i++) {
            std::cout << shape_tiles[i];
            if (i < shape_tiles.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]\n";
        std::cout << "  tile_id_inc: [";
        for (size_t i = 0; i < coord_inc.size(); i++) {
            std::cout << coord_inc[i];
            if (i < coord_inc.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]\n";
        std::cout << "  initial tile_coord: [";
        for (size_t i = 0; i < tile_coord.size(); i++) {
            std::cout << tile_coord[i];
            if (i < tile_coord.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]\n\n";
        std::cout << "Execution trace:\n";
    }

    // Main loop (matching kernel_main)
    for (uint32_t i = 0; i < rt_args.num_tiles_arg; i++) {
        // Simulate: cb_reserve_back(cb_id, 1);
        // Simulate: noc_async_read_tile(src_tile_id, s, l1_write_addr);
        // Simulate: noc_async_read_barrier();
        // Simulate: cb_push_back(cb_id, 1);

        result.tile_ids_read.push_back(src_tile_id);
        result.tile_coords.push_back(tile_coord);

        // Mark this tile as accessed in the debug array (increment to detect multiple accesses)
        if (src_tile_id < input_tile_access.size()) {
            if (src_tile_id == 0) {
                std::cout << "  Iteration " << i << ": src_tile_id=" << src_tile_id << ", tile_coord=[";
                for (size_t j = 0; j < tile_coord.size(); j++) {
                    std::cout << tile_coord[j];
                    if (j < tile_coord.size() - 1) {
                        std::cout << ", ";
                    }
                }
                std::cout << "]\n";
            }
            input_tile_access[src_tile_id]++;
        } else {
            if (verbose) {
                std::cout << "  WARNING: src_tile_id " << src_tile_id << " exceeds input_tile_access size "
                          << input_tile_access.size() << "\n";
            }
        }

        if (verbose) {
            std::cout << "  Iteration " << i << ": src_tile_id=" << src_tile_id << ", tile_coord=[";
            for (size_t j = 0; j < tile_coord.size(); j++) {
                std::cout << tile_coord[j];
                if (j < tile_coord.size() - 1) {
                    std::cout << ", ";
                }
            }
            std::cout << "]\n";
        }

        if (1) {
            uint32_t manual_src_tile_id = tile_coord[2];
            manual_src_tile_id += tile_coord[1] * 5;
            manual_src_tile_id += tile_coord[0] * 120;
            if (manual_src_tile_id != src_tile_id or input_tile_access[src_tile_id] > 1) {
                std::cout << "  src_tile_id_start: " << rt_args.core_tile_id << "\n";
                std::cout << "        -- id error(" << input_tile_access[src_tile_id] << ") : " << src_tile_id << ", "
                          << manual_src_tile_id << "\n";
            }
        }

        for (int32_t j = static_cast<int32_t>(num_dims) - 1; j >= 1; j--) {
            tile_coord[j] += coord_inc[j];
            src_tile_id += coord_inc[j] * tile_id_acc[j];
            if (tile_coord[j] >= shape_tiles[j]) {
                tile_coord[j] -= shape_tiles[j];
                tile_coord[j - 1] += 1;
                src_tile_id += tile_id_acc[j - 1] - shape_tiles[j] * tile_id_acc[j];
            }
        }

#if 0  // to do
       // Update tile_id and tile_coord (matching kernel logic)
        src_tile_id += tile_id_stride;
        tile_coord[num_dims - 1] += tile_id_stride;

        // Handle coordinate wrapping and apply tile_id_inc
        for (int32_t j = static_cast<int32_t>(num_dims) - 1; j >= 0; j--) {
            if (tile_coord[j] >= shape_tiles[j]) {
                tile_coord[j] -= shape_tiles[j];
                if (j > 0) {
                    tile_coord[j - 1]++;
                }
                src_tile_id += tile_id_inc[j];
            } else {
                break;
            }
	    if (tile_coord[j] >= shape_tiles[j]) j++;
        }
#endif
    }

    result.final_tile_id = src_tile_id;
    result.final_tile_coord = tile_coord;

    if (verbose) {
        std::cout << "\nFinal state:\n";
        std::cout << "  final_tile_id: " << result.final_tile_id << "\n";
        std::cout << "  final_tile_coord: [";
        for (size_t i = 0; i < result.final_tile_coord.size(); i++) {
            std::cout << result.final_tile_coord[i];
            if (i < result.final_tile_coord.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]\n";
        std::cout << "  Total tiles read: " << result.tile_ids_read.size() << "\n";
    }

    return result;
}

// Simulate the writer kernel execution
// This models slice_writer_unary_tile_row_col_interleaved.cpp
static WriterKernelSimResult simulate_writer_kernel(
    const WriterCompileTimeArgs& ct_args,
    const WriterRuntimeArgs& rt_args,
    std::vector<uint32_t>& output_tile_access,  // Debug array: mark accessed output tiles
    bool verbose = false) {
    WriterKernelSimResult result;
    result.tile_ids_written.reserve(rt_args.num_tiles_arg);
    result.tile_coords.reserve(rt_args.num_tiles_arg);

    // Initialize state (matching kernel_main)
    uint32_t out_tile_id = rt_args.core_tile_id;
    const uint32_t tile_id_stride = ct_args.out_tile_stride;
    const uint32_t num_dims = ct_args.rank;
    const std::vector<uint32_t>& shape_tiles = rt_args.shape_tiles;
    std::vector<uint32_t> tile_coord = rt_args.tile_coord;

    if (verbose) {
        std::cout << "\n=== Writer Kernel Simulation ===\n";
        std::cout << "Compile-time args:\n";
        std::cout << "  cb_id: " << ct_args.cb_id << "\n";
        std::cout << "  tile_id_stride: " << tile_id_stride << "\n";
        std::cout << "  num_dims: " << num_dims << "\n";
        std::cout << "  size_tile: " << ct_args.single_tile_size << "\n";
        std::cout << "Runtime args:\n";
        std::cout << "  out_addr: 0x" << std::hex << rt_args.dst_buffer_addr << std::dec << "\n";
        std::cout << "  out_tile_id_start: " << rt_args.core_tile_id << "\n";
        std::cout << "  num_tiles: " << rt_args.num_tiles_arg << "\n";
        std::cout << "  shape_tiles: [";
        for (size_t i = 0; i < shape_tiles.size(); i++) {
            std::cout << shape_tiles[i];
            if (i < shape_tiles.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]\n\n";
        std::cout << "Execution trace:\n";
    }

    // Main loop (matching kernel_main)
    for (uint32_t i = 0; i < rt_args.num_tiles_arg; i++) {
        // Simulate: cb_wait_front(cb_id, 1);
        // Simulate: noc_async_write_page(out_tile_id, s, l1_read_addr);
        // Simulate: noc_async_writes_flushed();
        // Simulate: cb_pop_front(cb_id, 1);

        result.tile_ids_written.push_back(out_tile_id);
        result.tile_coords.push_back(tile_coord);

        // Mark this tile as accessed in the debug array
        if (out_tile_id < output_tile_access.size()) {
            output_tile_access[out_tile_id] = 1;
        } else {
            if (verbose) {
                std::cout << "  WARNING: out_tile_id " << out_tile_id << " exceeds output_tile_access size "
                          << output_tile_access.size() << "\n";
            }
        }

        if (verbose) {
            std::cout << "  Iteration " << i << ": out_tile_id=" << out_tile_id << ", tile_coord=[";
            for (size_t j = 0; j < tile_coord.size(); j++) {
                std::cout << tile_coord[j];
                if (j < tile_coord.size() - 1) {
                    std::cout << ", ";
                }
            }
            std::cout << "]\n";
        }

        // Update tile_id and tile_coord (matching kernel logic)
        out_tile_id += tile_id_stride;
        tile_coord[num_dims - 1] += tile_id_stride;

        // Handle coordinate wrapping (writer doesn't use tile_id_inc)
        for (int32_t j = static_cast<int32_t>(num_dims) - 1; j >= 1; j--) {
            while (tile_coord[j] >= shape_tiles[j]) {
                tile_coord[j] -= shape_tiles[j];
                tile_coord[j - 1]++;
            }
        }
    }

    result.final_tile_id = out_tile_id;
    result.final_tile_coord = tile_coord;

    if (verbose) {
        std::cout << "\nFinal state:\n";
        std::cout << "  final_tile_id: " << result.final_tile_id << "\n";
        std::cout << "  final_tile_coord: [";
        for (size_t i = 0; i < result.final_tile_coord.size(); i++) {
            std::cout << result.final_tile_coord[i];
            if (i < result.final_tile_coord.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]\n";
        std::cout << "  Total tiles written: " << result.tile_ids_written.size() << "\n";
    }

    return result;
}

// Demonstrate kernel execution for a specific core
static void demonstrate_kernel_execution(
    const ReaderCompileTimeArgs& reader_ct_args,
    const WriterCompileTimeArgs& writer_ct_args,
    const ReaderRuntimeArgs& reader_rt_args,
    const WriterRuntimeArgs& writer_rt_args,
    uint32_t core_idx,
    std::vector<uint32_t>& input_tile_access,   // Debug array for input tiles
    std::vector<uint32_t>& output_tile_access,  // Debug array for output tiles
    bool verbose = false) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "Kernel Execution Simulation for Core " << core_idx << "\n";
    std::cout << std::string(80, '=') << "\n";

    // bool verbose_local = (core_idx == 0) ? true : verbose;
    bool verbose_local = verbose;

    // Simulate reader kernel
    ReaderKernelSimResult reader_result =
        simulate_reader_kernel(reader_ct_args, reader_rt_args, input_tile_access, verbose_local);

    // Simulate writer kernel
    WriterKernelSimResult writer_result =
        simulate_writer_kernel(writer_ct_args, writer_rt_args, output_tile_access, verbose_local);

    // Summary
    std::cout << "\n=== Summary for Core " << core_idx << " ===\n";
    std::cout << "Reader kernel:\n";
    std::cout << "  Tiles read: " << reader_result.tile_ids_read.size() << "\n";
    if (!reader_result.tile_ids_read.empty()) {
        std::cout << "  First tile ID: " << reader_result.tile_ids_read[0] << "\n";
        std::cout << "  Last tile ID: " << reader_result.tile_ids_read.back() << "\n";
    }

    std::cout << "Writer kernel:\n";
    std::cout << "  Tiles written: " << writer_result.tile_ids_written.size() << "\n";
    if (!writer_result.tile_ids_written.empty()) {
        std::cout << "  First tile ID: " << writer_result.tile_ids_written[0] << "\n";
        std::cout << "  Last tile ID: " << writer_result.tile_ids_written.back() << "\n";
    }

    // Verify reader and writer process same number of tiles
    if (reader_result.tile_ids_read.size() != writer_result.tile_ids_written.size()) {
        std::cout << "  WARNING: Mismatch in number of tiles!\n";
    }
}

// Print the tile access arrays for debugging
static void print_tile_access_arrays(
    const std::vector<uint32_t>& input_tile_access,
    const std::vector<uint32_t>& output_tile_access,
    uint32_t num_input_tiles,
    uint32_t num_output_tiles,
    const Shape& input_padded_shape,
    const Shape& output_padded_shape) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "Tile Access Debug Arrays\n";
    std::cout << std::string(80, '=') << "\n\n";

    // Count accessed tiles
    uint32_t input_accessed_count = 0;
    uint32_t output_accessed_count = 0;

    for (uint32_t val : input_tile_access) {
        if (val == 1) {
            input_accessed_count++;
        }
    }

    for (uint32_t val : output_tile_access) {
        if (val == 1) {
            output_accessed_count++;
        }
    }

    std::cout << "Input tile access array:\n";
    std::cout << "  Total input tiles: " << num_input_tiles << "\n";
    std::cout << "  Tiles accessed: " << input_accessed_count << "\n";
    std::cout << "  Tiles not accessed: " << (num_input_tiles - input_accessed_count) << "\n";

    std::cout << "\nOutput tile access array:\n";
    std::cout << "  Total output tiles: " << num_output_tiles << "\n";
    std::cout << "  Tiles accessed: " << output_accessed_count << "\n";
    std::cout << "  Tiles not accessed: " << (num_output_tiles - output_accessed_count) << "\n";

    // Print access pattern (for smaller arrays, print full pattern)
    const uint32_t max_print_size = 1000;

    if (num_input_tiles <= max_print_size) {
        std::cout << "\nInput tile access pattern (1=accessed, 0=not accessed):\n";
        uint32_t src_tiles_per_row = input_padded_shape[input_padded_shape.size() - 1] / TILE_WIDTH;
        uint32_t src_tiles_per_col = input_padded_shape[input_padded_shape.size() - 2] / TILE_HEIGHT;

        for (uint32_t i = 0; i < num_input_tiles; i++) {
            if (i > 0 && i % src_tiles_per_row == 0) {
                std::cout << "\n";
            }
            std::cout << input_tile_access[i];
        }
        std::cout << "\n";
    } else {
        std::cout << "\nInput tile access array too large to print (size: " << num_input_tiles << ")\n";
        std::cout << "First 100 elements: ";
        for (uint32_t i = 0; i < std::min(100u, num_input_tiles); i++) {
            std::cout << input_tile_access[i];
        }
        std::cout << "...\n";
    }

    if (num_output_tiles <= max_print_size) {
        std::cout << "\nOutput tile access pattern (1=accessed, 0=not accessed):\n";
        uint32_t out_tiles_per_row = output_padded_shape[output_padded_shape.size() - 1] / TILE_WIDTH;

        for (uint32_t i = 0; i < num_output_tiles; i++) {
            if (i > 0 && i % out_tiles_per_row == 0) {
                std::cout << "\n";
            }
            std::cout << output_tile_access[i];
        }
        std::cout << "\n";
    } else {
        std::cout << "\nOutput tile access array too large to print (size: " << num_output_tiles << ")\n";
        std::cout << "First 100 elements: ";
        for (uint32_t i = 0; i < std::min(100u, num_output_tiles); i++) {
            std::cout << output_tile_access[i];
        }
        std::cout << "...\n";
    }

    // Check for unaccessed tiles (potential issues)
    std::vector<uint32_t> unaccessed_input_tiles;
    std::vector<uint32_t> unaccessed_output_tiles;

    for (uint32_t i = 0; i < num_input_tiles; i++) {
        if (input_tile_access[i] == 0) {
            unaccessed_input_tiles.push_back(i);
        }
    }

    for (uint32_t i = 0; i < num_output_tiles; i++) {
        if (output_tile_access[i] == 0) {
            unaccessed_output_tiles.push_back(i);
        }
    }

    if (!unaccessed_input_tiles.empty()) {
        std::cout << "\nWARNING: " << unaccessed_input_tiles.size() << " input tiles were not accessed!\n";
        if (unaccessed_input_tiles.size() <= 20) {
            std::cout << "Unaccessed input tile IDs: ";
            for (uint32_t id : unaccessed_input_tiles) {
                std::cout << id << " ";
            }
            std::cout << "\n";
        }
    }

    if (!unaccessed_output_tiles.empty()) {
        std::cout << "\nWARNING: " << unaccessed_output_tiles.size() << " output tiles were not accessed!\n";
        if (unaccessed_output_tiles.size() <= 20) {
            std::cout << "Unaccessed output tile IDs: ";
            for (uint32_t id : unaccessed_output_tiles) {
                std::cout << id << " ";
            }
            std::cout << "\n";
        }
    }

    // Check for multiple accesses (potential issues)
    std::vector<uint32_t> multiple_access_input;
    std::vector<uint32_t> multiple_access_output;

    for (uint32_t i = 0; i < num_input_tiles; i++) {
        if (input_tile_access[i] > 1) {
            multiple_access_input.push_back(i);
        }
    }

    for (uint32_t i = 0; i < num_output_tiles; i++) {
        if (output_tile_access[i] > 1) {
            multiple_access_output.push_back(i);
        }
    }

    if (!multiple_access_input.empty()) {
        std::cout << "\nWARNING: " << multiple_access_input.size() << " input tiles were accessed multiple times!\n";
        if (multiple_access_input.size() <= 20) {
            std::cout << "Multiple-access input tile IDs: ";
            for (uint32_t id : multiple_access_input) {
                std::cout << id << "(" << input_tile_access[id] << "x) ";
            }
            std::cout << "\n";
        }
    }

    if (!multiple_access_output.empty()) {
        std::cout << "\nWARNING: " << multiple_access_output.size() << " output tiles were accessed multiple times!\n";
        if (multiple_access_output.size() <= 20) {
            std::cout << "Multiple-access output tile IDs: ";
            for (uint32_t id : multiple_access_output) {
                std::cout << id << "(" << output_tile_access[id] << "x) ";
            }
            std::cout << "\n";
        }
    }

    std::cout << "\n" << std::string(80, '=') << "\n";
}

// Main function that simulates slice_program_factory_tile_interleaved::create()
static void simulate_slice_tile_interleaved_create(
    const Shape& input_padded_shape,
    const Shape& output_padded_shape,
    const Shape& slice_start,
    const Shape& step,
    bool verbose_kernels = false) {
    const uint32_t rank = static_cast<uint32_t>(input_padded_shape.size());

    std::cout << "=== Slice Tile Interleaved Program Factory Simulation ===\n\n";
    std::cout << "Input padded shape: [";
    for (size_t i = 0; i < input_padded_shape.size(); i++) {
        std::cout << input_padded_shape[i];
        if (i < input_padded_shape.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]\n";

    std::cout << "Output padded shape: [";
    for (size_t i = 0; i < output_padded_shape.size(); i++) {
        std::cout << output_padded_shape[i];
        if (i < output_padded_shape.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]\n";

    std::cout << "Slice start: [";
    for (size_t i = 0; i < slice_start.size(); i++) {
        std::cout << slice_start[i];
        if (i < slice_start.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]\n";

    std::cout << "Rank: " << rank << "\n\n";

    // Calculate number of tiles
    uint32_t num_output_tiles = 1;
    for (uint32_t dim : output_padded_shape) {
        num_output_tiles *= dim;
    }
    num_output_tiles /= TILE_HW;

    uint32_t num_input_tiles = 1;
    for (uint32_t dim : input_padded_shape) {
        num_input_tiles *= dim;
    }
    num_input_tiles /= TILE_HW;

    std::cout << "Total input tiles: " << num_input_tiles << "\n";
    std::cout << "Total output tiles: " << num_output_tiles << "\n";

    // Create debug arrays to track tile access
    std::vector<uint32_t> input_tile_access(num_input_tiles, 0);
    std::vector<uint32_t> output_tile_access(num_output_tiles, 0);

    // Simulate split_work_to_cores
    SplitWorkResult split = simulate_split_work_to_cores(num_output_tiles);
    std::cout << "Split work result:\n";
    std::cout << "  num_cores: " << split.num_cores << "\n";
    std::cout << "  num_tiles_per_core: " << split.num_tiles_per_core << "\n";
    std::cout << "  num_tiles_per_core_cliff: " << split.num_tiles_per_core_cliff << "\n";
    std::cout << "  num_cores_in_core_group: " << split.num_cores_in_core_group << "\n";
    std::cout << "  num_cores_in_core_group_cliff: " << split.num_cores_in_core_group_cliff << "\n\n";

    // Calculate tiles per row/col
    uint32_t out_tiles_per_row = output_padded_shape[rank - 1] / TILE_WIDTH;
    uint32_t out_tiles_per_col = output_padded_shape[rank - 2] / TILE_HEIGHT;
    uint32_t src_tiles_per_row = input_padded_shape[rank - 1] / TILE_WIDTH;
    uint32_t src_tiles_per_col = input_padded_shape[rank - 2] / TILE_HEIGHT;

    std::cout << "Tiles per row/col:\n";
    std::cout << "  src_tiles_per_row: " << src_tiles_per_row << "\n";
    std::cout << "  src_tiles_per_col: " << src_tiles_per_col << "\n";
    std::cout << "  out_tiles_per_row: " << out_tiles_per_row << "\n";
    std::cout << "  out_tiles_per_col: " << out_tiles_per_col << "\n\n";

    // Prepare compile-time arguments
    const uint32_t single_tile_size = 1024;  // Example: 1024 bytes per tile
    const uint32_t src_tile_stride = split.num_cores;
    const uint32_t out_tile_stride = split.num_cores;

    ReaderCompileTimeArgs reader_ct_args = {
        .cb_id = 0, .src_tile_stride = src_tile_stride, .rank = rank, .single_tile_size = single_tile_size};

    WriterCompileTimeArgs writer_ct_args = {
        .cb_id = 0, .out_tile_stride = out_tile_stride, .rank = rank, .single_tile_size = single_tile_size};

    std::cout << "Reader compile-time args:\n";
    std::cout << "  cb_id: " << reader_ct_args.cb_id << "\n";
    std::cout << "  src_tile_stride: " << reader_ct_args.src_tile_stride << "\n";
    std::cout << "  rank: " << reader_ct_args.rank << "\n";
    std::cout << "  single_tile_size: " << reader_ct_args.single_tile_size << "\n\n";

    std::cout << "Writer compile-time args:\n";
    std::cout << "  cb_id: " << writer_ct_args.cb_id << "\n";
    std::cout << "  out_tile_stride: " << writer_ct_args.out_tile_stride << "\n";
    std::cout << "  rank: " << writer_ct_args.rank << "\n";
    std::cout << "  single_tile_size: " << writer_ct_args.single_tile_size << "\n\n";

    // Calculate shape tiles and tile ID increments
    std::vector<uint32_t> src_shape_tiles(rank);
    std::vector<uint32_t> out_shape_tiles(rank);
    std::vector<uint32_t> src_coord_inc(rank, 0);
    std::vector<uint32_t> src_tile_id_acc(rank);
    std::vector<uint32_t> src_tile_start(rank);
    std::vector<uint32_t> out_tile_start(rank, 0);

    uint32_t size_acc_src = 1;
    for (int32_t i = static_cast<int32_t>(rank) - 1; i >= 0; i--) {
        src_tile_id_acc[i] = size_acc_src;
        if (i == static_cast<int32_t>(rank) - 1) {
            // Last dimension (width)
            out_shape_tiles[i] = out_tiles_per_row;
            src_shape_tiles[i] = src_tiles_per_row;
            src_tile_start[i] = slice_start[i] / TILE_WIDTH;
            size_acc_src *= src_tiles_per_row;
        } else if (i == static_cast<int32_t>(rank) - 2) {
            // Second-to-last dimension (height)
            out_shape_tiles[i] = out_tiles_per_col;
            src_shape_tiles[i] = src_tiles_per_col;
            src_tile_start[i] = slice_start[i] / TILE_HEIGHT;
            size_acc_src *= src_tiles_per_col;
        } else {
            // Other dimensions
            int size_unpadded_dim = static_cast<int>(input_padded_shape[i]) - static_cast<int>(output_padded_shape[i]);
            out_shape_tiles[i] = output_padded_shape[i];
            src_shape_tiles[i] = input_padded_shape[i];
            src_tile_start[i] = slice_start[i];
            size_acc_src *= input_padded_shape[i];
        }
    }
    src_coord_inc[rank - 1] = out_tile_stride;
    for (int32_t i = static_cast<int32_t>(rank) - 1; i >= 1; i--) {
        if (src_coord_inc[i] >= out_shape_tiles[i]) {
            src_coord_inc[i - 1] += (src_coord_inc[i] / out_shape_tiles[i]);
            src_coord_inc[i] = src_coord_inc[i] % out_shape_tiles[i];
        }
    }

    std::cout << "Shape tiles and increments:\n";
    std::cout << "  src_shape_tiles: [";
    for (size_t i = 0; i < src_shape_tiles.size(); i++) {
        std::cout << src_shape_tiles[i];
        if (i < src_shape_tiles.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]\n";

    std::cout << "  out_shape_tiles: [";
    for (size_t i = 0; i < out_shape_tiles.size(); i++) {
        std::cout << out_shape_tiles[i];
        if (i < out_shape_tiles.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]\n";

    std::cout << "  src_tile_id_acc: [";
    for (size_t i = 0; i < src_tile_id_acc.size(); i++) {
        std::cout << src_tile_id_acc[i];
        if (i < src_tile_id_acc.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]\n";

    std::cout << "  src_coord_inc: [";
    for (size_t i = 0; i < src_coord_inc.size(); i++) {
        std::cout << src_coord_inc[i];
        if (i < src_coord_inc.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]\n";

    std::cout << "  src_tile_start: [";
    for (size_t i = 0; i < src_tile_start.size(); i++) {
        std::cout << src_tile_start[i];
        if (i < src_tile_start.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]\n\n";

    // Generate runtime arguments for each core and run kernels to populate debug arrays
    std::vector<uint32_t> tile_coord(rank, 0);
    const uint32_t src_buffer_addr = 0x10000000;  // Example address
    const uint32_t dst_buffer_addr = 0x20000000;  // Example address

    std::cout << "Runtime arguments per core:\n";
    std::cout << std::string(80, '-') << "\n";

    // Run kernels for all cores to populate debug arrays
    for (uint32_t core_idx = 0; core_idx < split.num_cores; core_idx++) {
        uint32_t num_tiles_arg = split.num_tiles_per_core;
        if (core_idx >= split.num_cores_in_core_group) {
            num_tiles_arg = split.num_tiles_per_core_cliff;
        }

        // Calculate tile IDs
        uint32_t src_core_tile_id = coord_to_tile_id(tile_coord, src_shape_tiles, src_tile_start);
        uint32_t out_core_tile_id = coord_to_tile_id(tile_coord, out_shape_tiles, out_tile_start);

        ReaderRuntimeArgs reader_rt_args = {
            .src_buffer_addr = src_buffer_addr,
            .core_tile_id = src_core_tile_id,  // source id
            .num_tiles_arg = num_tiles_arg,
            .out_shape_tiles = out_shape_tiles,
            .tile_coord = tile_coord,  // output coordinate
            .tile_id_acc = src_tile_id_acc,
            .coord_inc = src_coord_inc};

        WriterRuntimeArgs writer_rt_args = {
            .dst_buffer_addr = dst_buffer_addr,
            .core_tile_id = out_core_tile_id,
            .num_tiles_arg = num_tiles_arg,
            .shape_tiles = out_shape_tiles,
            .tile_coord = tile_coord  // output coordinate
        };

        // Print core information
        std::cout << "Core " << core_idx << ":\n";
        std::cout << "  Reader RT Args:\n";
        std::cout << "    src_buffer_addr: 0x" << std::hex << reader_rt_args.src_buffer_addr << std::dec << "\n";
        std::cout << "    src_core_tile_id: " << reader_rt_args.core_tile_id << "\n";
        std::cout << "    num_tiles_arg: " << reader_rt_args.num_tiles_arg << "\n";
        std::cout << "    tile_coord: [";
        for (size_t i = 0; i < reader_rt_args.tile_coord.size(); i++) {
            std::cout << reader_rt_args.tile_coord[i];
            if (i < reader_rt_args.tile_coord.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]\n";

        std::cout << "  Writer RT Args:\n";
        std::cout << "    dst_buffer_addr: 0x" << std::hex << writer_rt_args.dst_buffer_addr << std::dec << "\n";
        std::cout << "    out_core_tile_id: " << writer_rt_args.core_tile_id << "\n";
        std::cout << "    num_tiles_arg: " << writer_rt_args.num_tiles_arg << "\n";
        std::cout << "    tile_coord: [";
        for (size_t i = 0; i < writer_rt_args.tile_coord.size(); i++) {
            std::cout << writer_rt_args.tile_coord[i];
            if (i < writer_rt_args.tile_coord.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]\n";
        std::cout << std::string(80, '-') << "\n";

        // Run kernels for this core to populate debug arrays
        simulate_reader_kernel(reader_ct_args, reader_rt_args, input_tile_access, false);
        simulate_writer_kernel(writer_ct_args, writer_rt_args, output_tile_access, false);

        // Advance tile_coord for next core
        tile_coord[rank - 1]++;
        for (int j = static_cast<int>(rank) - 1; j >= 1; j--) {
            if (tile_coord[j] >= out_shape_tiles[j]) {
                const uint32_t carry = tile_coord[j] / out_shape_tiles[j];
                tile_coord[j] = tile_coord[j] % out_shape_tiles[j];
                tile_coord[j - 1] += carry;
            } else {
                break;
            }
        }
    }

    // Demonstrate kernel execution for first core (with verbose output)
    if (split.num_cores > 0) {
        // Recalculate for first core
        std::vector<uint32_t> first_core_tile_coord(rank, 0);
        std::vector<uint32_t> first_src_tile_coord(rank);
        uint32_t first_core_num_tiles = split.num_tiles_per_core;

        uint32_t first_src_tile_id = coord_to_tile_id(first_core_tile_coord, src_shape_tiles, src_tile_start);
        uint32_t first_out_tile_id = coord_to_tile_id(first_core_tile_coord, out_shape_tiles, out_tile_start);

        first_src_tile_coord = src_tile_start;

        ReaderRuntimeArgs first_reader_rt_args = {
            .src_buffer_addr = src_buffer_addr,
            .core_tile_id = first_src_tile_id,
            .num_tiles_arg = first_core_num_tiles,
            .out_shape_tiles = out_shape_tiles,
            .tile_coord = first_src_tile_coord,
            .tile_id_acc = src_tile_id_acc,
            .coord_inc = src_coord_inc};

        WriterRuntimeArgs first_writer_rt_args = {
            .dst_buffer_addr = dst_buffer_addr,
            .core_tile_id = first_out_tile_id,
            .num_tiles_arg = first_core_num_tiles,
            .shape_tiles = out_shape_tiles,
            .tile_coord = first_core_tile_coord};

#if 0
        // Demonstrate kernel execution with verbose output
        demonstrate_kernel_execution(
            reader_ct_args,
            writer_ct_args,
            first_reader_rt_args,
            first_writer_rt_args,
            0,
            input_tile_access,
            output_tile_access,
            verbose_kernels
        );
#endif
    }

    // Print debug arrays
    print_tile_access_arrays(
        input_tile_access,
        output_tile_access,
        num_input_tiles,
        num_output_tiles,
        input_padded_shape,
        output_padded_shape);

    std::cout << "\n=== Simulation Complete ===\n";
}

int main(int argc, char* argv[]) {
    // Default test case: 3D tensor
    Shape input_padded_shape = {33, 1, 7, 32, 64};
    Shape output_padded_shape = {33, 1, 7, 32, 64};
    Shape slice_start = {0, 0, 0};
    Shape step = {1, 1, 1, 1, 1};
    bool verbose_kernels = false;

    int rank = input_padded_shape.size();

    // Parse command line arguments if provided
    // Usage: prog [input0 input1 input2 output0 output1 output2 start0 start1 start2] [--verbose]
    // Or: prog [--verbose] to use defaults with verbose kernel output
    if (argc >= 2 && std::string(argv[1]) == "--verbose") {
        verbose_kernels = true;
        if (argc == 11) {
            // Skip --verbose and parse shapes
            input_padded_shape = {
                static_cast<uint32_t>(std::stoi(argv[2])),
                static_cast<uint32_t>(std::stoi(argv[3])),
                static_cast<uint32_t>(std::stoi(argv[4]))};
            output_padded_shape = {
                static_cast<uint32_t>(std::stoi(argv[5])),
                static_cast<uint32_t>(std::stoi(argv[6])),
                static_cast<uint32_t>(std::stoi(argv[7]))};
            slice_start = {
                static_cast<uint32_t>(std::stoi(argv[8])),
                static_cast<uint32_t>(std::stoi(argv[9])),
                static_cast<uint32_t>(std::stoi(argv[10]))};
        }
    } else if (argc == 10) {
        input_padded_shape = {
            static_cast<uint32_t>(std::stoi(argv[1])),
            static_cast<uint32_t>(std::stoi(argv[2])),
            static_cast<uint32_t>(std::stoi(argv[3]))};
        output_padded_shape = {
            static_cast<uint32_t>(std::stoi(argv[4])),
            static_cast<uint32_t>(std::stoi(argv[5])),
            static_cast<uint32_t>(std::stoi(argv[6]))};
        slice_start = {
            static_cast<uint32_t>(std::stoi(argv[7])),
            static_cast<uint32_t>(std::stoi(argv[8])),
            static_cast<uint32_t>(std::stoi(argv[9]))};
    }

    // Ensure dimensions are tile-aligned
    auto pad_to_tile = [](uint32_t v, uint32_t tile_size) {
        return (v % tile_size) ? (v + (tile_size - v % tile_size)) : v;
    };

    input_padded_shape[rank - 2] = pad_to_tile(input_padded_shape[rank - 2], TILE_HEIGHT);
    input_padded_shape[rank - 1] = pad_to_tile(input_padded_shape[rank - 1], TILE_WIDTH);
    output_padded_shape[rank - 2] = pad_to_tile(output_padded_shape[rank - 2], TILE_HEIGHT);
    output_padded_shape[rank - 1] = pad_to_tile(output_padded_shape[rank - 1], TILE_WIDTH);

    try {
        simulate_slice_tile_interleaved_create(
            input_padded_shape, output_padded_shape, slice_start, step, verbose_kernels);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
