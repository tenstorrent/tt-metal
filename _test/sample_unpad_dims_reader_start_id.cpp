// Standalone sample that mirrors:
//  - SliceTileProgramFactory runtime-arg math in:
//      ttnn/.../slice_program_factory_tile.cpp  (set_slice_runtime_args_tile)
//  - Reader kernel src_tile_id progression in:
//      ttnn/.../kernels/dataflow/reader_unary_unpad_dims_interleaved_start_id.cpp
//
// Focus: verify how start_id + id_per_dim drive src_tile_id (ignoring the actual noc reads).
//
// Build:
//   g++ -std=c++20 -O2 -Wall -Wextra -pedantic _test/sample_unpad_dims_reader_start_id.cpp -o sample_unpad

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

static constexpr uint32_t TILE_WIDTH = 32;
static constexpr uint32_t TILE_HEIGHT = 32;

using Shape = std::vector<uint32_t>;  // padded shapes (rank >= 2)

struct SplitWorkResult {
    // Mirrors split_work_to_cores outputs (simplified for this sample)
    uint32_t num_cores = 1;
    uint32_t num_cores_in_group_1 = 1;  // "core_group_1" (more work)
    uint32_t num_cores_in_group_2 = 0;  // "core_group_2" (less work)
    uint32_t tiles_per_core_group_1 = 0;
    uint32_t tiles_per_core_group_2 = 0;
};

struct CommonReaderState {
    // These correspond to "common runtime args" in SliceTileProgramFactory
    std::vector<uint32_t> num_unpadded_tiles_per_dim;  // size num_dims
    std::vector<uint32_t> num_padded_tiles_per_dim;    // size num_dims
    std::vector<uint32_t> accumulated_total_per_dim;   // size num_dims
    uint32_t start_offset = 0;                         // get_tiled_start_offset equivalent (tile-id space)
};

struct PerCoreReaderArgs {
    // These correspond to per-core runtime args set for the reader kernel
    uint32_t start_id = 0;             // arg 0
    uint32_t num_tiles = 0;            // arg 1
    std::vector<uint32_t> id_per_dim;  // arg 2.. (size num_dims)
};

static uint32_t get_tiled_start_offset_like_ttnn(const Shape& input_padded_shape, const Shape& slice_start) {
    // Matches the intent of get_tiled_start_offset(input_tensor, slice_start):
    // - last dim: element index / TILE_WIDTH
    // - second last dim: element index / TILE_HEIGHT
    // - higher dims: assumed already in "tile units" (like ttnn tiled tensors)
    const int rank = static_cast<int>(input_padded_shape.size());
    if (rank < 2 || static_cast<int>(slice_start.size()) != rank) {
        throw std::invalid_argument("shape/start must have same rank >= 2");
    }

    const uint32_t num_total_Xt = input_padded_shape[rank - 1] / TILE_WIDTH;
    const uint32_t num_total_Yt = input_padded_shape[rank - 2] / TILE_HEIGHT;

    uint32_t start_id = 0;
    start_id += (slice_start[rank - 1] / TILE_WIDTH);
    start_id += (slice_start[rank - 2] / TILE_HEIGHT) * num_total_Xt;

    // accumulated_total_per_dim[1] == num_total_Yt * num_total_Xt
    uint32_t acc = num_total_Yt * num_total_Xt;
    for (int i = 2; i < rank; ++i) {
        const uint32_t dim_index = rank - 1 - i;  // walk from third-last toward dim0
        start_id += slice_start[dim_index] * acc;
        acc *= input_padded_shape[dim_index];
    }
    return start_id;
}

static CommonReaderState compute_common_reader_state(
    const Shape& input_padded_shape, const Shape& output_padded_shape, const Shape& slice_start) {
    const int rank = static_cast<int>(input_padded_shape.size());
    if (rank < 2 || output_padded_shape.size() != input_padded_shape.size() ||
        slice_start.size() != input_padded_shape.size()) {
        throw std::invalid_argument("input/output/start must have same rank >= 2");
    }

    const uint32_t num_dims = static_cast<uint32_t>(rank);

    CommonReaderState st;
    st.num_unpadded_tiles_per_dim.assign(num_dims, 0);
    st.num_padded_tiles_per_dim.assign(num_dims, 0);
    st.accumulated_total_per_dim.assign(num_dims, 0);

    // Exactly like set_common_reader_args() in slice_program_factory_tile.cpp
    const uint32_t num_unpadded_Xt = output_padded_shape[rank - 1] / TILE_WIDTH;
    const uint32_t num_total_Xt = input_padded_shape[rank - 1] / TILE_WIDTH;
    const uint32_t num_padded_Xt = num_total_Xt - num_unpadded_Xt;

    const uint32_t num_unpadded_Yt = output_padded_shape[rank - 2] / TILE_HEIGHT;
    const uint32_t num_total_Yt = input_padded_shape[rank - 2] / TILE_HEIGHT;
    const uint32_t num_padded_Yt = (num_total_Yt - num_unpadded_Yt) * num_total_Xt;

    st.num_unpadded_tiles_per_dim[0] = num_unpadded_Xt;
    st.num_unpadded_tiles_per_dim[1] = num_unpadded_Yt;
    st.num_padded_tiles_per_dim[0] = num_padded_Xt;
    st.num_padded_tiles_per_dim[1] = num_padded_Yt;
    st.accumulated_total_per_dim[0] = num_total_Xt;
    st.accumulated_total_per_dim[1] = num_total_Yt * num_total_Xt;

    for (int32_t i = 2; i < static_cast<int32_t>(num_dims); ++i) {
        // Uses raw padded_shape values for dims >= 2, same as the factory code
        const uint32_t num_unpadded_dim = output_padded_shape[rank - 1 - i];
        const uint32_t num_total_dim = input_padded_shape[rank - 1 - i];
        const uint32_t num_padded_dim = (num_total_dim - num_unpadded_dim) * st.accumulated_total_per_dim[i - 1];
        st.num_unpadded_tiles_per_dim[i] = num_unpadded_dim;
        st.num_padded_tiles_per_dim[i] = num_padded_dim;
        st.accumulated_total_per_dim[i] = num_total_dim * st.accumulated_total_per_dim[i - 1];
    }

    st.start_offset = get_tiled_start_offset_like_ttnn(input_padded_shape, slice_start);
    return st;
}

static PerCoreReaderArgs compute_per_core_reader_args_like_ttnn(
    const CommonReaderState& st, uint32_t num_tiles_per_core, uint32_t num_tiles_written_so_far) {
    const uint32_t num_dims = static_cast<uint32_t>(st.num_unpadded_tiles_per_dim.size());
    PerCoreReaderArgs args;
    args.num_tiles = num_tiles_per_core;
    args.id_per_dim.assign(num_dims, 0);

    // Mirrors set_reader_rt_args() in slice_program_factory_tile.cpp
    args.id_per_dim[0] = num_tiles_written_so_far % st.num_unpadded_tiles_per_dim[0];
    uint32_t unpadded_written = num_tiles_written_so_far / st.num_unpadded_tiles_per_dim[0];
    uint32_t start_id = args.id_per_dim[0] + st.start_offset;
    for (uint32_t j = 1; j < num_dims; ++j) {
        args.id_per_dim[j] = unpadded_written % st.num_unpadded_tiles_per_dim[j];
        unpadded_written = unpadded_written / st.num_unpadded_tiles_per_dim[j];
        start_id += args.id_per_dim[j] * st.accumulated_total_per_dim[j - 1];
    }
    args.start_id = start_id;
    return args;
}

// Mirrors the kernel's src_tile_id + id_per_dim update logic exactly (ignoring noc reads).
static std::vector<uint32_t> simulate_reader_kernel_src_tile_ids(
    uint32_t start_id,
    uint32_t num_tiles,
    std::vector<uint32_t> id_per_dim,
    const std::vector<uint32_t>& num_unpadded_tiles_per_dim,
    const std::vector<uint32_t>& num_padded_tiles_per_dim) {
    const uint32_t num_dims = static_cast<uint32_t>(num_unpadded_tiles_per_dim.size());
    std::vector<uint32_t> ids;
    ids.reserve(num_tiles);

    uint32_t src_tile_id = start_id;
    for (uint32_t i = 0; i < num_tiles; ++i) {
        ids.push_back(src_tile_id);
        src_tile_id++;
        for (uint32_t j = 0; j < num_dims; ++j) {
            id_per_dim[j]++;
            if (id_per_dim[j] == num_unpadded_tiles_per_dim[j]) {
                id_per_dim[j] = 0;
                src_tile_id += num_padded_tiles_per_dim[j];
            } else {
                break;
            }
        }
    }
    return ids;
}

int main() {
    // Example shapes (feel free to edit)
    // Rank-3 example: [D0, H, W] in elements for last two dims, and "tile units" for dim0 in tiled layout.
#if 0
    Shape input = {768, 768, 160};
    Shape output = {686, 640, 128};
    Shape slice_start = {4, 32, 32};
#else
    Shape input = {768, 768, 128};
    Shape output = {686, 686, 128};
    Shape slice_start = {0, 0, 0};
#endif

    // Pad H/W and start H/W to multiples of tile, like typical tiled paths
    auto pad32 = [](uint32_t v) { return (v % 32) ? (v + (32 - v % 32)) : v; };
    input[1] = pad32(input[1]);
    input[2] = pad32(input[2]);
    output[1] = pad32(output[1]);
    output[2] = pad32(output[2]);
    slice_start[1] = pad32(slice_start[1]);
    slice_start[2] = pad32(slice_start[2]);

    const auto st = compute_common_reader_state(input, output, slice_start);

    std::cout << "CommonReaderState\n";
    std::cout << "  start_offset=" << st.start_offset << "\n";
    std::cout << "  num_unpadded_tiles_per_dim:";
    for (auto v : st.num_unpadded_tiles_per_dim) {
        std::cout << " " << v;
    }
    std::cout << "\n";
    std::cout << "  num_padded_tiles_per_dim:";
    for (auto v : st.num_padded_tiles_per_dim) {
        std::cout << " " << v;
    }
    std::cout << "\n";
    std::cout << "  accumulated_total_per_dim:";
    for (auto v : st.accumulated_total_per_dim) {
        std::cout << " " << v;
    }
    std::cout << "\n";

    // Total unpadded tiles (same as output.physical_volume()/TILE_HW for tiled tensors)
    const uint32_t num_unpadded_tiles = output[0] * (output[1] / TILE_HEIGHT) * (output[2] / TILE_WIDTH);

    // Provide (or hardcode) a split_work_to_cores result:
    // (Here: naive even split for demonstration.)
    SplitWorkResult split;
    split.num_cores = 64;
    split.num_cores_in_group_1 = 16;
    split.num_cores_in_group_2 = split.num_cores - split.num_cores_in_group_1;
    split.tiles_per_core_group_1 = 944;
    split.tiles_per_core_group_2 =
        (num_unpadded_tiles - split.num_cores_in_group_1 * split.tiles_per_core_group_1) / split.num_cores_in_group_2;

    // Print out SplitWorkResult 'split'
    std::cout << "SplitWorkResult:\n";
    std::cout << "  num_unpadded_tiles=" << num_unpadded_tiles << "\n";
    std::cout << "  num_cores=" << split.num_cores << "\n";
    std::cout << "  num_cores_in_group_1=" << split.num_cores_in_group_1 << "\n";
    std::cout << "  num_cores_in_group_2=" << split.num_cores_in_group_2 << "\n";
    std::cout << "  tiles_per_core_group_1=" << split.tiles_per_core_group_1 << "\n";
    std::cout << "  tiles_per_core_group_2=" << split.tiles_per_core_group_2 << "\n";

    // Calculate the total number of tiles distributed across all cores
    uint32_t total_tiles = split.num_cores_in_group_1 * split.tiles_per_core_group_1 +
                           split.num_cores_in_group_2 * split.tiles_per_core_group_2;
    std::cout << "Total tiles distributed across cores = " << total_tiles << "\n";
    if (total_tiles != num_unpadded_tiles) {
        std::cerr << "ERROR: Sum of tiles distributed across cores (" << total_tiles
                  << ") does not match num_unpadded_tiles (" << num_unpadded_tiles << ")\n";
        throw std::runtime_error("Tile distribution error: mismatch between total_tiles and num_unpadded_tiles");
    }

    // Simulate per-core runtime args and kernel src_tile_id sequence
    uint32_t num_tiles_written = 0;
    std::vector<uint32_t> last_ids;
    for (uint32_t core = 0; core < split.num_cores; ++core) {
        const uint32_t tiles_this_core = split.tiles_per_core_group_1;
        if (num_tiles_written >= num_unpadded_tiles) {
            break;
        }
        const uint32_t capped_tiles = std::min(tiles_this_core, num_unpadded_tiles - num_tiles_written);

        auto per_core = compute_per_core_reader_args_like_ttnn(st, capped_tiles, num_tiles_written);
        auto ids = simulate_reader_kernel_src_tile_ids(
            per_core.start_id,
            per_core.num_tiles,
            per_core.id_per_dim,
            st.num_unpadded_tiles_per_dim,
            st.num_padded_tiles_per_dim);

        std::cout << "core " << core << " num_tiles_written=" << num_tiles_written << " start_id=" << per_core.start_id
                  << " num_tiles=" << per_core.num_tiles << "\n";

        if (!last_ids.empty() && !ids.empty()) {
            // quick continuity check: next core's first id should be >= previous core's last id
            std::cout << "  prev_last_id=" << last_ids.back() << " next_first_id=" << ids.front() << "\n";
        }

        last_ids = std::move(ids);
        num_tiles_written += capped_tiles;
    }
}
