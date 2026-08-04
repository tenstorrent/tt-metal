// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/command_queue_fixture.hpp"

#include <algorithm>
#include <functional>
#include <random>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/distributed.hpp>
#include "tt_metal/test_utils/deprecated/tensor.hpp"
#include <tt-metalium/tilize_utils.hpp>

using std::vector;
using namespace tt;
using std::string;
using namespace tt::tt_metal;

static std::vector<bfloat16> make_identity_scale_tile(uint32_t tile_height = tt::constants::TILE_HEIGHT) {
    std::vector<bfloat16> tile(tile_height * tt::constants::TILE_WIDTH, static_cast<bfloat16>(1.0f));
    return tile;
}

// Tilize a row-major (rows x cols) buffer into GENUINE 16x32 tiles: each tile is two 16x16 faces
// F0 (cols 0-15) then F1 (cols 16-31), both spanning rows 0-15 of the tile; tiles in row-major order.
// tilize_nfaces() only produces 32x32 tiles, so a 16x32 (num_faces=2) tile needs this explicit layout —
// this is exactly the layout the SDPA matmul packs into cb_qkt_im, and is what exposes the tiny reduce bug.
static std::vector<bfloat16> tilize_16x32(const std::vector<bfloat16>& rm, uint32_t rows, uint32_t cols) {
    constexpr uint32_t TH = 16, TW = 32, FH = 16, FW = 16;
    std::vector<bfloat16> out(rows * cols);
    uint32_t idx = 0;
    for (uint32_t ti = 0; ti < rows / TH; ++ti) {
        for (uint32_t tj = 0; tj < cols / TW; ++tj) {
            for (uint32_t face = 0; face < 2; ++face) {
                const uint32_t c0 = tj * TW + face * FW;
                for (uint32_t r = 0; r < FH; ++r) {
                    for (uint32_t c = 0; c < FW; ++c) {
                        out[idx++] = rm[(ti * TH + r) * cols + (c0 + c)];
                    }
                }
            }
        }
    }
    return out;
}

// Inverse of tilize_16x32.
static std::vector<bfloat16> untilize_16x32(const std::vector<bfloat16>& til, uint32_t rows, uint32_t cols) {
    constexpr uint32_t TH = 16, TW = 32, FH = 16, FW = 16;
    std::vector<bfloat16> out(rows * cols);
    uint32_t idx = 0;
    for (uint32_t ti = 0; ti < rows / TH; ++ti) {
        for (uint32_t tj = 0; tj < cols / TW; ++tj) {
            for (uint32_t face = 0; face < 2; ++face) {
                const uint32_t c0 = tj * TW + face * FW;
                for (uint32_t r = 0; r < FH; ++r) {
                    for (uint32_t c = 0; c < FW; ++c) {
                        out[(ti * TH + r) * cols + (c0 + c)] = til[idx++];
                    }
                }
            }
        }
    }
    return out;
}

static std::vector<bfloat16> make_prev_max_matrix(
    uint32_t q_chunk_size,
    float min_val,
    float max_val,
    std::vector<bfloat16>& prev_max_first_col_out,
    uint32_t tile_height = tt::constants::TILE_HEIGHT) {
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(min_val, max_val);

    const uint32_t rows = q_chunk_size * tile_height;
    const uint32_t cols = tt::constants::TILE_WIDTH;
    std::vector<bfloat16> mat(rows * cols);
    prev_max_first_col_out.resize(rows);

    for (uint32_t r = 0; r < rows; ++r) {
        float v = dist(rng);
        prev_max_first_col_out[r] = static_cast<bfloat16>(v);
        for (uint32_t c = 0; c < cols; ++c) {
            mat[(r * cols) + c] = static_cast<bfloat16>(v);
        }
    }
    return mat;
}

static std::vector<bfloat16> golden_reduce_c(
    const std::vector<bfloat16>& qk_im_rm,
    const std::vector<bfloat16>& prev_max_first_col,
    uint32_t q_chunk_size,
    uint32_t k_chunk_size,
    bool do_eltwise_max,
    uint32_t num_faces = 4) {
    // Genuine 16x32 tiny tile (num_faces=2) has tile height 16; full 32x32 (num_faces=4) has height 32.
    // Every row is a real q-row now (no phantom second-face-row to skip).
    const uint32_t tile_height = (num_faces == 2) ? tt::constants::FACE_HEIGHT : tt::constants::TILE_HEIGHT;
    const uint32_t rows = q_chunk_size * tile_height;
    const uint32_t cols = k_chunk_size * tt::constants::TILE_WIDTH;
    const uint32_t stats_cols = tt::constants::TILE_WIDTH;

    bool prev_max_larger = false;

    // Produce tiles: rows tiles. We'll set first column to the row max (replicated down the column).
    std::vector<bfloat16> out(rows * stats_cols, static_cast<bfloat16>(0.0f));

    for (uint32_t r = 0; r < rows; ++r) {
        auto row_begin = qk_im_rm.begin() + r * cols;
        auto row_end = row_begin + cols;
        float row_max = -std::numeric_limits<float>::infinity();
        if (row_begin != row_end) {
            row_max = static_cast<float>(*std::max_element(row_begin, row_end, [](bfloat16 a, bfloat16 b) {
                return static_cast<float>(a) < static_cast<float>(b);
            }));
        }
        if (do_eltwise_max) {
            float working_row_max = row_max;
            row_max = std::max(row_max, static_cast<float>(prev_max_first_col[r]));
            if (row_max > working_row_max) {
                prev_max_larger = true;
            }
        }
        out[(r * stats_cols) + 0] = static_cast<bfloat16>(row_max);
    }

    // Expand to tiles (each row has a 32x32 tile). Fill only first column; others left as zero and not checked.
    std::vector<bfloat16> tilized(rows * stats_cols * tt::constants::TILE_WIDTH, static_cast<bfloat16>(0.0f));
    // However, for comparison we will only use untilized first column from device output; no need to mirror tile layout
    // here.
    log_info(LogTest, "prev_max_larger: {}", prev_max_larger);
    return out;  // row-major (rows x 32), first column holds value
}

static float compare_first_col_mse(
    const std::vector<bfloat16>& result_rm, const std::vector<bfloat16>& golden_first_col_rm, uint32_t num_faces = 4) {
    // result_rm is rows x 32 row-major (from untilize); compare only column 0 vs golden_first_col_rm.
    // With genuine 16x32 tiny tiles every row carries valid output, so no rows are excluded.
    (void)num_faces;
    const uint32_t rows = golden_first_col_rm.size() / tt::constants::TILE_WIDTH;
    float mse = 0.0f;
    uint32_t counted = 0;
    for (uint32_t r = 0; r < rows; ++r) {
        float a = static_cast<float>(result_rm[(r * tt::constants::TILE_WIDTH) + 0]);
        float b = static_cast<float>(golden_first_col_rm[(r * tt::constants::TILE_WIDTH) + 0]);
        float d = a - b;
        mse += d * d;
        ++counted;
    }
    mse /= (counted > 0 ? counted : 1);
    return mse;
}

static bool test_sdpa_reduce_c(
    const std::shared_ptr<tt_metal::distributed::MeshDevice>& mesh_device,
    uint32_t q_chunk_size,
    uint32_t k_chunk_size,
    bool fp32_dest_acc_en,
    bool do_eltwise_max,
    const std::string& kernel_path,
    const std::string& kernel_name,
    uint32_t num_faces = 4) {
    bool pass = true;

    log_info(
        LogTest,
        "Running {} test with q_chunk_size: {}, k_chunk_size: {}, fp32_dest_acc_en: {}, "
        "do_eltwise_max: {}, num_faces: {}",
        kernel_name,
        q_chunk_size,
        k_chunk_size,
        fp32_dest_acc_en,
        do_eltwise_max,
        num_faces);

    // Get device and command queue from mesh
    tt_metal::IDevice* device = mesh_device->get_devices().at(0);
    tt_metal::distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue(0);

    tt_metal::Program program = tt_metal::CreateProgram();

    CoreCoord core = {0, 0};

    auto cb_df = tt::DataFormat::Float16_b;
    // Genuine 16x32 tile (num_faces=2) => tile height 16; full 32x32 (num_faces=4) => height 32.
    const uint32_t tile_height = (num_faces == 2) ? tt::constants::FACE_HEIGHT : tt::constants::TILE_HEIGHT;
    const bool tiny = (tile_height == tt::constants::FACE_HEIGHT);
    auto cb_tile_size = tile_height * tt::constants::TILE_WIDTH * sizeof(bfloat16);

    uint32_t qk_im_num_tiles = q_chunk_size * k_chunk_size;
    uint32_t stats_num_tiles = q_chunk_size;

    auto qk_im_buffer_config = tt::tt_metal::ShardedBufferConfig{
        .device = device,
        .size = qk_im_num_tiles * cb_tile_size,
        .page_size = cb_tile_size,
        .buffer_type = tt::tt_metal::BufferType::L1,
        .buffer_layout = tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = tt::tt_metal::ShardSpecBuffer(
            CoreRangeSet(std::set<CoreRange>({CoreRange(core, core)})),
            {q_chunk_size * tile_height, k_chunk_size * tt::constants::TILE_WIDTH},
            tt::tt_metal::ShardOrientation::ROW_MAJOR,
            {tile_height, tt::constants::TILE_WIDTH},
            {q_chunk_size, k_chunk_size})};

    auto stats_buffer_config = tt::tt_metal::ShardedBufferConfig{
        .device = device,
        .size = stats_num_tiles * cb_tile_size,
        .page_size = cb_tile_size,
        .buffer_type = tt::tt_metal::BufferType::L1,
        .buffer_layout = tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = tt::tt_metal::ShardSpecBuffer(
            CoreRangeSet(std::set<CoreRange>({CoreRange(core, core)})),
            {stats_num_tiles * tile_height, tt::constants::TILE_WIDTH},
            tt::tt_metal::ShardOrientation::ROW_MAJOR,
            {tile_height, tt::constants::TILE_WIDTH},
            {stats_num_tiles, 1})};

    auto one_tile_buffer_config = tt::tt_metal::ShardedBufferConfig{
        .device = device,
        .size = cb_tile_size,
        .page_size = cb_tile_size,
        .buffer_type = tt::tt_metal::BufferType::L1,
        .buffer_layout = tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = tt::tt_metal::ShardSpecBuffer(
            CoreRangeSet(std::set<CoreRange>({CoreRange(core, core)})),
            {tile_height, tt::constants::TILE_WIDTH},
            tt::tt_metal::ShardOrientation::ROW_MAJOR,
            {tile_height, tt::constants::TILE_WIDTH},
            {1, 1})};

    // Create sharded buffers for CB inputs/outputs
    auto qk_im_buffer = CreateBuffer(qk_im_buffer_config);
    auto prev_max_buffer = CreateBuffer(stats_buffer_config);
    auto out_max_buffer = CreateBuffer(stats_buffer_config);
    auto identity_scale_buffer = CreateBuffer(one_tile_buffer_config);

    // Create CBs and point them to sharded buffers
    auto cb_qk_im_id = tt::CBIndex::c_0;
    auto cb_qk_im_config = tt::tt_metal::CircularBufferConfig(qk_im_num_tiles * cb_tile_size, {{cb_qk_im_id, cb_df}})
                               .set_page_size(cb_qk_im_id, cb_tile_size)
                               .set_globally_allocated_address(*qk_im_buffer);
    tt_metal::CreateCircularBuffer(program, core, cb_qk_im_config);

    auto cb_prev_max_id = tt::CBIndex::c_1;
    auto cb_prev_max_config =
        tt::tt_metal::CircularBufferConfig(stats_num_tiles * cb_tile_size, {{cb_prev_max_id, cb_df}})
            .set_page_size(cb_prev_max_id, cb_tile_size)
            .set_globally_allocated_address(*prev_max_buffer);
    tt_metal::CreateCircularBuffer(program, core, cb_prev_max_config);

    auto cb_out_max_id = tt::CBIndex::c_2;
    auto cb_out_max_config =
        tt::tt_metal::CircularBufferConfig(stats_num_tiles * cb_tile_size, {{cb_out_max_id, cb_df}})
            .set_page_size(cb_out_max_id, cb_tile_size)
            .set_globally_allocated_address(*out_max_buffer);
    tt_metal::CreateCircularBuffer(program, core, cb_out_max_config);

    auto cb_identity_scale_id = tt::CBIndex::c_3;
    auto cb_identity_scale_config =
        tt::tt_metal::CircularBufferConfig(1 * cb_tile_size, {{cb_identity_scale_id, cb_df}})
            .set_page_size(cb_identity_scale_id, cb_tile_size)
            .set_globally_allocated_address(*identity_scale_buffer);
    tt_metal::CreateCircularBuffer(program, core, cb_identity_scale_config);

    std::vector<uint32_t> compute_kernel_args = {
        cb_qk_im_id,
        cb_prev_max_id,
        cb_out_max_id,
        cb_identity_scale_id,
        q_chunk_size,
        k_chunk_size,
        static_cast<uint32_t>(do_eltwise_max ? 1 : 0),
        num_faces};
    std::map<std::string, std::string> compute_defines;
    compute_defines["EXP_APPROX_MODE"] = "0";
    compute_defines["REDUCE_GRANULARITY"] = "1";

    tt_metal::CreateKernel(
        program,
        kernel_path,
        core,
        tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi2,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = false,
            .compile_args = compute_kernel_args,
            .defines = compute_defines});

    // Prepare inputs. For a genuine 16x32 tiny tile the operand rows are q_chunk_size*16 and the tile
    // layout is F0(16x16)|F1(16x16) (tilize_16x32); the full 32x32 path keeps tilize_nfaces.
    SHAPE qk_im_shape = {1, 1, q_chunk_size * tile_height, k_chunk_size * 32};
    tt::deprecated::Tensor<bfloat16> qk_im_tensor = tt::deprecated::initialize_tensor<bfloat16>(
        qk_im_shape, tt::deprecated::Initialize::RANDOM, -50, 50, 0 /* seed */);

    vector<uint32_t> qk_im;
    auto qk_im_tilized = tiny ? tilize_16x32(qk_im_tensor.get_values(), q_chunk_size * tile_height, k_chunk_size * 32)
                              : tilize_nfaces(qk_im_tensor.get_values(), q_chunk_size * 32, k_chunk_size * 32);
    qk_im = pack_bfloat16_vec_into_uint32_vec(qk_im_tilized);
    tt_metal::detail::WriteToBuffer(qk_im_buffer, qk_im);

    std::vector<bfloat16> prev_max_first_col;
    auto prev_max_rm = make_prev_max_matrix(q_chunk_size, 25.0f, 65.0f, prev_max_first_col, tile_height);
    auto prev_max_tilized = tiny ? tilize_16x32(prev_max_rm, q_chunk_size * tile_height, 32)
                                 : tilize_nfaces(prev_max_rm, q_chunk_size * 32, 32);
    auto prev_max_uint_vec = pack_bfloat16_vec_into_uint32_vec(prev_max_tilized);
    tt_metal::detail::WriteToBuffer(prev_max_buffer, prev_max_uint_vec);

    auto identity_scale_tile = make_identity_scale_tile(tile_height);
    auto identity_scale_uint_vec = pack_bfloat16_vec_into_uint32_vec(identity_scale_tile);
    tt_metal::detail::WriteToBuffer(identity_scale_buffer, identity_scale_uint_vec);

    // Execute program using MeshWorkload
    tt_metal::distributed::MeshWorkload workload;
    tt_metal::distributed::MeshCoordinate zero_coord =
        tt_metal::distributed::MeshCoordinate::zero_coordinate(mesh_device->shape().dims());
    tt_metal::distributed::MeshCoordinateRange device_range =
        tt_metal::distributed::MeshCoordinateRange(zero_coord, zero_coord);
    workload.add_program(device_range, std::move(program));
    tt_metal::distributed::EnqueueMeshWorkload(cq, workload, false);
    tt_metal::distributed::Finish(cq);

    // Read outputs
    std::vector<uint32_t> out_max_vec;
    tt_metal::detail::ReadFromBuffer(out_max_buffer, out_max_vec);
    auto out_max_bfp16 = unpack_uint32_vec_into_bfloat16_vec(out_max_vec);
    auto out_max_rm = tiny ? untilize_16x32(out_max_bfp16, q_chunk_size * tile_height, 32)
                           : untilize_nfaces(out_max_bfp16, q_chunk_size * 32, 32);

    // Golden
    auto golden_first_col_rm = golden_reduce_c(
        qk_im_tensor.get_values(), prev_max_first_col, q_chunk_size, k_chunk_size, do_eltwise_max, num_faces);

    float mse = compare_first_col_mse(out_max_rm, golden_first_col_rm, num_faces);
    const float max_mse = 0.0f;  // expect exact max match in first column
    log_info(LogTest, "{} first-col mse: {} (do_eltwise: {})", kernel_name, mse, do_eltwise_max);
    if (mse > max_mse) {
        log_error(LogTest, "{} first-col mse: {} > {} (do_eltwise: {})", kernel_name, mse, max_mse, do_eltwise_max);
        pass = false;
    }

    return pass;
}

// NIGHTLY_ prefix ensures this test only runs in nightly CI pipelines
TEST_F(UnitMeshCQSingleCardSharedFixture, NIGHTLY_SdpaReduceC) {
    bool pass = true;

    /**
     * Parameters to sweep over for correctness.
     */
    // std::vector<uint32_t> q_chunk_sizes = {1};
    // std::vector<uint32_t> k_chunk_sizes = {1};
    // std::vector<bool> fp32_dest_acc_ens = {true};
    // std::vector<bool> do_eltwise = {true};
    std::vector<uint32_t> q_chunk_sizes = {1, 2, 4, 8};
    std::vector<uint32_t> k_chunk_sizes = {1, 2, 4, 8, 16};
    std::vector<bool> fp32_dest_acc_ens = {false, true};
    std::vector<bool> do_eltwise = {false, true};

    /**
     * These parameters are the same as the SDPA sprint-2 perfomance test parameters.
     * Uncomment to measure perf of the test we care most about.
     */
    // std::vector<uint32_t> q_chunk_sizes = {8};
    // std::vector<uint32_t> k_chunk_sizes = {16};
    // std::vector<uint32_t> head_dims = {128};
    // std::vector<bool> fp32_dest_acc_ens = {false};
    // std::vector<bool> do_eltwise = {false, true};

    // Test both implementations
    std::vector<std::pair<std::string, std::string>> kernel_variants = {
        {"tests/tt_metal/tt_metal/test_kernels/misc/sdpa/reduce_c/compute.cpp", "reduce_c"},
        {"tests/tt_metal/tt_metal/test_kernels/misc/sdpa/reduce_block_max_row/compute.cpp", "reduce_block_max_row"}};

    for (const auto& [kernel_path, kernel_name] : kernel_variants) {
        log_info(LogTest, "Testing kernel variant: {}", kernel_name);
        for (uint32_t q_chunk_size : q_chunk_sizes) {
            for (uint32_t k_chunk_size : k_chunk_sizes) {
                for (bool fp32_dest_acc_en : fp32_dest_acc_ens) {
                    for (bool do_elt : do_eltwise) {
                        bool this_passed = test_sdpa_reduce_c(
                            devices_[0],
                            q_chunk_size,
                            k_chunk_size,
                            fp32_dest_acc_en,
                            do_elt,
                            kernel_path,
                            kernel_name);
                        if (!this_passed) {
                            log_error(
                                LogTest,
                                "Test Failed for kernel: {}, q_chunk_size: {}, k_chunk_size: {}"
                                "fp32_dest_acc_en: "
                                "{}, do_eltwise: {}",
                                kernel_name,
                                q_chunk_size,
                                k_chunk_size,
                                fp32_dest_acc_en,
                                do_elt);
                        }
                        pass &= this_passed;
                    }
                }
            }
        }
    }

    // 16x32 tiny-tile (num_faces=2, one face-row) case for the block-based reduce.
    // Threads num_faces=2 through reduce_block_max_row_init / reduce_block_max_row on both the
    // UNPACK and MATH sides. Only the first face-row (rows 0-15 of each 32-row tile) is reduced;
    // the golden/comparison exclude the second face-row accordingly.
    {
        const std::string tiny_kernel_path =
            "tests/tt_metal/tt_metal/test_kernels/misc/sdpa/reduce_block_max_row/compute.cpp";
        const std::string tiny_kernel_name = "reduce_block_max_row_tiny_16x32";
        constexpr uint32_t tiny_num_faces = 2;
        for (uint32_t q_chunk_size : q_chunk_sizes) {
            for (uint32_t k_chunk_size : k_chunk_sizes) {
                for (bool fp32_dest_acc_en : fp32_dest_acc_ens) {
                    // 16x32 tiny-tile reduce_block_max_row is only supported in non-fp32 dest mode
                    // (the LLK static_asserts fp32 + num_faces==2). fp32 16x32 is a future item.
                    if (fp32_dest_acc_en) {
                        continue;
                    }
                    for (bool do_elt : do_eltwise) {
                        bool this_passed = test_sdpa_reduce_c(
                            devices_[0],
                            q_chunk_size,
                            k_chunk_size,
                            fp32_dest_acc_en,
                            do_elt,
                            tiny_kernel_path,
                            tiny_kernel_name,
                            tiny_num_faces);
                        if (!this_passed) {
                            log_error(
                                LogTest,
                                "Test Failed for kernel: {}, q_chunk_size: {}, k_chunk_size: {}"
                                "fp32_dest_acc_en: {}, do_eltwise: {}, num_faces: {}",
                                tiny_kernel_name,
                                q_chunk_size,
                                k_chunk_size,
                                fp32_dest_acc_en,
                                do_elt,
                                tiny_num_faces);
                        }
                        pass &= this_passed;
                    }
                }
            }
        }
    }

    // Coverage for the RUNTIME reduce_block_max_row path (used by streaming SDPA via reduce_c_row_group),
    // which the compile-time kernel variants above do not exercise. num_faces=2 uses a GENUINE 16x32
    // tile (tilize_16x32) so the tiny-tile unpack tile-descriptor / multi-tile stride is actually tested
    // — this is what caught the reduce dropping the second k-face (F1) for real 16x32 tiles.
    {
        const std::string rt_kernel_path =
            "tests/tt_metal/tt_metal/test_kernels/misc/sdpa/reduce_block_max_row_runtime/compute.cpp";
        for (uint32_t nf : {4u, 2u}) {
            const std::string rt_kernel_name =
                (nf == 2) ? "reduce_block_max_row_runtime_tiny_16x32" : "reduce_block_max_row_runtime_32x32";
            for (uint32_t q_chunk_size : {1u, 2u}) {
                for (uint32_t k_chunk_size : {1u, 2u, 4u}) {
                    bool this_passed = test_sdpa_reduce_c(
                        devices_[0],
                        q_chunk_size,
                        k_chunk_size,
                        /*fp32_dest_acc_en=*/false,
                        /*do_eltwise_max=*/false,
                        rt_kernel_path,
                        rt_kernel_name,
                        nf);
                    if (!this_passed) {
                        log_error(
                            LogTest,
                            "REPRO Failed: kernel {}, q {}, k {}, num_faces {}",
                            rt_kernel_name,
                            q_chunk_size,
                            k_chunk_size,
                            nf);
                    }
                    pass &= this_passed;
                }
            }
        }
    }

    ASSERT_TRUE(pass);
}
