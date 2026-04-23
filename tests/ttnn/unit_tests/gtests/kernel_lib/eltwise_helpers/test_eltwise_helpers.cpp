// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cmath>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-logger/tt-logger.hpp>

#include "tests/tt_metal/tt_metal/common/command_queue_fixture.hpp"

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

namespace {

// =============================================================================
// Constants
// =============================================================================

static constexpr uint32_t kTileElems = 32 * 32;             // 1024 BF16 values/tile
static constexpr uint32_t kTileBytesBF16 = kTileElems * 2;  // 2048 bytes/tile
static constexpr uint32_t kTileU32s = kTileBytesBF16 / 4;   // 512 uint32_t/tile

// =============================================================================
// Test infrastructure
// =============================================================================

struct EltwiseTestCfg {
    std::string compute_kernel;                  // relative path from tt-metal root
    std::map<std::string, std::string> defines;  // compile-time defines for op selection
    uint32_t num_input_cbs;                      // 1 = unary, 2 = binary
    uint32_t rows;                               // EltwiseTileShape rows
    uint32_t cols;                               // EltwiseTileShape cols (= num_tiles for flat)
    uint32_t b_tile_count;    // B tiles for binary (Wt for ROW, Ht for COL, 1 for SCALAR, rows*cols for NONE)
    uint32_t num_output_cbs;  // 1 for normal, 2 for persistent-input test
    tt::DataFormat data_format{tt::DataFormat::Float16_b};
};

// Runs a kernel test: creates program, runs on device, returns output data as float.
// inputs[0] = A tiles, inputs[1] = B tiles (if binary)
static std::vector<float> run_eltwise_test(
    std::shared_ptr<MeshDevice>& mesh_device,
    const EltwiseTestCfg& cfg,
    std::vector<std::vector<uint32_t>> inputs,   // packed uint32 input data per CB
    uint32_t out_tile_count,                     // how many output tiles to read back
    std::vector<uint32_t>* out_extra = nullptr)  // optional second output buffer (persistent test)
{
    const uint32_t n_a_tiles = cfg.rows * cfg.cols;
    const uint32_t tile_bytes = kTileBytesBF16;  // BF16 only for now
    const uint32_t page_bytes = tile_bytes;

    Program program = CreateProgram();
    CoreCoord core = {0, 0};

    // ── Input DRAM buffers ──────────────────────────────────────────────────
    std::vector<std::shared_ptr<MeshBuffer>> in_bufs;
    for (uint32_t c = 0; c < cfg.num_input_cbs; ++c) {
        const uint32_t n_tiles = (c == 0) ? n_a_tiles : cfg.b_tile_count;
        DeviceLocalBufferConfig local_cfg{.page_size = page_bytes, .buffer_type = BufferType::DRAM};
        ReplicatedBufferConfig rep_cfg{.size = n_tiles * tile_bytes};
        in_bufs.push_back(MeshBuffer::create(rep_cfg, local_cfg, mesh_device.get()));
    }

    // ── Output DRAM buffer(s) ───────────────────────────────────────────────
    DeviceLocalBufferConfig out_local{.page_size = page_bytes, .buffer_type = BufferType::DRAM};
    ReplicatedBufferConfig out_rep{.size = out_tile_count * tile_bytes};
    auto out_buf = MeshBuffer::create(out_rep, out_local, mesh_device.get());

    std::shared_ptr<MeshBuffer> out_buf2;
    if (cfg.num_output_cbs == 2) {
        ReplicatedBufferConfig out_rep2{.size = out_tile_count * tile_bytes};
        out_buf2 = MeshBuffer::create(out_rep2, out_local, mesh_device.get());
    }

    // ── Circular Buffers ───────────────────────────────────────────────────
    // Input CBs
    for (uint32_t c = 0; c < cfg.num_input_cbs; ++c) {
        const uint32_t n_tiles = (c == 0) ? n_a_tiles : cfg.b_tile_count;
        CircularBufferConfig cb_cfg(n_tiles * tile_bytes, {{c, cfg.data_format}}).set_page_size(c, tile_bytes);
        CreateCircularBuffer(program, core, cb_cfg);
    }

    // Output CB(s) at c_16 (and c_17 for persistent test)
    {
        CircularBufferConfig cb_out(out_tile_count * tile_bytes, {{CBIndex::c_16, cfg.data_format}})
            .set_page_size(CBIndex::c_16, tile_bytes);
        CreateCircularBuffer(program, core, cb_out);
    }
    if (cfg.num_output_cbs == 2) {
        CircularBufferConfig cb_out2(out_tile_count * tile_bytes, {{CBIndex::c_17, cfg.data_format}})
            .set_page_size(CBIndex::c_17, tile_bytes);
        CreateCircularBuffer(program, core, cb_out2);
    }

    // ── Dataflow: Reader ────────────────────────────────────────────────────
    const char* reader_path = (cfg.num_input_cbs == 1)
                                  ? "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_8bank.cpp"
                                  : "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_dual_8bank.cpp";

    std::vector<uint32_t> reader_ct_args;
    for (uint32_t c = 0; c < cfg.num_input_cbs; ++c) {
        TensorAccessorArgs(*in_bufs[c]).append_to(reader_ct_args);
    }
    auto reader_kernel = CreateKernel(
        program,
        reader_path,
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = reader_ct_args});

    // ── Dataflow: Writer ────────────────────────────────────────────────────
    const uint32_t out_cb_idx = CBIndex::c_16;
    std::vector<uint32_t> writer_ct_args{out_cb_idx};
    TensorAccessorArgs(*out_buf).append_to(writer_ct_args);
    auto writer_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_8bank.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = writer_ct_args});

    // ── Compute Kernel ──────────────────────────────────────────────────────
    // Broadcast and multi-exec kernels pass rows+cols; others pass n_tiles
    const bool uses_shape_args = (cfg.rows > 1 || cfg.cols != n_a_tiles);
    auto compute_kernel = CreateKernel(program, cfg.compute_kernel, core, ComputeConfig{.defines = cfg.defines});

    // ── Set Runtime Args ────────────────────────────────────────────────────
    if (cfg.num_input_cbs == 1) {
        // reader_unary_8bank: [0]=addr, [1]=0, [2]=0, [3]=num_tiles
        SetRuntimeArgs(program, reader_kernel, core, {in_bufs[0]->address(), 0u, 0u, n_a_tiles});
    } else {
        // reader_dual_8bank: [0]=src0_addr, [1]=0, [2]=src0_tiles, [3]=src1_addr, [4]=0, [5]=src1_tiles, [6]=0
        SetRuntimeArgs(
            program,
            reader_kernel,
            core,
            {in_bufs[0]->address(), 0u, n_a_tiles, in_bufs[1]->address(), 0u, cfg.b_tile_count, 0u});
    }
    // writer_unary_8bank: [0]=dst_addr, [1]=0, [2]=num_tiles
    SetRuntimeArgs(program, writer_kernel, core, {out_buf->address(), 0u, out_tile_count});

    if (uses_shape_args) {
        SetRuntimeArgs(program, compute_kernel, core, {cfg.rows, cfg.cols});
    } else if (cfg.compute_kernel.find("multi_exec") != std::string::npos) {
        // multi_exec: [0]=num_blocks, [1]=tiles_per_block
        SetRuntimeArgs(program, compute_kernel, core, {3u, n_a_tiles / 3u});
    } else if (cfg.compute_kernel.find("persistent") != std::string::npos) {
        SetRuntimeArgs(program, compute_kernel, core, {n_a_tiles});
    } else {
        SetRuntimeArgs(program, compute_kernel, core, {n_a_tiles});
    }

    // ── Write inputs ────────────────────────────────────────────────────────
    MeshWorkload workload;
    workload.add_program(MeshCoordinateRange(mesh_device->shape()), std::move(program));

    auto& cq = mesh_device->mesh_command_queue();
    for (uint32_t c = 0; c < cfg.num_input_cbs; ++c) {
        WriteShard(cq, in_bufs[c], inputs[c], MeshCoordinate(0, 0));
    }

    // ── Run ─────────────────────────────────────────────────────────────────
    EnqueueMeshWorkload(cq, workload, false);
    Finish(cq);

    // ── Read output ──────────────────────────────────────────────────────────
    std::vector<uint32_t> result;
    ReadShard(cq, result, out_buf, MeshCoordinate(0, 0));

    if (out_extra && out_buf2) {
        ReadShard(cq, *out_extra, out_buf2, MeshCoordinate(0, 0));
    }

    // Unpack BF16 uint32 to float
    auto bf_vals = unpack_uint32_vec_into_bfloat16_vec(result);
    std::vector<float> out_float;
    out_float.reserve(bf_vals.size());
    for (auto& v : bf_vals) {
        out_float.push_back(v.to_float());
    }
    return out_float;
}

// Validate actual vs expected with PCC threshold
static void validate_pcc(
    const std::vector<float>& actual, const std::vector<float>& expected, float threshold = 0.9999f) {
    ASSERT_EQ(actual.size(), expected.size());
    // Compute PCC
    const size_t n = actual.size();
    double sum_a = 0, sum_e = 0, sum_aa = 0, sum_ee = 0, sum_ae = 0;
    for (size_t i = 0; i < n; ++i) {
        sum_a += actual[i];
        sum_e += expected[i];
        sum_aa += actual[i] * actual[i];
        sum_ee += expected[i] * expected[i];
        sum_ae += actual[i] * expected[i];
    }
    double mean_a = sum_a / n, mean_e = sum_e / n;
    double cov = sum_ae / n - mean_a * mean_e;
    double var_a = sum_aa / n - mean_a * mean_a;
    double var_e = sum_ee / n - mean_e * mean_e;
    double denom = std::sqrt(var_a * var_e);
    double pcc_val = (denom < 1e-12) ? 1.0 : cov / denom;
    EXPECT_GE(pcc_val, threshold) << "PCC " << pcc_val << " below threshold " << threshold;
}

// Generate random BF16 data packed as uint32, num_tiles * kTileElems values
static std::vector<uint32_t> make_bf16_data(
    uint32_t num_tiles, float scale = 1.0f, float offset = 0.0f, int seed = 42) {
    return create_random_vector_of_bfloat16(
        num_tiles * kTileBytesBF16, /*rand_max_float=*/static_cast<int>(scale * 100), seed, offset);
}

// ─────────────────────────────────────────────────────────────────────────────
// Golden computations (in float, BF16-rounded via bfloat16 type)
// ─────────────────────────────────────────────────────────────────────────────

static std::vector<float> golden_sfpu(const std::vector<uint32_t>& src_packed, std::function<float(float)> op) {
    auto bfvec = unpack_uint32_vec_into_bfloat16_vec(src_packed);
    std::vector<float> out;
    out.reserve(bfvec.size());
    for (auto& v : bfvec) {
        float r = op(v.to_float());
        out.push_back(bfloat16(r).to_float());  // round to BF16
    }
    return out;
}

static std::vector<float> golden_fpu(
    const std::vector<uint32_t>& a_packed,
    const std::vector<uint32_t>& b_packed,
    std::function<float(float, float)> op) {
    auto av = unpack_uint32_vec_into_bfloat16_vec(a_packed);
    auto bv = unpack_uint32_vec_into_bfloat16_vec(b_packed);
    EXPECT_EQ(av.size(), bv.size());
    std::vector<float> out;
    out.reserve(av.size());
    for (size_t i = 0; i < av.size(); ++i) {
        float r = op(av[i].to_float(), bv[i].to_float());
        out.push_back(bfloat16(r).to_float());
    }
    return out;
}

static std::vector<float> golden_broadcast(
    const std::vector<uint32_t>& a_packed,
    const std::vector<uint32_t>& b_packed,
    std::function<float(float, float)> op,
    uint32_t rows,
    uint32_t cols,
    uint32_t b_rows,
    uint32_t b_cols)  // b_tile_count = b_rows * b_cols
{
    auto av = unpack_uint32_vec_into_bfloat16_vec(a_packed);
    auto bv = unpack_uint32_vec_into_bfloat16_vec(b_packed);
    // av is rows*cols tiles, bv is b_tile_count tiles
    // tile-level broadcast: b[tile_b] broadcasts to a[tile_a]
    // Within each tile, all 1024 elements have the same broadcast
    const uint32_t tile_elems = kTileElems;
    std::vector<float> out;
    out.reserve(rows * cols * tile_elems);
    for (uint32_t ht = 0; ht < rows; ++ht) {
        for (uint32_t wt = 0; wt < cols; ++wt) {
            const uint32_t a_tile = ht * cols + wt;
            // Broadcast tile index (mirrors FpuBinaryOp b_tile_idx logic)
            uint32_t b_tile;
            if (b_rows == 1 && b_cols == 1) {
                b_tile = 0;  // SCALAR
            } else if (b_rows == 1) {
                b_tile = wt;  // ROW: b has 1 row of cols tiles
            } else if (b_cols == 1) {
                b_tile = ht;  // COL: b has rows tiles, 1 col
            } else {
                b_tile = ht * cols + wt;  // NONE
            }
            for (uint32_t e = 0; e < tile_elems; ++e) {
                float a_val = av[a_tile * tile_elems + e].to_float();
                float b_val = bv[b_tile * tile_elems + e].to_float();
                out.push_back(bfloat16(op(a_val, b_val)).to_float());
            }
        }
    }
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Fixtures
// ─────────────────────────────────────────────────────────────────────────────

static const std::string kKernelBase = "tests/ttnn/unit_tests/gtests/kernel_lib/eltwise_helpers/kernels/compute/";

}  // namespace

// =============================================================================
// Test 1: SFPU-only chain — (exp | relu | sqrt) × (BF16 | FP32 | BFP8) × (PerTile | Bulk)
// =============================================================================

class EltwiseSfpuChainTest : public UnitMeshCQSingleCardSharedFixture,
                             public testing::WithParamInterface<std::tuple<std::string, std::string, std::string>> {};

TEST_P(EltwiseSfpuChainTest, Run) {
    auto [op, dtype_str, policy_str] = GetParam();
    auto& mesh_device = this->devices_[0];

    const uint32_t n_tiles = 4;
    std::map<std::string, std::string> defines{{"SFPU_OP_" + op, "1"}};
    if (policy_str == "PerTile") {
        defines["OUTPUT_POLICY_PER_TILE"] = "1";
    }

    // Positive values for sqrt, any for exp/relu
    float scale = (op == "SQRT") ? 2.0f : 3.0f;
    float offset = (op == "SQRT") ? 0.01f : (op == "RELU" ? -1.5f : 0.0f);
    auto a = make_bf16_data(n_tiles, scale, offset);

    std::function<float(float)> golden_fn;
    if (op == "EXP") {
        golden_fn = [](float x) { return std::exp(x); };
    }
    if (op == "RELU") {
        golden_fn = [](float x) { return std::max(0.0f, x); };
    }
    if (op == "SQRT") {
        golden_fn = [](float x) { return std::sqrt(x); };
    }

    auto expected = golden_sfpu(a, golden_fn);

    EltwiseTestCfg cfg{
        .compute_kernel = kKernelBase + "compute_sfpu_chain.cpp",
        .defines = defines,
        .num_input_cbs = 1,
        .rows = 1,
        .cols = n_tiles,
        .b_tile_count = 0,
        .num_output_cbs = 1};

    auto actual = run_eltwise_test(mesh_device, cfg, {a}, n_tiles);
    validate_pcc(actual, expected);
}

INSTANTIATE_TEST_SUITE_P(
    SfpuChain,
    EltwiseSfpuChainTest,
    testing::Combine(
        testing::Values("EXP", "RELU", "SQRT"),
        testing::Values("BF16"),  // TODO: FP32, BFP8 (require pack reconfig)
        testing::Values("Bulk", "PerTile")),
    [](const auto& info) {
        auto [op, dtype, policy] = info.param;
        return op + "_" + dtype + "_" + policy;
    });

// =============================================================================
// Test 2: FPU-only chain — (add | sub | mul) × (BF16)
// =============================================================================

class EltwiseFpuChainTest : public UnitMeshCQSingleCardSharedFixture,
                            public testing::WithParamInterface<std::string> {};

TEST_P(EltwiseFpuChainTest, Run) {
    const std::string op = GetParam();
    auto& mesh_device = this->devices_[0];

    const uint32_t n_tiles = 4;
    auto a = make_bf16_data(n_tiles, 2.0f, 0.0f, 11);
    auto b = make_bf16_data(n_tiles, 2.0f, 0.0f, 99);

    std::function<float(float, float)> golden_fn;
    if (op == "ADD") {
        golden_fn = [](float x, float y) { return x + y; };
    }
    if (op == "SUB") {
        golden_fn = [](float x, float y) { return x - y; };
    }
    if (op == "MUL") {
        golden_fn = [](float x, float y) { return x * y; };
    }

    auto expected = golden_fpu(a, b, golden_fn);

    EltwiseTestCfg cfg{
        .compute_kernel = kKernelBase + "compute_fpu_chain.cpp",
        .defines = {{"FPU_OP_" + op, "1"}},
        .num_input_cbs = 2,
        .rows = 1,
        .cols = n_tiles,
        .b_tile_count = n_tiles,
        .num_output_cbs = 1};

    auto actual = run_eltwise_test(mesh_device, cfg, {a, b}, n_tiles);
    validate_pcc(actual, expected);
}

INSTANTIATE_TEST_SUITE_P(FpuChain, EltwiseFpuChainTest, testing::Values("ADD", "SUB", "MUL"));

// =============================================================================
// Test 3: FPU+SFPU chain — (add|mul) × (relu|sqrt) × BF16
// =============================================================================

class EltwiseFpuSfpuChainTest : public UnitMeshCQSingleCardSharedFixture,
                                public testing::WithParamInterface<std::pair<std::string, std::string>> {};

TEST_P(EltwiseFpuSfpuChainTest, Run) {
    auto [fpu_op, sfpu_op] = GetParam();
    auto& mesh_device = this->devices_[0];

    const uint32_t n_tiles = 4;
    // Ensure values produce non-negative results for sqrt: use positive values
    float offset = (sfpu_op == "SQRT") ? 0.5f : -1.0f;
    float scale = (sfpu_op == "SQRT") ? 1.0f : 2.0f;
    auto a = make_bf16_data(n_tiles, scale, offset, 7);
    auto b = make_bf16_data(n_tiles, scale, offset, 13);

    // FPU golden
    std::function<float(float, float)> fpu_fn;
    if (fpu_op == "ADD") {
        fpu_fn = [](float x, float y) { return x + y; };
    }
    if (fpu_op == "MUL") {
        fpu_fn = [](float x, float y) { return x * y; };
    }
    // SFPU golden
    std::function<float(float)> sfpu_fn;
    if (sfpu_op == "RELU") {
        sfpu_fn = [](float x) { return std::max(0.0f, x); };
    }
    if (sfpu_op == "SQRT") {
        sfpu_fn = [](float x) { return std::sqrt(std::max(0.0f, x)); };
    }

    // Compose: sfpu_fn(fpu_fn(a, b))
    auto fpu_result_packed = [&]() {
        auto av = unpack_uint32_vec_into_bfloat16_vec(a);
        auto bv = unpack_uint32_vec_into_bfloat16_vec(b);
        std::vector<bfloat16> mid;
        mid.reserve(av.size());
        for (size_t i = 0; i < av.size(); ++i) {
            mid.push_back(bfloat16(fpu_fn(av[i].to_float(), bv[i].to_float())));
        }
        return pack_bfloat16_vec_into_uint32_vec(mid);
    }();
    auto expected = golden_sfpu(fpu_result_packed, sfpu_fn);

    std::map<std::string, std::string> defines{{"FPU_OP_" + fpu_op, "1"}, {"SFPU_OP_" + sfpu_op, "1"}};

    EltwiseTestCfg cfg{
        .compute_kernel = kKernelBase + "compute_fpu_sfpu_chain.cpp",
        .defines = defines,
        .num_input_cbs = 2,
        .rows = 1,
        .cols = n_tiles,
        .b_tile_count = n_tiles,
        .num_output_cbs = 1};

    auto actual = run_eltwise_test(mesh_device, cfg, {a, b}, n_tiles);
    validate_pcc(actual, expected);
}

INSTANTIATE_TEST_SUITE_P(
    FpuSfpuChain,
    EltwiseFpuSfpuChainTest,
    testing::Values(
        std::pair{"ADD", "RELU"}, std::pair{"ADD", "SQRT"}, std::pair{"MUL", "RELU"}, std::pair{"MUL", "SQRT"}));

// =============================================================================
// Test 4: FPU broadcast — (ROW|COL|SCALAR) × (add|mul) × BF16
// =============================================================================

struct BcastParams {
    std::string bcast;
    std::string op;
};

class EltwiseFpuBroadcastTest : public UnitMeshCQSingleCardSharedFixture,
                                public testing::WithParamInterface<BcastParams> {};

TEST_P(EltwiseFpuBroadcastTest, Run) {
    auto [bcast, op] = GetParam();
    auto& mesh_device = this->devices_[0];

    const uint32_t Ht = 2, Wt = 4;
    const uint32_t n_a = Ht * Wt;

    uint32_t b_rows = 1, b_cols = 1, b_count = 1;
    if (bcast == "ROW") {
        b_rows = 1;
        b_cols = Wt;
        b_count = Wt;
    } else if (bcast == "COL") {
        b_rows = Ht;
        b_cols = 1;
        b_count = Ht;
    } else {
        b_rows = 1;
        b_cols = 1;
        b_count = 1;
    }  // SCALAR

    auto a = make_bf16_data(n_a, 2.0f, 0.0f, 31);
    auto b = make_bf16_data(b_count, 1.5f, 0.5f, 77);

    std::function<float(float, float)> fn;
    if (op == "ADD") {
        fn = [](float x, float y) { return x + y; };
    }
    if (op == "MUL") {
        fn = [](float x, float y) { return x * y; };
    }

    auto expected = golden_broadcast(a, b, fn, Ht, Wt, b_rows, b_cols);

    std::map<std::string, std::string> defines{{"BCAST_" + bcast, "1"}};
    if (op == "MUL") {
        defines["FPU_OP_MUL"] = "1";
    }

    EltwiseTestCfg cfg{
        .compute_kernel = kKernelBase + "compute_fpu_broadcast.cpp",
        .defines = defines,
        .num_input_cbs = 2,
        .rows = Ht,
        .cols = Wt,
        .b_tile_count = b_count,
        .num_output_cbs = 1};

    auto actual = run_eltwise_test(mesh_device, cfg, {a, b}, n_a);
    validate_pcc(actual, expected);
}

INSTANTIATE_TEST_SUITE_P(
    FpuBroadcast,
    EltwiseFpuBroadcastTest,
    testing::Values(
        BcastParams{"ROW", "ADD"},
        BcastParams{"ROW", "MUL"},
        BcastParams{"COL", "ADD"},
        BcastParams{"COL", "MUL"},
        BcastParams{"SCALAR", "ADD"},
        BcastParams{"SCALAR", "MUL"}),
    [](const auto& info) { return info.param.bcast + "_" + info.param.op; });

// =============================================================================
// Test 5: FPU broadcast + SFPU — (ROW|COL|SCALAR) × (relu|exp) × BF16
// =============================================================================

struct BcastSfpuParams {
    std::string bcast;
    std::string sfpu;
};

class EltwiseFpuBroadcastSfpuTest : public UnitMeshCQSingleCardSharedFixture,
                                    public testing::WithParamInterface<BcastSfpuParams> {};

TEST_P(EltwiseFpuBroadcastSfpuTest, Run) {
    auto [bcast, sfpu] = GetParam();
    auto& mesh_device = this->devices_[0];

    const uint32_t Ht = 2, Wt = 4, n_a = Ht * Wt;
    uint32_t b_rows = 1, b_cols = Wt, b_count = Wt;  // default ROW
    if (bcast == "COL") {
        b_rows = Ht;
        b_cols = 1;
        b_count = Ht;
    }
    if (bcast == "SCALAR") {
        b_rows = 1;
        b_cols = 1;
        b_count = 1;
    }

    auto a = make_bf16_data(n_a, 1.5f, 0.1f, 55);
    auto b = make_bf16_data(b_count, 1.0f, 0.1f, 66);

    auto add_fn = [](float x, float y) { return x + y; };
    auto fpu_expected = golden_broadcast(a, b, add_fn, Ht, Wt, b_rows, b_cols);

    // Pack fpu_expected back to uint32 for SFPU pass
    std::vector<bfloat16> fpu_bf;
    fpu_bf.reserve(fpu_expected.size());
    for (float v : fpu_expected) {
        fpu_bf.push_back(bfloat16(v));
    }
    auto fpu_packed = pack_bfloat16_vec_into_uint32_vec(fpu_bf);

    std::function<float(float)> sfpu_fn;
    if (sfpu == "RELU") {
        sfpu_fn = [](float x) { return std::max(0.0f, x); };
    }
    if (sfpu == "EXP") {
        sfpu_fn = [](float x) { return std::exp(std::min(x, 5.0f)); };
    }

    auto expected = golden_sfpu(fpu_packed, sfpu_fn);

    std::map<std::string, std::string> defines{{"BCAST_" + bcast, "1"}};
    if (sfpu == "EXP") {
        defines["SFPU_OP_EXP"] = "1";
    }

    EltwiseTestCfg cfg{
        .compute_kernel = kKernelBase + "compute_fpu_broadcast_sfpu.cpp",
        .defines = defines,
        .num_input_cbs = 2,
        .rows = Ht,
        .cols = Wt,
        .b_tile_count = b_count,
        .num_output_cbs = 1};

    auto actual = run_eltwise_test(mesh_device, cfg, {a, b}, n_a);
    validate_pcc(actual, expected);
}

INSTANTIATE_TEST_SUITE_P(
    FpuBroadcastSfpu,
    EltwiseFpuBroadcastSfpuTest,
    testing::Values(
        BcastSfpuParams{"ROW", "RELU"},
        BcastSfpuParams{"ROW", "EXP"},
        BcastSfpuParams{"COL", "RELU"},
        BcastSfpuParams{"COL", "EXP"},
        BcastSfpuParams{"SCALAR", "RELU"},
        BcastSfpuParams{"SCALAR", "EXP"}),
    [](const auto& info) { return info.param.bcast + "_" + info.param.sfpu; });

// =============================================================================
// Test 6: Multi-exec — init fires once per call, N blocks all correct
// =============================================================================

class EltwiseMultiExecTest : public UnitMeshCQSingleCardSharedFixture,
                             public testing::WithParamInterface<std::string> {};

TEST_P(EltwiseMultiExecTest, Run) {
    const std::string chain_type = GetParam();
    auto& mesh_device = this->devices_[0];

    const uint32_t num_blocks = 3, tiles_per_block = 2;
    const uint32_t n_total = num_blocks * tiles_per_block;

    std::vector<std::vector<uint32_t>> inputs;
    std::vector<float> expected;

    if (chain_type == "SFPU_ONLY") {
        auto a = make_bf16_data(n_total, 3.0f, -1.5f, 1);
        inputs = {a};
        expected = golden_sfpu(a, [](float x) { return std::max(0.0f, x); });
    } else {
        auto a = make_bf16_data(n_total, 2.0f, 0.0f, 2);
        auto b = make_bf16_data(n_total, 2.0f, 0.0f, 3);
        inputs = {a, b};
        expected = golden_fpu(a, b, [](float x, float y) { return x + y; });
        if (chain_type == "FPU_SFPU") {
            // Apply relu after add
            for (float& v : expected) {
                v = std::max(0.0f, v);
            }
        }
    }

    const uint32_t num_input_cbs = (chain_type == "SFPU_ONLY") ? 1 : 2;
    EltwiseTestCfg cfg{
        .compute_kernel = kKernelBase + "compute_multi_exec.cpp",
        .defines = {{"CHAIN_" + chain_type, "1"}},
        .num_input_cbs = num_input_cbs,
        .rows = 1,
        .cols = n_total,
        .b_tile_count = (num_input_cbs == 2) ? n_total : 0u,
        .num_output_cbs = 1};

    auto actual = run_eltwise_test(mesh_device, cfg, inputs, n_total);
    validate_pcc(actual, expected);
}

INSTANTIATE_TEST_SUITE_P(MultiExec, EltwiseMultiExecTest, testing::Values("SFPU_ONLY", "FPU_ONLY", "FPU_SFPU"));

// =============================================================================
// Test 7: Persistent input — B (WaitUpfrontNoPop) reused across two blocks
// =============================================================================

TEST_F(UnitMeshCQSingleCardSharedFixture, PersistentInput) {
    auto& mesh_device = this->devices_[0];

    const uint32_t n_tiles = 4;
    const uint32_t n_out = n_tiles * 2;  // kernel writes 2 blocks to cb_out

    auto a = make_bf16_data(n_tiles * 2, 2.0f, 0.0f, 5);  // 2 blocks of A (A stream)
    auto b = make_bf16_data(n_tiles, 1.0f, 0.5f, 6);      // 1 block of B (reused)

    // Expected: block 0 = A[0..n_tiles-1] + B, block 1 = A[n_tiles..2n-1] + B (same B)
    auto av = unpack_uint32_vec_into_bfloat16_vec(a);
    auto bv = unpack_uint32_vec_into_bfloat16_vec(b);
    std::vector<float> expected;
    expected.reserve(n_out * kTileElems);
    for (uint32_t i = 0; i < n_out * kTileElems; ++i) {
        float a_val = av[i].to_float();
        float b_val = bv[i % (n_tiles * kTileElems)].to_float();
        expected.push_back(bfloat16(a_val + b_val).to_float());
    }

    EltwiseTestCfg cfg{
        .compute_kernel = kKernelBase + "compute_persistent_input.cpp",
        .defines = {},
        .num_input_cbs = 2,
        .rows = 1,
        .cols = n_tiles * 2,      // reader pushes 2*n_tiles A tiles total
        .b_tile_count = n_tiles,  // reader pushes n_tiles B tiles (reused)
        .num_output_cbs = 1};

    // Host passes n_a_tiles = 2*n_tiles as the compute kernel arg.
    // Kernel divides by 2 to get tiles_per_block and calls eltwise_op twice.
    auto actual = run_eltwise_test(mesh_device, cfg, {a, b}, n_out);
    validate_pcc(actual, expected);
}
