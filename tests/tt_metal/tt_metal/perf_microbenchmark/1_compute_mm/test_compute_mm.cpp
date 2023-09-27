// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <chrono>
#include <functional>
#include <random>

#include "common/bfloat16.hpp"
#include "common/bfloat8.hpp"
#include "test_tiles.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/tt_metal/perf_microbenchmark/common/util_device_profiler.hpp"

using namespace tt;
////////////////////////////////////////////////////////////////////////////////
// This benchmark measures the compute performance of matmul.
// When in the slow dispatch mode, it uses LaunchProgram API and measures performance via device profiler.
// In the fast dispatch mode, it uses EnqueueProgram API and measures the execution time.
// Regarding kernels, the compute kernel used “bmm_large_block_zm_fused_bias_activation.cpp” as is and
// the data movement kernels were implemented with reference to kernels of multi_core_reuse_mcast_2d_optimized bmm op in
// tt_dnn. Matmul parameters such as in0_block_w, out_subblock_h, out_subblock_w are set considering to the given input
// shape and L1 size.
//
// Disclaimer
// - This benchmark uses a little trick for both inputs (M x K and K x N) to support as large an input as possible. Only
// the first block of each input (per_core_Mt x in0_block_w, in0_block_w x per_core_Nt) is stored in L1 and used
// repeatedly for the total number of blocks.
// - Currently, TT's matmul implementation may not be able to use all Tensix cores for certain input shapes. In that
// case, only some cores are used with a warning message.
// - To measure performance in the slow dispatch mode, build tt_metal project with the profiler build flag
// (ENABLE_PROFILER=1) first. This benchmark copied device profiler's internal code to get the "t0 to any riscfw end"
// cycles. If device profiler is changed, it also should be updated. Otherwise, it may get inappropriate cycle value.
//
// TODO:
// - For validation, the output is compared with cpu-ref mm code. This benchamrk uses gold_mm function modified version
// of gold_bmm function from test_gold_impls.hpp. As K increases, the error in the results of both versions also
// increases so it is required to find appropriate atol and rtol or alternatives.
//
// Usage example:
//   ./test_compute_mm --m <size in elements> --n <size in elements> --k <size in elements> --slow-dispatch-mode <0 for
//   fast dispatch mode, 1 for slow dispatch mode>
//
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////
//                      Function Forward Declaration
////////////////////////////////////////////////////////////////////////////
uint32_t get_l1_size(tt::ARCH arch);

double get_tt_npu_rpeak_tflops(tt::ARCH arch, CoreCoord grid_size, int tt_npu_clock);

tuple<uint32_t, uint32_t, uint32_t> get_aligned_input_tile_num(uint32_t M, uint32_t N, uint32_t K);

uint32_t get_in0_block_w(
    uint32_t per_core_Mt, uint32_t per_core_Nt, uint32_t Kt, uint32_t single_tile_size, uint32_t l1_size);

CoreCoord get_core_range(
    uint32_t num_blocks_rows, uint32_t num_blocks_cols, uint32_t max_num_rows, uint32_t max_num_cols);

tuple<MathFidelity, bool> get_compute_params(tt::ARCH arch);

tuple<uint32_t, uint32_t> get_out_subblock_params(uint32_t per_core_Mt, uint32_t per_core_Nt);

tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> get_all_buffers_addresses(
    uint32_t per_core_Mt, uint32_t per_core_Nt, uint32_t in0_block_w, uint32_t single_tile_size);

std::vector<float> generate_fp32_random(uint32_t num_elems, int32_t rand_max_val);

template <typename T>
std::vector<T> tilize(std::vector<T> data, int rows, int cols);

template <typename T>
std::vector<T> untilize(std::vector<T> data, int rows, int cols);

template <typename T>
std::vector<T> get_row_slice(std::vector<T> data, int start_row_index, int num_rows, int rows, int cols);

template <typename T>
std::vector<T> get_col_slice(std::vector<T> data, int start_col_index, int num_cols, int rows, int cols);

void prepare_inputs(
    tt_metal::Device* device,
    CoreCoord core_range,
    uint32_t Mt,
    uint32_t Nt,
    uint32_t Kt,
    uint32_t per_core_Mt,
    uint32_t per_core_Nt,
    uint32_t in0_block_w,
    uint32_t single_tile_size,
    uint32_t activations_addr,
    uint32_t weights_addr,
    uint32_t in2_cb_addr,
    std::vector<std::vector<float>>& in0_bfp8_unpack_slice,
    std::vector<std::vector<float>>& in1_bfp8_unpack_slice);

tt_metal::Program create_program(
    tt_metal::Device* device,
    tt::DataFormat cb_data_format,
    MathFidelity math_fidelity,
    uint32_t single_tile_size,
    CoreCoord core_range,
    uint32_t B,
    uint32_t Mt,
    uint32_t Nt,
    uint32_t Kt,
    bool bcast_batch,
    uint32_t in0_block_w,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t per_core_Mt,
    uint32_t per_core_Nt,
    uint32_t in0_cb_addr,
    uint32_t in1_cb_addr,
    uint32_t zero_cb_addr,
    uint32_t out_cb_addr,
    uint32_t in0_addr,
    uint32_t in1_addr,
    uint32_t out_addr);

int get_tt_npu_clock(tt_metal::Device *device);

inline vector<float> gold_mm(
    const vector<uint32_t> shapeA,
    const vector<float>& A,
    const vector<uint32_t>& shapeB,
    const vector<float>& B,
    const uint32_t& num_blocks,
    bool acc16);

bool validation(
    tt_metal::Device* device,
    CoreCoord core_range,
    uint32_t Mt,
    uint32_t Nt,
    uint32_t Kt,
    uint32_t per_core_Mt,
    uint32_t per_core_Nt,
    uint32_t in0_block_w,
    uint32_t out_addr,
    uint32_t single_tile_size,
    bool fp32_dest_acc_en,
    std::vector<std::vector<float>>& in0_bfp8_unpack_slice,
    std::vector<std::vector<float>>& in1_bfp8_unpack_slice);

////////////////////////////////////////////////////////////////////////////
//                      Main
////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
    bool pass = true;
    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Initial Runtime Args Parse
        ////////////////////////////////////////////////////////////////////////////
        std::vector<std::string> input_args(argv, argv + argc);
        uint32_t M;
        uint32_t N;
        uint32_t K;
        uint32_t slow_dispatch_mode;
        try {
            std::tie(M, input_args) = test_args::get_command_option_uint32_and_remaining_args(input_args, "--m", 11264);
            std::tie(N, input_args) = test_args::get_command_option_uint32_and_remaining_args(input_args, "--n", 3072);
            std::tie(K, input_args) = test_args::get_command_option_uint32_and_remaining_args(input_args, "--k", 768);
            std::tie(slow_dispatch_mode, input_args) =
                test_args::get_command_option_uint32_and_remaining_args(input_args, "--slow-dispatch-mode", 1);
        } catch (const std::exception& e) {
            log_fatal(LogTest, "Command line arguments found exception", e.what());
        }

        ////////////////////////////////////////////////////////////////////////////
        //                      Env and Device Setup
        ////////////////////////////////////////////////////////////////////////////
        if (slow_dispatch_mode) {
            setenv("TT_METAL_SLOW_DISPATCH_MODE", "1", true);
            setenv("TT_METAL_DEVICE_PROFILER", "1", true);

#if !defined(PROFILER)
            log_error("In the slow dispatch mode, device profiler is used to get the performance");
            log_error("Build the tt_metal project with the profiler build flag (ENABLE_PROFILER=1)");
            TT_ASSERT(false);
#endif
        }

        int pci_express_slot = 0;
        tt_metal::Device* device = tt_metal::CreateDevice(pci_express_slot);
        const tt::ARCH arch = device->arch();
        ////////////////////////////////////////////////////////////////////////////
        //                      Check Input Args
        ////////////////////////////////////////////////////////////////////////////
        uint32_t l1_size = get_l1_size(arch);
        auto [Mt, Nt, Kt] = get_aligned_input_tile_num(M, N, K);
        log_info(LogTest, "Input M, N, K = {}, {}, {} / {}, {}, {} tile(s)", M, N, K, Mt, Nt, Kt);

        tt::DataFormat data_format = tt::DataFormat::Bfp8_b;
        uint32_t single_tile_size = tt_metal::detail::TileSize(data_format);
        TT_ASSERT(single_tile_size == (256 * 4) + (16 * 4));

        auto grid_size = device->compute_with_storage_grid_size();
        uint32_t num_cores_y = grid_size.y;
        uint32_t num_cores_x = grid_size.x;
        uint32_t per_core_Mt = (Mt - 1) / num_cores_y + 1;
        uint32_t per_core_Nt = (Nt - 1) / num_cores_x + 1;
        uint32_t in0_block_w = get_in0_block_w(per_core_Mt, per_core_Nt, Kt, single_tile_size, l1_size);
        if (in0_block_w == 0) {
            log_error(LogTest, "M, N, K = {}, {}, {} cannot be tested due to insufficient L1 memory.", M, N, K);
            TT_ASSERT(false);
        }

        uint32_t num_blocks_y = (Mt - 1) / per_core_Mt + 1;
        uint32_t num_blocks_x = (Nt - 1) / per_core_Nt + 1;
        uint32_t num_blocks_total = num_blocks_y * num_blocks_x;
        TT_ASSERT(num_blocks_total <= num_cores_x * num_cores_y);
        CoreCoord core_range = get_core_range(num_blocks_y, num_blocks_x, num_cores_y, num_cores_x);
        if (core_range.y != num_cores_y || core_range.x != num_cores_x) {
            log_warning(
                LogTest,
                "This run only use {} cores instead {} cores",
                core_range.y * core_range.x,
                num_cores_y * num_cores_x);
        }
        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        auto [math_fidelity, fp32_dest_acc_en] = get_compute_params(arch);
        auto [out_subblock_h, out_subblock_w] = get_out_subblock_params(per_core_Mt, per_core_Nt);
        auto [in0_cb_addr, in1_cb_addr, zero_cb_addr, out_cb_addr, in0_addr, in1_addr, out_addr] =
            get_all_buffers_addresses(per_core_Mt, per_core_Nt, in0_block_w, single_tile_size);

        auto program = create_program(
            device,
            data_format,
            math_fidelity,
            single_tile_size,
            core_range,
            1,
            Mt,
            Nt,
            Kt,
            false,
            in0_block_w,
            out_subblock_h,
            out_subblock_w,
            per_core_Mt,
            per_core_Nt,
            in0_cb_addr,
            in1_cb_addr,
            zero_cb_addr,
            out_cb_addr,
            in0_addr,
            in1_addr,
            out_addr);

        ////////////////////////////////////////////////////////////////////////////
        //                      Input Setup
        ////////////////////////////////////////////////////////////////////////////
        // for cpu-ref mm
        std::vector<std::vector<float>> in0_bfp8_unpack_slice;
        std::vector<std::vector<float>> in1_bfp8_unpack_slice;
        prepare_inputs(
            device,
            core_range,
            Mt,
            Nt,
            Kt,
            per_core_Mt,
            per_core_Nt,
            in0_block_w,
            single_tile_size,
            in0_addr,
            in1_addr,
            zero_cb_addr,
            in0_bfp8_unpack_slice,
            in1_bfp8_unpack_slice);

        ////////////////////////////////////////////////////////////////////////////
        //                      Kernel Execution and Perf Profiling
        ////////////////////////////////////////////////////////////////////////////
        constexpr int giga_byte = 1000000;
        constexpr long long tera_byte = 1000000000000LL;
        int tt_npu_clock = get_tt_npu_clock(device);
        double rpeak_tflops = get_tt_npu_rpeak_tflops(arch, grid_size, tt_npu_clock);
        double rmax_tflops;
        unsigned long elapsed_us;
        uint64_t num_of_matmul_ops =
            (2 * static_cast<uint64_t>(Kt) * 32 - 1) * (static_cast<uint64_t>(Mt) * static_cast<uint64_t>(Nt) * 1024);
        log_debug(LogTest, "number of matmul ops: {}", num_of_matmul_ops);

        if (slow_dispatch_mode) {
            log_debug(LogTest, "calling LaunchProgram");
            LaunchProgram(device, program);
            log_debug(LogTest, "LaunchProgram done");

            uint64_t t0_to_any_riscfw_end = get_t0_to_any_riscfw_end_cycle(device, program);
            double cycle_time = 1 / static_cast<double>(tt_npu_clock) / giga_byte;
            auto execution_time = t0_to_any_riscfw_end * cycle_time;
            rmax_tflops = static_cast<double>(num_of_matmul_ops) / execution_time / tera_byte;

            log_debug(LogTest, "cycle time {:.8f}s", cycle_time);
            log_debug(LogTest, "t0_to_any_riscfw_end {}", t0_to_any_riscfw_end);
        }
        else {
            log_debug(LogTest, "calling EnqueueProgram");
            std::chrono::duration<double, std::nano> duration;
            auto t_begin = std::chrono::high_resolution_clock::now();
            EnqueueProgram(*::detail::GLOBAL_CQ, program, false);
            Finish(*::detail::GLOBAL_CQ);
            log_debug(LogTest, "EnqueProgram done");
            auto t_end = std::chrono::high_resolution_clock::now();
            duration = t_end - t_begin;
            rmax_tflops = static_cast<double>(num_of_matmul_ops) / duration.count() / 1000;
            log_debug(LogTest, "time duration: {} ns, rmax_tflops {}", duration.count(), rmax_tflops);
        }

        double rmax_per_rpeak = rmax_tflops / rpeak_tflops;
        log_info(LogTest, "Rmax(TFLOPS) {:.3f}, Rpeak {:.3f}, Rmax / Rpeak {:.2f}%", rmax_tflops, rpeak_tflops, rmax_per_rpeak * 100);
        bool performance_result = true;
        if (rmax_per_rpeak < 0.9) {
            performance_result = false;
        }

        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        bool validation_result = validation(
            device,
            core_range,
            Mt,
            Nt,
            Kt,
            per_core_Mt,
            per_core_Nt,
            in0_block_w,
            out_addr,
            single_tile_size,
            fp32_dest_acc_en,
            in0_bfp8_unpack_slice,
            in1_bfp8_unpack_slice);

        // Determine if it passes performance goal
        if (!validation_result) {
            log_error(LogTest, "Matmul validation between TT NPU and cpu-ref failed");
            pass = false;
        }

        if (!performance_result) {
            log_error(
                LogTest,
                "The compute performance does not meet the criteria. "
                "Current: Rmax / Rpeak = {:.2f}%, goal: > 90%",
                rmax_per_rpeak * 100);
            pass = false;
        }

        pass &= tt_metal::CloseDevice(device);

    } catch (const std::exception& e) {
        pass = false;
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        log_fatal(LogTest, "Test Failed");
    }

    TT_ASSERT(pass);

    return 0;
}

////////////////////////////////////////////////////////////////////////////
//                      Function Implementation
////////////////////////////////////////////////////////////////////////////
uint32_t get_l1_size(tt::ARCH arch) {
    constexpr uint32_t GS_L1_SIZE = 1048576;
    constexpr uint32_t WH_L1_SIZE = 1499136;

    uint32_t l1_size = 0;
    if (arch == tt::ARCH::WORMHOLE || arch == tt::ARCH::WORMHOLE_B0) {
        l1_size = WH_L1_SIZE;
    } else if (arch == tt::ARCH::GRAYSKULL) {
        l1_size = GS_L1_SIZE;
    }
    return l1_size;
}

double get_tt_npu_rpeak_tflops(tt::ARCH arch, CoreCoord grid_size, int tt_npu_clock) {
    constexpr double WH_FPU_BFP8_TFLOPS_PER_TENSIX = 2.05;
    constexpr double GS_FPU_BFP8_TFLOPS_PER_TENSIX = 0.58;

    double rpeak_tflops = 0.0f;
    double clock = static_cast<double>(tt_npu_clock) / 1000;
    uint32_t num_compute_core = grid_size.x * grid_size.y;
    if (arch == tt::ARCH::WORMHOLE || arch == tt::ARCH::WORMHOLE_B0) {
        rpeak_tflops =
            WH_FPU_BFP8_TFLOPS_PER_TENSIX * static_cast<double>(num_compute_core) * static_cast<double>(clock);
    } else if (arch == tt::ARCH::GRAYSKULL) {
        rpeak_tflops =
            GS_FPU_BFP8_TFLOPS_PER_TENSIX * static_cast<double>(num_compute_core) * static_cast<double>(clock);
    }

    log_debug(LogTest, "Rpeak {} TFLOPS", rpeak_tflops);
    return rpeak_tflops;
}

tuple<uint32_t, uint32_t, uint32_t> get_aligned_input_tile_num(uint32_t M, uint32_t N, uint32_t K) {
    auto align_to_tile = [](uint32_t value) -> uint32_t {
        return ((value + (constants::TILE_WIDTH - 1)) / constants::TILE_WIDTH) * constants::TILE_WIDTH;
    };

    TT_ASSERT(M != 0 && N != 0 && K != 0, "Matmul input size should not be zero");

    uint32_t M_aligned = align_to_tile(M);
    uint32_t N_aligned = align_to_tile(N);
    uint32_t K_aligned = align_to_tile(K);

    if (M % constants::TILE_WIDTH || N % constants::TILE_WIDTH || K % constants::TILE_WIDTH)
        log_info(LogTest, "M, N, K = {}, {}, {} are aligned to {}, {}, {}", M, N, K, M_aligned, N_aligned, K_aligned);

    uint32_t Mt = M_aligned / constants::TILE_WIDTH;
    uint32_t Nt = N_aligned / constants::TILE_WIDTH;
    uint32_t Kt = K_aligned / constants::TILE_WIDTH;
    return {Mt, Nt, Kt};
}

uint32_t get_in0_block_w(
    uint32_t per_core_Mt, uint32_t per_core_Nt, uint32_t Kt, uint32_t single_tile_size, uint32_t l1_size) {
    std::vector<uint32_t> in0_block_w_choices = {4, 2, 1};
    uint32_t num_buffer = 2;  // double buffering
    uint32_t in0_block_w = 0;
    uint32_t base_addr = L1_UNRESERVED_BASE;
    for (auto choice : in0_block_w_choices) {
        if (Kt % choice != 0)
            continue;

        uint32_t in0_cb_size = per_core_Mt * choice * num_buffer * single_tile_size;
        uint32_t in1_cb_size = per_core_Nt * choice * num_buffer * single_tile_size;
        uint32_t zero_cb_size = single_tile_size;
        uint32_t intermediate_cb_size = per_core_Mt * per_core_Nt * single_tile_size;

        uint32_t total_cb_size = in0_cb_size + in1_cb_size + zero_cb_size + intermediate_cb_size;

        // only taking first blocks from in0 and in1
        uint32_t per_core_in0_size = per_core_Mt * choice * single_tile_size;
        uint32_t per_core_in1_size = per_core_Nt * choice * single_tile_size;
        uint32_t per_core_out_size = per_core_Mt * per_core_Nt * single_tile_size;

        uint32_t total_buffer_size = per_core_in0_size + per_core_in1_size + per_core_out_size;
        if (base_addr + total_cb_size + total_buffer_size <= l1_size) {
            in0_block_w = choice;
            break;
        }
    }
    return in0_block_w;
}

CoreCoord get_core_range(
    uint32_t num_blocks_rows, uint32_t num_blocks_cols, uint32_t max_num_rows, uint32_t max_num_cols) {
    CoreCoord core_range(0, 0);
    if (num_blocks_rows <= max_num_rows && num_blocks_cols <= max_num_cols) {
        core_range.x = num_blocks_cols;
        core_range.y = num_blocks_rows;
    }
    return core_range;
}

tuple<MathFidelity, bool> get_compute_params(tt::ARCH arch) {
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    bool fp32_dest_acc_en = false;
    if (arch == tt::ARCH::WORMHOLE || arch == tt::ARCH::WORMHOLE_B0) {
        math_fidelity = MathFidelity::HiFi2;
        fp32_dest_acc_en = true;
    } else if (arch == tt::ARCH::GRAYSKULL) {
        math_fidelity = MathFidelity::HiFi4;
        fp32_dest_acc_en = false;
    }
    return {math_fidelity, fp32_dest_acc_en};
}

tuple<uint32_t, uint32_t> get_out_subblock_params(uint32_t per_core_Mt, uint32_t per_core_Nt) {
    constexpr std::array<tuple<uint32_t, uint32_t>, 20> SUBBLOCK_HW_CHOICES = {{
        {4, 2}, {2, 4}, {8, 1}, {1, 8}, {7, 1}, {1, 7}, {3, 2}, {2, 3}, {6, 1}, {1, 6},
        {5, 1}, {1, 5}, {2, 2}, {4, 1}, {1, 4}, {3, 1}, {1, 3}, {2, 1}, {1, 2}, {1, 1},
    }};

    for (auto& subblock_hw : SUBBLOCK_HW_CHOICES) {
        auto subblock_h = std::get<0>(subblock_hw);
        auto subblock_w = std::get<1>(subblock_hw);
        if (per_core_Mt % subblock_h == 0 and per_core_Nt % subblock_w == 0)
            return {subblock_h, subblock_w};
    }

    return {1, 1};
}

tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> get_all_buffers_addresses(
    uint32_t per_core_Mt, uint32_t per_core_Nt, uint32_t in0_block_w, uint32_t single_tile_size) {
    uint32_t num_buffer = 2;  // double buffering
    uint32_t in0_cb_addr = L1_UNRESERVED_BASE;
    uint32_t in0_cb_size = per_core_Mt * in0_block_w * num_buffer * single_tile_size;
    uint32_t in1_cb_addr = in0_cb_addr + in0_cb_size;
    uint32_t in1_cb_size = per_core_Nt * in0_block_w * num_buffer * single_tile_size;
    uint32_t zero_cb_addr = in1_cb_addr + in1_cb_size;
    uint32_t zero_cb_size = single_tile_size;
    uint32_t out_cb_addr = zero_cb_addr + zero_cb_size;
    uint32_t out_cb_size = per_core_Mt * per_core_Nt * single_tile_size;

    uint32_t per_core_in0_tiles = per_core_Mt * in0_block_w;
    uint32_t per_core_in1_tiles = per_core_Nt * in0_block_w;
    uint32_t per_core_out_tiles = per_core_Mt * per_core_Nt;
    uint32_t in0_addr = out_cb_addr + out_cb_size;
    uint32_t in1_addr = in0_addr + (per_core_in0_tiles * single_tile_size);
    uint32_t out_addr = in1_addr + (per_core_in1_tiles * single_tile_size);

    return {in0_cb_addr, in1_cb_addr, zero_cb_addr, out_cb_addr, in0_addr, in1_addr, out_addr};
}

tt_metal::Program create_program(
    tt_metal::Device* device,
    tt::DataFormat cb_data_format,
    MathFidelity math_fidelity,
    uint32_t single_tile_size,
    CoreCoord core_range,
    uint32_t B,
    uint32_t Mt,
    uint32_t Nt,
    uint32_t Kt,
    bool bcast_batch,
    uint32_t in0_block_w,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t per_core_Mt,
    uint32_t per_core_Nt,
    uint32_t in0_cb_addr,
    uint32_t in1_cb_addr,
    uint32_t zero_cb_addr,
    uint32_t out_cb_addr,
    uint32_t in0_addr,
    uint32_t in1_addr,
    uint32_t out_addr) {
    tt_metal::Program program{};

    uint32_t num_buffer = 2;  // double buffer
    uint32_t in0_block_tiles = per_core_Mt * in0_block_w;
    uint32_t in0_CB_tiles = in0_block_tiles * num_buffer;
    uint32_t in0_CB_size = in0_CB_tiles * single_tile_size;
    uint32_t in1_block_tiles = per_core_Nt * in0_block_w;
    uint32_t in1_CB_tiles = in1_block_tiles * num_buffer;
    uint32_t in1_CB_size = in1_CB_tiles * single_tile_size;
    uint32_t out_block_tiles = per_core_Mt * per_core_Nt;
    uint32_t out_CB_tiles = out_block_tiles;  // No double buffer
    uint32_t out_CB_size = out_CB_tiles * single_tile_size;

    // Compute kernel compile time args
    uint32_t num_blocks = (Kt / in0_block_w);

    uint32_t in0_num_subblocks = (per_core_Mt / out_subblock_h);
    uint32_t in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks;
    uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;

    uint32_t in1_num_subblocks = (per_core_Nt / out_subblock_w);
    uint32_t in1_block_num_tiles = out_subblock_w * in0_block_w * in1_num_subblocks;
    uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;

    uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;

    vector<uint32_t> compute_kernel_args = {
        in0_block_w,             // in0_block_w
        in0_num_subblocks,       // in0_num_subblocks
        in0_block_num_tiles,     // in0_block_num_tiles
        in0_subblock_num_tiles,  // in0_subblock_num_tiles

        in1_num_subblocks,    // in1_num_subblocks
        in1_block_num_tiles,  // in1_block_num_tiles
        in1_per_core_w,       // in1_per_core_w

        num_blocks,  // num_blocks

        out_subblock_h,          // out_subblock_h
        out_subblock_w,          // out_subblock_w
        out_subblock_num_tiles,  // out_subblock_num_tiles
        B                        // batch
    };

    CoreRange all_cores{
        .start = {(std::size_t)0, (std::size_t)0},
        .end = {(std::size_t)core_range.x - 1, (std::size_t)core_range.y - 1}};

    // Create circular buffers
    uint32_t src0_cb_index = 0;
    auto cb_src0 = tt_metal::CreateCircularBuffers(
        program, src0_cb_index, all_cores, in0_CB_tiles, in0_CB_size, cb_data_format, in0_cb_addr);

    uint32_t src1_cb_index = 1;
    auto cb_src1 = tt_metal::CreateCircularBuffers(
        program, src1_cb_index, all_cores, in1_CB_tiles, in1_CB_size, cb_data_format, in1_cb_addr);

    // Dummy cb to store one tile of zeros for padding
    uint32_t in2_CB_tiles = 1;  // No double buffer
    // CB for padding; only need these in the senders
    // NOTE: For first core, initialize cb to the larger tile size to prevent accidentally writing 0 to L1 space during
    // cb init in the kernels
    uint32_t src2_cb_index = 2;
    auto in0_in1_sender_cb_src2 = tt_metal::CreateCircularBuffers(
        program, src2_cb_index, all_cores, in2_CB_tiles, in2_CB_tiles * single_tile_size, cb_data_format, zero_cb_addr);

    uint32_t output_cb_index = 16;  // output operands start at index 16
    uint32_t interm0_cb_index = 24;
    auto cb_output = tt_metal::CreateCircularBuffers(
        program,
        {output_cb_index, interm0_cb_index},
        CoreRangeSet({all_cores}),
        out_CB_tiles,
        out_CB_size,
        cb_data_format,
        out_cb_addr);

    // Create reader and writer kernels per core
    auto mm_in0_reader_kernel_id = tt_metal::CreateDataMovementKernel(
        program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/1_compute_mm/kernels/in0_reader_bmm_tile_layout.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_0_default});

    auto mm_in1_reader_writer_kernel_id = tt_metal::CreateDataMovementKernel(
        program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/1_compute_mm/kernels/in1_reader_writer_bmm_tile_layout.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_1_default});

    // Create compute kernel
    bool fp32_dest_acc_en = false;
    // Gelu currently has better accuracy when run in approx mode
    bool math_approx_mode = false;
    auto mm_kernel_id = tt_metal::CreateComputeKernel(
        program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/1_compute_mm/kernels/bmm_large_block_zm_fused_bias_activation.cpp",
        all_cores,
        tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_kernel_args});

    // Parameters for last row, col, or block
    uint32_t last_block_h = Mt % per_core_Mt == 0 ? per_core_Mt : Mt % per_core_Mt;
    uint32_t last_block_w = Nt % per_core_Nt == 0 ? per_core_Nt : Nt % per_core_Nt;
    uint32_t last_block_num_nonzero_subblocks_h = (last_block_h - 1) / out_subblock_h + 1;
    uint32_t last_block_num_nonzero_subblocks_w = (last_block_w - 1) / out_subblock_w + 1;
    uint32_t last_subblock_of_last_block_h =
        last_block_h % out_subblock_h == 0 ? out_subblock_h : last_block_h % out_subblock_h;
    uint32_t last_subblock_of_last_block_w =
        last_block_w % out_subblock_w == 0 ? out_subblock_w : last_block_w % out_subblock_w;
    uint32_t last_block_padded_subblock_tiles_addr_skip =
        single_tile_size * (out_subblock_w - last_subblock_of_last_block_w);
    uint32_t last_block_padded_block_tiles_w_skip =
        (out_subblock_w * out_subblock_h) * (per_core_Nt / out_subblock_w - last_block_num_nonzero_subblocks_w);
    uint32_t last_block_padded_block_tiles_h_skip =
        (per_core_Mt / out_subblock_h - last_block_num_nonzero_subblocks_h) * (per_core_Nt * out_subblock_h);

    for (int output_idx_y = 0; output_idx_y < core_range.y; output_idx_y++) {
        for (int output_idx_x = 0; output_idx_x < core_range.x; output_idx_x++) {
            int core_idx_x = output_idx_x;
            int core_idx_y = output_idx_y;
            CoreCoord core = {(std::size_t)core_idx_x, (std::size_t)core_idx_y};
            auto phy_core = device->worker_core_from_logical_core(core);

            // Write runtime args to device
            std::vector<uint32_t> mm_in0_reader_args = {
                (std::uint32_t)in0_addr,     // in0_buffer->address(), // in0_tensor_addr
                (std::uint32_t)0,            // K * per_core_Mt * output_idx_y, // in0_tensor_start_tile_id
                (std::uint32_t)1,            // in0_tensor_stride_w
                (std::uint32_t)in0_block_w,  // K, // in0_tensor_stride_h
                (std::uint32_t)in0_block_w,  // in0_tensor_next_block_stride

                (std::uint32_t)in0_block_w,                // in0_block_w
                (std::uint32_t)per_core_Mt,                // in0_block_h
                (std::uint32_t)in0_block_w * per_core_Mt,  // in0_block_num_tiles

                (std::uint32_t)num_blocks,  // num_blocks

                (std::uint32_t)Mt * Kt,      // MtKt
                (std::uint32_t)Kt * Nt,      // KtNt
                (std::uint32_t)B,            // batch
                (std::uint32_t)bcast_batch,  // bcast_B

                (std::uint32_t)phy_core.x,
                (std::uint32_t)phy_core.y,
            };

            uint32_t in1_tensor_stride_h = per_core_Nt;
            uint32_t out_tensor_stride_h = per_core_Nt;
            if (core_idx_x == core_range.x - 1) {
                in1_tensor_stride_h = last_block_w;
                out_tensor_stride_h = last_block_w;
            }

            std::vector<uint32_t> mm_in1_reader_writer_args = {
                (std::uint32_t)in1_addr,                   // in1_buffer->address(), // in1_tensor_addr
                (std::uint32_t)0,                          // per_core_Nt * output_idx_x, //in1_tensor_start_tile_id
                (std::uint32_t)1,                          // in1_tensor_stride_w
                (std::uint32_t)in1_tensor_stride_h,        // in1_tensor_stride_h
                (std::uint32_t)in0_block_w * per_core_Nt,  // in1_tensor_next_block_stride

                (std::uint32_t)per_core_Nt,                // in1_block_w
                (std::uint32_t)in0_block_w,                // in1_block_h
                (std::uint32_t)per_core_Nt * in0_block_w,  // in1_block_num_tiles

                (std::uint32_t)num_blocks,  // num_blocks

                (std::uint32_t)Mt * Kt,      // MtKt
                (std::uint32_t)Kt * Nt,      // KtNt
                (std::uint32_t)B,            // batch
                (std::uint32_t)bcast_batch,  // bcast_B

                (std::uint32_t)phy_core.x,
                (std::uint32_t)phy_core.y,

                (std::uint32_t)out_addr,             // out_buffer->address(), // out_tensor_addr
                (std::uint32_t)0,                    // output_idx_x * per_core_Nt + output_idx_y * per_core_Mt * N, //
                                                     // out_tensor_start_tile_id
                (std::uint32_t)1,                    // out_tensor_stride_w
                (std::uint32_t)out_tensor_stride_h,  // out_tensor_stride_h
                (std::uint32_t)out_subblock_w,       // out_tensor_next_subblock_stride_w
                (std::uint32_t)out_subblock_h * out_tensor_stride_h,  // out_tensor_next_subblock_stride_h

                (std::uint32_t)out_subblock_w,                     // out_subblock_w
                (std::uint32_t)out_subblock_h,                     // out_subblock_h
                (std::uint32_t)(out_subblock_w * out_subblock_h),  // out_subblocks_w * out_subblocks_h
                (std::uint32_t)(per_core_Nt / out_subblock_w),     // out_num_subblocks_w
                (std::uint32_t)(per_core_Mt / out_subblock_h),     // out_num_subblocks_h

                (std::uint32_t)Mt * Nt  // MtNt
            };

            if (core_idx_y == core_range.y - 1) {
                mm_in0_reader_args.push_back(last_block_h);  // last_block_h
            } else {
                mm_in0_reader_args.push_back(per_core_Mt);
            }

            if (core_idx_x == core_range.x - 1) {
                mm_in1_reader_writer_args.push_back(last_block_w);
            } else {
                mm_in1_reader_writer_args.push_back(per_core_Nt);
            }

            if (core_idx_x == core_range.x - 1 && core_idx_y == core_range.y - 1) {
                // padding args (WRITER)
                mm_in1_reader_writer_args.push_back(last_block_num_nonzero_subblocks_h);
                mm_in1_reader_writer_args.push_back(last_subblock_of_last_block_h);
                mm_in1_reader_writer_args.push_back(last_block_padded_block_tiles_h_skip);
                mm_in1_reader_writer_args.push_back(last_block_num_nonzero_subblocks_w);
                mm_in1_reader_writer_args.push_back(last_subblock_of_last_block_w);
                mm_in1_reader_writer_args.push_back(last_block_padded_subblock_tiles_addr_skip);
                mm_in1_reader_writer_args.push_back(last_block_padded_block_tiles_w_skip);
            } else if (core_idx_y == core_range.y - 1) {
                // padding args (WRITER)
                mm_in1_reader_writer_args.push_back(last_block_num_nonzero_subblocks_h);
                mm_in1_reader_writer_args.push_back(last_subblock_of_last_block_h);
                mm_in1_reader_writer_args.push_back(last_block_padded_block_tiles_h_skip);
                mm_in1_reader_writer_args.push_back(per_core_Nt / out_subblock_w);
                mm_in1_reader_writer_args.push_back(out_subblock_w);
                mm_in1_reader_writer_args.push_back(0);
                mm_in1_reader_writer_args.push_back(0);
            } else if (core_idx_x == core_range.x - 1) {
                // padding args (WRITER)
                mm_in1_reader_writer_args.push_back(per_core_Mt / out_subblock_h);
                mm_in1_reader_writer_args.push_back(out_subblock_h);
                mm_in1_reader_writer_args.push_back(0);
                mm_in1_reader_writer_args.push_back(last_block_num_nonzero_subblocks_w);
                mm_in1_reader_writer_args.push_back(last_subblock_of_last_block_w);
                mm_in1_reader_writer_args.push_back(last_block_padded_subblock_tiles_addr_skip);
                mm_in1_reader_writer_args.push_back(last_block_padded_block_tiles_w_skip);
            } else {
                // padding args (WRITER)
                mm_in1_reader_writer_args.push_back(per_core_Mt / out_subblock_h);
                mm_in1_reader_writer_args.push_back(out_subblock_h);
                mm_in1_reader_writer_args.push_back(0);
                mm_in1_reader_writer_args.push_back(per_core_Nt / out_subblock_w);
                mm_in1_reader_writer_args.push_back(out_subblock_w);
                mm_in1_reader_writer_args.push_back(0);
                mm_in1_reader_writer_args.push_back(0);
            }
            tt_metal::SetRuntimeArgs(program, mm_in0_reader_kernel_id, core, mm_in0_reader_args);
            tt_metal::SetRuntimeArgs(program, mm_in1_reader_writer_kernel_id, core, mm_in1_reader_writer_args);
        }
    }
    return std::move(program);
}

std::vector<float> generate_fp32_random(uint32_t num_elems, int32_t rand_max_val = 100) {
    std::vector<float> vec(num_elems);
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    auto rand_float = std::bind(std::uniform_real_distribution<float>(0, rand_max_val), std::mt19937(seed));
    for (uint32_t i = 0; i < num_elems; ++i) {
        vec.at(i) = static_cast<float>(rand_float());
    }
    return vec;
}

// Given a tensor that is row-major datums, make it tilized
// so that its row major within a tile, and each tile's data
// is contiguous
template <typename T>
std::vector<T> tilize(std::vector<T> data, int rows, int cols) {
    TT_ASSERT(rows % 32 == 0);
    TT_ASSERT(cols % 32 == 0);
    int num_tiles_r = rows / 32;
    int num_tiles_c = cols / 32;
    std::vector<T> result;
    for (auto r = 0; r < num_tiles_r; r++) {
        for (auto c = 0; c < num_tiles_c; c++) {
            for (auto j = 0; j < 32; j++) {      // tile rows
                for (auto i = 0; i < 32; i++) {  // tile cols
                    // each row of tiles is 32x32 * num_tiles_c
                    // each row within the row of tiles is cols
                    // each col of tiles is 32
                    // pick row of tiles, pick the row within the tile, pick col tile
                    int index = r * 32 * 32 * num_tiles_c + j * cols + c * 32 + i;
                    result.push_back(data.at(index));
                }
            }
        }
    }
    return result;
}

// Given a tilized data (each tile's data is contiguous and row major within the
// tile) transform it back to row major full tensor. (This function inverts the
// tilize() function)
template <typename T>
std::vector<T> untilize(std::vector<T> data, int rows, int cols) {
    TT_ASSERT(rows % 32 == 0);
    TT_ASSERT(cols % 32 == 0);
    int num_tiles_r = rows / 32;
    int num_tiles_c = cols / 32;
    std::vector<T> result;
    for (auto r = 0; r < num_tiles_r; r++) {
        for (auto i = 0; i < 32; i++) {
            for (auto c = 0; c < num_tiles_c; c++) {
                int offset = r * 32 * 32 * num_tiles_c + c * 32 * 32 + i * 32;
                for (auto j = 0; j < 32; j++) {
                    result.push_back(data.at(offset + j));
                }
            }
        }
    }

    return result;
}

template <typename T>
std::vector<T> get_row_slice(std::vector<T> data, int start_row_index, int num_rows, int rows, int cols) {
    std::vector<T> result;
    for (int i = start_row_index * cols; i < (start_row_index + num_rows) * cols; i++) {
        result.push_back(data.at(i));
    }
    return result;
}

template <typename T>
std::vector<T> get_col_slice(std::vector<T> data, int start_col_index, int num_cols, int rows, int cols) {
    std::vector<T> result;
    for (int r = 0; r < rows; r++) {
        for (int c = start_col_index; c < (start_col_index + num_cols); c++) {
            result.push_back(data.at(r * cols + c));
        }
    }
    return result;
}

void prepare_inputs(
    tt_metal::Device* device,
    CoreCoord core_range,
    uint32_t Mt,
    uint32_t Nt,
    uint32_t Kt,
    uint32_t per_core_Mt,
    uint32_t per_core_Nt,
    uint32_t in0_block_w,
    uint32_t single_tile_size,
    uint32_t in0_addr,
    uint32_t in1_addr,
    uint32_t zero_cb_addr,
    std::vector<std::vector<float>>& in0_bfp8_unpack_slice,
    std::vector<std::vector<float>>& in1_bfp8_unpack_slice) {
    bool pass = true;
    auto in0_vec = generate_fp32_random(Mt * Kt * constants::TILE_HW);
    auto identity = generate_fp32_random(Kt * Nt * constants::TILE_HW);
    std::vector<uint32_t> zero_cb(single_tile_size / sizeof(uint32_t), 0);

    uint32_t num_cores_y = core_range.y;
    uint32_t num_cores_x = core_range.x;

    uint32_t last_block_h = Mt % per_core_Mt == 0 ? per_core_Mt : Mt % per_core_Mt;
    uint32_t last_block_w = Nt % per_core_Nt == 0 ? per_core_Nt : Nt % per_core_Nt;

    for (int r = 0; r < num_cores_y; r++) {
        int num_r = (r == num_cores_y - 1) ? (last_block_h) : (per_core_Mt);
        std::vector<float> activation_slice =
            get_row_slice(in0_vec, r * per_core_Mt * 32, num_r * 32, Mt * 32, Kt * 32);

        auto activation_slice2 = get_col_slice(activation_slice, 0, in0_block_w * 32, num_r * 32, Kt * 32);
        auto in0_tilized = tilize(activation_slice2, num_r * 32, in0_block_w * 32);
        std::vector<uint32_t> in0 =
            pack_fp32_vec_as_bfp8_tiles(in0_tilized, /*row_major_input=*/true, /*is_exp_a=*/false);

        // for cpu-ref mm
        auto unpack_vec = unpack_bfp8_tiles_into_float_vec(in0, true, false);
        auto untilize_vec = untilize(unpack_vec, num_r * 32, in0_block_w * 32);
        in0_bfp8_unpack_slice.push_back(untilize_vec);

        for (int c = 0; c < num_cores_x; c++) {
            int num_c = (c == num_cores_x - 1) ? (last_block_w) : (per_core_Nt);
            std::vector<float> in1_slice = get_col_slice(identity, c * per_core_Nt * 32, num_c * 32, Kt * 32, Nt * 32);
            auto in1_slice2 = get_row_slice(in1_slice, 0, in0_block_w * 32, Kt * 32, num_c * 32);

            CoreCoord core = {(std::size_t)c, (std::size_t)r};
            pass &= tt_metal::detail::WriteToDeviceL1(device, core, in0_addr, in0);
            TT_ASSERT(pass);

            auto identity_tilized = tilize(in1_slice2, in0_block_w * 32, num_c * 32);
            std::vector<uint32_t> in1 =
                pack_fp32_vec_as_bfp8_tiles(identity_tilized, /*row_major_input=*/true, /*is_exp_a=*/false);
            pass &= tt_metal::detail::WriteToDeviceL1(device, core, in1_addr, in1);
            TT_ASSERT(pass);

            pass &= tt_metal::detail::WriteToDeviceL1(device, core, zero_cb_addr, zero_cb);
            TT_ASSERT(pass);

            // for cpu-ref mm
            auto unpack_vec2 = unpack_bfp8_tiles_into_float_vec(in1, true, false);
            auto untilize_vec2 = untilize(unpack_vec2, in0_block_w * 32, num_c * 32);
            in1_bfp8_unpack_slice.push_back(untilize_vec2);
        }
    }
}

inline vector<float> gold_mm(
    const vector<uint32_t> shapeA,
    const vector<float>& A,
    const vector<uint32_t>& shapeB,
    const vector<float>& B,
    const uint32_t& num_blocks,
    bool acc16 = false) {
    TT_ASSERT(shapeB[0] == 1 && shapeA[0] == 1);
    uint32_t nb = shapeA[1];
    TT_ASSERT(shapeB[1] == nb);
    uint32_t M = shapeA[2];
    uint32_t K = shapeA[3];
    TT_ASSERT(shapeB[2] == K);
    uint32_t N = shapeB[3];

    vector<uint32_t> shapeC{1, nb, M, N};
    TensAddr addrC(shapeC);
    TensAddr addrA(shapeA);
    TensAddr addrB(shapeB);
    vector<float> resultf(addrC.numel());
    std::fill(resultf.begin(), resultf.end(), 0);

    for (int ib = 0; ib < nb; ib++)
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++)
                for (int k = 0; k < K; k++) {
                    auto offsA = addrA.offs(0, ib, m, k);
                    auto offsB = addrB.offs(0, ib, k, n);
                    auto offsC = addrC.offs(0, ib, m, n);

                    float aa = bfloat16(A[offsA]).to_float();
                    float bb = bfloat16(B[offsB]).to_float();
                    resultf[offsC] += aa * bb;
                    if (acc16)
                        resultf[offsC] = bfloat16(resultf[offsC]).to_float();
                }

    // write back to fp16 after we accumulated in fp32
    for (int ib = 0; ib < nb; ib++)
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++) {
                auto offsC = addrC.offs(0, ib, m, n);

                float cc = resultf[offsC];
                for (int block = 1; block < num_blocks; block++) {
                    resultf[offsC] += cc;
                    if (acc16)
                        resultf[offsC] = bfloat16(resultf[offsC]).to_float();
                }
            }
    return resultf;
}

bool validation(
    tt_metal::Device* device,
    CoreCoord core_range,
    uint32_t Mt,
    uint32_t Nt,
    uint32_t Kt,
    uint32_t per_core_Mt,
    uint32_t per_core_Nt,
    uint32_t in0_block_w,
    uint32_t out_addr,
    uint32_t single_tile_size,
    bool fp32_dest_acc_en,
    std::vector<std::vector<float>>& in0_bfp8_unpack_slice,
    std::vector<std::vector<float>>& in1_bfp8_unpack_slice) {
    auto comparison_function = [](float a, float b) {
        const float rtol = 0.05f;  // TODO(AP): need a spec for reference
        const float atol = 0.05f;
        float maxabs = fmaxf(fabsf(a), fabsf(b));
        float absdiff = fabsf(a - b);
        auto result = (absdiff <= atol) || absdiff < rtol * maxabs;
        return result;
    };

    bool pass = true;
    uint32_t num_cores_y = core_range.y;
    uint32_t num_cores_x = core_range.x;
    uint32_t num_blocks = Kt / in0_block_w;
    uint32_t last_block_h = Mt % per_core_Mt == 0 ? per_core_Mt : Mt % per_core_Mt;
    uint32_t last_block_w = Nt % per_core_Nt == 0 ? per_core_Nt : Nt % per_core_Nt;
    uint32_t diff_count = 0;
    for (int r = 0; r < num_cores_y; ++r) {
        for (int c = 0; c < num_cores_x; ++c) {
            CoreCoord core = {(size_t)c, (size_t)r};
            std::vector<uint32_t> result_vec;
            uint32_t num_r = (r == num_cores_y - 1) ? (last_block_h) : (per_core_Mt);
            uint32_t num_c = (c == num_cores_x - 1) ? (last_block_w) : (per_core_Nt);
            tt_metal::detail::ReadFromDeviceL1(device, core, out_addr, num_r * num_c * single_tile_size, result_vec);
            auto result_flat_layout = unpack_bfp8_tiles_into_float_vec(result_vec, true, false);
            auto result_untilized = untilize(result_flat_layout, num_r * 32, num_c * 32);

            // cpu ref
            std::vector<uint32_t> shapeA = {1, 1, num_r * 32, in0_block_w * 32};
            std::vector<uint32_t> shapeB = {1, 1, in0_block_w * 32, num_c * 32};
            auto per_core_golden = gold_mm(
                shapeA, in0_bfp8_unpack_slice[r], shapeB, in1_bfp8_unpack_slice[c], num_blocks, fp32_dest_acc_en);

            if (result_untilized.size() != per_core_golden.size()) {
                pass = false;
            } else {
                for (int i = 0; i < result_untilized.size(); ++i) {
                    float a = result_untilized.at(i);
                    float b = per_core_golden.at(i);
                    if (not comparison_function(a, b)) {
                        diff_count++;
                        pass = false;

                    }
                }
            }
        }
    }

    uint32_t total_count = Mt * Nt * constants::TILE_HW;
    if (!pass) {
        log_error(LogTest, "{}/{} elements are not matched", diff_count, total_count);
    }
    return pass;
}
