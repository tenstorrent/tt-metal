// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <cerrno>
#include <fmt/base.h>
#include <cstdlib>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/bfloat8.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tt_metal_profiler.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/distributed.hpp>
#include <hostdevcommon/common_values.hpp>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <ratio>
#include <set>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt-metalium/base_types.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/hal_types.hpp>
#include "hostdevcommon/kernel_structs.h"
#include <tt-metalium/kernel_types.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include "test_common.hpp"
#include <tt-metalium/tt_backend_api_types.hpp>
#include "tt_metal/test_utils/deprecated/tensor.hpp"
#include "tt_metal/tt_metal/perf_microbenchmark/common/util.hpp"
#include <umd/device/types/arch.hpp>

using std::vector;
using namespace tt;
////////////////////////////////////////////////////////////////////////////////
// This benchmark measures the compute performance of matmul. When in the slow
// dispatch mode, it uses LaunchProgram API and measures performance via device
// profiler. In the fast dispatch mode, it uses EnqueueMeshWorkload API and measures
// the execution time. Regarding kernels, the compute kernel used
// “bmm_large_block_zm_fused_bias_activation.cpp” as is and the data movement
// kernels were implemented with reference to kernels of
// multi_core_reuse_mcast_2d_optimized bmm op in tt_dnn. Matmul parameters such
// as in0_block_w, out_subblock_h, out_subblock_w are set considering to the
// given input shape and L1 size.
//
// Disclaimer:
//   - This benchmark uses a little trick for both inputs (M x K and K x N) to
//   support as large an input as possible. Only the first block of each input
//   (per_core_Mt x in0_block_w, in0_block_w x per_core_Nt) is stored in L1 and
//   used repeatedly for the total number of blocks.
//   - For validation, this benchmark sets in1 buffer as the identity matrix
//   so validation function checks out buffer where the first block of in0 buffer
//   is stored like a pattern.
//   - Currently, TT's matmul implementation may not be able to use all Tensix
//   cores for certain input shapes. In that case, only some cores are used with
//   a warning message.
//   - To measure performance in the slow dispatch mode, build tt_metal project
//   with Tracy enabled. This benchmark
//   copied device profiler's internal code to get the "t0 to any riscfw end"
//   cycles. If device profiler is changed, it also should be updated.
//   Otherwise, it may get inappropriate cycle value.
//
// Usage example:
//   ./test_compute_mm
//     --m <size in elements>
//     --n <size in elements>
//     --k <size in elements>
//     --dtype <data type for single core test, 0: bfp8_b, 1: fp16>
//     --fidel <math fidelity for single core test, 0: LoFi, 1: HiFi2>
//     --block <block version of matmul for single core test, 0: matmul_tiles, 1: matmul_block>
//     --packer <packer l1 acc mode for single core test, 0: no enabled, 1: enable>
//     --one-core <single core test, 0: multi-core test, 1: single-core test>
//     --num-blocks <number of blocks to feed into compute kernel in single core test>
//     --fast-dispatch (set to use fast dispatch mode)
//     --num-tests <count of tests>
//     --bypass-check (set to bypass checking performance criteria fulfillment)
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////
//                      Function Forward Declaration
////////////////////////////////////////////////////////////////////////////
uint32_t get_l1_size(tt::ARCH arch);

double get_tt_npu_rpeak_tflops(tt::ARCH arch, CoreCoord grid_size, int tt_npu_clock);

std::tuple<uint32_t, uint32_t, uint32_t> get_aligned_input_tile_num(uint32_t M, uint32_t N, uint32_t K);

uint32_t get_in0_block_w(
    uint32_t per_core_Mt,
    uint32_t per_core_Nt,
    uint32_t Kt,
    uint32_t single_tile_size,
    uint32_t l1_size,
    uint32_t l1_unreserved_base);

CoreCoord get_core_range(
    uint32_t num_blocks_rows, uint32_t num_blocks_cols, uint32_t max_num_rows, uint32_t max_num_cols);

std::tuple<MathFidelity, bool> get_compute_params(tt::ARCH arch);

std::tuple<uint32_t, uint32_t> get_out_subblock_params(uint32_t per_core_Mt, uint32_t per_core_Nt, uint32_t choice);

std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> get_all_buffers_addresses(
    uint32_t per_core_Mt,
    uint32_t per_core_Nt,
    uint32_t in0_block_w,
    uint32_t single_tile_size,
    uint32_t l1_unreserved_base);

std::vector<float> generate_fp32_random(uint32_t num_elems, int32_t rand_max_val);

template <typename T>
std::vector<T> get_row_slice(std::vector<T> data, int start_row_index, int num_rows, int rows, int cols);

template <typename T>
std::vector<T> get_col_slice(std::vector<T> data, int start_col_index, int num_cols, int rows, int cols);

void prepare_inputs(
    tt_metal::distributed::MeshDevice* device,
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
    uint32_t in2_cb_addr,
    bool dtype,
    std::vector<std::vector<float>>& in0_bfp8_unpack_slice,
    std::vector<std::vector<float>>& in1_bfp8_unpack_slice);

tt_metal::Program create_program_single_core(
    tt_metal::distributed::MeshDevice* device,
    tt::DataFormat cb_data_format,
    MathFidelity math_fidelity,
    bool fp32_dest_acc_en,
    uint32_t single_tile_size,
    CoreCoord core_range,
    uint32_t Mt,
    uint32_t Nt,
    uint32_t Kt,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& in0_cb_addr,
    const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& in1_cb_addr,
    const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& out_cb_addr,
    bool matmul_block,
    bool packer_l1,
    uint32_t num_blocks,
    uint32_t interm_cb_dtype);

tt_metal::Program create_program(
    tt_metal::distributed::MeshDevice* device,
    tt::DataFormat cb_data_format,
    MathFidelity math_fidelity,
    bool fp32_dest_acc_en,
    uint32_t single_tile_size,
    CoreCoord core_range,
    uint32_t Mt,
    uint32_t Nt,
    uint32_t Kt,
    uint32_t in0_block_w,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t per_core_Mt,
    uint32_t per_core_Nt,
    uint32_t in0_cb_addr,
    uint32_t in1_cb_addr,
    uint32_t in2_cb_addr,
    uint32_t out_cb_addr,
    uint32_t in0_addr,
    uint32_t in1_addr,
    uint32_t out_addr,
    bool matmul_block,
    bool packer_l1_acc);

bool validation_single_core(
    tt_metal::distributed::MeshDevice* device,
    const tt::deprecated::Tensor<bfloat16>& tensor_in0,
    const tt::deprecated::Tensor<bfloat16>& tensor_in1,
    uint32_t num_blocks,
    uint32_t Mt,
    uint32_t Nt,
    uint32_t Kt,
    const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& out_buffer);

bool validation(
    tt_metal::distributed::MeshDevice* device,
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

bool validation_single_core_fp8(
    tt_metal::distributed::MeshDevice* device,
    const tt::deprecated::Tensor<float>& tensor_in0,
    const tt::deprecated::Tensor<float>& tensor_in1,
    uint32_t num_blocks,
    uint32_t Mt,
    uint32_t Nt,
    uint32_t Kt,
    const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& out_buffer);

std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> create_and_transfer_data_sharded_cb(
    tt_metal::distributed::MeshDevice* device, const vector<uint32_t>& activations, uint32_t Mt, uint32_t Nt);

std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> create_and_transfer_data_sharded_cb_fp8(
    tt_metal::distributed::MeshDevice* device, const vector<uint32_t>& activations, uint32_t Mt, uint32_t Nt);

////////////////////////////////////////////////////////////////////////////
//                      Main
////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
    bool pass = true;
    bool bypass_check = false;
    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Initial Runtime Args Parse
        ////////////////////////////////////////////////////////////////////////////
        std::vector<std::string> input_args(argv, argv + argc);
        uint32_t M;
        uint32_t N;
        uint32_t K;
        uint32_t dtype = 0;  // bfp8
        uint32_t fidel = 0;  // lofi
        uint32_t num_tests = 10;
        uint32_t num_blocks = 1;
        bool matmul_block = false;
        bool packer_l1 = false;
        bool fp32 = false;
        uint32_t interm_cb_dtype = 0;
        uint32_t subblock_choice = 0;
        bool single_core = false;
        bool fast_dispatch_mode = false;
        try {
            std::tie(M, input_args) = test_args::get_command_option_uint32_and_remaining_args(input_args, "--m", 11264);
            std::tie(N, input_args) = test_args::get_command_option_uint32_and_remaining_args(input_args, "--n", 3072);
            std::tie(K, input_args) = test_args::get_command_option_uint32_and_remaining_args(input_args, "--k", 768);
            std::tie(dtype, input_args) =
                test_args::get_command_option_uint32_and_remaining_args(input_args, "--dtype", 0);
            std::tie(fidel, input_args) =
                test_args::get_command_option_uint32_and_remaining_args(input_args, "--fidel", 0);
            std::tie(matmul_block, input_args) =
                test_args::get_command_option_uint32_and_remaining_args(input_args, "--block", 0);
            std::tie(packer_l1, input_args) =
                test_args::get_command_option_uint32_and_remaining_args(input_args, "--packer", 0);
            std::tie(fp32, input_args) =
                test_args::get_command_option_uint32_and_remaining_args(input_args, "--fp32", 0);
            std::tie(interm_cb_dtype, input_args) =
                test_args::get_command_option_uint32_and_remaining_args(input_args, "--interm-cb", 1);
            std::tie(subblock_choice, input_args) =
                test_args::get_command_option_uint32_and_remaining_args(input_args, "--subblock-index", 0);
            std::tie(single_core, input_args) =
                test_args::get_command_option_uint32_and_remaining_args(input_args, "--one-core", 0);
            std::tie(num_blocks, input_args) =
                test_args::get_command_option_uint32_and_remaining_args(input_args, "--num-blocks", 1);
            std::tie(num_tests, input_args) =
                test_args::get_command_option_uint32_and_remaining_args(input_args, "--num-tests", 10);
            std::tie(fast_dispatch_mode, input_args) =
                test_args::has_command_option_and_remaining_args(input_args, "--fast-dispatch");

            std::tie(bypass_check, input_args) =
                test_args::has_command_option_and_remaining_args(input_args, "--bypass-check");

            test_args::validate_remaining_args(input_args);
        } catch (const std::exception& e) {
            log_error(LogTest, "Command line arguments found exception", e.what());
        }

        if (not single_core) {
            TT_FATAL(dtype == 0, "multi core test only supports bfp8_b");
            TT_FATAL(packer_l1 == 0, "multi core test does not support packer_l1 arg");
        }

        ////////////////////////////////////////////////////////////////////////////
        //                      Env and Device Setup
        ////////////////////////////////////////////////////////////////////////////
        if (single_core) {
            TT_FATAL(fast_dispatch_mode, "single core test only supports in fast dispatch mode");
        } else if (!fast_dispatch_mode) {
            setenv("TT_METAL_SLOW_DISPATCH_MODE", "1", true);

            bool device_profiler = tt::tt_metal::MetalContext::instance().rtoptions().get_profiler_enabled();
            TT_FATAL(
                device_profiler,
                "Before running the program, do one of the following in a shell: "
                "either export the environment variable by executing export TT_METAL_DEVICE_PROFILER=1, "
                "or run the program with TT_METAL_DEVICE_PROFILER=1 prefixed to the command");
        }

        int pci_express_slot = 0;
        auto mesh_device_map = tt::tt_metal::distributed::MeshDevice::create_unit_meshes(
            {pci_express_slot},
            DEFAULT_L1_SMALL_SIZE,
            DEFAULT_TRACE_REGION_SIZE,
            1 /* num_command_queues */,
            tt::tt_metal::MetalContext::instance().rtoptions().get_dispatch_core_config());

        const std::shared_ptr<tt_metal::distributed::MeshDevice>& device = mesh_device_map.at(pci_express_slot);
        uint32_t l1_unreserved_base = device->allocator()->get_base_allocator_addr(HalMemType::L1);
        const tt::ARCH arch = device->arch();
        ////////////////////////////////////////////////////////////////////////////
        //                      Check Input Args
        ////////////////////////////////////////////////////////////////////////////
        uint32_t l1_size = get_l1_size(arch);
        auto [Mt, Nt, Kt] = get_aligned_input_tile_num(M, N, K);
        log_info(LogTest, "Input M, N, K = {}, {}, {} / {}, {}, {} tile(s)", M, N, K, Mt, Nt, Kt);

        tt::DataFormat data_format = tt::DataFormat::Bfp8_b;
        if (single_core) {
            data_format = dtype == 0 ? tt::DataFormat::Bfp8_b : tt::DataFormat::Float16_b;
        }
        uint32_t single_tile_size = tt::tile_size(data_format);
        TT_FATAL(single_tile_size == (dtype == 0 ? (256 * 4) + (16 * 4) : 2048), "Unexpected tile size");

        auto grid_size = device->compute_with_storage_grid_size();
        if (single_core) {
            grid_size.x = 1;
            grid_size.y = 1;
        }

        uint32_t num_cores_y = grid_size.y;
        uint32_t num_cores_x = grid_size.x;
        uint32_t per_core_Mt = ((Mt - 1) / num_cores_y) + 1;
        uint32_t per_core_Nt = ((Nt - 1) / num_cores_x) + 1;
        uint32_t in0_block_w =
            get_in0_block_w(per_core_Mt, per_core_Nt, Kt, single_tile_size, l1_size, l1_unreserved_base);
        if (in0_block_w == 0) {
            log_error(
                LogTest,
                "M, N, K = {}, {}, {} cannot be tested due to insufficient L1 "
                "memory.",
                M,
                N,
                K);
            TT_FATAL(false, "Multi core matmul not supported with L1 memory constraints");
        }

        uint32_t num_blocks_y = ((Mt - 1) / per_core_Mt) + 1;
        uint32_t num_blocks_x = ((Nt - 1) / per_core_Nt) + 1;
        uint32_t num_blocks_total = num_blocks_y * num_blocks_x;
        TT_FATAL(
            num_blocks_total <= num_cores_x * num_cores_y,
            "num_blocks_total {} exceeds available cores {}",
            num_blocks_total,
            num_cores_x * num_cores_y);
        CoreCoord core_range = get_core_range(num_blocks_y, num_blocks_x, num_cores_y, num_cores_x);
        if (core_range.y != num_cores_y || core_range.x != num_cores_x) {
            log_warning(
                LogTest,
                "This run only use {} cores instead {} cores",
                core_range.y * core_range.x,
                num_cores_y * num_cores_x);
        }

        ////////////////////////////////////////////////////////////////////////////
        //                      Tensor Setup
        ////////////////////////////////////////////////////////////////////////////
        std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> input_buffer0;
        std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> input_buffer1;
        std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> output_buffer;
        SHAPE shape_in0 = {1, 1, M, K};
        tt::deprecated::Tensor<bfloat16> tensor_in0_fp16 = tt::deprecated::initialize_tensor<bfloat16>(
            shape_in0,
            tt::deprecated::Initialize::ONES,
            0,
            100,
            std::chrono::system_clock::now().time_since_epoch().count());
        tt::deprecated::Tensor<float> tensor_in0_fp8 = tt::deprecated::initialize_tensor<float>(
            shape_in0,
            tt::deprecated::Initialize::ONES,
            0,
            100,
            std::chrono::system_clock::now().time_since_epoch().count());
        SHAPE shape_in1 = {1, 1, K, N};
        tt::deprecated::Tensor<bfloat16> tensor_in1_fp16 = tt::deprecated::initialize_tensor<bfloat16>(
            shape_in1,
            tt::deprecated::Initialize::ONES,
            0,
            100,
            std::chrono::system_clock::now().time_since_epoch().count());
        tt::deprecated::Tensor<float> tensor_in1_fp8 = tt::deprecated::initialize_tensor<float>(
            shape_in1,
            tt::deprecated::Initialize::ONES,
            0,
            100,
            std::chrono::system_clock::now().time_since_epoch().count());

        if (single_core) {
            if (dtype == 1) {
                // in0
                auto activations_tilized = tilize_swizzled(tensor_in0_fp16.get_values(), M, K);
                auto activations_tile_layout =
                    convert_layout_tile_swizzled_to_tile_nfaces(tt::stl::make_const_span(activations_tilized));
                vector<uint32_t> activations = pack_bfloat16_vec_into_uint32_vec(activations_tile_layout);
                input_buffer0 = create_and_transfer_data_sharded_cb(device.get(), activations, Mt, Kt);

                // in1
                auto identity_tilized = tilize_swizzled(tensor_in1_fp16.get_values(), K, N);
                auto weights_tile_layout =
                    convert_layout_tile_swizzled_to_tile_nfaces(tt::stl::make_const_span(identity_tilized));
                auto weights = pack_bfloat16_vec_into_uint32_vec(weights_tile_layout);
                input_buffer1 = create_and_transfer_data_sharded_cb(device.get(), weights, Kt, Nt);

                // output
                SHAPE output_hsape = {1, 1, M, N};
                tt::deprecated::Tensor<bfloat16> out_tensor = tt::deprecated::initialize_tensor<bfloat16>(
                    output_hsape,
                    tt::deprecated::Initialize::ZEROS,
                    0,
                    100,
                    std::chrono::system_clock::now().time_since_epoch().count());
                vector<uint32_t> outputs = pack_bfloat16_vec_into_uint32_vec(out_tensor.get_values());
                output_buffer = create_and_transfer_data_sharded_cb(device.get(), outputs, Mt, Nt);

            } else {
                // in0
                auto activations_tilized = tilize_swizzled(tensor_in0_fp8.get_values(), M, K);
                std::vector<uint32_t> activations =
                    pack_as_bfp8_tiles(tt::stl::make_const_span(activations_tilized), true, false);
                input_buffer0 = create_and_transfer_data_sharded_cb_fp8(device.get(), activations, Mt, Kt);

                // in1
                auto identity_tilized = tilize_swizzled(tensor_in1_fp8.get_values(), K, N);
                auto weights = pack_as_bfp8_tiles(tt::stl::make_const_span(identity_tilized), true, false);
                input_buffer1 = create_and_transfer_data_sharded_cb_fp8(device.get(), weights, Kt, Nt);

                // output
                SHAPE output_hsape = {1, 1, M, N};
                tt::deprecated::Tensor<float> out_tensor = tt::deprecated::initialize_tensor<float>(
                    output_hsape,
                    tt::deprecated::Initialize::ZEROS,
                    0,
                    100,
                    std::chrono::system_clock::now().time_since_epoch().count());
                auto output_tilized = tilize_swizzled(out_tensor.get_values(), M, N);
                auto outputs = pack_as_bfp8_tiles(tt::stl::make_const_span(output_tilized), true, false);
                output_buffer = create_and_transfer_data_sharded_cb_fp8(device.get(), outputs, Mt, Nt);
            }
        }

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        auto [math_fidelity, fp32_dest_acc_en] = get_compute_params(arch);
        if (single_core) {
            math_fidelity = fidel == 0 ? MathFidelity::LoFi : MathFidelity::HiFi2;
            fp32_dest_acc_en = fp32 != 0;
        }
        auto [out_subblock_h, out_subblock_w] = get_out_subblock_params(per_core_Mt, per_core_Nt, subblock_choice);
        auto [in0_cb_addr, in1_cb_addr, in2_cb_addr, out_cb_addr, in0_addr, in1_addr, out_addr] =
            get_all_buffers_addresses(per_core_Mt, per_core_Nt, in0_block_w, single_tile_size, l1_unreserved_base);

        if (fp32_dest_acc_en and (out_subblock_h * out_subblock_w > 4)) {
            if (out_subblock_w >= 4) {
                out_subblock_h = 1;
                out_subblock_w = tt::tt_metal::find_max_block_size(out_subblock_w, 4);
            } else {
                while (out_subblock_h * out_subblock_w > 4) {
                    uint32_t div = tt::tt_metal::find_max_divisor(out_subblock_h, out_subblock_h - 1);
                    out_subblock_h = tt::tt_metal::find_max_block_size(out_subblock_h, div);
                }
            }
        }

        log_debug(LogTest, "grid_size.x {}", grid_size.x);
        log_debug(LogTest, "grid_size.y {}", grid_size.y);
        log_debug(LogTest, "per_core_Mt {}", per_core_Mt);
        log_debug(LogTest, "per_core_Nt {}", per_core_Nt);
        log_debug(LogTest, "in0_block_w {}", in0_block_w);
        log_debug(LogTest, "out_subblock_h {}", out_subblock_h);
        log_debug(LogTest, "out_subblock_w {}", out_subblock_w);

        tt::tt_metal::Program program;
        if (single_core) {
            program = create_program_single_core(
                device.get(),
                data_format,
                math_fidelity,
                fp32_dest_acc_en,
                single_tile_size,
                core_range,
                Mt,
                Nt,
                Kt,
                out_subblock_h,
                out_subblock_w,
                input_buffer0,
                input_buffer1,
                output_buffer,
                matmul_block,
                packer_l1,
                num_blocks,
                interm_cb_dtype);
        } else {
            program = create_program(
                device.get(),
                data_format,
                math_fidelity,
                fp32_dest_acc_en,
                single_tile_size,
                core_range,
                Mt,
                Nt,
                Kt,
                in0_block_w,
                out_subblock_h,
                out_subblock_w,
                per_core_Mt,
                per_core_Nt,
                in0_cb_addr,
                in1_cb_addr,
                in2_cb_addr,
                out_cb_addr,
                in0_addr,
                in1_addr,
                out_addr,
                matmul_block,
                packer_l1);
        }

        ////////////////////////////////////////////////////////////////////////////
        //                      Input Setup
        ////////////////////////////////////////////////////////////////////////////
        // for validation
        std::vector<std::vector<float>> in0_bfp8_unpack_slice;
        std::vector<std::vector<float>> in1_bfp8_unpack_slice;
        if (not single_core) {
            prepare_inputs(
                device.get(),
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
                in2_cb_addr,
                dtype,
                in0_bfp8_unpack_slice,
                in1_bfp8_unpack_slice);
        }

        ////////////////////////////////////////////////////////////////////////////
        //                      Kernel Execution and Perf Profiling
        ////////////////////////////////////////////////////////////////////////////
        constexpr int giga_byte = 1000000;
        constexpr long long tera_byte = 1000000000000LL;
        int tt_npu_clock = get_tt_npu_clock(device->get_devices()[0]);
        double rpeak_tflops = get_tt_npu_rpeak_tflops(arch, grid_size, tt_npu_clock);
        std::vector<double> rmax_tflops;
        uint64_t num_of_matmul_ops =
            (2 * static_cast<uint64_t>(Kt) * 32 - 1) * (static_cast<uint64_t>(Mt) * static_cast<uint64_t>(Nt) * 1024);
        log_debug(LogTest, "number of matmul ops: {}", num_of_matmul_ops);

        log_info(LogTest, "Num tests {}", num_tests);
        // Create MeshWorkload
        auto mesh_workload = tt_metal::distributed::MeshWorkload();
        mesh_workload.add_program(tt::tt_metal::distributed::MeshCoordinateRange{{0, 0}, {0, 0}}, std::move(program));

        for (uint32_t i = 0; i < num_tests; ++i) {
            if (!fast_dispatch_mode) {
                log_debug(LogTest, "calling EnqueueMeshWorkload");
                tt_metal::distributed::EnqueueMeshWorkload(device->mesh_command_queue(), mesh_workload, true);
                log_debug(LogTest, "EnqueueMeshWorkload done");

                uint64_t t0_to_any_riscfw_end = get_t0_to_any_riscfw_end_cycle(
                    device->get_devices()[0], mesh_workload.get_programs().begin()->second);
                double cycle_time = 1 / static_cast<double>(tt_npu_clock) / giga_byte;
                auto execution_time = t0_to_any_riscfw_end * cycle_time;
                rmax_tflops.push_back(static_cast<double>(num_of_matmul_ops) / execution_time / tera_byte);

                log_debug(LogTest, "cycle time {:.8f}s", cycle_time);
                log_debug(LogTest, "t0_to_any_riscfw_end {}", t0_to_any_riscfw_end);
                log_info(
                    LogTest,
                    "time duration: {:.5}us ({}cycles) rmax_tflops {:.2f}",
                    execution_time,
                    t0_to_any_riscfw_end,
                    rmax_tflops[i]);
            } else {
                log_debug(LogTest, "calling EnqueueMeshWorkload");
                std::chrono::duration<double, std::nano> duration{};
                auto t_begin = std::chrono::high_resolution_clock::now();
                tt_metal::distributed::EnqueueMeshWorkload(device->mesh_command_queue(), mesh_workload, false);
                tt_metal::distributed::Finish(device->mesh_command_queue());
                auto t_end = std::chrono::high_resolution_clock::now();
                log_debug(LogTest, "EnqueueMeshWorkload done");
                tt_metal::ReadMeshDeviceProfilerResults(*device);

                if (single_core) {
                    uint64_t t0_to_any_riscfw_end =
                        get_t0_to_any_riscfw_end_cycle(device.get(), mesh_workload.get_programs().begin()->second);
                    double cycle_time = 1 / static_cast<double>(tt_npu_clock) / giga_byte;
                    auto execution_time = t0_to_any_riscfw_end * cycle_time;
                    rmax_tflops.push_back(static_cast<double>(num_of_matmul_ops) / execution_time / tera_byte);

                    log_debug(LogTest, "cycle time {:.8f}s", cycle_time);
                    log_debug(LogTest, "t0_to_any_riscfw_end {}", t0_to_any_riscfw_end);
                    log_info(
                        LogTest,
                        "time duration: {:.5}us ({}cycles) rmax_tflops {:.2f}",
                        execution_time,
                        t0_to_any_riscfw_end,
                        rmax_tflops[i]);
                } else {
                    duration = t_end - t_begin;
                    rmax_tflops.push_back(static_cast<double>(num_of_matmul_ops) / duration.count() / 1000);
                    log_info(LogTest, "time duration: {:.5} ns, rmax_tflops {:.2f}", duration.count(), rmax_tflops[i]);
                }
            }
        }

        auto avg_rmax_tflops = calculate_average(rmax_tflops);
        double rmax_per_rpeak = avg_rmax_tflops / rpeak_tflops;
        log_info(
            LogTest,
            "Avg Rmax(TFLOPS) {:.3f}, Rpeak {:.3f}, Rmax / Rpeak {:.2f}%",
            avg_rmax_tflops,
            rpeak_tflops,
            rmax_per_rpeak * 100);
        bool performance_result = true;
        if (rmax_per_rpeak < 0.9) {
            performance_result = false;
        }

        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        bool validation_result = true;
        if (single_core) {
            if (dtype == 1) {
                validation_result = validation_single_core(
                    device.get(), tensor_in0_fp16, tensor_in1_fp16, num_blocks, Mt, Nt, Kt, output_buffer);
            } else {
                validation_result = validation_single_core_fp8(
                    device.get(), tensor_in0_fp8, tensor_in1_fp8, num_blocks, Mt, Nt, Kt, output_buffer);
            }
        } else {
            validation_result = validation(
                device.get(),
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
        }

        if ((!validation_result || !performance_result) && !bypass_check) {
            log_error(
                LogTest,
                "The compute performance does not meet the criteria. "
                "Current: Rmax / Rpeak = {:.2f}%, goal: > 90%",
                rmax_per_rpeak * 100);
            pass = false;
        }

        pass &= device->close();

        // for csv
        log_info(tt::LogTest, "CSV_MICROBENCHMARK:title:test_compute_mm");
        log_info(tt::LogTest, "CSV_INPUT:M:{}:N:{}:K:{}:fast-dispatch:{}", M, N, K, fast_dispatch_mode);
        log_info(tt::LogTest, "CSV_OUTPUT:RMax(TFLOPS):{:.2f}", avg_rmax_tflops);
        log_info(tt::LogTest, "CSV_RESULT:pass:{}", pass);

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
        log_error(LogTest, "Test Failed");
    }

    return 0;
}

////////////////////////////////////////////////////////////////////////////
//                      Function Implementation
////////////////////////////////////////////////////////////////////////////
uint32_t get_l1_size(tt::ARCH arch) {
    constexpr uint32_t GS_L1_SIZE = 1048576;
    constexpr uint32_t WH_L1_SIZE = 1499136;
    constexpr uint32_t BH_L1_SIZE = 1499136;

    uint32_t l1_size = 0;
    if (arch == tt::ARCH::WORMHOLE_B0) {
        l1_size = WH_L1_SIZE;
    } else if (arch == tt::ARCH::GRAYSKULL) {
        l1_size = GS_L1_SIZE;
    } else if (arch == tt::ARCH::BLACKHOLE) {
        l1_size = BH_L1_SIZE;
    }
    return l1_size;
}

double get_tt_npu_rpeak_tflops(tt::ARCH arch, CoreCoord grid_size, int tt_npu_clock) {
    constexpr double BH_FPU_BFP8_TFLOPS_PER_TENSIX = 2.97;
    constexpr double WH_FPU_BFP8_TFLOPS_PER_TENSIX = 2.05;
    constexpr double GS_FPU_BFP8_TFLOPS_PER_TENSIX = 0.58;

    double rpeak_tflops = 0.0f;
    double clock = static_cast<double>(tt_npu_clock) / 1000;
    uint32_t num_compute_core = grid_size.x * grid_size.y;
    if (arch == tt::ARCH::WORMHOLE_B0) {
        rpeak_tflops =
            WH_FPU_BFP8_TFLOPS_PER_TENSIX * static_cast<double>(num_compute_core) * static_cast<double>(clock);
    } else if (arch == tt::ARCH::GRAYSKULL) {
        rpeak_tflops =
            GS_FPU_BFP8_TFLOPS_PER_TENSIX * static_cast<double>(num_compute_core) * static_cast<double>(clock);
    } else if (arch == tt::ARCH::BLACKHOLE) {
        rpeak_tflops =
            BH_FPU_BFP8_TFLOPS_PER_TENSIX * static_cast<double>(num_compute_core) * static_cast<double>(clock);
    }

    log_debug(LogTest, "Rpeak {} TFLOPS", rpeak_tflops);
    return rpeak_tflops;
}

std::tuple<uint32_t, uint32_t, uint32_t> get_aligned_input_tile_num(uint32_t M, uint32_t N, uint32_t K) {
    auto align_to_tile = [](uint32_t value) -> uint32_t {
        return ((value + (constants::TILE_WIDTH - 1)) / constants::TILE_WIDTH) * constants::TILE_WIDTH;
    };

    TT_FATAL(M != 0 && N != 0 && K != 0, "Matmul input size should not be zero");

    uint32_t M_aligned = align_to_tile(M);
    uint32_t N_aligned = align_to_tile(N);
    uint32_t K_aligned = align_to_tile(K);

    if (M % constants::TILE_WIDTH || N % constants::TILE_WIDTH || K % constants::TILE_WIDTH) {
        log_info(LogTest, "M, N, K = {}, {}, {} are aligned to {}, {}, {}", M, N, K, M_aligned, N_aligned, K_aligned);
    }

    uint32_t Mt = M_aligned / constants::TILE_WIDTH;
    uint32_t Nt = N_aligned / constants::TILE_WIDTH;
    uint32_t Kt = K_aligned / constants::TILE_WIDTH;
    return {Mt, Nt, Kt};
}

uint32_t get_in0_block_w(
    uint32_t per_core_Mt,
    uint32_t per_core_Nt,
    uint32_t Kt,
    uint32_t single_tile_size,
    uint32_t l1_size,
    uint32_t l1_unreserved_base) {
    std::vector<uint32_t> in0_block_w_choices = {4, 2, 1};
    uint32_t num_buffer = 2;  // double buffering
    uint32_t in0_block_w = 0;
    uint32_t base_addr = l1_unreserved_base;
    for (auto choice : in0_block_w_choices) {
        if (Kt % choice != 0) {
            continue;
        }

        uint32_t in0_cb_size = per_core_Mt * choice * num_buffer * single_tile_size;
        uint32_t in1_cb_size = per_core_Nt * choice * num_buffer * single_tile_size;
        uint32_t in2_cb_size = single_tile_size;
        uint32_t intermediate_cb_size = per_core_Mt * per_core_Nt * single_tile_size;

        uint32_t total_cb_size = in0_cb_size + in1_cb_size + in2_cb_size + intermediate_cb_size;

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

std::tuple<MathFidelity, bool> get_compute_params(tt::ARCH arch) {
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    bool fp32_dest_acc_en = false;
    if (arch == tt::ARCH::WORMHOLE_B0 or arch == tt::ARCH::BLACKHOLE) {
        math_fidelity = MathFidelity::HiFi2;
        // TODO: apply packer_l1_acc
        // TODO: need to consider whether to set these variablias as arguments
        fp32_dest_acc_en = false;
    } else if (arch == tt::ARCH::GRAYSKULL) {
        math_fidelity = MathFidelity::HiFi4;
        fp32_dest_acc_en = false;
    }
    return {math_fidelity, fp32_dest_acc_en};
}

std::tuple<uint32_t, uint32_t> get_out_subblock_params(
    uint32_t per_core_Mt, uint32_t per_core_Nt, uint32_t choice = 0) {
    constexpr std::array<std::tuple<uint32_t, uint32_t>, 20> SUBBLOCK_HW_CHOICES = {{
        {4, 2}, {2, 4}, {8, 1}, {1, 8}, {7, 1}, {1, 7}, {3, 2}, {2, 3}, {6, 1}, {1, 6},
        {5, 1}, {1, 5}, {2, 2}, {4, 1}, {1, 4}, {3, 1}, {1, 3}, {2, 1}, {1, 2}, {1, 1},
    }};

    uint32_t index = 0;
    for (const auto& subblock_hw : SUBBLOCK_HW_CHOICES) {
        auto subblock_h = std::get<0>(subblock_hw);
        auto subblock_w = std::get<1>(subblock_hw);
        if (per_core_Mt % subblock_h == 0 and per_core_Nt % subblock_w == 0) {
            if (index >= choice) {
                return {subblock_h, subblock_w};
            }
            index++;
        }
    }

    return {1, 1};
}

std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> get_all_buffers_addresses(
    uint32_t per_core_Mt,
    uint32_t per_core_Nt,
    uint32_t in0_block_w,
    uint32_t single_tile_size,
    uint32_t l1_unreserved_base) {
    uint32_t num_buffer = 2;  // double buffering
    uint32_t in0_cb_addr = l1_unreserved_base;
    uint32_t in0_cb_size = per_core_Mt * in0_block_w * num_buffer * single_tile_size;
    uint32_t in1_cb_addr = in0_cb_addr + in0_cb_size;
    uint32_t in1_cb_size = per_core_Nt * in0_block_w * num_buffer * single_tile_size;
    uint32_t in2_cb_addr = in1_cb_addr + in1_cb_size;
    uint32_t in2_cb_size = single_tile_size;
    uint32_t out_cb_addr = in2_cb_addr + in2_cb_size;
    uint32_t out_cb_size = per_core_Mt * per_core_Nt * single_tile_size;

    uint32_t per_core_in0_tiles = per_core_Mt * in0_block_w;
    uint32_t per_core_in1_tiles = per_core_Nt * in0_block_w;
    uint32_t in0_addr = out_cb_addr + out_cb_size;
    uint32_t in1_addr = in0_addr + (per_core_in0_tiles * single_tile_size);
    uint32_t out_addr = in1_addr + (per_core_in1_tiles * single_tile_size);

    return {in0_cb_addr, in1_cb_addr, in2_cb_addr, out_cb_addr, in0_addr, in1_addr, out_addr};
}

tt_metal::Program create_program_single_core(
    tt_metal::distributed::MeshDevice* /*device*/,
    tt::DataFormat cb_data_format,
    MathFidelity math_fidelity,
    bool fp32_dest_acc_en,
    uint32_t single_tile_size,
    CoreCoord core_range,
    uint32_t Mt,
    uint32_t Nt,
    uint32_t Kt,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& in0_cb_addr,
    const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& in1_cb_addr,
    const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& out_cb_addr,
    bool matmul_block,
    bool packer_l1,
    uint32_t num_blocks,
    uint32_t interm_cb_dtype) {
    tt_metal::Program program{};

    log_debug(tt::LogTest, "cb_data_format: {} ", cb_data_format);
    log_debug(tt::LogTest, "math_fidelity: {} ", math_fidelity);
    log_debug(tt::LogTest, "single_tile_size: {} ", single_tile_size);
    log_debug(tt::LogTest, "fp32_dest_acc_en: {} ", fp32_dest_acc_en);

    uint32_t num_buffer = 1;  // No double buffer
    uint32_t in0_block_tiles = Mt * Kt;
    uint32_t in0_CB_tiles = in0_block_tiles * num_buffer;
    uint32_t in1_block_tiles = Nt * Kt;
    uint32_t in1_CB_tiles = in1_block_tiles * num_buffer;
    uint32_t out_block_tiles = Mt * Nt;
    uint32_t out_CB_tiles = out_block_tiles;  // No double buffer
    uint32_t out_CB_size = out_CB_tiles * single_tile_size;

    uint32_t in0_num_subblocks = (Mt / out_subblock_h);
    uint32_t in0_block_num_tiles = out_subblock_h * Kt * in0_num_subblocks;
    uint32_t in0_subblock_num_tiles = out_subblock_h * Kt;

    uint32_t in1_num_subblocks = (Nt / out_subblock_w);
    uint32_t in1_block_num_tiles = out_subblock_w * Kt * in1_num_subblocks;
    uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;

    uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;

    vector<uint32_t> compute_kernel_args = {
        Kt,                      // in0_block_w
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
        1,                       // batch
        Mt * Nt,
        0};

    vector<uint32_t> reader_kernel_args = {
        Mt * Kt,
        num_blocks,
    };
    vector<uint32_t> writer_kernel_args = {
        Nt * Kt,
        num_blocks,
    };

    log_debug(tt::LogTest, "in0_cb_addr: {}", (*in0_cb_addr).address());
    log_debug(tt::LogTest, "in1_cb_addr: {}", (*in1_cb_addr).address());
    log_debug(tt::LogTest, "out_cb_addr: {}", (*out_cb_addr).address());
    log_debug(tt::LogTest, "cb_data_format: {}", cb_data_format);
    log_debug(tt::LogTest, "single_tile_size: {}", single_tile_size);
    if (matmul_block) {
        log_debug(tt::LogTest, "matmul_block");
    } else {
        log_debug(tt::LogTest, "matmul_tiles");
    }
    if (packer_l1) {
        log_debug(tt::LogTest, "packer_l1");
    } else {
        log_debug(tt::LogTest, "no packer_l1");
    }

    CoreRange all_cores(
        {(std::size_t)0, (std::size_t)0}, {(std::size_t)core_range.x - 1, (std::size_t)core_range.y - 1});

    // Create circular buffers
    uint32_t src0_cb_index = tt::CBIndex::c_0;
    tt_metal::CircularBufferConfig cb_src0_config =
        tt_metal::CircularBufferConfig(in0_CB_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, single_tile_size)
            .set_globally_allocated_address(*in0_cb_addr->get_backing_buffer());
    tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    uint32_t src1_cb_index = tt::CBIndex::c_1;
    tt_metal::CircularBufferConfig cb_src1_config =
        tt_metal::CircularBufferConfig(in1_CB_tiles * single_tile_size, {{src1_cb_index, cb_data_format}})
            .set_page_size(src1_cb_index, single_tile_size)
            .set_globally_allocated_address(*in1_cb_addr->get_backing_buffer());
    tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);

    uint32_t out_cb_index = tt::CBIndex::c_16;
    uint32_t interm0_cb_index = tt::CBIndex::c_24;

    if (fp32_dest_acc_en) {
        if (interm_cb_dtype == 1) {
            tt_metal::CircularBufferConfig cb_interm_config =
                tt_metal::CircularBufferConfig(out_CB_tiles * 4096, {{interm0_cb_index, tt::DataFormat::Float32}})
                    .set_page_size(interm0_cb_index, 4096);
            tt_metal::CreateCircularBuffer(program, all_cores, cb_interm_config);
        } else {
            tt_metal::CircularBufferConfig cb_interm_config =
                tt_metal::CircularBufferConfig(out_CB_tiles * 2048, {{interm0_cb_index, tt::DataFormat::Float16_b}})
                    .set_page_size(interm0_cb_index, 2048);
            tt_metal::CreateCircularBuffer(program, all_cores, cb_interm_config);
        }

        tt_metal::CircularBufferConfig cb_out_config =
            tt_metal::CircularBufferConfig(out_CB_size, {{out_cb_index, cb_data_format}})
                .set_page_size(out_cb_index, single_tile_size)
                .set_globally_allocated_address(*out_cb_addr->get_backing_buffer());
        tt_metal::CreateCircularBuffer(program, all_cores, cb_out_config);
    } else if (packer_l1 and cb_data_format == tt::DataFormat::Bfp8_b) {
        tt_metal::CircularBufferConfig cb_interm_config =
            tt_metal::CircularBufferConfig(out_CB_tiles * 2048, {{interm0_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(interm0_cb_index, 2048);
        tt_metal::CreateCircularBuffer(program, all_cores, cb_interm_config);

        tt_metal::CircularBufferConfig cb_out_config =
            tt_metal::CircularBufferConfig(out_CB_size, {{out_cb_index, cb_data_format}})
                .set_page_size(out_cb_index, single_tile_size)
                .set_globally_allocated_address(*out_cb_addr->get_backing_buffer());
        tt_metal::CreateCircularBuffer(program, all_cores, cb_out_config);
    } else {
        std::map<uint8_t, tt::DataFormat> partials_and_out_data_format_spec = {
            {out_cb_index, cb_data_format}, {interm0_cb_index, cb_data_format}};
        tt_metal::CircularBufferConfig cb_out_config =
            tt_metal::CircularBufferConfig(out_CB_size, partials_and_out_data_format_spec)
                .set_page_size(interm0_cb_index, single_tile_size)
                .set_page_size(out_cb_index, single_tile_size)
                .set_globally_allocated_address(*out_cb_addr->get_backing_buffer());
        tt_metal::CreateCircularBuffer(program, CoreRangeSet({all_cores}), cb_out_config);
    }

    log_debug(tt::LogTest, "in0_CB_size: {}", in0_CB_tiles * single_tile_size);
    log_debug(tt::LogTest, "in1_CB_size: {}", in1_CB_tiles * single_tile_size);
    log_debug(tt::LogTest, "interm_CB_size: {}", out_CB_tiles * 4096);
    log_debug(tt::LogTest, "out_CB_size: {}", out_CB_size);
    log_debug(
        tt::LogTest,
        "total_CB_size: {}",
        in0_CB_tiles * single_tile_size + in1_CB_tiles * single_tile_size + out_CB_tiles * 4096 + out_CB_size);

    // Create reader and writer kernels per core
    tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/1_compute_mm/kernels/"
        "in0_reader_bmm_single_core.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = reader_kernel_args});

    tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/1_compute_mm/kernels/"
        "in1_reader_writer_bmm_single_core.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_1_default,
            .compile_args = writer_kernel_args});

    // Create compute kernel
    // Gelu currently has better accuracy when run in approx mode
    std::map<std::string, std::string> mm_kernel_defines;
    if (packer_l1) {
        mm_kernel_defines["PACKER_L1_ACC"] = "1";
    }
    if (fp32_dest_acc_en) {
        mm_kernel_defines["FP32_DEST_ACC_EN"] = "1";
    }
    bool math_approx_mode = false;
    tt_metal::CreateKernel(
        program,
        matmul_block ? "tests/tt_metal/tt_metal/perf_microbenchmark/1_compute_mm/kernels/"
                       "bmm_large_block_zm_fused_bias_activation_copy.cpp"
                     : "tests/tt_metal/tt_metal/perf_microbenchmark/1_compute_mm/kernels/"
                       "bmm_large_block_zm_fused_bias_activation.cpp",
        all_cores,
        tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_kernel_args,
            .defines = mm_kernel_defines});

    return program;
}

tt_metal::Program create_program(
    tt_metal::distributed::MeshDevice* device,
    tt::DataFormat cb_data_format,
    MathFidelity math_fidelity,
    bool fp32_dest_acc_en,
    uint32_t single_tile_size,
    CoreCoord core_range,
    uint32_t Mt,
    uint32_t Nt,
    uint32_t Kt,
    uint32_t in0_block_w,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t per_core_Mt,
    uint32_t per_core_Nt,
    uint32_t /*in0_cb_addr*/,
    uint32_t /*in1_cb_addr*/,
    uint32_t in2_cb_addr,
    uint32_t /*out_cb_addr*/,
    uint32_t in0_addr,
    uint32_t in1_addr,
    uint32_t out_addr,
    bool matmul_block,
    bool packer_l1) {
    tt_metal::Program program{};

    uint32_t num_buffer = 2;  // double buffer
    uint32_t in0_block_tiles = per_core_Mt * in0_block_w;
    uint32_t in0_CB_tiles = in0_block_tiles * num_buffer;
    uint32_t in1_block_tiles = per_core_Nt * in0_block_w;
    uint32_t in1_CB_tiles = in1_block_tiles * num_buffer;
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
        1,                       // batch
        per_core_Mt * per_core_Nt};

    CoreRange all_cores(
        {(std::size_t)0, (std::size_t)0}, {(std::size_t)core_range.x - 1, (std::size_t)core_range.y - 1});

    // Create circular buffers
    uint32_t src0_cb_index = tt::CBIndex::c_0;
    tt_metal::CircularBufferConfig cb_src0_config =
        tt_metal::CircularBufferConfig(in0_CB_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    uint32_t src1_cb_index = tt::CBIndex::c_1;
    tt_metal::CircularBufferConfig cb_src1_config =
        tt_metal::CircularBufferConfig(in1_CB_tiles * single_tile_size, {{src1_cb_index, cb_data_format}})
            .set_page_size(src1_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);

    // Dummy cb to store one tile of zeros for padding
    uint32_t in2_CB_tiles = 1;  // No double buffer
    // CB for padding; only need these in the senders
    // NOTE: For first core, initialize cb to the larger tile size to prevent
    // accidentally writing 0 to L1 space during cb init in the kernels
    uint32_t src2_cb_index = tt::CBIndex::c_2;
    tt_metal::CircularBufferConfig cb_src2_config =
        tt_metal::CircularBufferConfig(in2_CB_tiles * single_tile_size, {{src2_cb_index, cb_data_format}})
            .set_page_size(src2_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, all_cores, cb_src2_config);

    uint32_t out_cb_index = tt::CBIndex::c_16;
    uint32_t interm0_cb_index = tt::CBIndex::c_24;
    std::map<uint8_t, tt::DataFormat> partials_and_out_data_format_spec = {
        {out_cb_index, cb_data_format}, {interm0_cb_index, cb_data_format}};
    tt_metal::CircularBufferConfig cb_out_config =
        tt_metal::CircularBufferConfig(out_CB_size, partials_and_out_data_format_spec)
            .set_page_size(out_cb_index, single_tile_size)
            .set_page_size(interm0_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, CoreRangeSet({all_cores}), cb_out_config);

    // Create reader and writer kernels per core
    auto mm_in0_reader_kernel_id = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/1_compute_mm/kernels/"
        "in0_reader_bmm_tile_layout.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_0_default});

    std::map<std::string, std::string> mm_in1_reader_writer_defines;
    mm_in1_reader_writer_defines["IN1_IS_IDENTITY"] = "1";
    auto mm_in1_reader_writer_kernel_id = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/1_compute_mm/kernels/"
        "in1_reader_writer_bmm_tile_layout.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_1_default,
            .defines = mm_in1_reader_writer_defines});

    // Create compute kernel
    // Gelu currently has better accuracy when run in approx mode
    std::map<std::string, std::string> mm_kernel_defines;
    if (packer_l1) {
        mm_kernel_defines["PACKER_L1_ACC"] = "1";
    }
    bool math_approx_mode = false;
    tt_metal::CreateKernel(
        program,
        matmul_block ? "tests/tt_metal/tt_metal/perf_microbenchmark/1_compute_mm/kernels/"
                       "bmm_large_block_zm_fused_bias_activation_block.cpp"
                     : "tests/tt_metal/tt_metal/perf_microbenchmark/1_compute_mm/kernels/"
                       "bmm_large_block_zm_fused_bias_activation.cpp",
        all_cores,
        tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_kernel_args,
            .defines = mm_kernel_defines});

    // Parameters for last row, col, or block
    uint32_t last_block_h = Mt % per_core_Mt == 0 ? per_core_Mt : Mt % per_core_Mt;
    uint32_t last_block_w = Nt % per_core_Nt == 0 ? per_core_Nt : Nt % per_core_Nt;
    uint32_t last_block_num_nonzero_subblocks_h = ((last_block_h - 1) / out_subblock_h) + 1;
    uint32_t last_block_num_nonzero_subblocks_w = ((last_block_w - 1) / out_subblock_w) + 1;
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
            std::array<uint32_t, 12> mm_in0_reader_args = {
                (std::uint32_t)in0_addr,     // in0_buffer->address(), // in0_tensor_addr
                (std::uint32_t)0,            // K * per_core_Mt * output_idx_y, //
                                             // in0_tensor_start_tile_id
                (std::uint32_t)1,            // in0_tensor_stride_w
                (std::uint32_t)in0_block_w,  // K, // in0_tensor_stride_h
                (std::uint32_t)in0_block_w,  // in0_tensor_next_block_stride

                (std::uint32_t)in0_block_w,                // in0_block_w
                (std::uint32_t)per_core_Mt,                // in0_block_h
                (std::uint32_t)in0_block_w * per_core_Mt,  // in0_block_num_tiles

                (std::uint32_t)num_blocks,  // num_blocks
                (std::uint32_t)phy_core.x,
                (std::uint32_t)phy_core.y,
            };

            uint32_t in1_tensor_stride_h = per_core_Nt;
            uint32_t out_tensor_stride_h = per_core_Nt;
            if (core_idx_x == core_range.x - 1) {
                in1_tensor_stride_h = last_block_w;
                out_tensor_stride_h = last_block_w;
            }

            std::array<uint32_t, 31> mm_in1_reader_writer_args = {
                (std::uint32_t)in1_addr,                   // in1_buffer->address(), // in1_tensor_addr
                (std::uint32_t)0,                          // per_core_Nt * output_idx_x,
                                                           // //in1_tensor_start_tile_id
                (std::uint32_t)1,                          // in1_tensor_stride_w
                (std::uint32_t)in1_tensor_stride_h,        // in1_tensor_stride_h
                (std::uint32_t)in0_block_w * per_core_Nt,  // in1_tensor_next_block_stride

                (std::uint32_t)per_core_Nt,                // in1_block_w
                (std::uint32_t)in0_block_w,                // in1_block_h
                (std::uint32_t)per_core_Nt * in0_block_w,  // in1_block_num_tiles

                (std::uint32_t)num_blocks,  // num_blocks

                (std::uint32_t)in2_cb_addr,
                (std::uint32_t)phy_core.x,
                (std::uint32_t)phy_core.y,

                (std::uint32_t)out_addr,                              // out_buffer->address(), // out_tensor_addr
                (std::uint32_t)0,                                     // output_idx_x * per_core_Nt + output_idx_y *
                                                                      // per_core_Mt * N, // out_tensor_start_tile_id
                (std::uint32_t)1,                                     // out_tensor_stride_w
                (std::uint32_t)out_tensor_stride_h,                   // out_tensor_stride_h
                (std::uint32_t)out_subblock_w,                        // out_tensor_next_subblock_stride_w
                (std::uint32_t)out_subblock_h * out_tensor_stride_h,  // out_tensor_next_subblock_stride_h

                (std::uint32_t)out_subblock_w,                     // out_subblock_w
                (std::uint32_t)out_subblock_h,                     // out_subblock_h
                (std::uint32_t)(out_subblock_w * out_subblock_h),  // out_subblocks_w * out_subblocks_h
                (std::uint32_t)(per_core_Nt / out_subblock_w),     // out_num_subblocks_w
                (std::uint32_t)(per_core_Mt / out_subblock_h),     // out_num_subblocks_h
            };

            if (core_idx_y == core_range.y - 1) {
                mm_in0_reader_args.back() = last_block_h;  // last_block_h
            } else {
                mm_in0_reader_args.back() = per_core_Mt;
            }

            // padding args (WRITER)
            if (core_idx_x == core_range.x - 1) {
                mm_in1_reader_writer_args[23] = last_block_w;
            } else {
                mm_in1_reader_writer_args[23] = per_core_Nt;
            }

            if (core_idx_x == core_range.x - 1 && core_idx_y == core_range.y - 1) {
                mm_in1_reader_writer_args[24] = last_block_num_nonzero_subblocks_h;
                mm_in1_reader_writer_args[25] = last_subblock_of_last_block_h;
                mm_in1_reader_writer_args[26] = last_block_padded_block_tiles_h_skip;
                mm_in1_reader_writer_args[27] = last_block_num_nonzero_subblocks_w;
                mm_in1_reader_writer_args[28] = last_subblock_of_last_block_w;
                mm_in1_reader_writer_args[29] = last_block_padded_subblock_tiles_addr_skip;
                mm_in1_reader_writer_args[30] = last_block_padded_block_tiles_w_skip;
            } else if (core_idx_y == core_range.y - 1) {
                mm_in1_reader_writer_args[24] = last_block_num_nonzero_subblocks_h;
                mm_in1_reader_writer_args[25] = last_subblock_of_last_block_h;
                mm_in1_reader_writer_args[26] = last_block_padded_block_tiles_h_skip;
                mm_in1_reader_writer_args[27] = per_core_Nt / out_subblock_w;
                mm_in1_reader_writer_args[28] = out_subblock_w;
                mm_in1_reader_writer_args[29] = 0;
                mm_in1_reader_writer_args[30] = 0;
            } else if (core_idx_x == core_range.x - 1) {
                mm_in1_reader_writer_args[24] = per_core_Mt / out_subblock_h;
                mm_in1_reader_writer_args[25] = out_subblock_h;
                mm_in1_reader_writer_args[26] = 0;
                mm_in1_reader_writer_args[27] = last_block_num_nonzero_subblocks_w;
                mm_in1_reader_writer_args[28] = last_subblock_of_last_block_w;
                mm_in1_reader_writer_args[29] = last_block_padded_subblock_tiles_addr_skip;
                mm_in1_reader_writer_args[30] = last_block_padded_block_tiles_w_skip;
            } else {
                mm_in1_reader_writer_args[24] = per_core_Mt / out_subblock_h;
                mm_in1_reader_writer_args[25] = out_subblock_h;
                mm_in1_reader_writer_args[26] = 0;
                mm_in1_reader_writer_args[27] = per_core_Nt / out_subblock_w;
                mm_in1_reader_writer_args[28] = out_subblock_w;
                mm_in1_reader_writer_args[29] = 0;
                mm_in1_reader_writer_args[30] = 0;
            }
            tt_metal::SetRuntimeArgs(program, mm_in0_reader_kernel_id, core, mm_in0_reader_args);
            tt_metal::SetRuntimeArgs(program, mm_in1_reader_writer_kernel_id, core, mm_in1_reader_writer_args);
        }
    }
    return program;
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

template <typename T>
std::vector<T> get_row_slice(std::vector<T> data, int start_row_index, int num_rows, int /*rows*/, int cols) {
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
            result.push_back(data.at((r * cols) + c));
        }
    }
    return result;
}

void prepare_inputs(
    tt_metal::distributed::MeshDevice* device,
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
    uint32_t in2_cb_addr,
    bool /*dtype*/,
    std::vector<std::vector<float>>& in0_bfp8_unpack_slice,
    std::vector<std::vector<float>>& /*in1_bfp8_unpack_slice*/) {
    bool pass = true;
    auto in0_vec = generate_fp32_random(Mt * Kt * constants::TILE_HW);
    std::vector<uint32_t> in2(single_tile_size / sizeof(uint32_t), 0);

    uint32_t num_cores_y = core_range.y;
    uint32_t num_cores_x = core_range.x;

    uint32_t last_block_h = Mt % per_core_Mt == 0 ? per_core_Mt : Mt % per_core_Mt;
    uint32_t last_block_w = Nt % per_core_Nt == 0 ? per_core_Nt : Nt % per_core_Nt;

    for (int r = 0; r < num_cores_y; r++) {
        int num_r = (r == num_cores_y - 1) ? (last_block_h) : (per_core_Mt);

        std::vector<float> in0_slice = get_row_slice(in0_vec, r * per_core_Mt * 32, num_r * 32, Mt * 32, Kt * 32);
        // only use the first block of in0_slice
        auto in0_block_slice = get_col_slice(in0_slice, 0, in0_block_w * 32, num_r * 32, Kt * 32);
        auto in0_block_tilized = tilize_swizzled(in0_block_slice, num_r * 32, in0_block_w * 32);
        std::vector<uint32_t> in0 = pack_as_bfp8_tiles(
            tt::stl::make_const_span(in0_block_tilized), /*row_major_input=*/true, /*is_exp_a=*/false);

        auto unpack_vec = unpack_bfp8_tiles_into_float_vec(in0, true, false);
        auto untilize_vec = untilize_swizzled(unpack_vec, num_r * 32, in0_block_w * 32);
        in0_bfp8_unpack_slice.push_back(untilize_vec);

        for (int c = 0; c < num_cores_x; c++) {
            int num_c = (c == num_cores_x - 1) ? (last_block_w) : (per_core_Nt);

            std::vector<float> in1_block_slice(in0_block_w * num_c * 1024, (float)0);
            int num_ones = std::min(in0_block_w, static_cast<uint32_t>(num_c)) * 32;
            for (int i = 0; i < num_ones; i++) {
                in1_block_slice.at((i * (num_c * 32)) + i) = (float)1;
            }

            auto in1_block_tilized = tilize_swizzled(in1_block_slice, in0_block_w * 32, num_c * 32);
            std::vector<uint32_t> in1 = pack_as_bfp8_tiles(
                tt::stl::make_const_span(in1_block_tilized), /*row_major_input=*/true, /*is_exp_a=*/false);

            // copy in0, in1, in2 to L1
            CoreCoord core = {(std::size_t)c, (std::size_t)r};
            auto* target_device = device->get_devices()[0];
            pass &= tt_metal::detail::WriteToDeviceL1(target_device, core, in0_addr, in0);
            TT_FATAL(pass, "Failed to write in0 to device L1");
            pass &= tt_metal::detail::WriteToDeviceL1(target_device, core, in1_addr, in1);
            TT_FATAL(pass, "Failed to write in1 to device L1");
            pass &= tt_metal::detail::WriteToDeviceL1(target_device, core, in2_cb_addr, in2);
            TT_FATAL(pass, "Failed to write in2 to device L1");
        }
    }
}

float to_float(bfloat16 bfloat16_num) { return static_cast<float>(bfloat16_num); }

bool validation_single_core(
    tt_metal::distributed::MeshDevice* device,
    const tt::deprecated::Tensor<bfloat16>& tensor_in0,
    const tt::deprecated::Tensor<bfloat16>& tensor_in1,
    uint32_t num_blocks,
    uint32_t Mt,
    uint32_t Nt,
    uint32_t Kt,
    const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& out_buffer) {
    bool pass = true;

    std::vector<uint32_t> result;
    tt_metal::distributed::ReadShard(device->mesh_command_queue(), result, out_buffer, {0, 0}, true);

    auto result_bfp16 = unpack_uint32_vec_into_bfloat16_vec(result);
    auto result_flat_layout = convert_layout_tile_nfaces_to_tile_swizzled(tt::stl::make_const_span(result_bfp16));
    auto result_untilized = untilize_swizzled(result_flat_layout, Mt * 32, Nt * 32);

    std::vector<float> golden_vec(Mt * Nt * 32 * 32, 0);  // Initialize with zeros
    const auto& values0 = tensor_in0.get_values();
    const auto& values1 = tensor_in1.get_values();

    for (size_t i = 0; i < Mt * 32; ++i) {
        for (size_t j = 0; j < Nt * 32; ++j) {
            float sum = 0;
            for (size_t k = 0; k < Kt * 32; ++k) {
                sum += to_float(values0[(i * Kt * 32) + k]) * to_float(values1[(k * Nt * 32) + j]);
            }
            golden_vec[(i * Nt * 32) + j] = sum * num_blocks;
        }
    }

    std::vector<float> result_vec;
    result_vec.reserve(result_untilized.size());
    for (auto val : result_untilized) {
        result_vec.push_back(to_float(static_cast<bfloat16>(val)));
    }

    // for (int i=0; i<result_vec.size(); ++i) {
    //     std::cout << "index: " << i << " " << golden_vec[i] << " " << result_vec[i] << std::endl;
    // }

    pass &= (golden_vec == result_vec);
    if (!pass) {
        log_error(LogTest, "validation single core failed");
    }
    return pass;
}

bool validation_single_core_fp8(
    tt_metal::distributed::MeshDevice* device,
    const tt::deprecated::Tensor<float>& tensor_in0,
    const tt::deprecated::Tensor<float>& tensor_in1,
    uint32_t num_blocks,
    uint32_t Mt,
    uint32_t Nt,
    uint32_t Kt,
    const std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>& out_buffer) {
    bool pass = true;

    std::vector<uint32_t> result;
    tt_metal::distributed::ReadShard(device->mesh_command_queue(), result, out_buffer, {0, 0}, true);

    auto result_bfp8 = unpack_bfp8_tiles_into_float_vec(result, true, false);
    auto result_untilized = untilize_swizzled(result_bfp8, Mt * 32, Nt * 32);

    std::vector<float> golden_vec(Mt * Nt * 32 * 32, 0);  // Initialize with zeros
    const auto& values0 = tensor_in0.get_values();
    const auto& values1 = tensor_in1.get_values();

    for (size_t i = 0; i < Mt * 32; ++i) {
        for (size_t j = 0; j < Nt * 32; ++j) {
            float sum = 0;
            for (size_t k = 0; k < Kt * 32; ++k) {
                sum += values0[(i * Kt * 32) + k] * values1[(k * Nt * 32) + j];
            }
            golden_vec[(i * Nt * 32) + j] = sum * num_blocks;
        }
    }

    // for (int i=0; i<tensor_in0.get_values().size(); ++i) {
    //     std::cout << golden_vec[i] << " " << result_untilized[i] << std::endl;
    // }

    pass &= (result_untilized == golden_vec);
    if (!pass) {
        log_error(LogTest, "validation single core failed");
    }
    return pass;
}

bool validation(
    tt_metal::distributed::MeshDevice* device,
    CoreCoord core_range,
    uint32_t Mt,
    uint32_t Nt,
    uint32_t Kt,
    uint32_t per_core_Mt,
    uint32_t per_core_Nt,
    uint32_t in0_block_w,
    uint32_t out_addr,
    uint32_t single_tile_size,
    bool /*fp32_dest_acc_en*/,
    std::vector<std::vector<float>>& in0_bfp8_unpack_slice,
    std::vector<std::vector<float>>& /*in1_bfp8_unpack_slice*/) {
    auto zero_vector = [](uint32_t r, uint32_t c) {
        std::vector<float> vec(r * c, (float)0);
        return vec;
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
            auto* target_device = device->get_devices()[0];
            tt_metal::detail::ReadFromDeviceL1(
                target_device, core, out_addr, num_r * num_c * single_tile_size, result_vec);
            auto result_flat_layout = unpack_bfp8_tiles_into_float_vec(result_vec, true, false);
            auto result_untilized = untilize_swizzled(result_flat_layout, num_r * 32, num_c * 32);

            uint32_t num_patterns = ((num_c - 1) / in0_block_w) + 1;
            uint32_t last_remain_c = num_c % in0_block_w == 0 ? in0_block_w : num_c % in0_block_w;

            for (int32_t i = 0; i < num_patterns; ++i) {
                auto pattern_w = (i == num_patterns - 1) ? (last_remain_c) : (in0_block_w);
                auto result_slice =
                    get_col_slice(result_untilized, i * in0_block_w * 32, pattern_w * 32, num_r * 32, num_c * 32);
                auto in0_block_slice =
                    (i < num_blocks)
                        ? (get_col_slice(in0_bfp8_unpack_slice[r], 0, pattern_w * 32, num_r * 32, in0_block_w * 32))
                        : (zero_vector(num_r * 32, pattern_w * 32));

                if (result_slice.size() != in0_block_slice.size()) {
                    pass = false;
                }
                for (int j = 0; j < result_slice.size(); ++j) {
                    float a = result_slice.at(i);
                    float b = in0_block_slice.at(i);
                    if (a != b) {
                        diff_count++;
                        pass = false;
                    }
                }
            }
        }
    }

    uint32_t total_count = Mt * Nt * constants::TILE_HW;
    if (!pass) {
        log_error(LogTest, "validation failed : {}/{} elements are not matched with golden", diff_count, total_count);
    }
    return pass;
}

std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> create_and_transfer_data_sharded_cb(
    tt_metal::distributed::MeshDevice* device, const vector<uint32_t>& activations, uint32_t Mt, uint32_t Nt) {
    uint32_t size_bytes = Mt * tt::constants::TILE_HEIGHT * Nt * tt::constants::TILE_WIDTH * 2;
    uint32_t page_size_bytes = tt::constants::TILE_HW * 2;

    ShardSpecBuffer shard_spec = ShardSpecBuffer(
        CoreRangeSet(std::set<CoreRange>({CoreRange(CoreCoord(0, 0))})),
        {Mt * tt::constants::TILE_HEIGHT, Nt * tt::constants::TILE_WIDTH},
        ShardOrientation::ROW_MAJOR,
        {tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH},
        {Mt, Nt});

    log_debug(tt::LogTest, "size_bytes: {}", size_bytes);
    log_debug(tt::LogTest, "page_size_bytes: {}", page_size_bytes);

    // Create MeshBuffer configuration
    auto mesh_buffer_config = tt::tt_metal::distributed::ReplicatedBufferConfig{.size = size_bytes};
    auto device_local_config = tt::tt_metal::distributed::DeviceLocalBufferConfig{
        .page_size = page_size_bytes,
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(shard_spec, TensorMemoryLayout::HEIGHT_SHARDED)};

    auto input_buffer = tt::tt_metal::distributed::MeshBuffer::create(mesh_buffer_config, device_local_config, device);

    // Write data to the mesh buffer
    auto& mesh_cq = device->mesh_command_queue();
    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(mesh_cq, input_buffer, activations, true);

    return input_buffer;
}

std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> create_and_transfer_data_sharded_cb_fp8(
    tt_metal::distributed::MeshDevice* device, const vector<uint32_t>& activations, uint32_t Mt, uint32_t Nt) {
    uint32_t size_bytes = Mt * Nt * 1088;
    uint32_t page_size_bytes = 1088;

    ShardSpecBuffer shard_spec = ShardSpecBuffer(
        CoreRangeSet(std::set<CoreRange>({CoreRange(CoreCoord(0, 0))})),
        {Mt * tt::constants::TILE_HEIGHT, Nt * tt::constants::TILE_WIDTH},
        ShardOrientation::ROW_MAJOR,
        {tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH},
        {Mt, Nt});

    log_debug(tt::LogTest, "size_bytes: {}", size_bytes);
    log_debug(tt::LogTest, "page_size_bytes: {}", page_size_bytes);

    // Create MeshBuffer configuration
    auto mesh_buffer_config = tt::tt_metal::distributed::ReplicatedBufferConfig{.size = size_bytes};
    auto device_local_config = tt::tt_metal::distributed::DeviceLocalBufferConfig{
        .page_size = page_size_bytes,
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(shard_spec, TensorMemoryLayout::HEIGHT_SHARDED)};

    auto input_buffer = tt::tt_metal::distributed::MeshBuffer::create(mesh_buffer_config, device_local_config, device);

    // Write data to the mesh buffer
    auto& mesh_cq = device->mesh_command_queue();
    tt::tt_metal::distributed::EnqueueWriteMeshBuffer(mesh_cq, input_buffer, activations, true);

    return input_buffer;
}
