#include <algorithm>
#include <chrono>
#include <functional>
#include <random>
#include <thread>

#include "common/bfloat16.hpp"
#include "test_tiles.hpp"
// #include "tt_metal/detail/util.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/llrt/tt_debug_print_server.hpp"
#include "tt_metal/test_utils/deprecated/tensor.hpp"

#define LAUNCH

using namespace tt;

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
      for (auto j = 0; j < 32; j++) {    // tile rows
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

std::vector<bfloat16> get_row_slice(std::vector<bfloat16> data,
                                    int total_row_slices, int row_slice_index,
                                    int rows, int cols) {
  std::vector<bfloat16> result;
  int rows_per_slice = rows / total_row_slices;

  for (int i = rows_per_slice * row_slice_index * cols;
       i < rows_per_slice * (row_slice_index + 1) * cols; i++) {
    result.push_back(data.at(i));
  }
  return result;
}

std::vector<bfloat16> get_col_slice(std::vector<bfloat16> data,
                                    int total_col_slices, int col_slice_index,
                                    int rows, int cols) {
  std::vector<bfloat16> result;
  int cols_per_slice = cols / total_col_slices;

  for (int r = 0; r < rows; r++) {
    for (int c = cols_per_slice * col_slice_index;
         c < cols_per_slice * (col_slice_index + 1); c++) {
      result.push_back(data.at(r * cols + c));
    }
  }
  return result;
}

// Transpose 2D matrix of tiles so that its column major of tiles instead of row
// major. this is usually used for activation so that blocks data is contiguous
// in memory until we have a more generalized read kernel that can read tiles
// from different location in memory to make up a block in the activations CB
std::vector<std::uint32_t> transpose_tiles(std::vector<std::uint32_t> data,
                                           int row_tiles, int col_tiles,
                                           int in0_block_w) {
  std::vector<std::uint32_t> result;
  int tile_size = 512;
  for (int c = 0; c < col_tiles; c += in0_block_w) {
    for (int r = 0; r < row_tiles; r++) {
      for (int k = 0; k < in0_block_w; k++) {
        int offset = tile_size * col_tiles * r + c * tile_size + k * tile_size;
        for (int i = 0; i < tile_size; i++) {
          result.push_back(data.at(offset + i));
        }
      }
    }
  }
  return result;
}

std::vector<bfloat16> select_columns(std::vector<bfloat16> data, int M, int K,
                                     int N) {
  if (N == K) {
    return data;
  }
  std::vector<bfloat16> result;
  if (N > K) {
    for (int i = 0; i < M * 32; i++) {
      for (int j = 0; j < K * 32; j++) {
        int offset = i * K * 32;
        result.push_back(data.at(offset + j));
      }
      for (int j = 0; j < (N - K) * 32; j++) {
        result.push_back((float)0);
      }
    }
  } else {
    for (int i = 0; i < M * 32; i++) {
      for (int j = 0; j < N * 32; j++) {
        int offset = i * K * 32;
        result.push_back(data.at(offset + j));
      }
    }
  }
  return result;
}

template <typename T>
std::vector<T> slice(std::vector<T> const &v, int m, int n) {
  auto first = v.cbegin() + m;
  auto last = v.cbegin() + n + 1;

  std::vector<T> vec(first, last);
  return vec;
}

void print_vec(std::vector<bfloat16> data, int rows, int cols, string name) {
  std::cout << name << ": " << std::endl;
  int index = 0;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      std::cout << data.at(index).to_float() << " ";
      index++;
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

int main(int argc, char **argv) {
  bool pass = true;
  try {
    ////////////////////////////////////////////////////////////////////////////
    //                      Initial Runtime Args Parse
    ////////////////////////////////////////////////////////////////////////////
    std::vector<std::string> input_args(argv, argv + argc);
    string profile_device_str = "";
    string dprint_str = "";
    string print_tensor_str = "";
    string debug_str = "";
    string cb_str = "";
    string Mt_str = "";
    string Nt_str = "";
    string Kt_str = "";
    string r_str = "";
    string c_str = "";
    string validation_str = "";
    string arch_name = "";
    try {
      std::tie(arch_name, input_args) =
          test_args::get_command_option_and_remaining_args(input_args, "--arch",
                                                           "grayskull");
      std::tie(profile_device_str, input_args) =
          test_args::get_command_option_and_remaining_args(input_args,
                                                           "--profile", "0");
      std::tie(debug_str, input_args) =
          test_args::get_command_option_and_remaining_args(input_args,
                                                           "--debug", "0");
      std::tie(dprint_str, input_args) =
          test_args::get_command_option_and_remaining_args(input_args,
                                                           "--dprint", "0");
      std::tie(print_tensor_str, input_args) =
          test_args::get_command_option_and_remaining_args(
              input_args, "--print_tensor", "0");
      std::tie(Mt_str, input_args) =
          test_args::get_command_option_and_remaining_args(input_args, "--mt",
                                                           "1");
      std::tie(Nt_str, input_args) =
          test_args::get_command_option_and_remaining_args(input_args, "--nt",
                                                           "1");
      std::tie(Kt_str, input_args) =
          test_args::get_command_option_and_remaining_args(input_args, "--kt",
                                                           "1");
      std::tie(r_str, input_args) =
          test_args::get_command_option_and_remaining_args(input_args, "--r",
                                                           "1");
      std::tie(c_str, input_args) =
          test_args::get_command_option_and_remaining_args(input_args, "--c",
                                                           "1");
      std::tie(validation_str, input_args) =
          test_args::get_command_option_and_remaining_args(input_args,
                                                           "--validation", "1");
    } catch (const std::exception &e) {
      log_fatal(tt::LogTest, "Command line arguments found exception",
                e.what());
    }

    const tt::ARCH arch = tt::get_arch_from_string(arch_name);
    bool profile_device = stoi(profile_device_str);
    bool dprint = stoi(dprint_str);
    bool print_tensor = stoi(print_tensor_str);
    bool validation = stoi(validation_str);
    bool debug = stoi(debug_str);

    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    int pci_express_slot = 0;
    tt_metal::Device *device = tt_metal::CreateDevice(arch, pci_express_slot);

    // TODO(jaehoon): check enable_fw_profile_hack removal
    // if (profile_device) {
    //  extern bool enable_fw_profile_hack;
    //  enable_fw_profile_hack = true;
    //}
    pass &= tt_metal::InitializeDevice(device);

    if (dprint) {
      tt_start_debug_print_server(device->cluster());
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Inputs Setup
    ////////////////////////////////////////////////////////////////////////////

    int num_cores_r = stoi(r_str);
    int num_cores_c = stoi(c_str);
    uint32_t Mt = stoi(Mt_str);
    uint32_t Nt = stoi(Nt_str);
    uint32_t Kt = stoi(Kt_str);

    if (debug) {
      log_info(LogTest, "row {} x col {} = {} cores", num_cores_r, num_cores_c,
               num_cores_r * num_cores_c);
    }

    log_info(LogTest, "M = {}, N = {}, K = {}", Mt, Nt, Kt);
    log_info(LogTest, "Activation = {}x{}", Mt * 32, Kt * 32);
    log_info(LogTest, "Weights = {}x{}", Kt * 32, Nt * 32);

    if (Mt % num_cores_r != 0) {
      log_fatal(LogTest, "Mt {} must be a multiple of num_cores_r {}", Mt,
                num_cores_r);
      TT_ASSERT(false);
    }

    if (Nt % num_cores_c != 0) {
      log_fatal(LogTest, "Nt {} must be a multiple of num_cores_c {}", Nt,
                num_cores_c);
      TT_ASSERT(false);
    }

    tt::DataFormat data_format = tt::DataFormat::Float16_b;
    uint32_t single_tile_size = 2 * 1024;
    uint32_t per_core_Mt = Mt / num_cores_r;
    uint32_t per_core_Nt = Nt / num_cores_c;
    uint32_t per_core_activations_tiles = per_core_Mt * Kt;
    uint32_t per_core_weights_tiles = per_core_Nt * Kt;
    uint32_t per_core_output_tiles = per_core_Mt * per_core_Nt;

    uint32_t activations_addr = 120 * 1024;
    uint32_t weights_addr =
        activations_addr + (per_core_activations_tiles * single_tile_size);
    uint32_t output_addr =
        weights_addr + (per_core_weights_tiles * single_tile_size);

    if (debug) {
      log_info(LogTest, "per core M = {}, N = {}, K = {}", per_core_Mt,
               per_core_Nt, Kt);
      log_info(LogTest, "per core activations tiles = {}, weights tiles = {}",
               per_core_activations_tiles, per_core_weights_tiles);
    }

    log_info(LogTest, "activations_addr ({} x 1024) {} tiles",
             activations_addr / 1024, per_core_activations_tiles);
    log_info(LogTest, "weights_addr ({} x 1024) {} tiles", weights_addr / 1024,
             per_core_weights_tiles);
    log_info(LogTest, "output_addr ({} x 1024) {} tiles", output_addr / 1024,
             per_core_output_tiles);

    if (output_addr + (per_core_output_tiles * single_tile_size) >
        1024 * 1024) {
      log_error(LogTest, "inputs and output CBs don't fit in L1");
      TT_ASSERT(false);
    }

    SHAPE shape = {1, 1, Mt * 32, Kt * 32};
    tt::deprecated::Tensor<bfloat16> tensor =
        tt::deprecated::initialize_tensor<bfloat16>(
            shape, tt::deprecated::Initialize::RANDOM, 100,
            std::chrono::system_clock::now().time_since_epoch().count());
    auto identity = create_identity_matrix(
        Kt * 32, Nt * 32, std::min(Kt, Nt) * 32);  // bflaot16 identity

    if (print_tensor) {
      print_vec(tensor.get_values(), 2, Kt * 32,
                std::string("Activation first row"));
      print_vec(identity, 2, Nt * 32, std::string("Weights first row"));
    }

    log_info(LogTest, "Slicing input tensors and copying them to L1");
    for (int r = 0; r < num_cores_r; r++) {
      std::vector<bfloat16> activation_slice =
          get_row_slice(tensor.get_values(), num_cores_r, r, Mt * 32, Kt * 32);
      for (int c = 0; c < num_cores_c; c++) {
        std::vector<bfloat16> weights_slice =
            get_col_slice(identity, num_cores_c, c, Kt * 32, Nt * 32);

        CoreCoord core = {(std::size_t)c, (std::size_t)r};
        auto activations_tilized =
            tilize(activation_slice, per_core_Mt * 32, Kt * 32);
        auto activations_tile_layout =
            convert_to_tile_layout(activations_tilized);
        auto activations =
            pack_bfloat16_vec_into_uint32_vec(activations_tile_layout);
        pass &= tt_metal::detail::WriteToDeviceL1(
            device, core, activations_addr, activations);
        TT_ASSERT(pass);

        auto identity_tilized =
            tilize(weights_slice, Kt * 32, per_core_Nt * 32);
        auto weights_tile_layout = convert_to_tile_layout(identity_tilized);
        auto weights = pack_bfloat16_vec_into_uint32_vec(weights_tile_layout);
        auto weights_tile_transposed =
            transpose_tiles(weights, Kt, per_core_Nt, 1);
        pass &= tt_metal::detail::WriteToDeviceL1(device, core, weights_addr,
                                                  weights_tile_transposed);
        TT_ASSERT(pass);
      }
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program program = tt_metal::Program();
    CoreCoord start_core = {0, 0};
    CoreCoord end_core = {(std::size_t)num_cores_c - 1,
                          (std::size_t)num_cores_r - 1};
    const CoreRange all_cores{.start = start_core, .end = end_core};

    log_info(LogTest, "start_core {},{}", start_core.x, start_core.y);
    log_info(LogTest, "end_core {},{}", end_core.x, end_core.y);

    // CB creation
    uint32_t cb_activations_index = 0;
    uint32_t cb_activations_tiles = per_core_activations_tiles;
    auto cb_activations = tt_metal::CreateCircularBuffers(
        program, cb_activations_index, all_cores, cb_activations_tiles,
        cb_activations_tiles * single_tile_size, data_format, activations_addr);

    uint32_t cb_weights_index = 1;
    uint32_t cb_weights_tiles = per_core_weights_tiles;
    auto cb_weights = tt_metal::CreateCircularBuffers(
        program, cb_weights_index, all_cores, cb_weights_tiles,
        cb_weights_tiles * single_tile_size, data_format, weights_addr);

    uint32_t cb_output_index = 16;
    uint32_t cb_output_tiles = per_core_output_tiles;
    auto cb_output = tt_metal::CreateCircularBuffers(
        program, cb_output_index, all_cores, cb_output_tiles,
        cb_output_tiles * single_tile_size, data_format, output_addr);

    // compute kernel setup
    vector<uint32_t> compute_kernel_args = {uint(per_core_Mt), uint(Kt),
                                            uint(per_core_Nt)};

    auto mm_kernel = tt_metal::CreateComputeKernel(
        program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/matmul/kernels/"
        "compute_local_l1.cpp",
        all_cores,
        tt_metal::ComputeConfig{.math_fidelity = MathFidelity::HiFi4,
                                .fp32_dest_acc_en = false,
                                .math_approx_mode = false,
                                .compile_args = compute_kernel_args});

    // TODO(jaehoon): check interface change
    // pass &= tt_metal::CompileProgram(device, program, profile_device);
    pass &= tt_metal::CompileProgram(device, program);

#ifndef LAUNCH
    CommandQueue cq(device);
#endif

    tt::log_assert(program.get_worker_core_range_set().ranges().size() >= 1,
                   "Invalid core range set");
    pass &= tt_metal::ConfigureDeviceWithProgram(device, program);

#ifdef LAUNCH
    log_info(LogTest, "Running {} core test with LaunchKernels",
             num_cores_r * num_cores_c);
    pass &= tt_metal::LaunchKernels(device, program);
#else
    log_info(LogTest, "Running {} core test with EnqueueProgram",
             num_cores_r * num_cores_c);
    EnqueueProgram(cq, program, false);
    Finish(cq);
#endif

    if (profile_device) {
      tt_metal::detail::DumpDeviceProfileResults(device, program);
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Validation & Teardown
    ////////////////////////////////////////////////////////////////////////////

    auto golden = select_columns(tensor.get_values(), Mt, Kt, Nt);
    if (validation) {
      log_info(LogTest, "Validation");
      for (int r = 0; r < num_cores_r; ++r) {
        auto golden_row =
            get_row_slice(golden, num_cores_r, r, Mt * 32, Nt * 32);
        for (int c = 0; c < num_cores_c; ++c) {
          auto per_core_golden = get_col_slice(golden_row, num_cores_c, c,
                                               per_core_Mt * 32, Nt * 32);

          CoreCoord core = {(size_t)c, (size_t)r};

          std::vector<uint32_t> result_vec;
          tt_metal::detail::ReadFromDeviceL1(device, core, output_addr,
                                             cb_output_tiles * single_tile_size,
                                             result_vec);
          auto result_bfp16 = unpack_uint32_vec_into_bfloat16_vec(result_vec);
          auto result_flat_layout = convert_to_flat_layout(result_bfp16);
          auto result_untilized =
              untilize(result_flat_layout, per_core_Mt * 32, per_core_Nt * 32);

          if (print_tensor) {
            print_vec(result_untilized, 2, Nt * 32,
                      std::string("result_untilized" + std::to_string(r) + " " +
                                  std::to_string(c)));
          }

          if (!(per_core_golden == result_untilized)) {
            log_error(LogTest, "{}/{} - comparision failed ", r, c);
            pass = false;
          } else {
            if (debug) {
              log_info(LogTest, "{}/{} - comparision passed", r, c);
            }
          }
        }
      }
    }
    pass &= tt_metal::CloseDevice(device);

  } catch (const std::exception &e) {
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
