// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <bit>
#include <chrono>
#include <fmt/base.h>
#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <random>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "impl/program/program_impl.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <memory>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/device.hpp>
#include "llk_device_fixture.hpp"
#include <tt-metalium/distributed.hpp>
#include "hostdevcommon/kernel_structs.h"
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/df/float32.hpp"
#include "tt_metal/test_utils/int8.hpp"
#include "tt_metal/test_utils/mx_utils.hpp"
#include "tt_metal/test_utils/packing.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include <tt-metalium/tile.hpp>
#include <umd/device/types/arch.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/int8.hpp>
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"

namespace tt::tt_metal {

using std::map;
using std::vector;
using namespace tt;
using namespace tt::test_utils;
using namespace tt::test_utils::df;

namespace unit_tests::sfpu_util {

const map<std::string, std::map<std::string, std::string>> sfpu_op_to_op_name = {
    // FIXME: #1157
    {"relu", {{"SFPU_OP_CHAIN_0", "relu_tile_init(); relu_tile(0);"}}},
    {"relu_min", {{"SFPU_OP_CHAIN_0", "relu_min_tile_init(); relu_min_tile(0, 0x40A33333u);"}}},  // 5.1f
    {"relu_max", {{"SFPU_OP_CHAIN_0", "relu_max_tile_init(); relu_max_tile(0, 0x40A33333u);"}}},  // 5.1f
    {"exponential", {{"SFPU_OP_CHAIN_0", "exp_tile_init(); exp_tile(0);"}}},
    {"reciprocal", {{"SFPU_OP_CHAIN_0", "recip_tile_init(); recip_tile(0);"}}},
    {"gelu", {{"SFPU_OP_CHAIN_0", "gelu_tile_init(); gelu_tile(0);"}}},
    {"gelu_accurate", {{"SFPU_OP_CHAIN_0", "gelu_tile_init<false>(); gelu_tile<false>(0);"}}},
    {"sqrt", {{"SFPU_OP_CHAIN_0", "sqrt_tile_init(); sqrt_tile(0);"}}},
    {"sigmoid", {{"SFPU_OP_CHAIN_0", "sigmoid_tile_init(); sigmoid_tile(0);"}}},
    {"silu", {{"SFPU_OP_CHAIN_0", "silu_tile_init(); silu_tile(0);"}}},
    {"log", {{"SFPU_OP_CHAIN_0", "log_tile_init(); log_tile(0);"}}},
    {"tanh", {{"SFPU_OP_CHAIN_0", "tanh_tile_init(); tanh_tile(0);"}}},
    {"sign", {{"SFPU_OP_CHAIN_0", "sign_tile_init(); sign_tile(0);"}}},
    {"rsqrt", {{"SFPU_OP_CHAIN_0", "rsqrt_tile_init(); rsqrt_tile(0);"}}},
    {"mul_unary", {{"SFPU_OP_CHAIN_0", "binop_with_scalar_tile_init(); mul_unary_tile(0, 0x40000000u);"}}},  // 2.0f
    {"square", {{"SFPU_OP_CHAIN_0", "square_tile_init(); square_tile(0);"}}},
    {"negative", {{"SFPU_OP_CHAIN_0", "negative_tile_init(); negative_tile(0);"}}},
    {"softplus",
     {{"SFPU_OP_CHAIN_0",
       "softplus_tile_init(); softplus_tile(0, /* beta */ 0x3F800000u, /* recip */0x3F800000u, /* threshold */ "
       "0x41A00000u);"}}},
    {"clamp", {{"SFPU_OP_CHAIN_0", "clamp_tile_init(); clamp_tile(0, 0xBF800000u, 0x3F800000u);"}}},  // [-1.0f, 1.0f]
    // Comparison-to-zero family (unary): result = 1.0f if predicate(x, 0) else 0.0f.
    {"eqz", {{"SFPU_OP_CHAIN_0", "eqz_tile_init(); eqz_tile(0);"}}},
    {"nez", {{"SFPU_OP_CHAIN_0", "nez_tile_init(); nez_tile(0);"}}},
    {"ltz", {{"SFPU_OP_CHAIN_0", "ltz_tile_init(); ltz_tile(0);"}}},
    {"gtz", {{"SFPU_OP_CHAIN_0", "gtz_tile_init(); gtz_tile(0);"}}},
    {"gez", {{"SFPU_OP_CHAIN_0", "gez_tile_init(); gez_tile(0);"}}},
    {"lez", {{"SFPU_OP_CHAIN_0", "lez_tile_init(); lez_tile(0);"}}},
    {"ceil", {{"SFPU_OP_CHAIN_0", "rounding_op_tile_init(); ceil_tile(0);"}}},
    {"floor", {{"SFPU_OP_CHAIN_0", "rounding_op_tile_init(); floor_tile(0);"}}},
    {"trunc", {{"SFPU_OP_CHAIN_0", "rounding_op_tile_init(); trunc_tile(0);"}}},
    {"frac", {{"SFPU_OP_CHAIN_0", "rounding_op_tile_init(); frac_tile(0);"}}},
    {"round", {{"SFPU_OP_CHAIN_0", "rounding_op_tile_init(); round_tile(0, 0 /* decimals */);"}}},
    // Scalars below match the tt-llk SFPU tests (tests/python_tests/helpers/sfpu_dispatch_constants.py).
    {"hardsigmoid",
     {{"SFPU_OP_ACTIVATIONS_INCLUDE", "1"}, {"SFPU_OP_CHAIN_0", "hardsigmoid_tile_init(); hardsigmoid_tile(0);"}}},
    {"softsign",
     {{"SFPU_OP_ACTIVATIONS_INCLUDE", "1"}, {"SFPU_OP_CHAIN_0", "softsign_tile_init(); softsign_tile(0);"}}},
    {"celu",
     {{"SFPU_OP_ACTIVATIONS_INCLUDE", "1"},
      {"SFPU_OP_CHAIN_0", "celu_tile_init(); celu_tile(0, 0x3F800000u, 0x3F800000u);"}}},  // alpha 1
    {"softshrink",
     {{"SFPU_OP_ACTIVATIONS_INCLUDE", "1"},
      {"SFPU_OP_CHAIN_0", "softshrink_tile_init(); softshrink_tile(0, 0x3F000000u);"}}},  // 0.5
    {"hardshrink",
     {{"SFPU_OP_ACTIVATIONS_INCLUDE", "1"},
      {"SFPU_OP_CHAIN_0", "hardshrink_tile_init(); hardshrink_tile(0, 0x3F000000u);"}}},  // 0.5
    {"elu", {{"SFPU_OP_CHAIN_0", "elu_tile_init(); elu_tile(0, 0x3F800000u);"}}},         // alpha 1
    {"selu",
     {{"SFPU_OP_SELU_INCLUDE", "1"}, {"SFPU_OP_CHAIN_0", "selu_tile_init(); selu_tile(0, 0x3F867D5Fu, 0x3FD62D7Du);"}}},
    {"hardtanh",
     {{"SFPU_OP_HARDTANH_INCLUDE", "1"},
      {"SFPU_OP_CHAIN_0", "hardtanh_tile_init(); hardtanh_tile(0, 0xBF800000u, 0x3F800000u);"}}},
    {"erf", {{"SFPU_OP_CHAIN_0", "erf_tile_init(); erf_tile(0);"}}},
    {"erfc", {{"SFPU_OP_CHAIN_0", "erfc_tile_init(); erfc_tile(0);"}}},
    {"erfinv", {{"SFPU_OP_ERFINV_INCLUDE", "1"}, {"SFPU_OP_CHAIN_0", "erfinv_tile_init(); erfinv_tile(0);"}}},
    {"cbrt", {{"SFPU_OP_CBRT_INCLUDE", "1"}, {"SFPU_OP_CHAIN_0", "cbrt_tile_init(); cbrt_tile(0);"}}},
    {"i0", {{"SFPU_OP_I0_INCLUDE", "1"}, {"SFPU_OP_CHAIN_0", "i0_tile_init(); i0_tile(0);"}}},
    {"i1", {{"SFPU_OP_I1_INCLUDE", "1"}, {"SFPU_OP_CHAIN_0", "i1_tile_init(); i1_tile(0);"}}},
    {"identity", {{"SFPU_OP_IDENTITY_INCLUDE", "1"}, {"SFPU_OP_CHAIN_0", "identity_tile_init(); identity_tile(0);"}}},
    {"hardmish", {{"SFPU_OP_HARDMISH_INCLUDE", "1"}, {"SFPU_OP_CHAIN_0", "hardmish_tile_init(); hardmish_tile(0);"}}},
    {"mish",
     {{"SFPU_OP_MISH_INCLUDE", "1"},
      {"SFPU_OP_CHAIN_0", "mish_tile_init<SFPU_OP_APPROX>(); mish_tile<SFPU_OP_APPROX>(0);"}}},
    {"isinf", {{"SFPU_OP_ISINF_ISNAN_INCLUDE", "1"}, {"SFPU_OP_CHAIN_0", "isinf_tile_init(); isinf_tile(0);"}}},
    {"isposinf",
     {{"SFPU_OP_ISINF_ISNAN_INCLUDE", "1"}, {"SFPU_OP_CHAIN_0", "isposinf_tile_init(); isposinf_tile(0);"}}},
    {"isneginf",
     {{"SFPU_OP_ISINF_ISNAN_INCLUDE", "1"}, {"SFPU_OP_CHAIN_0", "isneginf_tile_init(); isneginf_tile(0);"}}},
    {"isnan", {{"SFPU_OP_ISINF_ISNAN_INCLUDE", "1"}, {"SFPU_OP_CHAIN_0", "isnan_tile_init(); isnan_tile(0);"}}},
    {"isfinite",
     {{"SFPU_OP_ISINF_ISNAN_INCLUDE", "1"}, {"SFPU_OP_CHAIN_0", "isfinite_tile_init(); isfinite_tile(0);"}}},
    {"lgamma_stirling",
     {{"SFPU_OP_LGAMMA_INCLUDE", "1"}, {"SFPU_OP_CHAIN_0", "lgamma_stirling_tile_init(); lgamma_stirling_tile(0);"}}},
    {"digamma", {{"SFPU_OP_DIGAMMA_INCLUDE", "1"}, {"SFPU_OP_CHAIN_0", "digamma_tile_init(); digamma_tile(0);"}}},
    // n = 1, scale = (-1)^(n+1) * n! = 1: trigamma.
    {"polygamma",
     {{"SFPU_OP_POLYGAMMA_INCLUDE", "1"},
      {"SFPU_OP_CHAIN_0", "polygamma_tile_init(); polygamma_tile(0, 0x3F800000u, 0x3F800000u);"}}},
    {"logical_not",
     {{"SFPU_OP_LOGICAL_NOT_INCLUDE", "1"},
      {"SFPU_OP_CHAIN_0", "logical_not_tile_init(); logical_not_tile<DataFormat::Float16_b>(0);"}}},
    {"prelu",
     {{"SFPU_OP_PRELU_INCLUDE", "1"},
      {"SFPU_OP_CHAIN_0", "prelu_tile_init(); prelu_tile(0, 0x3E800000u);"}}},  // slope 0.25
    {"rdiv",
     {{"SFPU_OP_RDIV_INCLUDE", "1"}, {"SFPU_OP_CHAIN_0", "rdiv_tile_init(); rdiv_tile(0, 0x40000000u);"}}},  // 2 / x
    {"rpow",
     {{"SFPU_OP_RPOW_INCLUDE", "1"}, {"SFPU_OP_CHAIN_0", "rpow_tile_init(); rpow_tile(0, 0x40000000u);"}}},  // 2 ** x
    {"fmod",
     {{"SFPU_OP_FMOD_INCLUDE", "1"},
      {"SFPU_OP_CHAIN_0", "fmod_tile_init(0x40000000u, 0x3F000000u); fmod_tile(0);"}}},  // fmod(x, 2)
    {"remainder",
     {{"SFPU_OP_REMAINDER_INCLUDE", "1"},
      {"SFPU_OP_CHAIN_0", "remainder_tile_init(0x40000000u, 0x3F000000u); remainder_tile(0);"}}},  // x mod 2
    {"tanh_derivative",
     {{"SFPU_OP_TANH_DERIVATIVE_INCLUDE", "1"},
      {"SFPU_OP_CHAIN_0", "tanh_derivative_tile_init(); tanh_derivative_tile(0);"}}},
    {"tanhshrink",
     {{"SFPU_OP_TANHSHRINK_INCLUDE", "1"}, {"SFPU_OP_CHAIN_0", "tanhshrink_tile_init(); tanhshrink_tile(0);"}}},
    // x > 5 ? x : 10
    {"threshold",
     {{"SFPU_OP_THRESHOLD_INCLUDE", "1"},
      {"SFPU_OP_CHAIN_0", "threshold_tile_init(); threshold_tile(0, 0x40A00000u, 0x41200000u);"}}},
    {"xielu",
     {{"SFPU_OP_XIELU_INCLUDE", "1"},
      {"SFPU_OP_CHAIN_0", "xielu_tile_init(); xielu_tile(0, 0x3F800000u, 0x3F800000u);"}}},
    // beta = 5 and its reciprocal
    {"softcap",
     {{"SFPU_OP_SOFTCAP_INCLUDE", "1"},
      {"SFPU_OP_CHAIN_0", "softcap_tile_init(); softcap_tile(0, 0x40A00000u, 0x3E4CCCCDu);"}}},
    {"unary_ne", {{"SFPU_OP_CHAIN_0", "unary_ne_tile_init(); unary_ne_tile(0, 0x3F000000u);"}}},  // vs 0.5
    {"unary_eq", {{"SFPU_OP_CHAIN_0", "unary_eq_tile_init(); unary_eq_tile(0, 0x3F000000u);"}}},
    {"unary_gt", {{"SFPU_OP_CHAIN_0", "unary_gt_tile_init(); unary_gt_tile(0, 0x3F000000u);"}}},
    {"unary_ge", {{"SFPU_OP_CHAIN_0", "unary_ge_tile_init(); unary_ge_tile(0, 0x3F000000u);"}}},
    {"unary_lt", {{"SFPU_OP_CHAIN_0", "unary_lt_tile_init(); unary_lt_tile(0, 0x3F000000u);"}}},
    {"unary_le", {{"SFPU_OP_CHAIN_0", "unary_le_tile_init(); unary_le_tile(0, 0x3F000000u);"}}},
    {"power", {{"SFPU_OP_CHAIN_0", "power_tile_init(); power_tile(0, 0x40000000u);"}}},  // x ** 2.0
    {"power_iterative", {{"SFPU_OP_CHAIN_0", "power_iterative_tile_init(); power_iterative_tile(0, 3u);"}}},
    {"exp2", {{"SFPU_OP_CHAIN_0", "exp2_tile_init(); exp2_tile(0);"}}},
    {"heaviside", {{"SFPU_OP_CHAIN_0", "heaviside_tile_init(); heaviside_tile(0, 0x3F000000u);"}}},  // 0.5 at 0
    {"expm1", {{"SFPU_OP_CHAIN_0", "expm1_tile_init(); expm1_tile(0);"}}},
    // base_scale = 1 / ln(10): log10
    {"log_with_base", {{"SFPU_OP_CHAIN_0", "log_with_base_tile_init(); log_with_base_tile(0, 0x3EDE5BD9u);"}}},
    {"tiled_prod", {{"SFPU_OP_CHAIN_0", "tiled_prod_tile_init(); tiled_prod_tile(0);"}}},
    {"alt_complex_rotate90", {{"SFPU_OP_CHAIN_0", "alt_complex_rotate90_tile_init(); alt_complex_rotate90_tile(0);"}}},
};

// digamma / trigamma in double: recurrence up to x >= 6, then the asymptotic series.
double digamma(double x) {
    double result = 0.0;
    while (x < 6.0) {
        result -= 1.0 / x;
        x += 1.0;
    }
    const double inv = 1.0 / x;
    const double inv2 = inv * inv;
    return result + std::log(x) - 0.5 * inv -
           inv2 * (1.0 / 12 - inv2 * (1.0 / 120 - inv2 * (1.0 / 252 - inv2 * (1.0 / 240 - inv2 / 132))));
}

double trigamma(double x) {
    double result = 0.0;
    while (x < 6.0) {
        result += 1.0 / (x * x);
        x += 1.0;
    }
    const double inv = 1.0 / x;
    const double inv2 = inv * inv;
    return result + inv + 0.5 * inv2 + inv * inv2 * (1.0 / 6 - inv2 * (1.0 / 30 - inv2 * (1.0 / 42 - inv2 / 30)));
}

// Modified Bessel I_n(x) for n = 0, 1 by its power series; libc++ has no std::cyl_bessel_i.
double bessel_i(int order, double x) {
    const double q = 0.25 * x * x;
    double term = order == 0 ? 1.0 : 0.5 * x;
    double sum = term;
    for (int k = 1; k < 200 && std::fabs(term) > 1e-17 * std::fabs(sum); ++k) {
        term *= q / (k * (k + order));
        sum += term;
    }
    return sum;
}

// Ops whose result is exact (rounding, a 0 / 1 predicate, or a copied value), compared bit-exactly.
bool is_exact_unary_sfpu_op(const std::string& op_name) {
    return op_name == "ceil" || op_name == "floor" || op_name == "trunc" || op_name == "frac" || op_name == "round" ||
           op_name == "isinf" || op_name == "isposinf" || op_name == "isneginf" || op_name == "isnan" ||
           op_name == "isfinite" || op_name == "logical_not" || op_name == "heaviside" || op_name == "identity" ||
           op_name.starts_with("unary_");
}

// Binary SFPU ops driven by `run_sfpu_binary_two_input_buffer`.
//
// Each entry maps an op name (the test parameter) to the kernel-side macro
// substitutions expanded by the SFPU binary compute kernel
// (`eltwise_sfpu_2_0.cpp`, Metal 2.0 dataflow-buffer based, SFPU_BINARY_OP):
//
//   * SFPU_OP_INIT_0  — runs once before the per-pair loop. Used to set up
//                       SFPU lookup tables / state (e.g. div reciprocal LUT).
//   * SFPU_OP_*_INCLUDE — the sfpu_split_includes.h define for the op's header.
//   * SFPU_OP_CHAIN_0 — runs once per (LHS, RHS) pair, inside an
//                       acquire/release section. By convention LHS lives at
//                       DST[0] and RHS at DST[1]; the result is written back
//                       to DST[0] so the packer reads from there.
//
// To add a new binary SFPU op: add an entry here, add a matching arm in
// sfpu_binary_function() or get_binary_int_operation_result() for golden compute,
// and (if its valid input range differs from div) add an arm in generate_packed_sfpu_binary_inputs().
const map<std::string, std::map<std::string, std::string>> sfpu_binary_op_to_op_name = {
    {"div_binary",
     {{"SFPU_OP_BINARY_DIV_INCLUDE", "1"},
      {"SFPU_OP_INIT_0", "div_binary_tile_init();"},
      {"SFPU_OP_CHAIN_0", "div_binary_tile(0, 1, 0);"}}},
    {"mul_float",
     {{"SFPU_OP_BINARY_DIV_INCLUDE", "1"},
      {"SFPU_OP_INIT_0", "mul_binary_tile_init();"},
      {"SFPU_OP_CHAIN_0", "mul_binary_tile(0, 1, 0);"}}},
    {"atan2",
     {{"SFPU_OP_BINARY_ATAN2_INCLUDE", "1"},
      {"SFPU_OP_INIT_0", "atan2_binary_tile_init();"},
      {"SFPU_OP_CHAIN_0", "atan2_binary_tile(0, 1, 0);"}}},
    // add_int: Int8 L1 inputs are promoted to sign-magnitude Int32 in DEST via copy_tile + fp32_dest_acc;
    // add_int_tile<Int32> (sign-mag on Quasar via ARCH_QUASAR) then adds in sign-mag space. Result in DST[0].
    {"add_int",
     {{"SFPU_OP_BINARY_ADD_INT_INCLUDE", "1"},
      {"SFPU_OP_INIT_0", "add_int_tile_init();"},
      {"SFPU_OP_CHAIN_0", "add_int_tile<DataFormat::Int32>(0, 1, 0);"}}},
    {"mul_int",
     {{"SFPU_OP_BINARY_MUL_INT_INCLUDE", "1"},
      {"SFPU_OP_INIT_0", "mul_int_tile_init<DataFormat::Int32>();"},
      {"SFPU_OP_CHAIN_0", "mul_int_tile<DataFormat::Int32>(0, 1, 0);"}}},
    {"gt_int",
     {{"SFPU_OP_BINARY_GT_INT_INCLUDE", "1"},
      {"SFPU_OP_INIT_0", "gt_int_tile_init<DataFormat::Int32>();"},
      {"SFPU_OP_CHAIN_0", "gt_int_tile<DataFormat::Int32>(0, 1, 0);"}}},
    {"binary_max",
     {{"SFPU_OP_BINARY_MAX_MIN_INCLUDE", "1"},
      {"SFPU_OP_INIT_0", "binary_max_tile_init();"},
      {"SFPU_OP_CHAIN_0", "binary_max_tile(0, 1, 0);"}}},
    {"binary_min",
     {{"SFPU_OP_BINARY_MAX_MIN_INCLUDE", "1"},
      {"SFPU_OP_INIT_0", "binary_min_tile_init();"},
      {"SFPU_OP_CHAIN_0", "binary_min_tile(0, 1, 0);"}}},
    {"binary_max_int32",
     {{"SFPU_OP_BINARY_MAX_MIN_INCLUDE", "1"},
      {"SFPU_OP_INIT_0", "binary_max_int32_tile_init();"},
      {"SFPU_OP_CHAIN_0", "binary_max_int32_tile(0, 1, 0);"}}},
    {"binary_min_int32",
     {{"SFPU_OP_BINARY_MAX_MIN_INCLUDE", "1"},
      {"SFPU_OP_INIT_0", "binary_min_int32_tile_init();"},
      {"SFPU_OP_CHAIN_0", "binary_min_int32_tile(0, 1, 0);"}}},
    {"copy_dest",
     {{"SFPU_OP_COPY_DEST_INCLUDE", "1"},
      {"SFPU_OP_INIT_0", "copy_dest_values_init();"},
      {"SFPU_OP_CHAIN_0", "copy_dest_values<DataFormat::Float16_b>(1, 0);"}}},
    {"copy_dest_int",
     {{"SFPU_OP_COPY_DEST_INCLUDE", "1"},
      {"SFPU_OP_INIT_0", "copy_dest_values_init();"},
      {"SFPU_OP_CHAIN_0", "copy_dest_values<DataFormat::Int32>(1, 0);"}}},
    // The integer ops below ride the Int8 -> sign-magnitude Int32 Dest path with non-negative operands,
    // where sign-magnitude and two's complement agree (their kernels assume two's complement).
    {"power_binary",
     {{"SFPU_OP_BINARY_DIV_INCLUDE", "1"},
      {"SFPU_OP_INIT_0", "power_binary_tile_init();"},
      {"SFPU_OP_CHAIN_0", "power_binary_tile(0, 1, 0);"}}},
    {"fmod_binary",
     {{"SFPU_OP_BINARY_FMOD_INCLUDE", "1"},
      {"SFPU_OP_INIT_0", "fmod_binary_tile_init();"},
      {"SFPU_OP_CHAIN_0", "fmod_binary_tile(0, 1, 0);"}}},
    {"remainder_binary",
     {{"SFPU_OP_BINARY_REMAINDER_INCLUDE", "1"},
      {"SFPU_OP_INIT_0", "remainder_binary_tile_init();"},
      {"SFPU_OP_CHAIN_0", "remainder_binary_tile(0, 1, 0);"}}},
    // RHS carries exp(-x), computed on the host.
    {"logsigmoid",
     {{"SFPU_OP_LOGSIGMOID_INCLUDE", "1"},
      {"SFPU_OP_INIT_0", "logsigmoid_tile_init();"},
      {"SFPU_OP_CHAIN_0", "logsigmoid_tile(0, 1, 0);"}}},
    // torch defaults: rtol 1e-5, atol 1e-8
    {"isclose",
     {{"SFPU_OP_BINARY_ISCLOSE_INCLUDE", "1"},
      {"SFPU_OP_INIT_0", "isclose_binary_tile_init();"},
      {"SFPU_OP_CHAIN_0", "isclose_binary_tile(0, 1, 0, 0x3727C5ACu, 0x322BCC77u);"}}},
    {"clamped_silu_glu",
     {{"SFPU_OP_CLAMPED_SILU_GLU_INCLUDE", "1"},
      {"SFPU_OP_INIT_0", "clamped_silu_glu_tile_init();"},
      {"SFPU_OP_CHAIN_0", "clamped_silu_glu_tile(0, 1, 0);"}}},
    {"situ_glu",
     {{"SFPU_OP_SITU_GLU_INCLUDE", "1"},
      {"SFPU_OP_INIT_0", "situ_glu_tile_init();"},
      {"SFPU_OP_CHAIN_0", "situ_glu_tile(0, 1, 0);"}}},
    // mask tile at DST[0] + 1
    {"mask",
     {{"SFPU_OP_MASK_INCLUDE", "1"}, {"SFPU_OP_INIT_0", "mask_tile_init();"}, {"SFPU_OP_CHAIN_0", "mask_tile(0, 1);"}}},
    {"mask_posinf",
     {{"SFPU_OP_MASK_INCLUDE", "1"},
      {"SFPU_OP_INIT_0", "mask_tile_init();"},
      {"SFPU_OP_CHAIN_0", "mask_posinf_tile(0, 1);"}}},
    // RHS carries log(x) (x >= 0.5 here), computed on the host.
    {"lgamma_stirling_float",
     {{"SFPU_OP_LGAMMA_INCLUDE", "1"},
      {"SFPU_OP_INIT_0", "lgamma_stirling_float_tile_init();"},
      {"SFPU_OP_CHAIN_0", "lgamma_stirling_float_tile(0, 1, 0);"}}},
    // Float32 in a 32-bit Dest, bf16 in L1.
    {"add_top_row",
     {{"SFPU_OP_COMPUTE_KERNEL_API_INCLUDE", "1"},
      {"SFPU_OP_INIT_0", "sfpu_add_top_row_init();"},
      {"SFPU_OP_CHAIN_0", "sfpu_add_top_row<DataFormat::Float32>(0, 1, 0);"}}},
    // Integer ops.
    {"add_top_row_int32",
     {{"SFPU_OP_COMPUTE_KERNEL_API_INCLUDE", "1"},
      {"SFPU_OP_INIT_0", "sfpu_add_top_row_init();"},
      {"SFPU_OP_CHAIN_0", "sfpu_add_top_row<DataFormat::Int32>(0, 1, 0);"}}},
    {"div_int32_floor",
     {{"SFPU_OP_BINARY_DIV_INT32_FLOOR_INCLUDE", "1"},
      {"SFPU_OP_INIT_0", "div_int32_floor_tile_init();"},
      {"SFPU_OP_CHAIN_0", "div_int32_floor_tile(0, 1, 0);"}}},
    {"div_int32_trunc",
     {{"SFPU_OP_BINARY_DIV_INT32_FLOOR_INCLUDE", "1"},
      {"SFPU_OP_INIT_0", "div_int32_trunc_tile_init();"},
      {"SFPU_OP_CHAIN_0", "div_int32_trunc_tile(0, 1, 0);"}}},
    // int32 / int32 -> Float32 result.
    {"div_int32",
     {{"SFPU_OP_BINARY_DIV_INT32_INCLUDE", "1"},
      {"SFPU_OP_INIT_0", "div_int32_tile_init();"},
      {"SFPU_OP_CHAIN_0", "div_int32_tile(0, 1, 0);"}}},
    {"fmod_int32",
     {{"SFPU_OP_BINARY_FMOD_INCLUDE", "1"},
      {"SFPU_OP_INIT_0", "fmod_int32_tile_init();"},
      {"SFPU_OP_CHAIN_0", "fmod_int32_tile(0, 1, 0);"}}},
    {"remainder_int32",
     {{"SFPU_OP_BINARY_REMAINDER_INCLUDE", "1"},
      {"SFPU_OP_INIT_0", "remainder_int32_tile_init();"},
      {"SFPU_OP_CHAIN_0", "remainder_int32_tile(0, 1, 0);"}}},
    {"bitwise_and_binary",
     {{"SFPU_OP_BINARY_BITWISE_INCLUDE", "1"},
      {"SFPU_OP_INIT_0", "binary_bitwise_tile_init();"},
      {"SFPU_OP_CHAIN_0", "bitwise_and_binary_tile<DataFormat::Int32>(0, 1, 0);"}}},
    {"bitwise_or_binary",
     {{"SFPU_OP_BINARY_BITWISE_INCLUDE", "1"},
      {"SFPU_OP_INIT_0", "binary_bitwise_tile_init();"},
      {"SFPU_OP_CHAIN_0", "bitwise_or_binary_tile<DataFormat::Int32>(0, 1, 0);"}}},
    {"bitwise_xor_binary",
     {{"SFPU_OP_BINARY_BITWISE_INCLUDE", "1"},
      {"SFPU_OP_INIT_0", "binary_bitwise_tile_init();"},
      {"SFPU_OP_CHAIN_0", "bitwise_xor_binary_tile<DataFormat::Int32>(0, 1, 0);"}}},
    // DST[1] - DST[0]
    {"rsub_int",
     {{"SFPU_OP_BINARY_SUB_INT_INCLUDE", "1"},
      {"SFPU_OP_INIT_0", "rsub_int_tile_init();"},
      {"SFPU_OP_CHAIN_0", "rsub_int_tile<DataFormat::Int32>(0, 1, 0);"}}},
    // DST[0] += DST[0 + 1]
    {"sfpu_add_int",
     {{"SFPU_OP_INT_SUM_INCLUDE", "1"},
      {"SFPU_OP_INIT_0", "sfpu_sum_int_init();"},
      {"SFPU_OP_CHAIN_0", "sfpu_add_int(0, 1);"}}},
    {"int_mask",
     {{"SFPU_OP_MASK_INCLUDE", "1"},
      {"SFPU_OP_INIT_0", "mask_tile_init();"},
      {"SFPU_OP_CHAIN_0", "mask_tile(0, 1, DataFormat::Int32);"}}},
};

// ---- Tile-layout ops (goldens mirror the tt-llk SFPU tests' UnarySFPUGolden) --------------------
//
// L1 holds tiles in face order; these goldens are written in (row, col) of the 32x32 tile, as the
// tt-llk ones are, so map between the two.
inline size_t tile_order_index(size_t row, size_t col) {
    const size_t face = (row / 16) * 2 + (col / 16);
    return face * 256 + (row % 16) * 16 + (col % 16);
}

// Ops that move values across the tile, so their golden works on whole tiles.
bool is_tile_layout_sfpu_op(const std::string& op_name) {
    return op_name == "tiled_prod" || op_name == "alt_complex_rotate90";
}

// One 1024-element tile, face order in and out. `store` rounds a value the way Dest holds it.
std::vector<float> tile_layout_golden(
    const std::string& op_name, const std::vector<float>& tile, const std::function<float(float)>& store) {
    std::vector<float> out = tile;
    const auto at = [&](size_t row, size_t col) -> float& { return out[tile_order_index(row, col)]; };
    if (op_name == "alt_complex_rotate90") {
        for (size_t row = 0; row < 32; ++row) {
            for (size_t col = 0; col < 32; col += 2) {
                const float re = tile[tile_order_index(row, col)], im = tile[tile_order_index(row, col + 1)];
                at(row, col) = -im;
                at(row, col + 1) = re;
            }
        }
        return out;
    }
    if (op_name == "tiled_prod") {
        // Every SFPU lane (row r of each 4-row group, column 2c + parity) keeps a running product over
        // its 8 values per face; a face's ninth step multiplies into the next face's first row, so the
        // chain runs through faces 0-3 (the last face's ninth step lands past the tile, not modelled).
        const auto position = [](size_t face, size_t step, size_t lane_row, size_t lane_col) {
            return std::pair{16 * (face / 2) + 4 * (step / 2) + lane_row, 16 * (face % 2) + 2 * lane_col + (step & 1)};
        };
        for (size_t lane_row = 0; lane_row < 4; ++lane_row) {
            for (size_t lane_col = 0; lane_col < 8; ++lane_col) {
                for (size_t face = 0; face < 4; ++face) {
                    float product = 1.0f;
                    for (size_t step = 0; step < 9; ++step) {
                        const size_t target_face = step == 8 ? face + 1 : face;
                        if (target_face == 4) {
                            break;
                        }
                        const auto [row, col] = position(target_face, step % 8, lane_row, lane_col);
                        product = product * at(row, col);
                        at(row, col) = store(product);
                    }
                }
            }
        }
        return out;
    }
    TT_THROW("Unsupported layout op_name in test");
}

// Integer layout ops on Int32 in Dest.
std::vector<int32_t> int_tile_layout_golden(const std::string& op_name, const std::vector<int32_t>& tile) {
    std::vector<int32_t> out = tile;
    const auto in = [&](size_t row, size_t col) { return static_cast<int64_t>(tile[tile_order_index(row, col)]); };
    if (op_name == "sum_int_col") {
        // Row r = 0..3 of every column becomes the sum of rows r, r+4, ..., r+28.
        for (size_t row = 0; row < 4; ++row) {
            for (size_t col = 0; col < 32; ++col) {
                int64_t sum = 0;
                for (size_t r = row; r < 32; r += 4) {
                    sum += in(r, col);
                }
                out[tile_order_index(row, col)] = static_cast<int32_t>(sum);
            }
        }
        return out;
    }
    if (op_name == "sum_int_row") {
        // Even column 2c < 16 of every row becomes x[2c] + x[2c+1] + x[2c+16] + x[2c+17].
        for (size_t row = 0; row < 32; ++row) {
            for (size_t col = 0; col < 16; col += 2) {
                out[tile_order_index(row, col)] =
                    static_cast<int32_t>(in(row, col) + in(row, col + 1) + in(row, col + 16) + in(row, col + 17));
            }
        }
        return out;
    }
    TT_THROW("Unsupported int layout op_name in test");
}

// ---- Int32 unary ops ------------------------------------------------------------------------------
//
// Int8 L1 (non-negative, so sign-magnitude == two's complement) copied into a 32-bit Dest, Int32 out.
// Scalars match the tt-llk SFPU tests.
const map<std::string, std::map<std::string, std::string>> sfpu_int32_unary_op_to_op_name = {
    {"bitwise_and",
     {{"SFPU_OP_BITWISE_INCLUDE", "1"},
      {"SFPU_OP_CHAIN_0", "bitwise_and_tile_init(); bitwise_and_tile<DataFormat::Int32>(0, 0x70FF00F5u);"}}},
    {"bitwise_or",
     {{"SFPU_OP_BITWISE_INCLUDE", "1"},
      {"SFPU_OP_CHAIN_0", "bitwise_or_tile_init(); bitwise_or_tile<DataFormat::Int32>(0, 0x70FF00F5u);"}}},
    {"bitwise_xor",
     {{"SFPU_OP_BITWISE_INCLUDE", "1"},
      {"SFPU_OP_CHAIN_0", "bitwise_xor_tile_init(); bitwise_xor_tile<DataFormat::Int32>(0, 0x70FF00F5u);"}}},
    {"left_shift",
     {{"SFPU_OP_SHIFT_INCLUDE", "1"},
      {"SFPU_OP_CHAIN_0", "left_shift_tile_init(); left_shift_tile<DataFormat::Int32>(0, 3u);"}}},
    {"right_shift",
     {{"SFPU_OP_SHIFT_INCLUDE", "1"},
      {"SFPU_OP_CHAIN_0", "right_shift_tile_init(); right_shift_tile<DataFormat::Int32>(0, 3u);"}}},
    {"rsub_unary_int32",
     {{"SFPU_OP_RSUB_INCLUDE", "1"},
      {"SFPU_OP_CHAIN_0", "rsub_unary_int32_tile_init(); rsub_unary_int32_tile(0, 0x7FFFFFFFu);"}}},
    {"logical_not_int32",
     {{"SFPU_OP_LOGICAL_NOT_INCLUDE", "1"},
      {"SFPU_OP_CHAIN_0", "logical_not_tile_init(); logical_not_tile<DataFormat::Int32>(0);"}}},
    {"sum_int_col",
     {{"SFPU_OP_INT_SUM_INCLUDE", "1"}, {"SFPU_OP_CHAIN_0", "sfpu_sum_int_init(); sfpu_sum_int_col(0);"}}},
    {"sum_int_row",
     {{"SFPU_OP_INT_SUM_INCLUDE", "1"}, {"SFPU_OP_CHAIN_0", "sfpu_sum_int_init(); sfpu_sum_int_row(0);"}}},
};

int32_t int32_unary_result(const std::string& op_name, int32_t x) {
    if (op_name == "bitwise_and") {
        return x & 0x70FF00F5;
    }
    if (op_name == "bitwise_or") {
        return x | 0x70FF00F5;
    }
    if (op_name == "bitwise_xor") {
        return x ^ 0x70FF00F5;
    }
    if (op_name == "left_shift") {
        return x << 3;
    }
    if (op_name == "right_shift") {
        return x >> 3;
    }
    if (op_name == "rsub_unary_int32") {
        return 0x7FFFFFFF - x;
    }
    if (op_name == "logical_not_int32") {
        return x == 0 ? 1 : 0;
    }
    TT_THROW("Unsupported int unary op_name in test");
}

// Int8 binary ops whose operands stay non-negative (see generate_non_negative_int8_binary_inputs).
bool is_non_negative_int8_binary_op(const std::string& op_name) {
    return op_name == "add_top_row_int32" || op_name == "div_int32_floor" || op_name == "div_int32_trunc" ||
           op_name == "div_int32" || op_name == "fmod_int32" || op_name == "remainder_int32" ||
           op_name == "bitwise_and_binary" || op_name == "bitwise_or_binary" || op_name == "bitwise_xor_binary" ||
           op_name == "rsub_int" || op_name == "sfpu_add_int" || op_name == "int_mask";
}

// Rows 0-3 of faces 0 and 1 in tile order: what sfpu_add_top_row writes (see ckernel_sfpu_add_top_row.h).
bool is_add_top_row_element(size_t element_in_tile) {
    return element_in_tile < 64 || (element_in_tile >= 256 && element_in_tile < 320);
}

// Non-negative Int8 operands, packed four per word (sign bit clear, so sign-magnitude == value).
// Divisors stay >= 1; rsub_int keeps RHS >= LHS so the result is non-negative too.
std::pair<vector<uint32_t>, vector<uint32_t>> generate_non_negative_int8_binary_inputs(
    const unsigned int numel, const std::string& op_name, const int seed) {
    std::mt19937 rng(seed);
    const bool divides = op_name.starts_with("div_int32") || op_name == "fmod_int32" || op_name == "remainder_int32";
    const bool masks = op_name == "int_mask";
    const bool is_rsub = op_name == "rsub_int";
    int rhs_min = is_rsub ? 64 : 0;
    if (divides) {
        rhs_min = 1;
    }
    std::uniform_int_distribution<int> lhs_dist(0, is_rsub ? 63 : 127);
    std::uniform_int_distribution<int> rhs_dist(rhs_min, masks ? 1 : 127);
    const size_t words = (numel + 3) / 4;
    vector<uint32_t> lhs(words, 0), rhs(words, 0);
    for (size_t i = 0; i < numel; ++i) {
        lhs[i / 4] |= static_cast<uint32_t>(lhs_dist(rng)) << (8 * (i % 4));
        rhs[i / 4] |= static_cast<uint32_t>(rhs_dist(rng)) << (8 * (i % 4));
    }
    return {lhs, rhs};
}

bool is_int8_binary_sfpu_op(const std::string& op_name) {
    return is_non_negative_int8_binary_op(op_name) or (op_name == "add_int") or (op_name == "mul_int") or
           (op_name == "gt_int") or (op_name == "binary_max_int32") or (op_name == "binary_min_int32") or
           (op_name == "copy_dest_int");
}

// Scalar golden for the unary SFPU ops, computed in float. Both the bf16 and
// the Float32 paths share this: bf16 wraps the result in bfloat16,
// Float32 consumes it directly. Keeping the math in one place is what lets the
// two data-format paths stay in sync.
float sfpu_function(const std::string& op_name, float input) {
    const double d = input;  // for the goldens computed in double
    if (op_name == "relu") {
        return fmaxf(input, 0.0f);
    }
    if (op_name == "relu_min") {
        return fmaxf(input, 5.1f);
    }
    if (op_name == "relu_max") {
        return fmaxf(0.0f, fminf(input, 5.1f));
    }
    if (op_name == "exponential") {
        return std::exp(input);
    }
    if (op_name == "reciprocal") {
        return 1 / input;
    }
    if (op_name == "gelu") {
        static constexpr float alpha = M_2_SQRTPI * M_SQRT1_2;
        auto x3 = input * input * input;
        return input * 0.5 * (1.0 + tanhf(alpha * (input + 0.044715 * x3)));
    }
    if (op_name == "gelu_accurate") {
        // In double: erf(x/sqrt(2)) tends to -1 in the negative tail, so 1 + erf cancels. Computed
        // in float the golden is already 1.8% off at x = -4.84, past this op's 1% rtol.
        const double x = input;
        return static_cast<float>(0.5 * x * (1.0 + erf(x * M_SQRT1_2)));
    }
    if (op_name == "sqrt") {
        return sqrtf(input);
    }
    if (op_name == "sigmoid") {
        return 1 / (1 + std::exp(-input));
    }
    if (op_name == "silu") {
        return input / (1 + std::exp(-input));
    }
    if (op_name == "log") {
        return logf(input);
    }
    if (op_name == "tanh") {
        return std::tanh(input);
    }
    if (op_name == "rsqrt") {
        return 1.0f / sqrtf(input);
    }
    if (op_name == "sign") {
        return static_cast<float>((input > 0.0f) - (input < 0.0f));
    }
    if (op_name == "mul_unary") {
        return bfloat16(static_cast<float>(input) * 2.0f);
    }
    if (op_name == "square") {
        return bfloat16(static_cast<float>(input) * static_cast<float>(input));
    }
    if (op_name == "negative") {
        return -input;
    }
    if (op_name == "softplus") {
        return (input > 20.0f) ? input : std::log1p(std::exp(input));
    }
    if (op_name == "clamp") {
        return std::clamp(input, -1.0f, 1.0f);
    }
    if (op_name == "eqz") {
        return bfloat16(static_cast<float>(input) == 0.0f ? 1.0f : 0.0f);
    }
    if (op_name == "nez") {
        return bfloat16(static_cast<float>(input) != 0.0f ? 1.0f : 0.0f);
    }
    if (op_name == "ltz") {
        return bfloat16(static_cast<float>(input) < 0.0f ? 1.0f : 0.0f);
    }
    if (op_name == "gtz") {
        return bfloat16(static_cast<float>(input) > 0.0f ? 1.0f : 0.0f);
    }
    if (op_name == "gez") {
        return bfloat16(static_cast<float>(input) >= 0.0f ? 1.0f : 0.0f);
    }
    if (op_name == "lez") {
        return bfloat16(static_cast<float>(input) <= 0.0f ? 1.0f : 0.0f);
    }
    if (op_name == "ceil") {
        return std::ceil(input);
    }
    if (op_name == "floor") {
        return std::floor(input);
    }
    if (op_name == "trunc") {
        return std::trunc(input);
    }
    if (op_name == "frac") {
        return input - std::trunc(input);
    }
    if (op_name == "round") {
        // Round-half-to-even (matches _round_even_). std::round is away-from-zero.
        return std::nearbyint(input);
    }
    if (op_name == "hardsigmoid") {
        return std::clamp(input / 6.0f + 0.5f, 0.0f, 1.0f);
    }
    if (op_name == "softsign") {
        return input / (1.0f + std::fabs(input));
    }
    if (op_name == "celu" || op_name == "elu") {
        return input > 0.0f ? input : static_cast<float>(std::expm1(d));
    }
    if (op_name == "softshrink") {
        if (input > 0.5f) {
            return input - 0.5f;
        }
        return input < -0.5f ? input + 0.5f : 0.0f;
    }
    if (op_name == "hardshrink") {
        return std::fabs(input) > 0.5f ? input : 0.0f;
    }
    if (op_name == "selu") {
        constexpr double scale = 1.0507009873554805, alpha = 1.6732632423543772;
        return static_cast<float>(scale * (input > 0.0f ? d : alpha * std::expm1(d)));
    }
    if (op_name == "hardtanh") {
        return std::clamp(input, -1.0f, 1.0f);
    }
    if (op_name == "erf") {
        return static_cast<float>(std::erf(d));
    }
    if (op_name == "erfc") {
        return static_cast<float>(std::erfc(d));
    }
    if (op_name == "erfinv") {
        // Newton on erf(y) = x from a rough start; |x| < 1 here.
        double y = 0.0;
        for (int i = 0; i < 60; ++i) {
            y -= (std::erf(y) - d) / (M_2_SQRTPI * std::exp(-y * y));
        }
        return static_cast<float>(y);
    }
    if (op_name == "cbrt") {
        return std::cbrt(input);
    }
    if (op_name == "i0") {
        return static_cast<float>(bessel_i(0, std::fabs(d)));
    }
    if (op_name == "i1") {
        return static_cast<float>(bessel_i(1, d));
    }
    if (op_name == "identity") {
        return input;
    }
    if (op_name == "hardmish") {
        return input * std::clamp(0.5f * input + 1.0f, 0.0f, 1.0f);
    }
    if (op_name == "mish") {
        return static_cast<float>(d * std::tanh(std::log1p(std::exp(d))));
    }
    if (op_name == "isinf") {
        return std::isinf(input) ? 1.0f : 0.0f;
    }
    if (op_name == "isposinf") {
        return (std::isinf(input) && input > 0.0f) ? 1.0f : 0.0f;
    }
    if (op_name == "isneginf") {
        return (std::isinf(input) && input < 0.0f) ? 1.0f : 0.0f;
    }
    if (op_name == "isnan") {
        return std::isnan(input) ? 1.0f : 0.0f;
    }
    if (op_name == "isfinite") {
        return std::isfinite(input) ? 1.0f : 0.0f;
    }
    if (op_name == "lgamma_stirling") {
        return static_cast<float>(std::lgamma(d));
    }
    if (op_name == "digamma") {
        return static_cast<float>(digamma(d));
    }
    if (op_name == "polygamma") {
        return static_cast<float>(trigamma(d));
    }
    if (op_name == "logical_not") {
        return input == 0.0f ? 1.0f : 0.0f;
    }
    if (op_name == "prelu") {
        return input >= 0.0f ? input : 0.25f * input;
    }
    if (op_name == "rdiv") {
        return 2.0f / input;
    }
    if (op_name == "rpow") {
        return static_cast<float>(std::exp2(d));
    }
    if (op_name == "fmod") {
        return static_cast<float>(std::fmod(d, 2.0));
    }
    if (op_name == "remainder") {
        return static_cast<float>(d - 2.0 * std::floor(d / 2.0));
    }
    if (op_name == "tanh_derivative") {
        const double t = std::tanh(d);
        return static_cast<float>(1.0 - t * t);
    }
    if (op_name == "tanhshrink") {
        return static_cast<float>(d - std::tanh(d));
    }
    if (op_name == "threshold") {
        return input > 5.0f ? input : 10.0f;
    }
    if (op_name == "xielu") {
        // alpha_p = alpha_n = 1, beta = 0.5 (the kernel's fixed beta).
        return static_cast<float>(input > 0.0f ? d * d + 0.5 * d : std::expm1(d) - d + 0.5 * d);
    }
    if (op_name == "softcap") {
        return static_cast<float>(5.0 * std::tanh(d / 5.0));
    }
    if (op_name == "unary_ne") {
        return input != 0.5f ? 1.0f : 0.0f;
    }
    if (op_name == "unary_eq") {
        return input == 0.5f ? 1.0f : 0.0f;
    }
    if (op_name == "unary_gt") {
        return input > 0.5f ? 1.0f : 0.0f;
    }
    if (op_name == "unary_ge") {
        return input >= 0.5f ? 1.0f : 0.0f;
    }
    if (op_name == "unary_lt") {
        return input < 0.5f ? 1.0f : 0.0f;
    }
    if (op_name == "unary_le") {
        return input <= 0.5f ? 1.0f : 0.0f;
    }
    if (op_name == "power") {
        return input * input;
    }
    if (op_name == "power_iterative") {
        return input * input * input;
    }
    if (op_name == "exp2") {
        return static_cast<float>(std::exp2(d));
    }
    if (op_name == "heaviside") {
        if (input < 0.0f) {
            return 0.0f;
        }
        return input == 0.0f ? 0.5f : 1.0f;
    }
    if (op_name == "expm1") {
        return static_cast<float>(std::expm1(d));
    }
    if (op_name == "log_with_base") {
        return static_cast<float>(std::log10(d));
    }
    TT_THROW("Unsupported op_name in test");
}

bfloat16 sfpu_function(const std::string& op_name, const bfloat16& input) {
    return bfloat16(sfpu_function(op_name, static_cast<float>(input)));
}

// Reference implementation for binary SFPU ops.
bfloat16 sfpu_binary_function(const std::string& op_name, const bfloat16& lhs_bf16, const bfloat16& rhs_bf16) {
    const float lhs = static_cast<float>(lhs_bf16), rhs = static_cast<float>(rhs_bf16);
    const double a = lhs, b = rhs;
    if (op_name == "div_binary") {
        return bfloat16(lhs / rhs);
    }
    if (op_name == "mul_float") {
        return bfloat16(lhs * rhs);
    }
    if (op_name == "atan2") {
        // Compute API convention: the first operand is y and the second is x.
        return bfloat16(std::atan2(lhs, rhs));
    }
    if (op_name == "binary_max") {
        return bfloat16(std::max(lhs, rhs));
    }
    if (op_name == "binary_min") {
        return bfloat16(std::min(lhs, rhs));
    }
    if (op_name == "copy_dest") {
        // copy_dest_values(1, 0) copies RHS onto DST[0].
        return rhs_bf16;
    }
    if (op_name == "power_binary") {
        return bfloat16(static_cast<float>(std::pow(a, b)));
    }
    if (op_name == "fmod_binary") {
        return bfloat16(static_cast<float>(std::fmod(a, b)));
    }
    if (op_name == "remainder_binary") {
        return bfloat16(static_cast<float>(a - b * std::floor(a / b)));
    }
    if (op_name == "logsigmoid") {
        return bfloat16(static_cast<float>(-std::log1p(std::exp(-a))));
    }
    if (op_name == "isclose") {
        return bfloat16(std::fabs(a - b) <= 1e-8 + 1e-5 * std::fabs(b) ? 1.0f : 0.0f);
    }
    if (op_name == "clamped_silu_glu") {
        const double gate = std::min(a, 10.0), up = std::clamp(b, -10.0, 10.0);
        return bfloat16(static_cast<float>(gate / (1.0 + std::exp(-gate)) * up));
    }
    if (op_name == "situ_glu") {
        const double sigmoid = 1.0 / (1.0 + std::exp(-a));
        return bfloat16(static_cast<float>(4.0 * std::tanh(a / 4.0) * sigmoid * 25.0 * std::tanh(b / 25.0)));
    }
    if (op_name == "mask") {
        return bfloat16(rhs == 0.0f ? 0.0f : lhs);
    }
    if (op_name == "mask_posinf") {
        return bfloat16(rhs == 0.0f ? std::numeric_limits<float>::infinity() : lhs);
    }
    if (op_name == "lgamma_stirling_float") {
        // The kernel's Stirling series takes log(x) from RHS, which is bf16-rounded: its
        // (x - 0.5) * log(x) term carries that rounding, so the golden does too.
        return bfloat16(static_cast<float>(std::lgamma(a) + (a - 0.5) * (b - std::log(a))));
    }
    if (op_name == "add_top_row") {
        return bfloat16(lhs + rhs);  // top rows only; the caller restores the rest
    }
    TT_THROW("Unsupported binary op_name in test");
}

// Ternary SFPU ops driven by `run_sfpu_ternary_three_input_buffer`.
//
// Each entry maps an op name to SFPU_OP_CHAIN_0 — the full per-tile body
// (init + compute) run once per (in0, in1, in2) triple inside an
// acquire/release section. Mirrors the unary pattern where init and compute
// are both part of the chain.
const map<std::string, std::map<std::string, std::string>> sfpu_ternary_op_to_op_name = {
    {"where",
     {{"SFPU_OP_WHERE_INCLUDE", "1"},
      {"SFPU_OP_CHAIN_0", "where_tile_init(); where_tile<DataFormat::Float16_b>(0, 1, 2, 3);"}}},
    // mac always writes DST[0], so its result is copied to DST[3], the tile this runner packs.
    {"mac",
     {{"SFPU_OP_MAC_INCLUDE", "1"},
      {"SFPU_OP_COPY_DEST_INCLUDE", "1"},
      {"SFPU_OP_CHAIN_0",
       "mac_tile_init<DataFormat::Float16_b>(); mac_tile<DataFormat::Float16_b>(0, 1, 2, 0); "
       "copy_dest_values_init(); copy_dest_values<DataFormat::Float16_b>(0, 3);"}}},
    {"lerp",
     {{"SFPU_OP_LERP_INCLUDE", "1"},
      {"SFPU_OP_CHAIN_0", "lerp_tile_init(); lerp_tile<DataFormat::Float16_b>(0, 1, 2, 3);"}}},
    // value = 0.5
    {"addcdiv",
     {{"SFPU_OP_ADDCDIV_INCLUDE", "1"},
      {"SFPU_OP_CHAIN_0", "addcdiv_tile_init(); addcdiv_tile<DataFormat::Float16_b>(0, 1, 2, 3, 0x3F000000u);"}}},
    {"addcmul",
     {{"SFPU_OP_ADDCMUL_INCLUDE", "1"},
      {"SFPU_OP_CHAIN_0", "addcmul_tile_init(); addcmul_tile<DataFormat::Float16_b>(0, 1, 2, 3, 0x3F000000u);"}}},
    {"snake_beta",
     {{"SFPU_OP_SNAKE_BETA_INCLUDE", "1"},
      {"SFPU_OP_CHAIN_0", "snake_beta_tile_init(); snake_beta_tile<DataFormat::Float16_b>(0, 1, 2, 3);"}}},
};

bfloat16 sfpu_ternary_function(
    const std::string& op_name, const bfloat16& in0, const bfloat16& in1, const bfloat16& in2) {
    const double a = static_cast<float>(in0), b = static_cast<float>(in1), c = static_cast<float>(in2);
    if (op_name == "mac") {
        return bfloat16(static_cast<float>(a * b + c));
    }
    if (op_name == "lerp") {
        return bfloat16(static_cast<float>(a + c * (b - a)));
    }
    if (op_name == "addcdiv") {
        return bfloat16(static_cast<float>(a + 0.5 * b / c));
    }
    if (op_name == "addcmul") {
        return bfloat16(static_cast<float>(a + 0.5 * b * c));
    }
    if (op_name == "snake_beta") {
        const double s = std::sin(b * a);
        return bfloat16(static_cast<float>(a + s * s / c));
    }
    if (op_name == "where") {
        // Condition selects in1 when "true"; treat a near-zero condition as "false".
        constexpr float condition_epsilon = 1e-3f;
        return (std::fabs(static_cast<float>(in0)) < condition_epsilon) ? in2 : in1;
    }
    TT_THROW("Unsupported ternary op_name in test");
}

// Reference for int8 binary SFPU ops after sign-magnitude decode.
int32_t get_binary_int_operation_result(const std::string& op_name, int lhs, int rhs) {
    if (op_name == "add_int") {
        return static_cast<int32_t>(lhs + rhs);
    }
    if (op_name == "mul_int") {
        return static_cast<int32_t>(lhs * rhs);
    }
    if (op_name == "gt_int") {
        return (lhs > rhs) ? 1 : 0;
    }
    if (op_name == "binary_max_int32") {
        return std::max(lhs, rhs);
    }
    if (op_name == "binary_min_int32") {
        return std::min(lhs, rhs);
    }
    if (op_name == "copy_dest_int") {
        return static_cast<int32_t>(rhs);
    }
    if (op_name == "add_top_row_int32" || op_name == "sfpu_add_int") {
        return lhs + rhs;
    }
    if (op_name == "div_int32_floor" || op_name == "div_int32_trunc") {
        return lhs / rhs;  // non-negative operands: floor == trunc
    }
    if (op_name == "fmod_int32" || op_name == "remainder_int32") {
        return lhs % rhs;
    }
    if (op_name == "bitwise_and_binary") {
        return lhs & rhs;
    }
    if (op_name == "bitwise_or_binary") {
        return lhs | rhs;
    }
    if (op_name == "bitwise_xor_binary") {
        return lhs ^ rhs;
    }
    if (op_name == "rsub_int") {
        return rhs - lhs;
    }
    if (op_name == "int_mask") {
        return rhs == 0 ? 0 : lhs;
    }
    TT_THROW("Unsupported int8 binary op_name in test");
}

// Builds packed sign-magnitude Int32 golden for int8 binary SFPU ops.
std::vector<uint32_t> compute_packed_int8_binary_golden(
    const std::vector<uint32_t>& packed_lhs, const std::vector<uint32_t>& packed_rhs, const std::string& op_name) {
    TT_FATAL(packed_lhs.size() == packed_rhs.size(), "lhs/rhs packed size mismatch");
    std::vector<uint32_t> packed_golden(packed_lhs.size() * 4);
    for (size_t w = 0; w < packed_lhs.size(); ++w) {
        for (int b = 0; b < 4; ++b) {
            const auto lhs_byte = static_cast<uint8_t>((packed_lhs[w] >> (b * 8)) & 0xFF);
            const auto rhs_byte = static_cast<uint8_t>((packed_rhs[w] >> (b * 8)) & 0xFF);
            const int lhs = sign_mag_byte_to_int8(lhs_byte);
            const int rhs = sign_mag_byte_to_int8(rhs_byte);
            const size_t element = w * 4 + b;
            if (op_name == "div_int32") {
                packed_golden[element] = std::bit_cast<uint32_t>(static_cast<float>(lhs) / static_cast<float>(rhs));
                continue;
            }
            const bool untouched = op_name == "add_top_row_int32" && !is_add_top_row_element(element % 1024);
            const int32_t result = untouched ? lhs : get_binary_int_operation_result(op_name, lhs, rhs);
            packed_golden[element] = int32_to_sign_mag_word(result);
        }
    }
    return packed_golden;
}
vector<uint32_t> generate_packed_sfpu_input(const unsigned int numel, const std::string& op_name, const int seed) {
    const auto uniform = [&](float lo, float hi) {
        return generate_packed_uniform_random_vector<uint32_t, bfloat16>(lo, hi, numel, seed);
    };
    const auto pick = [&](vector<bfloat16> values) {
        return generate_packed_random_vector_from_vector<uint32_t, bfloat16>(values, numel, seed);
    };
    if (op_name == "isinf" || op_name == "isposinf" || op_name == "isneginf" || op_name == "isnan" ||
        op_name == "isfinite") {
        constexpr float inf = std::numeric_limits<float>::infinity();
        return pick({-inf, inf, std::numeric_limits<float>::quiet_NaN(), -2.0f, -0.5f, 0.0f, 1.0f, 3.0f});
    }
    if (op_name.starts_with("unary_")) {
        // The comparisons are against 0.5: include it and its neighbours.
        return pick({-1.0f, 0.0f, 0.25f, 0.5f, 0.75f, 1.0f, 2.0f});
    }
    if (op_name == "logical_not" || op_name == "heaviside") {
        return pick({-2.0f, -0.5f, 0.0f, 0.0f, 0.5f, 1.0f, 3.0f});
    }
    if (op_name == "erfinv") {
        return uniform(-0.95f, 0.95f);
    }
    if (op_name == "lgamma_stirling" || op_name == "digamma" || op_name == "polygamma") {
        return uniform(0.5f, 8.0f);
    }
    if (op_name == "log_with_base") {
        return uniform(0.01f, 10.0f);
    }
    if (op_name == "rdiv") {
        auto packed = uniform(-4.0f, 4.0f);
        auto values = unpack_vector<bfloat16, uint32_t>(packed);
        for (auto& v : values) {
            const float f = static_cast<float>(v);
            v = bfloat16(std::copysign(std::max(std::fabs(f), 0.25f), f));
        }
        return pack_vector<uint32_t, bfloat16>(values);
    }
    if (op_name == "threshold") {
        return uniform(0.0f, 10.0f);
    }
    if (op_name == "softcap") {
        return uniform(-20.0f, 20.0f);
    }
    if (op_name == "fmod" || op_name == "remainder") {
        return uniform(-8.0f, 8.0f);
    }
    if (op_name == "cbrt") {
        return uniform(-8.0f, 8.0f);
    }
    if (op_name == "mish") {
        // Both branches (x < 0, x >= 0) and the x >= 8 saturation.
        return uniform(-10.0f, 10.0f);
    }
    if (op_name == "rpow" || op_name == "exp2" || op_name == "hardsigmoid" || op_name == "softsign" ||
        op_name == "tanh_derivative" || op_name == "i0" || op_name == "i1") {
        return uniform(-4.0f, 4.0f);
    }
    if (op_name == "power_iterative" || op_name == "expm1") {
        return uniform(-2.0f, 2.0f);
    }
    if (op_name == "celu" || op_name == "elu" || op_name == "selu" || op_name == "softshrink" ||
        op_name == "hardshrink" || op_name == "hardtanh" || op_name == "erf" || op_name == "erfc" ||
        op_name == "identity" || op_name == "hardmish" || op_name == "prelu" || op_name == "tanhshrink" ||
        op_name == "xielu" || op_name == "power") {
        return uniform(-3.0f, 3.0f);
    }
    if (op_name == "tiled_prod") {
        // A lane's running product spans ~33 values: keep them near 1.
        return generate_packed_uniform_random_vector<uint32_t, bfloat16>(0.9f, 1.1f, numel, seed);
    }
    if ((op_name == "sqrt") || (op_name == "log") || (op_name == "rsqrt")) {
        return generate_packed_uniform_random_vector<uint32_t, bfloat16>(0.0001f, 4.0f, numel, seed);
    }
    if ((op_name == "exponential") || (op_name == "gelu") || (op_name == "reciprocal")) {
        auto possible_values = vector<bfloat16>({-1.0f, -0.5f, 0.5f, 1.0f});
        return generate_packed_random_vector_from_vector<uint32_t, bfloat16>(possible_values, numel, seed);
    }
    if (op_name == "gelu_accurate") {
        return generate_packed_uniform_random_vector<uint32_t, bfloat16>(-5.0f, 5.0f, numel, seed);
    }
    if ((op_name == "eqz") || (op_name == "nez") || (op_name == "ltz") || (op_name == "gtz") || (op_name == "gez") ||
        (op_name == "lez")) {
        // Include exact zeros so the eqz/nez/lez/gez at-zero branches are exercised.
        auto possible_values = vector<bfloat16>({-1.0f, -0.5f, 0.0f, 0.5f, 1.0f});
        return generate_packed_random_vector_from_vector<uint32_t, bfloat16>(possible_values, numel, seed);
    }
    if (op_name == "softplus") {
        return generate_packed_uniform_random_vector<uint32_t, bfloat16>(-5.0f, 30.0f, numel, seed);
    }
    if (op_name == "clamp") {
        return generate_packed_uniform_random_vector<uint32_t, bfloat16>(-2.0f, 2.0f, numel, seed);
    }
    if ((op_name == "relu_min") || (op_name == "relu_max")) {
        return generate_packed_uniform_random_vector<uint32_t, bfloat16>(-2.0f, 10.0f, numel, seed);
    }
    if ((op_name == "ceil") || (op_name == "floor") || (op_name == "trunc") || (op_name == "frac") ||
        (op_name == "round")) {
        // Half-integers distinguish floor/ceil from trunc and exercise round-half-to-even.
        auto possible_values =
            vector<bfloat16>({-2.5f, -1.5f, -0.5f, 0.0f, 0.5f, 1.5f, 2.5f, -2.0f, 2.0f, -0.25f, 3.75f});
        return generate_packed_random_vector_from_vector<uint32_t, bfloat16>(possible_values, numel, seed);
    }
    return generate_packed_uniform_random_vector<uint32_t, bfloat16>(-1.0f, 1.0f, numel, seed);
}

// Mirror the LLK `_prepare_div_inputs` helper: uniform in [-4, 4], then snap
// |x| < 0.25 to ±0.25 keeping the sign. Every element ends up in
// [-4, -0.25] ∪ [0.25, 4]. This exercises both halves of the sfpi `setsgn`
// path and avoids sub-normal divisors / spurious 0/0 -> NaN.
static vector<uint32_t> generate_div_operand(const unsigned int numel, const int seed) {
    auto packed = generate_packed_uniform_random_vector<uint32_t, bfloat16>(-4.0f, 4.0f, numel, seed);
    auto unpacked = unpack_vector<bfloat16, uint32_t>(packed);
    for (auto& v : unpacked) {
        float f = static_cast<float>(v);
        float sign = (f >= 0.0f) ? 1.0f : -1.0f;
        float magnitude = std::fabs(f);
        magnitude = std::max(magnitude, 0.25f);
        v = bfloat16(sign * magnitude);
    }
    return pack_vector<uint32_t, bfloat16>(unpacked);
}

// Per-operand stimuli for binary SFPU ops. LHS and RHS use independent seeds so
// their signs and magnitudes vary independently.
std::pair<vector<uint32_t>, vector<uint32_t>> generate_packed_sfpu_binary_inputs(
    const unsigned int numel, const std::string& op_name, const int seed) {
    if (is_non_negative_int8_binary_op(op_name)) {
        return generate_non_negative_int8_binary_inputs(numel, op_name, seed);
    }
    const auto uniform = [&](float lo, float hi, int s) {
        return generate_packed_uniform_random_vector<uint32_t, bfloat16>(lo, hi, numel, s);
    };
    const auto map_lhs = [&](const vector<uint32_t>& packed_lhs, auto&& fn) {
        auto values = unpack_vector<bfloat16, uint32_t>(packed_lhs);
        for (auto& v : values) {
            v = bfloat16(fn(static_cast<float>(v)));
        }
        return pack_vector<uint32_t, bfloat16>(values);
    };
    if (op_name == "power_binary") {
        return {uniform(0.5f, 4.0f, seed), uniform(-2.0f, 2.0f, seed + 1)};
    }
    if (op_name == "fmod_binary" || op_name == "remainder_binary") {
        auto rhs = map_lhs(
            uniform(-4.0f, 4.0f, seed + 1), [](float f) { return std::copysign(std::max(std::fabs(f), 0.5f), f); });
        return {uniform(-8.0f, 8.0f, seed), rhs};
    }
    if (op_name == "logsigmoid") {
        auto lhs = uniform(-6.0f, 6.0f, seed);
        return {lhs, map_lhs(lhs, [](float f) { return std::exp(-f); })};
    }
    if (op_name == "isclose") {
        // Half the lanes equal, half one bf16 step or more apart.
        auto lhs = uniform(-4.0f, 4.0f, seed);
        auto rhs = unpack_vector<bfloat16, uint32_t>(lhs);
        for (size_t i = 0; i < rhs.size(); i += 2) {
            rhs[i] = bfloat16(static_cast<float>(rhs[i]) * 1.01f + 0.01f);
        }
        return {lhs, pack_vector<uint32_t, bfloat16>(rhs)};
    }
    if (op_name == "clamped_silu_glu" || op_name == "situ_glu") {
        return {uniform(-12.0f, 12.0f, seed), uniform(-30.0f, 30.0f, seed + 1)};
    }
    if (op_name == "mask" || op_name == "mask_posinf") {
        auto mask_values = vector<bfloat16>({0.0f, 1.0f});
        return {
            uniform(-4.0f, 4.0f, seed),
            generate_packed_random_vector_from_vector<uint32_t, bfloat16>(mask_values, numel, seed + 1)};
    }
    if (op_name == "lgamma_stirling_float") {
        auto lhs = uniform(0.5f, 8.0f, seed);
        return {lhs, map_lhs(lhs, [](float f) { return std::log(f); })};
    }
    if (op_name == "add_top_row") {
        return {uniform(-4.0f, 4.0f, seed), uniform(-4.0f, 4.0f, seed + 1)};
    }
    if (op_name == "div_binary" || op_name == "mul_float") {
        // Reuse the div operand generator: values in [-4,-0.25] ∪ [0.25,4]. For mul this
        // is simply well-conditioned, finite bf16 input with independent lhs/rhs signs.
        auto lhs = generate_div_operand(numel, seed);
        auto rhs = generate_div_operand(numel, seed + 1);
        return {lhs, rhs};
    }
    if (op_name == "atan2") {
        return {
            generate_packed_uniform_random_vector<uint32_t, bfloat16>(-5.0f, 5.0f, numel, seed),
            generate_packed_uniform_random_vector<uint32_t, bfloat16>(-5.0f, 5.0f, numel, seed + 1)};
    }
    if (op_name == "binary_max" || op_name == "binary_min" || op_name == "copy_dest") {
        auto lhs = generate_packed_uniform_random_vector<uint32_t, bfloat16>(-4.0f, 4.0f, numel, seed);
        auto rhs = generate_packed_uniform_random_vector<uint32_t, bfloat16>(-4.0f, 4.0f, numel, seed + 1);
        return {lhs, rhs};
    }
    if (is_int8_binary_sfpu_op(op_name)) {
        auto lhs = create_random_vector_of_int8(numel, seed);
        auto rhs = create_random_vector_of_int8(numel, seed + 1);
        return {lhs, rhs};
    }
    TT_THROW("Unsupported binary op_name in test");
}

// Returns (in0, in1, in2) packed operand vectors for ternary SFPU ops.
std::tuple<vector<uint32_t>, vector<uint32_t>, vector<uint32_t>> generate_packed_sfpu_ternary_inputs(
    const unsigned int numel, const std::string& op_name, const int seed) {
    if (op_name == "mac" || op_name == "lerp" || op_name == "addcdiv" || op_name == "addcmul" ||
        op_name == "snake_beta") {
        // in2 is a divisor for addcdiv and snake_beta (beta): keep it in [0.5, 2].
        return {
            generate_packed_uniform_random_vector<uint32_t, bfloat16>(-2.0f, 2.0f, numel, seed),
            generate_packed_uniform_random_vector<uint32_t, bfloat16>(-2.0f, 2.0f, numel, seed + 1),
            generate_packed_uniform_random_vector<uint32_t, bfloat16>(0.5f, 2.0f, numel, seed + 2)};
    }
    if (op_name == "where") {
        auto possible_cond = vector<bfloat16>({-1.0f, 0.0f, 1.0f});
        auto packed_cond = generate_packed_random_vector_from_vector<uint32_t, bfloat16>(possible_cond, numel, seed);
        auto packed_true_val = generate_packed_uniform_random_vector<uint32_t, bfloat16>(-1.0f, 1.0f, numel, seed + 1);
        auto packed_false_val = generate_packed_uniform_random_vector<uint32_t, bfloat16>(-1.0f, 1.0f, numel, seed + 2);
        return {packed_cond, packed_true_val, packed_false_val};
    }
    TT_THROW("Unsupported ternary op_name in test");
}

// Per-op (rtol, atol) for the device-vs-golden comparison. Shared by the bf16
// and Float32 close-checks so the tolerances live in one place. Defaults match
// is_close()'s own defaults for the "everything else" bucket.
std::pair<float, float> sfpu_tolerance(const std::string& op_name, bool fp32_dest = false) {
    if (op_name == "lgamma_stirling_float") {
        // The kernel's Stirling series (four Bernoulli terms) is up to ~0.053 off for z just above 0.5,
        // outside its z ~ 1 and z ~ 2 Taylor windows. The tt-llk test gates it with torch.isclose at
        // rtol = atol = 0.05, i.e. atol + rtol * |golden| ~ 0.08 there; is_close takes the larger of
        // the two bounds instead, so atol carries that whole margin.
        return {0.05f, 0.08f};
    }
    if (op_name == "erfinv" || op_name == "lgamma_stirling" || op_name == "digamma" || op_name == "polygamma" ||
        op_name == "softcap" || op_name == "i0" || op_name == "i1" || op_name == "log_with_base") {
        return {0.06f, 0.02f};
    }
    if (op_name == "tanh") {
        return {0.175f, 0.1f};
    }
    if ((op_name == "gelu") || (op_name == "relu")) {
        return {0.15f, 0.001f};
    }
    if (op_name == "gelu_accurate") {
        return {0.01f, 0.001f};
    }
    if (op_name == "exponential") {
        // 16-bit Dest runs the approximate (HW LUT) exp; 32-bit Dest runs the fp32-accurate
        // path, so hold it to a much tighter tolerance.
        return fp32_dest ? std::pair<float, float>{0.02f, 0.01f} : std::pair<float, float>{0.1f, 0.1f};
    }
    if (op_name == "log") {
        return {0.03f, 0.02f};
    }
    if (op_name == "softplus") {
        return {0.06f, 0.02f};
    }
    return {0.06f, 0.006f};
}

bool is_close_packed_sfpu_output_f32(
    const std::vector<uint32_t>& vec_a, const std::vector<uint32_t>& vec_b, const std::string& op_name);

bool is_close_packed_sfpu_output(
    const std::vector<uint32_t>& vec_a, const std::vector<uint32_t>& vec_b, const std::string& op_name) {
    if (op_name == "div_int32") {
        return is_close_packed_sfpu_output_f32(vec_a, vec_b, op_name);
    }
    if (is_int8_binary_sfpu_op(op_name) || op_name == "binary_max" || op_name == "binary_min" ||
        op_name == "copy_dest" || op_name == "mask" || op_name == "mask_posinf" || op_name == "isclose") {
        return vec_a == vec_b;
    }
    if (is_exact_unary_sfpu_op(op_name)) {
        return is_close_packed_vectors<bfloat16, uint32_t>(
            vec_a, vec_b, [](const bfloat16& a, const bfloat16& b) { return a == b; });
    }
    if (op_name == "where") {
        // Matches the LLK pytest's torch.isclose(rtol=0.05, atol=0.05) for
        // Float16 / Float16_b / Float32 (helpers/utils.py:tolerances).
        return is_close_packed_vectors<bfloat16, uint32_t>(
            vec_a, vec_b, [](const bfloat16& a, const bfloat16& b) { return is_close(a, b, 0.05f, 0.05f); });
    }
    const auto [rtol, atol] = sfpu_tolerance(op_name);
    return is_close_packed_vectors<bfloat16, uint32_t>(
        vec_a, vec_b, [&](const bfloat16& a, const bfloat16& b) { return is_close(a, b, rtol, atol); });
}

// Float32 close-check. Shares sfpu_function() / sfpu_tolerance() / the
// uniform-random input generator with the bf16 path; only this comparison stays
// separate because Float32 has 1:1 element-to-word packing and unpack_vector
// requires sizeof(PackType) > sizeof(ValueType), so the bf16-oriented
// is_close_packed_vectors helper won't instantiate. bit_cast per element
// instead, matching test_transpose.cpp:404-410.
bool is_close_packed_sfpu_output_f32(
    const std::vector<uint32_t>& vec_a, const std::vector<uint32_t>& vec_b, const std::string& op_name) {
    if (vec_a.size() != vec_b.size()) {
        return false;
    }
    const auto [rtol, atol] = sfpu_tolerance(op_name, /*fp32_dest=*/true);
    for (size_t i = 0; i < vec_a.size(); ++i) {
        const float a = std::bit_cast<float>(vec_a[i]);
        const float b = std::bit_cast<float>(vec_b[i]);
        if (!is_close(a, b, rtol, atol)) {
            return false;
        }
    }
    return true;
}

// ---- Typecast (data-conversion) test helpers ----------------------------------------------------
//
// Pack each endpoint format, decode it back, and build a host golden so the metal test exercises each
// conversion symmetrically (mirrors the tt-llk typecast test).

inline bool typecast_is_mx(tt::DataFormat fmt) {
    return fmt == tt::DataFormat::MxFp8R || fmt == tt::DataFormat::MxFp8P;
}

// Integer endpoints whose value compares exactly (and that the SFPU round-to-nearest produces).
inline bool typecast_is_int(tt::DataFormat fmt) {
    return fmt == tt::DataFormat::Int32 || fmt == tt::DataFormat::Int16 || fmt == tt::DataFormat::UInt8;
}

inline bool typecast_is_unsigned(tt::DataFormat fmt) { return fmt == tt::DataFormat::UInt8; }

// Device-side ckernel::DataFormat enum NAME (the kernel chain references formats by name, since the
// compute kernel compiles against the device enum whose values differ from the host tt::DataFormat).
inline std::string typecast_device_format_name(tt::DataFormat fmt) {
    switch (fmt) {
        case tt::DataFormat::Float16_b: return "Float16_b";
        case tt::DataFormat::Float32: return "Float32";
        case tt::DataFormat::Int32: return "Int32";
        case tt::DataFormat::Int16: return "Int16";
        case tt::DataFormat::UInt8: return "UInt8";
        case tt::DataFormat::MxFp8R: return "MxFp8R";
        case tt::DataFormat::MxFp8P: return "MxFp8P";
        default: TT_THROW("typecast test: unsupported format {}", static_cast<int>(fmt));
    }
}

// Pack tile-ordered whole-number floats into `fmt`'s on-device L1 tile encoding.
inline std::vector<uint32_t> typecast_pack(tt::DataFormat fmt, const std::vector<float>& vals) {
    switch (fmt) {
        case tt::DataFormat::Float16_b: {
            std::vector<bfloat16> bf(vals.begin(), vals.end());
            return pack_vector<uint32_t, bfloat16>(bf);
        }
        case tt::DataFormat::Float32: {
            std::vector<uint32_t> out;
            out.reserve(vals.size());
            for (const float v : vals) {
                out.push_back(float32(v).to_packed());
            }
            return out;
        }
        case tt::DataFormat::Int32: {
            // Quasar Int32 in L1 is sign-magnitude (same encoding as the int8->int32 binary path).
            std::vector<uint32_t> out;
            out.reserve(vals.size());
            for (const float v : vals) {
                out.push_back(int32_to_sign_mag_word(static_cast<int32_t>(std::lround(v))));
            }
            return out;
        }
        case tt::DataFormat::Int16: {
            // Quasar Int16 in L1 is sign-magnitude 16-bit (SMAG16): sign in bit 15, 14-bit magnitude;
            // two elements packed per 32-bit word (element 2i in the low half, 2i+1 in the high half).
            std::vector<uint32_t> out((vals.size() + 1) / 2, 0);
            for (size_t i = 0; i < vals.size(); ++i) {
                const int32_t s = static_cast<int32_t>(std::lround(vals[i]));
                const uint16_t enc =
                    static_cast<uint16_t>((s < 0 ? 0x8000u : 0u) | (static_cast<uint32_t>(std::abs(s)) & 0x7fffu));
                out[i / 2] |= static_cast<uint32_t>(enc) << (16 * (i % 2));
            }
            return out;
        }
        case tt::DataFormat::UInt8: {
            // UInt8 in L1 is raw unsigned bytes (four per 32-bit word).
            std::vector<uint32_t> out((vals.size() + 3) / 4, 0);
            for (size_t i = 0; i < vals.size(); ++i) {
                const int32_t s = std::clamp<int32_t>(static_cast<int32_t>(std::lround(vals[i])), 0, 255);
                out[i / 4] |= static_cast<uint32_t>(s & 0xff) << (8 * (i % 4));
            }
            return out;
        }
        case tt::DataFormat::MxFp8R:
        case tt::DataFormat::MxFp8P: return pack_as_mx_tiles(fmt, vals, /*row_major_input=*/false);
        default: TT_THROW("typecast test: unsupported pack format {}", static_cast<int>(fmt));
    }
}

// Decode `fmt`'s L1 tile bytes back into tile-ordered floats (inverse of typecast_pack).
inline std::vector<float> typecast_decode(tt::DataFormat fmt, const std::vector<uint32_t>& bytes) {
    switch (fmt) {
        case tt::DataFormat::Float16_b: {
            auto bf = unpack_vector<bfloat16, uint32_t>(bytes);
            std::vector<float> out(bf.size());
            for (size_t i = 0; i < bf.size(); ++i) {
                out[i] = static_cast<float>(bf[i]);
            }
            return out;
        }
        case tt::DataFormat::Float32: {
            std::vector<float> out(bytes.size());
            for (size_t i = 0; i < bytes.size(); ++i) {
                out[i] = float32(bytes[i]).to_float();
            }
            return out;
        }
        case tt::DataFormat::Int32: {
            std::vector<float> out(bytes.size());
            for (size_t i = 0; i < bytes.size(); ++i) {
                const uint32_t w = bytes[i];
                const int32_t mag = static_cast<int32_t>(w & 0x7fffffffu);
                out[i] = static_cast<float>((w & 0x80000000u) ? -mag : mag);
            }
            return out;
        }
        case tt::DataFormat::Int16: {
            std::vector<float> out(bytes.size() * 2);
            for (size_t i = 0; i < out.size(); ++i) {
                const uint16_t h = static_cast<uint16_t>(bytes[i / 2] >> (16 * (i % 2)));
                const int32_t mag = h & 0x7fff;
                out[i] = static_cast<float>((h & 0x8000) ? -mag : mag);
            }
            return out;
        }
        case tt::DataFormat::UInt8: {
            std::vector<float> out(bytes.size() * 4);
            for (size_t i = 0; i < out.size(); ++i) {
                out[i] = static_cast<float>((bytes[i / 4] >> (8 * (i % 4))) & 0xffu);
            }
            return out;
        }
        case tt::DataFormat::MxFp8R:
        case tt::DataFormat::MxFp8P: return mx_to_floats(fmt, bytes, /*row_major_output=*/false);
        default: TT_THROW("typecast test: unsupported decode format {}", static_cast<int>(fmt));
    }
}

// Whole-number stimulus, range chosen so every endpoint represents it losslessly: small magnitudes
// for MX (block-float steps), non-negative for an unsigned endpoint (UInt8), wider signed otherwise.
inline std::vector<float> generate_typecast_input(
    size_t numel, int seed, tt::DataFormat in_fmt, tt::DataFormat out_fmt) {
    const bool mx = typecast_is_mx(in_fmt) || typecast_is_mx(out_fmt);
    const bool has_unsigned = typecast_is_unsigned(in_fmt) || typecast_is_unsigned(out_fmt);
    const float lo = (mx || has_unsigned) ? 0.0f : -64.0f;
    float hi = 64.0f;
    if (mx) {
        hi = 8.0f;
    } else if (has_unsigned) {
        hi = 32.0f;
    }
    auto packed = generate_packed_uniform_random_vector<uint32_t, bfloat16>(lo, hi, numel, seed);
    auto bf = unpack_vector<bfloat16, uint32_t>(packed);
    std::vector<float> out(bf.size());
    for (size_t i = 0; i < bf.size(); ++i) {
        out[i] = std::round(static_cast<float>(bf[i]));
    }
    return out;
}

// Expected tile-ordered output values of a SRC->DST typecast of `vals` (packed_in is its SRC bytes).
inline std::vector<float> typecast_golden(
    tt::DataFormat in_fmt, tt::DataFormat out_fmt, const std::vector<uint32_t>& packed_in) {
    // Effective source = what the SRC encoding actually represents (after any input quantization).
    std::vector<float> golden = typecast_decode(in_fmt, packed_in);
    if (typecast_is_int(out_fmt)) {
        for (auto& v : golden) {
            v = static_cast<float>(std::lround(v));  // float->int rounds to nearest
            if (typecast_is_unsigned(out_fmt)) {
                v = std::clamp(v, 0.0f, 255.0f);
            }
        }
    }
    // Fold in the destination encoding's quantization so the golden matches what the packer emits.
    if (typecast_is_mx(out_fmt) || out_fmt == tt::DataFormat::Float16_b) {
        golden = typecast_decode(out_fmt, typecast_pack(out_fmt, golden));
    }
    return golden;
}

inline bool typecast_compare(tt::DataFormat out_fmt, const std::vector<float>& got, const std::vector<float>& want) {
    if (got.size() != want.size()) {
        return false;
    }
    const bool exact = typecast_is_int(out_fmt);
    const float rtol = typecast_is_mx(out_fmt) ? 0.1f : 0.05f;
    const float atol = typecast_is_mx(out_fmt) ? 0.1f : 0.05f;
    for (size_t i = 0; i < got.size(); ++i) {
        if (exact) {
            if (got[i] != want[i]) {
                return false;
            }
        } else if (std::fabs(got[i] - want[i]) > atol + rtol * std::fabs(want[i])) {
            return false;
        }
    }
    return true;
}

}  // namespace unit_tests::sfpu_util

namespace unit_tests::compute::sfpu {

struct SfpuConfig {
    size_t num_tiles = 0;
    size_t tile_byte_size = 0;
    tt::DataFormat l1_input_data_format = tt::DataFormat::Invalid;
    tt::DataFormat l1_output_data_format = tt::DataFormat::Invalid;
    CoreRangeSet cores;
    std::string sfpu_op;
    bool approx_mode = true;
    bool dst_full_sync_en = true;  // SyncFull by default (matches today's implicit behavior)
    bool unpack_to_dest =
        false;  // route input DFB to Dest (unpack_modes=UnpackToDest); pair with en_32bit_dest for 32-bit Dest
    bool en_32bit_dest = false;
};

// Builds a DataflowBufferSpec. `entry_size` is derived from `fmt` so that input and output
// DFBs are correctly sized even when their formats differ (e.g. Int8 in → Int32 out).
experimental::DataflowBufferSpec make_dfb_spec(
    const experimental::DFBSpecName& id, const SfpuConfig& cfg, tt::DataFormat fmt) {
    return {
        .unique_id = id,
        .entry_size = static_cast<uint32_t>(tt::tile_size(fmt)),
        .num_entries = static_cast<uint32_t>(cfg.num_tiles),
        .data_format_metadata = fmt,
    };
}

// Converts a string→string defines map to the CompilerOptions::Defines vector form.
experimental::KernelSpec::CompilerOptions::Defines to_kernel_defines(const std::map<std::string, std::string>& m) {
    experimental::KernelSpec::CompilerOptions::Defines defines;
    for (const auto& [k, v] : m) {
        defines.emplace(k, v);
    }
    return defines;
}

/// Builds and runs the single-input SFPU pipeline on one core and returns the raw DST bytes:
///
///   DRAM(in) -> reader_unary -> in DFB(in_fmt) -> eltwise_sfpu(`defines`) -> out DFB(out_fmt) -> writer_unary -> DRAM
///
/// Generic over the (input, output) data formats in `cfg`, so scalar-math unary ops (in == out) and
/// data-conversion ops like typecast (in != out, differing tile widths) share one harness. Cross-arch:
/// keeps the Gen1+Gen2 data-movement config and the MeshWorkload dispatch path. Callers supply the
/// compute `defines` (op selection / chain) and the packed SRC bytes, and verify the returned DST bytes.
std::vector<uint32_t> run_sfpu_pipeline(
    distributed::MeshDevice& mesh_device,
    const SfpuConfig& test_config,
    const std::map<std::string, std::string>& defines,
    const std::vector<uint32_t>& packed_input) {
    auto& cq = mesh_device.mesh_command_queue();
    const size_t in_bytes = test_config.num_tiles * tt::tile_size(test_config.l1_input_data_format);
    const size_t out_bytes = test_config.num_tiles * tt::tile_size(test_config.l1_output_data_format);

    auto input_dram_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = in_bytes},
        {.page_size = in_bytes, .buffer_type = BufferType::DRAM},
        &mesh_device);
    auto output_dram_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = out_bytes},
        {.page_size = out_bytes, .buffer_type = BufferType::DRAM},
        &mesh_device);

    // Every parametrization of these tests uses a single-core CoreRangeSet of {0, 0};
    // MakeProgramFromSpec models the kernel set per single-core WorkUnit.
    TT_FATAL(
        test_config.cores.ranges().size() == 1,
        "sfpu test expects a single CoreRange (got {})",
        test_config.cores.size());
    const CoreRange& core_range = *test_config.cores.ranges().begin();
    TT_FATAL(core_range.start_coord == core_range.end_coord, "sfpu test expects a single-core CoreRange");
    const CoreCoord core = core_range.start_coord;
    const experimental::NodeCoord node{core.x, core.y};

    const experimental::DFBSpecName IN_DFB{"in_dfb"};
    const experimental::DFBSpecName OUT_DFB{"out_dfb"};
    const experimental::KernelSpecName READER{"reader"};
    const experimental::KernelSpecName WRITER{"writer"};
    const experimental::KernelSpecName COMPUTE{"compute"};

    const experimental::DataflowBufferSpec in_dfb_spec =
        make_dfb_spec(IN_DFB, test_config, test_config.l1_input_data_format);
    const experimental::DataflowBufferSpec out_dfb_spec =
        make_dfb_spec(OUT_DFB, test_config, test_config.l1_output_data_format);

    experimental::DataMovementHardwareConfig reader_hw_config;
    if (mesh_device.arch() == tt::ARCH::QUASAR) {
        reader_hw_config = experimental::DataMovementHardwareConfig{
            .config_2xx =
                experimental::DataMovementHardwareConfig::DataMovement2XXConfig{
                    .disable_dfb_implicit_sync_for_all = true,
                },
        };
    } else {
        reader_hw_config = experimental::DataMovementHardwareConfig{
            .config_1xx =
                experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_1,
                    .noc = tt_metal::NOC::RISCV_1_default,
                },
        };
    }

    experimental::KernelSpec reader_spec{
        .unique_id = READER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {{
            .dfb_spec_name = IN_DFB,
            .accessor_name = "out",
            .endpoint_type = experimental::DFBEndpointType::PRODUCER,
            .access_pattern = experimental::DFBAccessPattern::STRIDED,
        }},
        .runtime_arg_schema = {.runtime_arg_names = {"src_addr", "bank_id", "num_tiles"}},
        .hw_config = reader_hw_config,
    };

    experimental::DataMovementHardwareConfig writer_hw_config;
    if (mesh_device.arch() == tt::ARCH::QUASAR) {
        writer_hw_config = experimental::DataMovementHardwareConfig{
            .config_2xx =
                experimental::DataMovementHardwareConfig::DataMovement2XXConfig{
                    .disable_dfb_implicit_sync_for_all = true,
                },
        };
    } else {
        writer_hw_config = experimental::DataMovementHardwareConfig{
            .config_1xx =
                experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_0,
                    .noc = tt_metal::NOC::RISCV_0_default,
                },
        };
    }

    experimental::KernelSpec writer_spec{
        .unique_id = WRITER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {{
            .dfb_spec_name = OUT_DFB,
            .accessor_name = "in",
            .endpoint_type = experimental::DFBEndpointType::CONSUMER,
            .access_pattern = experimental::DFBAccessPattern::STRIDED,
        }},
        .runtime_arg_schema = {.runtime_arg_names = {"dst_addr", "bank_id", "num_tiles"}},
        .hw_config = writer_hw_config,
    };

    experimental::ComputeHardwareConfig compute_hw_config;
    experimental::ComputeHardwareConfig::ComputeUnpackModes unpack_modes{};
    if (test_config.unpack_to_dest) {
        unpack_modes = {{IN_DFB, tt::tt_metal::UnpackMode::UnpackToDest}};
    }
    const bool fp32_dest_acc_en = test_config.en_32bit_dest;
    compute_hw_config = experimental::ComputeHardwareConfig{
        .sfpu_precision_mode =
            test_config.approx_mode ? tt::tt_metal::Precision::Approximate : tt::tt_metal::Precision::Precise,
        .enable_32_bit_dest = fp32_dest_acc_en,
        .double_buffer_dest = !test_config.dst_full_sync_en,
        .unpack_modes = unpack_modes,
    };

    experimental::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_sfpu_2_0.cpp",
        .num_threads = 1,
        .compiler_options = {.defines = to_kernel_defines(defines)},
        .dfb_bindings =
            {{
                 .dfb_spec_name = IN_DFB,
                 .accessor_name = "in",
                 .endpoint_type = experimental::DFBEndpointType::CONSUMER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = OUT_DFB,
                 .accessor_name = "out",
                 .endpoint_type = experimental::DFBEndpointType::PRODUCER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             }},
        .compile_time_args =
            {{"per_core_block_cnt", static_cast<uint32_t>(test_config.num_tiles)}, {"per_core_block_size", 1u}},
        .hw_config = compute_hw_config,
    };

    experimental::WorkUnitSpec wu{
        .name = "main",
        .kernels = {READER, WRITER, COMPUTE},
        .target_nodes = node,
    };

    experimental::ProgramSpec spec{
        .name = "sfpu_compute",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers = {in_dfb_spec, out_dfb_spec},
        .work_units = {wu},
    };

    Program program = experimental::MakeProgramFromSpec(mesh_device, spec);

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = READER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node,
                {{"src_addr", input_dram_buffer->address()},
                 {"bank_id", 0u},
                 {"num_tiles", static_cast<uint32_t>(test_config.num_tiles)}}),
        },
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = WRITER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node,
                {{"dst_addr", output_dram_buffer->address()},
                 {"bank_id", 0u},
                 {"num_tiles", static_cast<uint32_t>(test_config.num_tiles)}}),
        },
        experimental::ProgramRunArgs::KernelRunArgs{.kernel = COMPUTE},
    };
    experimental::SetProgramRunArgs(program, params);

    distributed::EnqueueWriteMeshBuffer(cq, input_dram_buffer, packed_input, /*blocking=*/true);
    LaunchProgram(mesh_device, std::move(program));

    std::vector<uint32_t> dest_buffer_data;
    distributed::EnqueueReadMeshBuffer(cq, dest_buffer_data, output_dram_buffer, /*blocking=*/true);
    return dest_buffer_data;
}

/// @brief Does Dram --> Reader --> CB --> Sfpu Compute --> CB --> Writer --> Dram. So far, enqueue APIs only added to
/// grayskull
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool run_sfpu_all_same_buffer(distributed::MeshDevice& mesh_device, const SfpuConfig& test_config) {
    const size_t byte_size = test_config.num_tiles * test_config.tile_byte_size;

    // Input
    const bool is_fp32 = (test_config.l1_input_data_format == tt::DataFormat::Float32);
    const bool is_relu_family =
        test_config.sfpu_op == "relu" || test_config.sfpu_op == "relu_min" || test_config.sfpu_op == "relu_max";
    TT_FATAL(!is_fp32 || is_relu_family, "Float32 SFPU path supports relu / relu_min / relu_max only in v1");
    const size_t element_size = is_fp32 ? sizeof(float) : sizeof(bfloat16);
    const size_t numel = byte_size / element_size;
    const auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    const bool relu_threshold_op = test_config.sfpu_op == "relu_min" || test_config.sfpu_op == "relu_max";
    std::vector<uint32_t> packed_input =
        is_fp32 ? generate_packed_uniform_random_vector<uint32_t, float>(
                      relu_threshold_op ? -2.0f : -1.0f, relu_threshold_op ? 10.0f : 1.0f, numel, seed)
                : sfpu_util::generate_packed_sfpu_input(numel, test_config.sfpu_op, seed);

    // Golden output
    std::vector<uint32_t> packed_golden;
    if (is_fp32) {
        // 1:1 element-to-word; bit_cast in/out per-element.
        packed_golden.resize(packed_input.size());
        for (size_t i = 0; i < packed_input.size(); ++i) {
            const float in = std::bit_cast<float>(packed_input[i]);
            packed_golden[i] = std::bit_cast<uint32_t>(sfpu_util::sfpu_function(test_config.sfpu_op, in));
        }
    } else {
        auto input = unpack_vector<bfloat16, uint32_t>(packed_input);
        std::vector<bfloat16> golden(input.size());
        if (sfpu_util::is_tile_layout_sfpu_op(test_config.sfpu_op)) {
            // Dest holds bf16 for a 16-bit Dest; with a 32-bit Dest the partial products stay fp32.
            const auto store = [&](float v) { return test_config.en_32bit_dest ? v : static_cast<float>(bfloat16(v)); };
            for (size_t base = 0; base < input.size(); base += 1024) {
                std::vector<float> tile(input.begin() + base, input.begin() + base + 1024);
                const auto out = sfpu_util::tile_layout_golden(test_config.sfpu_op, tile, store);
                std::transform(out.begin(), out.end(), golden.begin() + base, [](float v) { return bfloat16(v); });
            }
        } else {
            std::transform(input.begin(), input.end(), golden.begin(), [&](const bfloat16& val) {
                return sfpu_util::sfpu_function(test_config.sfpu_op, val);
            });
        }
        packed_golden = pack_vector<uint32_t, bfloat16>(golden);
    }

    std::map<std::string, std::string> sfpu_defines = sfpu_util::sfpu_op_to_op_name.at(test_config.sfpu_op);
    sfpu_defines["SFPU_UNARY_OP"] = "1";
    // For chains that take the approximation mode as a template argument (mish_tile): APPROX is
    // declared only on the math TRISC, so name the fixture's mode as a literal instead.
    sfpu_defines["SFPU_OP_APPROX"] = test_config.approx_mode ? "true" : "false";
    sfpu_defines["SFPU_OP_EXP_INCLUDE"] = "1";
    sfpu_defines["SFPU_OP_GELU_INCLUDE"] = "1";
    sfpu_defines["SFPU_OP_RECIP_INCLUDE"] = "1";
    sfpu_defines["SFPU_OP_SQRT_INCLUDE"] = "1";
    sfpu_defines["SFPU_OP_RSQRT_INCLUDE"] = "1";
    sfpu_defines["SFPU_OP_ERF_ERFC_INCLUDE"] = "1";
    sfpu_defines["SFPU_OP_ELU_INCLUDE"] = "1";
    sfpu_defines["SFPU_OP_NEG_INCLUDE"] = "1";
    sfpu_defines["SFPU_OP_SOFTPLUS_INCLUDE"] = "1";
    sfpu_defines["SFPU_OP_CLAMP_INCLUDE"] = "1";
    sfpu_defines["SFPU_OP_RELU_FAMILY_INCLUDE"] = "1";
    sfpu_defines["SFPU_OP_ROUND_FAMILY_INCLUDE"] = "1";
    sfpu_defines["SFPU_OP_COMPUTE_KERNEL_API_INCLUDE"] = "1";
    sfpu_defines["SFPU_OP_BINOP_WITH_SCALAR_INCLUDE"] = "1";
    sfpu_defines["SFPU_OP_UNARY_COMP_INCLUDE"] = "1";

    const auto dest_buffer_data = run_sfpu_pipeline(mesh_device, test_config, sfpu_defines, packed_input);
    return is_fp32 ? sfpu_util::is_close_packed_sfpu_output_f32(dest_buffer_data, packed_golden, test_config.sfpu_op)
                   : sfpu_util::is_close_packed_sfpu_output(dest_buffer_data, packed_golden, test_config.sfpu_op);
}

/// Int32 unary SFPU ops: Int8 L1 (non-negative) -> 32-bit Dest -> Int32 L1, compared exactly.
bool run_sfpu_int32_unary(distributed::MeshDevice& mesh_device, const SfpuConfig& test_config) {
    const size_t numel = test_config.num_tiles * 1024;
    const auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    const auto packed_input =
        sfpu_util::generate_non_negative_int8_binary_inputs(numel, test_config.sfpu_op, seed).first;

    std::vector<int32_t> values(numel);
    for (size_t i = 0; i < numel; ++i) {
        values[i] = sign_mag_byte_to_int8(static_cast<uint8_t>(packed_input[i / 4] >> (8 * (i % 4))));
    }
    std::vector<int32_t> expected(numel);
    if (test_config.sfpu_op == "sum_int_col" || test_config.sfpu_op == "sum_int_row") {
        for (size_t base = 0; base < numel; base += 1024) {
            const std::vector<int32_t> tile(values.begin() + base, values.begin() + base + 1024);
            const auto out = sfpu_util::int_tile_layout_golden(test_config.sfpu_op, tile);
            std::copy(out.begin(), out.end(), expected.begin() + base);
        }
    } else {
        std::transform(values.begin(), values.end(), expected.begin(), [&](int32_t x) {
            return sfpu_util::int32_unary_result(test_config.sfpu_op, x);
        });
    }
    std::vector<uint32_t> packed_golden(numel);
    std::transform(expected.begin(), expected.end(), packed_golden.begin(), int32_to_sign_mag_word);

    std::map<std::string, std::string> sfpu_defines = sfpu_util::sfpu_int32_unary_op_to_op_name.at(test_config.sfpu_op);
    sfpu_defines["SFPU_UNARY_OP"] = "1";
    const auto dest_buffer_data = run_sfpu_pipeline(mesh_device, test_config, sfpu_defines, packed_input);
    return dest_buffer_data == packed_golden;
}

namespace {

// Validates that cfg describes a single-core CoreRange and returns the Quasar NodeCoord.
experimental::NodeCoord extract_single_core_node(const SfpuConfig& cfg, const char* context) {
    TT_FATAL(cfg.cores.ranges().size() == 1, "{} expects a single CoreRange (got {})", context, cfg.cores.size());
    const CoreRange& cr = *cfg.cores.ranges().begin();
    TT_FATAL(cr.start_coord == cr.end_coord, "{} expects a single-core CoreRange", context);
    return {cr.start_coord.x, cr.start_coord.y};
}

// Builds a writer_unary KernelSpec bound to a single output DFB.
experimental::KernelSpec make_writer_unary_quasar_spec(
    const experimental::KernelSpecName& kernel_id, const experimental::DFBSpecName& out_dfb_id) {
    return {
        .unique_id = kernel_id,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {{
            .dfb_spec_name = out_dfb_id,
            .accessor_name = "in",
            .endpoint_type = experimental::DFBEndpointType::CONSUMER,
            .access_pattern = experimental::DFBAccessPattern::STRIDED,
        }},
        .runtime_arg_schema = {.runtime_arg_names = {"dst_addr", "bank_id", "num_tiles"}},
        .hw_config =
            experimental::DataMovementHardwareConfig{
                .config_2xx =
                    experimental::DataMovementHardwareConfig::DataMovement2XXConfig{
                        .disable_dfb_implicit_sync_for_all = true,
                    },
            },
    };
}

// Creates a writer_unary kernel on the legacy (non-Quasar) path.
tt_metal::KernelHandle create_legacy_writer_kernel(tt_metal::Program& program, const SfpuConfig& cfg) {
    return tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary.cpp",
        cfg.cores,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});
}

// Compiles `spec`, sets `params`, uploads each input buffer, launches on the device, and
// returns the raw output tile data. Shared by the binary and ternary Quasar SFPU runners.
std::vector<uint32_t> sfpu_quasar_run(
    distributed::MeshDevice& mesh_device,
    const experimental::ProgramSpec& spec,
    const experimental::ProgramRunArgs& params,
    const std::vector<std::pair<std::shared_ptr<distributed::MeshBuffer>, const std::vector<uint32_t>*>>& inputs,
    const std::shared_ptr<distributed::MeshBuffer>& out_buf) {
    auto program = experimental::MakeProgramFromSpec(mesh_device, spec);
    experimental::SetProgramRunArgs(program, params);
    for (const auto& [buf, data] : inputs) {
        slow_dispatch::WriteToBuffer(*buf, *data);
    }
    LaunchProgram(mesh_device, std::move(program));
    std::vector<uint32_t> dest;
    slow_dispatch::ReadFromBuffer(*out_buf, dest);
    return dest;
}

}  // namespace

/// High-level flow:
///
///   DRAM(LHS) ─┐
///              ├─> Reader ─> in0/in1 DFB ─> SFPU Compute (eltwise_sfpu_2_0.cpp, SFPU_BINARY_OP) ─> out DFB ─> Writer
///              ─> DRAM(out)
///   DRAM(RHS) ─┘
///
/// @param mesh_device Device under test.
/// @param test_config - Configuration of the test -- see struct
/// @return
bool run_sfpu_binary_two_input_buffer(distributed::MeshDevice& mesh_device, const SfpuConfig& test_config) {
    const size_t per_buffer_byte_size_input = test_config.num_tiles * tt::tile_size(test_config.l1_input_data_format);
    const size_t per_buffer_byte_size_output = test_config.num_tiles * tt::tile_size(test_config.l1_output_data_format);

    auto input0_dram_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = per_buffer_byte_size_input},
        {.page_size = per_buffer_byte_size_input, .buffer_type = tt::tt_metal::BufferType::DRAM},
        &mesh_device);
    auto input1_dram_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = per_buffer_byte_size_input},
        {.page_size = per_buffer_byte_size_input, .buffer_type = tt::tt_metal::BufferType::DRAM},
        &mesh_device);
    auto output_dram_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = per_buffer_byte_size_output},
        {.page_size = per_buffer_byte_size_output, .buffer_type = tt::tt_metal::BufferType::DRAM},
        &mesh_device);

    const bool is_int8_op = sfpu_util::is_int8_binary_sfpu_op(test_config.sfpu_op);
    const size_t element_size = is_int8_op ? sizeof(int8_t) : sizeof(bfloat16);
    const uint32_t numel = per_buffer_byte_size_input / element_size;
    const int seed = std::chrono::system_clock::now().time_since_epoch().count();
    auto [packed_lhs, packed_rhs] = sfpu_util::generate_packed_sfpu_binary_inputs(numel, test_config.sfpu_op, seed);

    std::vector<uint32_t> packed_golden;
    if (is_int8_op) {
        packed_golden = sfpu_util::compute_packed_int8_binary_golden(packed_lhs, packed_rhs, test_config.sfpu_op);
    } else {
        auto lhs = unpack_vector<bfloat16, uint32_t>(packed_lhs);
        auto rhs = unpack_vector<bfloat16, uint32_t>(packed_rhs);
        std::vector<bfloat16> golden(lhs.size());
        std::transform(lhs.begin(), lhs.end(), rhs.begin(), golden.begin(), [&](const bfloat16& a, const bfloat16& b) {
            return sfpu_util::sfpu_binary_function(test_config.sfpu_op, a, b);
        });
        if (test_config.sfpu_op == "add_top_row") {
            // Only the top rows are written; the rest of DST[0] keeps the LHS tile.
            for (size_t i = 0; i < golden.size(); ++i) {
                if (!sfpu_util::is_add_top_row_element(i % 1024)) {
                    golden[i] = lhs[i];
                }
            }
        }
        packed_golden = pack_vector<uint32_t, bfloat16>(golden);
    }

    std::map<std::string, std::string> sfpu_defines = sfpu_util::sfpu_binary_op_to_op_name.at(test_config.sfpu_op);
    sfpu_defines["SFPU_BINARY_OP"] = "1";

    const auto node = extract_single_core_node(test_config, "Metal 2.0 binary SFPU path");
    const experimental::DFBSpecName IN0_DFB{"in0_dfb"};
    const experimental::DFBSpecName IN1_DFB{"in1_dfb"};
    const experimental::DFBSpecName OUT_DFB{"out_dfb"};
    const experimental::KernelSpecName READER{"reader"};
    const experimental::KernelSpecName WRITER{"writer"};
    const experimental::KernelSpecName COMPUTE{"compute"};

    experimental::KernelSpec reader_spec{
        .unique_id = READER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_binary_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings =
            {{
                 .dfb_spec_name = IN0_DFB,
                 .accessor_name = "in0",
                 .endpoint_type = experimental::DFBEndpointType::PRODUCER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = IN1_DFB,
                 .accessor_name = "in1",
                 .endpoint_type = experimental::DFBEndpointType::PRODUCER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             }},
        .runtime_arg_schema =
            {.runtime_arg_names = {"src0_addr", "src0_bank_id", "src1_addr", "src1_bank_id", "num_tiles"}},
        .hw_config =
            experimental::DataMovementHardwareConfig{
                .config_2xx =
                    experimental::DataMovementHardwareConfig::DataMovement2XXConfig{
                        .disable_dfb_implicit_sync_for_all = true,
                    },
            },
    };

    experimental::ComputeHardwareConfig compute_hw_config;
    compute_hw_config = experimental::ComputeHardwareConfig{
        .sfpu_precision_mode =
            test_config.approx_mode ? tt::tt_metal::Precision::Approximate : tt::tt_metal::Precision::Precise,
        .enable_32_bit_dest = is_int8_op || test_config.sfpu_op == "add_top_row",
    };

    experimental::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_sfpu_2_0.cpp",
        .num_threads = 1,
        .compiler_options = {.defines = to_kernel_defines(sfpu_defines)},
        .dfb_bindings =
            {{
                 .dfb_spec_name = IN0_DFB,
                 .accessor_name = "in0",
                 .endpoint_type = experimental::DFBEndpointType::CONSUMER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = IN1_DFB,
                 .accessor_name = "in1",
                 .endpoint_type = experimental::DFBEndpointType::CONSUMER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = OUT_DFB,
                 .accessor_name = "out",
                 .endpoint_type = experimental::DFBEndpointType::PRODUCER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             }},
        .compile_time_args =
            {{"per_core_block_cnt", 1u}, {"per_core_block_size", static_cast<uint32_t>(test_config.num_tiles)}},
        .hw_config = compute_hw_config,
    };

    experimental::ProgramSpec spec{
        .name = "sfpu_binary_compute",
        .kernels = {reader_spec, make_writer_unary_quasar_spec(WRITER, OUT_DFB), compute_spec},
        .dataflow_buffers =
            {make_dfb_spec(IN0_DFB, test_config, test_config.l1_input_data_format),
             make_dfb_spec(IN1_DFB, test_config, test_config.l1_input_data_format),
             make_dfb_spec(OUT_DFB, test_config, test_config.l1_output_data_format)},
        .work_units = {experimental::WorkUnitSpec{
            .name = "main", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = node}},
    };

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = READER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node,
                {{"src0_addr", input0_dram_buffer->address()},
                 {"src0_bank_id", 0u},
                 {"src1_addr", input1_dram_buffer->address()},
                 {"src1_bank_id", 0u},
                 {"num_tiles", static_cast<uint32_t>(test_config.num_tiles)}})},
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = WRITER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node,
                {{"dst_addr", output_dram_buffer->address()},
                 {"bank_id", 0u},
                 {"num_tiles", static_cast<uint32_t>(test_config.num_tiles)}}),
        },
        experimental::ProgramRunArgs::KernelRunArgs{.kernel = COMPUTE},
    };

    const auto dest = sfpu_quasar_run(
        mesh_device,
        spec,
        params,
        {{input0_dram_buffer, &packed_lhs}, {input1_dram_buffer, &packed_rhs}},
        output_dram_buffer);
    return sfpu_util::is_close_packed_sfpu_output(dest, packed_golden, test_config.sfpu_op);
}

/// High-level flow:
///
///   DRAM(in0) ─┐
///   DRAM(in1) ─┼─> Reader ─> in0/in1/in2 DFB ─> SFPU Compute (eltwise_sfpu_2_0.cpp, SFPU_TERNARY_OP) ─> out DFB ─>
///   Writer ─> DRAM(out) DRAM(in2) ─┘
///
/// @param mesh_device Device under test.
/// @param test_config - Configuration of the test -- see struct
/// @return
bool run_sfpu_ternary_three_input_buffer(distributed::MeshDevice& mesh_device, const SfpuConfig& test_config) {
    const size_t per_buffer_byte_size = test_config.num_tiles * test_config.tile_byte_size;
    auto input0_dram_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = per_buffer_byte_size},
        {.page_size = per_buffer_byte_size, .buffer_type = tt::tt_metal::BufferType::DRAM},
        &mesh_device);
    auto input1_dram_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = per_buffer_byte_size},
        {.page_size = per_buffer_byte_size, .buffer_type = tt::tt_metal::BufferType::DRAM},
        &mesh_device);
    auto input2_dram_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = per_buffer_byte_size},
        {.page_size = per_buffer_byte_size, .buffer_type = tt::tt_metal::BufferType::DRAM},
        &mesh_device);
    auto output_dram_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = per_buffer_byte_size},
        {.page_size = per_buffer_byte_size, .buffer_type = tt::tt_metal::BufferType::DRAM},
        &mesh_device);

    const size_t numel = per_buffer_byte_size / sizeof(bfloat16);
    const int seed = std::chrono::system_clock::now().time_since_epoch().count();
    auto [packed_in0, packed_in1, packed_in2] =
        sfpu_util::generate_packed_sfpu_ternary_inputs(numel, test_config.sfpu_op, seed);

    auto in0 = unpack_vector<bfloat16, uint32_t>(packed_in0);
    auto in1 = unpack_vector<bfloat16, uint32_t>(packed_in1);
    auto in2 = unpack_vector<bfloat16, uint32_t>(packed_in2);
    std::vector<bfloat16> golden(in0.size());
    for (size_t i = 0; i < in0.size(); ++i) {
        golden[i] = sfpu_util::sfpu_ternary_function(test_config.sfpu_op, in0[i], in1[i], in2[i]);
    }
    std::vector<uint32_t> packed_golden = pack_vector<uint32_t, bfloat16>(golden);

    std::map<std::string, std::string> sfpu_defines = sfpu_util::sfpu_ternary_op_to_op_name.at(test_config.sfpu_op);
    sfpu_defines["SFPU_TERNARY_OP"] = "1";

    std::vector<uint32_t> dest_buffer_data;
    if (mesh_device.arch() == ARCH::QUASAR) {
        const experimental::DFBSpecName IN0_DFB{"in0_dfb"};
        const experimental::DFBSpecName IN1_DFB{"in1_dfb"};
        const experimental::DFBSpecName IN2_DFB{"in2_dfb"};
        const experimental::DFBSpecName OUT_DFB{"out_dfb"};
        const experimental::KernelSpecName READER{"reader"};
        const experimental::KernelSpecName WRITER{"writer"};
        const experimental::KernelSpecName COMPUTE{"compute"};

        const auto node = extract_single_core_node(test_config, "Metal 2.0 ternary SFPU path");

        experimental::KernelSpec reader_spec{
            .unique_id = READER,
            .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_binary_2_0.cpp",
            .num_threads = 1,
            .compiler_options = {.defines = {{"LOAD_BUF2_DATA", "1"}}},
            .dfb_bindings =
                {{
                     .dfb_spec_name = IN0_DFB,
                     .accessor_name = "in0",
                     .endpoint_type = experimental::DFBEndpointType::PRODUCER,
                     .access_pattern = experimental::DFBAccessPattern::STRIDED,
                 },
                 {
                     .dfb_spec_name = IN1_DFB,
                     .accessor_name = "in1",
                     .endpoint_type = experimental::DFBEndpointType::PRODUCER,
                     .access_pattern = experimental::DFBAccessPattern::STRIDED,
                 },
                 {
                     .dfb_spec_name = IN2_DFB,
                     .accessor_name = "in2",
                     .endpoint_type = experimental::DFBEndpointType::PRODUCER,
                     .access_pattern = experimental::DFBAccessPattern::STRIDED,
                 }},
            .runtime_arg_schema =
                {.runtime_arg_names =
                     {"src0_addr",
                      "src0_bank_id",
                      "src1_addr",
                      "src1_bank_id",
                      "num_tiles",
                      "src2_addr",
                      "src2_bank_id"}},
            .hw_config =
                experimental::DataMovementHardwareConfig{
                    .config_2xx =
                        experimental::DataMovementHardwareConfig::DataMovement2XXConfig{
                            .disable_dfb_implicit_sync_for_all = true,
                        },
                },
        };

        experimental::ComputeHardwareConfig compute_hw_config;
        compute_hw_config = experimental::ComputeHardwareConfig{
            .sfpu_precision_mode =
                test_config.approx_mode ? tt::tt_metal::Precision::Approximate : tt::tt_metal::Precision::Precise,
        };

        experimental::KernelSpec compute_spec{
            .unique_id = COMPUTE,
            .source = "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_sfpu_2_0.cpp",
            .num_threads = 1,
            .compiler_options = {.defines = to_kernel_defines(sfpu_defines)},
            .dfb_bindings =
                {{
                     .dfb_spec_name = IN0_DFB,
                     .accessor_name = "in0",
                     .endpoint_type = experimental::DFBEndpointType::CONSUMER,
                     .access_pattern = experimental::DFBAccessPattern::STRIDED,
                 },
                 {
                     .dfb_spec_name = IN1_DFB,
                     .accessor_name = "in1",
                     .endpoint_type = experimental::DFBEndpointType::CONSUMER,
                     .access_pattern = experimental::DFBAccessPattern::STRIDED,
                 },
                 {
                     .dfb_spec_name = IN2_DFB,
                     .accessor_name = "in2",
                     .endpoint_type = experimental::DFBEndpointType::CONSUMER,
                     .access_pattern = experimental::DFBAccessPattern::STRIDED,
                 },
                 {
                     .dfb_spec_name = OUT_DFB,
                     .accessor_name = "out",
                     .endpoint_type = experimental::DFBEndpointType::PRODUCER,
                     .access_pattern = experimental::DFBAccessPattern::STRIDED,
                 }},
            .compile_time_args =
                {{"per_core_block_cnt", 1u}, {"per_core_block_size", static_cast<uint32_t>(test_config.num_tiles)}},
            .hw_config = compute_hw_config,
        };

        experimental::ProgramSpec spec{
            .name = "sfpu_ternary_compute",
            .kernels = {reader_spec, make_writer_unary_quasar_spec(WRITER, OUT_DFB), compute_spec},
            .dataflow_buffers =
                {make_dfb_spec(IN0_DFB, test_config, test_config.l1_input_data_format),
                 make_dfb_spec(IN1_DFB, test_config, test_config.l1_input_data_format),
                 make_dfb_spec(IN2_DFB, test_config, test_config.l1_input_data_format),
                 make_dfb_spec(OUT_DFB, test_config, test_config.l1_output_data_format)},
            .work_units = {experimental::WorkUnitSpec{
                .name = "main", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = node}},
        };

        experimental::ProgramRunArgs params;
        params.kernel_run_args = {
            experimental::ProgramRunArgs::KernelRunArgs{
                .kernel = READER,
                .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                    node,
                    {{"src0_addr", input0_dram_buffer->address()},
                     {"src0_bank_id", 0u},
                     {"src1_addr", input1_dram_buffer->address()},
                     {"src1_bank_id", 0u},
                     {"num_tiles", static_cast<uint32_t>(test_config.num_tiles)},
                     {"src2_addr", input2_dram_buffer->address()},
                     {"src2_bank_id", 0u}}),
            },
            experimental::ProgramRunArgs::KernelRunArgs{
                .kernel = WRITER,
                .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                    node,
                    {{"dst_addr", output_dram_buffer->address()},
                     {"bank_id", 0u},
                     {"num_tiles", static_cast<uint32_t>(test_config.num_tiles)}}),
            },
            experimental::ProgramRunArgs::KernelRunArgs{.kernel = COMPUTE},
        };
        dest_buffer_data = sfpu_quasar_run(
            mesh_device,
            spec,
            params,
            {{input0_dram_buffer, &packed_in0}, {input1_dram_buffer, &packed_in1}, {input2_dram_buffer, &packed_in2}},
            output_dram_buffer);
    } else {
        auto& cq = mesh_device.mesh_command_queue();
        tt_metal::Program program = tt_metal::CreateProgram();

        // reader_binary.cpp with LOAD_BUF2_DATA:
        //   {src0_addr, src0_bank, src1_addr, src1_bank, num_tiles, src2_addr, src2_bank}
        vector<uint32_t> reader_rt_args = {
            (uint32_t)input0_dram_buffer->address(),
            0u,
            (uint32_t)input1_dram_buffer->address(),
            0u,
            (uint32_t)test_config.num_tiles,
            (uint32_t)input2_dram_buffer->address(),
            0u,
        };
        vector<uint32_t> writer_rt_args = {
            (uint32_t)output_dram_buffer->address(), 0u, (uint32_t)test_config.num_tiles};
        // eltwise_sfpu.cpp ternary path reads {per_core_block_cnt, per_core_block_size} as compile-time args.
        vector<uint32_t> compute_kernel_args = {1u, (uint32_t)test_config.num_tiles};

        for (const CoreRange& core_range : test_config.cores.ranges()) {
            auto make_input_cb = [&](tt::CBIndex idx) {
                return tt_metal::CircularBufferConfig(per_buffer_byte_size, {{idx, test_config.l1_input_data_format}})
                    .set_page_size(idx, test_config.tile_byte_size);
            };
            tt_metal::CreateCircularBuffer(program, core_range, make_input_cb(tt::CBIndex::c_0));
            tt_metal::CreateCircularBuffer(program, core_range, make_input_cb(tt::CBIndex::c_1));
            tt_metal::CreateCircularBuffer(program, core_range, make_input_cb(tt::CBIndex::c_2));

            tt_metal::CircularBufferConfig l1_output_cb_config =
                tt_metal::CircularBufferConfig(
                    per_buffer_byte_size, {{tt::CBIndex::c_16, test_config.l1_output_data_format}})
                    .set_page_size(tt::CBIndex::c_16, test_config.tile_byte_size);
            tt_metal::CreateCircularBuffer(program, core_range, l1_output_cb_config);

            auto reader_kernel = tt_metal::CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_binary.cpp",
                test_config.cores,
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_1,
                    .noc = tt_metal::NOC::RISCV_1_default,
                    .defines = {{"LOAD_BUF2_DATA", "1"}}});

            auto writer_kernel = create_legacy_writer_kernel(program, test_config);

            tt_metal::CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_sfpu.cpp",
                test_config.cores,
                tt_metal::ComputeConfig{
                    .dst_full_sync_en = true,
                    .math_approx_mode = test_config.approx_mode,
                    .compile_args = compute_kernel_args,
                    .defines = sfpu_defines});

            for (const CoreCoord& core_coord : core_range) {
                SetRuntimeArgs(program, reader_kernel, core_coord, reader_rt_args);
                SetRuntimeArgs(program, writer_kernel, core_coord, writer_rt_args);
            }
        }

        distributed::EnqueueWriteMeshBuffer(cq, input0_dram_buffer, packed_in0, /*blocking=*/true);
        distributed::EnqueueWriteMeshBuffer(cq, input1_dram_buffer, packed_in1, /*blocking=*/true);
        distributed::EnqueueWriteMeshBuffer(cq, input2_dram_buffer, packed_in2, /*blocking=*/true);
        LaunchProgram(mesh_device, std::move(program));
        distributed::EnqueueReadMeshBuffer(cq, dest_buffer_data, output_dram_buffer, /*blocking=*/true);
    }

    return sfpu_util::is_close_packed_sfpu_output(dest_buffer_data, packed_golden, test_config.sfpu_op);
}

/// High-level flow (single input, differing in/out formats):
///
///   DRAM(in, SRC) -> Reader -> in DFB(SRC) -> SFPU typecast(SRC->DST) -> out DFB(DST) -> Writer -> DRAM(out, DST)
///
/// MX <-> float pairs issue no SFPU op (the unpack/pack gasket performs the conversion); the kernel
/// chain still runs copy_tile + pack_tile, so the conversion happens symmetrically on both threads.
bool run_sfpu_typecast(
    distributed::MeshDevice& mesh_device, tt::DataFormat in_fmt, tt::DataFormat out_fmt, size_t num_tiles) {
    const size_t numel = num_tiles * tt::constants::TILE_HW;
    const int seed = std::chrono::system_clock::now().time_since_epoch().count();
    auto vals = sfpu_util::generate_typecast_input(numel, seed, in_fmt, out_fmt);
    auto packed_in = sfpu_util::typecast_pack(in_fmt, vals);
    auto golden = sfpu_util::typecast_golden(in_fmt, out_fmt, packed_in);

    // Flag selection mirrors the typecast op / the tt-llk typecast test (see test_eltwise_unary_typecast.py:
    // _preserve_fp32_precision, _production_dest_acc, unpack_to_dest), so the compute-side datapath matches
    // the LLK reference these conversions pass under.
    const bool in_is_32bit = in_fmt == tt::DataFormat::Float32 || in_fmt == tt::DataFormat::Int32;
    const bool out_is_32bit = out_fmt == tt::DataFormat::Float32 || out_fmt == tt::DataFormat::Int32;

    // preserve_fp32_precision: a Float32 input, an 8-bit-output promotion from a bf16/MX source, or a
    // UInt8 input.
    const bool preserve_fp32 = in_fmt == tt::DataFormat::Float32 ||
                               (out_fmt == tt::DataFormat::UInt8 &&
                                (in_fmt == tt::DataFormat::Float16_b || sfpu_util::typecast_is_mx(in_fmt))) ||
                               in_fmt == tt::DataFormat::UInt8;

    // dest_acc (fp32_dest_acc_en): a 32-bit endpoint or preserve_fp32. Narrow integer pairs (e.g. Int16 ->
    // Float16_b) stay in a 16-bit Dest, exactly as production runs them.
    const bool fp32_dest_acc = preserve_fp32 || in_is_32bit || out_is_32bit;

    // unpack-to-Dest only changes the datapath for genuine 32-bit inputs (the narrow-input unpack MOP gates
    // on is_32bit_input, so the flag is inert for Int16/UInt8 -- they reach Dest via FPU A2D datacopy).
    // Wiring UnpackToDestFp32 for a narrow input drives the wrong datapath (UInt8 hangs, Int16 corrupts).
    const bool unpack_to_dest = in_is_32bit;

    // typecast_tile_init<IN, OUT>() + typecast_tile<IN, OUT>(0), with the format pair baked into the
    // template args via the device-side ckernel::DataFormat enum names.
    std::map<std::string, std::string> defines;
    defines["SFPU_UNARY_OP"] = "1";
    defines["SFPU_OP_TYPECAST_INCLUDE"] = "1";
    const std::string tmpl = "<static_cast<uint32_t>(DataFormat::" + sfpu_util::typecast_device_format_name(in_fmt) +
                             "), static_cast<uint32_t>(DataFormat::" + sfpu_util::typecast_device_format_name(out_fmt) +
                             ")>";
    defines["SFPU_OP_CHAIN_0"] = "typecast_tile_init" + tmpl + "(); typecast_tile" + tmpl + "(0);";

    const CoreRange core_range({0, 0}, {0, 0});
    SfpuConfig cfg{
        .num_tiles = num_tiles,
        .l1_input_data_format = in_fmt,
        .l1_output_data_format = out_fmt,
        .cores = CoreRangeSet({core_range}),
        .approx_mode = false,
        .unpack_to_dest = unpack_to_dest,
        .en_32bit_dest = fp32_dest_acc,
    };

    const auto dest = run_sfpu_pipeline(mesh_device, cfg, defines, packed_in);
    const auto got = sfpu_util::typecast_decode(out_fmt, dest);
    return sfpu_util::typecast_compare(out_fmt, got, golden);
}

}  // namespace unit_tests::compute::sfpu

void run_quasar_sfpu_unpack_to_dest_fp32(
    distributed::MeshDevice& dev, size_t num_tiles, const std::string& sfpu_op, bool dst_full_sync_en) {
    CoreRange core_range({0, 0}, {0, 0});
    CoreRangeSet core_range_set({core_range});
    unit_tests::compute::sfpu::SfpuConfig cfg{
        .num_tiles = num_tiles,
        .tile_byte_size = 4 * 32 * 32,
        .l1_input_data_format = tt::DataFormat::Float32,
        .l1_output_data_format = tt::DataFormat::Float32,
        .cores = core_range_set,
        .sfpu_op = sfpu_op,
        .approx_mode = false,
        .dst_full_sync_en = dst_full_sync_en,
        .unpack_to_dest = true,
        .en_32bit_dest = true,
    };
    log_info(
        tt::LogTest, "Quasar SFPU FP32: op={} num_tiles={} dst_full_sync_en={}", sfpu_op, num_tiles, dst_full_sync_en);
    EXPECT_TRUE(unit_tests::compute::sfpu::run_sfpu_all_same_buffer(dev, cfg));
}

void run_quasar_sfpu_unpack_to_dest_16b(
    distributed::MeshDevice& dev, size_t num_tiles, const std::string& sfpu_op, bool dst_full_sync_en) {
    CoreRange core_range({0, 0}, {0, 0});
    CoreRangeSet core_range_set({core_range});
    unit_tests::compute::sfpu::SfpuConfig cfg{
        .num_tiles = num_tiles,
        .tile_byte_size = 2 * 32 * 32,
        .l1_input_data_format = tt::DataFormat::Float16_b,
        .l1_output_data_format = tt::DataFormat::Float16_b,
        .cores = core_range_set,
        .sfpu_op = sfpu_op,
        .approx_mode = false,
        .dst_full_sync_en = dst_full_sync_en,
        .unpack_to_dest = true,  // 16-bit operand unpack-to-dest (fp32_dest_acc_en stays false)
    };
    log_info(
        tt::LogTest,
        "Quasar SFPU 16b->DEST: op={} num_tiles={} dst_full_sync_en={}",
        sfpu_op,
        num_tiles,
        dst_full_sync_en);
    EXPECT_TRUE(unit_tests::compute::sfpu::run_sfpu_all_same_buffer(dev, cfg));
}

// Unary SFPU ops an arch has no compute-API implementation for, so building the kernel would fail
// with "not declared in this scope". Skip them there so the suite reflects actual coverage instead
// of a hard kernel-build failure. softcap has kernels on Blackhole and Quasar only (softcap.h).
inline bool is_unary_sfpu_op_unsupported(tt::ARCH arch, const std::string& sfpu_op) {
    return arch == tt::ARCH::WORMHOLE_B0 && sfpu_op == "softcap";
}

class SingleCoreSingleMeshDeviceSfpuParameterizedFixture
    : public LLKMeshDeviceFixture,
      public testing::WithParamInterface<std::tuple<size_t, std::string>> {};
TEST_P(SingleCoreSingleMeshDeviceSfpuParameterizedFixture, TensixSfpuCompute) {
    size_t num_tiles = std::get<0>(GetParam());
    std::string sfpu_op = std::get<1>(GetParam());

    if (is_unary_sfpu_op_unsupported(arch_, sfpu_op)) {
        GTEST_SKIP() << "SFPU unary op '" << sfpu_op << "' has no compute-API implementation on this arch";
    }

    CoreRange core_range({0, 0}, {0, 0});
    CoreRangeSet core_range_set({core_range});
    unit_tests::compute::sfpu::SfpuConfig test_config = {
        .num_tiles = num_tiles,
        .tile_byte_size = 2 * 32 * 32,
        .l1_input_data_format = tt::DataFormat::Float16_b,
        .l1_output_data_format = tt::DataFormat::Float16_b,
        .cores = core_range_set,
        .sfpu_op = sfpu_op,
        .approx_mode = false};
    log_info(tt::LogTest, "Testing SFPU_OP={} num_tiles={}", sfpu_op, num_tiles);
    for (auto& device : this->devices_) {
        EXPECT_TRUE(run_sfpu_all_same_buffer(*device, test_config));
    }
}

INSTANTIATE_TEST_SUITE_P(
    SingleCoreSfpuCompute,
    SingleCoreSingleMeshDeviceSfpuParameterizedFixture,
    ::testing::Values(
        std::make_tuple(1, "eqz"),
        std::make_tuple(1, "nez"),
        std::make_tuple(1, "ltz"),
        std::make_tuple(1, "gtz"),
        std::make_tuple(1, "gez"),
        std::make_tuple(1, "lez"),
        std::make_tuple(1, "ceil"),
        std::make_tuple(4, "ceil"),
        std::make_tuple(1, "floor"),
        std::make_tuple(4, "floor"),
        std::make_tuple(1, "trunc"),
        std::make_tuple(4, "trunc"),
        std::make_tuple(1, "frac"),
        std::make_tuple(4, "frac"),
        std::make_tuple(1, "round"),
        std::make_tuple(4, "round"),
        std::make_tuple(4, "eqz"),
        std::make_tuple(4, "nez"),
        std::make_tuple(4, "ltz"),
        std::make_tuple(4, "gtz"),
        std::make_tuple(4, "gez"),
        std::make_tuple(4, "lez"),
        std::make_tuple(1, "square"),
        std::make_tuple(4, "square"),
        std::make_tuple(1, "negative"),
        std::make_tuple(4, "negative"),
        std::make_tuple(1, "softplus"),
        std::make_tuple(4, "softplus"),
        std::make_tuple(1, "clamp"),
        std::make_tuple(4, "clamp"),
        std::make_tuple(1, "relu"),
        std::make_tuple(1, "relu_min"),
        std::make_tuple(1, "relu_max"),
        std::make_tuple(1, "exponential"),
        std::make_tuple(1, "reciprocal"),
        std::make_tuple(1, "gelu"),
        std::make_tuple(1, "gelu_accurate"),
        std::make_tuple(1, "sqrt"),
        std::make_tuple(1, "sigmoid"),
        std::make_tuple(1, "silu"),
        std::make_tuple(1, "log"),
        std::make_tuple(1, "tanh"),
        std::make_tuple(1, "sign"),
        std::make_tuple(1, "rsqrt"),
        std::make_tuple(1, "mul_unary"),
        std::make_tuple(4, "relu"),
        std::make_tuple(4, "relu_min"),
        std::make_tuple(4, "relu_max"),
        std::make_tuple(4, "exponential"),
        std::make_tuple(4, "reciprocal"),
        std::make_tuple(4, "gelu"),
        std::make_tuple(4, "gelu_accurate"),
        std::make_tuple(4, "sqrt"),
        std::make_tuple(4, "sigmoid"),
        std::make_tuple(4, "silu"),
        std::make_tuple(4, "log"),
        std::make_tuple(4, "tanh"),
        std::make_tuple(4, "sign"),
        std::make_tuple(4, "rsqrt"),
        std::make_tuple(4, "mul_unary"),
        std::make_tuple(1, "hardsigmoid"),
        std::make_tuple(1, "softsign"),
        std::make_tuple(1, "celu"),
        std::make_tuple(1, "softshrink"),
        std::make_tuple(1, "hardshrink"),
        std::make_tuple(1, "elu"),
        std::make_tuple(1, "selu"),
        std::make_tuple(1, "hardtanh"),
        std::make_tuple(1, "erf"),
        std::make_tuple(1, "erfc"),
        std::make_tuple(1, "erfinv"),
        std::make_tuple(1, "cbrt"),
        std::make_tuple(1, "i0"),
        std::make_tuple(1, "i1"),
        std::make_tuple(1, "identity"),
        std::make_tuple(1, "hardmish"),
        std::make_tuple(1, "mish"),
        std::make_tuple(1, "isinf"),
        std::make_tuple(1, "isposinf"),
        std::make_tuple(1, "isneginf"),
        std::make_tuple(1, "isnan"),
        std::make_tuple(1, "isfinite"),
        std::make_tuple(1, "lgamma_stirling"),
        std::make_tuple(1, "digamma"),
        std::make_tuple(1, "polygamma"),
        std::make_tuple(1, "logical_not"),
        std::make_tuple(1, "prelu"),
        std::make_tuple(1, "rdiv"),
        std::make_tuple(1, "rpow"),
        std::make_tuple(1, "fmod"),
        std::make_tuple(1, "remainder"),
        std::make_tuple(1, "tanh_derivative"),
        std::make_tuple(1, "tanhshrink"),
        std::make_tuple(1, "threshold"),
        std::make_tuple(1, "xielu"),
        std::make_tuple(1, "softcap"),
        std::make_tuple(1, "unary_ne"),
        std::make_tuple(1, "unary_eq"),
        std::make_tuple(1, "unary_gt"),
        std::make_tuple(1, "unary_ge"),
        std::make_tuple(1, "unary_lt"),
        std::make_tuple(1, "unary_le"),
        std::make_tuple(1, "power"),
        std::make_tuple(1, "power_iterative"),
        std::make_tuple(1, "exp2"),
        std::make_tuple(1, "heaviside"),
        std::make_tuple(1, "expm1"),
        std::make_tuple(1, "log_with_base"),
        std::make_tuple(1, "tiled_prod"),
        std::make_tuple(1, "alt_complex_rotate90")),
    [](const testing::TestParamInfo<std::tuple<size_t, std::string>>& info) {
        return std::get<1>(info.param) + "_" + std::to_string(std::get<0>(info.param)) + "tiles";
    });

class SingleCoreSingleMeshDeviceSfpuParameterizedApproxFixture
    : public LLKMeshDeviceFixture,
      public testing::WithParamInterface<std::tuple<size_t, std::string>> {};

TEST_P(SingleCoreSingleMeshDeviceSfpuParameterizedApproxFixture, TensixSfpuCompute) {
    size_t num_tiles = std::get<0>(GetParam());
    std::string sfpu_op = std::get<1>(GetParam());

    if (is_unary_sfpu_op_unsupported(arch_, sfpu_op)) {
        GTEST_SKIP() << "SFPU unary op '" << sfpu_op << "' has no compute-API implementation on this arch";
    }
    if (((arch_ == tt::ARCH::WORMHOLE_B0) && (sfpu_op == "relu" || sfpu_op == "relu_min" || sfpu_op == "relu_max")) ||
        ((arch_ == tt::ARCH::WORMHOLE_B0) && (sfpu_op == "exponential")) ||
        ((arch_ == tt::ARCH::WORMHOLE_B0) && (sfpu_op == "log"))) {
        GTEST_SKIP();
    }
    CoreRange core_range({0, 0}, {0, 0});
    CoreRangeSet core_range_set({core_range});
    unit_tests::compute::sfpu::SfpuConfig test_config = {
        .num_tiles = num_tiles,
        .tile_byte_size = 2 * 32 * 32,
        .l1_input_data_format = tt::DataFormat::Float16_b,
        .l1_output_data_format = tt::DataFormat::Float16_b,
        .cores = core_range_set,
        .sfpu_op = sfpu_op,
        .approx_mode = true};
    log_info(tt::LogTest, "Testing SFPU_OP={} num_tiles={}", sfpu_op, num_tiles);
    for (auto& device : this->devices_) {
        EXPECT_TRUE(run_sfpu_all_same_buffer(*device, test_config));
    }
}
INSTANTIATE_TEST_SUITE_P(
    SingleCoreSfpuCompute,
    SingleCoreSingleMeshDeviceSfpuParameterizedApproxFixture,
    ::testing::Values(
        std::make_tuple(1, "relu"),
        std::make_tuple(1, "relu_min"),
        std::make_tuple(1, "relu_max"),
        std::make_tuple(1, "exponential"),
        std::make_tuple(1, "reciprocal"),
        std::make_tuple(1, "gelu"),
        std::make_tuple(1, "sqrt"),
        std::make_tuple(1, "sigmoid"),
        std::make_tuple(1, "silu"),
        std::make_tuple(1, "log"),
        std::make_tuple(1, "tanh"),
        std::make_tuple(1, "sign"),
        std::make_tuple(1, "rsqrt"),
        std::make_tuple(1, "mul_unary"),
        std::make_tuple(4, "relu"),
        std::make_tuple(4, "relu_min"),
        std::make_tuple(4, "relu_max"),
        std::make_tuple(4, "exponential"),
        std::make_tuple(4, "reciprocal"),
        std::make_tuple(4, "gelu"),
        std::make_tuple(4, "sqrt"),
        std::make_tuple(4, "sigmoid"),
        std::make_tuple(4, "silu"),
        std::make_tuple(4, "log"),
        std::make_tuple(4, "tanh"),
        std::make_tuple(4, "sign"),
        std::make_tuple(4, "rsqrt"),
        std::make_tuple(4, "mul_unary"),
        std::make_tuple(1, "hardsigmoid"),
        std::make_tuple(1, "softsign"),
        std::make_tuple(1, "celu"),
        std::make_tuple(1, "softshrink"),
        std::make_tuple(1, "hardshrink"),
        std::make_tuple(1, "elu"),
        std::make_tuple(1, "selu"),
        std::make_tuple(1, "hardtanh"),
        std::make_tuple(1, "erf"),
        std::make_tuple(1, "erfc"),
        std::make_tuple(1, "erfinv"),
        std::make_tuple(1, "cbrt"),
        std::make_tuple(1, "i0"),
        std::make_tuple(1, "i1"),
        std::make_tuple(1, "identity"),
        std::make_tuple(1, "hardmish"),
        std::make_tuple(1, "mish"),
        std::make_tuple(1, "isinf"),
        std::make_tuple(1, "isposinf"),
        std::make_tuple(1, "isneginf"),
        std::make_tuple(1, "isnan"),
        std::make_tuple(1, "isfinite"),
        std::make_tuple(1, "lgamma_stirling"),
        std::make_tuple(1, "digamma"),
        std::make_tuple(1, "polygamma"),
        std::make_tuple(1, "logical_not"),
        std::make_tuple(1, "prelu"),
        std::make_tuple(1, "rdiv"),
        std::make_tuple(1, "rpow"),
        std::make_tuple(1, "fmod"),
        std::make_tuple(1, "remainder"),
        std::make_tuple(1, "tanh_derivative"),
        std::make_tuple(1, "tanhshrink"),
        std::make_tuple(1, "threshold"),
        std::make_tuple(1, "xielu"),
        std::make_tuple(1, "softcap"),
        std::make_tuple(1, "unary_ne"),
        std::make_tuple(1, "unary_eq"),
        std::make_tuple(1, "unary_gt"),
        std::make_tuple(1, "unary_ge"),
        std::make_tuple(1, "unary_lt"),
        std::make_tuple(1, "unary_le"),
        std::make_tuple(1, "power"),
        std::make_tuple(1, "power_iterative"),
        std::make_tuple(1, "exp2"),
        std::make_tuple(1, "heaviside"),
        std::make_tuple(1, "expm1"),
        std::make_tuple(1, "log_with_base"),
        std::make_tuple(1, "tiled_prod"),
        std::make_tuple(1, "alt_complex_rotate90")),
    [](const testing::TestParamInfo<std::tuple<size_t, std::string>>& info) {
        return std::get<1>(info.param) + "_" + std::to_string(std::get<0>(info.param)) + "tiles";
    });

class SingleCoreSingleMeshDeviceSfpuParameterized32BitDestFixture
    : public LLKMeshDeviceFixture,
      public testing::WithParamInterface<std::tuple<size_t, std::string>> {};
TEST_P(SingleCoreSingleMeshDeviceSfpuParameterized32BitDestFixture, TensixSfpuCompute) {
    size_t num_tiles = std::get<0>(GetParam());
    std::string sfpu_op = std::get<1>(GetParam());

    if (is_unary_sfpu_op_unsupported(arch_, sfpu_op)) {
        GTEST_SKIP() << "SFPU unary op '" << sfpu_op << "' has no compute-API implementation on this arch";
    }

    CoreRange core_range({0, 0}, {0, 0});
    CoreRangeSet core_range_set({core_range});
    unit_tests::compute::sfpu::SfpuConfig test_config = {
        .num_tiles = num_tiles,
        .tile_byte_size = 2 * 32 * 32,
        .l1_input_data_format = tt::DataFormat::Float16_b,
        .l1_output_data_format = tt::DataFormat::Float16_b,
        .cores = core_range_set,
        .sfpu_op = sfpu_op,
        .approx_mode = false,
        .en_32bit_dest = true};
    log_info(tt::LogTest, "Testing SFPU_OP={} num_tiles={}", sfpu_op, num_tiles);
    for (auto& device : this->devices_) {
        EXPECT_TRUE(run_sfpu_all_same_buffer(*device, test_config));
    }
}

INSTANTIATE_TEST_SUITE_P(
    SingleCoreSfpuCompute,
    SingleCoreSingleMeshDeviceSfpuParameterized32BitDestFixture,
    ::testing::Values(
        std::make_tuple(1, "ceil"),
        std::make_tuple(4, "ceil"),
        std::make_tuple(1, "floor"),
        std::make_tuple(4, "floor"),
        std::make_tuple(1, "trunc"),
        std::make_tuple(4, "trunc"),
        std::make_tuple(1, "frac"),
        std::make_tuple(4, "frac"),
        std::make_tuple(1, "round"),
        std::make_tuple(4, "round"),
        std::make_tuple(1, "negative"),
        std::make_tuple(4, "negative"),
        std::make_tuple(1, "softplus"),
        std::make_tuple(4, "softplus"),
        std::make_tuple(1, "clamp"),
        std::make_tuple(4, "clamp"),
        std::make_tuple(1, "relu"),
        std::make_tuple(1, "relu_min"),
        std::make_tuple(1, "relu_max"),
        std::make_tuple(1, "exponential"),
        std::make_tuple(1, "reciprocal"),
        std::make_tuple(1, "gelu"),
        std::make_tuple(1, "gelu_accurate"),
        std::make_tuple(1, "sqrt"),
        std::make_tuple(1, "sigmoid"),
        std::make_tuple(1, "silu"),
        std::make_tuple(1, "log"),
        std::make_tuple(1, "tanh"),
        std::make_tuple(1, "sign"),
        std::make_tuple(1, "rsqrt"),
        std::make_tuple(4, "relu"),
        std::make_tuple(4, "relu_min"),
        std::make_tuple(4, "relu_max"),
        std::make_tuple(4, "exponential"),
        std::make_tuple(4, "reciprocal"),
        std::make_tuple(4, "gelu"),
        std::make_tuple(4, "gelu_accurate"),
        std::make_tuple(4, "sqrt"),
        std::make_tuple(4, "sigmoid"),
        std::make_tuple(4, "silu"),
        std::make_tuple(4, "log"),
        std::make_tuple(4, "tanh"),
        std::make_tuple(4, "sign"),
        std::make_tuple(4, "rsqrt"),
        std::make_tuple(1, "hardsigmoid"),
        std::make_tuple(1, "softsign"),
        std::make_tuple(1, "celu"),
        std::make_tuple(1, "softshrink"),
        std::make_tuple(1, "hardshrink"),
        std::make_tuple(1, "elu"),
        std::make_tuple(1, "selu"),
        std::make_tuple(1, "hardtanh"),
        std::make_tuple(1, "erf"),
        std::make_tuple(1, "erfc"),
        std::make_tuple(1, "erfinv"),
        std::make_tuple(1, "cbrt"),
        std::make_tuple(1, "i0"),
        std::make_tuple(1, "i1"),
        std::make_tuple(1, "identity"),
        std::make_tuple(1, "hardmish"),
        std::make_tuple(1, "mish"),
        std::make_tuple(1, "isinf"),
        std::make_tuple(1, "isposinf"),
        std::make_tuple(1, "isneginf"),
        std::make_tuple(1, "isnan"),
        std::make_tuple(1, "isfinite"),
        std::make_tuple(1, "lgamma_stirling"),
        std::make_tuple(1, "digamma"),
        std::make_tuple(1, "polygamma"),
        std::make_tuple(1, "logical_not"),
        std::make_tuple(1, "prelu"),
        std::make_tuple(1, "rdiv"),
        std::make_tuple(1, "rpow"),
        std::make_tuple(1, "fmod"),
        std::make_tuple(1, "remainder"),
        std::make_tuple(1, "tanh_derivative"),
        std::make_tuple(1, "tanhshrink"),
        std::make_tuple(1, "threshold"),
        std::make_tuple(1, "xielu"),
        std::make_tuple(1, "softcap"),
        std::make_tuple(1, "unary_ne"),
        std::make_tuple(1, "unary_eq"),
        std::make_tuple(1, "unary_gt"),
        std::make_tuple(1, "unary_ge"),
        std::make_tuple(1, "unary_lt"),
        std::make_tuple(1, "unary_le"),
        std::make_tuple(1, "power"),
        std::make_tuple(1, "power_iterative"),
        std::make_tuple(1, "exp2"),
        std::make_tuple(1, "heaviside"),
        std::make_tuple(1, "expm1"),
        std::make_tuple(1, "log_with_base"),
        std::make_tuple(1, "tiled_prod"),
        std::make_tuple(1, "alt_complex_rotate90")),
    [](const testing::TestParamInfo<std::tuple<size_t, std::string>>& info) {
        return std::get<1>(info.param) + "_" + std::to_string(std::get<0>(info.param)) + "tiles";
    });

class SingleCoreSingleMeshDeviceSfpuParameterized32BitDestApproxFixture
    : public LLKMeshDeviceFixture,
      public testing::WithParamInterface<std::tuple<size_t, std::string>> {};

TEST_P(SingleCoreSingleMeshDeviceSfpuParameterized32BitDestApproxFixture, TensixSfpuCompute) {
    size_t num_tiles = std::get<0>(GetParam());
    std::string sfpu_op = std::get<1>(GetParam());

    if (is_unary_sfpu_op_unsupported(arch_, sfpu_op)) {
        GTEST_SKIP() << "SFPU unary op '" << sfpu_op << "' has no compute-API implementation on this arch";
    }
    if (((arch_ == tt::ARCH::WORMHOLE_B0) && (sfpu_op == "relu" || sfpu_op == "relu_min" || sfpu_op == "relu_max")) ||
        ((arch_ == tt::ARCH::WORMHOLE_B0) && (sfpu_op == "exponential")) ||
        ((arch_ == tt::ARCH::WORMHOLE_B0) && (sfpu_op == "log"))) {
        GTEST_SKIP();
    }
    CoreRange core_range({0, 0}, {0, 0});
    CoreRangeSet core_range_set({core_range});
    unit_tests::compute::sfpu::SfpuConfig test_config = {
        .num_tiles = num_tiles,
        .tile_byte_size = 2 * 32 * 32,
        .l1_input_data_format = tt::DataFormat::Float16_b,
        .l1_output_data_format = tt::DataFormat::Float16_b,
        .cores = core_range_set,
        .sfpu_op = sfpu_op,
        .approx_mode = true,
        .en_32bit_dest = true};
    log_info(tt::LogTest, "Testing SFPU_OP={} num_tiles={}", sfpu_op, num_tiles);
    for (auto& device : this->devices_) {
        EXPECT_TRUE(run_sfpu_all_same_buffer(*device, test_config));
    }
}
INSTANTIATE_TEST_SUITE_P(
    SingleCoreSfpuCompute,
    SingleCoreSingleMeshDeviceSfpuParameterized32BitDestApproxFixture,
    ::testing::Values(
        std::make_tuple(1, "relu"),
        std::make_tuple(1, "relu_min"),
        std::make_tuple(1, "relu_max"),
        std::make_tuple(1, "exponential"),
        std::make_tuple(1, "reciprocal"),
        std::make_tuple(1, "gelu"),
        std::make_tuple(1, "sqrt"),
        std::make_tuple(1, "sigmoid"),
        std::make_tuple(1, "silu"),
        std::make_tuple(1, "log"),
        std::make_tuple(1, "tanh"),
        std::make_tuple(1, "sign"),
        std::make_tuple(1, "rsqrt"),
        std::make_tuple(4, "relu"),
        std::make_tuple(4, "relu_min"),
        std::make_tuple(4, "relu_max"),
        std::make_tuple(4, "exponential"),
        std::make_tuple(4, "reciprocal"),
        std::make_tuple(4, "gelu"),
        std::make_tuple(4, "sqrt"),
        std::make_tuple(4, "sigmoid"),
        std::make_tuple(4, "silu"),
        std::make_tuple(4, "log"),
        std::make_tuple(4, "tanh"),
        std::make_tuple(4, "sign"),
        std::make_tuple(4, "rsqrt"),
        std::make_tuple(1, "hardsigmoid"),
        std::make_tuple(1, "softsign"),
        std::make_tuple(1, "celu"),
        std::make_tuple(1, "softshrink"),
        std::make_tuple(1, "hardshrink"),
        std::make_tuple(1, "elu"),
        std::make_tuple(1, "selu"),
        std::make_tuple(1, "hardtanh"),
        std::make_tuple(1, "erf"),
        std::make_tuple(1, "erfc"),
        std::make_tuple(1, "erfinv"),
        std::make_tuple(1, "cbrt"),
        std::make_tuple(1, "i0"),
        std::make_tuple(1, "i1"),
        std::make_tuple(1, "identity"),
        std::make_tuple(1, "hardmish"),
        std::make_tuple(1, "mish"),
        std::make_tuple(1, "isinf"),
        std::make_tuple(1, "isposinf"),
        std::make_tuple(1, "isneginf"),
        std::make_tuple(1, "isnan"),
        std::make_tuple(1, "isfinite"),
        std::make_tuple(1, "lgamma_stirling"),
        std::make_tuple(1, "digamma"),
        std::make_tuple(1, "polygamma"),
        std::make_tuple(1, "logical_not"),
        std::make_tuple(1, "prelu"),
        std::make_tuple(1, "rdiv"),
        std::make_tuple(1, "rpow"),
        std::make_tuple(1, "fmod"),
        std::make_tuple(1, "remainder"),
        std::make_tuple(1, "tanh_derivative"),
        std::make_tuple(1, "tanhshrink"),
        std::make_tuple(1, "threshold"),
        std::make_tuple(1, "xielu"),
        std::make_tuple(1, "softcap"),
        std::make_tuple(1, "unary_ne"),
        std::make_tuple(1, "unary_eq"),
        std::make_tuple(1, "unary_gt"),
        std::make_tuple(1, "unary_ge"),
        std::make_tuple(1, "unary_lt"),
        std::make_tuple(1, "unary_le"),
        std::make_tuple(1, "power"),
        std::make_tuple(1, "power_iterative"),
        std::make_tuple(1, "exp2"),
        std::make_tuple(1, "heaviside"),
        std::make_tuple(1, "expm1"),
        std::make_tuple(1, "log_with_base"),
        std::make_tuple(1, "tiled_prod"),
        std::make_tuple(1, "alt_complex_rotate90")),
    [](const testing::TestParamInfo<std::tuple<size_t, std::string>>& info) {
        return std::get<1>(info.param) + "_" + std::to_string(std::get<0>(info.param)) + "tiles";
    });

// Binary SFPU parameterized test fixture (mirrors the unary fixture above).
//
// Each test instance is identified by (num_tiles, op_name). The op_name picks
// up macro substitutions from sfpu_binary_op_to_op_name and a host-side
// reference from sfpu_binary_function or get_binary_int_operation_result(). The name generator
// suffixes each instance with its op name (e.g. div_binary_1tiles) so a single op can be run standalone
// via --gtest_filter='*div_binary*' / '*add_int*' / '*mul_int*', while still sharing the single
// MeshDevice that LLKMeshDeviceFixture opens once per suite (no per-test device).
class SingleCoreSingleMeshDeviceSfpuBinaryParameterizedFixture
    : public LLKMeshDeviceFixture,
      public testing::WithParamInterface<std::tuple<size_t, std::string>> {};

TEST_P(SingleCoreSingleMeshDeviceSfpuBinaryParameterizedFixture, TensixSfpuBinaryCompute) {
    size_t num_tiles = std::get<0>(GetParam());
    std::string sfpu_op = std::get<1>(GetParam());

    if (MetalContext::instance().get_cluster().arch() == ARCH::WORMHOLE_B0 ||
        MetalContext::instance().get_cluster().arch() == ARCH::BLACKHOLE) {
        GTEST_SKIP() << "Binary SFPU op test not fixed for WH/BH";
    }

    // add_int/mul_int: Int8 L1 inputs promoted to sign-mag Int32 output. div_binary stays bfloat16.
    const bool is_int8_op = unit_tests::sfpu_util::is_int8_binary_sfpu_op(sfpu_op);
    const tt::DataFormat data_format_input = is_int8_op ? tt::DataFormat::Int8 : tt::DataFormat::Float16_b;
    tt::DataFormat data_format_output = is_int8_op ? tt::DataFormat::Int32 : tt::DataFormat::Float16_b;
    if (sfpu_op == "div_int32") {
        data_format_output = tt::DataFormat::Float32;
    }
    const size_t tile_byte_size = is_int8_op ? tt::tile_size(tt::DataFormat::Int8) : 2 * 32 * 32;

    CoreRange core_range({0, 0}, {0, 0});
    CoreRangeSet core_range_set({core_range});
    unit_tests::compute::sfpu::SfpuConfig test_config = {
        .num_tiles = num_tiles,
        .tile_byte_size = tile_byte_size,
        .l1_input_data_format = data_format_input,
        .l1_output_data_format = data_format_output,
        .cores = core_range_set,
        .sfpu_op = sfpu_op,
        .approx_mode = false};
    log_info(tt::LogTest, "Testing binary SFPU_OP={} num_tiles={}", sfpu_op, num_tiles);
    for (auto& device : this->devices_) {
        EXPECT_TRUE(unit_tests::compute::sfpu::run_sfpu_binary_two_input_buffer(*device, test_config));
    }
}

// TODO: BinarySFPU ops here can only do 1 tile due to the hardcoding in the macros to indicies (0,1,2)
INSTANTIATE_TEST_SUITE_P(
    SingleCoreSfpuBinaryCompute,
    SingleCoreSingleMeshDeviceSfpuBinaryParameterizedFixture,
    ::testing::Values(
        std::make_tuple(1, "div_binary"),
        std::make_tuple(1, "mul_float"),
        std::make_tuple(1, "atan2"),
        std::make_tuple(1, "add_int"),
        std::make_tuple(1, "mul_int"),
        std::make_tuple(1, "gt_int"),
        std::make_tuple(1, "binary_max"),
        std::make_tuple(1, "binary_min"),
        std::make_tuple(1, "binary_max_int32"),
        std::make_tuple(1, "binary_min_int32"),
        std::make_tuple(1, "copy_dest"),
        std::make_tuple(1, "copy_dest_int"),
        std::make_tuple(1, "power_binary"),
        std::make_tuple(1, "fmod_binary"),
        std::make_tuple(1, "remainder_binary"),
        std::make_tuple(1, "logsigmoid"),
        std::make_tuple(1, "isclose"),
        std::make_tuple(1, "clamped_silu_glu"),
        std::make_tuple(1, "situ_glu"),
        std::make_tuple(1, "mask"),
        std::make_tuple(1, "mask_posinf"),
        std::make_tuple(1, "lgamma_stirling_float"),
        std::make_tuple(1, "add_top_row"),
        std::make_tuple(1, "add_top_row_int32"),
        std::make_tuple(1, "div_int32_floor"),
        std::make_tuple(1, "div_int32_trunc"),
        std::make_tuple(1, "div_int32"),
        std::make_tuple(1, "fmod_int32"),
        std::make_tuple(1, "remainder_int32"),
        std::make_tuple(1, "bitwise_and_binary"),
        std::make_tuple(1, "bitwise_or_binary"),
        std::make_tuple(1, "bitwise_xor_binary"),
        std::make_tuple(1, "rsub_int"),
        std::make_tuple(1, "sfpu_add_int"),
        std::make_tuple(1, "int_mask")),
    [](const testing::TestParamInfo<std::tuple<size_t, std::string>>& info) {
        return std::get<1>(info.param) + "_" + std::to_string(std::get<0>(info.param)) + "tiles";
    });

class SingleCoreSingleMeshDeviceSfpuTernaryParameterizedFixture
    : public LLKMeshDeviceFixture,
      public testing::WithParamInterface<std::tuple<size_t, std::string>> {};

TEST_P(SingleCoreSingleMeshDeviceSfpuTernaryParameterizedFixture, TensixSfpuTernaryCompute) {
    size_t num_tiles = std::get<0>(GetParam());
    std::string sfpu_op = std::get<1>(GetParam());

    if (MetalContext::instance().get_cluster().arch() == ARCH::WORMHOLE_B0 ||
        MetalContext::instance().get_cluster().arch() == ARCH::BLACKHOLE) {
        GTEST_SKIP() << "Ternary SFPU op test not fixed for WH/BH";
    }

    CoreRange core_range({0, 0}, {0, 0});
    CoreRangeSet core_range_set({core_range});
    unit_tests::compute::sfpu::SfpuConfig test_config = {
        .num_tiles = num_tiles,
        .tile_byte_size = 2 * 32 * 32,
        .l1_input_data_format = tt::DataFormat::Float16_b,
        .l1_output_data_format = tt::DataFormat::Float16_b,
        .cores = core_range_set,
        .sfpu_op = sfpu_op,
        .approx_mode = false};
    log_info(tt::LogTest, "Testing ternary SFPU_OP={} num_tiles={}", sfpu_op, num_tiles);
    for (auto& device : this->devices_) {
        EXPECT_TRUE(unit_tests::compute::sfpu::run_sfpu_ternary_three_input_buffer(*device, test_config));
    }
}

INSTANTIATE_TEST_SUITE_P(
    SingleCoreSfpuTernaryCompute,
    SingleCoreSingleMeshDeviceSfpuTernaryParameterizedFixture,
    ::testing::Values(
        std::make_tuple(1, "where"),
        std::make_tuple(1, "mac"),
        std::make_tuple(1, "lerp"),
        std::make_tuple(1, "addcdiv"),
        std::make_tuple(1, "addcmul"),
        std::make_tuple(1, "snake_beta")),
    [](const testing::TestParamInfo<std::tuple<size_t, std::string>>& info) {
        return std::get<1>(info.param) + "_" + std::to_string(std::get<0>(info.param)) + "tiles";
    });

// Int32 unary ops: Int8 L1 -> 32-bit Dest -> Int32 L1; 16-bit Dest cannot hold them, so there is no
// bf16 / approx split.
class SingleCoreSingleMeshDeviceSfpuInt32UnaryParameterizedFixture
    : public LLKMeshDeviceFixture,
      public testing::WithParamInterface<std::tuple<size_t, std::string>> {};

TEST_P(SingleCoreSingleMeshDeviceSfpuInt32UnaryParameterizedFixture, TensixSfpuCompute) {
    const size_t num_tiles = std::get<0>(GetParam());
    const std::string sfpu_op = std::get<1>(GetParam());
    // Same Int8 -> sign-magnitude Int32 Dest path as the binary integer ops.
    if (arch_ == tt::ARCH::WORMHOLE_B0 || arch_ == tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << "Int32 unary SFPU op test not fixed for WH/BH";
    }
    CoreRange core_range({0, 0}, {0, 0});
    unit_tests::compute::sfpu::SfpuConfig test_config = {
        .num_tiles = num_tiles,
        .tile_byte_size = tt::tile_size(tt::DataFormat::Int8),
        .l1_input_data_format = tt::DataFormat::Int8,
        .l1_output_data_format = tt::DataFormat::Int32,
        .cores = CoreRangeSet({core_range}),
        .sfpu_op = sfpu_op,
        .approx_mode = false,
        .en_32bit_dest = true};
    log_info(tt::LogTest, "Testing Int32 SFPU_OP={} num_tiles={}", sfpu_op, num_tiles);
    for (auto& device : this->devices_) {
        EXPECT_TRUE(unit_tests::compute::sfpu::run_sfpu_int32_unary(*device, test_config));
    }
}

INSTANTIATE_TEST_SUITE_P(
    SingleCoreSfpuCompute,
    SingleCoreSingleMeshDeviceSfpuInt32UnaryParameterizedFixture,
    ::testing::Values(
        std::make_tuple(1, "bitwise_and"),
        std::make_tuple(1, "bitwise_or"),
        std::make_tuple(1, "bitwise_xor"),
        std::make_tuple(1, "left_shift"),
        std::make_tuple(1, "right_shift"),
        std::make_tuple(1, "rsub_unary_int32"),
        std::make_tuple(1, "logical_not_int32"),
        std::make_tuple(1, "sum_int_col"),
        std::make_tuple(1, "sum_int_row")),
    [](const testing::TestParamInfo<std::tuple<size_t, std::string>>& info) {
        return std::get<1>(info.param) + "_" + std::to_string(std::get<0>(info.param)) + "tiles";
    });

TEST_F(QuasarMeshDeviceSingleCardFixture, QuasarSfpuRelu) {
    // 1 and 4-tile, SyncFull and SyncHalf
    for (const char* sfpu_op : {"relu", "relu_min", "relu_max"}) {
        for (const uint32_t num_tiles : {1u, 4u}) {
            for (const bool dst_full_sync_en : {true, false}) {
                SCOPED_TRACE(
                    std::string(sfpu_op) + " num_tiles=" + std::to_string(num_tiles) +
                    (dst_full_sync_en ? " SyncFull" : " SyncHalf"));
                run_quasar_sfpu_unpack_to_dest_fp32(this->device(), num_tiles, sfpu_op, dst_full_sync_en);
            }
        }
    }
}

TEST_F(QuasarMeshDeviceSingleCardFixture, QuasarSfpuUnpackToDest16b) {
    // 16-bit operand explicitly unpacked to Dest
    for (const char* sfpu_op : {"relu", "relu_min", "relu_max"}) {
        for (const bool dst_full_sync_en : {true, false}) {
            for (uint32_t num_tiles : {1u, 4u}) {
                log_info(
                    tt::LogTest,
                    "Quasar SFPU 16b->DEST: op={} num_tiles={} {}",
                    sfpu_op,
                    num_tiles,
                    dst_full_sync_en ? "SyncFull" : "SyncHalf");
                run_quasar_sfpu_unpack_to_dest_16b(this->device(), num_tiles, sfpu_op, dst_full_sync_en);
            }
        }
    }
}

// Typecast test fixture: one (in_format -> out_format) pair per instance, covering the Quasar typecast
// matrix over Float16_b, Float32, Int32, Int16 (SMAG16), UInt8, and MX (MxFp8P / MxFp8R). UInt16 maps to
// Int16; UInt32 / MxFp4 are out of scope. Unsupported pairs are skipped by the gate in the test body.
class SingleCoreSingleMeshDeviceSfpuTypecastFixture
    : public LLKMeshDeviceFixture,
      public testing::WithParamInterface<std::tuple<tt::DataFormat, tt::DataFormat>> {};

TEST_P(SingleCoreSingleMeshDeviceSfpuTypecastFixture, TensixSfpuTypecast) {
    const auto in_fmt = std::get<0>(GetParam());
    const auto out_fmt = std::get<1>(GetParam());

    if (MetalContext::instance().get_cluster().arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Typecast compute-API test is currently Quasar-only";
    }

    // Typecast pairs not yet wired through the metal2 compute-API datapath (they hang or throw today):
    //  * any UInt8 endpoint;
    //  * any Int16 endpoint (the data_format.cpp Int16 enablement is reverted until the full narrow-int
    //    datapath lands, so an Int16 endpoint trips the format-consistency / pack_src derivation);
    //  * a non-Float32 input widening into a 32-bit Int output (Float16_b/MX -> Int32).
    // Float32 <-> Int32 and Float16_b/MX <-> Float32 still run.
    // Int16/UInt8 support through the metal2 compute-API path is tracked in tenstorrent/tt-metal#48601.
    const bool uint8_endpoint = (in_fmt == tt::DataFormat::UInt8 || out_fmt == tt::DataFormat::UInt8);
    const bool int16_endpoint = (in_fmt == tt::DataFormat::Int16 || out_fmt == tt::DataFormat::Int16);
    const bool widen_to_int32 = (out_fmt == tt::DataFormat::Int32 && in_fmt != tt::DataFormat::Float32);
    if (uint8_endpoint || int16_endpoint || widen_to_int32) {
        GTEST_SKIP() << "typecast format not yet supported through the metal2 compute-API path "
                        "(tenstorrent/tt-metal#48601)";
    }

    log_info(
        tt::LogTest,
        "Testing typecast {} -> {}",
        unit_tests::sfpu_util::typecast_device_format_name(in_fmt),
        unit_tests::sfpu_util::typecast_device_format_name(out_fmt));
    for (auto& device : this->devices_) {
        EXPECT_TRUE(unit_tests::compute::sfpu::run_sfpu_typecast(*device, in_fmt, out_fmt, 1));
    }
}

// Every directed endpoint pair (in != out), minus pairs that are not real conversions: MxFp8P <-> MxFp8R
// (both arrive as Float16_b in Dest) and UInt8 -> Int32/Int16 (not part of the matrix).
static std::vector<std::tuple<tt::DataFormat, tt::DataFormat>> typecast_pairs() {
    const tt::DataFormat endpoints[] = {
        tt::DataFormat::Float16_b,
        tt::DataFormat::Float32,
        tt::DataFormat::Int32,
        tt::DataFormat::Int16,
        tt::DataFormat::UInt8,
        tt::DataFormat::MxFp8P,
        tt::DataFormat::MxFp8R};
    std::vector<std::tuple<tt::DataFormat, tt::DataFormat>> pairs;
    for (const tt::DataFormat in : endpoints) {
        for (const tt::DataFormat out : endpoints) {
            if (in == out) {
                continue;
            }
            const bool mx_to_mx =
                unit_tests::sfpu_util::typecast_is_mx(in) && unit_tests::sfpu_util::typecast_is_mx(out);
            const bool uint8_to_wide_int =
                in == tt::DataFormat::UInt8 && (out == tt::DataFormat::Int32 || out == tt::DataFormat::Int16);
            if (!mx_to_mx && !uint8_to_wide_int) {
                pairs.emplace_back(in, out);
            }
        }
    }
    return pairs;
}

INSTANTIATE_TEST_SUITE_P(
    SingleCoreSfpuTypecast,
    SingleCoreSingleMeshDeviceSfpuTypecastFixture,
    ::testing::ValuesIn(typecast_pairs()),
    [](const testing::TestParamInfo<std::tuple<tt::DataFormat, tt::DataFormat>>& info) {
        return unit_tests::sfpu_util::typecast_device_format_name(std::get<0>(info.param)) + "_to_" +
               unit_tests::sfpu_util::typecast_device_format_name(std::get<1>(info.param));
    });

}  // namespace tt::tt_metal
