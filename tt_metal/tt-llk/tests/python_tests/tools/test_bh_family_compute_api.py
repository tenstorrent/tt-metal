# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``compute_api`` family (tools.llk_state_audit.bh.families.compute_api).

The synthetic tree holds compute-API functions copied verbatim from the folded Blackhole tree
(patch 02, the FOLD) and from main. The llk_api / llk_lib callees keep their real signatures,
and their bodies are cut down to the sinks the tests need.
"""

import tempfile
import unittest
from pathlib import Path

from test_bh_model import write_tree

from tools.llk_state_audit.bh import effects_db
from tools.llk_state_audit.bh.families import compute_api

API = "tt_metal/hw/inc/api/compute"
LLK_API = "tt_metal/hw/ckernels/blackhole/metal/llk_api"
LLK_LIB = "tt_metal/tt-llk/tt_llk_blackhole/llk_lib"
COMMON = "tt_metal/tt-llk/tt_llk_blackhole/common/inc"

FILES = {
    "tt_metal/hw/inc/internal/tt-1xx/blackhole/cfg_defines.h": """
// Registers for ALU
#define STACC_RELU_ApplyRelu_ADDR32 2
#define STACC_RELU_ApplyRelu_SHAMT 2
#define STACC_RELU_ApplyRelu_MASK 0x3c
#define STACC_RELU_ApplyRelu_RMW STACC_RELU_ApplyRelu_ADDR32, STACC_RELU_ApplyRelu_SHAMT, STACC_RELU_ApplyRelu_MASK
#define STACC_RELU_ReluThreshold_ADDR32 2
#define STACC_RELU_ReluThreshold_SHAMT 6
#define STACC_RELU_ReluThreshold_MASK 0x3fffc0
// Registers for THCON
#define THCON_SEC0_REG3_Base_address_ADDR32 76
#define THCON_SEC0_REG3_Base_address_SHAMT 0
#define THCON_SEC0_REG3_Base_address_MASK 0xffffffff
#define THCON_SEC0_REG3_Base_cntx1_address_ADDR32 77
#define THCON_SEC0_REG3_Base_cntx1_address_SHAMT 0
#define THCON_SEC0_REG3_Base_cntx1_address_MASK 0xffffffff
#define THCON_SEC1_REG3_Base_address_ADDR32 124
#define THCON_SEC1_REG3_Base_address_SHAMT 0
#define THCON_SEC1_REG3_Base_address_MASK 0xffffffff
#define THCON_SEC1_REG3_Base_cntx1_address_ADDR32 125
#define THCON_SEC1_REG3_Base_cntx1_address_SHAMT 0
#define THCON_SEC1_REG3_Base_cntx1_address_MASK 0xffffffff
""",
    f"{COMMON}/ckernel_gpr_map.h": """
namespace ckernel {
struct p_gpr_pack { constexpr static std::uint32_t TMP0 = 28; };
}
""",
    f"{COMMON}/ckernel_ops.h": """
#define TTI_STALLWAIT(stall_res, wait_res) TTI_INSN(TT_OP_STALLWAIT(stall_res, wait_res))
#define TTI_UNPACR(a, b, c, d, e, f, g, h, i, j, k, l, m) TTI_INSN(TT_OP_UNPACR(a, b, c, d, e, f, g, h, i, j, k, l, m))
#define TTI_UNPACR_NOP(a, b, c, d, e, f, g, h, i) TTI_INSN(TT_OP_UNPACR_NOP(a, b, c, d, e, f, g, h, i))
#define TTI_SETADCZW(a, b, c, d, e, f) TTI_INSN(TT_OP_SETADCZW(a, b, c, d, e, f))
#define TTI_RMWCIB0(mask, data, addr) TTI_INSN(TT_OP_RMWCIB0(mask, data, addr))
#define TTI_ATGETM(index) TTI_INSN(TT_OP_ATGETM(index))
#define TTI_ATRELM(index) TTI_INSN(TT_OP_ATRELM(index))
""",
    f"{COMMON}/ckernel.h": """
namespace ckernel {
template <std::uint32_t CfgAddr32, std::uint32_t Shamt, std::uint32_t Mask>
inline void cfg_reg_rmw_tensix(std::uint32_t val)
{
    TTI_RMWCIB0(Mask & 0xff, val << Shamt, CfgAddr32);
}

inline void t6_mutex_acquire(const std::uint8_t index)
{
    TTI_ATGETM(index);
}

inline void t6_mutex_release(const std::uint8_t index)
{
    TTI_ATRELM(index);
}
}
""",
    # llk_lib (main): RISC-side cfg[] stores of the unpacker base addresses
    f"{LLK_LIB}/llk_unpack_common.h": """
inline void _llk_unpack_configure_addresses_(const std::uint32_t address_a, const std::uint32_t address_b, volatile std::uint32_t tt_reg_ptr *cfg)
{
    LLK_ASSERT(is_valid_L1_address(address_a), "L1 address_a must be in valid L1 memory region");
    LLK_ASSERT(is_valid_L1_address(address_b), "L1 address_b must be in valid L1 memory region");

    if (0 == unp_cfg_context)
    {
        cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address_a;
        cfg[THCON_SEC1_REG3_Base_address_ADDR32] = address_b;
    }
    else
    {
        cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = address_a;
        cfg[THCON_SEC1_REG3_Base_cntx1_address_ADDR32] = address_b;
    }
}
""",
    # llk_lib (folded tree): relu config through a Tensix-side RMW
    f"{LLK_LIB}/llk_pack_common.h": """
TT_ALWAYS_INLINE void _llk_pack_relu_config_(const ckernel::ReluConfig& relu_config)
{
    const std::uint32_t mode = static_cast<std::uint32_t>(relu_config.get_mode());
    const std::uint32_t val  = (relu_config.get_threshold() << STACC_RELU_ReluThreshold_SHAMT) | (mode << STACC_RELU_ApplyRelu_SHAMT);
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK);
    t6_mutex_acquire(mutex::REG_RMW);
    cfg_reg_rmw_tensix<STACC_RELU_ApplyRelu_RMW>(val);
    t6_mutex_release(mutex::REG_RMW);
}
""",
    # llk_api: real signatures, bodies reduced
    f"{LLK_API}/llk_pack_common_api.h": """
TT_ALWAYS_INLINE void llk_pack_relu_config(const ckernel::ReluConfig& relu_config) {
    _llk_pack_relu_config_(relu_config);
}
inline void llk_pack_reconfig_l1_acc(const std::uint32_t enable) { _llk_pack_reconfig_l1_acc_(enable); }
""",
    f"{LLK_API}/llk_pack_tile_api.h": """
template <
    PackMode pack_mode = PackMode::Default,
    bool zero_output = false,
    bool skip_addrmod_config = false,
    bool skip_packer_strides = false>
inline void llk_pack_init(const std::uint32_t pack_output, const std::uint32_t num_tiles = 1) {
    static_assert(pack_mode != PackMode::Tilize, "PackMode::Tilize requires an explicit input_operand");
}
template <bool is_fp32_dest_acc_en, bool out_of_order_output = false, PackMode pack_mode = PackMode::Default>
inline void llk_pack(std::uint32_t tile_index, std::uint32_t output, std::uint32_t output_tile_index = 0) {
    std::uint8_t output_id = get_output_id(output);
}
""",
    f"{LLK_API}/llk_math_binary_api.h": """
template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    MathFidelity math_fidelity = MathFidelity::LoFi,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void llk_math_eltwise_binary_init(
    const std::uint32_t operand_A, const std::uint32_t operand_B, const std::uint32_t acc_to_dest = 0) {
}
template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    bool is_fp32_dest_acc_en,
    MathFidelity math_fidelity = MathFidelity::LoFi,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void llk_math_eltwise_binary(
    const std::uint32_t operand_A, const std::uint32_t operand_B, std::uint32_t dst_index, const bool clear_fp32_dst_acc) {
}
""",
    f"{LLK_API}/llk_unpack_AB_api.h": """
template <BroadcastType BType = BroadcastType::NONE>
inline void llk_unpack_AB_init(const std::uint32_t operandA, const std::uint32_t operandB, const std::uint32_t transpose = 0) {
}
template <BroadcastType BType = BroadcastType::NONE>
inline void llk_unpack_AB(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t tile_index_a,
    const std::uint32_t tile_index_b,
    const std::uint32_t bcast_row_idx = 0) {
}
""",
    f"{LLK_API}/llk_unpack_A_api.h": """
template <
    BroadcastType BType = BroadcastType::NONE,
    bool acc_to_dest = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool unpack_to_dest = false>
inline void llk_unpack_A_init(
    const std::uint32_t transpose_of_faces,
    const std::uint32_t within_face_16x16_transpose,
    const std::uint32_t operand) {
    const std::uint32_t operand_id = get_operand_id(operand);
}
template <
    BroadcastType BType = BroadcastType::NONE,
    bool acc_to_dest = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool unpack_to_dest = false>
inline void llk_unpack_A(const std::uint32_t operand, const std::uint32_t tile_index) {
    std::uint32_t operand_id = get_operand_id(operand);
}
""",
    f"{LLK_API}/llk_math_unary_datacopy_api.h": """
template <
    DataCopyType type,
    bool is_fp32_dest_acc_en,
    BroadcastType src_b_bcast_type = BroadcastType::NONE,
    bool is_int_fpu_en = false,
    PackMode pack_mode = PackMode::Default>
inline void llk_math_eltwise_unary_datacopy_init(const std::uint32_t operand) {
    const std::uint32_t operand_id = get_operand_id(operand);
}
template <
    DataCopyType type,
    bool is_fp32_dest_acc_en,
    BroadcastType src_b_bcast_type = BroadcastType::NONE,
    bool unpack_to_dest = false>
inline void llk_math_eltwise_unary_datacopy(std::uint32_t dst_index, std::uint32_t operand) {
}
""",
    f"{LLK_API}/llk_math_matmul_api.h": """
template <MathFidelity math_fidelity, int THROTTLE_LEVEL = 0>
inline void llk_math_matmul_init(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t transpose = 0,
    const std::uint32_t ct_dim = 1,
    const std::uint32_t rt_dim = 1) {
}
template <MathFidelity math_fidelity, int THROTTLE_LEVEL = 0>
inline void llk_math_matmul(const std::uint32_t dst_index, const std::uint32_t ct_dim = 1, const std::uint32_t rt_dim = 1) {
}
""",
    f"{LLK_API}/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h": """
#define _SFPU_EXPAND(...) __VA_ARGS__
#define SFPU_UNARY_CALL(DST_SYNC, DST_ACCUM, FN, TEMPLATES, DST_IDX, VECTOR_MODE, ...) \\
    (::ckernel::_sfpu_check_<DST_SYNC>(DST_IDX, VECTOR_MODE),               \\
     _llk_math_eltwise_unary_sfpu_params_(                                             \\
         ::ckernel::sfpu::FN<_SFPU_EXPAND TEMPLATES>, DST_IDX, VECTOR_MODE, ##__VA_ARGS__))
#define SFPU_UNARY_INIT_FN(OP, INIT_FN, TEMPLATES) \\
    ::ckernel::llk_math_eltwise_unary_sfpu_init<::SfpuType::OP>(INIT_FN<_SFPU_EXPAND TEMPLATES>)
""",
    f"{LLK_API}/llk_sfpu/llk_math_eltwise_unary_sfpu_init.h": """
namespace ckernel {
template <SfpuType sfpu_op, class F, class... ARGS>
inline void llk_math_eltwise_unary_sfpu_init(F&& init_func, ARGS&&... args) {
    SAN_HOOK(unsupported());
    _llk_math_eltwise_unary_sfpu_init_<sfpu_op>();
    init_func(std::forward<ARGS>(args)...);
}
}
""",
    f"{LLK_API}/llk_unpack_common_api.h": """
template <bool is_fp32_dest_acc_en>
inline void llk_unpack_hw_configure(const std::uint32_t unpA_operand, const std::uint32_t unpB_operand) {
}
""",
    f"{LLK_API}/llk_pack_api.h": """
template <bool is_fp32_dest_acc_en>
inline void llk_pack_hw_configure(const std::uint32_t pack_output) {
}
template <bool is_fp32_dest_acc_en, PackMode pack_mode = PackMode::Default>
inline void llk_pack_dest_init(const std::uint32_t pack_output = 16) {
}
""",
    f"{LLK_API}/llk_math_common_api.h": """
template <bool is_fp32_dest_acc_en>
inline void llk_math_pack_sync_init() {
}
template <bool is_fp32_dest_acc_en>
inline void llk_math_hw_configure(const std::uint32_t srca_operand, const std::uint32_t srcb_operand) {
}
""",
    # ---------------------------------------------------------------- compute API
    f"{API}/sentinel/compute_kernel_sentinel.h": """
namespace ckernel {
ALWI void state_configure(std::uint32_t cb, std::uint32_t call_line) {
}
}
""",
    f"{API}/reconfig_data_format.h": """
namespace ckernel {
template <SrcOrder src_order = SrcOrder::Regular>
ALWI void reconfig_data_format(const std::uint32_t icb0_new_operand, const std::uint32_t icb1_new_operand) {
    LLK_SAN_FUNCTION();
    constexpr bool reverse = (src_order == SrcOrder::Reverse);
    detail::reconfig_df_both<false, false>(
        reverse ? icb1_new_operand : icb0_new_operand, reverse ? icb0_new_operand : icb1_new_operand);
}
ALWI void reconfig_data_format_srca(const std::uint32_t srca_new_operand) {
    LLK_SAN_FUNCTION();
    detail::reconfig_df_srca<false, false>(srca_new_operand);
}
template <bool is_tile_dim_reconfig_en = false>
ALWI void pack_reconfig_data_format(const std::uint32_t new_cb_id) {
    detail::pack_reconfig_df<is_tile_dim_reconfig_en>(new_cb_id);
}
}
""",
    f"{API}/eltwise_binary.h": """
namespace ckernel {
template <bool full_init, EltwiseBinaryType eltwise_binary_type>
ALWI void binary_tiles_init(std::uint32_t icb0, std::uint32_t icb1, bool acc_to_dest = false, std::uint32_t call_line = __builtin_LINE()) {
    MATH((llk_math_eltwise_binary_init<eltwise_binary_type, NONE, MATH_FIDELITY>(icb0, icb1, acc_to_dest)));
    if constexpr (full_init) {
        UNPACK((llk_unpack_AB_init<BroadcastType::NONE>(icb0, icb1)));
    }
}
ALWI void add_init(std::uint32_t icb0, std::uint32_t icb1, bool acc_to_dest = false, std::uint32_t call_line = __builtin_LINE()) {
    binary_tiles_init<true /* full_init */, EltwiseBinaryType::ELWADD>(icb0, icb1, acc_to_dest, call_line);
}
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void add_tiles(
    std::uint32_t icb0, std::uint32_t icb1, std::uint32_t itile0, std::uint32_t itile1, std::uint32_t idst, bool acc_to_dest = false) {
    reconfig_data_format(icb0, icb1);
    add_init(icb0, icb1, acc_to_dest);
    UNPACK((llk_unpack_AB(icb0, icb1, itile0, itile1)));
    MATH((llk_math_eltwise_binary<
          EltwiseBinaryType::ELWADD,
          BroadcastType::NONE,
          is_fp32_dest_acc_en,
          MathFidelity::LoFi,
          EltwiseBinaryReuseDestType::NONE>(icb0, icb1, idst, true)));
}
}
""",
    f"{API}/pack.h": """
namespace ckernel {
template <bool out_of_order_output = false, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void pack_tile(
    std::uint32_t ifrom_dst,
    std::uint32_t icb,
    std::uint32_t output_tile_index = 0,
    bool l1_acc = false,
    ReluConfig relu = ReluConfig::none()) {
    LLK_SAN_FUNCTION();
#ifndef ARCH_QUASAR
    pack_reconfig_data_format(icb);
    PACK((llk_pack_init<PackMode::Default, false, true, true>(icb, 1)));
    PACK((llk_pack_reconfig_l1_acc(l1_acc ? 1 : 0)));
    PACK((llk_pack_relu_config(relu)));
    PACK((llk_pack<is_fp32_dest_acc_en, out_of_order_output, PackMode::Default>(ifrom_dst, icb, output_tile_index)));
#else
    PACK((llk_pack<out_of_order_output>(ifrom_dst, icb, output_tile_index)));
#endif
}
ALWI void pack_reconfig_l1_acc(const std::uint32_t l1_acc_en) { PACK((llk_pack_reconfig_l1_acc(l1_acc_en))); }
ALWI void pack_relu_config(const ReluConfig& config) { PACK((llk_pack_relu_config(config))); }
}
""",
    f"{API}/reduce.h": """
namespace ckernel {
template <PoolType reduce_type, ReduceDim reduce_dim, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void reduce_init(
    std::uint32_t icb, std::uint32_t icb_scaler, std::uint32_t ocb, std::uint32_t call_line = __builtin_LINE()) {
    UNPACK((llk_unpack_AB_init<BroadcastType::NONE>(icb, icb_scaler)));
}
ALWI void reduce_uninit(std::uint32_t icb = 0) {
#ifdef ARCH_QUASAR
    UNPACK((llk_unpack_AB_reduce_uninit(icb)));
#elif defined(ARCH_BLACKHOLE)
    MATH((llk_math_reduce_uninit()));
#endif
}
template <PoolType reduce_type, ReduceDim reduce_dim, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void reduce_tile(
    std::uint32_t icb,
    std::uint32_t icb_scaler,
    std::uint32_t itile,
    std::uint32_t itile_scaler,
    std::uint32_t idst,
    std::uint32_t ocb) {
    if constexpr ((reduce_dim == ReduceDim::REDUCE_ROW) && (reduce_type != PoolType::MAX)) {
        reconfig_data_format(icb_scaler, icb);
    } else {
        reconfig_data_format(icb, icb_scaler);
    }
    reduce_init<reduce_type, reduce_dim, is_fp32_dest_acc_en>(icb, icb_scaler, ocb);
    reduce_uninit(icb);
}
}
""",
    f"{API}/eltwise_unary/exp.h": """
namespace ckernel {
template <
    bool approx = false,
    std::uint32_t scale = 0x3F800000,
    InputClamping input_clamping = InputClamping::ClampToNegative, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void exp_tile_init() {
    MATH(SFPU_UNARY_INIT_FN(
        exponential,
        sfpu::exp_init,
        (approx, scale, (input_clamping == InputClamping::ClampToNegative), is_fp32_dest_acc_en)));
}
template <
    bool approx = false,
    bool scale_en = false,
    InputClamping input_clamping = InputClamping::ClampToNegative,
    int iterations = 8,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE,
    std::uint32_t init_scale = 0x3F800000>
ALWI void exp_tile(std::uint32_t idst, VectorMode vector_mode = VectorMode::RC, std::uint16_t scale = p_sfpu::kCONST_1_FP16B) {
    exp_tile_init<approx, init_scale, input_clamping, is_fp32_dest_acc_en>();
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_exponential,
        (approx, is_fp32_dest_acc_en, scale_en, iterations, (input_clamping == InputClamping::ClampToNegative)),
        idst,
        vector_mode,
        scale));
}
}
""",
    f"{API}/tile_move_copy.h": """
namespace ckernel {
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void copy_init(
    std::uint32_t cbid,
    std::uint32_t transpose = 0,
    std::uint32_t transpose_within_16x16_face = false,
    std::uint32_t call_line = __builtin_LINE()) {
    LLK_SAN_FUNCTION();
#ifndef ARCH_QUASAR
    state_configure(cbid, call_line);
#else
    LLK_ASSERT(transpose_within_16x16_face == false, "Transpose within face not supported on Quasar");
    LLK_ASSERT(transpose == 0, "Transpose not supported on Quasar");
#endif
    UNPACK((llk_unpack_A_init<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(
        transpose, transpose_within_16x16_face, cbid)));
#ifndef ARCH_QUASAR
    MATH((llk_math_eltwise_unary_datacopy_init<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE>(cbid)));
#endif
}
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void copy_tile(std::uint32_t in_cb_id, std::uint32_t in_tile_index, std::uint32_t dst_tile_index) {
#ifndef ARCH_QUASAR
    LLK_SAN_FUNCTION();
#endif
    reconfig_data_format_srca(in_cb_id);
    copy_init<is_fp32_dest_acc_en>(in_cb_id);
    UNPACK((llk_unpack_A<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(
        in_cb_id, in_tile_index)));
    MATH((llk_math_eltwise_unary_datacopy<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, UnpackToDestEn>(
        dst_tile_index, in_cb_id)));
}
}
""",
    f"{API}/tilize.h": """
namespace ckernel {
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void tilize_init(std::uint32_t icb, std::uint32_t block, std::uint32_t ocb, std::uint32_t call_line = __builtin_LINE()) {
    UNPACK((llk_unpack_A_init<BroadcastType::NONE>(0, 0, icb)));
}
#if (defined(REDUCE_OP) and defined(REDUCE_DIM)) or defined(__DOXYGEN__)
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void tilizeA_B_reduce_init(
    std::uint32_t icb0, std::uint32_t icb1_scaler, std::uint32_t block, std::uint32_t ocb, std::uint32_t num_faces = 4, std::uint32_t face_r_dim = 16) {
    UNPACK((llk_unpack_AB_init<BroadcastType::NONE>(icb0, icb1_scaler)));
}
#endif
namespace tilize_detail {
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void tilize_block_body(
    std::uint32_t icb, std::uint32_t block, std::uint32_t ocb, std::uint32_t input_tile_index = 0, std::uint32_t output_tile_index = 0) {
    UNPACK((llk_unpack_tilize_block(icb, block, input_tile_index)));
}
}  // namespace tilize_detail
ALWI void tilize_uninit(std::uint32_t icb, std::uint32_t ocb) {
    UNPACK((llk_unpack_A_init<BroadcastType::NONE>(0, 0, icb)));
}
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void tilize_block(
    std::uint32_t icb,
    std::uint32_t block,
    std::uint32_t ocb,
    std::uint32_t input_tile_index = 0,
    std::uint32_t output_tile_index = 0,
    std::uint32_t row_ct_dim = 0) {
#ifndef ARCH_QUASAR
    reconfig_data_format_srca(icb);
    pack_reconfig_data_format(ocb);
#endif
    tilize_init<is_fp32_dest_acc_en>(icb, row_ct_dim ? row_ct_dim : block, ocb);
    tilize_detail::tilize_block_body<is_fp32_dest_acc_en>(icb, block, ocb, input_tile_index, output_tile_index);
    tilize_uninit(icb, ocb);
}
}
""",
    f"{API}/matmul.h": """
#ifndef MM_THROTTLE
#define MM_THROTTLE 0
#endif
namespace ckernel {
#ifdef ARCH_BLACKHOLE
#define MM_THROTTLE_MAX 5
volatile tt_l1_ptr std::uint32_t* throttle_ptr =
    reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(MEM_L1_ARC_FW_SCRATCH);
static std::uint32_t throttled_mop_status = 0;
ALWI void matmul_block_math_dynamic_throttle(
    std::uint32_t in0_cb_id,
    std::uint32_t in1_cb_id,
    std::uint32_t idst,
    const std::uint32_t transpose,
    std::uint32_t ct_dim,
    std::uint32_t rt_dim) {
    LLK_SAN_FUNCTION();
#ifndef ARCH_QUASAR
    volatile std::uint32_t mm_throttle_en = *(throttle_ptr) % 2;
    if (mm_throttle_en) {
        if (throttled_mop_status != 1) {
            MATH((
                llk_math_matmul_init<MATH_FIDELITY, MM_THROTTLE_MAX>(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim)));
            throttled_mop_status = 1;
        }
        MATH((llk_math_matmul<MATH_FIDELITY, MM_THROTTLE_MAX>(idst, ct_dim, rt_dim)));
    } else {
        if (throttled_mop_status != 0) {
            MATH((llk_math_matmul_init<MATH_FIDELITY, MM_THROTTLE>(in0_cb_id, in1_cb_id, transpose, ct_dim, rt_dim)));
            throttled_mop_status = 0;
        }
        MATH((llk_math_matmul<MATH_FIDELITY, MM_THROTTLE>(idst, ct_dim, rt_dim)));
    }
#endif
}
#endif
}
""",
    f"{API}/compute_kernel_hw_startup.h": """
namespace ckernel {
template <SrcOrder src_order = SrcOrder::Regular>
ALWI void compute_kernel_hw_startup(std::uint32_t icb0, std::uint32_t icb1, std::uint32_t ocb) {
    LLK_SAN_FUNCTION();
    constexpr bool reverse = (src_order == SrcOrder::Reverse);
    const std::uint32_t src_a_cb = reverse ? icb1 : icb0;
    const std::uint32_t src_b_cb = reverse ? icb0 : icb1;
#ifndef ARCH_QUASAR
    UNPACK((llk_unpack_hw_configure<DST_ACCUM_MODE>(src_a_cb, src_b_cb)));

    MATH((llk_math_pack_sync_init<DST_ACCUM_MODE>()));
    MATH((llk_math_hw_configure<DST_ACCUM_MODE>(src_a_cb, src_b_cb)));

    PACK((llk_pack_hw_configure<DST_ACCUM_MODE>(ocb)));
    PACK((llk_pack_init<PackMode::Default>(ocb)));
    PACK((llk_pack_dest_init<DST_ACCUM_MODE, PackMode::Default>(ocb)));
#endif
}
}
""",
    f"{API}/experimental/sdpa_weighted_reduce.h": """
namespace ckernel {
#ifdef TRISC_UNPACK
inline void weighted_reduce_unpack_impl(
    const std::uint32_t weights_cb, const std::uint32_t qk_cb, const std::uint32_t qk_tile_index = 0) {
    const std::uint32_t qk_id = get_operand_id(qk_cb);            // operandA -> SrcA
    const std::uint32_t weights_id = get_operand_id(weights_cb);  // operandB -> SrcB
    const std::uint32_t address_a =
        get_local_cb_interface(qk_id).fifo_rd_ptr - 1 + qk_tile_index * get_local_cb_interface(qk_id).fifo_page_size;
    const std::uint32_t address_b = get_local_cb_interface(weights_id).fifo_rd_ptr - 1;

    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer();

    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);  // reset both unpackers' counters
    // Wait for a free context, program SrcA/SrcB base addresses, release to unpacker.
    wait_for_next_context(2);
    _llk_unpack_configure_addresses_(address_a, address_b, cfg);
    semaphore_post(semaphore::UNPACK_SYNC);
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

    TTI_UNPACR(SrcA, 0b00010001, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    TTI_UNPACR(SrcB, 0b00000000, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
}
// Same sequence with the STALLWAIT removed: the base-address stores are not fenced before UNPACR.
inline void weighted_reduce_unpack_unfenced(const std::uint32_t address_a, const std::uint32_t address_b) {
    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer();
    _llk_unpack_configure_addresses_(address_a, address_b, cfg);
    semaphore_post(semaphore::UNPACK_SYNC);
    TTI_UNPACR(SrcA, 0b00010001, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
}
#endif
}
""",
}


def by_pattern(effects, pattern):
    return [e for e in effects if e.pattern == pattern]


def bindings_of(effects, callee, pattern=None):
    """{param: (value, source, access, condition)} for direct binding effects on ``callee``."""
    out = {}
    for e in effects:
        if pattern is not None and e.pattern != pattern:
            continue
        if e.pattern not in ("compute_api.bind", "compute_api.fold_bind"):
            continue
        if e.resource.startswith(f"param:{callee}."):
            out.setdefault(e.resource.split(".", 1)[1], []).append(
                (e.value, e.detail, e.access, e.condition)
            )
    return out


class ComputeApiFamilyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = write_tree(Path(cls._tmp.name), FILES)
        cls.db = effects_db.build(cls.root, families=["compute_api"])

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def fn(self, name, path_part=None):
        found = [
            f
            for f in self.db.by_name(name)
            if f.layer == "compute_api" and (path_part is None or path_part in f.path)
        ]
        self.assertEqual(len(found), 1, [f.key for f in found])
        return found[0]

    # ---- 1. add_tiles: folded reconfig + init, acc_to_dest forwarded ------------------------- #
    def test_add_tiles_fold_markers_and_forwarded_param(self) -> None:
        fn = self.fn("add_tiles")
        self.assertEqual(fn.role, "execute")
        markers = {e.resource: e for e in by_pattern(fn.effects, "compute_api.fold")}
        self.assertEqual(
            sorted(markers),
            [
                "param:fold.add_tiles.add_init",
                "param:fold.add_tiles.reconfig_data_format",
            ],
        )
        init = markers["param:fold.add_tiles.add_init"]
        self.assertEqual(init.value, ".init")
        self.assertEqual(init.detail, ".fwd=.icb0,.icb1,.acc_to_dest .pin= .dflt=")
        self.assertEqual(init.evidence, f"{API}/eltwise_binary.h:17")
        reconfig = markers["param:fold.add_tiles.reconfig_data_format"]
        self.assertEqual(reconfig.value, ".reconfig")
        self.assertEqual(
            reconfig.detail,
            ".fwd=.icb0_new_operand,.icb1_new_operand .pin= .dflt=.src_order",
        )
        self.assertEqual(
            bindings_of(fn.effects, "add_init", "compute_api.fold_bind"),
            {
                "acc_to_dest": [("acc_to_dest", "explicit", "write", "")],
                "icb0": [("icb0", "explicit", "write", "")],
                "icb1": [("icb1", "explicit", "write", "")],
            },
        )
        # the execute callee is a read binding with the pinned LoFi fidelity
        execute = bindings_of(fn.effects, "llk_math_eltwise_binary", "compute_api.bind")
        self.assertEqual(
            execute["math_fidelity"], [("MathFidelity::LoFi", "explicit", "read", "")]
        )
        self.assertEqual(execute["dst_index"], [("idst", "explicit", "read", "")])
        self.assertTrue(
            all(
                e.thread == "math"
                for e in fn.effects
                if "llk_math_eltwise_binary." in e.resource
            )
        )

    def test_add_tiles_transitive_binding_depends_on_param(self) -> None:
        fn = self.fn("add_tiles")
        effects = self.db.effects(fn.key)
        acc = [
            e
            for e in effects
            if e.resource == "param:llk_math_eltwise_binary_init.acc_to_dest"
        ]
        self.assertEqual(
            [(e.value, e.value_kind, e.params) for e in acc],
            [("acc_to_dest", "param", ("acc_to_dest",))],
        )
        self.assertEqual(
            [step[0].split("(")[0] for step in acc[0].via],
            ["ckernel::add_init", "ckernel::binary_tiles_init<bool,EltwiseBinaryType>"],
        )
        # a call that passes acc_to_dest=true binds the init to true
        (call_effect,) = [
            e
            for e in self.db.call_effects(
                "add_tiles", args=("cb_a", "cb_b", "0", "0", "0", "true")
            )
            if e.resource == "param:llk_math_eltwise_binary_init.acc_to_dest"
        ]
        self.assertEqual(call_effect.value, "true")

    # ---- 2. pack_tile: relu / l1_acc state reset by the fold ---------------------------------- #
    def test_pack_tile_folded_config_calls(self) -> None:
        fn = self.fn("pack_tile")
        markers = {
            e.resource: (e.value, e.detail, e.thread)
            for e in by_pattern(fn.effects, "compute_api.fold")
        }
        self.assertEqual(
            markers,
            {
                "param:fold.pack_tile.llk_pack_init": (
                    ".init",
                    ".fwd=.pack_output .pin=.pack_mode,.zero_output,.skip_addrmod_config,.skip_packer_strides,.num_tiles .dflt=",
                    "pack",
                ),
                "param:fold.pack_tile.llk_pack_reconfig_l1_acc": (
                    ".reconfig",
                    ".fwd=.enable .pin= .dflt=",
                    "pack",
                ),
                "param:fold.pack_tile.llk_pack_relu_config": (
                    ".config",
                    ".fwd=.relu_config .pin= .dflt=",
                    "pack",
                ),
                "param:fold.pack_tile.pack_reconfig_data_format": (
                    ".reconfig",
                    ".fwd=.new_cb_id .pin= .dflt=.is_tile_dim_reconfig_en",
                    "any",
                ),
            },
        )
        self.assertEqual(
            bindings_of(fn.effects, "llk_pack_reconfig_l1_acc"),
            {"enable": [("l1_acc ? 1 : 0", "explicit", "write", "")]},
        )
        # the folded relu config writes STACC_RELU with the default ReluConfig::none() for a plain call
        relu = [
            e
            for e in self.db.call_effects("pack_tile", args=("0", "cb_out"))
            if e.resource == "param:llk_pack_relu_config.relu_config"
        ]
        self.assertEqual([e.value for e in relu], ["ReluConfig::none()"])
        hardware = {
            e.resource
            for e in self.db.effects(fn.key)
            if e.resource.startswith("CFG.STACC_RELU")
        }
        self.assertTrue(
            hardware,
            "pack_tile must reach the STACC_RELU RMW through llk_pack_relu_config",
        )

    # ---- 3. reduce_tile: branch-dependent reconfig operand order ------------------------------- #
    def test_reduce_tile_branch_conditions(self) -> None:
        fn = self.fn("reduce_tile")
        reconfig = bindings_of(
            fn.effects, "reconfig_data_format", "compute_api.fold_bind"
        )
        row = "(reduce_dim == ReduceDim::REDUCE_ROW) && (reduce_type != PoolType::MAX)"
        self.assertEqual(
            sorted(reconfig["icb0_new_operand"]),
            [
                ("icb", "explicit", "write", f"!({row})"),
                ("icb_scaler", "explicit", "write", row),
            ],
        )
        markers = sorted(e.resource for e in by_pattern(fn.effects, "compute_api.fold"))
        self.assertEqual(
            markers,
            [
                "param:fold.reduce_tile.reconfig_data_format",
                "param:fold.reduce_tile.reconfig_data_format",
                "param:fold.reduce_tile.reduce_init",
                "param:fold.reduce_tile.reduce_uninit",
            ],
        )
        uninit = [
            e
            for e in by_pattern(fn.effects, "compute_api.fold")
            if e.resource.endswith("reduce_uninit")
        ]
        self.assertEqual(
            [(e.value, e.detail) for e in uninit],
            [(".uninit", ".fwd=.icb .pin= .dflt=")],
        )
        # substitution folds the if constexpr: a REDUCE_ROW SUM call keeps only the swapped order
        effects = self.db.call_effects(
            "reduce_tile",
            template_args=("PoolType::SUM", "ReduceDim::REDUCE_ROW"),
            args=("cb_in", "cb_scaler", "0", "0", "0", "cb_out"),
        )
        first = sorted(
            e.value
            for e in effects
            if e.resource == "param:reconfig_data_format.icb0_new_operand"
        )
        self.assertEqual(first, ["cb_scaler"])

    # ---- 4. exp_tile: init_scale defaults to 1.0 inside the fold ------------------------------ #
    def test_exp_tile_init_scale_default_and_kernel_override(self) -> None:
        fn = self.fn("exp_tile")
        self.assertEqual(
            bindings_of(fn.effects, "exp_tile_init", "compute_api.fold_bind"),
            {
                "approx": [("approx", "explicit", "write", "")],
                "input_clamping": [("input_clamping", "explicit", "write", "")],
                "is_fp32_dest_acc_en": [
                    ("is_fp32_dest_acc_en", "explicit", "write", "")
                ],
                "scale": [("init_scale", "explicit", "write", "")],
            },
        )
        kernel = """
void MAIN {
    exp_tile_init<true, 0x3F000000>();
    for (std::uint32_t i = 0; i < 8; ++i) {
        exp_tile<true>(i);
    }
}
"""
        (main,) = self.db.add_sources([("ttnn/kernels/exp_demo.cpp", kernel)])
        scales = sorted(
            (e.value, e.detail, len(e.via))
            for e in self.db.effects(main.key)
            if e.resource == "param:exp_tile_init.scale"
        )
        # the kernel's explicit init (direct) and the folded re-init with the default (via exp_tile)
        self.assertEqual(
            scales, [("0x3F000000", "explicit", 0), ("0x3F800000", "explicit", 1)]
        )

    def test_exp_tile_init_binds_sfpu_callable(self) -> None:
        fn = self.fn("exp_tile_init")
        self.assertEqual(fn.role, "init")
        init_func = bindings_of(fn.effects, "llk_math_eltwise_unary_sfpu_init")[
            "init_func"
        ]
        self.assertEqual(
            init_func,
            [
                (
                    "sfpu::exp_init<approx, scale, (input_clamping == InputClamping::ClampToNegative), is_fp32_dest_acc_en>",
                    "explicit",
                    "write",
                    "",
                )
            ],
        )
        self.assertEqual(
            bindings_of(fn.effects, "llk_math_eltwise_unary_sfpu_init")["sfpu_op"],
            [("SfpuType::exponential", "explicit", "write", "")],
        )

    # ---- 5. copy_tile: transpose flags pinned to copy_init defaults --------------------------- #
    def test_copy_tile_defaults_pinned(self) -> None:
        fn = self.fn("copy_tile")
        (marker,) = [
            e
            for e in by_pattern(fn.effects, "compute_api.fold")
            if e.resource.endswith(".copy_init")
        ]
        self.assertEqual(
            marker.detail,
            ".fwd=.is_fp32_dest_acc_en,.cbid .pin= .dflt=.transpose,.transpose_within_16x16_face",
        )
        init = bindings_of(fn.effects, "copy_init")
        self.assertEqual(init["transpose"], [("0", "default", "write", "")])
        self.assertEqual(
            init["transpose_within_16x16_face"], [("false", "default", "write", "")]
        )
        self.assertNotIn("call_line", init)
        # UnpackToDestEn is a kernel-wide define consumed as a template argument
        defines = {
            (e.resource, e.value) for e in by_pattern(fn.effects, "compute_api.define")
        }
        self.assertIn(
            ("param:define.UnpackToDestEn", ".llk_unpack_A.unpack_to_dest"), defines
        )

    # ---- 6. tilize_block: expression argument from a ternary ---------------------------------- #
    def test_tilize_block_expression_binding(self) -> None:
        fn = self.fn("tilize_block")
        init = bindings_of(fn.effects, "tilize_init", "compute_api.fold_bind")
        self.assertEqual(
            init["block"],
            [("row_ct_dim ? row_ct_dim : block", "explicit", "write", "")],
        )
        markers = sorted(
            e.resource.rsplit(".", 1)[1]
            for e in by_pattern(fn.effects, "compute_api.fold")
        )
        self.assertEqual(
            markers,
            [
                "pack_reconfig_data_format",
                "reconfig_data_format_srca",
                "tilize_init",
                "tilize_uninit",
            ],
        )
        # tilize_block_body is an execute: plain read bindings, no fold marker
        body = bindings_of(fn.effects, "tilize_block_body")
        self.assertEqual(body["block"], [("block", "explicit", "read", "")])
        effects = self.db.call_effects("tilize_block", args=("cb_in", "4", "cb_out"))
        (block,) = [e for e in effects if e.resource == "param:tilize_init.block"]
        self.assertEqual(block.value, "0 ? 0 : 4")

    # ---- 7. matmul dynamic throttle: conditional init and SW shadow read ----------------------- #
    def test_matmul_dynamic_throttle_conditional_init(self) -> None:
        fn = self.fn("matmul_block_math_dynamic_throttle")
        level = bindings_of(
            fn.effects, "llk_math_matmul_init", "compute_api.fold_bind"
        )["THROTTLE_LEVEL"]
        self.assertEqual(
            sorted(level),
            [
                (
                    "MM_THROTTLE",
                    "explicit",
                    "write",
                    "!(mm_throttle_en) && throttled_mop_status != 0",
                ),
                (
                    "MM_THROTTLE_MAX",
                    "explicit",
                    "write",
                    "mm_throttle_en && throttled_mop_status != 1",
                ),
            ],
        )
        reads = sorted(
            (e.resource, e.condition)
            for e in by_pattern(fn.effects, "compute_api.sw_read")
        )
        self.assertEqual(
            reads,
            [
                ("SW.throttled_mop_status", "!(mm_throttle_en)"),
                ("SW.throttled_mop_status", "mm_throttle_en"),
            ],
        )
        defines = sorted(
            {e.resource for e in by_pattern(fn.effects, "compute_api.define")}
        )
        self.assertEqual(
            defines, ["param:define.MATH_FIDELITY", "param:define.MM_THROTTLE"]
        )

    # ---- 8. hw_startup: kernel defines reach state set only outside init ---------------------- #
    def test_hw_startup_bindings_and_defines(self) -> None:
        fn = self.fn("compute_kernel_hw_startup")
        self.assertEqual(fn.role, "hw_startup")
        unpack = bindings_of(fn.effects, "llk_unpack_hw_configure")
        self.assertEqual(
            unpack["is_fp32_dest_acc_en"], [("DST_ACCUM_MODE", "explicit", "write", "")]
        )
        self.assertEqual(
            unpack["unpA_operand"], [("src_a_cb", "explicit", "write", "")]
        )
        (src_a,) = [
            e
            for e in fn.effects
            if e.resource == "param:llk_unpack_hw_configure.unpA_operand"
        ]
        self.assertEqual(
            src_a.params, ("icb0", "icb1", "src_order")
        )  # through the src_a_cb local
        defines = sorted(
            e.value
            for e in by_pattern(fn.effects, "compute_api.define")
            if e.resource == "param:define.DST_ACCUM_MODE"
        )
        self.assertEqual(
            defines,
            [
                ".llk_math_hw_configure.is_fp32_dest_acc_en",
                ".llk_math_pack_sync_init.is_fp32_dest_acc_en",
                ".llk_pack_dest_init.is_fp32_dest_acc_en",
                ".llk_pack_hw_configure.is_fp32_dest_acc_en",
                ".llk_unpack_hw_configure.is_fp32_dest_acc_en",
            ],
        )
        self.assertFalse(
            by_pattern(fn.effects, "compute_api.fold"), "hw_startup is not an execute"
        )

    def test_define_gate(self) -> None:
        fn = self.fn("tilizeA_B_reduce_init")
        gates = sorted(
            (e.resource, e.value) for e in by_pattern(fn.effects, "compute_api.define")
        )
        self.assertEqual(
            gates,
            [("param:define.REDUCE_DIM", ".gate"), ("param:define.REDUCE_OP", ".gate")],
        )

    # ---- 9. ordering of RISC-side cfg stores ---------------------------------------------------- #
    def test_weighted_reduce_unpack_ordered_by_stallwait(self) -> None:
        fn = self.fn("weighted_reduce_unpack_impl")
        facts = sorted(
            (e.resource, e.value, e.evidence, e.condition, e.thread)
            for e in by_pattern(fn.effects, "compute_api.order")
        )
        lib = f"{LLK_LIB}/llk_unpack_common.h"
        self.assertEqual(
            facts,
            [
                (
                    "param:order.CFG.THCON_SEC0_REG3_Base_address",
                    ".ordered",
                    f"{lib}:9",
                    "0 == unp_cfg_context",
                    "unpack",
                ),
                (
                    "param:order.CFG.THCON_SEC0_REG3_Base_cntx1_address",
                    ".ordered",
                    f"{lib}:14",
                    "!(0 == unp_cfg_context)",
                    "unpack",
                ),
                (
                    "param:order.CFG.THCON_SEC1_REG3_Base_address",
                    ".ordered",
                    f"{lib}:10",
                    "0 == unp_cfg_context",
                    "unpack",
                ),
                (
                    "param:order.CFG.THCON_SEC1_REG3_Base_cntx1_address",
                    ".ordered",
                    f"{lib}:15",
                    "!(0 == unp_cfg_context)",
                    "unpack",
                ),
            ],
        )
        (row,) = [
            r
            for r in compute_api.ordering_facts(self.db)
            if r["function"] == "weighted_reduce_unpack_impl"
            and r["resource"].endswith("SEC0_REG3_Base_address")
        ]
        wr = f"{API}/experimental/sdpa_weighted_reduce.h"
        self.assertEqual(
            (row["fence"], row["use"], row["at"]), (f"{wr}:19", f"{wr}:21", f"{wr}:17")
        )

    def test_unfenced_store_is_unordered(self) -> None:
        fn = self.fn("weighted_reduce_unpack_unfenced")
        verdicts = sorted(
            {e.value for e in by_pattern(fn.effects, "compute_api.order")}
        )
        self.assertEqual(verdicts, [".unordered"])
        (row,) = [
            r
            for r in compute_api.ordering_facts(self.db)
            if r["function"] == "weighted_reduce_unpack_unfenced"
            and r["resource"] == "CFG.THCON_SEC1_REG3_Base_address"
        ]
        self.assertEqual(row["fence"], "")

    def test_order_facts_branch_and_replay_rules(self) -> None:
        """Synthetic event sequences: a sibling-branch consumer does not decide the verdict."""
        Event = compute_api.Event
        store = Event("store", "cfg", "CFG.X", "a.h:1", "c", "unpack", -1)
        branch_use = Event("use", "UNPACR_NOP", "", "a.h:2", "c && zero", "unpack", -1)
        branch_fence = Event("fence", "cfg", "", "a.h:3", "c && !(zero)", "unpack", -1)
        main_use = Event("use", "UNPACR", "", "a.h:4", "c && !(zero)", "unpack", -1)
        fd = self.fn("weighted_reduce_unpack_impl").definition
        (fact,) = compute_api.order_facts(
            fd, [store, branch_use, branch_fence, main_use]
        )
        self.assertEqual((fact.verdict, fact.use.evidence), ("conditional", "a.h:2"))
        fenced = Event("fence", "all", "", "a.h:3", "", "unpack", -1)
        (fact,) = compute_api.order_facts(
            fd, [store, fenced, Event("use", "UNPACR", "", "a.h:4", "", "unpack", -1)]
        )
        self.assertEqual(fact.verdict, "ordered")
        # config-only instructions (SETC16, ATGETM) and other threads are not consumers
        (fact,) = compute_api.order_facts(
            fd,
            [
                store,
                Event("use", "SETC16", "", "a.h:5", "", "unpack", -1),
                Event("use", "UNPACR", "", "a.h:6", "", "pack", -1),
                Event("use", "UNPACR", "", "a.h:7", "", "unpack", -1),
            ],
        )
        self.assertEqual((fact.verdict, fact.use.evidence), ("unordered", "a.h:7"))

    # ---- 10. roles ---------------------------------------------------------------------------- #
    def test_roles(self) -> None:
        self.assertEqual(self.fn("pack_reconfig_l1_acc").role, "config")
        self.assertEqual(self.fn("pack_relu_config").role, "config")
        self.assertEqual(self.fn("state_configure").role, "helper")
        self.assertEqual(self.fn("reduce_uninit").role, "uninit")
        self.assertEqual(self.fn("reconfig_data_format_srca").role, "reconfig")
        self.assertEqual(self.fn("tilize_block_body").role, "execute")

    # ---- 11. checker helpers ------------------------------------------------------------------ #
    def test_folded_executes_and_outside_init_state(self) -> None:
        rows = compute_api.folded_executes(self.db)
        pack = [
            r
            for r in rows
            if r["execute"] == "pack_tile" and r["callee"] == "llk_pack_relu_config"
        ]
        self.assertEqual(len(pack), 1)
        self.assertEqual(pack[0]["forwarded"], ["relu_config"])
        self.assertEqual(
            pack[0]["values"], {"relu_config": {"source": "explicit", "value": "relu"}}
        )
        self.assertEqual(
            sorted({r["execute"] for r in rows}),
            [
                "add_tiles",
                "copy_tile",
                "exp_tile",
                "matmul_block_math_dynamic_throttle",
                "pack_tile",
                "reduce_tile",
                "tilize_block",
            ],
        )
        state = {r["resource"]: r for r in compute_api.outside_init_state(self.db)}
        relu = [r for name, r in state.items() if name.startswith("CFG.STACC_RELU")]
        self.assertTrue(relu)
        self.assertEqual(relu[0]["writers"], ["pack_relu_config"])
        self.assertEqual(relu[0]["folded_executes"], ["pack_tile"])
        self.assertEqual(relu[0]["init_writers_via_setup"], [])
        self.assertIn("param:define.REDUCE_OP", state)

    def test_effects_json_is_deterministic(self) -> None:
        first = effects_db.dumps_effects(self.db.to_json())
        second = effects_db.dumps_effects(
            effects_db.build(self.root, families=["compute_api"]).to_json()
        )
        self.assertEqual(first, second)
        self.assertNotIn(str(self.root), first)

    def test_canonical_value(self) -> None:
        canon = compute_api.canonical_value
        self.assertEqual(canon("::SfpuType::exponential"), "SfpuType::exponential")
        self.assertEqual(
            canon("sfpu::exp_init< a, (x > y), b >"), "sfpu::exp_init<a, (x > y), b>"
        )
        self.assertEqual(canon("f<g<1> >"), "f<g<1>>")
        for text in ("a < b ? x : y", "x > y", "x << 2", "l1_acc ? 1 : 0"):
            self.assertEqual(canon(text), text)

    def test_location_codec(self) -> None:
        location = "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_A.h:42"
        token = compute_api.encode_location(location)
        self.assertEqual(
            token, "tt_metal::tt$llk::tt_llk_blackhole::llk_lib::llk_unpack_A.h:42"
        )
        self.assertEqual(compute_api.decode_location(token), location)


if __name__ == "__main__":
    unittest.main()
