// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <settings.hpp>
#include <build.hpp>
#include <iostream>
#include <string>

namespace tt::tt_metal {

JitBuildOptions::JitBuildOptions(const JitBuildEnv& env) :
    build_env(env), fp32_dest_acc_en(false), bfp8_pack_precise(false) {}

void JitBuildOptions::set_name(const string& n) {
    name = n;
    path = build_env.get_out_kernel_root_path() + n;
}

void JitBuildOptions::set_hlk_math_fidelity_all_cores(MathFidelity math_fidelity) {
    hlk_desc.set_hlk_math_fidelity(math_fidelity);
}

void JitBuildOptions::set_hlk_math_approx_mode_all_cores(bool approx_mode) {
    hlk_desc.set_hlk_math_approx_mode(approx_mode);
}

void JitBuildOptions::set_hlk_args_all_cores(void* args, size_t size) { hlk_desc.set_hlk_args(args, size); }

void JitBuildOptions::set_cb_dataformat_all_cores(CBIndex cb_id, DataFormat data_format) {
    set_hlk_operand_dataformat_all_cores((HlkOperand)cb_id, data_format);
}

void JitBuildOptions::set_cb_tile_dims_all_cores(
    CBIndex cb_id,
    uint32_t num_faces,
    uint32_t partial_face,
    uint32_t face_r_dim,
    uint32_t narrow_tile,
    uint32_t tile_r_dim,
    uint32_t tile_c_dim) {
    hlk_desc.set_buf_num_faces((int)cb_id, num_faces);
    hlk_desc.set_buf_partial_face((int)cb_id, partial_face);
    hlk_desc.set_buf_face_r_dim((int)cb_id, face_r_dim);
    hlk_desc.set_buf_narrow_tile((int)cb_id, narrow_tile);
    hlk_desc.set_buf_tile_r_dim((int)cb_id, tile_r_dim);
    hlk_desc.set_buf_tile_c_dim((int)cb_id, tile_c_dim);
}

void JitBuildOptions::set_cb_tile_size_all_cores(CBIndex cb_id, uint32_t tile_size) {
    hlk_desc.set_buf_tile_size((int)cb_id, tile_size);
}

void JitBuildOptions::set_hlk_operand_dataformat_all_cores(HlkOperand op_id, DataFormat data_format) {
    hlk_desc.set_buf_dataformat((int)op_id, data_format);
}

}  // namespace tt::tt_metal
