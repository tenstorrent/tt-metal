// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>

#include "hlk_desc.hpp"
#include "hostdevcommon/kernel_structs.h"

namespace tt::tt_metal {

class JitBuildEnv;

class JitBuildOptions {
public:
    // general config
    const JitBuildEnv& build_env;
    std::string name;
    std::string path;

    // HLK config
    tt::tt_hlk_desc hlk_desc;

    // We can keep for future WH support, otherwise not used in GS
    bool fp32_dest_acc_en;
    std::vector<UnpackToDestMode> unpack_to_dest_mode;
    bool bfp8_pack_precise;

    bool dst_full_sync_en;

    JitBuildOptions(const JitBuildEnv& env);
    void set_name(const std::string& name);

    void set_hlk_math_fidelity_all_cores(MathFidelity math_fidelity);
    void set_hlk_math_approx_mode_all_cores(bool approx_mode);
    void set_hlk_args_all_cores(void* args, size_t size);

    void set_cb_dataformat_all_cores(CBIndex cb_id, DataFormat data_format);
    void set_cb_tile_dims_all_cores(
        CBIndex cb_id,
        uint32_t num_faces,
        uint32_t partial_face,
        uint32_t face_r_dim,
        uint32_t narrow_tile,
        uint32_t tile_r_dim,
        uint32_t tile_c_dim);
    void set_cb_tile_size_all_cores(CBIndex cb_id, uint32_t tile_size);
    // old API name
    void set_hlk_operand_dataformat_all_cores(HlkOperand op_id, DataFormat data_format);
};

}  // namespace tt::tt_metal
