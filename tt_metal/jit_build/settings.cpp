// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/common/assert.hpp"
#include "tt_metal/common/core_coord.h"
#include "jit_build/settings.hpp"
#include "jit_build/build.hpp"
#include <iostream>
#include <string>

namespace tt::tt_metal
{

    JitBuildOptions::JitBuildOptions(const JitBuildEnv& env) :
      build_env(env),
      fp32_dest_acc_en(false),
      preserve_fp32_precision(false) {}

    void JitBuildOptions::set_name(const string& n)
    {
        name = n;
        path = build_env.get_out_kernel_root_path() + n;
    }

    void JitBuildOptions::set_hlk_file_name_all_cores(std::string file_name)
    {
        hlk_desc.set_hlk_file_name(file_name);
    }

    void JitBuildOptions::set_hlk_math_fidelity_all_cores(MathFidelity math_fidelity)
    {
        hlk_desc.set_hlk_math_fidelity(math_fidelity);
    }

    void JitBuildOptions::set_hlk_math_approx_mode_all_cores(bool approx_mode)
    {
        hlk_desc.set_hlk_math_approx_mode(approx_mode);
    }

    void JitBuildOptions::set_hlk_args_all_cores(void *args, size_t size)
    {
        hlk_desc.set_hlk_args(args, size);
    }

    void JitBuildOptions::set_cb_dataformat_all_cores(CB cb_id, DataFormat data_format) {
        set_hlk_operand_dataformat_all_cores((HlkOperand)cb_id, data_format);
    }

    void JitBuildOptions::set_hlk_operand_dataformat_all_cores(HlkOperand op_id, DataFormat data_format)
    {
        static_assert(HlkOperand::in7 == int(HlkOperand::param0)-1);
        static_assert(HlkOperand::param7 == int(HlkOperand::out0)-1);
        static_assert(HlkOperand::out7 == int(HlkOperand::intermed0)-1);
        if (op_id <= HlkOperand::in7) {
            hlk_desc.set_input_buf_dataformat((int)op_id, data_format);
        } else if (op_id <= HlkOperand::param7) {
            hlk_desc.set_param_buf_dataformat((int)op_id - ((int)HlkOperand::in7+1), data_format);
        } else if (op_id <= HlkOperand::out7) {
            hlk_desc.set_output_buf_dataformat((int)op_id - ((int)HlkOperand::param7+1), data_format);
        } else if (op_id <= HlkOperand::intermed7) {
            hlk_desc.set_intermediate_buf_dataformat((int)op_id - ((int)HlkOperand::out7+1), data_format);
        } else {
            std::cout << "Error: incorrect operand identifier" << std::endl;
            TT_ASSERT(false);
        }
    }

} // end namespace tt
