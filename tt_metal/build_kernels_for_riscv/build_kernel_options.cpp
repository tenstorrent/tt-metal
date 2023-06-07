#include "common/assert.hpp"
#include "common/core_coord.h"
#include "build_kernels_for_riscv/build_kernel_options.hpp"
#include <iostream>
#include <string>

namespace tt
{

    build_kernel_for_riscv_options_t::build_kernel_for_riscv_options_t() :
      name(""),
      outpath(get_firmware_compile_outpath()),
      fp32_dest_acc_en(false),
      fw_build_(true) {}

    build_kernel_for_riscv_options_t::build_kernel_for_riscv_options_t(std::string type, std::string name) :
      name(name),
      outpath(get_kernel_compile_outpath()),
      fp32_dest_acc_en(false),
      fw_build_(false) {}

    void build_kernel_for_riscv_options_t::set_hlk_file_name_all_cores(std::string file_name)
    {
        hlk_desc.set_hlk_file_name(file_name);
    }

    void build_kernel_for_riscv_options_t::set_hlk_math_fidelity_all_cores(MathFidelity math_fidelity)
    {
        hlk_desc.set_hlk_math_fidelity(math_fidelity);
    }

    void build_kernel_for_riscv_options_t::set_hlk_math_approx_mode_all_cores(bool approx_mode)
    {
        hlk_desc.set_hlk_math_approx_mode(approx_mode);
    }

    void build_kernel_for_riscv_options_t::set_hlk_args_all_cores(void *args, size_t size)
    {
        hlk_desc.set_hlk_args(args, size);
    }

    void build_kernel_for_riscv_options_t::set_cb_dataformat_all_cores(CB cb_id, DataFormat data_format) {
        set_hlk_operand_dataformat_all_cores((HlkOperand)cb_id, data_format);
    }

    void build_kernel_for_riscv_options_t::set_hlk_operand_dataformat_all_cores(HlkOperand op_id, DataFormat data_format)
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
