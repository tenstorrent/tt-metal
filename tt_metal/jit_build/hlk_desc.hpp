// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include "hostdevcommon/kernel_structs.h"
#include "tt_metal/common/base_types.hpp"
#include "tt_metal/common/tt_backend_api_types.hpp"
#include "tt_metal/common/assert.hpp"
#include "tt_metal/common/utils.hpp"

namespace tt
{
/**
 * @brief A descriptor of the high-level kernel. Contains circular buffer formats, HLK filename, HLK args ptr/size.
*/

class tt_hlk_desc
{
    private:
    // data formats spec for the I/O operands (i.e., buffers)
    MathFidelity math_fidelity;
    bool approximation_mode;

    std::string hlk_file_name; // HLK kernel file name (user writes)
    void *hlk_args; // void ptr to user-defined hlk_args_t struct (user writes)
    size_t hlk_args_size; // size of hlk_args_t in bytes (result of sizeof())

    public:
    DataFormat input_buf_dataformat_arr[8];
    DataFormat param_buf_dataformat_arr[8];
    DataFormat output_buf_dataformat_arr[8];
    DataFormat intermediate_buf_dataformat_arr[8];

    tt_hlk_desc()
    {
        math_fidelity = MathFidelity::Invalid;
        hlk_file_name = "";
        hlk_args = nullptr;
        hlk_args_size = 0;
        approximation_mode = true;

        for (int i = 0; i < 8; ++i)
        {
            input_buf_dataformat_arr[i] = DataFormat::Invalid;
            param_buf_dataformat_arr[i] = DataFormat::Invalid;
            output_buf_dataformat_arr[i] = DataFormat::Invalid;
            intermediate_buf_dataformat_arr[i] = DataFormat::Invalid;
        }
    }

    tt_hlk_desc(tt_hlk_desc &in)
    {
        for(int i=0;i<8;++i)
        {
            input_buf_dataformat_arr[i]  = in.input_buf_dataformat_arr[i] ;
            param_buf_dataformat_arr[i] = in.param_buf_dataformat_arr[i] ;
            output_buf_dataformat_arr[i] = in.output_buf_dataformat_arr[i];
            intermediate_buf_dataformat_arr[i] = in.intermediate_buf_dataformat_arr[i];
        }

        math_fidelity = in.math_fidelity;
        hlk_file_name = in.hlk_file_name;
        hlk_args = in.hlk_args;
        hlk_args_size = in.hlk_args_size;
        approximation_mode = in.approximation_mode;
    }

    DataFormat get_input_buf_dataformat(int buf_idx) const
    {
        return input_buf_dataformat_arr[buf_idx];
    }

    void set_input_buf_dataformat(int buf_idx, DataFormat data_format)
    {
        input_buf_dataformat_arr[buf_idx] = data_format;
    }

    DataFormat get_param_buf_dataformat(int buf_idx) const
    {
        return param_buf_dataformat_arr[buf_idx];
    }

    void set_param_buf_dataformat(int buf_idx, DataFormat data_format)
    {
        param_buf_dataformat_arr[buf_idx] = data_format;
    }

    DataFormat get_output_buf_dataformat(int buf_idx) const
    {
        return output_buf_dataformat_arr[buf_idx];
    }

    void set_output_buf_dataformat(int buf_idx, DataFormat data_format)
    {
        output_buf_dataformat_arr[buf_idx] = data_format;
    }

    DataFormat get_intermediate_buf_dataformat(int buf_idx) const
    {
        return intermediate_buf_dataformat_arr[buf_idx];
    }

    void set_intermediate_buf_dataformat(int buf_idx, DataFormat data_format)
    {
        intermediate_buf_dataformat_arr[buf_idx] = data_format;
    }

    void set_hlk_args(void* args, size_t size)
    {
        hlk_args = args;
        hlk_args_size = size;
    }

    void* get_hlk_args() const
    {
        return hlk_args;
    }

    void set_hlk_file_name(std::string file_name)
    {
        hlk_file_name = file_name;
    }

    const std::string & get_hlk_file_name() const
    {
        return hlk_file_name;
    }

    void set_hlk_math_fidelity(MathFidelity math_fi)
    {
        math_fidelity = math_fi;
    }

    MathFidelity get_hlk_math_fidelity() const
    {
        return math_fidelity;
    }

    void set_hlk_math_approx_mode(bool approx_mode)
    {
        approximation_mode = approx_mode;
    }

    bool get_hlk_math_approx_mode() const
    {
        return approximation_mode;
    }

    // rk: added by fw-dma-test-2 team
    size_t get_hlk_args_size() const
    {
        return hlk_args_size;
    }

    const DataFormat* get_input_buf_dataformats() const
    {
        return input_buf_dataformat_arr;
    }

    const DataFormat* get_param_buf_dataformats() const
    {
        return param_buf_dataformat_arr;
    }

    const DataFormat* get_output_buf_dataformats() const
    {
        return output_buf_dataformat_arr;
    }

    const DataFormat* get_intermediate_buf_dataformats() const
    {
        return intermediate_buf_dataformat_arr;
    }
};  // tt_hlk_desc
}  // namespace tt

// Hash for hlk_args
inline void hash_hlk_args(size_t& seed, void *hlk_args, size_t hlk_args_size) {
    char buffer[hlk_args_size];
    memcpy(buffer, hlk_args, hlk_args_size);

    for (int i = 0; i < hlk_args_size; i++) {
        tt::utils::hash_combine(seed, std::hash<char>{}(buffer[i]));
    }
}

template<>
struct std::hash<tt::tt_hlk_desc>
{
    std::size_t operator()(tt::tt_hlk_desc const& obj) const noexcept
    {
        std::size_t hash_value = 0;
        for (int i = 0; i < 8; i++)
        {
            tt::utils::hash_combine(hash_value, hash<tt::DataFormat>{}(obj.get_input_buf_dataformat(i)));
            tt::utils::hash_combine(hash_value, hash<tt::DataFormat>{}(obj.get_param_buf_dataformat(i)));
            tt::utils::hash_combine(hash_value, hash<tt::DataFormat>{}(obj.get_output_buf_dataformat(i)));
            tt::utils::hash_combine(hash_value, hash<tt::DataFormat>{}(obj.get_intermediate_buf_dataformat(i)));
        }
        tt::utils::hash_combine(hash_value, hash<MathFidelity>{}(obj.get_hlk_math_fidelity()));
        tt::utils::hash_combine(hash_value, hash<bool>{}(obj.get_hlk_math_approx_mode()));

        // Get hash for hlk_args here
        void *hlk_args = obj.get_hlk_args();
        size_t hlk_args_size = obj.get_hlk_args_size();
        if (hlk_args != nullptr and hlk_args_size > 0)
        {
            hash_hlk_args(hash_value, hlk_args, hlk_args_size);
        }
        else if (hlk_args == nullptr and hlk_args_size == 0)
        {
        }
        else
        {
            TT_ASSERT("Mismatching values, either hlk_args == nullptr and hlk_args_size == 0 or hlk_args != nullptr and hlk_args_size > 0!");
        }

        return hash_value;
    }
};
