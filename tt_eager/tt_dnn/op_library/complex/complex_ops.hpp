/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <array>
#include "tt_dnn/op_library/composite/composite_ops.hpp"

namespace tt {

namespace tt_metal {

/**
 * Representation:
 *
 * support complex tensors as N,H,W,C rank-4 tensor with last dim of size divisible by 2 to represent
 * real and imaginary components 0:N/2 being real and N/2:N being imaginary.
 */

namespace utility {
    bool is_complex_shape(const Tensor& input);
}

// make complex
Tensor mk_complex(const Tensor& input_r, const Tensor& input_i, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

Tensor is_real(const Tensor& input, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
Tensor is_imag(const Tensor& input, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
Tensor real(const Tensor& input, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
Tensor imag(const Tensor& input, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

//Tensor pol2cart(const Tensor& input, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
//Tensor cart2pol(const Tensor& input, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
Tensor conj(const Tensor& input, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
Tensor angle(const Tensor& input, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

inline
Tensor complex_add(const Tensor& input_a, const Tensor& input_b,  const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    return add(input_a,input_b,{},output_mem_config);
}

inline
Tensor complex_sub(const Tensor& input_a, const Tensor& input_b,  const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    return sub(input_a,input_b,{},output_mem_config);
}

Tensor complex_abs(const Tensor& input, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
Tensor complex_mul(const Tensor& input_a, const Tensor& input_b,  const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
Tensor complex_div(const Tensor& input_a, const Tensor& input_b,  const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
Tensor complex_recip(const Tensor& input, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

//////// 2-tensor representation without split-concat and associated dimensional restrictions ////////


struct ComplexTensor {
    std::array<Tensor,2> m_real_imag;
    ComplexTensor(std::array<Tensor,2> val): m_real_imag(val) {
      TT_ASSERT( m_real_imag[0].shape() == m_real_imag[1].shape() , "Tensor shapes of real and imag should be identical");
    }

    Tensor operator[](uint32_t index) const {
        return m_real_imag[index];
    }

    Tensor get_real() {
        return m_real_imag[0];
    }
    Tensor get_imag() {
        return m_real_imag[1];
    }
};


ComplexTensor type2_mk_complex(const Tensor& input_r, const Tensor& input_i);


Tensor type2_is_real(const ComplexTensor& input, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
Tensor type2_is_imag(const ComplexTensor& input, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
Tensor type2_real(const ComplexTensor& input, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
Tensor type2_imag(const ComplexTensor& input, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);


ComplexTensor type2_conj(const ComplexTensor& input, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
Tensor type2_angle(const ComplexTensor& input, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

inline
ComplexTensor type2_complex_add(ComplexTensor& input_a,  ComplexTensor& input_b,  const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    return ComplexTensor({ add(input_a[0],input_b[0],{},output_mem_config),
             add(input_a[1],input_b[1],{},output_mem_config) });
}

inline
ComplexTensor type2_complex_sub(ComplexTensor& input_a,  ComplexTensor& input_b,  const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    return ComplexTensor({ sub(input_a[0],input_b[0],{},output_mem_config),
             sub(input_a[1],input_b[1],{},output_mem_config) });
}


Tensor type2_complex_abs(const ComplexTensor& input, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
ComplexTensor type2_complex_mul(const ComplexTensor& input_a, const ComplexTensor& input_b,  const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
ComplexTensor type2_complex_div(const ComplexTensor& input_a, const ComplexTensor& input_b,  const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
ComplexTensor type2_complex_recip(const ComplexTensor& input, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);


} //namespace tt_metal

} //namespace tt
