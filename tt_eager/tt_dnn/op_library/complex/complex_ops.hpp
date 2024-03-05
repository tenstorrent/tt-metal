// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

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

Tensor complex_add(const Tensor& input_a, const Tensor& input_b,  const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
Tensor complex_sub(const Tensor& input_a, const Tensor& input_b,  const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

Tensor complex_abs(const Tensor& input, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
Tensor complex_mul(const Tensor& input_a, const Tensor& input_b,  const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
Tensor complex_div(const Tensor& input_a, const Tensor& input_b,  const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
Tensor complex_recip(const Tensor& input, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

//////// 2-tensor representation without split-concat and associated dimensional restrictions ////////

class ComplexTensor {
    private:
        std::array<Tensor,2> m_real_imag;

    public:

        ComplexTensor(std::array<Tensor,2> val): m_real_imag(val) {
            TT_ASSERT( m_real_imag[0].get_legacy_shape() == m_real_imag[1].get_legacy_shape() , "Tensor shapes of real and imag should be identical");
        }

        const Tensor& operator[](uint32_t index) const {
            return m_real_imag[index];
        }

        const Tensor& real() const {
            return m_real_imag[0];
        }

        const Tensor& imag() const {
            return m_real_imag[1];
        }

        void deallocate() {
            m_real_imag[0].deallocate();
            m_real_imag[1].deallocate();
        }
};

Tensor is_real(const ComplexTensor& input, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
Tensor is_imag(const ComplexTensor& input, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
Tensor real(const ComplexTensor& input, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
Tensor imag(const ComplexTensor& input, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

ComplexTensor conj(const ComplexTensor& input, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
Tensor angle(const ComplexTensor& input, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

ComplexTensor complex_add(const ComplexTensor& input_a, const ComplexTensor& input_b, const MemoryConfig& output_mem_config);
ComplexTensor complex_sub(const ComplexTensor& input_a, const ComplexTensor& input_b, const MemoryConfig& output_mem_config);

Tensor complex_abs(const ComplexTensor& input, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
ComplexTensor complex_mul(const ComplexTensor& input_a, const ComplexTensor& input_b,  const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
ComplexTensor complex_div(const ComplexTensor& input_a, const ComplexTensor& input_b,  const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
ComplexTensor complex_recip(const ComplexTensor& input, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

//polar operator: return a complex value tensor
Tensor polar(const Tensor& input_a, const Tensor& input_b, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
ComplexTensor polar(const ComplexTensor& input, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

} //namespace tt_metal

} //namespace tt
