// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/complex/complex_ops.hpp"
#include "tt_dnn/op_library/concat/concat_op.hpp"
#include "tt_dnn/op_library/bmm/bmm_op.hpp"
#include "tt_dnn/op_library/reshape/reshape_op.hpp"
#include "tt_dnn/op_library/split/split_last_dim_two_chunks_tiled.hpp"
#include "tt_numpy/functions.hpp"
#include "tt_eager/tensor/tensor_utils.hpp"

namespace tt {

namespace tt_metal {

//TODO: add profiling hooks

#define CHECK_FOR_COMPLEX(input) do {\
  TT_ASSERT( utility::is_complex_shape(input), "works for complex shape only"); \
  /* TT_ASSERT( input.get_legacy_shape()[0] == 1, "tensor should have batch size 1"); */ \
  } while(0);

Tensor mk_complex(const Tensor& input_r, const Tensor& input_i, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> inputs = {input_r,input_i};
    return concat(inputs, -1, output_mem_config);
}

namespace utility {
    bool is_complex_shape(const Tensor& input) {
        const Shape& shape = input.get_legacy_shape();
        return shape[-1]%(2*TILE_WIDTH) == 0; //last dim should be partitionable
    }
}


Tensor _is_real(const Tensor& input, const MemoryConfig& output_mem_config) {
    CHECK_FOR_COMPLEX(input);
    std::vector<Tensor> ab = split_last_dim_two_chunks_tiled(input,output_mem_config);
    return eqz(ab[1],output_mem_config); //imaginary portion = 0
}
Tensor is_real(const Tensor& input, const MemoryConfig& output_mem_config) {
    return operation::decorate_as_composite(__func__, _is_real)(input, output_mem_config);
}

Tensor is_imag(const Tensor& input, const MemoryConfig& output_mem_config) {
    CHECK_FOR_COMPLEX(input);
    std::vector<Tensor> ab = split_last_dim_two_chunks_tiled(input,output_mem_config);
    return eqz(ab[0],output_mem_config); //real portion = 0
}

Tensor real(const Tensor& input, const MemoryConfig& output_mem_config) {
	CHECK_FOR_COMPLEX(input);
    std::vector<Tensor> ab = split_last_dim_two_chunks_tiled(input,output_mem_config);
    return ab[0];
}

Tensor imag(const Tensor& input, const MemoryConfig& output_mem_config) {
    CHECK_FOR_COMPLEX(input);
    std::vector<Tensor> ab = split_last_dim_two_chunks_tiled(input,output_mem_config);
    return ab[1];
}

Tensor conj(const Tensor& input, const MemoryConfig& output_mem_config) {
    CHECK_FOR_COMPLEX(input);
    std::vector<Tensor> ab = split_last_dim_two_chunks_tiled(input,output_mem_config);
    return mk_complex(ab[0],neg(ab[1],output_mem_config));
}

Tensor complex_abs(const Tensor& input, const MemoryConfig& output_mem_config) {
    CHECK_FOR_COMPLEX(input);
    std::vector<Tensor> ab = split_last_dim_two_chunks_tiled(input,output_mem_config);
    return hypot(ab[0],ab[1],output_mem_config);
}

Tensor complex_recip(const Tensor& input, const MemoryConfig& output_mem_config) {
    CHECK_FOR_COMPLEX(input);
    std::vector<Tensor> ab = split_last_dim_two_chunks_tiled(input,output_mem_config);

    Tensor a_plus_b = add(ab[0],ab[1],{},output_mem_config);
    Tensor a_minus_b = sub(ab[0],ab[1],{},output_mem_config);
    Tensor asqr_plus_bsqr = add(square(ab[0],output_mem_config),square(ab[1],output_mem_config),
                                {},output_mem_config);
    Tensor inv_dr = recip( asqr_plus_bsqr, output_mem_config );
    Tensor conj_im = mul( neg(ab[1],output_mem_config), inv_dr, {}, output_mem_config);
    Tensor conj_re = mul( ab[0], inv_dr, {}, output_mem_config);
    return mk_complex( conj_re, conj_im, output_mem_config );
}

Tensor complex_add(const Tensor& input_a, const Tensor& input_b,  const MemoryConfig& output_mem_config) {
    return add(input_a,input_b,{},output_mem_config);
}

Tensor complex_sub(const Tensor& input_a, const Tensor& input_b,  const MemoryConfig& output_mem_config) {
    return sub(input_a,input_b,{},output_mem_config);
}

Tensor complex_mul(const Tensor& input_a, const Tensor& input_b,  const MemoryConfig& output_mem_config) {
    CHECK_FOR_COMPLEX(input_a);
    CHECK_FOR_COMPLEX(input_b);
    std::vector<Tensor> ab = split_last_dim_two_chunks_tiled(input_a,output_mem_config);
    std::vector<Tensor> cd = split_last_dim_two_chunks_tiled(input_b,output_mem_config);
    Tensor re_part = sub( mul(ab[0],cd[0],{},output_mem_config), mul(ab[1],cd[1],{},output_mem_config), {}, output_mem_config );
    Tensor im_part = add( mul(ab[0],cd[1],{},output_mem_config), mul(ab[1],cd[0],{},output_mem_config), {}, output_mem_config );
    return mk_complex( re_part, im_part, output_mem_config);
}


// z_a/z_b = z_a*recip(z_b) = z_a*conj(z_b)/(z_b*conj(z_b))
Tensor complex_div(const Tensor& input_a, const Tensor& input_b,  const MemoryConfig& output_mem_config) {
    CHECK_FOR_COMPLEX(input_a);
    CHECK_FOR_COMPLEX(input_b);
    return complex_mul( input_a, complex_recip( input_b , output_mem_config ), output_mem_config  );
}

// theta = /_x + iy = atan2(y,x)
Tensor angle(const Tensor& input, const MemoryConfig& output_mem_config) {
    CHECK_FOR_COMPLEX(input);
    std::vector<Tensor> ab = split_last_dim_two_chunks_tiled(input,output_mem_config);
    return neg( atan2(ab[1],ab[0],output_mem_config), output_mem_config );
}

#undef CHECK_FOR_COMPLEX

///// type-2 implementation ////
Tensor is_real(const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    return eqz( input[1], output_mem_config);
}

Tensor is_imag(const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    return eqz( input[0], output_mem_config);
}

Tensor real(const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    return input[0];
}

Tensor imag(const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    return input[1];
}

ComplexTensor conj(const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    return ComplexTensor({input[0], neg(input[1],output_mem_config)});
}

Tensor angle(const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    return neg( atan2(input[1],input[0],output_mem_config), output_mem_config );
}

Tensor complex_abs(const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    return hypot(input[0],input[1],output_mem_config);
}

ComplexTensor complex_mul(const ComplexTensor& ab, const ComplexTensor& cd,  const MemoryConfig& output_mem_config) {
    // (a + ib)*(c + id) = (ac - bd) + i(bc + ad)
    Tensor re_part = sub( mul(ab[0],cd[0],{},output_mem_config), mul(ab[1],cd[1],{},output_mem_config), {}, output_mem_config );
    Tensor im_part = add( mul(ab[0],cd[1],{},output_mem_config), mul(ab[1],cd[0],{},output_mem_config), {}, output_mem_config );
    return ComplexTensor({ re_part, im_part });
}

ComplexTensor complex_div(const ComplexTensor& input_a, const ComplexTensor& input_b,  const MemoryConfig& output_mem_config) {
    return complex_mul( input_a, complex_recip( input_b , output_mem_config ), output_mem_config  );
}

ComplexTensor complex_recip(const ComplexTensor& ab, const MemoryConfig& output_mem_config) {
    Tensor a_plus_b = add(ab[0],ab[1],{},output_mem_config);
    Tensor a_minus_b = sub(ab[0],ab[1],{},output_mem_config);
    Tensor asqr_plus_bsqr = add(square(ab[0],output_mem_config),square(ab[1],output_mem_config),
                                {},output_mem_config);
    Tensor inv_dr = recip( asqr_plus_bsqr, output_mem_config );
    Tensor conj_im = mul( neg(ab[1],output_mem_config), inv_dr, {}, output_mem_config);
    Tensor conj_re = mul( ab[0], inv_dr, {}, output_mem_config);
    return ComplexTensor({ conj_re, conj_im});
}

ComplexTensor complex_add(const ComplexTensor& input_a, const ComplexTensor& input_b, const MemoryConfig& output_mem_config) {
    return ComplexTensor({ add(input_a[0],input_b[0],{},output_mem_config),
             add(input_a[1],input_b[1],{},output_mem_config) });
}

ComplexTensor complex_sub(const ComplexTensor& input_a, const ComplexTensor& input_b, const MemoryConfig& output_mem_config) {
    return ComplexTensor({ sub(input_a[0],input_b[0],{},output_mem_config),
             sub(input_a[1],input_b[1],{},output_mem_config) });
}

// level-1 type polar
Tensor polar(const Tensor& input_a, const Tensor& input_b, const MemoryConfig& output_mem_config) {
    Tensor c = cos(input_b,output_mem_config);
    Tensor r = mul(input_a,c,{},output_mem_config);
    c.deallocate();

    Tensor s = sin(input_b,output_mem_config);
    Tensor i = mul(input_a,s,{},output_mem_config);
    s.deallocate();
    return mk_complex( r, i, output_mem_config);
}

ComplexTensor polar(const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    const Tensor& input_a = input.real();
    const Tensor& input_b = input.imag();
    Tensor c = cos(input_b,output_mem_config);
    Tensor r = mul(input_a,c,{},output_mem_config);
    c.deallocate();

    Tensor s = sin(input_b,output_mem_config);
    Tensor i = mul(input_a,s,{},output_mem_config);
    s.deallocate();

    return ComplexTensor({r,i});
}

}//namespace tt_metal

}//namespace tt
