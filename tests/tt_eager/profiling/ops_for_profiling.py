# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import tt_lib
import ttnn


def subalpha(x, y):
    tt_lib.tensor.subalpha(x, y, 5)


def addalpha(x, y):
    tt_lib.tensor.addalpha(x, y, 5)


def isclose(x, y):
    tt_lib.tensor.isclose(x, y, rtol=0.00001, atol=0.0000001)


def where_binary_1(x, y):
    tt_lib.tensor.where(x, 5, y)


def where_binary_2(x, y):
    tt_lib.tensor.where(x, y, 5)


def bcast_add_h(x, y):
    tt_lib.tensor.bcast(x, y, tt_lib.tensor.BcastOpMath.ADD, tt_lib.tensor.BcastOpDim.H)


def bcast_add_w(x, y):
    tt_lib.tensor.bcast(x, y, tt_lib.tensor.BcastOpMath.ADD, tt_lib.tensor.BcastOpDim.W)


def bcast_add_hw(x, y):
    tt_lib.tensor.bcast(x, y, tt_lib.tensor.BcastOpMath.ADD, tt_lib.tensor.BcastOpDim.HW)


def bcast_sub_h(x, y):
    tt_lib.tensor.bcast(x, y, tt_lib.tensor.BcastOpMath.SUB, tt_lib.tensor.BcastOpDim.H)


def bcast_sub_w(x, y):
    tt_lib.tensor.bcast(x, y, tt_lib.tensor.BcastOpMath.SUB, tt_lib.tensor.BcastOpDim.W)


def bcast_sub_hw(x, y):
    tt_lib.tensor.bcast(x, y, tt_lib.tensor.BcastOpMath.SUB, tt_lib.tensor.BcastOpDim.HW)


def bcast_mul_h(x, y):
    tt_lib.tensor.bcast(x, y, tt_lib.tensor.BcastOpMath.MUL, tt_lib.tensor.BcastOpDim.H)


def bcast_mul_w(x, y):
    tt_lib.tensor.bcast(x, y, tt_lib.tensor.BcastOpMath.MUL, tt_lib.tensor.BcastOpDim.W)


def bcast_mul_hw(x, y):
    tt_lib.tensor.bcast(x, y, tt_lib.tensor.BcastOpMath.MUL, tt_lib.tensor.BcastOpDim.HW)


def bcast_h_shape_func(input_shape):
    input_shape_1 = [input_shape[-4], input_shape[-3], 32, input_shape[-1]]
    return input_shape, input_shape_1


def bcast_w_shape_func(input_shape):
    input_shape_1 = [input_shape[-4], input_shape[-3], input_shape[-2], 32]
    return input_shape, input_shape_1


def bcast_hw_shape_func(input_shape):
    input_shape_1 = [input_shape[-4], input_shape[-3], 32, 32]
    return input_shape, input_shape_1


def complex_add(x, y):
    tt_lib.tensor.complex_add(
        x, y, tt_lib.tensor.MemoryConfig(tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM)
    )


def complex_sub(x, y):
    tt_lib.tensor.complex_sub(
        x, y, tt_lib.tensor.MemoryConfig(tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM)
    )


def complex_mul(x, y):
    tt_lib.tensor.complex_mul(
        x, y, tt_lib.tensor.MemoryConfig(tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM)
    )


def complex_div(x, y):
    tt_lib.tensor.complex_div(
        x, y, tt_lib.tensor.MemoryConfig(tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM)
    )


def polar_binary(x, y):
    tt_lib.tensor.polar(
        x, y, tt_lib.tensor.MemoryConfig(tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM)
    )


def concat_0(x, y):
    tt_lib.tensor.concat([x, y], 0)


def concat_1(x, y):
    tt_lib.tensor.concat([x, y], 1)


def concat_2(x, y):
    tt_lib.tensor.concat([x, y], 2)


def concat_3(x, y):
    tt_lib.tensor.concat([x, y], 3)


def lerp_binary(x, y):
    tt_lib.tensor.lerp(x, y, 0.7)


def embeddings_shape_func(input_shape):
    input_shape_0 = (input_shape[0], 1, 1, input_shape[-1])
    input_shape_1 = (input_shape[0], 1, 1, input_shape[-1])
    return input_shape_0, input_shape_1


def unary_mul_bw(x, y):
    tt_lib.tensor.unary_mul_bw(x, y, 3)


def unary_add_bw(x, y):
    tt_lib.tensor.unary_add_bw(x, y, 3)


def unary_div_bw(x, y):
    tt_lib.tensor.unary_div_bw(x, y, 3)


def rdiv_bw(x, y):
    tt_lib.tensor.rdiv_bw(x, y, 3)


def unary_pow_bw(x, y):
    tt_lib.tensor.unary_pow_bw(x, y, 3)


def clamp_bw(x, y):
    tt_lib.tensor.clamp_bw(x, y, 0.1, 0.9)


def clamp_min_bw(x, y):
    tt_lib.tensor.clamp_min_bw(x, y, 0.1)


def clamp_max_bw(x, y):
    tt_lib.tensor.clamp_max_bw(x, y, 0.9)


def gelu_bw_none(x, y):
    tt_lib.tensor.gelu_bw(x, y, approximate="none")


def gelu_bw_tanh(x, y):
    tt_lib.tensor.gelu_bw(x, y, approximate="tanh")


def bias_gelu_unary_bw_none(x, y):
    tt_lib.tensor.bias_gelu_unary_bw(x, y, bias=3.1, approximate="None")


def bias_gelu_unary_bw_tanh(x, y):
    tt_lib.tensor.bias_gelu_unary_bw(x, y, bias=3.1, approximate="tanh")


def softplus_bw(x, y):
    tt_lib.tensor.softplus_bw(x, y, beta=2, threshold=10)


def polygamma_bw(x, y):
    tt_lib.tensor.polygamma_bw(x, y, n=3)


def elu_bw(x, y):
    tt_lib.tensor.elu_bw(x, y, alpha=0.7)


def hardtanh_bw(x, y):
    tt_lib.tensor.hardtanh_bw(x, y, min=-0.8, max=0.8)


def rpow_bw(x, y):
    tt_lib.tensor.rpow_bw(x, y, exponent=3.1)


def threshold_bw(x, y):
    tt_lib.tensor.threshold_bw(x, y, threshold=0.7, value=10)


def unary_eq_bw(x, y):
    tt_lib.tensor.unary_eq_bw(x, y, other=0.7)


def logiteps_bw(x, y):
    tt_lib.tensor.logiteps_bw(x, y, eps=0.0001)


def unary_fmod_bw(x, y):
    tt_lib.tensor.unary_fmod_bw(x, y, scalar=0.5)


def unary_remainder_bw(x, y):
    tt_lib.tensor.unary_remainder_bw(x, y, scalar=0.5)


def repeat_bw(x, y):
    tt_lib.tensor.repeat_bw(x, y, shape=(1, 1, 1, 4))


def repeat_bw_shape_func(input_shape):
    input_shape_1 = [1, 1, input_shape[2], input_shape[3]]
    return input_shape, input_shape_1


def unary_div_no_nan_bw(x, y):
    tt_lib.tensor.unary_div_no_nan_bw(x, y, scalar=3.0)


def mseloss(x, y):
    tt_lib.tensor.mseloss(
        x,
        y,
        reduce_mode=tt_lib.tensor.LossReductionMode.SUM,
        output_mem_config=tt_lib.tensor.MemoryConfig(
            tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM
        ),
    )


def maeloss(x, y):
    tt_lib.tensor.maeloss(
        x,
        y,
        reduce_mode=tt_lib.tensor.LossReductionMode.SUM,
        output_mem_config=tt_lib.tensor.MemoryConfig(
            tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM
        ),
    )


def primary_moreh_softmax_backward_0(x, y):
    tt_lib.operations.primary.moreh_softmax_backward(x, y, dim=0)


def primary_moreh_softmax_backward_1(x, y):
    tt_lib.operations.primary.moreh_softmax_backward(x, y, dim=1)


def primary_moreh_softmax_backward_2(x, y):
    tt_lib.operations.primary.moreh_softmax_backward(x, y, dim=2)


def primary_moreh_softmax_backward_3(x, y):
    tt_lib.operations.primary.moreh_softmax_backward(x, y, dim=3)


def primary_moreh_softmin_backward_0(x, y):
    tt_lib.operations.primary.moreh_softmin_backward(x, y, dim=0)


def primary_moreh_softmin_backward_1(x, y):
    tt_lib.operations.primary.moreh_softmin_backward(x, y, dim=1)


def primary_moreh_softmin_backward_2(x, y):
    tt_lib.operations.primary.moreh_softmin_backward(x, y, dim=2)


def primary_moreh_softmin_backward_3(x, y):
    tt_lib.operations.primary.moreh_softmin_backward(x, y, dim=3)


def primary_moreh_logsoftmax_backward_0(x, y):
    tt_lib.operations.primary.moreh_logsoftmax_backward(x, y, dim=0)


def primary_moreh_logsoftmax_backward_1(x, y):
    tt_lib.operations.primary.moreh_logsoftmax_backward(x, y, dim=1)


def primary_moreh_logsoftmax_backward_2(x, y):
    tt_lib.operations.primary.moreh_logsoftmax_backward(x, y, dim=2)


def primary_moreh_logsoftmax_backward_3(x, y):
    tt_lib.operations.primary.moreh_logsoftmax_backward(x, y, dim=3)


def primary_scale_mask_softmax_in_place(x, y):
    ttnn.scale_mask_softmax_in_place(x, scale=3.3, mask=y)


def scale_mask_softmax_in_place_shape_func(input_shape):
    return input_shape, [1, 1, input_shape[-2], input_shape[-1]]


def primaru_moreh_mean_0(x, y):
    tt_lib.operations.primary.moreh_mean(x, y, dims=[0])


def primaru_moreh_mean_0_shape_func(input_shape):
    return input_shape, [1, input_shape[1], input_shape[2], input_shape[3]]


def primaru_moreh_mean_01(x, y):
    tt_lib.operations.primary.moreh_mean(x, y, dims=[0, 1])


def primaru_moreh_mean_01_shape_func(input_shape):
    return input_shape, [1, 1, input_shape[2], input_shape[3]]


def primaru_moreh_mean_012(x, y):
    tt_lib.operations.primary.moreh_mean(x, y, dims=[0, 1, 2])


def primaru_moreh_mean_012_shape_func(input_shape):
    return input_shape, [1, 1, 1, input_shape[3]]


def primaru_moreh_mean_0123(x, y):
    tt_lib.operations.primary.moreh_mean(x, y, dims=[0, 1, 2, 3])


def primaru_moreh_mean_0123_shape_func(input_shape):
    return input_shape, [1, 1, 1, 1]


def primaru_moreh_mean_013(x, y):
    tt_lib.operations.primary.moreh_mean(x, y, dims=[0, 1, 3])


def primaru_moreh_mean_013_shape_func(input_shape):
    return input_shape, [1, 1, input_shape[2], 1]


def primaru_moreh_mean_023(x, y):
    tt_lib.operations.primary.moreh_mean(x, y, dims=[0, 2, 3])


def primaru_moreh_mean_023_shape_func(input_shape):
    return input_shape, [1, input_shape[1], 1, 1]


def primaru_moreh_mean_1(x, y):
    tt_lib.operations.primary.moreh_mean(x, y, dims=[1])


def primaru_moreh_mean_1_shape_func(input_shape):
    return input_shape, [input_shape[0], 1, input_shape[2], input_shape[3]]


def primaru_moreh_mean_12(x, y):
    tt_lib.operations.primary.moreh_mean(x, y, dims=[1, 2])


def primaru_moreh_mean_12_shape_func(input_shape):
    return input_shape, [input_shape[0], 1, 1, input_shape[3]]


def primaru_moreh_mean_123(x, y):
    tt_lib.operations.primary.moreh_mean(x, y, dims=[1, 2, 3])


def primaru_moreh_mean_123_shape_func(input_shape):
    return input_shape, [input_shape[0], 1, 1, 1]


def primaru_moreh_mean_13(x, y):
    tt_lib.operations.primary.moreh_mean(x, y, dims=[1, 3])


def primaru_moreh_mean_13_shape_func(input_shape):
    return input_shape, [input_shape[0], 1, input_shape[2], 1]


def primaru_moreh_mean_2(x, y):
    tt_lib.operations.primary.moreh_mean(x, y, dims=[2])


def primaru_moreh_mean_2_shape_func(input_shape):
    return input_shape, [input_shape[0], input_shape[1], 1, input_shape[3]]


def primaru_moreh_mean_23(x, y):
    tt_lib.operations.primary.moreh_mean(x, y, dims=[2, 3])


def primaru_moreh_mean_23_shape_func(input_shape):
    return input_shape, [input_shape[0], input_shape[1], 1, 1]


def primaru_moreh_mean_3(x, y):
    tt_lib.operations.primary.moreh_mean(x, y, dims=[3])


def primaru_moreh_mean_3_shape_func(input_shape):
    return input_shape, [input_shape[0], input_shape[1], input_shape[2], 1]


def angle_bw(x, y):
    tt_lib.tensor.angle_bw(x, y, False)


all_binary_ops = [
    {
        "op": ttnn.add,
        "name": "ttnn.add",
    },
    {
        "op": ttnn.sub,
        "name": "ttnn.sub",
    },
    {
        "op": ttnn.mul,
        "name": "ttnn.mul",
    },
    {
        "op": tt_lib.tensor.div,
        "name": "tt_lib.tensor.div",
    },
    {
        "op": tt_lib.tensor.hypot,
        "name": "tt_lib.tensor.hypot",
    },
    {
        "op": ttnn.squared_difference,
        "name": "ttnn.squared_difference",
    },
    {
        "op": ttnn.logaddexp,
        "name": "ttnn.logaddexp",
    },
    {
        "op": ttnn.logaddexp2,
        "name": "ttnn.logaddexp2",
    },
    {
        "op": tt_lib.tensor.atan2,
        "name": "tt_lib.tensor.atan2",
    },
    {
        "op": ttnn.logical_xor,
        "name": "ttnn.logical_xor",
    },
    {
        "op": subalpha,
        "name": "tt_lib.tensor.subalpha",
    },
    {
        "op": addalpha,
        "name": "tt_lib.tensor.addalpha",
    },
    {
        "op": ttnn.ldexp,
        "name": "ttnn.ldexp",
    },
    {
        "op": ttnn.bias_gelu,
        "name": "ttnn.bias_gelu",
    },
    {
        "op": ttnn.logical_and,
        "name": "ttnn.logical_and",
    },
    {
        "op": tt_lib.tensor.assign,
        "name": "tt_lib.tensor.assign_binary",
    },
    {
        "op": isclose,
        "name": "tt_lib.tensor.isclose",
    },
    {
        "op": ttnn.logical_or,
        "name": "ttnn.logical_or",
    },
    {
        "op": ttnn.gt,
        "name": "ttnn.gt",
    },
    {
        "op": ttnn.ge,
        "name": "ttnn.ge",
    },
    {
        "op": ttnn.lt,
        "name": "ttnn.lt",
    },
    {
        "op": ttnn.le,
        "name": "ttnn.le",
    },
    {
        "op": ttnn.eq,
        "name": "ttnn.eq",
    },
    {
        "op": ttnn.ne,
        "name": "ttnn.ne",
    },
    {
        "op": where_binary_1,
        "name": "tt_lib.tensor.where_binary_x_const_y",
    },
    {
        "op": where_binary_2,
        "name": "tt_lib.tensor.where_binary_x_y_const",
    },
    {
        "op": ttnn.matmul,
        "name": "ttnn.matmul",
    },
    {
        "op": tt_lib.tensor.copy,
        "name": "tt_lib.tensor.copy",
    },
    {
        "op": bcast_add_h,
        "name": "tt_lib.tensor.bcast_add_h",
        "shape_func": bcast_h_shape_func,
    },
    {
        "op": bcast_add_w,
        "name": "tt_lib.tensor.bcast_add_w",
        "shape_func": bcast_w_shape_func,
    },
    {
        "op": bcast_add_hw,
        "name": "tt_lib.tensor.bcast_add_hw",
        "shape_func": bcast_hw_shape_func,
    },
    {
        "op": bcast_sub_h,
        "name": "tt_lib.tensor.bcast_sub_h",
        "shape_func": bcast_h_shape_func,
    },
    {
        "op": bcast_sub_w,
        "name": "tt_lib.tensor.bcast_sub_w",
        "shape_func": bcast_w_shape_func,
    },
    {
        "op": bcast_sub_hw,
        "name": "tt_lib.tensor.bcast_sub_hw",
        "shape_func": bcast_hw_shape_func,
    },
    {
        "op": bcast_mul_h,
        "name": "tt_lib.tensor.bcast_mul_h",
        "shape_func": bcast_h_shape_func,
    },
    {
        "op": bcast_mul_w,
        "name": "tt_lib.tensor.bcast_mul_w",
        "shape_func": bcast_w_shape_func,
    },
    {
        "op": bcast_mul_hw,
        "name": "tt_lib.tensor.bcast_mul_hw",
        "shape_func": bcast_hw_shape_func,
    },
    {
        "op": complex_add,
        "name": "tt_lib.tensor.complex_add",
    },
    {
        "op": complex_sub,
        "name": "tt_lib.tensor.complex_sub",
    },
    {
        "op": complex_mul,
        "name": "tt_lib.tensor.complex_mul",
    },
    {
        "op": complex_div,
        "name": "tt_lib.tensor.complex_div",
    },
    {
        "op": polar_binary,
        "name": "tt_lib.tensor.polar_binary",
    },
    {
        "op": concat_0,
        "name": "tt_lib.tensor.concat_dim_0",
    },
    {
        "op": concat_1,
        "name": "tt_lib.tensor.concat_dim_1",
    },
    {
        "op": concat_2,
        "name": "tt_lib.tensor.concat_dim_2",
    },
    {
        "op": concat_3,
        "name": "tt_lib.tensor.concat_dim_3",
    },
    {
        "op": lerp_binary,
        "name": "tt_lib.tensor.lerp_binary",
    },
    {
        "op": tt_lib.tensor.xlogy,
        "name": "tt_lib.tensor.xlogy",
    },
    {
        "op": tt_lib.tensor.embeddings,
        "name": "tt_lib.tensor.embeddings",
        "layout": "ROW_MAJOR",
        "shape_func": embeddings_shape_func,
    },
    {
        "op": tt_lib.tensor.nextafter,
        "name": "tt_lib.tensor.nextafter",
    },
    {
        "op": tt_lib.tensor.conj_bw,
        "name": "tt_lib.tensor.conj_bw",
    },
    {
        "op": unary_mul_bw,
        "name": "tt_lib.tensor.unary_mul_bw",
    },
    {
        "op": unary_add_bw,
        "name": "tt_lib.tensor.unary_add_bw",
    },
    {
        "op": tt_lib.tensor.unary_assign_bw,
        "name": "tt_lib.tensor.unary_assign_bw",
    },
    {
        "op": unary_div_bw,
        "name": "tt_lib.tensor.unary_div_bw",
    },
    {
        "op": rdiv_bw,
        "name": "tt_lib.tensor.rdiv_bw",
    },
    {
        "op": tt_lib.tensor.sqrt_bw,
        "name": "tt_lib.tensor.sqrt_bw",
    },
    {
        "op": tt_lib.tensor.tan_bw,
        "name": "tt_lib.tensor.tan_bw",
    },
    {
        "op": tt_lib.tensor.exp_bw,
        "name": "tt_lib.tensor.exp_bw",
    },
    {
        "op": tt_lib.tensor.exp2_bw,
        "name": "tt_lib.tensor.exp2_bw",
    },
    {
        "op": tt_lib.tensor.expm1_bw,
        "name": "tt_lib.tensor.expm1_bw",
    },
    {
        "op": unary_pow_bw,
        "name": "tt_lib.tensor.unary_pow_bw",
    },
    {
        "op": tt_lib.tensor.tanh_bw,
        "name": "tt_lib.tensor.tanh_bw",
    },
    {
        "op": tt_lib.tensor.unary_sub_bw,
        "name": "tt_lib.tensor.unary_sub_bw",
    },
    {
        "op": tt_lib.tensor.log_bw,
        "name": "tt_lib.tensor.log_bw",
    },
    {
        "op": tt_lib.tensor.abs_bw,
        "name": "tt_lib.tensor.abs_bw",
    },
    {
        "op": tt_lib.tensor.rsqrt_bw,
        "name": "tt_lib.tensor.rsqrt_bw",
    },
    {
        "op": tt_lib.tensor.neg_bw,
        "name": "tt_lib.tensor.neg_bw",
    },
    {
        "op": tt_lib.tensor.relu_bw,
        "name": "tt_lib.tensor.relu_bw",
    },
    {
        "op": clamp_bw,
        "name": "tt_lib.tensor.clamp_bw",
    },
    {
        "op": clamp_min_bw,
        "name": "tt_lib.tensor.clamp_min_bw",
    },
    {
        "op": clamp_max_bw,
        "name": "tt_lib.tensor.clamp_max_bw",
    },
    {
        "op": gelu_bw_none,
        "name": "tt_lib.tensor.gelu_bw_none",
    },
    {
        "op": gelu_bw_tanh,
        "name": "tt_lib.tensor.gelu_bw_tanh",
    },
    {
        "op": bias_gelu_unary_bw_none,
        "name": "tt_lib.tensor.bias_gelu_unary_bw_none",
    },
    {
        "op": bias_gelu_unary_bw_tanh,
        "name": "tt_lib.tensor.bias_gelu_unary_bw_tanh",
    },
    {
        "op": tt_lib.tensor.hardsigmoid_bw,
        "name": "tt_lib.tensor.hardsigmoid_bw",
    },
    {
        "op": tt_lib.tensor.i0_bw,
        "name": "tt_lib.tensor.i0_bw",
    },
    {
        "op": tt_lib.tensor.hardshrink_bw,
        "name": "tt_lib.tensor.hardshrink_bw",
    },
    {
        "op": tt_lib.tensor.softshrink_bw,
        "name": "tt_lib.tensor.softshrink_bw",
    },
    {
        "op": tt_lib.tensor.hardswish_bw,
        "name": "tt_lib.tensor.hardswish_bw",
    },
    {
        "op": softplus_bw,
        "name": "tt_lib.tensor.softplus_bw",
    },
    {
        "op": polygamma_bw,
        "name": "tt_lib.tensor.polygamma_bw",
        "num_repeats": 3,
    },
    {
        "op": tt_lib.tensor.atan_bw,
        "name": "tt_lib.tensor.atan_bw",
    },
    {
        "op": tt_lib.tensor.atanh_bw,
        "name": "tt_lib.tensor.atanh_bw",
    },
    {
        "op": tt_lib.tensor.asin_bw,
        "name": "tt_lib.tensor.asin_bw",
    },
    {
        "op": tt_lib.tensor.asinh_bw,
        "name": "tt_lib.tensor.asinh_bw",
    },
    {
        "op": tt_lib.tensor.cosh_bw,
        "name": "tt_lib.tensor.cosh_bw",
        "num_repeats": 3,
    },
    {
        "op": tt_lib.tensor.cos_bw,
        "name": "tt_lib.tensor.cos_bw",
    },
    {
        "op": tt_lib.tensor.acosh_bw,
        "name": "tt_lib.tensor.acosh_bw",
    },
    {
        "op": tt_lib.tensor.acos_bw,
        "name": "tt_lib.tensor.acos_bw",
    },
    {
        "op": tt_lib.tensor.erfinv_bw,
        "name": "tt_lib.tensor.erfinv_bw",
    },
    {
        "op": tt_lib.tensor.leaky_relu_bw,
        "name": "tt_lib.tensor.leaky_relu_bw",
    },
    {
        "op": elu_bw,
        "name": "tt_lib.tensor.elu_bw",
    },
    {
        "op": hardtanh_bw,
        "name": "tt_lib.tensor.hardtanh_bw",
    },
    {
        "op": tt_lib.tensor.sin_bw,
        "name": "tt_lib.tensor.sin_bw",
    },
    {
        "op": tt_lib.tensor.sinh_bw,
        "name": "tt_lib.tensor.sinh_bw",
    },
    {
        "op": tt_lib.tensor.celu_bw,
        "name": "tt_lib.tensor.celu_bw",
    },
    {
        "op": tt_lib.tensor.log10_bw,
        "name": "tt_lib.tensor.log10_bw",
    },
    {
        "op": tt_lib.tensor.log1p_bw,
        "name": "tt_lib.tensor.log1p_bw",
    },
    {
        "op": tt_lib.tensor.erf_bw,
        "name": "tt_lib.tensor.erf_bw",
    },
    {
        "op": tt_lib.tensor.erfc_bw,
        "name": "tt_lib.tensor.erfc_bw",
    },
    {
        "op": tt_lib.tensor.digamma_bw,
        "name": "tt_lib.tensor.digamma_bw",
        "num_repeats": 2,
    },
    {
        "op": tt_lib.tensor.deg2rad_bw,
        "name": "tt_lib.tensor.deg2rad_bw",
    },
    {
        "op": tt_lib.tensor.rad2deg_bw,
        "name": "tt_lib.tensor.rad2deg_bw",
    },
    {
        "op": tt_lib.tensor.reciprocal_bw,
        "name": "tt_lib.tensor.reciprocal_bw",
    },
    {
        "op": tt_lib.tensor.relu6_bw,
        "name": "tt_lib.tensor.relu6_bw",
    },
    {
        "op": rpow_bw,
        "name": "tt_lib.tensor.rpow_bw",
    },
    {
        "op": tt_lib.tensor.silu_bw,
        "name": "tt_lib.tensor.silu_bw",
    },
    {
        "op": tt_lib.tensor.selu_bw,
        "name": "tt_lib.tensor.selu_bw",
    },
    {
        "op": tt_lib.tensor.square_bw,
        "name": "tt_lib.tensor.square_bw",
    },
    {
        "op": tt_lib.tensor.lgamma_bw,
        "name": "tt_lib.tensor.lgamma_bw",
    },
    {
        "op": tt_lib.tensor.trunc_bw,
        "name": "tt_lib.tensor.trunc_bw",
    },
    {
        "op": tt_lib.tensor.frac_bw,
        "name": "tt_lib.tensor.frac_bw",
    },
    {
        "op": tt_lib.tensor.log_sigmoid_bw,
        "name": "tt_lib.tensor.log_sigmoid_bw",
    },
    {
        "op": tt_lib.tensor.tanhshrink_bw,
        "name": "tt_lib.tensor.tanhshrink_bw",
    },
    {
        "op": threshold_bw,
        "name": "tt_lib.tensor.threshold_bw",
    },
    {
        "op": unary_eq_bw,
        "name": "tt_lib.tensor.unary_eq_bw",
    },
    {
        "op": tt_lib.tensor.logit_bw,
        "name": "tt_lib.tensor.logit_bw",
    },
    {
        "op": logiteps_bw,
        "name": "tt_lib.tensor.logiteps_bw",
    },
    {
        "op": tt_lib.tensor.softsign_bw,
        "name": "tt_lib.tensor.softsign_bw",
    },
    {
        "op": tt_lib.tensor.sign_bw,
        "name": "tt_lib.tensor.sign_bw",
    },
    {
        "op": tt_lib.tensor.ceil_bw,
        "name": "tt_lib.tensor.ceil_bw",
    },
    {
        "op": tt_lib.tensor.log2_bw,
        "name": "tt_lib.tensor.log2_bw",
    },
    {
        "op": unary_fmod_bw,
        "name": "tt_lib.tensor.unary_fmod_bw",
    },
    {
        "op": unary_remainder_bw,
        "name": "tt_lib.tensor.unary_remainder_bw",
    },
    {
        "op": tt_lib.tensor.imag_bw,
        "name": "tt_lib.tensor.imag_bw",
    },
    {
        "op": tt_lib.tensor.real_bw,
        "name": "tt_lib.tensor.real_bw",
    },
    {
        "op": tt_lib.tensor.multigammaln_bw,
        "name": "tt_lib.tensor.multigammaln_bw",
    },
    {
        "op": repeat_bw,
        "name": "tt_lib.tensor.repeat_bw",
        "shape_func": repeat_bw_shape_func,
    },
    {
        "op": unary_div_no_nan_bw,
        "name": "tt_lib.tensor.unary_div_no_nan_bw",
    },
    {
        "op": mseloss,
        "name": "tt_lib.tensor.mseloss",
    },
    {
        "op": maeloss,
        "name": "tt_lib.tensor.maeloss",
    },
    {
        "op": primary_moreh_softmax_backward_0,
        "name": "tt_lib.operations.primary.moreh_softmax_backward_dim_0",
    },
    {
        "op": primary_moreh_softmax_backward_1,
        "name": "tt_lib.operations.primary.moreh_softmax_backward_dim_1",
    },
    {
        "op": primary_moreh_softmax_backward_2,
        "name": "tt_lib.operations.primary.moreh_softmax_backward_dim_2",
    },
    {
        "op": primary_moreh_softmax_backward_3,
        "name": "tt_lib.operations.primary.moreh_softmax_backward_dim_3",
    },
    {
        "op": primary_moreh_softmin_backward_0,
        "name": "tt_lib.operations.primary.moreh_softmin_backward_dim_0",
    },
    {
        "op": primary_moreh_softmin_backward_1,
        "name": "tt_lib.operations.primary.moreh_softmin_backward_dim_1",
    },
    {
        "op": primary_moreh_softmin_backward_2,
        "name": "tt_lib.operations.primary.moreh_softmin_backward_dim_2",
    },
    {
        "op": primary_moreh_softmin_backward_3,
        "name": "tt_lib.operations.primary.moreh_softmin_backward_dim_3",
    },
    {
        "op": primary_moreh_logsoftmax_backward_0,
        "name": "tt_lib.operations.primary.moreh_logsoftmax_backward_dim_0",
    },
    {
        "op": primary_moreh_logsoftmax_backward_1,
        "name": "tt_lib.operations.primary.moreh_logsoftmax_backward_dim_1",
    },
    {
        "op": primary_moreh_logsoftmax_backward_2,
        "name": "tt_lib.operations.primary.moreh_logsoftmax_backward_dim_2",
    },
    {
        "op": primary_moreh_logsoftmax_backward_3,
        "name": "tt_lib.operations.primary.moreh_logsoftmax_backward_dim_3",
    },
    {
        "op": primary_scale_mask_softmax_in_place,
        "name": "ttnn.scale_mask_softmax_in_place",
        "shape_func": scale_mask_softmax_in_place_shape_func,
    },
    {
        "op": primaru_moreh_mean_0,
        "name": "tt_lib.operations.primary.moreh_mean_dims_0",
        "shape_func": primaru_moreh_mean_0_shape_func,
    },
    {
        "op": primaru_moreh_mean_01,
        "name": "tt_lib.operations.primary.moreh_mean_dims_01",
        "shape_func": primaru_moreh_mean_01_shape_func,
    },
    # {
    #     "op": primaru_moreh_mean_012,
    #     "name": "tt_lib.operations.primary.moreh_mean_dims_012",
    #     "shape_func": primaru_moreh_mean_012_shape_func,
    # },
    # {
    #     "op": primaru_moreh_mean_013,
    #     "name": "tt_lib.operations.primary.moreh_mean_dims_013",
    #     "shape_func": primaru_moreh_mean_013_shape_func,
    # },
    {
        "op": primaru_moreh_mean_1,
        "name": "tt_lib.operations.primary.moreh_mean_dims_1",
        "shape_func": primaru_moreh_mean_1_shape_func,
    },
    # {
    #     "op": primaru_moreh_mean_12,
    #     "name": "tt_lib.operations.primary.moreh_mean_dims_12",
    #     "shape_func": primaru_moreh_mean_12_shape_func,
    # },
    # {
    #     "op": primaru_moreh_mean_13,
    #     "name": "tt_lib.operations.primary.moreh_mean_dims_13",
    #     "shape_func": primaru_moreh_mean_13_shape_func,
    # },
    # {
    #     "op": primaru_moreh_mean_2,
    #     "name": "tt_lib.operations.primary.moreh_mean_dims_2",
    #     "shape_func": primaru_moreh_mean_2_shape_func,
    # },
    # {
    #     "op": primaru_moreh_mean_23,
    #     "name": "tt_lib.operations.primary.moreh_mean_dims_23",
    #     "shape_func": primaru_moreh_mean_23_shape_func,
    # },
    # {
    #     "op": primaru_moreh_mean_3,
    #     "name": "tt_lib.operations.primary.moreh_mean_dims_3",
    #     "shape_func": primaru_moreh_mean_3_shape_func,
    # },
    {
        "op": tt_lib.operations.primary.moreh_mean_backward,
        "name": "tt_lib.operations.primary.moreh_mean_backward",
    },
    {
        "op": angle_bw,
        "name": "tt_lib.tensor.angle_bw",
    },
]


# To make
# {
#     "op": conv,
#     "name": "tt_lib.tensor.conv",
# },


# Crashing
# {
#     "op": primaru_moreh_mean_0123,
#     "name": "tt_lib.operations.primary.moreh_mean_dims_0123",
#     "shape_func": primaru_moreh_mean_0123_shape_func,
# },
# {
#     "op": primaru_moreh_mean_023,
#     "name": "tt_lib.operations.primary.moreh_mean_dims_023",
#     "shape_func": primaru_moreh_mean_023_shape_func,
# },
# {
#     "op": primaru_moreh_mean_123,
#     "name": "tt_lib.operations.primary.moreh_mean_dims_123",
#     "shape_func": primaru_moreh_mean_123_shape_func,
# },


def add_unary(x):
    tt_lib.tensor.add_unary(x, 5.0)


def sub_unary(x):
    tt_lib.tensor.sub_unary(x, 5.0)


def mul_unary(x):
    tt_lib.tensor.mul_unary(x, 5.0)


def div_unary(x):
    tt_lib.tensor.div_unary(x, 5.0)


def relu_min(x):
    tt_lib.tensor.relu_min(x, 0.1)


def relu_max(x):
    tt_lib.tensor.relu_max(x, 0.1)


def clip(x):
    tt_lib.tensor.clip(x, 0.1, 0.9)


def polyval(x):
    tt_lib.tensor.polyval(x, [1, 2, 3])


def leaky_relu(x):
    tt_lib.tensor.leaky_relu(x, 68)


def softshrink(x):
    tt_lib.tensor.softshrink(x, 70)


def hardshrink(x):
    tt_lib.tensor.hardshrink(x, 1)


def elu(x):
    tt_lib.tensor.elu(x, 2)


def heaviside(x):
    tt_lib.tensor.heaviside(x, 0.5)


def logical_xori(x):
    tt_lib.tensor.logical_xori(x, 2)


def bias_gelu_unary(x):
    tt_lib.tensor.bias_gelu_unary(x, 2)


def logit(x):
    tt_lib.tensor.logit(x, 0.0001)


def logical_andi(x):
    tt_lib.tensor.logical_andi(x, 2)


def logical_ori(x):
    tt_lib.tensor.logical_ori(x, 2)


def polygamma(x):
    tt_lib.tensor.polygamma(x, 2)


def where_unary(x):
    tt_lib.tensor.where(x, 2, 3)


def threshold(x):
    tt_lib.tensor.threshold(x, 0.5, 3)


def reshape(x):
    shape = x.get_legacy_shape()
    tt_lib.tensor.reshape(x, shape[-4], shape[-3], shape[-1], shape[-2])


def transpose(x):
    tt_lib.tensor.transpose(x, dim0=2, dim1=3)


def permute(x):
    tt_lib.tensor.permute(x, [1, 0, 3, 2])


def tilize(x):
    tt_lib.tensor.tilize(x)


def tilize_with_val_padding(x):
    shape = x.get_legacy_shape()

    output_tensor_shape = [shape[-4], shape[-3], shape[-2] + 32, shape[-1] + 32]

    tt_lib.tensor.tilize_with_val_padding(x, output_tensor_shape, 1.0)


def untilize_with_unpadding(x):
    shape = x.get_legacy_shape()

    unpadded_shape_end = [
        shape[0] - 1,
        shape[1] - 1,
        shape[2] - 33,
        shape[3] - 33,
    ]

    tt_lib.tensor.untilize_with_unpadding(x, unpadded_shape_end)


def pad(x):
    shape = x.get_legacy_shape()

    padding = [
        (0, 0),
        (0, 0),
        (0, 32),
        (0, 32),
    ]

    ttnn.pad(x, padding, 1)


def unpad(x):
    shape = x.get_legacy_shape()

    output_tensor_end = [
        shape[0] - 1,
        shape[1] - 1,
        shape[2] - 33,
        shape[3] - 33,
    ]

    tt_lib.tensor.unpad(x, output_tensor_start=(0, 0, 0, 0), output_tensor_end=output_tensor_end)


def typecast(x):
    tt_lib.tensor.typecast(x, tt_lib.tensor.DataType.BFLOAT8_B)


def arange(x):
    tt_lib.tensor.arange(0, 100, 2, x.device())


def full(x):
    tt_lib.tensor.full(
        shape=x.get_legacy_shape(), fill_value=2, data_type=x.get_dtype(), layout=x.get_layout(), device=x.device()
    )


def full_like(x):
    tt_lib.tensor.full_like(x, 2.0)


def ones(x):
    tt_lib.tensor.ones(shape=x.get_legacy_shape(), data_type=x.get_dtype(), layout=x.get_layout(), device=x.device())


def zeros(x):
    tt_lib.tensor.zeros(shape=x.get_legacy_shape(), data_type=x.get_dtype(), layout=x.get_layout(), device=x.device())


def empty(x):
    tt_lib.tensor.empty(shape=x.get_legacy_shape(), data_type=x.get_dtype(), layout=x.get_layout(), device=x.device())


def tril(x):
    tt_lib.tensor.tril(x, 1)


def triu(x):
    tt_lib.tensor.triu(x, 1)


def reduce_sum_h(x):
    tt_lib.tensor.reduce(x, tt_lib.tensor.ReduceOpMath.SUM, tt_lib.tensor.ReduceOpDim.H, 1.0)


def reduce_sum_w(x):
    tt_lib.tensor.reduce(x, tt_lib.tensor.ReduceOpMath.SUM, tt_lib.tensor.ReduceOpDim.W, 1.0)


def reduce_sum_hw(x):
    tt_lib.tensor.reduce(x, tt_lib.tensor.ReduceOpMath.SUM, tt_lib.tensor.ReduceOpDim.HW, 1.0)


def reduce_min_h(x):
    tt_lib.tensor.reduce(x, tt_lib.tensor.ReduceOpMath.MIN, tt_lib.tensor.ReduceOpDim.H, 1.0)


def reduce_min_w(x):
    tt_lib.tensor.reduce(x, tt_lib.tensor.ReduceOpMath.MIN, tt_lib.tensor.ReduceOpDim.W, 1.0)


def reduce_min_hw(x):
    tt_lib.tensor.reduce(x, tt_lib.tensor.ReduceOpMath.MIN, tt_lib.tensor.ReduceOpDim.HW, 1.0)


def reduce_max_h(x):
    tt_lib.tensor.reduce(x, tt_lib.tensor.ReduceOpMath.MAX, tt_lib.tensor.ReduceOpDim.H, 1.0)


def reduce_max_w(x):
    tt_lib.tensor.reduce(x, tt_lib.tensor.ReduceOpMath.MAX, tt_lib.tensor.ReduceOpDim.W, 1.0)


def reduce_max_hw(x):
    tt_lib.tensor.reduce(x, tt_lib.tensor.ReduceOpMath.MAX, tt_lib.tensor.ReduceOpDim.HW, 1.0)


def rpow(x):
    tt_lib.tensor.rpow(x, 3)


def rsub(x):
    tt_lib.tensor.rsub(x, 3)


def rdiv(x):
    tt_lib.tensor.rdiv(x, 3)


def real(x):
    tt_lib.tensor.real(
        x, tt_lib.tensor.MemoryConfig(tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM)
    )


def imag(x):
    tt_lib.tensor.imag(
        x, tt_lib.tensor.MemoryConfig(tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM)
    )


def complex_abs(x):
    tt_lib.tensor.complex_abs(
        x, tt_lib.tensor.MemoryConfig(tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM)
    )


def conj(x):
    tt_lib.tensor.conj(
        x, tt_lib.tensor.MemoryConfig(tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM)
    )


def complex_recip(x):
    tt_lib.tensor.complex_recip(
        x, tt_lib.tensor.MemoryConfig(tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM)
    )


def sum_0(x):
    tt_lib.tensor.sum(x, 0)


def sum_1(x):
    tt_lib.tensor.sum(x, 1)


def sum_2(x):
    tt_lib.tensor.sum(x, 2)


def sum_3(x):
    tt_lib.tensor.sum(x, 3)


def erf_slow(x):
    tt_lib.tensor.erf(x, fast_and_approx=False)


def erfc_slow(x):
    tt_lib.tensor.erfc(x, fast_and_approx=False)


def rsqrt_slow(x):
    tt_lib.tensor.rsqrt(x, fast_and_approx=False)


def fill_rm(x):
    shape = x.get_legacy_shape()

    tt_lib.tensor.fill_rm(
        N=shape[0],
        C=shape[1],
        H=shape[2],
        W=shape[3],
        hOnes=shape[2] - 32,
        wOnes=shape[3] - 32,
        any=x,
        val_hi=10,
        val_lo=5,
    )


def fill_ones_rm(x):
    shape = x.get_legacy_shape()

    tt_lib.tensor.fill_ones_rm(
        N=shape[0], C=shape[1], H=shape[2], W=shape[3], hOnes=shape[2] - 32, wOnes=shape[3] - 32, any=x
    )


def groupnorm_no_weights(x):
    tt_lib.tensor.groupnorm(input=x, group_size=32, eps=0.0001)


def convert_conv_weight_tensor_to_tiled_layout(x):
    tt_lib.tensor.convert_conv_weight_tensor_to_tiled_layout(x, in1_block_h=32, in1_block_w=32)


def logical_noti(x):
    tt_lib.tensor.logical_noti(x, 2)


def glu_1(x):
    tt_lib.tensor.glu(x, -1)


def geglu_1(x):
    tt_lib.tensor.geglu(x, -1)


def reglu_1(x):
    tt_lib.tensor.reglu(x, -1)


def swiglu_1(x):
    tt_lib.tensor.swiglu(x, -1)


def glu_2(x):
    tt_lib.tensor.glu(x, -2)


def geglu_2(x):
    tt_lib.tensor.geglu(x, -2)


def reglu_2(x):
    tt_lib.tensor.reglu(x, -2)


def swiglu_2(x):
    tt_lib.tensor.swiglu(x, -2)


def repeat(x):
    tt_lib.tensor.repeat(x, (1, 1, 1, 4))


def repeat_interleave_0(x):
    tt_lib.tensor.repeat_interleave(x, 4, 0)


def repeat_interleave_1(x):
    tt_lib.tensor.repeat_interleave(x, 4, 1)


def repeat_interleave_2(x):
    tt_lib.tensor.repeat_interleave(x, 4, 2)


def pow_int(x):
    tt_lib.tensor.pow(x, 3)


def pow_float(x):
    tt_lib.tensor.pow(x, 3.3)


def argmax_1(x):
    tt_lib.tensor.argmax(x, dim=-1)


def argmax_2(x):
    tt_lib.tensor.argmax(x, dim=-2)


def argmax_3(x):
    tt_lib.tensor.argmax(x, dim=-3)


def argmax_4(x):
    tt_lib.tensor.argmax(x, dim=-4)


def argmax_all(x):
    tt_lib.tensor.argmax(x, dim=-1, all=True)


def argmin_1(x):
    tt_lib.tensor.argmin(x, dim=-1)


def argmin_2(x):
    tt_lib.tensor.argmin(x, dim=-2)


def argmin_3(x):
    tt_lib.tensor.argmin(x, dim=-3)


def argmin_4(x):
    tt_lib.tensor.argmin(x, dim=-4)


def argmin_all(x):
    tt_lib.tensor.argmin(x, dim=-1, all=True)


def primary_moreh_softmax_0(x):
    tt_lib.operations.primary.moreh_softmax(x, dim=0)


def primary_moreh_softmax_1(x):
    tt_lib.operations.primary.moreh_softmax(x, dim=1)


def primary_moreh_softmax_2(x):
    tt_lib.operations.primary.moreh_softmax(x, dim=2)


def primary_moreh_softmax_3(x):
    tt_lib.operations.primary.moreh_softmax(x, dim=3)


def primary_moreh_softmin_0(x):
    tt_lib.operations.primary.moreh_softmin(x, dim=0)


def primary_moreh_softmin_1(x):
    tt_lib.operations.primary.moreh_softmin(x, dim=1)


def primary_moreh_softmin_2(x):
    tt_lib.operations.primary.moreh_softmin(x, dim=2)


def primary_moreh_softmin_3(x):
    tt_lib.operations.primary.moreh_softmin(x, dim=3)


def primary_moreh_logsoftmax_0(x):
    tt_lib.operations.primary.moreh_logsoftmax(x, dim=0)


def primary_moreh_logsoftmax_1(x):
    tt_lib.operations.primary.moreh_logsoftmax(x, dim=1)


def primary_moreh_logsoftmax_2(x):
    tt_lib.operations.primary.moreh_logsoftmax(x, dim=2)


def primary_moreh_logsoftmax_3(x):
    tt_lib.operations.primary.moreh_logsoftmax(x, dim=3)


def primary_moreh_norm_0(x):
    tt_lib.operations.primary.moreh_norm(x, p=2.0, dim=0)


def primary_moreh_norm_1(x):
    tt_lib.operations.primary.moreh_norm(x, p=2.0, dim=1)


def primary_moreh_norm_2(x):
    tt_lib.operations.primary.moreh_norm(x, p=2.0, dim=2)


def primary_moreh_norm_3(x):
    tt_lib.operations.primary.moreh_norm(x, p=2.0, dim=3)


from tt_lib.fused_ops.softmax import softmax as fused_softmax


all_unary_ops = [
    {
        "op": add_unary,
        "name": "tt_lib.tensor.add_unary",
    },
    {
        "op": sub_unary,
        "name": "tt_lib.tensor.sub_unary",
    },
    {
        "op": mul_unary,
        "name": "tt_lib.tensor.mul_unary",
    },
    {
        "op": div_unary,
        "name": "tt_lib.tensor.div_unary",
    },
    {
        "op": tt_lib.tensor.gelu,
        "name": "tt_lib.tensor.gelu",
    },
    {
        "op": tt_lib.tensor.relu,
        "name": "tt_lib.tensor.relu",
    },
    {
        "op": tt_lib.tensor.relu6,
        "name": "tt_lib.tensor.relu6",
    },
    {
        "op": relu_min,
        "name": "tt_lib.tensor.relu_min",
    },
    {
        "op": relu_max,
        "name": "tt_lib.tensor.relu_max",
    },
    {
        "op": tt_lib.tensor.exp,
        "name": "tt_lib.tensor.exp",
    },
    {
        "op": tt_lib.tensor.recip,
        "name": "tt_lib.tensor.recip",
    },
    {
        "op": tt_lib.tensor.sqrt,
        "name": "tt_lib.tensor.sqrt",
    },
    {
        "op": tt_lib.tensor.log,
        "name": "tt_lib.tensor.log",
    },
    {
        "op": tt_lib.tensor.log2,
        "name": "tt_lib.tensor.log2",
    },
    {
        "op": tt_lib.tensor.log10,
        "name": "tt_lib.tensor.log10",
    },
    {
        "op": tt_lib.tensor.log1p,
        "name": "tt_lib.tensor.log1p",
    },
    {
        "op": tt_lib.tensor.tanh,
        "name": "tt_lib.tensor.tanh",
    },
    {
        "op": clip,
        "name": "tt_lib.tensor.clip",
    },
    {
        "op": tt_lib.tensor.hardtanh,
        "name": "tt_lib.tensor.hardtanh",
    },
    {
        "op": tt_lib.tensor.deg2rad,
        "name": "tt_lib.tensor.deg2rad",
    },
    {
        "op": tt_lib.tensor.rad2deg,
        "name": "tt_lib.tensor.rad2deg",
    },
    {
        "op": tt_lib.tensor.cbrt,
        "name": "tt_lib.tensor.cbrt",
    },
    {
        "op": tt_lib.tensor.softplus,
        "name": "tt_lib.tensor.softplus",
    },
    {
        "op": tt_lib.tensor.mish,
        "name": "tt_lib.tensor.mish",
    },
    {
        "op": polyval,
        "name": "tt_lib.tensor.polyval",
    },
    {
        "op": tt_lib.tensor.sign,
        "name": "tt_lib.tensor.sign",
    },
    {
        "op": tt_lib.tensor.abs,
        "name": "tt_lib.tensor.abs",
    },
    {
        "op": tt_lib.tensor.silu,
        "name": "tt_lib.tensor.silu",
    },
    {
        "op": ttnn.square,
        "name": "ttnn.square",
    },
    {
        "op": tt_lib.tensor.neg,
        "name": "tt_lib.tensor.neg",
    },
    {
        "op": tt_lib.tensor.add1,
        "name": "tt_lib.tensor.add1",
    },
    {
        "op": tt_lib.tensor.sigmoid,
        "name": "tt_lib.tensor.sigmoid",
    },
    {
        "op": tt_lib.tensor.sigmoid_accurate,
        "name": "tt_lib.tensor.sigmoid_accurate",
    },
    {
        "op": tt_lib.tensor.hardsigmoid,
        "name": "tt_lib.tensor.hardsigmoid",
    },
    {
        "op": tt_lib.tensor.swish,
        "name": "tt_lib.tensor.swish",
    },
    {
        "op": tt_lib.tensor.hardswish,
        "name": "tt_lib.tensor.hardswish",
    },
    {
        "op": leaky_relu,
        "name": "tt_lib.tensor.leaky_relu",
    },
    {
        "op": tt_lib.tensor.softsign,
        "name": "tt_lib.tensor.softsign",
    },
    {
        "op": softshrink,
        "name": "tt_lib.tensor.softshrink",
    },
    {
        "op": hardshrink,
        "name": "tt_lib.tensor.hardshrink",
    },
    {
        "op": tt_lib.tensor.cos,
        "name": "tt_lib.tensor.cos",
    },
    {
        "op": tt_lib.tensor.sin,
        "name": "tt_lib.tensor.sin",
    },
    {
        "op": tt_lib.tensor.cosh,
        "name": "tt_lib.tensor.cosh",
    },
    {
        "op": tt_lib.tensor.sinh,
        "name": "tt_lib.tensor.sinh",
    },
    {
        "op": tt_lib.tensor.acos,
        "name": "tt_lib.tensor.acos",
    },
    {
        "op": tt_lib.tensor.asin,
        "name": "tt_lib.tensor.asin",
    },
    {
        "op": elu,
        "name": "tt_lib.tensor.elu",
    },
    {
        "op": tt_lib.tensor.exp2,
        "name": "tt_lib.tensor.exp2",
    },
    {
        "op": tt_lib.tensor.tanhshrink,
        "name": "tt_lib.tensor.tanhshrink",
    },
    {
        "op": heaviside,
        "name": "tt_lib.tensor.heaviside",
    },
    {
        "op": tt_lib.tensor.atan,
        "name": "tt_lib.tensor.atan",
    },
    {
        "op": tt_lib.tensor.atanh,
        "name": "tt_lib.tensor.atanh",
    },
    {
        "op": logical_xori,
        "name": "tt_lib.tensor.logical_xori",
    },
    {
        "op": tt_lib.tensor.logical_not_unary,
        "name": "tt_lib.tensor.logical_not_unary",
    },
    {
        "op": bias_gelu_unary,
        "name": "tt_lib.tensor.bias_gelu_unary",
    },
    {
        "op": tt_lib.tensor.isfinite,
        "name": "tt_lib.tensor.isfinite",
    },
    {
        "op": tt_lib.tensor.isinf,
        "name": "tt_lib.tensor.isinf",
    },
    {
        "op": tt_lib.tensor.isposinf,
        "name": "tt_lib.tensor.isposinf",
    },
    {
        "op": tt_lib.tensor.isneginf,
        "name": "tt_lib.tensor.isneginf",
    },
    {
        "op": tt_lib.tensor.isnan,
        "name": "tt_lib.tensor.isnan",
    },
    {
        "op": logit,
        "name": "tt_lib.tensor.logit",
    },
    {
        "op": tt_lib.tensor.lgamma,
        "name": "tt_lib.tensor.lgamma",
        "num_repeats": 3,
    },
    {
        "op": logical_andi,
        "name": "tt_lib.tensor.logical_andi",
    },
    {
        "op": tt_lib.tensor.erfinv,
        "name": "tt_lib.tensor.erfinv",
    },
    {
        "op": tt_lib.tensor.multigammaln,
        "name": "tt_lib.tensor.multigammaln",
    },
    {
        "op": tt_lib.tensor.assign,
        "name": "tt_lib.tensor.assign_unary",
    },
    {
        "op": tt_lib.tensor.i0,
        "name": "tt_lib.tensor.i0",
    },
    {
        "op": tt_lib.tensor.digamma,
        "name": "tt_lib.tensor.digamma",
    },
    {
        "op": tt_lib.tensor.tan,
        "name": "tt_lib.tensor.tan",
    },
    {
        "op": logical_ori,
        "name": "tt_lib.tensor.logical_ori",
    },
    {
        "op": polygamma,
        "name": "tt_lib.tensor.polygamma",
    },
    {
        "op": tt_lib.tensor.gtz,
        "name": "tt_lib.tensor.gtz",
    },
    {
        "op": tt_lib.tensor.gez,
        "name": "tt_lib.tensor.gez",
    },
    {
        "op": tt_lib.tensor.ltz,
        "name": "tt_lib.tensor.ltz",
    },
    {
        "op": tt_lib.tensor.lez,
        "name": "tt_lib.tensor.lez",
    },
    {
        "op": tt_lib.tensor.eqz,
        "name": "tt_lib.tensor.eqz",
    },
    {
        "op": tt_lib.tensor.nez,
        "name": "tt_lib.tensor.nez",
    },
    {
        "op": where_unary,
        "name": "tt_lib.tensor.where_unary_x_const_const",
    },
    {
        "op": threshold,
        "name": "tt_lib.tensor.threshold",
    },
    {
        "op": reshape,
        "name": "tt_lib.tensor.reshape",
    },
    {
        "op": transpose,
        "name": "tt_lib.tensor.transpose",
    },
    {
        "op": permute,
        "name": "tt_lib.tensor.permute",
    },
    {
        "op": tilize,
        "name": "tt_lib.tensor.tilize",
    },
    {
        "op": tt_lib.tensor.untilize,
        "name": "tt_lib.tensor.untilize",
    },
    {
        "op": tilize_with_val_padding,
        "name": "tt_lib.tensor.tilize_with_val_padding",
        "layout": "ROW_MAJOR",
    },
    {
        "op": untilize_with_unpadding,
        "name": "tt_lib.tensor.untilize_with_unpadding",
    },
    {
        "op": tt_lib.tensor.tilize_with_zero_padding,
        "name": "tt_lib.tensor.tilize_with_zero_padding",
    },
    {
        "op": pad,
        "name": "ttnn.pad",
    },
    {
        "op": unpad,
        "name": "tt_lib.tensor.unpad",
    },
    {
        "op": tt_lib.tensor.clone,
        "name": "tt_lib.tensor.clone",
    },
    {
        "op": typecast,
        "name": "tt_lib.tensor.typecast",
    },
    {
        "op": arange,
        "name": "tt_lib.tensor.arange",
    },
    {
        "op": full,
        "name": "tt_lib.tensor.full",
    },
    {
        "op": ones,
        "name": "tt_lib.tensor.ones",
    },
    {
        "op": tt_lib.tensor.ones_like,
        "name": "tt_lib.tensor.ones_like",
    },
    {
        "op": zeros,
        "name": "tt_lib.tensor.zeros",
    },
    {
        "op": tt_lib.tensor.zeros_like,
        "name": "tt_lib.tensor.zeros_like",
    },
    {
        "op": full_like,
        "name": "tt_lib.tensor.full_like",
    },
    {
        "op": tt_lib.tensor.split_last_dim_two_chunks_tiled,
        "name": "tt_lib.tensor.split_last_dim_two_chunks_tiled",
    },
    {
        "op": empty,
        "name": "tt_lib.tensor.empty",
    },
    {
        "op": tril,
        "name": "tt_lib.tensor.tril",
        "num_repeats": 3,
    },
    {
        "op": triu,
        "name": "tt_lib.tensor.triu",
        "num_repeats": 3,
    },
    {
        "op": reduce_sum_h,
        "name": "tt_lib.tensor.reduce_sum_h",
    },
    {
        "op": reduce_sum_w,
        "name": "tt_lib.tensor.reduce_sum_w",
    },
    {
        "op": reduce_sum_hw,
        "name": "tt_lib.tensor.reduce_sum_hw",
    },
    {
        "op": reduce_min_h,
        "name": "tt_lib.tensor.reduce_min_h",
    },
    {
        "op": reduce_min_w,
        "name": "tt_lib.tensor.reduce_min_w",
    },
    {
        "op": reduce_min_hw,
        "name": "tt_lib.tensor.reduce_min_hw",
    },
    {
        "op": reduce_max_h,
        "name": "tt_lib.tensor.reduce_max_h",
    },
    {
        "op": reduce_max_w,
        "name": "tt_lib.tensor.reduce_max_w",
    },
    {
        "op": reduce_max_hw,
        "name": "tt_lib.tensor.reduce_max_hw",
    },
    {
        "op": tt_lib.tensor.global_min,
        "name": "tt_lib.tensor.global_min",
    },
    {
        "op": tt_lib.tensor.global_max,
        "name": "tt_lib.tensor.global_max",
    },
    {
        "op": tt_lib.tensor.global_sum,
        "name": "tt_lib.tensor.global_sum",
    },
    {
        "op": tt_lib.tensor.global_mean,
        "name": "tt_lib.tensor.global_mean",
    },
    {
        "op": rpow,
        "name": "tt_lib.tensor.rpow",
    },
    {
        "op": rsub,
        "name": "tt_lib.tensor.rsub",
    },
    {
        "op": rdiv,
        "name": "tt_lib.tensor.rdiv",
    },
    {
        "op": real,
        "name": "tt_lib.tensor.real",
    },
    {
        "op": imag,
        "name": "tt_lib.tensor.imag",
    },
    {
        "op": complex_abs,
        "name": "tt_lib.tensor.complex_abs",
    },
    {
        "op": conj,
        "name": "tt_lib.tensor.conj",
    },
    {
        "op": complex_recip,
        "name": "tt_lib.tensor.complex_recip",
    },
    {
        "op": sum_0,
        "name": "tt_lib.tensor.sum_dim_0",
        "num_repeats": 2,
    },
    {
        "op": sum_1,
        "name": "tt_lib.tensor.sum_dim_1",
    },
    {
        "op": sum_2,
        "name": "tt_lib.tensor.sum_dim_2",
    },
    {
        "op": sum_3,
        "name": "tt_lib.tensor.sum_dim_3",
    },
    {
        "op": tt_lib.tensor.log_sigmoid,
        "name": "tt_lib.tensor.log_sigmoid",
    },
    {
        "op": tt_lib.tensor.expm1,
        "name": "tt_lib.tensor.expm1",
    },
    {
        "op": tt_lib.tensor.asinh,
        "name": "tt_lib.tensor.asinh",
    },
    {
        "op": tt_lib.tensor.acosh,
        "name": "tt_lib.tensor.acosh",
    },
    {
        "op": tt_lib.tensor.erf,
        "name": "tt_lib.tensor.erf_fast_and_approx_True",
    },
    {
        "op": erf_slow,
        "name": "tt_lib.tensor.erf_fast_and_approx_False",
    },
    {
        "op": tt_lib.tensor.erfc,
        "name": "tt_lib.tensor.erfc_fast_and_approx_True",
    },
    {
        "op": erfc_slow,
        "name": "tt_lib.tensor.erfc_fast_and_approx_False",
    },
    {
        "op": tt_lib.tensor.rsqrt,
        "name": "tt_lib.tensor.rsqrt_fast_and_approx_True",
    },
    {
        "op": rsqrt_slow,
        "name": "tt_lib.tensor.rsqrt_fast_and_approx_False",
    },
    {
        "op": tt_lib.tensor.signbit,
        "name": "tt_lib.tensor.signbit",
    },
    {
        "op": fill_rm,
        "name": "tt_lib.tensor.fill_rm",
    },
    {
        "op": fill_ones_rm,
        "name": "tt_lib.tensor.fill_ones_rm",
    },
    {
        "op": groupnorm_no_weights,
        "name": "tt_lib.tensor.groupnorm_no_weights",
    },
    {
        "op": tt_lib.tensor.mean_hw,
        "name": "tt_lib.tensor.mean_hw",
    },
    {
        "op": tt_lib.tensor.var_hw,
        "name": "tt_lib.tensor.var_hw",
    },
    {
        "op": logical_noti,
        "name": "tt_lib.tensor.logical_noti",
    },
    {
        "op": tt_lib.tensor.std_hw,
        "name": "tt_lib.tensor.std_hw",
    },
    {
        "op": tt_lib.tensor.normalize_hw,
        "name": "tt_lib.tensor.normalize_hw",
    },
    {
        "op": tt_lib.tensor.normalize_global,
        "name": "tt_lib.tensor.normalize_global",
    },
    {
        "op": glu_1,
        "name": "tt_lib.tensor.glu_dim_3",
    },
    {
        "op": geglu_1,
        "name": "tt_lib.tensor.geglu_dim_3",
    },
    {
        "op": reglu_1,
        "name": "tt_lib.tensor.reglu_dim_3",
    },
    {
        "op": swiglu_1,
        "name": "tt_lib.tensor.swiglu_dim_3",
    },
    {
        "op": glu_2,
        "name": "tt_lib.tensor.glu_dim_2",
    },
    {
        "op": geglu_2,
        "name": "tt_lib.tensor.geglu_dim_2",
    },
    {
        "op": reglu_2,
        "name": "tt_lib.tensor.reglu_dim_2",
    },
    {
        "op": swiglu_2,
        "name": "tt_lib.tensor.swiglu_dim_2",
    },
    {
        "op": repeat,
        "name": "tt_lib.tensor.repeat",
    },
    {
        "op": repeat_interleave_0,
        "name": "tt_lib.tensor.repeat_interleave_dim_0",
    },
    {
        "op": repeat_interleave_1,
        "name": "tt_lib.tensor.repeat_interleave_dim_1",
        "num_repeats": 2,
    },
    {
        "op": repeat_interleave_2,
        "name": "tt_lib.tensor.repeat_interleave_dim_2",
        "num_repeats": 2,
    },
    {
        "op": pow_int,
        "name": "tt_lib.tensor.pow_int",
    },
    {
        "op": pow_float,
        "name": "tt_lib.tensor.pow_float",
    },
    {
        "op": tt_lib.tensor.identity,
        "name": "tt_lib.tensor.identity",
    },
    {
        "op": argmax_1,
        "name": "tt_lib.tensor.argmax_dim_3",
        "num_repeats": 2,
    },
    {
        "op": argmax_2,
        "name": "tt_lib.tensor.argmax_dim_2",
        "num_repeats": 2,
    },
    {
        "op": argmax_3,
        "name": "tt_lib.tensor.argmax_dim_1",
        "num_repeats": 2,
    },
    {
        "op": argmax_all,
        "name": "tt_lib.tensor.argmax_all",
        "num_repeats": 2,
    },
    {
        "op": argmin_1,
        "name": "tt_lib.tensor.argmin_dim_3",
        "num_repeats": 2,
    },
    {
        "op": argmin_2,
        "name": "tt_lib.tensor.argmin_dim_2",
        "num_repeats": 2,
    },
    {
        "op": argmin_3,
        "name": "tt_lib.tensor.argmin_dim_1",
        "num_repeats": 2,
    },
    {
        "op": argmin_all,
        "name": "tt_lib.tensor.argmin_all",
        "num_repeats": 2,
    },
    {
        "op": tt_lib.tensor.fill_zero_bw,
        "name": "tt_lib.tensor.fill_zero_bw",
    },
    {
        "op": tt_lib.tensor.fill_bw,
        "name": "tt_lib.tensor.fill_bw",
    },
    {
        "op": tt_lib.tensor.lt_bw,
        "name": "tt_lib.tensor.lt_bw",
    },
    {
        "op": tt_lib.tensor.gt_bw,
        "name": "tt_lib.tensor.gt_bw",
    },
    {
        "op": tt_lib.tensor.ne_bw,
        "name": "tt_lib.tensor.ne_bw",
    },
    {
        "op": tt_lib.tensor.ge_bw,
        "name": "tt_lib.tensor.ge_bw",
    },
    {
        "op": tt_lib.tensor.le_bw,
        "name": "tt_lib.tensor.le_bw",
    },
    {
        "op": tt_lib.tensor.floor_bw,
        "name": "tt_lib.tensor.floor_bw",
    },
    {
        "op": tt_lib.tensor.round_bw,
        "name": "tt_lib.tensor.round_bw",
    },
    {
        "op": ttnn.softmax_in_place,
        "name": "ttnn.softmax_in_place",
    },
    {
        "op": primary_moreh_softmax_0,
        "name": "tt_lib.operations.primary.moreh_softmax_dim_0",
    },
    {
        "op": primary_moreh_softmax_1,
        "name": "tt_lib.operations.primary.moreh_softmax_dim_1",
    },
    {
        "op": primary_moreh_softmax_2,
        "name": "tt_lib.operations.primary.moreh_softmax_dim_2",
    },
    {
        "op": primary_moreh_softmax_3,
        "name": "tt_lib.operations.primary.moreh_softmax_dim_3",
    },
    {
        "op": primary_moreh_softmin_0,
        "name": "tt_lib.operations.primary.moreh_softmin_dim_0",
    },
    {
        "op": primary_moreh_softmin_1,
        "name": "tt_lib.operations.primary.moreh_softmin_dim_1",
    },
    {
        "op": primary_moreh_softmin_2,
        "name": "tt_lib.operations.primary.moreh_softmin_dim_2",
    },
    {
        "op": primary_moreh_softmin_3,
        "name": "tt_lib.operations.primary_moreh_softmin_dim_3",
    },
    {
        "op": primary_moreh_logsoftmax_0,
        "name": "tt_lib.operations.primary.moreh_logsoftmax_dim_0",
    },
    {
        "op": primary_moreh_logsoftmax_1,
        "name": "tt_lib.operations.primary.moreh_logsoftmax_dim_1",
    },
    {
        "op": primary_moreh_logsoftmax_2,
        "name": "tt_lib.operations.primary.moreh_logsoftmax_dim_2",
    },
    {
        "op": primary_moreh_logsoftmax_3,
        "name": "tt_lib.operations.primary.moreh_logsoftmax_dim_3",
    },
    {
        "op": primary_moreh_norm_0,
        "name": "tt_lib.operations.primary.moreh_norm_dim_0",
    },
    {
        "op": primary_moreh_norm_1,
        "name": "tt_lib.operations.primary.moreh_norm_dim_1",
    },
    {
        "op": primary_moreh_norm_2,
        "name": "tt_lib.operations.primary.moreh_norm_dim_2",
    },
    {
        "op": primary_moreh_norm_3,
        "name": "tt_lib.operations.primary.moreh_norm_dim_3",
    },
    {
        "op": fused_softmax,
        "name": "tt_lib.fused_ops.softmax.softmax",
    },
]

#  Unsupported storage type
# {
#     "op": convert_conv_weight_tensor_to_tiled_layout,
#     "name": "tt_lib.tensor.convert_conv_weight_tensor_to_tiled_layout",
#     "layout": "ROW_MAJOR",
# },


# Very slow - And crashes sometimes
# {
#     "op": argmin_4,
#     "name": "tt_lib.tensor.argmin_dim_0",
# },
# {
#     "op": argmax_4,
#     "name": "tt_lib.tensor.argmax_dim_0",
# },


def layernorm(x, y, z):
    tt_lib.tensor.layernorm(input=x, eps=0.0001, gamma=y, beta=z)


def primary_layernorm(x, y, z):
    tt_lib.operations.primary.layernorm(input=x, eps=0.0001, gamma=y, beta=z)


def norm_shapes_func(input_shape):
    input_shape_12 = [input_shape[0], input_shape[1], 32, input_shape[3]]
    return input_shape, input_shape_12, input_shape_12


def add_layernorm(x, y, z):
    tt_lib.tensor.add_layernorm(a=x, b=x, eps=0.0001, gamma=y, beta=z)


def primary_add_layernorm(x, y, z):
    tt_lib.operations.primary.add_layernorm(a=x, b=x, eps=0.0001, gamma=y, beta=z)


def groupnorm(x, y, z):
    tt_lib.tensor.groupnorm(input=x, group_size=32, eps=0.0001, gamma=y, beta=z)


def primary_moreh_groupnorm(x, y, z):
    tt_lib.operations.primary.moreh_groupnorm(
        input=x, num_groups=4, eps=0.0001, gamma=y, beta=y, are_required_outputs=(True, True, True), mean=z, rstd=z
    )


def primary_moreh_groupnorm_shape_func(input_shape):
    N, C, _, _ = [2, 4, 512, 512]
    num_groups = 4

    gamma_beta_shape = [1, 1, 1, C]
    mean_rstd_shape = [1, 1, N, num_groups]

    return input_shape, gamma_beta_shape, mean_rstd_shape


def rmsnorm(x, y, z):
    tt_lib.tensor.rmsnorm(input=x, eps=0.0001, gamma=y, beta=z)


def addcmul(x, y, z):
    tt_lib.tensor.addcmul(x, y, z, 2)


def addcdiv(x, y, z):
    tt_lib.tensor.addcdiv(x, y, z, 2)


def lamb_optimizer(x, y, z):
    tt_lib.tensor.lamb_optimizer(x, x, y, z, beta1=0.8, beta2=0.99, step_size=1e-3, eps=1e-6, weight_decay=0.02)


def addalpha_bw(x, y, z):
    ttnn.addalpha_bw(x, y, z, alpha=5)


def addcmul_bw(x, y, z):
    tt_lib.tensor.addcmul_bw(x, x, y, z, value=5)


def addcdiv_bw(x, y, z):
    tt_lib.tensor.addcdiv_bw(x, x, y, z, value=5)


def where_bw(x, y, z):
    tt_lib.tensor.where_bw(x, y, z, z)


def bias_gelu_bw_none(x, y, z):
    ttnn.bias_gelu_bw(x, y, z, "none")


def bias_gelu_bw_tanh(x, y, z):
    ttnn.bias_gelu_bw(x, y, z, "tanh")


def lerp_bw_1(x, y, z):
    ttnn.lerp_bw(x, y, z, 0.7)


def lerp_bw_2(x, y, z):
    ttnn.lerp_bw(x, x, y, z)


def concat_bw_0(x, y, z):
    ttnn.concat_bw(x, y, z, 0)


def concat_bw_0_shape_func(input_shape):
    input_shape_0 = [2 * input_shape[0], input_shape[1], input_shape[2], input_shape[3]]
    return input_shape_0, input_shape, input_shape


def concat_bw_1(x, y, z):
    ttnn.concat_bw(x, y, z, 1)


def concat_bw_1_shape_func(input_shape):
    input_shape_0 = [input_shape[0], 2 * input_shape[1], input_shape[2], input_shape[3]]
    return input_shape_0, input_shape, input_shape


def concat_bw_2(x, y, z):
    ttnn.concat_bw(x, y, z, 2)


def concat_bw_2_shape_func(input_shape):
    input_shape_0 = [input_shape[0], input_shape[1], 2 * input_shape[2], input_shape[3]]
    return input_shape_0, input_shape, input_shape


def concat_bw_3(x, y, z):
    ttnn.concat_bw(x, y, z, 3)


def concat_bw_3_shape_func(input_shape):
    input_shape_0 = [input_shape[0], input_shape[1], input_shape[2], 2 * input_shape[3]]
    return input_shape_0, input_shape, input_shape


def subalpha_bw(x, y, z):
    ttnn.subalpha_bw(x, y, z, alpha=3)


def div_bw(x, y, z):
    ttnn.div_bw(x, y, z, mode="None")


def primary_moreh_norm_backward(x, y, z):
    tt_lib.operations.primary.moreh_norm_backward(x, y, z, p=2.0)


from tt_lib.fused_ops.linear import Linear as Fused_Linear


fused_linear_op = None


def fused_linear(x, weight, bias):
    global fused_linear_op

    if fused_linear_op is None:
        shape = x.get_legacy_shape()
        fused_linear_op = Fused_Linear(shape[-2], shape[-1], weight, bias, device=x.device())

    fused_linear_op(x)


def fused_linear_shape_func(input_shape):
    x_shape = [1, 1, input_shape[-2], input_shape[-1]]
    weight_shape = [1, 1, input_shape[-2], input_shape[-1]]
    bias_shape = [1, 1, 32, input_shape[-1]]
    return x_shape, weight_shape, bias_shape


all_ternary_ops = [
    {
        "op": tt_lib.tensor.mac,
        "name": "tt_lib.tensor.mac",
    },
    {
        "op": tt_lib.tensor.where,
        "name": "tt_lib.tensor.where",
    },
    {
        "op": tt_lib.tensor.lerp,
        "name": "tt_lib.tensor.lerp",
    },
    {
        "op": layernorm,
        "name": "tt_lib.tensor.layernorm",
        "shape_func": norm_shapes_func,
    },
    {
        "op": primary_layernorm,
        "name": "tt_lib.operations.primary.layernorm",
        "shape_func": norm_shapes_func,
    },
    {
        "op": groupnorm,
        "name": "tt_lib.tensor.groupnorm",
    },
    {
        "op": rmsnorm,
        "name": "tt_lib.tensor.rmsnorm",
        "shape_func": norm_shapes_func,
    },
    {
        "op": add_layernorm,
        "name": "tt_lib.tensor.add_layernorm",
        "shape_func": norm_shapes_func,
    },
    {
        "op": primary_add_layernorm,
        "name": "tt_lib.operations.primary.add_layernorm",
        "shape_func": norm_shapes_func,
    },
    {
        "op": addcmul,
        "name": "tt_lib.tensor.addcmul",
    },
    {
        "op": addcdiv,
        "name": "tt_lib.tensor.addcdiv",
    },
    {
        "op": lamb_optimizer,
        "name": "tt_lib.tensor.lamb_optimizer",
        "num_repeats": 2,
    },
    {
        "op": addalpha_bw,
        "name": "ttnn.addalpha_bw",
    },
    {
        "op": addcmul_bw,
        "name": "tt_lib.tensor.addcmul_bw",
    },
    {
        "op": addcdiv_bw,
        "name": "tt_lib.tensor.addcdiv_bw",
    },
    {
        "op": ttnn.binary_assign_bw,
        "name": "ttnn.binary_assign_bw",
    },
    {
        "op": ttnn.binary_eq_bw,
        "name": "ttnn.binary_eq_bw",
    },
    {
        "op": ttnn.binary_ge_bw,
        "name": "ttnn.binary_ge_bw",
    },
    {
        "op": ttnn.binary_gt_bw,
        "name": "ttnn.binary_gt_bw",
    },
    {
        "op": ttnn.binary_lt_bw,
        "name": "ttnn.binary_lt_bw",
    },
    {
        "op": ttnn.binary_ne_bw,
        "name": "ttnn.binary_ne_bw",
    },
    {
        "op": div_bw,
        "name": "ttnn.div_bw",
        "num_repeats": 3,
    },
    {
        "op": ttnn.mul_bw,
        "name": "ttnn.mul_bw",
    },
    {
        "op": ttnn.max_bw,
        "name": "ttnn.max_bw",
    },
    {
        "op": ttnn.min_bw,
        "name": "ttnn.min_bw",
    },
    {
        "op": ttnn.add_bw,
        "name": "ttnn.add_bw",
    },
    # {
    #     "op": tt_lib.tensor.embedding_bw,
    #     "name": "tt_lib.tensor.embedding_bw",
    # },
    {
        "op": where_bw,
        "name": "tt_lib.tensor.where_bw",
    },
    {
        "op": ttnn.sub_bw,
        "name": "ttnn.sub_bw",
    },
    {
        "op": ttnn.rsub_bw,
        "name": "ttnn.rsub_bw",
    },
    {
        "op": ttnn.atan2_bw,
        "name": "ttnn.atan2_bw",
    },
    {
        "op": ttnn.hypot_bw,
        "name": "ttnn.hypot_bw",
    },
    {
        "op": ttnn.binary_le_bw,
        "name": "ttnn.binary_le_bw",
    },
    {
        "op": bias_gelu_bw_none,
        "name": "ttnn.bias_gelu_bw_none",
    },
    {
        "op": bias_gelu_bw_tanh,
        "name": "ttnn.bias_gelu_bw_tanh",
    },
    {
        "op": ttnn.squared_difference_bw,
        "name": "ttnn.squared_difference_bw",
    },
    {
        "op": lerp_bw_1,
        "name": "ttnn.lerp_bw_float_weight",
    },
    {
        "op": lerp_bw_2,
        "name": "ttnn.lerp_bw_tensor_weight",
    },
    {
        "op": ttnn.ldexp_bw,
        "name": "ttnn.ldexp_bw",
    },
    {
        "op": ttnn.xlogy_bw,
        "name": "ttnn.xlogy_bw",
    },
    {
        "op": ttnn.logaddexp_bw,
        "name": "ttnn.logaddexp_bw",
    },
    {
        "op": ttnn.logaddexp2_bw,
        "name": "ttnn.logaddexp2_bw",
    },
    {
        "op": concat_bw_0,
        "name": "ttnn.concat_bw_dim_0",
        "shape_func": concat_bw_0_shape_func,
    },
    {
        "op": concat_bw_1,
        "name": "ttnn.concat_bw_dim_1",
        "shape_func": concat_bw_1_shape_func,
    },
    {
        "op": concat_bw_2,
        "name": "ttnn.concat_bw_dim_2",
        "shape_func": concat_bw_2_shape_func,
    },
    {
        "op": concat_bw_3,
        "name": "ttnn.concat_bw_dim_3",
        "shape_func": concat_bw_3_shape_func,
    },
    {
        "op": subalpha_bw,
        "name": "ttnn.subalpha_bw",
    },
    {
        "op": primary_moreh_norm_backward,
        "name": "tt_lib.tensor.moreh_norm_backward",
    },
    {
        "op": fused_linear,
        "name": "tt_lib.fused_ops.linear.Linear",
        "shape_func": fused_linear_shape_func,
    },
]

# Gets stuck
# {
#     "op": primary_moreh_groupnorm,
#     "name": "tt_lib.operations.primary.moreh_groupnorm",
#     "shape_func": primary_moreh_groupnorm_shape_func,
# },
# {
#     "op": primary_moreh_groupnorm_backward,
#     "name": "tt_lib.operations.primary.moreh_groupnorm_backward",
#     "shape_func": primary_moreh_groupnorm_backward_shape_func,
# }


# Seems depricated
# {
#     "op": fused_layernorm,
#     "name": "tt_lib.fused_ops.layernorm.Layernorm",
#     "shape_func": norm_shapes_func,
# },
