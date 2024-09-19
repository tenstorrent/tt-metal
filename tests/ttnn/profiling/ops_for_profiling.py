# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import tt_lib
import ttnn


def subalpha(x, y):
    ttnn.subalpha(x, y, 5)


def addalpha(x, y):
    ttnn.addalpha(x, y, 5)


def isclose(x, y):
    ttnn.isclose(x, y, rtol=0.00001, atol=0.0000001)


def where_binary_1(x, y):
    ttnn.where(x, 5, y)


def where_binary_2(x, y):
    ttnn.where(x, y, 5)


def bcast_add_h(x, y):
    ttnn.bcast(x, y, ttnn.BcastOpMath.ADD, ttnn.BcastOpDim.H)


def bcast_add_w(x, y):
    ttnn.bcast(x, y, ttnn.BcastOpMath.ADD, ttnn.BcastOpDim.W)


def bcast_add_hw(x, y):
    ttnn.bcast(x, y, ttnn.BcastOpMath.ADD, ttnn.BcastOpDim.HW)


def bcast_sub_h(x, y):
    ttnn.bcast(x, y, ttnn.BcastOpMath.SUB, ttnn.BcastOpDim.H)


def bcast_sub_w(x, y):
    ttnn.bcast(x, y, ttnn.BcastOpMath.SUB, ttnn.BcastOpDim.W)


def bcast_sub_hw(x, y):
    ttnn.bcast(x, y, ttnn.BcastOpMath.SUB, ttnn.BcastOpDim.HW)


def bcast_mul_h(x, y):
    ttnn.bcast(x, y, ttnn.BcastOpMath.MUL, ttnn.BcastOpDim.H)


def bcast_mul_w(x, y):
    ttnn.bcast(x, y, ttnn.BcastOpMath.MUL, ttnn.BcastOpDim.W)


def bcast_mul_hw(x, y):
    ttnn.bcast(x, y, ttnn.BcastOpMath.MUL, ttnn.BcastOpDim.HW)


def bcast_h_shape_func(input_shape):
    input_shape_1 = [input_shape[-4], input_shape[-3], 32, input_shape[-1]]
    return input_shape, input_shape_1


def bcast_w_shape_func(input_shape):
    input_shape_1 = [input_shape[-4], input_shape[-3], input_shape[-2], 32]
    return input_shape, input_shape_1


def bcast_hw_shape_func(input_shape):
    input_shape_1 = [input_shape[-4], input_shape[-3], 32, 32]
    return input_shape, input_shape_1


def bcast_hw_shape_func_11(input_shape):
    input_shape_1 = [input_shape[-4], input_shape[-3], 1, 1]
    return input_shape, input_shape_1


def bcast_h_shape_func_1(input_shape):
    input_shape_1 = [input_shape[-4], input_shape[-3], 1, input_shape[-1]]
    return input_shape, input_shape_1


def bcast_w_shape_func_1(input_shape):
    input_shape_1 = [input_shape[-4], input_shape[-3], input_shape[-2], 1]
    return input_shape, input_shape_1


def concat_0(x, y):
    ttnn.concat([x, y], 0)


def concat_1(x, y):
    ttnn.concat([x, y], 1)


def concat_2(x, y):
    ttnn.concat([x, y], 2)


def concat_3(x, y):
    ttnn.concat([x, y], 3)


def lerp_binary(x, y):
    ttnn.lerp(x, y, 0.7)


def embeddings_shape_func(input_shape):
    input_shape_0 = (input_shape[0], 1, 1, input_shape[-1])
    input_shape_1 = (input_shape[0], 1, 1, input_shape[-1])
    return input_shape_0, input_shape_1


def logical_and_(x, y):
    ttnn.logical_and_(x, y)


def logical_or_(x, y):
    ttnn.logical_or_(x, y)


def logical_xor_(x, y):
    ttnn.logical_xor_(x, y)


def unary_add_bw(x, y):
    ttnn.add_bw(x, y, 3)


def rdiv_bw(x, y):
    ttnn.rdiv_bw(x, y, 3)


def unary_pow_bw(x, y):
    ttnn.pow_bw(x, y, 3)


def clamp_bw(x, y):
    ttnn.clamp_bw(x, y, min=0.1, max=0.9)


def clamp_min_bw(x, y):
    ttnn.clamp_bw(x, y, min=0.1)


def clamp_max_bw(x, y):
    ttnn.clamp_bw(x, y, max=0.9)


def gelu_bw_none(x, y):
    ttnn.gelu_bw(x, y, approximate="none")


def gelu_bw_tanh(x, y):
    ttnn.gelu_bw(x, y, approximate="tanh")


def bias_gelu_unary_bw_none(x, y):
    ttnn.bias_gelu_bw(x, y, bias=3.1, approximate="none")


def bias_gelu_unary_bw_tanh(x, y):
    ttnn.bias_gelu_bw(x, y, bias=3.1, approximate="tanh")


def softplus_bw(x, y):
    ttnn.softplus_bw(x, y, beta=2, threshold=10)


def polygamma_bw(x, y):
    ttnn.polygamma_bw(x, y, 3)


def elu_bw(x, y):
    ttnn.elu_bw(x, y, alpha=0.7)


def hardtanh_bw(x, y):
    ttnn.hardtanh_bw(x, y, min=-0.8, max=0.8)


def rpow_bw(x, y):
    ttnn.rpow_bw(x, y, 3.1)


def threshold_bw(x, y):
    ttnn.threshold_bw(x, y, 0.7, 10)


def unary_eq_bw(x, y):
    tt_lib.tensor.unary_eq_bw(x, y, other=0.7)


def logiteps_bw(x, y):
    ttnn.logiteps_bw(x, y, eps=0.0001)


def fmod_bw(x, y):
    ttnn.fmod_bw(x, y, 0.5)


def remainder_bw(x, y):
    ttnn.remainder_bw(x, y, 0.5)


def repeat_bw(x, y):
    ttnn.repeat_bw(x, y, shape=(1, 1, 1, 4))


def repeat_bw_shape_func(input_shape):
    input_shape_1 = [1, 1, input_shape[2], input_shape[3]]
    return input_shape, input_shape_1


def div_no_nan_bw(x, y):
    ttnn.div_no_nan_bw(x, y, 3.0)


def mseloss(x, y):
    ttnn.mse_loss(x, y, reduction=ttnn.LossReductionMode.MEAN)


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


def primary_moreh_mean_0(x):
    ttnn.operations.moreh.mean(x, dim=[0])


def primary_moreh_mean_01(x):
    ttnn.operations.moreh.mean(x, dim=[0, 1])


def primary_moreh_mean_012(x):
    ttnn.operations.moreh.mean(x, dim=[0, 1, 2])


def primary_moreh_mean_0123(x):
    ttnn.operations.moreh.mean(x, dim=[0, 1, 2, 3])


def primary_moreh_mean_013(x):
    ttnn.operations.moreh.mean(x, dim=[0, 1, 3])


def primary_moreh_mean_023(x):
    ttnn.operations.moreh.mean(x, dim=[0, 2, 3])


def primary_moreh_mean_1(x):
    ttnn.operations.moreh.mean(x, dim=[1])


def primary_moreh_mean_12(x):
    ttnn.operations.moreh.mean(x, dim=[1, 2])


def primary_moreh_mean_123(x):
    ttnn.operations.moreh.mean(x, dim=[1, 2, 3])


def primary_moreh_mean_13(x):
    ttnn.operations.moreh.mean(x, dim=[1, 3])


def primary_moreh_mean_2(x):
    ttnn.operations.moreh.mean(x, dim=[2])


def primary_moreh_mean_23(x):
    ttnn.operations.moreh.mean(x, dim=[2, 3])


def primary_moreh_mean_3(x):
    ttnn.operations.moreh.mean(x, dim=[3])


def primary_moreh_mean_backward(x, y):
    ttnn.operations.moreh.mean_backward(x, dim=[0], keepdim=True, input_grad=y)


def celu_bw(x, y):
    ttnn.celu_bw(x, y, alpha=1)


def hardshrink_bw(x, y):
    ttnn.hardshrink_bw(x, y, lambd=0.5)


def leaky_relu_bw(x, y):
    ttnn.leaky_relu_bw(x, y, negative_slope=0.3)


def softshrink_bw(x, y):
    ttnn.softshrink_bw(x, y, lambd=0.5)


def unary_div_bw(x, y):
    ttnn.div_bw(x, y, 3.0, round_mode="None")


all_binary_ops = [
    {
        "op": ttnn.assign,
        "name": "ttnn.assign_binary",
    },
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
        "op": ttnn.mul,
        "name": "ttnn.mul_bcast_h",
        "shape_func": bcast_h_shape_func_1,
    },
    {
        "op": ttnn.mul,
        "name": "ttnn.mul_bcast_w",
        "shape_func": bcast_w_shape_func_1,
    },
    {
        "op": ttnn.mul,
        "name": "ttnn.mul_bcast_hw",
        "shape_func": bcast_hw_shape_func_11,
    },
    {
        "op": ttnn.divide,
        "name": "ttnn.divide",
    },
    {
        "op": ttnn.div,
        "name": "ttnn.div",
    },
    {
        "op": ttnn.div_no_nan,
        "name": "ttnn.div_no_nan",
    },
    {
        "op": ttnn.hypot,
        "name": "ttnn.hypot",
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
        "op": ttnn.atan2,
        "name": "ttnn.atan2",
    },
    {
        "op": ttnn.logical_xor,
        "name": "ttnn.logical_xor",
    },
    {
        "op": subalpha,
        "name": "ttnn.subalpha",
    },
    {
        "op": addalpha,
        "name": "ttnn.addalpha",
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
        "op": isclose,
        "name": "ttnn.isclose",
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
        "name": "ttnn.where_binary_x_const_y",
    },
    {
        "op": where_binary_2,
        "name": "ttnn.where_binary_x_y_const",
    },
    {
        "op": ttnn.matmul,
        "name": "ttnn.matmul",
    },
    {
        "op": ttnn.copy,
        "name": "ttnn.copy",
    },
    {
        "op": bcast_add_h,
        "name": "ttnn.bcast_add_h",
        "shape_func": bcast_h_shape_func,
    },
    {
        "op": bcast_add_w,
        "name": "ttnn.bcast_add_w",
        "shape_func": bcast_w_shape_func,
    },
    {
        "op": bcast_add_hw,
        "name": "ttnn.bcast_add_hw",
        "shape_func": bcast_hw_shape_func,
    },
    {
        "op": bcast_sub_h,
        "name": "ttnn.bcast_sub_h",
        "shape_func": bcast_h_shape_func,
    },
    {
        "op": bcast_sub_w,
        "name": "ttnn.bcast_sub_w",
        "shape_func": bcast_w_shape_func,
    },
    {
        "op": bcast_sub_hw,
        "name": "ttnn.bcast_sub_hw",
        "shape_func": bcast_hw_shape_func,
    },
    {
        "op": bcast_mul_h,
        "name": "ttnn.bcast_mul_h",
        "shape_func": bcast_h_shape_func,
    },
    {
        "op": bcast_mul_w,
        "name": "ttnn.bcast_mul_w",
        "shape_func": bcast_w_shape_func,
    },
    {
        "op": bcast_mul_hw,
        "name": "ttnn.bcast_mul_hw",
        "shape_func": bcast_hw_shape_func,
    },
    # {
    #     "op": ttnn.add,
    #     "name": "ttnn.complex_add",
    #     "is_complex": [True, True],
    #     "need_out_mem_cfg": True,
    # },
    # {
    #     "op": ttnn.sub,
    #     "name": "ttnn.complex_sub",
    #     "is_complex": [True, True],
    #     "need_out_mem_cfg": True,
    # },
    # {
    #     "op": ttnn.multiply,
    #     "name": "ttnn.complex_mul",
    #     "is_complex": [True, True],
    #     "need_out_mem_cfg": True,
    # },
    # {
    #     "op": ttnn.divide,
    #     "name": "ttnn.complex_div",
    #     "is_complex": [True, True],
    #     "need_out_mem_cfg": True,
    # },
    {
        "op": concat_0,
        "name": "ttnn.concat_dim_0",
    },
    {
        "op": concat_1,
        "name": "ttnn.concat_dim_1",
    },
    {
        "op": concat_2,
        "name": "ttnn.concat_dim_2",
    },
    {
        "op": concat_3,
        "name": "ttnn.concat_dim_3",
    },
    {
        "op": lerp_binary,
        "name": "ttnn.lerp_binary",
    },
    {
        "op": logical_and_,
        "name": "ttnn.logical_and_",
    },
    {
        "op": logical_or_,
        "name": "ttnn.logical_or_",
    },
    {
        "op": logical_xor_,
        "name": "ttnn.logical_xor_",
    },
    {
        "op": ttnn.xlogy,
        "name": "ttnn.xlogy",
    },
    {
        "op": ttnn.embedding,
        "name": "ttnn.embedding",
        "layout": "ROW_MAJOR",
        "shape_func": embeddings_shape_func,
    },
    {
        "op": ttnn.nextafter,
        "name": "ttnn.nextafter",
    },
    {
        "op": ttnn.conj_bw,
        "name": "ttnn.conj_bw",
        "is_complex": [True, True],
        "need_out_mem_cfg": True,
    },
    {
        "op": unary_add_bw,
        "name": "ttnn.unary_add_bw",
    },
    {
        "op": ttnn.assign_bw,
        "name": "ttnn.assign_bw",
    },
    {
        "op": unary_div_bw,
        "name": "ttnn.unary_div_bw",
    },
    {
        "op": rdiv_bw,
        "name": "ttnn.rdiv_bw",
    },
    {
        "op": ttnn.sqrt_bw,
        "name": "ttnn.sqrt_bw",
    },
    {
        "op": ttnn.tan_bw,
        "name": "ttnn.tan_bw",
    },
    {
        "op": ttnn.exp_bw,
        "name": "ttnn.exp_bw",
    },
    {
        "op": ttnn.exp2_bw,
        "name": "ttnn.exp2_bw",
    },
    {
        "op": ttnn.expm1_bw,
        "name": "ttnn.expm1_bw",
    },
    {
        "op": unary_pow_bw,
        "name": "ttnn.unary_pow_bw",
    },
    {
        "op": ttnn.tanh_bw,
        "name": "ttnn.tanh_bw",
    },
    {
        "op": ttnn.log_bw,
        "name": "ttnn.log_bw",
    },
    {
        "op": ttnn.abs_bw,
        "name": "ttnn.abs_bw",
    },
    {
        "op": ttnn.rsqrt_bw,
        "name": "ttnn.rsqrt_bw",
    },
    {
        "op": ttnn.neg_bw,
        "name": "ttnn.neg_bw",
    },
    {
        "op": ttnn.relu_bw,
        "name": "ttnn.relu_bw",
    },
    {
        "op": clamp_bw,
        "name": "ttnn.clamp_bw",
    },
    {
        "op": clamp_min_bw,
        "name": "ttnn.clamp_min_bw",
    },
    {
        "op": clamp_max_bw,
        "name": "ttnn.clamp_max_bw",
    },
    {
        "op": gelu_bw_none,
        "name": "ttnn.gelu_bw_none",
    },
    {
        "op": gelu_bw_tanh,
        "name": "ttnn.gelu_bw_tanh",
    },
    {
        "op": bias_gelu_unary_bw_none,
        "name": "ttnn.bias_gelu_unary_bw_none",
    },
    {
        "op": bias_gelu_unary_bw_tanh,
        "name": "ttnn.bias_gelu_unary_bw_tanh",
    },
    {
        "op": ttnn.hardsigmoid_bw,
        "name": "ttnn.hardsigmoid_bw",
    },
    {
        "op": ttnn.i0_bw,
        "name": "ttnn.i0_bw",
    },
    {
        "op": hardshrink_bw,
        "name": "ttnn.hardshrink_bw",
    },
    {
        "op": softshrink_bw,
        "name": "ttnn.softshrink_bw",
    },
    {
        "op": ttnn.hardswish_bw,
        "name": "ttnn.hardswish_bw",
    },
    {
        "op": softplus_bw,
        "name": "ttnn.softplus_bw",
    },
    {
        "op": polygamma_bw,
        "name": "ttnn.polygamma_bw",
        "num_repeats": 3,
    },
    {
        "op": ttnn.atan_bw,
        "name": "ttnn.atan_bw",
    },
    {
        "op": ttnn.atanh_bw,
        "name": "ttnn.atanh_bw",
    },
    {
        "op": ttnn.asin_bw,
        "name": "ttnn.asin_bw",
    },
    {
        "op": ttnn.asinh_bw,
        "name": "ttnn.asinh_bw",
    },
    {
        "op": ttnn.cosh_bw,
        "name": "ttnn.cosh_bw",
        "num_repeats": 3,
    },
    {
        "op": ttnn.cos_bw,
        "name": "ttnn.cos_bw",
    },
    {
        "op": ttnn.acosh_bw,
        "name": "ttnn.acosh_bw",
    },
    {
        "op": ttnn.acos_bw,
        "name": "ttnn.acos_bw",
    },
    {
        "op": ttnn.erfinv_bw,
        "name": "ttnn.erfinv_bw",
    },
    {
        "op": leaky_relu_bw,
        "name": "ttnn.leaky_relu_bw",
    },
    {
        "op": elu_bw,
        "name": "ttnn.elu_bw",
    },
    {
        "op": hardtanh_bw,
        "name": "ttnn.hardtanh_bw",
    },
    {
        "op": ttnn.sin_bw,
        "name": "ttnn.sin_bw",
    },
    {
        "op": ttnn.sinh_bw,
        "name": "ttnn.sinh_bw",
    },
    {
        "op": celu_bw,
        "name": "ttnn.celu_bw",
    },
    {
        "op": ttnn.log10_bw,
        "name": "ttnn.log10_bw",
    },
    {
        "op": ttnn.log1p_bw,
        "name": "ttnn.log1p_bw",
    },
    {
        "op": ttnn.erf_bw,
        "name": "ttnn.erf_bw",
    },
    {
        "op": ttnn.erfc_bw,
        "name": "ttnn.erfc_bw",
    },
    {
        "op": ttnn.digamma_bw,
        "name": "ttnn.digamma_bw",
        "num_repeats": 2,
    },
    {
        "op": ttnn.deg2rad_bw,
        "name": "ttnn.deg2rad_bw",
    },
    {
        "op": ttnn.rad2deg_bw,
        "name": "ttnn.rad2deg_bw",
    },
    {
        "op": ttnn.reciprocal_bw,
        "name": "ttnn.reciprocal_bw",
    },
    {
        "op": ttnn.relu6_bw,
        "name": "ttnn.relu6_bw",
    },
    {
        "op": rpow_bw,
        "name": "ttnn.rpow_bw",
    },
    {
        "op": ttnn.silu_bw,
        "name": "ttnn.silu_bw",
    },
    {
        "op": ttnn.selu_bw,
        "name": "ttnn.selu_bw",
    },
    {
        "op": ttnn.square_bw,
        "name": "ttnn.square_bw",
    },
    {
        "op": ttnn.lgamma_bw,
        "name": "ttnn.lgamma_bw",
    },
    {
        "op": ttnn.trunc_bw,
        "name": "ttnn.trunc_bw",
    },
    {
        "op": ttnn.frac_bw,
        "name": "ttnn.frac_bw",
    },
    {
        "op": ttnn.log_sigmoid_bw,
        "name": "ttnn.log_sigmoid_bw",
    },
    {
        "op": ttnn.tanhshrink_bw,
        "name": "ttnn.tanhshrink_bw",
    },
    {
        "op": threshold_bw,
        "name": "ttnn.threshold_bw",
    },
    {
        "op": unary_eq_bw,
        "name": "tt_lib.tensor.unary_eq_bw",
    },
    {
        "op": ttnn.logit_bw,
        "name": "ttnn.logit_bw",
    },
    {
        "op": logiteps_bw,
        "name": "ttnn.logiteps_bw",
    },
    {
        "op": ttnn.softsign_bw,
        "name": "ttnn.softsign_bw",
    },
    {
        "op": ttnn.sign_bw,
        "name": "ttnn.sign_bw",
    },
    {
        "op": ttnn.ceil_bw,
        "name": "ttnn.ceil_bw",
    },
    {
        "op": ttnn.log2_bw,
        "name": "ttnn.log2_bw",
    },
    {
        "op": fmod_bw,
        "name": "ttnn.fmod_bw",
    },
    {
        "op": remainder_bw,
        "name": "ttnn.remainder_bw",
    },
    {
        "op": ttnn.imag_bw,
        "name": "ttnn.imag_bw",
        "is_complex": [False, True],
        "need_out_mem_cfg": True,
    },
    {
        "op": ttnn.real_bw,
        "name": "ttnn.real_bw",
        "is_complex": [False, True],
        "need_out_mem_cfg": True,
    },
    {
        "op": ttnn.multigammaln_bw,
        "name": "ttnn.multigammaln_bw",
    },
    {
        "op": repeat_bw,
        "name": "ttnn.repeat_bw",
        "shape_func": repeat_bw_shape_func,
    },
    {
        "op": div_no_nan_bw,
        "name": "ttnn.div_no_nan_bw",
    },
    {
        "op": mseloss,
        "name": "ttnn.mse_loss",
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
        "op": ttnn.angle_bw,
        "name": "ttnn.angle_bw",
        "is_complex": [False, True],
        "need_out_mem_cfg": True,
    },
    {
        "op": ttnn.fill_bw,
        "name": "ttnn.fill_bw",
    },
    {
        "op": ttnn.fill_zero_bw,
        "name": "ttnn.fill_zero_bw",
    },
    {
        "op": ttnn.floor_bw,
        "name": "ttnn.floor_bw",
    },
    {
        "op": ttnn.round_bw,
        "name": "ttnn.round_bw",
    },
    {
        "op": primary_moreh_mean_backward,
        "name": "ttnn.operations.moreh.mean_backward",
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
    ttnn.add(x, 5.0)


def sub_unary(x):
    ttnn.sub(x, 5.0)


def mul_unary(x):
    ttnn.mul(x, 5.0)


def relu_min(x):
    ttnn.relu_min(x, 0.1)


def relu_max(x):
    ttnn.relu_max(x, 0.1)


def clip(x):
    ttnn.clip(x, 0.1, 0.9)


def polyval(x):
    ttnn.polyval(x, [1, 2, 3])


def leaky_relu(x):
    ttnn.leaky_relu(x, 68)


def softshrink(x):
    ttnn.softshrink(x, lambd=70)


def hardshrink(x):
    ttnn.hardshrink(x, lambd=1)


def elu(x):
    ttnn.elu(x, 2)


def heaviside(x):
    ttnn.heaviside(x, 0.5)


def logit(x):
    ttnn.logit(x, eps=0.0001)


def polygamma(x):
    ttnn.polygamma(x, 2)


def where_unary(x):
    ttnn.where(x, 2, 3)


def threshold(x):
    ttnn.threshold(x, 0.5, 3)


def reshape(x):
    shape = x.get_legacy_shape()
    ttnn.reshape(x, [shape[-4], shape[-3], shape[-1], shape[-2]])


def transpose_01(x):
    ttnn.transpose(x, 0, 1)


def transpose_02(x):
    ttnn.transpose(x, 0, 2)


def transpose_03(x):
    ttnn.transpose(x, 0, 3)


def transpose_12(x):
    ttnn.transpose(x, 1, 2)


def transpose_13(x):
    ttnn.transpose(x, 1, 3)


def transpose_23(x):
    ttnn.transpose(x, 2, 3)


def permute(x):
    ttnn.permute(x, [1, 0, 3, 2])


def tilize(x):
    ttnn.tilize(x)


def tilize_with_val_padding(x):
    shape = x.get_legacy_shape()

    output_tensor_shape = [shape[-4], shape[-3], shape[-2] + 32, shape[-1] + 32]

    ttnn.tilize_with_val_padding(x, output_tensor_shape, 1.0)


def untilize_with_unpadding(x):
    shape = x.get_legacy_shape()

    unpadded_shape_end = [
        shape[0] - 1,
        shape[1] - 1,
        shape[2] - 33,
        shape[3] - 33,
    ]

    ttnn.untilize_with_unpadding(x, output_tensor_end=unpadded_shape_end)


def pad(x):
    shape = x.get_legacy_shape()

    padding = [
        (0, 0),
        (0, 0),
        (0, 32),
        (0, 32),
    ]

    ttnn.pad(x, padding, 1)


def ttnn_slice(x):
    shape = x.get_legacy_shape()

    output_tensor_end = (
        shape[0] - 1,
        shape[1] - 1,
        shape[2] - 33,
        shape[3] - 33,
    )

    ttnn.slice(x, (0, 0, 0, 0), output_tensor_end)


def typecast(x):
    ttnn.typecast(x, ttnn.bfloat8_b)


def arange(x):
    ttnn.arange(0, 100, 2, device=x.device())


def full(x):
    ttnn.full(shape=x.get_legacy_shape(), fill_value=2, dtype=x.get_dtype(), layout=x.get_layout(), device=x.device())


def full_like(x):
    ttnn.full_like(x, 2.0)


def ones(x):
    ttnn.ones(shape=x.get_legacy_shape(), dtype=x.get_dtype(), layout=x.get_layout(), device=x.device())


def zeros(x):
    ttnn.zeros(shape=x.get_legacy_shape(), dtype=x.get_dtype(), layout=x.get_layout(), device=x.device())


def empty(x):
    ttnn.empty(shape=x.get_legacy_shape(), dtype=x.get_dtype(), layout=x.get_layout(), device=x.device())


def sum_dim_0(x):
    ttnn.sum(x, dim=0)


def sum_dim_1(x):
    ttnn.sum(x, dim=1)


def sum_dim_2(x):
    ttnn.sum(x, dim=2)


def sum_dim_3(x):
    ttnn.sum(x, dim=3)


def sum_dim_23(x):
    ttnn.sum(x, dim=(2, 3))


def min_dim_2(x):
    ttnn.min(x, dim=2)


def min_dim_3(x):
    ttnn.min(x, dim=3)


def min_dim_23(x):
    ttnn.min(x, dim=(2, 3))


def max_dim_2(x):
    ttnn.max(x, dim=2)


def max_dim_3(x):
    ttnn.max(x, dim=3)


def max_dim_23(x):
    ttnn.max(x, dim=(2, 3))


def rpow(x):
    ttnn.rpow(x, 3)


def rsub(x):
    ttnn.rsub(x, 3)


def rdiv(x):
    ttnn.rdiv(x, 3)


def erf_slow(x):
    ttnn.erf(x, fast_and_approximate_mode=False)


def erfc_slow(x):
    ttnn.erfc(x, fast_and_approximate_mode=False)


def rsqrt_slow(x):
    ttnn.rsqrt(x, fast_and_approximate_mode=False)


def fill_rm(x):
    shape = x.get_legacy_shape()

    ttnn.fill_rm(
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

    ttnn.fill_ones_rm(N=shape[0], C=shape[1], H=shape[2], W=shape[3], hOnes=shape[2] - 32, wOnes=shape[3] - 32, any=x)


def group_norm_no_weights(x):
    ttnn.group_norm(x, num_groups=32, epsilon=0.00001, weight=None, bias=None)


def convert_conv_weight_tensor_to_tiled_layout(x):
    tt_lib.tensor.convert_conv_weight_tensor_to_tiled_layout(x, in1_block_h=32, in1_block_w=32)


def logical_not_(x):
    ttnn.logical_not_(x)


def glu_1(x):
    ttnn.glu(x, -1)


def geglu_1(x):
    ttnn.geglu(x, -1)


def reglu_1(x):
    ttnn.reglu(x, -1)


def swiglu_1(x):
    ttnn.swiglu(x, -1)


def glu_2(x):
    ttnn.glu(x, -2)


def geglu_2(x):
    ttnn.geglu(x, -2)


def reglu_2(x):
    ttnn.reglu(x, -2)


def swiglu_2(x):
    ttnn.swiglu(x, -2)


def repeat(x):
    ttnn.repeat(x, ttnn.Shape((1, 1, 1, 4)))


def repeat_interleave_0(x):
    ttnn.repeat_interleave(x, 4, 0)


def repeat_interleave_1(x):
    ttnn.repeat_interleave(x, 4, 1)


def repeat_interleave_2(x):
    ttnn.repeat_interleave(x, 4, 2)


def pow_int(x):
    ttnn.pow(x, 3)


def pow_float(x):
    ttnn.pow(x, 3.3)


def argmax_dim_3(x):
    ttnn.argmax(x, dim=3)


def argmax_dim_2(x):
    ttnn.argmax(x, dim=2)


def argmax_dim_None(x):
    ttnn.argmax(x, dim=None)


def argmax_shape_func(input_shape):
    return [1, 1, 128, 128]


def argmin_dim_3(x):
    ttnn.argmin(x, dim=3)


def argmin_dim_2(x):
    ttnn.argmin(x, dim=2)


def argmin_dim_1(x):
    ttnn.argmin(x, dim=1)


def argmin_dim_0(x):
    ttnn.argmin(x, dim=0)


def argmin_all(x):
    ttnn.argmin(x, dim=None)


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


def split_dim_3(x):
    ttnn.split(x, 2, 3)


def split_dim_2(x):
    ttnn.split(x, 2, 2)


def assign_unary(x):
    ttnn.assign(x, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=x.get_dtype())


from tt_lib.fused_ops.softmax import softmax as fused_softmax


all_unary_ops = [
    {
        "op": assign_unary,
        "name": "ttnn.assign_unary",
    },
    {
        "op": add_unary,
        "name": "ttnn.add_unary",
    },
    {
        "op": sub_unary,
        "name": "ttnn.sub_unary",
    },
    {
        "op": mul_unary,
        "name": "ttnn.mul_unary",
    },
    {
        "op": ttnn.gelu,
        "name": "ttnn.gelu",
    },
    {
        "op": ttnn.relu,
        "name": "ttnn.relu",
    },
    {
        "op": ttnn.relu6,
        "name": "ttnn.relu6",
    },
    {
        "op": relu_min,
        "name": "ttnn.relu_min",
    },
    {
        "op": relu_max,
        "name": "ttnn.relu_max",
    },
    {
        "op": ttnn.exp,
        "name": "ttnn.exp",
    },
    {
        "op": ttnn.reciprocal,
        "name": "ttnn.reciprocal",
    },
    {
        "op": ttnn.sqrt,
        "name": "ttnn.sqrt",
    },
    {
        "op": ttnn.log,
        "name": "ttnn.log",
    },
    {
        "op": ttnn.log2,
        "name": "ttnn.log2",
    },
    {
        "op": ttnn.log10,
        "name": "ttnn.log10",
    },
    {
        "op": ttnn.log1p,
        "name": "ttnn.log1p",
    },
    {
        "op": ttnn.tanh,
        "name": "ttnn.tanh",
    },
    {
        "op": clip,
        "name": "ttnn.clip",
    },
    {
        "op": ttnn.hardtanh,
        "name": "ttnn.hardtanh",
    },
    {
        "op": ttnn.deg2rad,
        "name": "ttnn.deg2rad",
    },
    {
        "op": ttnn.rad2deg,
        "name": "ttnn.rad2deg",
    },
    {
        "op": ttnn.cbrt,
        "name": "ttnn.cbrt",
    },
    {
        "op": ttnn.softplus,
        "name": "ttnn.softplus",
    },
    {
        "op": ttnn.mish,
        "name": "ttnn.mish",
    },
    {
        "op": polyval,
        "name": "ttnn.polyval",
    },
    {
        "op": ttnn.sign,
        "name": "ttnn.sign",
    },
    {
        "op": ttnn.abs,
        "name": "ttnn.abs",
    },
    {
        "op": ttnn.silu,
        "name": "ttnn.silu",
    },
    {
        "op": ttnn.square,
        "name": "ttnn.square",
    },
    {
        "op": ttnn.neg,
        "name": "ttnn.neg",
    },
    {
        "op": ttnn.sigmoid,
        "name": "ttnn.sigmoid",
    },
    {
        "op": ttnn.sigmoid_accurate,
        "name": "ttnn.sigmoid_accurate",
    },
    {
        "op": ttnn.hardsigmoid,
        "name": "ttnn.hardsigmoid",
    },
    {
        "op": ttnn.swish,
        "name": "ttnn.swish",
    },
    {
        "op": ttnn.hardswish,
        "name": "ttnn.hardswish",
    },
    {
        "op": leaky_relu,
        "name": "ttnn.leaky_relu",
    },
    {
        "op": ttnn.softsign,
        "name": "ttnn.softsign",
    },
    {
        "op": softshrink,
        "name": "ttnn.softshrink",
    },
    {
        "op": hardshrink,
        "name": "ttnn.hardshrink",
    },
    {
        "op": ttnn.cos,
        "name": "ttnn.cos",
    },
    {
        "op": ttnn.sin,
        "name": "ttnn.sin",
    },
    {
        "op": ttnn.cosh,
        "name": "ttnn.cosh",
    },
    {
        "op": ttnn.sinh,
        "name": "ttnn.sinh",
    },
    {
        "op": ttnn.acos,
        "name": "ttnn.acos",
    },
    {
        "op": ttnn.asin,
        "name": "ttnn.asin",
    },
    {
        "op": elu,
        "name": "ttnn.elu",
    },
    {
        "op": ttnn.exp2,
        "name": "ttnn.exp2",
    },
    {
        "op": ttnn.tanhshrink,
        "name": "ttnn.tanhshrink",
    },
    {
        "op": heaviside,
        "name": "ttnn.heaviside",
    },
    {
        "op": ttnn.atan,
        "name": "ttnn.atan",
    },
    {
        "op": ttnn.atanh,
        "name": "ttnn.atanh",
    },
    {
        "op": ttnn.logical_not,
        "name": "ttnn.logical_not",
    },
    {
        "op": ttnn.isfinite,
        "name": "ttnn.isfinite",
    },
    {
        "op": ttnn.isinf,
        "name": "ttnn.isinf",
    },
    {
        "op": ttnn.isposinf,
        "name": "ttnn.isposinf",
    },
    {
        "op": ttnn.isneginf,
        "name": "ttnn.isneginf",
    },
    {
        "op": ttnn.isnan,
        "name": "ttnn.isnan",
    },
    {
        "op": logit,
        "name": "ttnn.logit",
    },
    {
        "op": ttnn.lgamma,
        "name": "ttnn.lgamma",
        "num_repeats": 3,
    },
    {
        "op": ttnn.erfinv,
        "name": "ttnn.erfinv",
    },
    {
        "op": ttnn.multigammaln,
        "name": "ttnn.multigammaln",
    },
    {
        "op": ttnn.i0,
        "name": "ttnn.i0",
    },
    {
        "op": ttnn.digamma,
        "name": "ttnn.digamma",
    },
    {
        "op": ttnn.tan,
        "name": "ttnn.tan",
    },
    {
        "op": polygamma,
        "name": "ttnn.polygamma",
    },
    {
        "op": ttnn.gtz,
        "name": "ttnn.gtz",
    },
    {
        "op": ttnn.gez,
        "name": "ttnn.gez",
    },
    {
        "op": ttnn.ltz,
        "name": "ttnn.ltz",
    },
    {
        "op": ttnn.lez,
        "name": "ttnn.lez",
    },
    {
        "op": ttnn.eqz,
        "name": "ttnn.eqz",
    },
    {
        "op": ttnn.nez,
        "name": "ttnn.nez",
    },
    {
        "op": where_unary,
        "name": "ttnn.where_x_const_const",
    },
    {
        "op": threshold,
        "name": "ttnn.threshold",
    },
    {
        "op": reshape,
        "name": "ttnn.reshape",
    },
    {
        "op": transpose_01,
        "name": "ttnn.transpose_01",
    },
    {
        "op": transpose_02,
        "name": "ttnn.transpose_02",
        "num_repeats": 3,
    },
    # {
    #     "op": transpose_03,
    #     "name": "ttnn.transpose_03",
    # },
    {
        "op": transpose_12,
        "name": "ttnn.transpose_12",
        "num_repeats": 3,
    },
    {
        "op": transpose_13,
        "name": "ttnn.transpose_13",
        "num_repeats": 3,
    },
    {
        "op": transpose_23,
        "name": "ttnn.transpose_23",
    },
    {
        "op": permute,
        "name": "ttnn.permute",
    },
    {
        "op": tilize,
        "name": "ttnn.tilize",
        "layout": "ROW_MAJOR",
    },
    {
        "op": ttnn.untilize,
        "name": "ttnn.untilize",
    },
    {
        "op": tilize_with_val_padding,
        "name": "ttnn.tilize_with_val_padding",
        "layout": "ROW_MAJOR",
    },
    {
        "op": untilize_with_unpadding,
        "name": "ttnn.untilize_with_unpadding",
    },
    {
        "op": ttnn.tilize_with_zero_padding,
        "name": "ttnn.tilize_with_zero_padding",
        "layout": "ROW_MAJOR",
    },
    {
        "op": pad,
        "name": "ttnn.pad",
    },
    {
        "op": ttnn_slice,
        "name": "ttnn.slice",
    },
    {
        "op": ttnn.clone,
        "name": "ttnn.clone",
    },
    {
        "op": typecast,
        "name": "ttnn.typecast",
    },
    {
        "op": arange,
        "name": "ttnn.arange",
    },
    {
        "op": full,
        "name": "ttnn.full",
    },
    {
        "op": ones,
        "name": "ttnn.ones",
    },
    {
        "op": ttnn.ones_like,
        "name": "ttnn.ones_like",
    },
    {
        "op": ttnn.empty_like,
        "name": "ttnn.empty_like",
    },
    {
        "op": zeros,
        "name": "ttnn.zeros",
    },
    {
        "op": ttnn.zeros_like,
        "name": "ttnn.zeros_like",
    },
    {
        "op": full_like,
        "name": "ttnn.full_like",
    },
    {
        "op": split_dim_3,
        "name": "ttnn.split_dim_3",
    },
    {
        "op": split_dim_2,
        "name": "ttnn.split_dim_2",
    },
    {
        "op": empty,
        "name": "ttnn.empty",
    },
    {
        "op": ttnn.tril,
        "name": "ttnn.tril",
        "num_repeats": 3,
    },
    {
        "op": ttnn.triu,
        "name": "ttnn.triu",
        "num_repeats": 3,
    },
    {
        "op": sum_dim_0,
        "name": "ttnn.sum_dim_0",
        "num_repeats": 2,
    },
    {
        "op": sum_dim_1,
        "name": "ttnn.sum_dim_1",
    },
    {
        "op": sum_dim_2,
        "name": "ttnn.sum_dim_2",
    },
    {
        "op": sum_dim_3,
        "name": "ttnn.sum_dim_3",
    },
    {
        "op": sum_dim_23,
        "name": "ttnn.sum_dim_23",
    },
    {
        "op": min_dim_2,
        "name": "ttnn.min_dim_2",
    },
    {
        "op": min_dim_3,
        "name": "ttnn.min_dim_3",
    },
    {
        "op": min_dim_23,
        "name": "ttnn.min_dim_23",
    },
    {
        "op": max_dim_2,
        "name": "ttnn.max_dim_2",
    },
    {
        "op": max_dim_3,
        "name": "ttnn.max_dim_3",
    },
    {
        "op": max_dim_23,
        "name": "ttnn.max_dim_23",
    },
    {
        "op": ttnn.min,
        "name": "ttnn.min",
    },
    {
        "op": ttnn.max,
        "name": "ttnn.max",
    },
    {
        "op": ttnn.sum,
        "name": "ttnn.sum",
    },
    {
        "op": ttnn.mean,
        "name": "ttnn.mean",
    },
    {
        "op": rpow,
        "name": "ttnn.rpow",
    },
    {
        "op": rsub,
        "name": "ttnn.rsub",
    },
    {
        "op": rdiv,
        "name": "ttnn.rdiv",
    },
    {
        "op": ttnn.real,
        "name": "ttnn.real",
        "is_complex": [True],
        "need_out_mem_cfg": True,
    },
    {
        "op": ttnn.imag,
        "name": "ttnn.imag",
        "is_complex": [True],
        "need_out_mem_cfg": True,
    },
    {
        "op": ttnn.abs,
        "name": "ttnn.complex_abs",
        "is_complex": [True],
        "need_out_mem_cfg": True,
    },
    {
        "op": ttnn.conj,
        "name": "ttnn.conj",
        "is_complex": [True],
        "need_out_mem_cfg": True,
    },
    {
        "op": ttnn.reciprocal,
        "name": "ttnn.complex_recip",
        "is_complex": [True],
        "need_out_mem_cfg": True,
    },
    {
        "op": ttnn.polar,
        "name": "ttnn.polar",
        "is_complex": [True],
        "need_out_mem_cfg": True,
    },
    {
        "op": ttnn.log_sigmoid,
        "name": "ttnn.log_sigmoid",
    },
    {
        "op": ttnn.expm1,
        "name": "ttnn.expm1",
    },
    {
        "op": ttnn.asinh,
        "name": "ttnn.asinh",
    },
    {
        "op": ttnn.acosh,
        "name": "ttnn.acosh",
    },
    {
        "op": ttnn.erf,
        "name": "ttnn.erf_fast_and_approx_True",
    },
    {
        "op": erf_slow,
        "name": "ttnn.erf_fast_and_approx_False",
    },
    {
        "op": ttnn.erfc,
        "name": "ttnn.erfc_fast_and_approx_True",
    },
    {
        "op": erfc_slow,
        "name": "ttnn.erfc_fast_and_approx_False",
    },
    {
        "op": ttnn.rsqrt,
        "name": "ttnn.rsqrt_fast_and_approx_True",
    },
    {
        "op": rsqrt_slow,
        "name": "ttnn.rsqrt_fast_and_approx_False",
    },
    {
        "op": ttnn.signbit,
        "name": "ttnn.signbit",
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
        "op": ttnn.mean,
        "name": "tt_lib.tensor.mean_hw",
    },
    {
        "op": ttnn.var_hw,
        "name": "ttnn.var_hw",
    },
    {
        "op": logical_not_,
        "name": "ttnn.logical_not_",
    },
    {
        "op": ttnn.std_hw,
        "name": "ttnn.std_hw",
    },
    {
        "op": ttnn.normalize_hw,
        "name": "ttnn.normalize_hw",
    },
    {
        "op": ttnn.normalize_global,
        "name": "ttnn.normalize_global",
    },
    {
        "op": glu_1,
        "name": "ttnn.glu_dim_3",
    },
    {
        "op": geglu_1,
        "name": "ttnn.geglu_dim_3",
    },
    {
        "op": reglu_1,
        "name": "ttnn.reglu_dim_3",
    },
    {
        "op": swiglu_1,
        "name": "ttnn.swiglu_dim_3",
    },
    {
        "op": glu_2,
        "name": "ttnn.glu_dim_2",
    },
    {
        "op": geglu_2,
        "name": "ttnn.geglu_dim_2",
    },
    {
        "op": reglu_2,
        "name": "ttnn.reglu_dim_2",
    },
    {
        "op": swiglu_2,
        "name": "ttnn.swiglu_dim_2",
    },
    {
        "op": repeat,
        "name": "ttnn.repeat",
    },
    {
        "op": repeat_interleave_0,
        "name": "ttnn.repeat_interleave_dim_0",
    },
    {
        "op": repeat_interleave_1,
        "name": "ttnn.repeat_interleave_dim_1",
        "num_repeats": 2,
    },
    {
        "op": repeat_interleave_2,
        "name": "ttnn.repeat_interleave_dim_2",
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
        "op": ttnn.identity,
        "name": "ttnn.identity",
    },
    {
        "op": argmax_dim_3,
        "name": "ttnn.argmax_dim_3",
        "shape_func": argmax_shape_func,
        "layout": "ROW_MAJOR",
        "num_repeats": 2,
    },
    {
        "op": argmax_dim_None,
        "name": "ttnn.argmax_dim_None",
        "shape_func": argmax_shape_func,
        "layout": "ROW_MAJOR",
        "num_repeats": 2,
    },
    # {
    #     "op": argmin_dim_3,
    #     "name": "ttnn.argmin_dim_3",
    #     "shape_func": argmax_shape_func,
    #     "layout": "ROW_MAJOR",
    #     "num_repeats": 2,
    # },
    # {
    #     "op": argmin_dim_2,
    #     "name": "ttnn.argmin_dim_2",
    #     "shape_func": argmax_shape_func,
    #     "layout": "ROW_MAJOR",
    #     "num_repeats": 2,
    # },
    # {
    #     "op": argmin_all,
    #     "name": "ttnn.argmin_all",
    #     "shape_func": argmax_shape_func,
    #     "layout": "ROW_MAJOR",
    #     "num_repeats": 2,
    # },
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
    {
        "op": primary_moreh_mean_0,
        "name": "ttnn.operations.moreh.mean_dims_0",
    },
    {
        "op": primary_moreh_mean_01,
        "name": "ttnn.operations.moreh.mean_dims_01",
    },
    {
        "op": primary_moreh_mean_012,
        "name": "ttnn.operations.moreh.mean_dims_012",
    },
    {
        "op": primary_moreh_mean_013,
        "name": "ttnn.operations.moreh.mean_dims_013",
    },
    {
        "op": primary_moreh_mean_023,
        "name": "ttnn.operations.moreh.mean_dims_023",
    },
    {
        "op": primary_moreh_mean_0123,
        "name": "ttnn.operations.moreh.mean_dims_0123",
    },
    {
        "op": primary_moreh_mean_1,
        "name": "ttnn.operations.moreh.mean_dims_1",
    },
    {
        "op": primary_moreh_mean_12,
        "name": "ttnn.operations.moreh.mean_dims_12",
    },
    {
        "op": primary_moreh_mean_13,
        "name": "ttnn.operations.moreh.mean_dims_13",
    },
    {
        "op": primary_moreh_mean_123,
        "name": "ttnn.operations.moreh.mean_dims_123",
    },
    {
        "op": primary_moreh_mean_2,
        "name": "ttnn.operations.moreh.mean_dims_2",
    },
    {
        "op": primary_moreh_mean_23,
        "name": "ttnn.operations.moreh.mean_dims_23",
    },
    {
        "op": primary_moreh_mean_3,
        "name": "ttnn.operations.moreh.mean_dims_3",
    },
]

# Crashes
# {
#     "op": group_norm_no_weights,
#     "name": "ttnn.group_norm_no_weights",
# },

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
    ttnn.layer_norm(input=x, epsilon=0.0001, weight=y, bias=z)


def primary_layernorm(x, y, z):
    ttnn.layer_norm(input=x, epsilon=0.0001, weight=y, bias=z)


def norm_shapes_func(input_shape):
    input_shape_12 = [input_shape[0], input_shape[1], 32, input_shape[3]]
    return input_shape, input_shape_12, input_shape_12


def add_layernorm(x, y, z):
    ttnn.layer_norm(x, residual_input_tensor=x, epsilon=0.0001, weight=y, bias=z)


def primary_add_layernorm(x, y, z):
    ttnn.layer_norm(x, residual_input_tensor=x, epsilon=0.0001, weight=y, bias=z)


def group_norm(x, y, z):
    ttnn.group_norm(x, num_groups=32, epsilon=0.0001, weight=y, bias=x)


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
    ttnn.rms_norm(input=x, epsilon=0.0001, weight=y, bias=z)


def addcmul(x, y, z):
    ttnn.addcmul(x, y, z, value=2)


def addcdiv(x, y, z):
    ttnn.addcdiv(x, y, z, value=2)


def addalpha_bw(x, y, z):
    ttnn.addalpha_bw(x, y, z, alpha=5)


def addcmul_bw(x, y, z):
    ttnn.addcmul_bw(x, x, y, z, alpha=5)


def addcdiv_bw(x, y, z):
    ttnn.addcdiv_bw(x, x, y, z, alpha=5)


def where_bw(x, y, z):
    ttnn.where_bw(x, y, z, z)


def bias_gelu_bw_none(x, y, z):
    ttnn.bias_gelu_bw(x, y, z, approximate="none")


def bias_gelu_bw_tanh(x, y, z):
    ttnn.bias_gelu_bw(x, y, z, approximate="tanh")


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
    ttnn.div_bw(x, y, z, round_mode="None")


def add_bw(x, y, z):
    ttnn.add_bw(x, y, z)


def primary_moreh_norm_backward(x, y, z):
    tt_lib.operations.primary.moreh_norm_backward(x, y, z, p=2.0)


def linear(x, weight, bias):
    ttnn.linear(x, weight, bias=bias)


def linear_shape_func(input_shape):
    N = input_shape[-1]
    x_shape = [1, input_shape[-2], N]
    weight_shape = [N, N]
    bias_shape = [1, N]
    return x_shape, weight_shape, bias_shape


all_ternary_ops = [
    {
        "op": ttnn.mac,
        "name": "ttnn.mac",
    },
    {
        "op": ttnn.where,
        "name": "ttnn.where",
    },
    {
        "op": ttnn.lerp,
        "name": "ttnn.lerp",
    },
    {
        "op": layernorm,
        "name": "ttnn.layer_norm",
        "shape_func": norm_shapes_func,
    },
    {
        "op": primary_layernorm,
        "name": "ttnn.layer_norm",
        "shape_func": norm_shapes_func,
    },
    {
        "op": rmsnorm,
        "name": "ttnn.rms_norm",
        "shape_func": norm_shapes_func,
    },
    {
        "op": add_layernorm,
        "name": "ttnn.layer_norm",
        "shape_func": norm_shapes_func,
    },
    {
        "op": primary_add_layernorm,
        "name": "ttnn.layer_norm",
        "shape_func": norm_shapes_func,
    },
    {
        "op": addcmul,
        "name": "ttnn.addcmul",
    },
    {
        "op": addcdiv,
        "name": "ttnn.addcdiv",
    },
    {
        "op": addalpha_bw,
        "name": "ttnn.addalpha_bw",
    },
    {
        "op": addcmul_bw,
        "name": "ttnn.addcmul_bw",
    },
    {
        "op": addcdiv_bw,
        "name": "ttnn.addcdiv_bw",
    },
    # {
    #     "op": ttnn.eq_bw,
    #     "name": "ttnn.eq_bw",
    # },
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
        "op": add_bw,
        "name": "ttnn.add_bw",
    },
    # {
    #     "op": tt_lib.tensor.embedding_bw,
    #     "name": "tt_lib.tensor.embedding_bw",
    # },
    {
        "op": where_bw,
        "name": "ttnn.where_bw",
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
        "op": linear,
        "name": "ttnn.linear",
        "shape_func": linear_shape_func,
    },
    # {
    #     "op": ttnn.ge_bw,
    #     "name": "ttnn.ge_bw",
    # },
    # {
    #     "op": ttnn.gt_bw,
    #     "name": "ttnn.gt_bw",
    # },
    # {
    #     "op": ttnn.le_bw,
    #     "name": "ttnn.le_bw",
    # },
    # {
    #     "op": ttnn.lt_bw,
    #     "name": "ttnn.lt_bw",
    # },
    # {
    #     "op": ttnn.ne_bw,
    #     "name": "ttnn.ne_bw",
    # },
]

# Crashes
# {
#     "op": group_norm,
#     "name": "ttnn.group_norm",
# },

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
