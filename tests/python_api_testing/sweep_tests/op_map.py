
from functools import partial

from python_api_testing.sweep_tests import (
    pytorch_ops,
    tt_lib_ops,
)



op_map = {
    ################################################
    ################# Helper-Funcs #################
    ################################################
    "linear":{
        "tt_lib_op": tt_lib_ops.linear,
        "pytorch_op": pytorch_ops.linear
    },
    ################################################
    #################### TT-DNN ####################
    ################################################
    # Eltwise unary
    "eltwise-ltz": {
        "tt_lib_op": tt_lib_ops.eltwise_ltz,
        "pytorch_op": pytorch_ops.ltz,
    },
    "eltwise-gtz": {
        "tt_lib_op": tt_lib_ops.eltwise_gtz,
        "pytorch_op": pytorch_ops.gtz,
    },
    "eltwise-lez": {
        "tt_lib_op": tt_lib_ops.eltwise_lez,
        "pytorch_op": pytorch_ops.lez,
    },
    "eltwise-gez": {
        "tt_lib_op": tt_lib_ops.eltwise_gez,
        "pytorch_op": pytorch_ops.gez,
    },
    "eltwise-eqz": {
        "tt_lib_op": tt_lib_ops.eltwise_eqz,
        "pytorch_op": pytorch_ops.eqz,
    },
    "eltwise-nez": {
        "tt_lib_op": tt_lib_ops.eltwise_nez,
        "pytorch_op": pytorch_ops.nez,
    },
    "eltwise-abs": {
        "tt_lib_op": tt_lib_ops.eltwise_abs,
        "pytorch_op": pytorch_ops.abs,
    },
    "eltwise-sign": {
        "tt_lib_op": tt_lib_ops.eltwise_sign,
        "pytorch_op": pytorch_ops.sign,
    },
    "eltwise-silu": {
        "tt_lib_op": tt_lib_ops.eltwise_silu,
        "pytorch_op": pytorch_ops.silu,
    },
    "eltwise-square": {
        "tt_lib_op": tt_lib_ops.eltwise_square,
        "pytorch_op": pytorch_ops.square,
    },
    "eltwise-mish": {
        "tt_lib_op": tt_lib_ops.eltwise_mish,
        "pytorch_op": pytorch_ops.mish,
    },
    "eltwise-softplus": {
        "tt_lib_op": tt_lib_ops.eltwise_softplus,
        "pytorch_op": pytorch_ops.softplus,
    },
    "eltwise-log1p": {
        "tt_lib_op": tt_lib_ops.eltwise_log1p,
        "pytorch_op": pytorch_ops.log1p,
    },
    "eltwise-add1": {
        "tt_lib_op": tt_lib_ops.eltwise_add1,
        "pytorch_op": pytorch_ops.add1,
    },
    "eltwise-neg": {
        "tt_lib_op": tt_lib_ops.eltwise_neg,
        "pytorch_op": pytorch_ops.neg,
    },
    "eltwise-swish": {
        "tt_lib_op": tt_lib_ops.eltwise_swish,
        "pytorch_op": pytorch_ops.swish,
    },
    "eltwise-cos": {
        "tt_lib_op": tt_lib_ops.eltwise_cos,
        "pytorch_op": pytorch_ops.cos,
    },
    "eltwise-sin": {
        "tt_lib_op": tt_lib_ops.eltwise_sin,
        "pytorch_op": pytorch_ops.sin,
    },
    "eltwise-exp": {
        "tt_lib_op": tt_lib_ops.eltwise_exp,
        "pytorch_op": pytorch_ops.exp,
    },
    "eltwise-recip": {
        "tt_lib_op": tt_lib_ops.eltwise_recip,
        "pytorch_op": pytorch_ops.recip,
    },
    "eltwise-sqrt": {
        "tt_lib_op": tt_lib_ops.eltwise_sqrt,
        "pytorch_op": pytorch_ops.sqrt,
    },
    "eltwise-gelu": {
        "tt_lib_op": tt_lib_ops.eltwise_gelu,
        "pytorch_op": pytorch_ops.gelu,
    },
    "eltwise-relu": {
        "tt_lib_op": tt_lib_ops.eltwise_relu,
        "pytorch_op": pytorch_ops.relu,
    },
    "eltwise-power": {
        "tt_lib_op": partial(tt_lib_ops.eltwise_power, exponent=7),
        "pytorch_op": partial(pytorch_ops.power, exponent=7),
    },
    "eltwise-relu_max": {
        "tt_lib_op": partial(tt_lib_ops.eltwise_relu_max, upper_limit=2),
        "pytorch_op": partial(pytorch_ops.relu_max, upper_limit=2),
    },
    "eltwise-relu_min": {
        "tt_lib_op": partial(tt_lib_ops.eltwise_relu_min, lower_limit=-2),
        "pytorch_op": partial(pytorch_ops.relu_min, lower_limit=-2),
    },
    "eltwise-polyval": {
        "tt_lib_op": partial(tt_lib_ops.eltwise_polyval, coeffs=[0.0, 1.0, 2.0, 1.0]),
        "pytorch_op": partial(pytorch_ops.polyval, coeffs=[0.0, 1.0, 2.0, 1.0]),
    },
    "eltwise-mac": {
        "tt_lib_op": tt_lib_ops.eltwise_mac,
        "pytorch_op": pytorch_ops.mac,
    },
    "eltwise-sigmoid": {
        "tt_lib_op": tt_lib_ops.eltwise_sigmoid,
        "pytorch_op": pytorch_ops.sigmoid,
    },
    "eltwise-log": {
        "tt_lib_op": tt_lib_ops.eltwise_log,
        "pytorch_op": pytorch_ops.log,
    },
    "eltwise-log2": {
        "tt_lib_op": tt_lib_ops.eltwise_log2,
        "pytorch_op": pytorch_ops.log2,
    },
    "eltwise-log10": {
        "tt_lib_op": tt_lib_ops.eltwise_log10,
        "pytorch_op": pytorch_ops.log10,
    },
    "eltwise-tanh": {
        "tt_lib_op": tt_lib_ops.eltwise_tanh,
        "pytorch_op": pytorch_ops.tanh,
    },
    # Eltwise binary
    "eltwise-add": {
        "tt_lib_op": tt_lib_ops.eltwise_add,
        "pytorch_op": pytorch_ops.add,
    },
    "eltwise-sub": {
        "tt_lib_op": tt_lib_ops.eltwise_sub,
        "pytorch_op": pytorch_ops.sub,
    },
    "eltwise-mul": {
        "tt_lib_op": tt_lib_ops.eltwise_mul,
        "pytorch_op": pytorch_ops.mul,
    },
    "eltwise-min": {
        "tt_lib_op": tt_lib_ops.eltwise_min,
        "pytorch_op": pytorch_ops.min,
    },
    "eltwise-max": {
        "tt_lib_op": tt_lib_ops.eltwise_max,
        "pytorch_op": pytorch_ops.max,
    },
    # Matmul
    "matmul": {
        "tt_lib_op": tt_lib_ops.matmul,
        "pytorch_op": pytorch_ops.matmul,
    },
    "bmm": {
        "tt_lib_op": tt_lib_ops.bmm,
        "pytorch_op": pytorch_ops.matmul,
    },
    # Broadcast
    "bcast-add-h": {
        "tt_lib_op": tt_lib_ops.bcast_add_h,
        "pytorch_op": pytorch_ops.add,
    },
    "bcast-add-w": {
        "tt_lib_op": tt_lib_ops.bcast_add_w,
        "pytorch_op": pytorch_ops.add,
    },
    "bcast-add-hw": {
        "tt_lib_op": tt_lib_ops.bcast_add_hw,
        "pytorch_op": pytorch_ops.add,
    },
    "bcast-sub-h": {
        "tt_lib_op": tt_lib_ops.bcast_sub_h,
        "pytorch_op": pytorch_ops.sub,
    },
    "bcast-sub-w": {
        "tt_lib_op": tt_lib_ops.bcast_sub_w,
        "pytorch_op": pytorch_ops.sub,
    },
    "bcast-sub-hw": {
        "tt_lib_op": tt_lib_ops.bcast_sub_hw,
        "pytorch_op": pytorch_ops.sub,
    },
    "bcast-mul-h": {
        "tt_lib_op": tt_lib_ops.bcast_mul_h,
        "pytorch_op": pytorch_ops.mul,
    },
    "bcast-mul-w": {
        "tt_lib_op": tt_lib_ops.bcast_mul_w,
        "pytorch_op": pytorch_ops.mul,
    },
    "bcast-mul-hw": {
        "tt_lib_op": tt_lib_ops.bcast_mul_hw,
        "pytorch_op": pytorch_ops.mul,
    },
    # Reduce
    "reduce-max-h": {
        "tt_lib_op": tt_lib_ops.reduce_max_h,
        "pytorch_op": partial(pytorch_ops.reduce_max, dims=(-2,)),
    },
    "reduce-max-w": {
        "tt_lib_op": tt_lib_ops.reduce_max_w,
        "pytorch_op": partial(pytorch_ops.reduce_max, dims=(-1,)),
    },
    "reduce-max-hw": {
        "tt_lib_op": tt_lib_ops.reduce_max_hw,
        "pytorch_op": partial(pytorch_ops.reduce_max, dims=(-2, -1)),
    },
    "reduce-sum-h": {
        "tt_lib_op": tt_lib_ops.reduce_sum_h,
        "pytorch_op": partial(pytorch_ops.reduce_sum, dims=(-2,)),
    },
    "reduce-sum-w": {
        "tt_lib_op": tt_lib_ops.reduce_sum_w,
        "pytorch_op": partial(pytorch_ops.reduce_sum, dims=(-1,)),
    },
    "reduce-sum-hw": {
        "tt_lib_op": tt_lib_ops.reduce_sum_hw,
        "pytorch_op": partial(pytorch_ops.reduce_sum, dims=(-2, -1)),
    },
    # Transpose
    "transpose-wh": {
        "tt_lib_op": tt_lib_ops.transpose_wh,
        "pytorch_op": partial(pytorch_ops.transpose, dim0=-2, dim1=-1),
    },
    "transpose-hc": {
        "tt_lib_op": tt_lib_ops.transpose_hc,
        "pytorch_op": partial(pytorch_ops.transpose, dim0=-3, dim1=-2),
    },
    "transpose-cn": {
        "tt_lib_op": tt_lib_ops.transpose_cn,
        "pytorch_op": partial(pytorch_ops.transpose, dim0=0, dim1=1),
    },
    "permute": {
        "tt_lib_op": tt_lib_ops.permute,
        "pytorch_op": pytorch_ops.permute,
    },
    "reshape": {
        "tt_lib_op": tt_lib_ops.reshape,
        "pytorch_op": pytorch_ops.reshape,
    },
    "tilize": {
        "tt_lib_op": tt_lib_ops.tilize,
        "pytorch_op": pytorch_ops.tilize,
    },
    "untilize": {
        "tt_lib_op": tt_lib_ops.untilize,
        "pytorch_op": pytorch_ops.untilize,
    },
    "tilize_with_zero_padding": {
        "tt_lib_op": tt_lib_ops.tilize_with_zero_padding,
        "pytorch_op": pytorch_ops.tilize_with_zero_padding,
    },
    "tilize_with_val_padding": {
        "tt_lib_op": tt_lib_ops.tilize_with_val_padding,
        "pytorch_op": pytorch_ops.tilize_with_val_padding,
    },
    "untilize_with_unpadding": {
        "tt_lib_op": tt_lib_ops.untilize_with_unpadding,
        "pytorch_op": pytorch_ops.untilize_with_unpadding,
    },
    "pad": {
        "tt_lib_op": tt_lib_ops.pad,
        "pytorch_op": pytorch_ops.pad,
    },
    "unpad": {
        "tt_lib_op": tt_lib_ops.unpad,
        "pytorch_op": pytorch_ops.unpad,
    },
    ################################################
    #################### Tensor ####################
    ################################################
    "datacopy": {
        "tt_lib_op": tt_lib_ops.datacopy,
        "pytorch_op": pytorch_ops.datacopy,
    },
    "tensor_pad": {
        "tt_lib_op": tt_lib_ops.tensor_pad,
        "pytorch_op": pytorch_ops.pad,
    },
    "tensor_unpad": {
        "tt_lib_op": tt_lib_ops.tensor_unpad,
        "pytorch_op": pytorch_ops.unpad,
    },
    "pad_to_tile": {
        "tt_lib_op": tt_lib_ops.pad_to_tile,
        "pytorch_op": pytorch_ops.pad_to_tile,
    },
    "unpad_from_tile": {
        "tt_lib_op": tt_lib_ops.unpad_from_tile,
        "pytorch_op": pytorch_ops.unpad_from_tile,
    },
}
