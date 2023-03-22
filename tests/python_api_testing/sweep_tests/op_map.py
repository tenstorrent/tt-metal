from functools import partial

from python_api_testing.sweep_tests import (
    pytorch_ops,
    ttlib_ops,
)

op_map = {
    # Sanity
    "datacopy": {
        "ttlib_op": ttlib_ops.datacopy,
        "pytorch_op": pytorch_ops.datacopy,
    },
    # Eltwise unary
    "eltwise-exp": {
        "ttlib_op": ttlib_ops.eltwise_exp,
        "pytorch_op": pytorch_ops.exp,
    },
    "eltwise-recip": {
        "ttlib_op": ttlib_ops.eltwise_recip,
        "pytorch_op": pytorch_ops.recip,
    },
    "eltwise-sqrt": {
        "ttlib_op": ttlib_ops.eltwise_sqrt,
        "pytorch_op": pytorch_ops.sqrt,
    },
    "eltwise-gelu": {
        "ttlib_op": ttlib_ops.eltwise_gelu,
        "pytorch_op": pytorch_ops.gelu,
    },
    "eltwise-relu": {
        "ttlib_op": ttlib_ops.eltwise_relu,
        "pytorch_op": pytorch_ops.relu,
    },
    "eltwise-sigmoid": {
        "ttlib_op": ttlib_ops.eltwise_sigmoid,
        "pytorch_op": pytorch_ops.sigmoid,
    },
    "eltwise-log": {
        "ttlib_op": ttlib_ops.eltwise_log,
        "pytorch_op": pytorch_ops.log,
    },
    "eltwise-tanh": {
        "ttlib_op": ttlib_ops.eltwise_tanh,
        "pytorch_op": pytorch_ops.tanh,
    },
    # Eltwise binary
    "eltwise-add": {
        "ttlib_op": ttlib_ops.eltwise_add,
        "pytorch_op": pytorch_ops.add,
    },
    "eltwise-sub": {
        "ttlib_op": ttlib_ops.eltwise_sub,
        "pytorch_op": pytorch_ops.sub,
    },
    "eltwise-mul": {
        "ttlib_op": ttlib_ops.eltwise_mul,
        "pytorch_op": pytorch_ops.mul,
    },
    # Matmul
    "matmul": {
        "ttlib_op": ttlib_ops.matmul,
        "pytorch_op": pytorch_ops.matmul,
    },
    "bmm": {
        "ttlib_op": ttlib_ops.bmm,
        "pytorch_op": pytorch_ops.matmul,
    },
    # Broadcast
    "bcast-add-h": {
        "ttlib_op": ttlib_ops.bcast_add_h,
        "pytorch_op": pytorch_ops.add,
    },
    "bcast-add-w": {
        "ttlib_op": ttlib_ops.bcast_add_w,
        "pytorch_op": pytorch_ops.add,
    },
    "bcast-add-hw": {
        "ttlib_op": ttlib_ops.bcast_add_hw,
        "pytorch_op": pytorch_ops.add,
    },
    "bcast-sub-h": {
        "ttlib_op": ttlib_ops.bcast_sub_h,
        "pytorch_op": pytorch_ops.sub,
    },
    "bcast-sub-w": {
        "ttlib_op": ttlib_ops.bcast_sub_w,
        "pytorch_op": pytorch_ops.sub,
    },
    "bcast-sub-hw": {
        "ttlib_op": ttlib_ops.bcast_sub_hw,
        "pytorch_op": pytorch_ops.sub,
    },
    "bcast-mul-h": {
        "ttlib_op": ttlib_ops.bcast_mul_h,
        "pytorch_op": pytorch_ops.mul,
    },
    "bcast-mul-w": {
        "ttlib_op": ttlib_ops.bcast_mul_w,
        "pytorch_op": pytorch_ops.mul,
    },
    "bcast-mul-hw": {
        "ttlib_op": ttlib_ops.bcast_mul_hw,
        "pytorch_op": pytorch_ops.mul,
    },
    # Reduce
    "reduce-max-h": {
        "ttlib_op": ttlib_ops.reduce_max_h,
        "pytorch_op": partial(pytorch_ops.reduce_max, dims=(-2,)),
    },
    "reduce-max-w": {
        "ttlib_op": ttlib_ops.reduce_max_w,
        "pytorch_op": partial(pytorch_ops.reduce_max, dims=(-1,)),
    },
    "reduce-max-hw": {
        "ttlib_op": ttlib_ops.reduce_max_hw,
        "pytorch_op": partial(pytorch_ops.reduce_max, dims=(-2, -1)),
    },
    "reduce-sum-h": {
        "ttlib_op": ttlib_ops.reduce_sum_h,
        "pytorch_op": partial(pytorch_ops.reduce_sum, dims=(-2,)),
    },
    "reduce-sum-w": {
        "ttlib_op": ttlib_ops.reduce_sum_w,
        "pytorch_op": partial(pytorch_ops.reduce_sum, dims=(-1,)),
    },
    "reduce-sum-hw": {
        "ttlib_op": ttlib_ops.reduce_sum_hw,
        "pytorch_op": partial(pytorch_ops.reduce_sum, dims=(-2, -1)),
    },
    # Transpose
    "transpose-wh": {
        "ttlib_op": ttlib_ops.transpose_wh,
        "pytorch_op": partial(pytorch_ops.transpose, dim0=-2, dim1=-1),
    },
    "transpose-hc": {
        "ttlib_op": ttlib_ops.transpose_hc,
        "pytorch_op": partial(pytorch_ops.transpose, dim0=-3, dim1=-2),
    },
}
