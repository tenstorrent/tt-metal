from functools import partial

from python_api_testing.sweep_tests import (
    pytorch_ops,
    ttmetal_ops,
)

op_map = {
    # Sanity
    "datacopy": {
        "ttmetal_op": ttmetal_ops.datacopy,
        "pytorch_op": pytorch_ops.datacopy,
    },
    # Eltwise unary
    "eltwise-exp": {
        "ttmetal_op": ttmetal_ops.eltwise_exp,
        "pytorch_op": pytorch_ops.exp,
    },
    "eltwise-recip": {
        "ttmetal_op": ttmetal_ops.eltwise_recip,
        "pytorch_op": pytorch_ops.recip,
    },
    "eltwise-sqrt": {
        "ttmetal_op": ttmetal_ops.eltwise_sqrt,
        "pytorch_op": pytorch_ops.sqrt,
    },
    "eltwise-gelu": {
        "ttmetal_op": ttmetal_ops.eltwise_gelu,
        "pytorch_op": pytorch_ops.gelu,
    },
    "eltwise-relu": {
        "ttmetal_op": ttmetal_ops.eltwise_relu,
        "pytorch_op": pytorch_ops.relu,
    },
    "eltwise-sigmoid": {
        "ttmetal_op": ttmetal_ops.eltwise_sigmoid,
        "pytorch_op": pytorch_ops.sigmoid,
    },
    "eltwise-log": {
        "ttmetal_op": ttmetal_ops.eltwise_log,
        "pytorch_op": pytorch_ops.log,
    },
    "eltwise-tanh": {
        "ttmetal_op": ttmetal_ops.eltwise_tanh,
        "pytorch_op": pytorch_ops.tanh,
    },
    # Eltwise binary
    "eltwise-add": {
        "ttmetal_op": ttmetal_ops.eltwise_add,
        "pytorch_op": pytorch_ops.add,
    },
    "eltwise-sub": {
        "ttmetal_op": ttmetal_ops.eltwise_sub,
        "pytorch_op": pytorch_ops.sub,
    },
    "eltwise-mul": {
        "ttmetal_op": ttmetal_ops.eltwise_mul,
        "pytorch_op": pytorch_ops.mul,
    },
    # Matmul
    "matmul": {
        "ttmetal_op": ttmetal_ops.matmul,
        "pytorch_op": pytorch_ops.matmul,
    },
    "bmm": {
        "ttmetal_op": ttmetal_ops.bmm,
        "pytorch_op": pytorch_ops.matmul,
    },
    # Broadcast
    "bcast-add-h": {
        "ttmetal_op": ttmetal_ops.bcast_add_h,
        "pytorch_op": pytorch_ops.add,
    },
    "bcast-add-w": {
        "ttmetal_op": ttmetal_ops.bcast_add_w,
        "pytorch_op": pytorch_ops.add,
    },
    "bcast-add-hw": {
        "ttmetal_op": ttmetal_ops.bcast_add_hw,
        "pytorch_op": pytorch_ops.add,
    },
    "bcast-sub-h": {
        "ttmetal_op": ttmetal_ops.bcast_sub_h,
        "pytorch_op": pytorch_ops.sub,
    },
    "bcast-sub-w": {
        "ttmetal_op": ttmetal_ops.bcast_sub_w,
        "pytorch_op": pytorch_ops.sub,
    },
    "bcast-sub-hw": {
        "ttmetal_op": ttmetal_ops.bcast_sub_hw,
        "pytorch_op": pytorch_ops.sub,
    },
    "bcast-mul-h": {
        "ttmetal_op": ttmetal_ops.bcast_mul_h,
        "pytorch_op": pytorch_ops.mul,
    },
    "bcast-mul-w": {
        "ttmetal_op": ttmetal_ops.bcast_mul_w,
        "pytorch_op": pytorch_ops.mul,
    },
    "bcast-mul-hw": {
        "ttmetal_op": ttmetal_ops.bcast_mul_hw,
        "pytorch_op": pytorch_ops.mul,
    },
    # Reduce
    "reduce-max-h": {
        "ttmetal_op": ttmetal_ops.reduce_max_h,
        "pytorch_op": partial(pytorch_ops.reduce_max, dims=(-2,)),
    },
    "reduce-max-w": {
        "ttmetal_op": ttmetal_ops.reduce_max_w,
        "pytorch_op": partial(pytorch_ops.reduce_max, dims=(-1,)),
    },
    "reduce-max-hw": {
        "ttmetal_op": ttmetal_ops.reduce_max_hw,
        "pytorch_op": partial(pytorch_ops.reduce_max, dims=(-2, -1)),
    },
    "reduce-sum-h": {
        "ttmetal_op": ttmetal_ops.reduce_sum_h,
        "pytorch_op": partial(pytorch_ops.reduce_sum, dims=(-2,)),
    },
    "reduce-sum-w": {
        "ttmetal_op": ttmetal_ops.reduce_sum_w,
        "pytorch_op": partial(pytorch_ops.reduce_sum, dims=(-1,)),
    },
    "reduce-sum-hw": {
        "ttmetal_op": ttmetal_ops.reduce_sum_hw,
        "pytorch_op": partial(pytorch_ops.reduce_sum, dims=(-2, -1)),
    },
    # Transpose
    "transpose-wh": {
        "ttmetal_op": ttmetal_ops.transpose_wh,
        "pytorch_op": partial(pytorch_ops.transpose, dim0=-2, dim1=-1),
    },
    "transpose-hc": {
        "ttmetal_op": ttmetal_ops.transpose_hc,
        "pytorch_op": partial(pytorch_ops.transpose, dim0=-3, dim1=-2),
    },
}
