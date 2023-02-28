from functools import partial

from python_api_testing.sweep_tests import (
    ll_buda_ops,
    pytorch_ops,
)

op_map = {
    # Sanity
    "datacopy": {
        "ll_buda_op": ll_buda_ops.datacopy,
        "pytorch_op": pytorch_ops.datacopy,
    },
    # Eltwise unary
    "eltwise-exp": {
        "ll_buda_op": ll_buda_ops.eltwise_exp,
        "pytorch_op": pytorch_ops.exp,
    },
    "eltwise-recip": {
        "ll_buda_op": ll_buda_ops.eltwise_recip,
        "pytorch_op": pytorch_ops.recip,
    },
    "eltwise-sqrt": {
        "ll_buda_op": ll_buda_ops.eltwise_sqrt,
        "pytorch_op": pytorch_ops.sqrt,
    },
    "eltwise-gelu": {
        "ll_buda_op": ll_buda_ops.eltwise_gelu,
        "pytorch_op": pytorch_ops.gelu,
    },
    "eltwise-relu": {
        "ll_buda_op": ll_buda_ops.eltwise_relu,
        "pytorch_op": pytorch_ops.relu,
    },
    # Eltwise binary
    "eltwise-add": {
        "ll_buda_op": ll_buda_ops.eltwise_add,
        "pytorch_op": pytorch_ops.add,
    },
    "eltwise-sub": {
        "ll_buda_op": ll_buda_ops.eltwise_sub,
        "pytorch_op": pytorch_ops.sub,
    },
    "eltwise-mul": {
        "ll_buda_op": ll_buda_ops.eltwise_mul,
        "pytorch_op": pytorch_ops.mul,
    },
    # Matmul
    "matmul": {
        "ll_buda_op": ll_buda_ops.matmul,
        "pytorch_op": pytorch_ops.matmul,
    },
    # Broadcast
    "bcast-add-h": {
        "ll_buda_op": ll_buda_ops.bcast_add_h,
        "pytorch_op": pytorch_ops.add,
    },
    "bcast-add-w": {
        "ll_buda_op": ll_buda_ops.bcast_add_w,
        "pytorch_op": pytorch_ops.add,
    },
    "bcast-add-hw": {
        "ll_buda_op": ll_buda_ops.bcast_add_hw,
        "pytorch_op": pytorch_ops.add,
    },
    "bcast-sub-h": {
        "ll_buda_op": ll_buda_ops.bcast_sub_h,
        "pytorch_op": pytorch_ops.sub,
    },
    "bcast-sub-w": {
        "ll_buda_op": ll_buda_ops.bcast_sub_w,
        "pytorch_op": pytorch_ops.sub,
    },
    "bcast-sub-hw": {
        "ll_buda_op": ll_buda_ops.bcast_sub_hw,
        "pytorch_op": pytorch_ops.sub,
    },
    "bcast-mul-h": {
        "ll_buda_op": ll_buda_ops.bcast_mul_h,
        "pytorch_op": pytorch_ops.mul,
    },
    "bcast-mul-w": {
        "ll_buda_op": ll_buda_ops.bcast_mul_w,
        "pytorch_op": pytorch_ops.mul,
    },
    "bcast-mul-hw": {
        "ll_buda_op": ll_buda_ops.bcast_mul_hw,
        "pytorch_op": pytorch_ops.mul,
    },
    # Reduce
    "reduce-max-h": {
        "ll_buda_op": ll_buda_ops.reduce_max_h,
        "pytorch_op": partial(pytorch_ops.reduce_max, dims=(-2,)),
    },
    "reduce-max-w": {
        "ll_buda_op": ll_buda_ops.reduce_max_w,
        "pytorch_op": partial(pytorch_ops.reduce_max, dims=(-1,)),
    },
    "reduce-max-hw": {
        "ll_buda_op": ll_buda_ops.reduce_max_hw,
        "pytorch_op": partial(pytorch_ops.reduce_max, dims=(-2, -1)),
    },
    "reduce-sum-h": {
        "ll_buda_op": ll_buda_ops.reduce_sum_h,
        "pytorch_op": partial(pytorch_ops.reduce_sum, dims=(-2,)),
    },
    "reduce-sum-w": {
        "ll_buda_op": ll_buda_ops.reduce_sum_w,
        "pytorch_op": partial(pytorch_ops.reduce_sum, dims=(-1,)),
    },
    "reduce-sum-hw": {
        "ll_buda_op": ll_buda_ops.reduce_sum_hw,
        "pytorch_op": partial(pytorch_ops.reduce_sum, dims=(-2, -1)),
    },
    # Transpose
    "transpose-wh": {
        "ll_buda_op": ll_buda_ops.transpose_wh,
        "pytorch_op": partial(pytorch_ops.transpose, dim0=-2, dim1=-1),
    },
    "transpose-hc": {
        "ll_buda_op": ll_buda_ops.transpose_hc,
        "pytorch_op": partial(pytorch_ops.transpose, dim0=-3, dim1=-2),
    },
}
