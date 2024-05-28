import tt_lib


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
    tt_lib.tensor.clamp_min_bw(x, y, 0.9)


def gelu_bw_none(x, y):
    tt_lib.tensor.gelu_bw(x, y, approximate="none")


def gelu_bw_tanh(x, y):
    tt_lib.tensor.gelu_bw(x, y, approximate="tanh")


all_binary_ops = [
    {
        "op": tt_lib.tensor.add,
        "name": "tt_lib.tensor.add",
    },
    {
        "op": tt_lib.tensor.sub,
        "name": "tt_lib.tensor.sub",
    },
    {
        "op": tt_lib.tensor.mul,
        "name": "tt_lib.tensor.mul",
    },
    {
        "op": tt_lib.tensor.div,
        "name": "tt_lib.tensor.div",
    },
    {
        "op": tt_lib.tensor.matmul,
        "name": "tt_lib.tensor.matmul",
    },
    {
        "op": tt_lib.tensor.hypot,
        "name": "tt_lib.tensor.hypot",
    },
    {
        "op": tt_lib.tensor.squared_difference,
        "name": "tt_lib.tensor.squared_difference",
    },
    {
        "op": tt_lib.tensor.logaddexp,
        "name": "tt_lib.tensor.logaddexp",
    },
    {
        "op": tt_lib.tensor.logaddexp2,
        "name": "tt_lib.tensor.logaddexp2",
    },
    {
        "op": tt_lib.tensor.atan2,
        "name": "tt_lib.tensor.atan2",
    },
    {
        "op": tt_lib.tensor.logical_xor,
        "name": "tt_lib.tensor.logical_xor",
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
        "op": tt_lib.tensor.ldexp,
        "name": "tt_lib.tensor.ldexp",
    },
    {
        "op": tt_lib.tensor.bias_gelu,
        "name": "tt_lib.tensor.bias_gelu",
    },
    {
        "op": tt_lib.tensor.logical_and,
        "name": "tt_lib.tensor.logical_and",
    },
    {
        "op": tt_lib.tensor.assign,
        "name": "tt_lib.tensor.assign(binary)",
    },
    {
        "op": isclose,
        "name": "tt_lib.tensor.isclose",
    },
    {
        "op": tt_lib.tensor.logical_or,
        "name": "tt_lib.tensor.logical_or",
    },
    {
        "op": tt_lib.tensor.gt,
        "name": "tt_lib.tensor.gt",
    },
    {
        "op": tt_lib.tensor.gte,
        "name": "tt_lib.tensor.gte",
    },
    {
        "op": tt_lib.tensor.lt,
        "name": "tt_lib.tensor.lt",
    },
    {
        "op": tt_lib.tensor.lte,
        "name": "tt_lib.tensor.lte",
    },
    {
        "op": tt_lib.tensor.eq,
        "name": "tt_lib.tensor.eq",
    },
    {
        "op": tt_lib.tensor.ne,
        "name": "tt_lib.tensor.ne",
    },
    {
        "op": where_binary_1,
        "name": "tt_lib.tensor.where(binary: x const y)",
    },
    {
        "op": where_binary_2,
        "name": "tt_lib.tensor.where(binary: x y const)",
    },
    {
        "op": tt_lib.tensor.matmul,
        "name": "tt_lib.tensor.matmul",
    },
    {
        "op": tt_lib.tensor.bmm,
        "name": "tt_lib.tensor.bmm",
    },
    {
        "op": tt_lib.tensor.copy,
        "name": "tt_lib.tensor.copy",
    },
    {
        "op": bcast_add_h,
        "name": "tt_lib.tensor.bcast(add h)",
        "bcast": True,
        "bcast_dim": tt_lib.tensor.BcastOpDim.H,
    },
    {
        "op": bcast_add_w,
        "name": "tt_lib.tensor.bcast(add w)",
        "bcast": True,
        "bcast_dim": tt_lib.tensor.BcastOpDim.W,
    },
    {
        "op": bcast_add_hw,
        "name": "tt_lib.tensor.bcast(add hw)",
        "bcast": True,
        "bcast_dim": tt_lib.tensor.BcastOpDim.HW,
    },
    {
        "op": bcast_sub_h,
        "name": "tt_lib.tensor.bcast(sub h)",
        "bcast": True,
        "bcast_dim": tt_lib.tensor.BcastOpDim.H,
    },
    {
        "op": bcast_sub_w,
        "name": "tt_lib.tensor.bcast(sub w)",
        "bcast": True,
        "bcast_dim": tt_lib.tensor.BcastOpDim.W,
    },
    {
        "op": bcast_sub_hw,
        "name": "tt_lib.tensor.bcast(sub hw)",
        "bcast": True,
        "bcast_dim": tt_lib.tensor.BcastOpDim.HW,
    },
    {
        "op": bcast_mul_h,
        "name": "tt_lib.tensor.bcast(mul h)",
        "bcast": True,
        "bcast_dim": tt_lib.tensor.BcastOpDim.H,
    },
    {
        "op": bcast_mul_w,
        "name": "tt_lib.tensor.bcast(mul w)",
        "bcast": True,
        "bcast_dim": tt_lib.tensor.BcastOpDim.W,
    },
    {
        "op": bcast_mul_hw,
        "name": "tt_lib.tensor.bcast(mul hw)",
        "bcast": True,
        "bcast_dim": tt_lib.tensor.BcastOpDim.HW,
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
        "name": "tt_lib.tensor.polar(binary)",
    },
    {
        "op": concat_0,
        "name": "tt_lib.tensor.concat(dim=0)",
    },
    {
        "op": concat_1,
        "name": "tt_lib.tensor.concat(dim=1)",
    },
    {
        "op": concat_2,
        "name": "tt_lib.tensor.concat(dim=2)",
    },
    {
        "op": concat_3,
        "name": "tt_lib.tensor.concat(dim=3)",
    },
    {
        "op": lerp_binary,
        "name": "tt_lib.tensor.lerp(binary)",
    },
    {
        "op": tt_lib.tensor.xlogy,
        "name": "tt_lib.tensor.xlogy",
    },
    {
        "op": tt_lib.tensor.embeddings,
        "name": "tt_lib.tensor.embeddings",
        "layout": "ROW_MAJOR",
        "embeddings_shapes": True,
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
        "op": tt_lib.tensor.binary_le_bw,
        "name": "tt_lib.tensor.binary_le_bw",
    },
    {
        "op": tt_lib.tensor.binary_le_bw,
        "name": "tt_lib.tensor.binary_le_bw",
    },
    {
        "op": gelu_bw_none,
        "name": "tt_lib.tensor.gelu_bw('none')",
    },
    {
        "op": gelu_bw_tanh,
        "name": "tt_lib.tensor.gelu_bw('tanh')",
    },
]

# # {
# #     "op": conv,
# #     "name": "tt_lib.tensor.conv",
# # },


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

    output_tensor_shape = (
        shape[-4],
        shape[-3],
        shape[-2] + 32,
        shape[-1] + 32,
    )

    tt_lib.tensor.pad(x, output_tensor_shape=output_tensor_shape, input_tensor_start=(0, 0, 0, 0), pad_value=1)


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
        "op": tt_lib.tensor.square,
        "name": "tt_lib.tensor.square",
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
        "op": tt_lib.tensor.multigammaln,
        "name": "tt_lib.tensor.multigammaln",
    },
    {
        "op": tt_lib.tensor.assign,
        "name": "tt_lib.tensor.assign(unary)",
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
        "name": "tt_lib.tensor.where(unary: x const const)",
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
        "name": "tt_lib.tensor.pad",
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
    },
    {
        "op": triu,
        "name": "tt_lib.tensor.triu",
    },
    {
        "op": reduce_sum_h,
        "name": "tt_lib.tensor.reduce(sum h)",
    },
    {
        "op": reduce_sum_w,
        "name": "tt_lib.tensor.reduce(sum w)",
    },
    {
        "op": reduce_sum_hw,
        "name": "tt_lib.tensor.reduce(sum hw)",
    },
    {
        "op": reduce_min_h,
        "name": "tt_lib.tensor.reduce(min h)",
    },
    {
        "op": reduce_min_w,
        "name": "tt_lib.tensor.reduce(min w)",
    },
    {
        "op": reduce_min_hw,
        "name": "tt_lib.tensor.reduce(min hw)",
    },
    {
        "op": reduce_max_h,
        "name": "tt_lib.tensor.reduce(max h)",
    },
    {
        "op": reduce_max_w,
        "name": "tt_lib.tensor.reduce(max w)",
    },
    {
        "op": reduce_max_hw,
        "name": "tt_lib.tensor.reduce(max hw)",
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
        "name": "tt_lib.tensor.sum(dim=0)",
    },
    {
        "op": sum_1,
        "name": "tt_lib.tensor.sum(dim=1)",
    },
    {
        "op": sum_2,
        "name": "tt_lib.tensor.sum(dim=2)",
    },
    {
        "op": sum_3,
        "name": "tt_lib.tensor.sum(dim=3)",
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
        "name": "tt_lib.tensor.erf(fast_and_approx=True)",
    },
    {
        "op": erf_slow,
        "name": "tt_lib.tensor.erf(fast_and_approx=False)",
    },
    {
        "op": tt_lib.tensor.erfc,
        "name": "tt_lib.tensor.erfc(fast_and_approx=True)",
    },
    {
        "op": erfc_slow,
        "name": "tt_lib.tensor.erfc(fast_and_approx=False)",
    },
    {
        "op": tt_lib.tensor.rsqrt,
        "name": "tt_lib.tensor.rsqrt(fast_and_approx=True)",
    },
    {
        "op": rsqrt_slow,
        "name": "tt_lib.tensor.rsqrt(fast_and_approx=False)",
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
        "name": "tt_lib.tensor.glu(dim=-1)",
    },
    {
        "op": geglu_1,
        "name": "tt_lib.tensor.geglu(dim=-1)",
    },
    {
        "op": reglu_1,
        "name": "tt_lib.tensor.reglu(dim=-1)",
    },
    {
        "op": swiglu_1,
        "name": "tt_lib.tensor.swiglu(dim=-1)",
    },
    {
        "op": glu_2,
        "name": "tt_lib.tensor.glu(dim=-2)",
    },
    {
        "op": geglu_2,
        "name": "tt_lib.tensor.geglu(dim=-2)",
    },
    {
        "op": reglu_2,
        "name": "tt_lib.tensor.reglu(dim=-2)",
    },
    {
        "op": swiglu_2,
        "name": "tt_lib.tensor.swiglu(dim=-2)",
    },
    {
        "op": repeat,
        "name": "tt_lib.tensor.repeat",
    },
    {
        "op": repeat_interleave_0,
        "name": "tt_lib.tensor.repeat_interleave(dim=0)",
    },
    {
        "op": repeat_interleave_1,
        "name": "tt_lib.tensor.repeat_interleave(dim=1)",
    },
    {
        "op": repeat_interleave_2,
        "name": "tt_lib.tensor.repeat_interleave(dim=2)",
    },
    {
        "op": pow_int,
        "name": "tt_lib.tensor.pow(int)",
    },
    {
        "op": pow_float,
        "name": "tt_lib.tensor.pow(float)",
    },
    {
        "op": tt_lib.tensor.identity,
        "name": "tt_lib.tensor.identity",
    },
    {
        "op": argmax_1,
        "name": "tt_lib.tensor.argmax(dim=-1)",
    },
    {
        "op": argmax_2,
        "name": "tt_lib.tensor.argmax(dim=-2)",
    },
    {
        "op": argmax_3,
        "name": "tt_lib.tensor.argmax(dim=-3)",
    },
    {
        "op": argmax_4,
        "name": "tt_lib.tensor.argmax(dim=-4)",
    },
    {
        "op": argmax_all,
        "name": "tt_lib.tensor.argmax(all)",
    },
    {
        "op": argmin_1,
        "name": "tt_lib.tensor.argmin(dim=-1)",
    },
    {
        "op": argmin_2,
        "name": "tt_lib.tensor.argmin(dim=-2)",
    },
    {
        "op": argmin_3,
        "name": "tt_lib.tensor.argmin(dim=-3)",
    },
    {
        "op": argmin_4,
        "name": "tt_lib.tensor.argmin(dim=-4)",
    },
    {
        "op": argmin_all,
        "name": "tt_lib.tensor.argmin(all)",
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
]

# {
#     "op": convert_conv_weight_tensor_to_tiled_layout, #  Unsupported storage type
#     "name": "tt_lib.tensor.convert_conv_weight_tensor_to_tiled_layout",
#     "layout": "ROW_MAJOR",
# },


def layernorm(x, y, z):
    tt_lib.tensor.layernorm(input=x, eps=0.0001, gamma=y, beta=z)


def add_layernorm(x, y, z):
    tt_lib.tensor.add_layernorm(a=x, b=x, eps=0.0001, gamma=y, beta=z)


def groupnorm(x, y, z):
    tt_lib.tensor.groupnorm(input=x, group_size=32, eps=0.0001, gamma=y, beta=z)


def rmsnorm(x, y, z):
    tt_lib.tensor.rmsnorm(input=x, eps=0.0001, gamma=y, beta=z)


def addcmul(x, y, z):
    tt_lib.tensor.addcmul(x, y, z, 2)


def addcdiv(x, y, z):
    tt_lib.tensor.addcdiv(x, y, z, 2)


def lamb_optimizer(x, y, z):
    tt_lib.tensor.lamb_optimizer(x, x, y, z, beta1=0.8, beta2=0.99, step_size=1e-3, eps=1e-6, weight_decay=0.02)


def addalpha_bw(x, y, z):
    tt_lib.tensor.addalpha_bw(x, y, z, alpha=5)


def addcmul_bw(x, y, z):
    tt_lib.tensor.addcmul_bw(x, x, y, z, value=5)


def addcdiv_bw(x, y, z):
    tt_lib.tensor.addcdiv_bw(x, x, y, z, value=5)


def where_bw(x, y, z):
    tt_lib.tensor.where_bw(x, y, z, z)


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
        "norm_shapes": True,
    },
    {
        "op": groupnorm,
        "name": "tt_lib.tensor.groupnorm",
    },
    {
        "op": rmsnorm,
        "name": "tt_lib.tensor.rmsnorm",
        "norm_shapes": True,
    },
    {
        "op": add_layernorm,
        "name": "tt_lib.tensor.add_layernorm",
        "norm_shapes": True,
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
    },
    {
        "op": addalpha_bw,
        "name": "tt_lib.tensor.addalpha_bw",
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
        "op": tt_lib.tensor.binary_assign_bw,
        "name": "tt_lib.tensor.binary_assign_bw",
    },
    {
        "op": tt_lib.tensor.div_bw,
        "name": "tt_lib.tensor.div_bw",
    },
    {
        "op": tt_lib.tensor.mul_bw,
        "name": "tt_lib.tensor.mul_bw",
    },
    {
        "op": tt_lib.tensor.max_bw,
        "name": "tt_lib.tensor.max_bw",
    },
    {
        "op": tt_lib.tensor.min_bw,
        "name": "tt_lib.tensor.min_bw",
    },
    {
        "op": tt_lib.tensor.add_bw,
        "name": "tt_lib.tensor.add_bw",
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
        "op": tt_lib.tensor.sub_bw,
        "name": "tt_lib.tensor.sub_bw",
    },
    {
        "op": tt_lib.tensor.rsub_bw,
        "name": "tt_lib.tensor.rsub_bw",
    },
    {
        "op": tt_lib.tensor.atan2_bw,
        "name": "tt_lib.tensor.atan2_bw",
    },
    {
        "op": tt_lib.tensor.hypot_bw,
        "name": "tt_lib.tensor.hypot_bw",
    },
]
