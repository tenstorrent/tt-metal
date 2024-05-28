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
]


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
]


all_ternary_ops = [
    {
        "op": tt_lib.tensor.mac,
        "name": "tt_lib.tensor.mac",
    },
    {
        "op": tt_lib.tensor.where,
        "name": "tt_lib.tensor.where",
    },
]
