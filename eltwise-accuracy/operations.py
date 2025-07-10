import ttnn
import torch
import math


UNARY_OPERATIONS = {
    # Exponential functions
    "exp": (torch.exp, ttnn.exp, math.exp, "exp"),
    "exp_approx": (
        torch.exp,
        lambda x, output_tensor: ttnn.exp(x, fast_and_approximate_mode=True, output_tensor=output_tensor),
        None,
        "exp",
    ),
    "exp_cond": (
        torch.exp,
        ttnn.exp,
        None,
        "exp",
    ),
    "exp_approx0": (
        torch.exp,
        ttnn.exp,
        None,
        "exp",
    ),
    "exp_approx_21f": (
        torch.exp,
        ttnn.exp,
        None,
        "exp",
    ),
    "exp_hybrid": (
        torch.exp,
        ttnn.exp,
        None,
        "exp",
    ),
    # "exp_accurate_python": (
    #     torch.exp,
    #     exp_accurate_python,
    #     None,
    #     "exp",
    # ),
    # "exp_python_alt1": (
    #     torch.exp,
    #     lambda x, output_tensor: exp_accurate_python(x, output_tensor, exp_regression=exp_regression_0p5_to_1_alt1),
    #     None,
    #     "exp",
    # ),
    "tanh": (
        torch.tanh,
        ttnn.tanh,
        math.tanh,
        "tanh",
    ),  # ttnn.tanh() does not support output_tensor ?
    "tanh_accurate": (
        torch.tanh,
        lambda x, output_tensor: ttnn.tanh(x, accuracy=True, output_tensor=output_tensor),
        math.tanh,
        "tanh",
    ),
    "cosh": (
        torch.cosh,
        lambda x, output_tensor: ttnn.cosh(x),
        math.cosh,
        "cosh",
    ),  # ttnn.cosh() does not support output_tensor ?
    "sinh": (
        torch.sinh,
        lambda x, output_tensor: ttnn.sinh(x),
        math.sinh,
        "sinh",
    ),  # ttnn.sinh() does not support output_tensor ?
    # Logarithmic functions
    "log": (torch.log, ttnn.log, math.log, "log"),
    "log10": (torch.log10, ttnn.log10, math.log10, "log10"),
    "log2": (torch.log2, ttnn.log2, math.log2, "log2"),
    "log1p": (torch.log1p, ttnn.log1p, math.log1p, "log1p"),
    "logaddexp": (torch.logaddexp, ttnn.logaddexp, None, "logaddexp"),
    "logaddexp2": (torch.logaddexp2, ttnn.logaddexp2, None, "logaddexp2"),
    # Activation functions
    "silu": (lambda x, out: torch.nn.SiLU()(x), ttnn.silu, None, "silu"),
    "gelu": (lambda x, out: torch.nn.GELU()(x), ttnn.gelu, None, "gelu"),
    "gelu_approx": (
        lambda x, out: torch.nn.GELU()(x),
        lambda x, output_tensor: ttnn.gelu(x, fast_and_approximate_mode=True, output_tensor=output_tensor),
        None,
        "gelu",
    ),
    "logit": (
        torch.logit,
        lambda x, output_tensor: ttnn.logit(x),
        None,
        "logit",
    ),  # ttnn.logit does not support output_tensor ?
    "swish": (
        lambda x, out: torch.nn.SiLU()(x),
        lambda x, output_tensor: ttnn.swish(x),
        None,
        "swish",
    ),  # ttnn.swish does not support output_tensor ?
    "mish": (lambda x, out: torch.nn.Mish()(x), ttnn.mish, None, "mish"),
    "elu": (
        lambda x, out: torch.nn.ELU()(x),
        lambda x, output_tensor: ttnn.elu(x, output_tensor=output_tensor, alpha=1.0),
        None,
        "elu",
    ),  # Unlike torch, ttnn.elu does not use alpha=1 by default
    "selu": (
        lambda x, out: torch.nn.SELU()(x),
        lambda x, output_tensor: ttnn.selu(x),
        None,
        "selu",
    ),  # ttnn.selu does not support output_tensor ?
    "softplus": (lambda x, out: torch.nn.Softplus()(x), ttnn.softplus, None, "softplus"),
    "softsign": (
        lambda x, out: torch.nn.Softsign()(x),
        lambda x, output_tensor: ttnn.softsign(x),
        None,
        "softsign",
    ),  # ttnn.softsign does not support output_tensor ?
    # Trigonometric functions
    "tan": (torch.tan, ttnn.tan, math.tan, "tan"),
    "atan": (torch.atan, ttnn.atan, math.atan, "atan"),
    "atan2": (torch.atan2, ttnn.atan2, math.atan2, "atan2"),
    "sin": (torch.sin, ttnn.sin, math.sin, "sin"),
    "cos": (torch.cos, ttnn.cos, math.cos, "cos"),
    # Miscellaneous functions
    "sqrt": (torch.sqrt, ttnn.sqrt, math.sqrt, "sqrt"),
    "rsqrt": (
        torch.rsqrt,
        lambda x, output_tensor: ttnn.rsqrt(x, fast_and_approximate_mode=False, output_tensor=output_tensor),
        None,
        "rsqrt",
    ),
    "rsqrt_approx": (
        torch.rsqrt,
        lambda x, output_tensor: ttnn.rsqrt(x, fast_and_approximate_mode=True, output_tensor=output_tensor),
        None,
        "rsqrt",
    ),
    "reciprocal": (
        torch.reciprocal,
        ttnn.reciprocal,
        None,
        "reciprocal",
    ),
    "digamma": (
        torch.digamma,
        lambda x, output_tensor: ttnn.digamma(x),
        None,
        "digamma",
    ),  # ttnn.digamma does not support output_tensor ?
    "lgamma": (
        torch.lgamma,
        lambda x, output_tensor: ttnn.lgamma(x),
        math.lgamma,
        "lgamma",
    ),  # ttnn.lgamma does not support output_tensor ?
    "tanhshrink": (
        lambda x, out: torch.nn.Tanhshrink()(x),
        lambda x, output_tensor: ttnn.tanhshrink(x),
        None,
        "tanhshrink",
    ),  # ttnn.tan
}


BINARY_OPERATIONS = {
    "pow": {
        "ttnn": ttnn.pow,
        "torch": torch.pow,
    },
    "pow21f": {
        "ttnn": ttnn.pow,
        "torch": torch.pow,
    },
}
