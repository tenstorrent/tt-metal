"""
Central configuration for activation functions.
This module should be imported by all Python scripts that need the activation list.
"""

# All 29 valid activations (excluding GLU variants which are 2D operations)
# These are activations that can be approximated with 1D LUTs
ALL_ACTIVATIONS = [
    "atanh",
    "celu",
    "cos",
    "cosh",
    "elu",
    "erf",
    "exp",
    "gelu",
    "hardshrink",
    "hardsigmoid",
    "hardswish",
    "hardtanh",
    "leaky_relu",
    "logsigmoid",
    "mish",
    "prelu",
    "relu",
    "relu6",
    "selu",
    "sigmoid",
    "sin",
    "sinh",
    "softplus",
    "softshrink",
    "softsign",
    "swish",
    "tanh",
    "tanhshrink",
    "threshold",
]

# Activations with native SFPU hardware support (subset that actually work)
# Note: Some activations fail with hardware errors (atanh, celu, cosh, etc.)
NATIVE_SFPU_ACTIVATIONS = [
    "cos",
    "elu",
    "erf",
    "exp",
    "gelu",
    "hardsigmoid",
    "leaky_relu",
    "mish",
    "prelu",
    "relu",
    "relu6",
    "selu",
    "sigmoid",
    "sin",
    "softplus",
    "softsign",
    "swish",
    "tanh",
    "threshold",
]

# Standard depths for piecewise approximations
PIECEWISE_DEPTHS = [4, 8, 16, 32]

# Quick-run depths for testing
QUICK_RUN_DEPTHS = [32]

# Expected config counts per method (FULL SWEEP)
# Native SFPU: 29 activations (1 config each)
# Piecewise methods: 29 activations × N depths (depth caps: constant/linear=32, quadratic/cubic=16, hexic/octic=8)
EXPECTED_CONFIGS = {
    "native_sfpu": 29,
    "piecewise_constant": 116,  # 29 activations × 4 depths (4, 8, 16, 32)
    "piecewise_linear": 116,  # 29 activations × 4 depths (4, 8, 16, 32)
    "piecewise_quadratic": 87,  # 29 activations × 3 depths (4, 8, 16)
    "piecewise_quadratic_remez": 87,  # 29 activations × 3 depths (4, 8, 16)
    "piecewise_cubic_remez": 87,  # 29 activations × 3 depths (4, 8, 16) (degree-3)
    "piecewise_hexic_remez": 58,  # 29 activations × 2 depths (4, 8) (degree-6)
    "piecewise_octic_remez": 58,  # 29 activations × 2 depths (4, 8) (degree-8)
}

# Expected config counts per method (QUICK-RUN MODE)
# Native SFPU: 29 activations (1 config each)
# Piecewise methods: 29 activations × 1 depth (32) = 29 configs each
EXPECTED_CONFIGS_QUICK = {
    "native_sfpu": 29,
    "piecewise_constant": 29,
    "piecewise_linear": 29,
    "piecewise_quadratic": 29,
    "piecewise_quadratic_remez": 29,
    "piecewise_cubic_remez": 29,
    "piecewise_hexic_remez": 29,  # 29 activations × 1 depth (degree-6)
    "piecewise_octic_remez": 29,  # 29 activations × 1 depth (degree-8)
}
