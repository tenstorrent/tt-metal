#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Ground truth activation function implementations.

This module provides reference implementations of activation functions used for:
- LUT generation (constant, linear, quadratic, cubic approximations)
- Accuracy validation (extract_accuracy.py)
- Plotting and visualization
- Domain range specifications for input generation

All functions use PyTorch as the primary source of truth for all activation
functions. This ensures consistency with what ML models actually use.
Falls back to mpmath for scalar inputs when needed by Remez algorithm.

Domain Ranges:
Domain ranges are loaded from JSON configs in activations/*.json (single source of truth).
Each activation has one range defining the input domain for LUT generation and testing.

Example Usage:
    >>> from ttpoly.groundtruth.activations import get_activation_domain
    >>> get_activation_domain('sigmoid')
    (-10.0, 10.0)
    >>> get_activation_domain('atanh')
    (-0.999, 0.999)
"""

import importlib
import numpy as np
import json
from pathlib import Path
from .spec_context import current_spec_root, spec_cache_key

# Import PyTorch (required)
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

# Try to import mpmath for Remez algorithm support (optional)
try:
    from mpmath import mp

    MPMATH_AVAILABLE = True
except ImportError:
    MPMATH_AVAILABLE = False


# ============================================================================
# JSON Configuration Loading
# ============================================================================

_ACTIVATION_CONFIG_CACHE = {}


def load_activation_config(name):
    """
    Load activation configuration from JSON file.

    Args:
        name: Activation function name (e.g., 'relu', 'tanh')

    Returns:
        Dictionary with activation metadata, or None if not found
    """
    key = spec_cache_key(name)
    if key in _ACTIVATION_CONFIG_CACHE:
        return _ACTIVATION_CONFIG_CACHE[key]

    root = current_spec_root()
    if root is not None and (Path(name).name != name or name in {".", ".."}):
        raise ValueError(f"invalid activation specification name: {name!r}")
    config_file = (root or Path(__file__).parent.parent.parent / "activations") / f"{name}.json"

    if not config_file.exists():
        if root is not None:
            raise FileNotFoundError(config_file)
        return None

    try:
        with open(config_file) as f:
            config = json.load(f)
            _ACTIVATION_CONFIG_CACHE[key] = config
            return config
    except (json.JSONDecodeError, IOError) as e:
        if root is not None:
            raise
        print(f"Warning: Failed to load config for {name}: {e}")
        return None


def get_activation_domain_from_config(name):
    """Get the fit/search campaign domain from an activation config."""
    config = load_activation_config(name)
    if config:
        from ttpoly.spec.activation_config import campaign_domain_from_config

        return campaign_domain_from_config(config)
    return None


def get_piecewise_breakpoints_from_config(name):
    """Get piecewise breakpoints for an activation from its JSON config."""
    config = load_activation_config(name)
    if config and "piecewise" in config:
        return config["piecewise"].get("breakpoints", [])
    return []


def get_critical_points_from_config(name):
    """
    Get critical points for an activation from its JSON config.

    Args:
        name: Activation function name

    Returns:
        List of {x, value} dicts, or empty list if none defined
    """
    config = load_activation_config(name)
    if config and "critical_points" in config:
        return [{"x": cp["x"], "value": cp["value"]} for cp in config["critical_points"]]
    return []


def get_all_activations():
    """
    Get list of all activation names from JSON config files.

    Returns:
        Sorted list of activation names
    """
    activations_dir = current_spec_root() or Path(__file__).parent.parent.parent / "activations"
    if not activations_dir.exists():
        return []
    return sorted([f.stem for f in activations_dir.glob("*.json")])


def get_sollya_expr_from_config(name, piece_index=0):
    """
    Get Sollya expression for an activation from its JSON config.

    Args:
        name: Activation function name
        piece_index: Which piece for piecewise functions (default: 0)

    Returns:
        Sollya expression string, or None if not found
    """
    config = load_activation_config(name)

    if not config:
        return None

    # Piecewise activations - check this first
    if "piecewise" in config:
        pieces = config["piecewise"].get("pieces", [])
        if piece_index < len(pieces):
            return pieces[piece_index]["sollya_expr"]

    # Simple (non-piecewise) activations
    if "sollya_expr" in config:
        return config["sollya_expr"]

    return None


def get_asymptotic_config_for_piece(name, piece_index=0):
    """
    Get asymptotic factorization config for a specific piece of a piecewise function.

    Args:
        name: Activation function name
        piece_index: Which piece (0 = first/leftmost piece)

    Returns:
        Dictionary with asymptotic config if enabled, None otherwise.
        Config includes: dominant_factor, correction_form, correction_degree, etc.
    """
    config = load_activation_config(name)

    if not config:
        return None

    if "piecewise" not in config:
        return None

    pieces = config["piecewise"].get("pieces", [])
    if piece_index >= len(pieces):
        return None

    piece = pieces[piece_index]
    asymptotic = piece.get("asymptotic", {})

    if not asymptotic.get("enabled", False):
        return None

    # Return full asymptotic config along with sollya_expr from piece
    return {
        "sollya_expr": piece["sollya_expr"],
        "dominant_factor": asymptotic["dominant_factor"],
        "dominant_class": asymptotic.get("dominant_class", "unknown"),
        "correction_form": asymptotic.get("correction_form", "polynomial"),
        "correction_degree": asymptotic.get("correction_degree", 8),
        "runtime_strategy": asymptotic.get("runtime_strategy", "use_exp"),
        "notes": asymptotic.get("notes", ""),
    }


def get_piece_index_for_segment(name, seg_start, seg_end):
    """
    Determine which piece index a segment falls into based on breakpoints.

    Args:
        name: Activation function name
        seg_start: Segment start
        seg_end: Segment end

    Returns:
        Piece index (0-based), or 0 if no piecewise config
    """
    config = load_activation_config(name)

    if not config or "piecewise" not in config:
        return 0

    breakpoints = config["piecewise"].get("breakpoints", [])

    # Use segment midpoint to determine which piece
    seg_mid = (seg_start + seg_end) / 2

    # Breakpoints define boundaries between pieces
    # e.g., breakpoints = [-5.5, 0.0, 5.5] means:
    #   piece 0: x < -5.5
    #   piece 1: -5.5 <= x < 0.0
    #   piece 2: 0.0 <= x < 5.5
    #   piece 3: x >= 5.5
    for i, bp in enumerate(breakpoints):
        if seg_mid < bp:
            return i

    return len(breakpoints)


# ============================================================================
# PyTorch Ground Truth Implementations
# ============================================================================


def _require_torch():
    """Return the torch module or raise a clear lazy dependency error."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for this ground-truth operation. Install with: pip install torch")
    return torch


def _to_torch(x):
    """Convert numpy array or scalar to torch tensor.

    Uses FP64 for ground truth computation to match tt-metal standard:
    https://github.com/nmauriceTT/ttnn-eltwise-op-tester

    Some activations still need stable formulas on top of FP64; GELU, for
    example, must use the erfc CDF identity instead of 1+erf in the negative tail.
    """
    torch_module = _require_torch()
    if isinstance(x, torch_module.Tensor):
        return x.to(torch_module.float64)
    if isinstance(x, np.ndarray):
        return torch_module.from_numpy(x.astype(np.float64))
    return torch_module.tensor(float(x), dtype=torch_module.float64)


def _to_numpy(x):
    """Convert torch tensor to numpy array or scalar."""
    if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        result = x.detach().cpu().numpy()
        # Return scalar if input was scalar
        return float(result) if result.ndim == 0 else result
    return x


def _is_mpmath_type(x):
    """Check if input is mpmath type (for Remez algorithm)."""
    return MPMATH_AVAILABLE and hasattr(x, "__class__") and "mpmath" in str(type(x))


def get_torch_native_function(func_name: str):
    """
    Get a torch-native version of an activation function for autograd.

    Returns a function that takes torch tensors and returns torch tensors,
    allowing PyTorch autograd to compute exact derivatives (no finite difference errors).

    Args:
        func_name: Name of activation function (e.g., 'gelu', 'sigmoid')

    Returns:
        Torch-native function or None if not available
    """
    if not TORCH_AVAILABLE:
        return None

    torch_natives = {
        "gelu": lambda x: 0.5
        * x
        * torch.special.erfc(-x / torch.sqrt(torch.tensor(2.0, dtype=x.dtype, device=x.device))),
        "gelu_bw": lambda x: 0.5
        * torch.special.erfc(-x / torch.sqrt(torch.tensor(2.0, dtype=x.dtype, device=x.device)))
        + x * torch.exp(-(x**2) / 2) / torch.sqrt(torch.tensor(2.0 * torch.pi, dtype=x.dtype, device=x.device)),
        "sigmoid": lambda x: torch.sigmoid(x),
        "sigmoid_bw": lambda x: torch.sigmoid(x) * (1 - torch.sigmoid(x)),
        "tanh": lambda x: torch.tanh(x),
        # sech(x)^2, NOT 1 - tanh(x)^2. The subtraction cancels catastrophically:
        # in fp64 tanh(19) rounds to exactly 1.0, so 1 - tanh(x)^2 returns
        # EXACTLY 0 for every |x| >= 19 while the true value is 1.26e-16 there and
        # stays a normal bf16 out to |x| = 44.35. tanh_bw's declared domain is
        # [-45, 45], so the naive form hands segmentation an identically-zero
        # function over more than half the domain. 1/cosh(x)^2 is exact to the
        # domain edge (cosh(45)^2 = 2.9e38, well inside fp64).
        "tanh_bw": lambda x: 1.0 / torch.cosh(x) ** 2,
        "relu": lambda x: torch.relu(x),
        "silu": lambda x: torch.nn.functional.silu(x),
        "silu_bw": lambda x: torch.sigmoid(x) * (1 + x * (1 - torch.sigmoid(x))),
        "swish": lambda x: torch.nn.functional.silu(x),  # swish = silu
        "leaky_relu": lambda x: torch.nn.functional.leaky_relu(x),
        "elu": lambda x: torch.nn.functional.elu(x),
        "softplus": lambda x: torch.nn.functional.softplus(x),
        "softplus_bw": lambda x: torch.sigmoid(x),  # derivative of softplus = sigmoid
        "mish": lambda x: x * torch.tanh(torch.nn.functional.softplus(x)),
    }
    return torch_natives.get(func_name)


def sigmoid(x):
    """
    Sigmoid activation: σ(x) = 1 / (1 + exp(-x))

    Range: (0, 1)
    Properties: Smooth, bounded, vanishing gradients at extremes
    Source: PyTorch torch.sigmoid()
    """
    # mpmath scalar path for Remez algorithm
    if _is_mpmath_type(x):
        return float(1.0 / (1.0 + mp.exp(-x)))

    # PyTorch path for all other inputs
    return _to_numpy(torch.sigmoid(_to_torch(x)))


def tanh(x):
    """
    Hyperbolic tangent: tanh(x)

    Range: (-1, 1)
    Properties: Smooth, bounded, symmetric around origin
    Source: PyTorch torch.tanh()
    """
    # mpmath scalar path for Remez algorithm
    if _is_mpmath_type(x):
        return float(mp.tanh(x))

    # PyTorch path for all other inputs
    return _to_numpy(torch.tanh(_to_torch(x)))


def gelu(x):
    """
    Gaussian Error Linear Unit: GELU(x) = 0.5 * x * erfc(-x / sqrt(2))

    Range: (-∞, ∞)
    Properties: Smooth, unbounded, widely used in transformers
    Source: PyTorch torch.nn.functional.gelu(approximate='none'), evaluated via
    the stable erfc identity to avoid 1+erf cancellation in the negative tail.
    """
    # mpmath scalar path for Remez algorithm
    if _is_mpmath_type(x):
        return float(0.5 * x * mp.erfc(-x / mp.sqrt(2.0)))

    # PyTorch path for all other inputs
    xt = _to_torch(x)
    return _to_numpy(
        0.5 * xt * torch.special.erfc(-xt / torch.sqrt(torch.tensor(2.0, dtype=xt.dtype, device=xt.device)))
    )


def relu(x):
    """
    Rectified Linear Unit: ReLU(x) = max(0, x)

    Range: [0, ∞)
    Properties: Simple, unbounded, non-smooth at x=0
    Source: PyTorch torch.nn.functional.relu()
    """
    # mpmath scalar path for Remez algorithm
    if _is_mpmath_type(x):
        return float(max(0.0, x))

    # PyTorch path for all other inputs
    return _to_numpy(torch.nn.functional.relu(_to_torch(x)))


def leaky_relu(x, alpha=0.01):
    """
    Leaky ReLU: leaky_relu(x) = max(αx, x)

    Range: (-∞, ∞)
    Properties: Prevents dead neurons, non-smooth at x=0
    Default: α = 0.01
    Source: PyTorch torch.nn.functional.leaky_relu()
    """
    # mpmath scalar path for Remez algorithm
    if _is_mpmath_type(x):
        return float(x if x > 0 else alpha * x)

    # PyTorch path for all other inputs
    return _to_numpy(torch.nn.functional.leaky_relu(_to_torch(x), alpha))


def softplus(x):
    """
    Softplus: softplus(x) = log(1 + exp(x))

    Range: (0, ∞)
    Properties: Smooth approximation to ReLU
    Source: PyTorch torch.nn.functional.softplus()
    Note: For x > 20, approximated as x to avoid overflow
    """
    # mpmath scalar path for Remez algorithm
    if _is_mpmath_type(x):
        return float(mp.log(1 + mp.exp(x)) if x <= 20 else x)

    # PyTorch path for all other inputs
    return _to_numpy(torch.nn.functional.softplus(_to_torch(x)))


def exp(x):
    """
    Exponential function: exp(x)

    Range: (0, ∞)
    Properties: Unbounded growth, large dynamic range
    Source: PyTorch torch.exp()
    """
    # mpmath scalar path for Remez algorithm
    if _is_mpmath_type(x):
        return float(mp.exp(x))

    # PyTorch path for all other inputs
    return _to_numpy(torch.exp(_to_torch(x)))


def elu(x, alpha=1.0):
    """
    Exponential Linear Unit: ELU(x) = x if x > 0 else α(exp(x) - 1)

    Range: (-α, ∞)
    Properties: Smooth, reduces vanishing gradient problem
    Default: α = 1.0
    Source: PyTorch torch.nn.functional.elu()
    """
    # mpmath scalar path for Remez algorithm
    if _is_mpmath_type(x):
        return float(x if x > 0 else alpha * (mp.exp(x) - 1))

    # PyTorch path for all other inputs
    return _to_numpy(torch.nn.functional.elu(_to_torch(x), alpha))


def selu(x):
    """
    Scaled ELU: SELU(x) = λ * ELU(x, α)

    Range: (-λα, ∞)
    Properties: Self-normalizing, maintains mean=0 and variance=1
    Source: PyTorch torch.nn.functional.selu()
    Parameters: λ ≈ 1.0507, α ≈ 1.6733 (derived theoretically)
    """
    # mpmath scalar path for Remez algorithm
    if _is_mpmath_type(x):
        scale = 1.0507009873554804934193349852946
        alpha = 1.6732632423543772848170429916717
        return scale * elu(x, alpha=alpha)

    # PyTorch path for all other inputs
    return _to_numpy(torch.nn.functional.selu(_to_torch(x)))


def mish(x):
    """
    Mish activation: mish(x) = x * tanh(softplus(x))

    Range: (-∞, ∞)
    Properties: Smooth, unbounded, self-regularized
    Source: PyTorch torch.nn.functional.mish()
    """
    # mpmath scalar path for Remez algorithm
    if _is_mpmath_type(x):
        return float(x * tanh(softplus(x)))

    # PyTorch path for all other inputs
    return _to_numpy(torch.nn.functional.mish(_to_torch(x)))


def hardsigmoid(x):
    """
    Hard Sigmoid (PyTorch standard): hardsigmoid(x) = clip(x/6 + 0.5, 0, 1)

    Piecewise definition:
    - 0 if x ≤ -3
    - 1 if x ≥ +3
    - x/6 + 1/2 otherwise

    Range: [0, 1]
    Properties: Piecewise linear approximation to sigmoid, faster
    Source: PyTorch torch.nn.functional.hardsigmoid()
    Reference: PyTorch nn.Hardsigmoid
    """
    # mpmath scalar path for Remez algorithm
    if _is_mpmath_type(x):
        return float(max(0.0, min(1.0, x / 6.0 + 0.5)))

    # PyTorch path for all other inputs
    return _to_numpy(torch.nn.functional.hardsigmoid(_to_torch(x)))


def softsign(x):
    """
    Softsign: softsign(x) = x / (1 + |x|)

    Range: (-1, 1)
    Properties: Similar to tanh but polynomial, approaches limits slower
    Source: PyTorch torch.nn.functional.softsign()
    """
    # mpmath scalar path for Remez algorithm
    if _is_mpmath_type(x):
        return float(x / (1 + abs(x)))

    # PyTorch path for all other inputs
    return _to_numpy(torch.nn.functional.softsign(_to_torch(x)))


def sin(x):
    """
    Sine function: sin(x)

    Range: [-1, 1]
    Properties: Periodic, smooth
    Source: PyTorch torch.sin()
    """
    # mpmath scalar path for Remez algorithm
    if _is_mpmath_type(x):
        return float(mp.sin(x))

    # PyTorch path for all other inputs
    return _to_numpy(torch.sin(_to_torch(x)))


def cos(x):
    """
    Cosine function: cos(x)

    Range: [-1, 1]
    Properties: Periodic, smooth
    Source: PyTorch torch.cos()
    """
    # mpmath scalar path for Remez algorithm
    if _is_mpmath_type(x):
        return float(mp.cos(x))

    # PyTorch path for all other inputs
    return _to_numpy(torch.cos(_to_torch(x)))


def erf(x):
    """
    Error function: erf(x) = (2/√π) ∫₀ˣ exp(-t²) dt

    Range: (-1, 1)
    Properties: Smooth, bounded, used in GELU
    Source: PyTorch torch.erf()
    """
    # mpmath scalar path for Remez algorithm
    if _is_mpmath_type(x):
        return float(mp.erf(x))

    # PyTorch path for all other inputs
    return _to_numpy(torch.erf(_to_torch(x)))


def erfinv(x):
    """
    Inverse error function: erfinv(erf(x)) = x

    Domain: (-1, 1)
    Range: (-∞, ∞)
    Properties: Odd function, erfinv(0) = 0, erfinv(±1) = ±∞
    Source: PyTorch torch.erfinv()
    """
    # mpmath scalar path for Remez algorithm
    if _is_mpmath_type(x):
        return float(mp.erfinv(x))

    # PyTorch path for all other inputs
    return _to_numpy(torch.erfinv(_to_torch(x)))


# ============================================================================
# Additional Activations (from kernel_bench and PyTorch)
# ============================================================================


def logsigmoid(x):
    """
    Log-Sigmoid: logsigmoid(x) = log(sigmoid(x)) = -log(1 + exp(-x))

    Range: (-∞, 0]
    Properties: Numerically stable log-probability, used in BCE loss
    Source: PyTorch torch.nn.functional.logsigmoid()
    Note: More stable than log(sigmoid(x)) for large |x|
    """
    # mpmath scalar path for Remez algorithm
    if _is_mpmath_type(x):
        return float(-mp.log(1 + mp.exp(-x)))

    # PyTorch path for all other inputs
    return _to_numpy(torch.nn.functional.logsigmoid(_to_torch(x)))


def sinh(x):
    """
    Hyperbolic Sine: sinh(x) = (exp(x) - exp(-x)) / 2

    Range: (-∞, ∞)
    Properties: Odd function, unbounded, used in attention mechanisms
    Source: PyTorch torch.sinh()
    """
    # mpmath scalar path for Remez algorithm
    if _is_mpmath_type(x):
        return float(mp.sinh(x))

    # PyTorch path for all other inputs
    return _to_numpy(torch.sinh(_to_torch(x)))


def cosh(x):
    """
    Hyperbolic Cosine: cosh(x) = (exp(x) + exp(-x)) / 2

    Range: [1, ∞)
    Properties: Even function, always >= 1, used with sinh
    Source: PyTorch torch.cosh()
    """
    # mpmath scalar path for Remez algorithm
    if _is_mpmath_type(x):
        return float(mp.cosh(x))

    # PyTorch path for all other inputs
    return _to_numpy(torch.cosh(_to_torch(x)))


def atanh(x):
    """
    Inverse Hyperbolic Tangent: atanh(x) = 0.5 * log((1+x)/(1-x))

    Range: (-1, 1) → (-∞, ∞)
    Properties: Inverse of tanh, used in Fisher transformations
    Source: PyTorch torch.atanh()
    Domain: Input must be in (-1, 1)
    """
    # mpmath scalar path for Remez algorithm
    if _is_mpmath_type(x):
        return float(0.5 * mp.log((1.0 + x) / (1.0 - x)))

    # PyTorch path for all other inputs
    return _to_numpy(torch.atanh(_to_torch(x)))


def softshrink(x, lambda_=0.5):
    """
    Soft Shrinkage: softshrink(x) = x - λ if x > λ; x + λ if x < -λ; else 0

    Range: (-∞, ∞)
    Properties: Piecewise linear, sparse coding, L1 regularization
    Default: λ = 0.5
    Source: PyTorch torch.nn.functional.softshrink()
    """
    # mpmath scalar path for Remez algorithm
    if _is_mpmath_type(x):
        if x > lambda_:
            return float(x - lambda_)
        elif x < -lambda_:
            return float(x + lambda_)
        else:
            return 0.0

    # PyTorch path for all other inputs
    return _to_numpy(torch.nn.functional.softshrink(_to_torch(x), lambda_))


def relu6(x):
    """
    ReLU6: relu6(x) = min(max(0, x), 6)

    Range: [0, 6]
    Properties: Bounded ReLU, used in MobileNets, quantization-friendly
    Source: PyTorch torch.nn.functional.relu6()
    """
    # mpmath scalar path for Remez algorithm
    if _is_mpmath_type(x):
        return float(max(0.0, min(6.0, x)))

    # PyTorch path for all other inputs
    return _to_numpy(torch.nn.functional.relu6(_to_torch(x)))


def hardtanh(x, min_val=-1.0, max_val=1.0):
    """
    Hard Tanh: hardtanh(x) = clip(x, min_val, max_val)

    Range: [min_val, max_val]
    Properties: Piecewise linear approximation to tanh, faster
    Default: min_val=-1, max_val=1
    Source: PyTorch torch.nn.functional.hardtanh()
    """
    # mpmath scalar path for Remez algorithm
    if _is_mpmath_type(x):
        return float(max(min_val, min(max_val, x)))

    # PyTorch path for all other inputs
    return _to_numpy(torch.nn.functional.hardtanh(_to_torch(x), min_val, max_val))


def celu(x, alpha=1.0):
    """
    Continuously Differentiable ELU: CELU(x) = max(0,x) + min(0, α*(exp(x/α)-1))

    Range: (-α, ∞)
    Properties: Smoother than ELU, better gradient flow
    Default: α = 1.0
    Source: PyTorch torch.nn.functional.celu()
    """
    # mpmath scalar path for Remez algorithm
    if _is_mpmath_type(x):
        return float(max(0, x) + min(0, alpha * (mp.exp(x / alpha) - 1)))

    # PyTorch path for all other inputs
    return _to_numpy(torch.nn.functional.celu(_to_torch(x), alpha))


def tanhshrink(x):
    """
    Tanh Shrink: tanhshrink(x) = x - tanh(x)

    Range: (-∞, ∞)
    Properties: Residual-like, approaches ±∞ linearly
    Source: PyTorch torch.nn.functional.tanhshrink()
    """
    # mpmath scalar path for Remez algorithm (arbitrary precision: no cancellation)
    if _is_mpmath_type(x):
        return float(x - tanh(x))

    # x - tanh(x) cancels catastrophically near 0 in fp64: it is exactly 0 below
    # |x| ~ 1.3e-8 while the true value is x^3/3, and loses relative accuracy
    # from |x| ~ 1e-3 down. Use the odd series there (relative error ~7e-17 at
    # the crossover); the direct form keeps >30 significant bits above it.
    xt = _to_torch(x)
    x2 = xt * xt
    series = xt * x2 * (1.0 / 3.0 + x2 * (-2.0 / 15.0 + x2 * (17.0 / 315.0)))
    direct = torch.nn.functional.tanhshrink(xt)
    return _to_numpy(torch.where(xt.abs() < 1e-3, series, direct))


def hardshrink(x, lambda_=0.5):
    """
    Hard Shrinkage: hardshrink(x) = x if |x| > λ; else 0

    Range: (-∞, ∞)
    Properties: Hard thresholding, sparse representations
    Default: λ = 0.5
    Source: PyTorch torch.nn.functional.hardshrink()
    """
    # mpmath scalar path for Remez algorithm
    if _is_mpmath_type(x):
        return float(x if abs(x) > lambda_ else 0.0)

    # PyTorch path for all other inputs
    return _to_numpy(torch.nn.functional.hardshrink(_to_torch(x), lambda_))


def prelu(x, alpha=0.25):
    """
    Parametric ReLU: PReLU(x) = max(0, x) + α*min(0, x)

    Range: (-∞, ∞)
    Properties: Learnable slope for negative values
    Default: α = 0.25 (typical learned value)
    Source: PyTorch torch.nn.functional.prelu()
    """
    # mpmath scalar path for Remez algorithm
    if _is_mpmath_type(x):
        return float(x if x > 0 else alpha * x)

    # PyTorch path for all other inputs
    x_t = _to_torch(x)
    return _to_numpy(torch.nn.functional.prelu(x_t, torch.tensor(alpha, dtype=x_t.dtype)))


def threshold(x, threshold_val=0.0, value=0.0):
    """
    Threshold: threshold(x) = x if x > threshold; else value

    Range: Depends on parameters
    Properties: Simple hard thresholding
    Defaults: threshold=0, value=0 (equivalent to ReLU)
    Source: PyTorch torch.nn.functional.threshold()
    """
    # mpmath scalar path for Remez algorithm
    if _is_mpmath_type(x):
        return float(x if x > threshold_val else value)

    # PyTorch path for all other inputs
    return _to_numpy(torch.nn.functional.threshold(_to_torch(x), threshold_val, value))


def hardswish(x):
    """
    Hard Swish (PyTorch standard): hardswish(x) = x * hardsigmoid(x)

    Piecewise definition:
    - 0 if x ≤ -3
    - x if x ≥ +3
    - x * (x + 3) / 6 otherwise

    Equivalent to: x * clip(x/6 + 0.5, 0, 1)

    Range: [0, ∞)
    Properties: Piecewise linear approximation to swish, MobileNetV3
    Source: PyTorch torch.nn.functional.hardswish()
    Reference: PyTorch nn.Hardswish
    """
    # mpmath scalar path for Remez algorithm
    if _is_mpmath_type(x):
        return float(x * hardsigmoid(x))

    # PyTorch path for all other inputs
    return _to_numpy(torch.nn.functional.hardswish(_to_torch(x)))


def digamma(x):
    """
    Digamma function: ψ(x) = d/dx[ln(Γ(x))] = Γ'(x)/Γ(x)

    Range: (0, ∞) → (-∞, ∞)
    Properties: Logarithmic derivative of gamma function, pole at x=0
    Source: PyTorch torch.digamma()

    For x > 0:
    - Monotonically increasing
    - Zero crossing at x ≈ 1.461632
    - Asymptotic: ψ(x) → ln(x) - 1/(2x) for large x
    """
    # mpmath scalar path for Remez algorithm
    if _is_mpmath_type(x):
        return float(mp.digamma(x))

    # PyTorch path for all other inputs
    return _to_numpy(torch.digamma(_to_torch(x)))


def lgamma(x):
    """
    Log-Gamma function: lgamma(x) = ln(|Γ(x)|)

    Range: (0, ∞) → (-∞, ∞)
    Properties: Natural logarithm of absolute value of gamma function, pole at x<=0
    Source: PyTorch torch.lgamma()

    For x > 0:
    - Monotonically increasing for x > 1.461632 (digamma zero crossing)
    - Local minimum at x ≈ 1.461632
    - Asymptotic: lgamma(x) → x*ln(x) - x for large x (Stirling's approximation)
    """
    # mpmath scalar path for Remez algorithm
    if _is_mpmath_type(x):
        return float(mp.loggamma(x))

    # PyTorch path for all other inputs
    return _to_numpy(torch.lgamma(_to_torch(x)))


# ============================================================================
# Additional Math Functions (sqrt, log, etc.)
# ============================================================================


def sqrt(x):
    """Square root."""
    return _to_numpy(torch.sqrt(_to_torch(x)))


def cbrt(x):
    """Cube root."""
    # PyTorch doesn't have cbrt, use power
    return _to_numpy(torch.pow(torch.abs(_to_torch(x)), 1 / 3) * torch.sign(_to_torch(x)))


def reciprocal(x):
    """Reciprocal: 1/x."""
    return _to_numpy(torch.reciprocal(_to_torch(x)))


def rsqrt(x):
    """Reciprocal square root: 1/sqrt(x)."""
    return _to_numpy(torch.rsqrt(_to_torch(x)))


def log(x):
    """Natural logarithm."""
    return _to_numpy(torch.log(_to_torch(x)))


def log2(x):
    """Base-2 logarithm."""
    return _to_numpy(torch.log2(_to_torch(x)))


def log10(x):
    """Base-10 logarithm."""
    return _to_numpy(torch.log10(_to_torch(x)))


def log1p(x):
    """log(1 + x), accurate for small x."""
    return _to_numpy(torch.log1p(_to_torch(x)))


def exp2(x):
    """Base-2 exponential: 2^x."""
    return _to_numpy(torch.exp2(_to_torch(x)))


def expm1(x):
    """exp(x) - 1, accurate for small x."""
    return _to_numpy(torch.expm1(_to_torch(x)))


def tan(x):
    """Tangent."""
    return _to_numpy(torch.tan(_to_torch(x)))


def atan(x):
    """Arctangent."""
    return _to_numpy(torch.atan(_to_torch(x)))


def identity(x):
    """Identity function: f(x) = x."""
    return _to_numpy(_to_torch(x))


# Cache for dynamically loaded activation functions
# ALL activations are loaded from activations/*.json - no hardcoded functions
# This dict is populated on-demand by get_activation_function()
ACTIVATION_FUNCTIONS = {}


def _eval_sollya_expr(x, expr):
    """Evaluate a Sollya expression string using numpy.

    Expressions come from local activations/*.json files (user-controlled).
    """
    # Convert Sollya exponentiation syntax (^) to Python (**)
    # This handles cases like x^2, x^(1/2), etc.
    expr = expr.replace("^", "**")

    safe_ns = {
        "x": x,
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
        "abs": np.abs,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "asin": np.arcsin,
        "acos": np.arccos,
        "atan": np.arctan,
        "sinh": np.sinh,
        "cosh": np.cosh,
        "tanh": np.tanh,
        "asinh": np.arcsinh,
        "acosh": np.arccosh,
        "atanh": np.arctanh,
        "pi": np.pi,
        "e": np.e,
        "log2": np.log2,
        "log10": np.log10,
        "log1p": np.log1p,
        "exp2": np.exp2,
        "expm1": np.expm1,
        "min": np.minimum,
        "max": np.maximum,
        # Sign-correct real (odd) cube root. Sollya has no cbrt builtin, so
        # this name never appears in a fitting `sollya_expr`; it exists for
        # `golden_expr` forms (cbrt.json) where "x^(1/3)" would be the complex
        # principal root (NaN under numpy) for x < 0.
        "cbrt": np.cbrt,
    }

    # Torch-backed gamma-family functions (used by sollya_expr in
    # multigammaln.json / polygamma.json). These are defined here so the
    # ground truth does not depend on scipy being installed. torch is the
    # accuracy ground truth for the rest of this file; we follow the same
    # convention here. Inputs/outputs stay as numpy float64.
    if TORCH_AVAILABLE:

        def _gt_lgamma(x):
            return _to_numpy(torch.lgamma(_to_torch(x)))

        def _gt_digamma(x):
            return _to_numpy(torch.digamma(_to_torch(x)))

        def _gt_polygamma(n, x):
            # torch.special.polygamma(n, input): nth derivative of digamma.
            return _to_numpy(torch.special.polygamma(int(n), _to_torch(x)))

        def _gt_multigammaln(x, p):
            # Multivariate log-gamma: sum_{i=0}^{p-1} lgamma(x - i/2)
            #                         + (p*(p-1)/4) * log(pi)
            t = _to_torch(x)
            p = int(p)
            acc = torch.zeros_like(t)
            for i in range(p):
                acc = acc + torch.lgamma(t - 0.5 * i)
            acc = acc + (p * (p - 1) / 4.0) * float(np.log(np.pi))
            return _to_numpy(acc)

        safe_ns.update(
            {
                "lgamma": _gt_lgamma,
                "digamma": _gt_digamma,
                "polygamma": _gt_polygamma,
                "multigammaln": _gt_multigammaln,
                # ratapprox parses `trigamma` natively but not `polygamma(1, x)`,
                # so trigamma is the canonical sollya_expr spelling for psi^(1).
                "trigamma": lambda x: _gt_polygamma(1, x),
            }
        )

    try:
        import scipy.special as _sp

        safe_ns.update(
            {
                "erf": _sp.erf,
                "erfc": _sp.erfc,
                "erfinv": _sp.erfinv,
                "i0": _sp.i0,
                "i1": _sp.i1,
            }
        )
        # Only fall back to scipy for the gamma family if torch is absent,
        # so torch stays the ground truth when both are installed.
        if not TORCH_AVAILABLE:
            safe_ns.update(
                {
                    "lgamma": _sp.gammaln,
                    "polygamma": _sp.polygamma,
                    "multigammaln": _sp.multigammaln,
                    "digamma": _sp.digamma,
                    "trigamma": lambda x: _sp.polygamma(1, x),
                }
            )
    except ImportError:
        pass
    return eval(expr, {"__builtins__": {}}, safe_ns)


def _tanhshrink_series_golden(x):
    """Cancellation-free tanhshrink golden (odd series below |x| = 1e-3).

    This is THE canonical stable tanhshrink form, referenced from
    ``activations/tanhshrink.json`` via ``"golden_impl":
    "tanhshrink_odd_series"``. fp64 ``x - tanh(x)`` is exactly 0 below
    |x| ~ 1.3e-8 (true value x^3/3) and bleeds relative accuracy below
    |x| ~ 1e-3; the odd series has relative error ~7e-17 at the crossover.
    The fp32 mirror in
    ``deployment/generic_lut_activation/activation_reference.hpp`` uses the
    same series with an fp32-appropriate crossover (0.1); keep the two in
    sync when touching either.
    """
    x = np.asarray(x, dtype=np.float64)
    x2 = x * x
    series = x * x2 * (1.0 / 3.0 + x2 * (-2.0 / 15.0 + x2 * (17.0 / 315.0)))
    return np.where(np.abs(x) < 1e-3, series, x - np.tanh(x))


# Numerically-stable golden implementations addressable from the spec.
# activations/<name>.json opts in via "golden_impl": "<key>" together with
# "golden_strategy": ["golden_impl", ...]. Use this ONLY for stable forms that
# cannot be written as a golden_expr while staying bit-identical (e.g. the
# tanhshrink branch-select: expression-encoding the series would round
# differently through np.power).
_STABLE_GOLDEN_IMPLS = {
    "tanhshrink_odd_series": _tanhshrink_series_golden,
}


# RETIRED name-keyed golden override layer — must stay EMPTY. Every golden
# branch fix now lives in the activation's spec (activations/<name>.json):
#   * cbrt       -> "golden_expr": "cbrt(x)" (sign-correct real root)
#   * gelu       -> "golden_strategy": ["sollya_expr"] (erfc form, see
#                    torch_gelu_bug.md)
#   * tanhshrink -> "golden_impl": "tanhshrink_odd_series"
# tests/test_golden_strategy_migration.py asserts this dict stays empty and
# that the spec-driven goldens are bit-identical to the retired overrides.
_NUMPY_OVERRIDES = {}


#: Legacy golden resolution order, used when a config carries no explicit
#: "golden_strategy". Do NOT reorder: goldens must stay bit-identical.
_DEFAULT_GOLDEN_STRATEGY = ("pytorch", "piecewise", "sollya_expr")

#: Strategy names a config's "golden_strategy" list may reference.
_KNOWN_GOLDEN_STRATEGIES = frozenset({"golden_expr", "golden_impl", "pytorch", "piecewise", "sollya_expr"})


def _create_activation_from_config(name):
    """
    Dynamically create an activation function from its JSON config.

    The config's optional ``"golden_strategy"`` (an ordered preference list)
    decides how the golden is built; without it the legacy order applies:

    1. ``pytorch``: resolve the function from the config's pytorch field
    2. ``piecewise``: build from the piecewise definition (numpy + breakpoints)
    3. ``sollya_expr``: evaluate the top-level sollya_expr with numpy

    Two further strategies are available only via an explicit
    ``"golden_strategy"``:

    * ``golden_expr``: evaluate the config's ``"golden_expr"`` — a
      numerically-correct closed form for goldens (extended namespace, e.g.
      ``cbrt``) that the fitting ``sollya_expr`` cannot express because Sollya
      must still parse the latter.
    * ``golden_impl``: use the registered stable implementation named by the
      config's ``"golden_impl"`` (see ``_STABLE_GOLDEN_IMPLS``).

    Returns:
        Callable activation function, or None if unable to create
    """
    config = load_activation_config(name)
    if config is None:
        return None

    # Retired override layer (kept as an emergency escape hatch; must be empty
    # — see _NUMPY_OVERRIDES above).
    _override = _NUMPY_OVERRIDES.get(str(name).lower())
    if _override is not None:

        def _override_activation(x, _fn=_override):
            if _is_mpmath_type(x):
                x = float(x)
            x_arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
            result = _fn(x_arr)
            return result if np.asarray(x).ndim > 0 else float(result[0])

        return _override_activation

    for strategy in config.get("golden_strategy", _DEFAULT_GOLDEN_STRATEGY):
        if strategy not in _KNOWN_GOLDEN_STRATEGIES:
            raise ValueError(
                f"activations/{name}.json: unknown golden_strategy entry "
                f"{strategy!r} (known: {sorted(_KNOWN_GOLDEN_STRATEGIES)})"
            )
        builder = _GOLDEN_STRATEGY_BUILDERS[strategy]
        activation = builder(name, config)
        if activation is not None:
            return activation

    return None


def _build_golden_expr_activation(name, config):
    """Strategy ``golden_expr``: numerically-correct golden closed form."""
    expr = config.get("golden_expr")
    if not expr:
        return None

    def _golden_expr_activation(x, _expr=expr):
        if _is_mpmath_type(x):
            x = float(x)
        x_arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
        result = _eval_sollya_expr(x_arr, _expr)
        return result if np.asarray(x).ndim > 0 else float(result[0])

    return _golden_expr_activation


def _build_golden_impl_activation(name, config):
    """Strategy ``golden_impl``: registered stable implementation."""
    impl_name = config.get("golden_impl")
    if not impl_name:
        return None
    impl = _STABLE_GOLDEN_IMPLS.get(impl_name)
    if impl is None:
        raise ValueError(
            f"activations/{name}.json: golden_impl {impl_name!r} is not a "
            f"registered stable golden (known: {sorted(_STABLE_GOLDEN_IMPLS)})"
        )

    def _golden_impl_activation(x, _fn=impl):
        if _is_mpmath_type(x):
            x = float(x)
        x_arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
        result = _fn(x_arr)
        return result if np.asarray(x).ndim > 0 else float(result[0])

    return _golden_impl_activation


def _build_pytorch_activation(name, config):
    """Strategy ``pytorch``: resolve from the config's pytorch field."""
    pytorch_info = config.get("pytorch", {})
    if pytorch_info and TORCH_AVAILABLE:
        module_path = pytorch_info.get("module", "")
        func_name = pytorch_info.get("function", "")
        pytorch_args = tuple(pytorch_info.get("args", []))
        pytorch_kwargs = dict(pytorch_info.get("kwargs", {}))
        if module_path and func_name:
            try:
                mod = importlib.import_module(module_path)
                torch_func = getattr(mod, func_name)
                # Verify it works with a single tensor input
                torch_func(torch.tensor([2.0], dtype=torch.float64), *pytorch_args, **pytorch_kwargs)

                def _pytorch_activation(x, _fn=torch_func, _args=pytorch_args, _kwargs=pytorch_kwargs):
                    if _is_mpmath_type(x):
                        x = float(x)
                    return _to_numpy(_fn(_to_torch(x), *_args, **_kwargs))

                return _pytorch_activation
            except Exception:
                pass
    return None


def _build_piecewise_activation(name, config):
    """Strategy ``piecewise``: build from the piecewise definition."""
    if "piecewise" not in config:
        return None
    from ttpoly.spec.activation_config import (
        piecewise_boundary_policy_from_config,
        piecewise_special_factors_from_config,
    )

    policy = piecewise_boundary_policy_from_config(config)
    if policy is None:
        raise ValueError(f"activations/{name}.json piecewise strategy has no typed boundary policy")
    breakpoints = policy.breakpoints
    boundary_owners = policy.boundary_owners
    pieces = sorted(config["piecewise"].get("pieces", []), key=lambda p: p["index"])
    special_factors = piecewise_special_factors_from_config(config) if policy.explicit else None

    def _piecewise_activation(
        x,
        _bp=breakpoints,
        _owners=boundary_owners,
        _pieces=pieces,
        _special=special_factors,
    ):
        if _is_mpmath_type(x):
            x = float(x)
        x_arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
        result = np.empty_like(x_arr)
        finite = np.isfinite(x_arr)
        bins = np.searchsorted(_bp, x_arr, side="right")
        for boundary_index, (breakpoint, owner) in enumerate(zip(_bp, _owners)):
            if owner == "left":
                bins[finite & (x_arr == breakpoint)] = boundary_index
        for i, piece in enumerate(_pieces):
            mask = finite & (bins == i)
            if not np.any(mask):
                continue
            expr = piece["sollya_expr"]
            if expr == "0":
                result[mask] = 0.0
            elif expr == "x":
                result[mask] = x_arr[mask]
            else:
                result[mask] = _eval_sollya_expr(x_arr[mask], expr)
        nonfinite = ~finite
        if np.any(nonfinite):
            if _special is None:
                # Compatibility for configs not yet migrated to explicit
                # ownership.  Their callers retain the historical all-right
                # route until their mathematical and special contracts are
                # proven and declared together.
                legacy_bins = np.digitize(x_arr[nonfinite], _bp)
                for i, piece in enumerate(_pieces):
                    mask = legacy_bins == i
                    if not np.any(mask):
                        continue
                    expr = piece["sollya_expr"]
                    if expr == "0":
                        result[np.flatnonzero(nonfinite)[mask]] = 0.0
                    elif expr == "x":
                        result[np.flatnonzero(nonfinite)[mask]] = x_arr[nonfinite][mask]
                    else:
                        result[np.flatnonzero(nonfinite)[mask]] = _eval_sollya_expr(x_arr[nonfinite][mask], expr)
            else:
                result[np.isnan(x_arr)] = _special["nan"]
                result[np.isposinf(x_arr)] = _special["pos_inf"]
                result[np.isneginf(x_arr)] = _special["neg_inf"]
        return result if np.asarray(x).ndim > 0 else float(result[0])

    return _piecewise_activation


def _build_sollya_expr_activation(name, config):
    """Strategy ``sollya_expr``: evaluate the top-level sollya_expr directly."""
    if "sollya_expr" not in config:
        return None
    expr = config["sollya_expr"]

    def _expr_activation(x, _expr=expr):
        if _is_mpmath_type(x):
            x = float(x)
        x_arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
        result = _eval_sollya_expr(x_arr, _expr)
        return result if np.asarray(x).ndim > 0 else float(result[0])

    return _expr_activation


_GOLDEN_STRATEGY_BUILDERS = {
    "golden_expr": _build_golden_expr_activation,
    "golden_impl": _build_golden_impl_activation,
    "pytorch": _build_pytorch_activation,
    "piecewise": _build_piecewise_activation,
    "sollya_expr": _build_sollya_expr_activation,
}


def get_activation(name):
    """
    Get activation function by name.

    Dynamically creates a function from the activation's JSON config
    (activations/<name>.json) and caches it for future lookups.

    JSON configs specify how to create the function via:
    1. pytorch.module + pytorch.function - Use PyTorch function
    2. piecewise - Build from breakpoints and piece expressions
    3. sollya_expr - Evaluate expression with numpy

    Args:
        name: Activation function name (case-insensitive)

    Returns:
        Activation function callable

    Raises:
        KeyError: If activation name not recognized and no JSON config exists
    """
    name_lower = name.lower()

    # Check cache first
    key = spec_cache_key(name_lower)
    if key in ACTIVATION_FUNCTIONS:
        return ACTIVATION_FUNCTIONS[key]

    # Load from JSON config (single source of truth)
    func = _create_activation_from_config(name_lower)
    if func is not None:
        ACTIVATION_FUNCTIONS[key] = func
        return func

    available = ", ".join(sorted(get_all_activations()))
    raise KeyError(
        f"Unknown activation '{name}'. "
        f"Create activations/{name_lower}.json with pytorch, piecewise, or sollya_expr field. "
        f"Available: {available}"
    )


# Alias for compatibility
get_activation_function = get_activation


def get_activation_domain(name):
    """
    Get domain range for an activation function from JSON config.

    Args:
        name: Activation function name (case-insensitive)

    Returns:
        Tuple of (min, max) from the JSON configuration

    Raises:
        KeyError: If activation name not recognized or JSON config missing

    Example:
        >>> get_activation_domain('sigmoid')
        (-10.0, 10.0)
        >>> get_activation_domain('atanh')
        (-0.999, 0.999)
    """
    name_lower = name.lower()

    # Load from JSON config (required)
    json_domain = get_activation_domain_from_config(name_lower)
    if json_domain is not None:
        return json_domain

    # No CSV fallback - JSON is required
    raise KeyError(f"No JSON config found for activation '{name}'. " f"Expected file: activations/{name_lower}.json")


def compute_ground_truth(activation_name, inputs):
    """
    Compute ground truth outputs for given activation and inputs.

    Args:
        activation_name: Name of activation function
        inputs: Input values (numpy array or scalar)

    Returns:
        Ground truth outputs (same shape as inputs)

    Example:
        >>> import numpy as np
        >>> x = np.linspace(-5, 5, 100)
        >>> y = compute_ground_truth('relu', x)
    """
    func = get_activation(activation_name)
    return func(inputs)


if __name__ == "__main__":
    """Quick validation that all functions work with PyTorch backend."""
    all_activations = get_all_activations()
    print("Ground Truth Activation Functions (JSON-based, PyTorch Backend)")
    print("=" * 80)
    print(f"Total JSON configs: {len(all_activations)}")
    print(f"Available: {', '.join(sorted(all_activations))}")
    print(f"PyTorch available: {TORCH_AVAILABLE}")
    print(f"mpmath available: {MPMATH_AVAILABLE}")
    print()

    # Test all functions with a sample input
    x_test = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    print("Testing with sample input:", x_test)
    print("-" * 80)

    for name in sorted(all_activations):
        try:
            func = get_activation(name)
            y = func(x_test)
            print(f"{name:15s}: {y}")
        except Exception as e:
            print(f"{name:15s}: ERROR - {e}")

    print()
    print("=" * 80)
    print(f"✓ Tested {len(all_activations)} activation functions from JSON configs")


# ============================================================================
# Sollya Expression Lookup
# ============================================================================


# Cache for Sollya expressions loaded from JSON
# Populated lazily by get_sollya_expression()
_SOLLYA_EXPRESSIONS_CACHE = {}


def get_sollya_expression(name):
    """
    Get Sollya expression for an activation function.

    Loads from activations/<name>.json's sollya_expr field.
    Results are cached for performance.

    Args:
        name: Activation function name (case-insensitive)

    Returns:
        Sollya expression string (e.g., 'asin(x)', 'exp(x) - 1')

    Raises:
        KeyError: If activation not found or has no sollya_expr
    """
    name_lower = name.lower()

    # Check cache
    key = spec_cache_key(name_lower)
    if key in _SOLLYA_EXPRESSIONS_CACHE:
        return _SOLLYA_EXPRESSIONS_CACHE[key]

    # Load from JSON
    config = load_activation_config(name_lower)
    if config and "sollya_expr" in config:
        expr = config["sollya_expr"]
        _SOLLYA_EXPRESSIONS_CACHE[key] = expr
        return expr

    available = ", ".join(
        sorted(
            n for n in get_all_activations() if load_activation_config(n) and "sollya_expr" in load_activation_config(n)
        )
    )
    raise KeyError(
        f"No sollya_expr found for '{name}'. "
        f"Add 'sollya_expr' to activations/{name_lower}.json. "
        f"Available: {available}"
    )


def get_sollya_expression_for_piece(func_name, piece_index=0):
    """
    Get Sollya expression for a specific piece of a piecewise function from JSON config.

    Args:
        func_name: Activation function name
        piece_index: Which piece (0 = first/leftmost piece)

    Returns:
        Sollya expression string

    Raises:
        ValueError: If JSON config not found or expression missing
    """
    # Load from JSON config (required)
    json_expr = get_sollya_expr_from_config(func_name, piece_index)
    if json_expr is not None:
        return json_expr

    # No fallback - JSON is required
    raise ValueError(
        f"No sollya expression found for '{func_name}' piece {piece_index}. "
        f"Expected JSON config at: activations/{func_name}.json"
    )


def supports_sollya(func_name):
    """Check if activation function can be expressed in Sollya (has JSON config)."""
    config = load_activation_config(func_name)
    if not config:
        return False
    # Check if it has sollya_expr (simple) or piecewise pieces with sollya_expr
    return "sollya_expr" in config or "piecewise" in config


def is_piecewise_linear_activation(func_name):
    """Check if activation is piecewise linear (not worth using Sollya for fitting)."""
    config = load_activation_config(func_name)
    if not config:
        return False
    classification = config.get("classification", [])
    return "piecewise_linear" in classification


def is_piecewise_smooth_activation(func_name):
    """Check if activation declares 'piecewise_smooth' in its JSON classification."""
    config = load_activation_config(func_name)
    if not config:
        return False
    classification = config.get("classification", [])
    return "piecewise_smooth" in classification
