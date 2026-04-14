# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Declarative test stimuli generation (v2) for LLK tests.

This module replaces the scattered boolean-flag API of stimuli_generator.py
with a declarative :class:`StimuliSpec` that supports:

* Multiple distributions per operand — ``"uniform"``, ``"gaussian"``,
  ``"ramp"``, ``"log_uniform"``, ``"constant"``, ``"sequential"``, or any
  custom callable.
* Arbitrary value bounds (``low`` / ``high``) per operand with no
  hard-coded caps.
* Per-face overrides via ``StimuliSpec.face_specs``.
* Format-aware SFPU domain presets through
  ``StimuliSpec.for_op(op, data_format)``, backed by a registry that maps
  every :class:`~helpers.llk_params.MathOperation` to its valid input domain
  and format-specific overflow threshold.
* Independent, reproducible seed per operand via ``StimuliSpec.seed``.

Quick-start
-----------
>>> from helpers.stimuli_generator_v2 import StimuliSpec, generate_stimuli_v2
>>> from helpers.format_config import DataFormat
>>> from helpers.llk_params import MathOperation
>>>
>>> # Two independent operands with explicit distributions
>>> srcA, tile_cnt_A, srcB, tile_cnt_B = generate_stimuli_v2(
...     stimuli_format_A=DataFormat.Float16_b,
...     spec_A=StimuliSpec.uniform(low=-10.0, high=10.0, seed=42),
...     spec_B=StimuliSpec.log_uniform(low=0.01, high=1.0, seed=7),
... )
>>>
>>> # Format-aware preset for a specific SFPU op — returns OperandSpecs
>>> operands = StimuliSpec.for_op(MathOperation.Acosh, DataFormat.Float16_b)
>>> srcA, cnt_A, srcB, cnt_B = generate_stimuli_v2(
...     spec_A=operands.spec_A, spec_B=operands.spec_B
... )
>>>
>>> # Asymmetric binary op — divisor avoids zero
>>> operands = StimuliSpec.for_op(MathOperation.SfpuElwdiv, DataFormat.Float16_b)
>>> # operands.spec_A → uniform[-2, 2]   (dividend)
>>> # operands.spec_B → log_uniform[0.1, 10]  (divisor, no zero)
>>>
>>> # Custom callable distribution — receives the per-operand generator
>>> spec = StimuliSpec(
...     distribution=lambda size, dtype, gen: torch.ones(size, dtype=dtype) * 3.14,
...     seed=99,
... )
"""

import math
from dataclasses import dataclass
from dataclasses import replace as _dataclass_replace
from typing import Callable, Dict, List, Optional, Union

import torch

from .bfp_format_utils import bfp4b_to_float16b
from .format_config import DataFormat
from .llk_params import MathOperation, format_dict
from .stimuli_generator import (
    _clamp_mx_tensors,
    calculate_tile_and_face_counts,
    calculate_tile_and_face_counts_w_tile_dimensions,
)
from .tile_constants import (
    DEFAULT_TILE_C_DIM,
    DEFAULT_TILE_R_DIM,
    FACE_C_DIM,
    MAX_FACE_R_DIM,
    MAX_NUM_FACES,
    get_tile_params,
)

# ─────────────────────────────────────────────────────────────────────────────
# StimuliSpec
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class StimuliSpec:
    """
    Declarative per-operand specification for LLK test stimuli generation.

    Parameters
    ----------
    distribution : str or callable
        How values are sampled.  Supported string values:

        ``"uniform"``
            Uniform random in ``[low, high]``.  For integer formats this maps
            to ``torch.randint(int(low), int(high) + 1)``.

        ``"gaussian"``
            Normal distribution with *mean* and *std*.  For integer formats
            the result is rounded and clamped to the representable range.

        ``"ramp"``
            Linearly spaced values from *low* to *high* (``torch.linspace``).
            For integer formats a float32 linspace is rounded and clamped.

        ``"log_uniform"``
            ``exp(Uniform(log(low), log(high)))``.  Both *low* and *high* must
            be strictly positive.

        ``"constant"``
            Every element is set to *value*.  Ignores *low*, *high*, *seed*.

        ``"sequential"``
            Values ``1, 2, 3, …, size`` (mirrors the legacy
            ``sequential=True`` flag).  Ignores *low*, *high*, *seed*.

        ``"uniform_linspace"``
            Deterministic, evenly spaced values from *low* to *high*
            (``torch.linspace``).  Equivalent to ``"ramp"`` but operates as
            a **global sweep** across the full tensor (short-circuits the
            face loop), producing one smooth curve instead of a per-face
            sawtooth.  Useful for plotting function shapes.

        ``"gaussian_linspace"``
            Deterministic sweep through the Gaussian domain via the
            inverse CDF (percent-point function).  Produces ordered values
            concentrated around *mean* and spreading into the tails at
            *std* scale.  No randomness involved — the output is fully
            determined by *mean*, *std*, and the element count.

        ``"log_uniform_linspace"``
            Deterministic, logarithmically spaced values from *low* to
            *high*.  Both bounds must be strictly positive.  Equivalent to
            ``torch.logspace`` in natural-log base.

        callable
            ``fn(size: int, dtype: torch.dtype, generator: Optional[torch.Generator]) -> torch.Tensor``.
            The *generator* argument carries the per-operand RNG state (or
            ``None`` when no seed is set), enabling reproducible custom
            distributions.  The caller is responsible for producing a 1-D
            tensor of exactly *size* elements and returning it as the
            requested *dtype*.

    low : float
        Lower bound for ``"uniform"``, ``"ramp"``, ``"log_uniform"``,
        ``"uniform_linspace"``, and ``"log_uniform_linspace"``.
        Defaults to ``0.0``.
    high : float
        Upper bound for ``"uniform"``, ``"ramp"``, ``"log_uniform"``,
        ``"uniform_linspace"``, and ``"log_uniform_linspace"``.
        Defaults to ``1.0``.
    value : float
        Constant fill value (only used by ``"constant"``).  Defaults to
        ``1.0``.
    mean : float
        Mean for ``"gaussian"`` and ``"gaussian_linspace"``.  Defaults to
        ``0.0``.
    std : float
        Standard deviation for ``"gaussian"`` and ``"gaussian_linspace"``.
        Defaults to ``1.0``.
    seed : int, optional
        Seed for a per-spec ``torch.Generator``.  ``None`` uses the global
        torch RNG state.  When an external generator is supplied to
        :func:`generate_face_v2` the *seed* field is ignored so the caller
        controls state across faces.
    face_specs : list[StimuliSpec], optional
        Per-face overrides.  Face *i* is generated with ``face_specs[i]``
        when ``i < len(face_specs)``; any remaining faces fall back to this
        spec.
    """

    distribution: Union[str, Callable] = "uniform"
    low: float = 0.0
    high: float = 1.0
    value: float = 1.0
    mean: float = 0.0
    std: float = 1.0
    seed: Optional[int] = None
    face_specs: Optional[List["StimuliSpec"]] = None

    # ── convenience constructors ──────────────────────────────────────────────

    @classmethod
    def constant(cls, value: float = 1.0, **kwargs) -> "StimuliSpec":
        """All elements equal to *value*."""
        return cls(distribution="constant", value=value, **kwargs)

    @classmethod
    def sequential(cls, **kwargs) -> "StimuliSpec":
        """Sequential values ``1, 2, 3, …``"""
        return cls(distribution="sequential", **kwargs)

    @classmethod
    def uniform(cls, low: float = 0.0, high: float = 1.0, **kwargs) -> "StimuliSpec":
        """Uniform random in ``[low, high]``."""
        return cls(distribution="uniform", low=low, high=high, **kwargs)

    @classmethod
    def gaussian(cls, mean: float = 0.0, std: float = 1.0, **kwargs) -> "StimuliSpec":
        """Normal distribution with *mean* and *std*."""
        return cls(distribution="gaussian", mean=mean, std=std, **kwargs)

    @classmethod
    def ramp(cls, low: float = 0.0, high: float = 1.0, **kwargs) -> "StimuliSpec":
        """Linearly spaced from *low* to *high* (``torch.linspace``)."""
        return cls(distribution="ramp", low=low, high=high, **kwargs)

    @classmethod
    def log_uniform(
        cls, low: float = 1e-4, high: float = 1.0, **kwargs
    ) -> "StimuliSpec":
        """
        Log-uniform in ``[low, high]``.  Both bounds must be strictly positive.
        """
        if low <= 0 or high <= 0:
            raise ValueError(
                f"log_uniform requires strictly positive low and high, "
                f"got low={low}, high={high}"
            )
        return cls(distribution="log_uniform", low=low, high=high, **kwargs)

    @classmethod
    def uniform_linspace(
        cls, low: float = 0.0, high: float = 1.0, **kwargs
    ) -> "StimuliSpec":
        """Deterministic evenly spaced sweep from *low* to *high*."""
        return cls(distribution="uniform_linspace", low=low, high=high, **kwargs)

    @classmethod
    def gaussian_linspace(
        cls, mean: float = 0.0, std: float = 1.0, **kwargs
    ) -> "StimuliSpec":
        """Deterministic sweep through the Gaussian domain (inverse CDF)."""
        return cls(distribution="gaussian_linspace", mean=mean, std=std, **kwargs)

    @classmethod
    def log_uniform_linspace(
        cls, low: float = 1e-4, high: float = 1.0, **kwargs
    ) -> "StimuliSpec":
        """Deterministic log-spaced sweep from *low* to *high* (both > 0)."""
        if low <= 0 or high <= 0:
            raise ValueError(
                f"log_uniform_linspace requires strictly positive low and high, "
                f"got low={low}, high={high}"
            )
        return cls(distribution="log_uniform_linspace", low=low, high=high, **kwargs)

    @classmethod
    def for_op(
        cls,
        op: MathOperation,
        data_format: DataFormat = DataFormat.Float16_b,
    ) -> "OperandSpecs":
        """
        Return :class:`OperandSpecs` with safe input domains for *op* and
        *data_format*.

        The registry maps each :class:`~helpers.llk_params.MathOperation` to
        either a fixed :class:`OperandSpecs` (format-independent domain) or a
        callable ``(DataFormat) -> OperandSpecs`` for operations whose safe
        range depends on the format's overflow threshold (e.g. ``Exp``
        overflows far sooner in Float16 than in Float16_b).

        For binary operations where the two operands have different valid
        domains (e.g. ``SfpuElwdiv``, ``SfpuXlogy``), the returned
        :class:`OperandSpecs` carries distinct ``spec_A`` and ``spec_B``.

        Parameters
        ----------
        op : MathOperation
            Target math operation.
        data_format : DataFormat
            Input data format; drives format-specific threshold selection for
            operations like :attr:`~helpers.llk_params.MathOperation.Exp`,
            :attr:`~helpers.llk_params.MathOperation.Exp2`, and
            :attr:`~helpers.llk_params.MathOperation.Square`.

        Returns
        -------
        OperandSpecs

        Raises
        ------
        KeyError
            If *op* is not in the registry.  This is intentional: silently
            falling back to a default spec can mask missing coverage when a
            new :class:`~helpers.llk_params.MathOperation` is added without
            a corresponding registry entry.
        """
        entry = _OP_DOMAIN_REGISTRY.get(op)
        if entry is None:
            registered = sorted(o.name for o in _OP_DOMAIN_REGISTRY)
            raise KeyError(
                f"MathOperation.{op.name} has no entry in the stimuli domain "
                f"registry.  Add an OperandSpecs entry to _OP_DOMAIN_REGISTRY.\n"
                f"Currently registered ({len(registered)}): {registered}"
            )
        if callable(entry):
            return entry(data_format)
        return entry


# ─────────────────────────────────────────────────────────────────────────────
# OperandSpecs
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class OperandSpecs:
    """
    Per-operand input domain specifications returned by
    :meth:`StimuliSpec.for_op`.

    Encodes the correct input domain for *each* operand of a math operation,
    including operations where operands must live in different domains:

    * ``SfpuElwdiv`` — divisor (srcB) must avoid zero
    * ``SfpuXlogy`` — srcA (x) requires x ≥ 0, srcB (y) requires y > 0

    Parameters
    ----------
    spec_A : StimuliSpec
        Specification for the first operand (srcA).
    spec_B : StimuliSpec, optional
        Specification for the second operand (srcB).  Defaults to a copy of
        *spec_A* when ``None``, so unary operations need only provide one spec.
    """

    spec_A: StimuliSpec
    spec_B: Optional[StimuliSpec] = None

    def __post_init__(self) -> None:
        if self.spec_B is None:
            # Copy the top-level spec so spec_A and spec_B are distinct
            # StimuliSpec instances. Nested fields (e.g. face_specs) are
            # still shared because this is a shallow copy.
            self.spec_B = _dataclass_replace(self.spec_A)


# ─────────────────────────────────────────────────────────────────────────────
# Format-specific domain builders (referenced by _OP_DOMAIN_REGISTRY)
# ─────────────────────────────────────────────────────────────────────────────


def _exp_spec(fmt: DataFormat) -> OperandSpecs:
    """
    Safe input range for ``exp(x)`` so the output does not overflow.

    Conservative overflow thresholds per format family:

    * MxFp8P    (max ≈ 448)    → exp(5)  ≈ 148   : [-5,  5]
    * Float16   (max ≈ 65504)  → exp(10) ≈ 22026  : [-10, 10]
    * MxFp8R    (max ≈ 57344)  → exp(10) ≈ 22026  : [-10, 10]
    * Float16_b (max ≈ 3.4e38) → exp(80) ≪ max    : [-80, 80]
    * Float32 / Tf32 / BFP: same as Float16_b
    """
    if fmt == DataFormat.MxFp8P:
        spec = StimuliSpec(distribution="uniform", low=-5.0, high=5.0)
    elif fmt in (DataFormat.Float16, DataFormat.MxFp8R):
        spec = StimuliSpec(distribution="uniform", low=-10.0, high=10.0)
    else:
        spec = StimuliSpec(distribution="uniform", low=-80.0, high=80.0)
    return OperandSpecs(spec_A=spec)


def _exp2_spec(fmt: DataFormat) -> OperandSpecs:
    """
    Safe input range for ``exp2(x) = 2^x``.

    * MxFp8P  : 2^7  = 128   < 448       : [-7,   7]
    * Float16 : 2^14 = 16384 < 65504     : [-14, 14]
    * MxFp8R  : 2^14 = 16384 < 57344     : [-14, 14]
    * Float16_b / Float32 / BFP: 2^100 ≈ 1.27e30 < 3.4e38 : [-100, 100]
    """
    if fmt == DataFormat.MxFp8P:
        spec = StimuliSpec(distribution="uniform", low=-7.0, high=7.0)
    elif fmt in (DataFormat.Float16, DataFormat.MxFp8R):
        spec = StimuliSpec(distribution="uniform", low=-14.0, high=14.0)
    else:
        spec = StimuliSpec(distribution="uniform", low=-100.0, high=100.0)
    return OperandSpecs(spec_A=spec)


def _square_spec(fmt: DataFormat) -> OperandSpecs:
    """
    Safe input range for ``square(x) = x^2`` so the output does not overflow.

    * MxFp8P  : sqrt(448)  ≈ 21  → [-20,  20]
    * Float16 : sqrt(65504) ≈ 256 → [-200, 200] (conservative)
    * MxFp8R  : sqrt(57344) ≈ 239 → [-200, 200] (conservative)
    * Float16_b / Float32 / BFP  → [-1000, 1000] (practical limit)
    """
    if fmt == DataFormat.MxFp8P:
        spec = StimuliSpec(distribution="uniform", low=-20.0, high=20.0)
    elif fmt in (DataFormat.Float16, DataFormat.MxFp8R):
        spec = StimuliSpec(distribution="uniform", low=-200.0, high=200.0)
    else:
        spec = StimuliSpec(distribution="uniform", low=-1000.0, high=1000.0)
    return OperandSpecs(spec_A=spec)


# ─────────────────────────────────────────────────────────────────────────────
# SFPU / FPU operation domain registry
# ─────────────────────────────────────────────────────────────────────────────
#
# Maps every MathOperation to either:
#   OperandSpecs          — format-independent safe input domains
#   callable              — (DataFormat) -> OperandSpecs for format-sensitive ops
#
# For unary operations spec_B is omitted (defaults to a copy of spec_A).
# For binary operations where operands require different domains the entry
# uses explicit spec_A and spec_B.
#
# Rationale for each entry is documented inline.

_OP_DOMAIN_REGISTRY: Dict[
    MathOperation,
    Union[OperandSpecs, Callable[[DataFormat], OperandSpecs]],
] = {
    # ── SFPU unary ────────────────────────────────────────────────────────────
    # abs: all reals; include negative branch
    MathOperation.Abs: OperandSpecs(
        spec_A=StimuliSpec(distribution="uniform", low=-10.0, high=10.0)
    ),
    # acosh: domain x >= 1
    MathOperation.Acosh: OperandSpecs(
        spec_A=StimuliSpec(distribution="uniform", low=1.0, high=10.0)
    ),
    # asinh: all reals
    MathOperation.Asinh: OperandSpecs(
        spec_A=StimuliSpec(distribution="uniform", low=-10.0, high=10.0)
    ),
    # atanh: domain |x| < 1; stay away from ±1 to avoid ±inf
    MathOperation.Atanh: OperandSpecs(
        spec_A=StimuliSpec(distribution="uniform", low=-0.95, high=0.95)
    ),
    # celu: exercises both the exponential branch (x < 0) and linear (x >= 0)
    MathOperation.Celu: OperandSpecs(
        spec_A=StimuliSpec(distribution="uniform", low=-5.0, high=5.0)
    ),
    # cos: cover the full unit circle
    MathOperation.Cos: OperandSpecs(
        spec_A=StimuliSpec(distribution="uniform", low=-math.pi, high=math.pi)
    ),
    # elu: exercises the exponential branch (x < 0)
    MathOperation.Elu: OperandSpecs(
        spec_A=StimuliSpec(distribution="uniform", low=-5.0, high=5.0)
    ),
    # exp: format-specific overflow threshold
    MathOperation.Exp: _exp_spec,
    # exp2: format-specific overflow threshold
    MathOperation.Exp2: _exp2_spec,
    # fill: the hardware ignores the input value; any range is fine
    MathOperation.Fill: OperandSpecs(
        spec_A=StimuliSpec(distribution="uniform", low=0.0, high=1.0)
    ),
    # gelu: Gaussian activation — gaussian distribution naturally exercises tails
    MathOperation.Gelu: OperandSpecs(
        spec_A=StimuliSpec(distribution="gaussian", mean=0.0, std=3.0)
    ),
    # hardsigmoid: linear region between -3 and 3, clipped outside
    MathOperation.Hardsigmoid: OperandSpecs(
        spec_A=StimuliSpec(distribution="uniform", low=-4.0, high=4.0)
    ),
    # log: domain x > 0; log-uniform spans several decades
    MathOperation.Log: OperandSpecs(
        spec_A=StimuliSpec(distribution="log_uniform", low=1e-4, high=1e3)
    ),
    # log1p: domain x > -1; log1p(x) = log(1 + x)
    MathOperation.Log1p: OperandSpecs(
        spec_A=StimuliSpec(distribution="uniform", low=-0.99, high=10.0)
    ),
    # neg: all reals
    MathOperation.Neg: OperandSpecs(
        spec_A=StimuliSpec(distribution="uniform", low=-10.0, high=10.0)
    ),
    # reciprocal: domain x != 0; log-uniform avoids zero and covers decades.
    # Only strictly positive values are generated here so that negative-
    # reciprocal paths are intentionally not exercised by default.
    # To test negative inputs, supply an explicit spec such as:
    #   StimuliSpec.uniform(low=-100.0, high=-0.1)
    MathOperation.Reciprocal: OperandSpecs(
        spec_A=StimuliSpec(distribution="log_uniform", low=0.1, high=100.0)
    ),
    # relu / relu_max / relu_min / threshold: include negatives (zero branch)
    MathOperation.Relu: OperandSpecs(
        spec_A=StimuliSpec(distribution="uniform", low=-5.0, high=5.0)
    ),
    MathOperation.ReluMax: OperandSpecs(
        spec_A=StimuliSpec(distribution="uniform", low=-5.0, high=5.0)
    ),
    MathOperation.ReluMin: OperandSpecs(
        spec_A=StimuliSpec(distribution="uniform", low=-5.0, high=5.0)
    ),
    MathOperation.Threshold: OperandSpecs(
        spec_A=StimuliSpec(distribution="uniform", low=-5.0, high=5.0)
    ),
    # rsqrt: domain x > 0; log-uniform covers a wide positive range
    MathOperation.Rsqrt: OperandSpecs(
        spec_A=StimuliSpec(distribution="log_uniform", low=1e-4, high=100.0)
    ),
    # sigmoid: cover both saturation regions
    MathOperation.Sigmoid: OperandSpecs(
        spec_A=StimuliSpec(distribution="uniform", low=-8.0, high=8.0)
    ),
    # silu: silu(x) = x * sigmoid(x); cover saturation + linear regions
    MathOperation.Silu: OperandSpecs(
        spec_A=StimuliSpec(distribution="uniform", low=-5.0, high=5.0)
    ),
    # sin: cover the full unit circle
    MathOperation.Sin: OperandSpecs(
        spec_A=StimuliSpec(distribution="uniform", low=-math.pi, high=math.pi)
    ),
    # sqrt: domain x >= 0
    MathOperation.Sqrt: OperandSpecs(
        spec_A=StimuliSpec(distribution="uniform", low=0.0, high=100.0)
    ),
    # square: format-specific overflow threshold
    MathOperation.Square: _square_spec,
    # tanh: cover saturation regions (saturates near ±1 for |x| > ~3)
    MathOperation.Tanh: OperandSpecs(
        spec_A=StimuliSpec(distribution="uniform", low=-5.0, high=5.0)
    ),
    # topk family: operation sorts/merges; any values are valid
    MathOperation.TopKLocalSort: OperandSpecs(
        spec_A=StimuliSpec(distribution="uniform", low=-10.0, high=10.0)
    ),
    MathOperation.TopKMerge: OperandSpecs(
        spec_A=StimuliSpec(distribution="uniform", low=-10.0, high=10.0)
    ),
    MathOperation.TopKRebuild: OperandSpecs(
        spec_A=StimuliSpec(distribution="uniform", low=-10.0, high=10.0)
    ),
    # ── FPU binary ────────────────────────────────────────────────────────────
    MathOperation.Elwadd: OperandSpecs(
        spec_A=StimuliSpec(distribution="uniform", low=-1.0, high=1.0)
    ),
    MathOperation.Elwmul: OperandSpecs(
        spec_A=StimuliSpec(distribution="uniform", low=-1.0, high=1.0)
    ),
    MathOperation.Elwsub: OperandSpecs(
        spec_A=StimuliSpec(distribution="uniform", low=-1.0, high=1.0)
    ),
    # ── SFPU binary ───────────────────────────────────────────────────────────
    MathOperation.SfpuElwadd: OperandSpecs(
        spec_A=StimuliSpec(distribution="uniform", low=-1.0, high=1.0)
    ),
    MathOperation.SfpuElwmul: OperandSpecs(
        spec_A=StimuliSpec(distribution="uniform", low=-1.0, high=1.0)
    ),
    MathOperation.SfpuElwsub: OperandSpecs(
        spec_A=StimuliSpec(distribution="uniform", low=-1.0, high=1.0)
    ),
    # div: srcA is the dividend (any value); srcB is the divisor.
    # The divisor uses log-uniform over strictly positive values so that
    # divide-by-zero and near-zero instability are avoided.
    # This means negative-divisor paths are intentionally not exercised here.
    # To test negative-divisor behaviour, supply an explicit spec_B such as:
    #   StimuliSpec.uniform(low=-10.0, high=-0.1)
    MathOperation.SfpuElwdiv: OperandSpecs(
        spec_A=StimuliSpec(distribution="uniform", low=-2.0, high=2.0),
        spec_B=StimuliSpec(distribution="log_uniform", low=0.1, high=10.0),
    ),
    MathOperation.SfpuElwrsub: OperandSpecs(
        spec_A=StimuliSpec(distribution="uniform", low=-1.0, high=1.0)
    ),
    # pow: srcA is the base (must be non-negative for non-integer exponents);
    # srcB is the exponent (non-negative to keep output finite)
    MathOperation.SfpuElwpow: OperandSpecs(
        spec_A=StimuliSpec(distribution="uniform", low=0.0, high=5.0),
        spec_B=StimuliSpec(distribution="uniform", low=0.0, high=5.0),
    ),
    # xlogy: computes x * log(y) element-wise
    # srcA (x): x >= 0 so xlogy(0, y) = 0 is well-defined
    # srcB (y): y > 0 so log(y) is finite; log-uniform spans several decades
    MathOperation.SfpuXlogy: OperandSpecs(
        spec_A=StimuliSpec(distribution="uniform", low=0.0, high=10.0),
        spec_B=StimuliSpec(distribution="log_uniform", low=1e-4, high=10.0),
    ),
    MathOperation.SfpuAddTopRow: OperandSpecs(
        spec_A=StimuliSpec(distribution="uniform", low=-1.0, high=1.0)
    ),
    # shift ops: operate on integer bit patterns; both operands in [0, 255]
    MathOperation.SfpuElwLeftShift: OperandSpecs(
        spec_A=StimuliSpec(distribution="uniform", low=0.0, high=255.0)
    ),
    MathOperation.SfpuElwLogicalRightShift: OperandSpecs(
        spec_A=StimuliSpec(distribution="uniform", low=0.0, high=255.0)
    ),
    MathOperation.SfpuElwRightShift: OperandSpecs(
        spec_A=StimuliSpec(distribution="uniform", low=0.0, high=255.0)
    ),
    # ── SFPU ternary ──────────────────────────────────────────────────────────
    # where(cond, a, b): selects a when cond != 0, else b
    # spec_A / spec_B cover the two value operands; the condition operand is
    # generated separately by the caller (not modelled here).
    # NOTE: TTNNWhere is an alias for SfpuWhere (same enum value) and is
    # implicitly covered by this entry.
    MathOperation.SfpuWhere: OperandSpecs(
        spec_A=StimuliSpec(distribution="uniform", low=-1.0, high=1.0),
        spec_B=StimuliSpec(distribution="uniform", low=-1.0, high=1.0),
    ),
    # ── Reduce ────────────────────────────────────────────────────────────────
    MathOperation.ReduceColumn: OperandSpecs(
        spec_A=StimuliSpec(distribution="uniform", low=-1.0, high=1.0)
    ),
    MathOperation.ReduceRow: OperandSpecs(
        spec_A=StimuliSpec(distribution="uniform", low=-1.0, high=1.0)
    ),
    MathOperation.ReduceScalar: OperandSpecs(
        spec_A=StimuliSpec(distribution="uniform", low=-1.0, high=1.0)
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────


# Set of distribution names that produce a deterministic global sweep and
# should short-circuit the face loop in _generate_source_tensor_v2.
_LINSPACE_DISTRIBUTIONS = frozenset(
    {
        "uniform_linspace",
        "gaussian_linspace",
        "log_uniform_linspace",
    }
)

# Small epsilon to keep the Gaussian inverse-CDF quantiles away from 0 and 1
# (which would map to ±inf).
_GAUSSIAN_LINSPACE_EPS = 1e-6


def _generate_linspace_tensor(
    spec: StimuliSpec, size: int, dtype: torch.dtype
) -> torch.Tensor:
    """
    Generate a deterministic linspace-style tensor for float formats.

    Works for ``"uniform_linspace"``, ``"gaussian_linspace"``, and
    ``"log_uniform_linspace"``.  Always computes in float32 and casts to
    *dtype* at the end to avoid precision issues in reduced formats.
    """
    dist = spec.distribution

    if dist == "uniform_linspace":
        return torch.linspace(spec.low, spec.high, size, dtype=torch.float32).to(
            dtype=dtype
        )

    if dist == "gaussian_linspace":
        # Inverse CDF of the standard normal: Φ⁻¹(p) = √2 · erfinv(2p − 1)
        p = torch.linspace(_GAUSSIAN_LINSPACE_EPS, 1.0 - _GAUSSIAN_LINSPACE_EPS, size)
        values = spec.mean + spec.std * math.sqrt(2.0) * torch.erfinv(2.0 * p - 1.0)
        return values.to(dtype=dtype)

    if dist == "log_uniform_linspace":
        if spec.low <= 0 or spec.high <= 0:
            raise ValueError(
                f"log_uniform_linspace requires strictly positive low and high, "
                f"got low={spec.low}, high={spec.high}"
            )
        log_low = math.log(spec.low)
        log_high = math.log(spec.high)
        return torch.exp(
            torch.linspace(log_low, log_high, size, dtype=torch.float32)
        ).to(dtype=dtype)

    raise ValueError(
        f"_generate_linspace_tensor called with unsupported distribution {dist!r}"
    )


def _get_dtype_for_format(stimuli_format: DataFormat) -> torch.dtype:
    """Return the torch dtype to use for *stimuli_format*."""
    if stimuli_format in (DataFormat.Bfp8_b, DataFormat.Bfp4_b):
        return torch.bfloat16
    if stimuli_format == DataFormat.Tf32:
        return torch.float32
    return format_dict[stimuli_format]


def _get_integer_bounds(stimuli_format: DataFormat) -> tuple[int, int]:
    """Return the valid integer range ``(min, max)`` inclusive for *stimuli_format*."""
    bounds: Dict[DataFormat, tuple[int, int]] = {
        DataFormat.Int8: (torch.iinfo(torch.int8).min + 1, torch.iinfo(torch.int8).max),
        DataFormat.UInt8: (0, 255),
        DataFormat.Int16: (
            torch.iinfo(torch.int16).min + 1,
            torch.iinfo(torch.int16).max,
        ),
        DataFormat.UInt16: (0, 65535),
        DataFormat.Int32: (
            torch.iinfo(torch.int32).min + 1,
            torch.iinfo(torch.int32).max,
        ),
        DataFormat.UInt32: (0, 2**32 - 1),
    }
    return bounds.get(
        stimuli_format, (torch.iinfo(torch.int8).min + 1, torch.iinfo(torch.int8).max)
    )


def _generate_integer_face(
    spec: StimuliSpec,
    stimuli_format: DataFormat,
    size: int,
    dtype: torch.dtype,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    """
    Generate one face of integer-format data according to *spec*.

    User-specified ``low`` / ``high`` are clamped to the representable range
    of *stimuli_format* so that the result is always a valid value for the
    hardware type.  Distributions that produce floats (``"gaussian"``,
    ``"log_uniform"``, ``"ramp"``) are rounded and clamped before casting.

    Collapsed ranges
    ~~~~~~~~~~~~~~~~
    If clamping the user bounds to the format's representable range makes
    ``high <= low``, a constant tensor filled with ``low`` is returned rather
    than silently adjusting the bounds.  This makes the degenerate condition
    observable: the caller gets a constant tensor and can decide whether the
    spec is misconfigured.

    ``log_uniform`` for integers
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Both ``low`` and ``high`` must be strictly positive (same requirement as
    the float path).  The raw log-uniform samples are rounded and clamped.
    Using ``log_uniform`` with signed integer formats is allowed as long as
    the *spec* bounds are positive (the result is always positive).
    """
    int_min, int_max = _get_integer_bounds(stimuli_format)
    # Clamp both bounds to the representable range before any comparison
    low = min(max(int(math.floor(spec.low)), int_min), int_max)
    high = min(max(int(math.ceil(spec.high)), int_min), int_max)

    # Degenerate range: spec bounds collapsed after clamping — return constant
    if high <= low:
        return torch.full((size,), low, dtype=dtype)

    distribution = spec.distribution

    if distribution == "uniform":
        return torch.randint(
            low=low, high=high + 1, size=(size,), dtype=dtype, generator=generator
        )

    if distribution == "ramp":
        return (
            torch.linspace(spec.low, spec.high, size, dtype=torch.float32)
            .round()
            .clamp(int_min, int_max)
            .to(dtype=dtype)
        )

    if distribution == "gaussian":
        raw = (
            torch.randn(size, dtype=torch.float32, generator=generator) * spec.std
            + spec.mean
        )
        return raw.round().clamp(int_min, int_max).to(dtype=dtype)

    if distribution == "log_uniform":
        if spec.low <= 0 or spec.high <= 0:
            raise ValueError(
                f"log_uniform requires strictly positive low and high bounds; "
                f"got low={spec.low}, high={spec.high}.  "
                f"For integer formats use a positive range such as "
                f"StimuliSpec.log_uniform(low=1, high=1000)."
            )
        log_low = math.log(spec.low)
        log_high = math.log(spec.high)
        raw_u = torch.rand(size, dtype=torch.float32, generator=generator)
        raw = torch.exp(raw_u * (log_high - log_low) + log_low)
        return raw.round().clamp(int_min, int_max).to(dtype=dtype)

    if distribution in _LINSPACE_DISTRIBUTIONS:
        raw = _generate_linspace_tensor(spec, size, torch.float32)
        return raw.round().clamp(int_min, int_max).to(dtype=dtype)

    raise ValueError(
        f"Unknown distribution {distribution!r} for integer format. "
        f"Expected one of 'uniform', 'gaussian', 'ramp', 'log_uniform', "
        f"'uniform_linspace', 'gaussian_linspace', 'log_uniform_linspace', "
        f"'constant', 'sequential', or a callable."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public: single-face generator
# ─────────────────────────────────────────────────────────────────────────────


def generate_face_v2(
    spec: StimuliSpec,
    stimuli_format: DataFormat = DataFormat.Float16_b,
    face_r_dim: int = MAX_FACE_R_DIM,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Generate a single face tensor of shape ``(face_r_dim * 16,)`` using *spec*.

    Parameters
    ----------
    spec : StimuliSpec
        Generation specification (distribution, bounds, seed, …).
    stimuli_format : DataFormat
        Target hardware format; determines the torch dtype and any
        format-specific constraints.
    face_r_dim : int
        Number of rows per face (1–16).  Defaults to ``16``.
    generator : torch.Generator, optional
        External RNG state to use.  When supplied, ``spec.seed`` is
        **ignored**, allowing the caller to share a single generator across
        multiple face calls for diverse-but-reproducible output.  When
        *generator* is ``None`` and ``spec.seed`` is set, a fresh generator
        is seeded from ``spec.seed``, giving identical results on every
        standalone call.

    Returns
    -------
    torch.Tensor
        1-D tensor with ``face_r_dim * 16`` elements.
    """
    size = face_r_dim * FACE_C_DIM
    dtype = _get_dtype_for_format(stimuli_format)

    # Create a local generator only when no external one was supplied
    if generator is None and spec.seed is not None:
        generator = torch.Generator()
        generator.manual_seed(spec.seed)

    distribution = spec.distribution

    # ── custom callable ───────────────────────────────────────────────────────
    if callable(distribution):
        result = distribution(size, dtype, generator)
        if not isinstance(result, torch.Tensor):
            raise TypeError(
                f"Custom distribution callable must return a torch.Tensor, "
                f"got {type(result).__name__}"
            )
        if result.ndim != 1:
            raise ValueError(
                f"Custom distribution callable must return a 1-D tensor; "
                f"got {result.ndim}-D tensor with shape {tuple(result.shape)}"
            )
        if len(result) != size:
            raise ValueError(
                f"Custom distribution callable returned {len(result)} elements "
                f"but {size} were expected "
                f"({face_r_dim} rows × {FACE_C_DIM} cols)"
            )
        return result.to(dtype=dtype)

    # ── format-independent distributions ─────────────────────────────────────
    if distribution == "constant":
        if stimuli_format.is_integer():
            int_min, int_max = _get_integer_bounds(stimuli_format)
            clamped = max(int_min, min(int(spec.value), int_max))
            return torch.full((size,), clamped, dtype=dtype)
        return torch.full((size,), spec.value, dtype=dtype)

    if distribution == "sequential":
        if stimuli_format.is_integer():
            int_min, int_max = _get_integer_bounds(stimuli_format)
            result = torch.arange(1, size + 1, dtype=torch.int64)
            result = result.clamp(min=int_min, max=int_max)
            return result.to(dtype=dtype)
        return torch.arange(1, size + 1, dtype=dtype)

    # ── integer formats ───────────────────────────────────────────────────────
    if stimuli_format.is_integer():
        return _generate_integer_face(spec, stimuli_format, size, dtype, generator)

    # ── float / BFP / MX formats ─────────────────────────────────────────────
    if distribution == "uniform":
        raw = torch.rand(size, dtype=dtype, generator=generator)
        return raw * (spec.high - spec.low) + spec.low

    if distribution == "gaussian":
        # NOTE: Gaussian is unbounded — extreme outliers in reduced-precision
        # formats (bfloat16, float16) may produce inf/NaN.  Set *std*
        # conservatively relative to the format's representable range.
        raw = torch.randn(size, dtype=dtype, generator=generator)
        return raw * spec.std + spec.mean

    if distribution == "ramp":
        return torch.linspace(spec.low, spec.high, size, dtype=dtype)

    if distribution == "log_uniform":
        if spec.low <= 0 or spec.high <= 0:
            raise ValueError(
                f"log_uniform requires strictly positive low and high, "
                f"got low={spec.low}, high={spec.high}"
            )
        log_low = math.log(spec.low)
        log_high = math.log(spec.high)
        raw = torch.rand(size, dtype=dtype, generator=generator)
        return torch.exp(raw * (log_high - log_low) + log_low).to(dtype=dtype)

    if distribution in _LINSPACE_DISTRIBUTIONS:
        return _generate_linspace_tensor(spec, size, dtype)

    raise ValueError(
        f"Unknown distribution {distribution!r}. "
        f"Expected one of 'uniform', 'gaussian', 'ramp', 'log_uniform', "
        f"'uniform_linspace', 'gaussian_linspace', 'log_uniform_linspace', "
        f"'constant', 'sequential', or a callable."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Private: full operand tensor
# ─────────────────────────────────────────────────────────────────────────────


def _generate_source_tensor_v2(
    stimuli_format: DataFormat,
    num_elements: int,
    face_r_dim: int,
    spec: StimuliSpec,
) -> torch.Tensor:
    """
    Generate a source tensor of *num_elements* using *spec*.

    A single ``torch.Generator`` is created once from ``spec.seed`` (when set)
    and reused across all faces, so the RNG state advances naturally and each
    face receives different values while remaining reproducible.

    Per-face overrides in ``spec.face_specs`` are applied: face *i* is
    generated with ``spec.face_specs[i]`` when that entry exists, inheriting
    the shared generator so random state is never reset mid-operand.

    BFP4_b post-quantisation (``bfp4b_to_float16b``) is applied after all
    faces are concatenated to simulate the hardware pack/unpack round-trip,
    matching the behaviour of the original generator.

    Notes
    -----
    The ``"sequential"`` and linspace distributions (``"uniform_linspace"``,
    ``"gaussian_linspace"``, ``"log_uniform_linspace"``) short-circuit the
    face loop and produce a single global sweep across *num_elements*.
    """
    dtype = _get_dtype_for_format(stimuli_format)

    if spec.distribution == "sequential":
        if stimuli_format.is_integer():
            int_min, int_max = _get_integer_bounds(stimuli_format)
            tensor = torch.arange(1, num_elements + 1, dtype=torch.int64)
            tensor = tensor.clamp(min=int_min, max=int_max).to(dtype=dtype)
        else:
            tensor = torch.arange(1, num_elements + 1, dtype=dtype)

        if stimuli_format == DataFormat.Bfp4_b:
            tensor = bfp4b_to_float16b(tensor)
        return tensor

    # ── linspace distributions: global sweep (no face loop) ──────────────
    if spec.distribution in _LINSPACE_DISTRIBUTIONS:
        if stimuli_format.is_integer():
            int_min, int_max = _get_integer_bounds(stimuli_format)
            tensor = _generate_linspace_tensor(spec, num_elements, torch.float32)
            tensor = tensor.round().clamp(int_min, int_max).to(dtype=dtype)
        else:
            tensor = _generate_linspace_tensor(spec, num_elements, dtype)

        if stimuli_format == DataFormat.Bfp4_b:
            tensor = bfp4b_to_float16b(tensor)
        return tensor

    elements_per_face = face_r_dim * FACE_C_DIM
    faces_needed = math.ceil(num_elements / elements_per_face)

    gen: Optional[torch.Generator] = None
    if spec.seed is not None:
        gen = torch.Generator()
        gen.manual_seed(spec.seed)

    face_tensors: list[torch.Tensor] = []
    for face_idx in range(faces_needed):
        face_spec = spec
        if spec.face_specs and face_idx < len(spec.face_specs):
            face_spec = spec.face_specs[face_idx]

        face_tensor = generate_face_v2(
            spec=face_spec,
            stimuli_format=stimuli_format,
            face_r_dim=face_r_dim,
            generator=gen,
        )
        face_tensors.append(face_tensor)

    tensor = torch.cat(face_tensors)[:num_elements]

    # Simulate the hardware pack/unpack round-trip for BFP4_b so that stimuli
    # already reflect the precision loss introduced by the format.
    if stimuli_format == DataFormat.Bfp4_b:
        tensor = bfp4b_to_float16b(tensor)

    return tensor


# ─────────────────────────────────────────────────────────────────────────────
# Public: top-level stimuli generator
# ─────────────────────────────────────────────────────────────────────────────


def generate_stimuli_v2(
    stimuli_format_A: DataFormat = DataFormat.Float16_b,
    input_dimensions_A: Optional[list] = None,
    stimuli_format_B: DataFormat = DataFormat.Float16_b,
    input_dimensions_B: Optional[list] = None,
    spec_A: Optional[StimuliSpec] = None,
    spec_B: Optional[StimuliSpec] = None,
    tile_dimensions: Optional[list] = None,
    face_r_dim: int = MAX_FACE_R_DIM,
    num_faces: int = MAX_NUM_FACES,
    output_format: Optional[DataFormat] = None,
) -> tuple[torch.Tensor, int, torch.Tensor, int]:
    """
    Generate test stimuli for two operands.

    This is the single v2 replacement for both
    :func:`~stimuli_generator.generate_stimuli` and
    :func:`~stimuli_generator.generate_stimuli_w_tile_dimensions`.

    When *tile_dimensions* is supplied the function operates in **dense mode**:
    tile counts and face layout are derived entirely from *tile_dimensions*,
    filling every element of each tile.  When *tile_dimensions* is ``None``
    the function uses the standard 32×32 tile path (controlled by *face_r_dim*
    and *num_faces* for the partial-face case).

    The two operands are generated independently: ``spec_A`` controls srcA and
    ``spec_B`` controls srcB.  Their seeds (if set) are completely independent,
    so the operands are uncorrelated even when the same distribution is used.

    Parameters
    ----------
    stimuli_format_A, stimuli_format_B : DataFormat
        Hardware data format for each operand.
    input_dimensions_A, input_dimensions_B : list of [int, int]
        ``[height, width]`` in elements.  Defaults to ``[32, 32]``.
    spec_A, spec_B : StimuliSpec, optional
        Generation spec for each operand.  Defaults to
        ``StimuliSpec(distribution="uniform", low=0.0, high=1.0)``.
    tile_dimensions : list of [int, int], optional
        ``[rows, cols]`` for the tile size (e.g. ``[8, 32]``).  When provided,
        enables dense mode: tile counts are computed from these dimensions and
        *face_r_dim* / *num_faces* are derived automatically, ignoring the
        values passed for those parameters.  When ``None`` (default), the
        standard 32×32 tile path is used.
    face_r_dim : int
        Rows per face (1–16).  Used only when *tile_dimensions* is ``None``.
        Defaults to ``16``.
    num_faces : int
        Faces per tile for the partial-face case.  Used only when
        *tile_dimensions* is ``None``.  Defaults to ``4``.
    output_format : DataFormat, optional
        When set, output values are clamped to prevent overflowing the output
        format for mixed MX-format pairs.

    Returns
    -------
    tuple
        ``(srcA_tensor, tile_cnt_A, srcB_tensor, tile_cnt_B)``

    Examples
    --------
    >>> # Standard 32×32 tile path
    >>> srcA, cnt_A, srcB, cnt_B = generate_stimuli_v2(
    ...     stimuli_format_A=DataFormat.Float16_b,
    ...     spec_A=StimuliSpec.gaussian(mean=0.0, std=1.0, seed=0),
    ...     spec_B=StimuliSpec.uniform(low=0.01, high=2.0, seed=1),
    ... )
    >>>
    >>> # Dense mode with custom tile dimensions
    >>> srcA, cnt_A, srcB, cnt_B = generate_stimuli_v2(
    ...     stimuli_format_A=DataFormat.Float16_b,
    ...     input_dimensions_A=[64, 64],
    ...     spec_A=StimuliSpec.uniform(low=-1.0, high=1.0),
    ...     tile_dimensions=[8, 32],
    ... )
    >>>
    >>> # Domain-safe preset for a specific SFPU op
    >>> operands = StimuliSpec.for_op(MathOperation.Acosh, DataFormat.Float16_b)
    >>> srcA, cnt_A, srcB, cnt_B = generate_stimuli_v2(
    ...     spec_A=operands.spec_A, spec_B=operands.spec_B
    ... )
    >>>
    >>> # Per-face overrides: alternating positive / negative faces
    >>> spec = StimuliSpec(
    ...     distribution="uniform",
    ...     low=0.1,
    ...     high=1.0,
    ...     face_specs=[
    ...         StimuliSpec.uniform(low=0.1, high=1.0),
    ...         StimuliSpec.uniform(low=-1.0, high=-0.1),
    ...         StimuliSpec.uniform(low=0.1, high=1.0),
    ...         StimuliSpec.uniform(low=-1.0, high=-0.1),
    ...     ],
    ... )
    """
    if input_dimensions_A is None:
        input_dimensions_A = [DEFAULT_TILE_R_DIM, DEFAULT_TILE_C_DIM]
    if input_dimensions_B is None:
        input_dimensions_B = [DEFAULT_TILE_R_DIM, DEFAULT_TILE_C_DIM]
    if spec_A is None:
        spec_A = StimuliSpec()
    if spec_B is None:
        spec_B = StimuliSpec()

    if tile_dimensions is not None:
        if face_r_dim != MAX_FACE_R_DIM:
            raise ValueError(
                f"tile_dimensions and face_r_dim are mutually exclusive: "
                f"when tile_dimensions is provided, face_r_dim is derived "
                f"automatically.  Got tile_dimensions={tile_dimensions}, "
                f"face_r_dim={face_r_dim}."
            )
        if num_faces != MAX_NUM_FACES:
            raise ValueError(
                f"tile_dimensions and num_faces are mutually exclusive: "
                f"when tile_dimensions is provided, num_faces is derived "
                f"automatically.  Got tile_dimensions={tile_dimensions}, "
                f"num_faces={num_faces}."
            )
        # Dense mode: derive face layout from tile_dimensions
        face_r_dim, num_faces_r_dim, num_faces_c_dim = get_tile_params(tile_dimensions)
        num_faces = num_faces_r_dim * num_faces_c_dim
        tile_cnt_A, tile_cnt_B, _ = calculate_tile_and_face_counts_w_tile_dimensions(
            input_dimensions_A,
            input_dimensions_B,
            face_r_dim,
            num_faces,
            tile_dimensions,
        )
    else:
        # Standard 32×32 tile path
        tile_cnt_A, tile_cnt_B, _ = calculate_tile_and_face_counts(
            input_dimensions_A, input_dimensions_B, face_r_dim, num_faces
        )

    num_elements_A = input_dimensions_A[0] * input_dimensions_A[1]
    num_elements_B = input_dimensions_B[0] * input_dimensions_B[1]

    srcA_tensor = _generate_source_tensor_v2(
        stimuli_format=stimuli_format_A,
        num_elements=num_elements_A,
        face_r_dim=face_r_dim,
        spec=spec_A,
    )
    srcB_tensor = _generate_source_tensor_v2(
        stimuli_format=stimuli_format_B,
        num_elements=num_elements_B,
        face_r_dim=face_r_dim,
        spec=spec_B,
    )

    srcA_tensor, srcB_tensor = _clamp_mx_tensors(
        srcA_tensor, srcB_tensor, stimuli_format_A, stimuli_format_B, output_format
    )

    return srcA_tensor, tile_cnt_A, srcB_tensor, tile_cnt_B
