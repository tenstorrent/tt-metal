# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import math
import warnings
from dataclasses import dataclass
from dataclasses import replace as _dataclass_replace
from typing import Callable, Dict, List, Optional, Set, Union

import torch

from .bfp_format_utils import bfp4b_to_float16b
from .format_config import (
    MXFP8_E4M3_MAX_NORMAL,
    MXFP8_E5M2_MAX_NORMAL,
    DataFormat,
)
from .llk_params import MathOperation, format_dict
from .tile_constants import (
    DEFAULT_TILE_C_DIM,
    DEFAULT_TILE_R_DIM,
    FACE_C_DIM,
    MAX_FACE_R_DIM,
    MAX_NUM_FACES,
    get_tile_params,
    validate_tile_dimensions,
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

        "uniform"
            Uniform random in [low, high].  For integer formats the bounds
            are narrowed to the tightest enclosing integers via
            torch.randint(ceil(low), floor(high) + 1), so only integers
            that actually lie within [low, high] are generated.

        "gaussian"
            Normal distribution with *mean* and *std*.  For integer formats
            the result is rounded and clamped to the representable range.

        "saw"
            Per-face sawtooth: linearly spaced values from *low* to *high*
            (torch.linspace), restarting every face.  For integer
            formats a float32 linspace is rounded and clamped.

        "log_uniform"
            exp(Uniform(log(low), log(high))).  Both *low* and *high* must
            be strictly positive.

        "constant"
            Every element is set to *value*.  Ignores *low*, *high*, *seed*.

        "sequential"
            Values 1, 2, 3, …, size.
            Ignores *low*, *high*, *seed*.

        "ramp"
            Continuous linear sweep: deterministic, evenly spaced values
            from *low* to *high* (torch.linspace) across the full
            tensor (short-circuits the face loop), producing one smooth
            ramp instead of a per-face sawtooth.  Useful for plotting
            function shapes.

        "gaussian_linspace"
            Deterministic sweep through the Gaussian domain via the
            inverse CDF (percent-point function).  Produces ordered values
            concentrated around *mean* and spreading into the tails at
            *std* scale.  No randomness involved — the output is fully
            determined by *mean*, *std*, and the element count.

        "log_uniform_linspace"
            Deterministic, logarithmically spaced values from *low* to
            *high*.  Both bounds must be strictly positive.  Equivalent to
            torch.logspace in natural-log base.

        "identity"
            Identity matrix pattern: *value* on the diagonal, zero
            elsewhere.  Requires 2-D input_dimensions.  For
            rectangular matrices the diagonal has min(rows, cols)
            entries.  Bypasses the face loop (tensor-level operation).
            Ignores *low*, *high*, *seed*.

        "face_identity"
            Per-face identity block: each face is treated as a
            (face_r_dim, FACE_C_DIM) matrix with *value* on the
            diagonal and zero elsewhere.  The diagonal length is
            min(face_r_dim, FACE_C_DIM).  Participates in the
            normal face loop, so it works naturally with face_specs
            and masked_faces.  Ignores *low*, *high*, *seed*.

        "custom"
            Explicit per-face values: the elements from *values* are
            written at the start of the flattened face, and every
            remaining element is zero.  Values are not repeated.
            Participates in the normal face loop, so it works naturally
            with face_specs and masked_faces.  Ignores *low*,
            *high*, *seed*, *value*.

        callable
            fn(size: int, dtype: torch.dtype, generator: Optional[torch.Generator]) -> torch.Tensor.
            The *generator* argument carries the per-operand RNG state (or
            "None" when no seed is set), enabling reproducible custom
            distributions.  The caller is responsible for producing a 1-D
            tensor of exactly *size* elements and returning it as the
            requested *dtype*.

    low : float
        Lower bound for "uniform", "saw", "ramp",
        "log_uniform", and "log_uniform_linspace".
        Defaults to 0.0.
    high : float
        Upper bound for "uniform", "saw", "ramp",
        "log_uniform", and "log_uniform_linspace".
        Defaults to 1.0.
    value : float
        Fill value used by "constant" (all elements), "identity"
        (diagonal elements), and "face_identity" (face diagonal
        elements).  Defaults to 1.0.
    values : list[float], optional
        Explicit value list for "custom".  Written at the start of
        the flattened face; remaining elements are zero.  For integer
        formats each value is rounded and clamped to the representable
        range.  Must not be empty when distribution="custom".
    mean : float
        Mean for "gaussian" and "gaussian_linspace".  Defaults to
        0.0.
    std : float
        Standard deviation for "gaussian" and "gaussian_linspace".
        Defaults to 1.0.
    seed : int, optional
        Seed for a per-spec torch.Generator.  "None" uses the global
        torch RNG state.  When an external generator is supplied to
        generate_face_v2 function the *seed* field is ignored so the caller
        controls state across faces.
    face_specs : list[StimuliSpec | None], optional
        Per-face overrides. face_specs itself may be "None"
        (no overrides at all).  Individual entries may also be "None",
        meaning "use the outer/base spec for this face."  Face *i* is
        generated with face_specs[i] when that entry is a
        StimuliSpec class, a "None" entry or an index beyond the
        list length falls back to the outer spec.
    masked_faces : set[int], optional
        Set of 0-based face indices whose output should be zeroed.
        Masking is applied *after* face generation (including any
        face_specs overrides), so it always wins.  Not compatible
        with global short-circuit distributions ("identity",
        "sequential", linspace variants) — raises ValueError
        if combined.
    """

    distribution: Union[str, Callable] = "uniform"
    low: float = 0.0
    high: float = 1.0
    value: float = 1.0
    values: Optional[List[float]] = None
    mean: float = 0.0
    std: float = 1.0
    seed: Optional[int] = None
    face_specs: Optional[List[Optional["StimuliSpec"]]] = None
    masked_faces: Optional[Set[int]] = None

    # ── convenience constructors ──────────────────────────────────────────────

    @classmethod
    def constant(cls, value: float = 1.0, **kwargs) -> "StimuliSpec":
        """All elements equal to *value*."""
        return cls(distribution="constant", value=value, **kwargs)

    @classmethod
    def identity(cls, diagonal_value: float = 1.0, **kwargs) -> "StimuliSpec":
        """Identity matrix: *diagonal_value* on the diagonal, zero elsewhere."""
        return cls(distribution="identity", value=diagonal_value, **kwargs)

    @classmethod
    def face_identity(cls, diagonal_value: float = 1.0, **kwargs) -> "StimuliSpec":
        """Per-face identity block: *diagonal_value* on the face diagonal, zero elsewhere."""
        return cls(distribution="face_identity", value=diagonal_value, **kwargs)

    @classmethod
    def custom(cls, values: List[float], **kwargs) -> "StimuliSpec":
        """Explicit values at the start of each face, zero-filled remainder."""
        return cls(distribution="custom", values=list(values), **kwargs)

    @classmethod
    def custom_faces(
        cls,
        face_values: Dict[int, List[float]],
        **kwargs,
    ) -> "StimuliSpec":
        """
        Build a spec that writes custom values on selected faces and leaves all
        other faces as zeros.

        Args:
            face_values: Mapping from 0-based face index to the value list for that face.
                Each list is written at the start of the flattened face; the rest
                of the face is zero-filled. Faces not present in the mapping are
                entirely zeros.
            **kwargs: Forwarded to the outer StimuliSpec (e.g. masked_faces).

        Returns:
            StimuliSpec with distribution="constant", value=0.0 as the base and a
            face_specs list that overlays per-face custom(values=...) specs
            where provided.
        """
        if not face_values:
            raise ValueError("face_values must be a non-empty dict")
        if any(idx < 0 for idx in face_values):
            raise ValueError(
                f"face_values keys must be non-negative, "
                f"got {sorted(k for k in face_values if k < 0)}"
            )
        if "face_specs" in kwargs:
            raise ValueError(
                "Cannot combine custom_faces() with an explicit face_specs "
                "argument — custom_faces builds face_specs internally"
            )
        max_idx = max(face_values)
        face_specs: List[Optional["StimuliSpec"]] = [None] * (max_idx + 1)
        for idx, vals in face_values.items():
            face_specs[idx] = cls.custom(values=vals)
        return cls(
            distribution="constant",
            value=0.0,
            face_specs=face_specs,
            **kwargs,
        )

    @classmethod
    def sequential(cls, **kwargs) -> "StimuliSpec":
        """Sequential values 1, 2, 3, …"""
        return cls(distribution="sequential", **kwargs)

    @classmethod
    def uniform(cls, low: float = 0.0, high: float = 1.0, **kwargs) -> "StimuliSpec":
        """Uniform random in [low, high]."""
        return cls(distribution="uniform", low=low, high=high, **kwargs)

    @classmethod
    def gaussian(cls, mean: float = 0.0, std: float = 1.0, **kwargs) -> "StimuliSpec":
        """Normal distribution with *mean* and *std*."""
        return cls(distribution="gaussian", mean=mean, std=std, **kwargs)

    @classmethod
    def saw(cls, low: float = 0.0, high: float = 1.0, **kwargs) -> "StimuliSpec":
        """Per-face sawtooth linspace from *low* to *high*, restarting every face."""
        return cls(distribution="saw", low=low, high=high, **kwargs)

    @classmethod
    def log_uniform(
        cls, low: float = 1e-4, high: float = 1.0, **kwargs
    ) -> "StimuliSpec":
        """Log-uniform in [low, high].  Both bounds must be strictly positive."""
        if low <= 0 or high <= 0:
            raise ValueError(
                f"log_uniform requires strictly positive low and high, "
                f"got low={low}, high={high}"
            )
        return cls(distribution="log_uniform", low=low, high=high, **kwargs)

    @classmethod
    def ramp(cls, low: float = 0.0, high: float = 1.0, **kwargs) -> "StimuliSpec":
        """Continuous linear sweep from *low* to *high* across the full tensor."""
        return cls(distribution="ramp", low=low, high=high, **kwargs)

    @classmethod
    def gaussian_linspace(
        cls, mean: float = 0.0, std: float = 1.0, **kwargs
    ) -> "StimuliSpec":
        """
        Deterministic sweep through a Gaussian: values spaced by inverting
        the normal CDF (no randomness, fully determined by mean/std/size).
        """
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
        """Return OperandSpecs with safe input domains for *op* and *data_format*.

        Args:
            op: Target math operation.
            data_format: Input data format; controls the numeric range and
            precision used to choose safe per-op input domains (e.g. tighter
            ranges for narrower MX/BFP formats).

        Returns:
            OperandSpecs with per-operand domain specs.

        Raises:
            KeyError: If *op* is not in the registry.
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
    """Per-operand input domain specs returned by StimuliSpec.for_op method.

    For binary ops where operands need different domains (e.g. divisor avoids
    zero), spec_A and spec_B differ; unary ops need only spec_A.
    spec_B defaults to a copy of spec_A when "None".
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
    """Safe input range for exp(x) per format to avoid overflow."""
    if fmt == DataFormat.MxFp8P:
        spec = StimuliSpec(distribution="uniform", low=-5.0, high=5.0)
    elif fmt in (DataFormat.Float16, DataFormat.MxFp8R):
        spec = StimuliSpec(distribution="uniform", low=-10.0, high=10.0)
    else:
        spec = StimuliSpec(distribution="uniform", low=-80.0, high=80.0)
    return OperandSpecs(spec_A=spec)


def _exp2_spec(fmt: DataFormat) -> OperandSpecs:
    """Safe input range for exp2(x) = 2^x per format to avoid overflow."""
    if fmt == DataFormat.MxFp8P:
        spec = StimuliSpec(distribution="uniform", low=-7.0, high=7.0)
    elif fmt in (DataFormat.Float16, DataFormat.MxFp8R):
        spec = StimuliSpec(distribution="uniform", low=-14.0, high=14.0)
    else:
        spec = StimuliSpec(distribution="uniform", low=-100.0, high=100.0)
    return OperandSpecs(spec_A=spec)


def _square_spec(fmt: DataFormat) -> OperandSpecs:
    """Safe input range for square(x) = x^2 per format to avoid overflow."""
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
# Tile / face count utilities
# ─────────────────────────────────────────────────────────────────────────────


def calculate_tile_and_face_counts(
    input_dimensions_A: list,
    input_dimensions_B: list,
    face_r_dim: int,
    num_faces: int,
) -> tuple[int, int, int]:
    """
    Calculate tile counts and faces to generate based on input dimensions and face configuration.
    This is the ORIGINAL function that always uses 32x32 tiles.

    Args:
        input_dimensions_A: [height, width] in elements for input A
        input_dimensions_B: [height, width] in elements for input B
        face_r_dim: Number of rows in a face (typically 16 for full faces)
        num_faces: Number of faces to generate for partial face case

    Returns:
        tuple: (tile_cnt_A, tile_cnt_B, faces_to_generate)
    """
    assert (
        face_r_dim == MAX_FACE_R_DIM or face_r_dim == input_dimensions_A[0]
    ), f"Invalid face_r_dim, got {face_r_dim}"

    # Handle partial faces
    if face_r_dim < MAX_FACE_R_DIM:
        # Partial face case: generate exactly num_faces worth of data
        tile_cnt_A, tile_cnt_B = 1, 1
        faces_to_generate = num_faces  # Generate exactly the right number of faces
    else:
        # Full tile case - always use 32x32 tiles
        tile_cnt_A = (
            input_dimensions_A[0]
            // DEFAULT_TILE_R_DIM
            * input_dimensions_A[1]
            // DEFAULT_TILE_C_DIM
        )
        tile_cnt_B = (
            input_dimensions_B[0]
            // DEFAULT_TILE_R_DIM
            * input_dimensions_B[1]
            // DEFAULT_TILE_C_DIM
        )
        faces_to_generate = MAX_NUM_FACES

    return tile_cnt_A, tile_cnt_B, faces_to_generate


def calculate_tile_and_face_counts_w_tile_dimensions(
    input_dimensions_A: list,
    input_dimensions_B: list,
    face_r_dim: int,
    num_faces: int,
    tile_dimensions: list,
) -> tuple[int, int, int]:
    """
    Calculate tile counts and faces to generate for variable tile dimensions (dense mode).

    Args:
        input_dimensions_A: [height, width] in elements for input A
        input_dimensions_B: [height, width] in elements for input B
        face_r_dim: Number of rows in a face (1, 2, 4, 8, or 16)
        num_faces: Number of faces per tile (1, 2, or 4)
        tile_dimensions: [rows, cols] for tile size

    Returns:
        tuple: (tile_cnt_A, tile_cnt_B, faces_to_generate)
    """
    validate_tile_dimensions(tile_dimensions)
    tile_r_dim, tile_c_dim = tile_dimensions

    # Calculate tile counts based on actual tile dimensions
    tile_cnt_A = (input_dimensions_A[0] // tile_r_dim) * (
        input_dimensions_A[1] // tile_c_dim
    )
    tile_cnt_B = (input_dimensions_B[0] // tile_r_dim) * (
        input_dimensions_B[1] // tile_c_dim
    )
    # Always generate all faces to fill the tile densely
    faces_to_generate = num_faces

    return tile_cnt_A, tile_cnt_B, faces_to_generate


def _clamp_mx_tensors(
    srcA_tensor: torch.Tensor,
    srcB_tensor: torch.Tensor,
    stimuli_format_A: DataFormat,
    stimuli_format_B: DataFormat,
    output_format: DataFormat = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Clamp tensors for MX format compatibility.

    Args:
        srcA_tensor: Source A tensor
        srcB_tensor: Source B tensor
        stimuli_format_A: Data format for source A
        stimuli_format_B: Data format for source B
        output_format: Optional output format for range constraints

    Returns:
        tuple: (clamped_srcA_tensor, clamped_srcB_tensor)
    """
    # Clamp inputs if both are different MX formats (use more restrictive MxFp8P)
    if stimuli_format_A.is_mx_format() and stimuli_format_B.is_mx_format():
        if stimuli_format_A != stimuli_format_B:
            srcA_tensor = torch.clamp(
                srcA_tensor, -MXFP8_E4M3_MAX_NORMAL, MXFP8_E4M3_MAX_NORMAL
            )
            srcB_tensor = torch.clamp(
                srcB_tensor, -MXFP8_E4M3_MAX_NORMAL, MXFP8_E4M3_MAX_NORMAL
            )

    # Clamp inputs based on output format to prevent excessive rounding errors
    if output_format == DataFormat.MxFp8P:
        srcA_tensor = torch.clamp(
            srcA_tensor, -MXFP8_E4M3_MAX_NORMAL, MXFP8_E4M3_MAX_NORMAL
        )
        srcB_tensor = torch.clamp(
            srcB_tensor, -MXFP8_E4M3_MAX_NORMAL, MXFP8_E4M3_MAX_NORMAL
        )
    elif output_format == DataFormat.MxFp8R:
        srcA_tensor = torch.clamp(
            srcA_tensor, -MXFP8_E5M2_MAX_NORMAL, MXFP8_E5M2_MAX_NORMAL
        )
        srcB_tensor = torch.clamp(
            srcB_tensor, -MXFP8_E5M2_MAX_NORMAL, MXFP8_E5M2_MAX_NORMAL
        )

    return srcA_tensor, srcB_tensor


# ─────────────────────────────────────────────────────────────────────────────
# Matmul helpers
# ─────────────────────────────────────────────────────────────────────────────


def _mask_tile(
    tile: torch.Tensor,
    num_faces: int,
    is_matrix_B: bool,
    face_r_dim: int = MAX_FACE_R_DIM,
) -> torch.Tensor:
    masked = tile.clone()
    if num_faces == 1:
        # Keep only f0
        masked[:MAX_FACE_R_DIM, FACE_C_DIM:] = 0  # Zero f1
        masked[face_r_dim:, :] = 0  # Zero f2, f3 and part of f0
    elif num_faces == 2:
        if is_matrix_B:
            # matrix B (In1/SrcA): keep partial f0, f2
            if face_r_dim < MAX_FACE_R_DIM:
                masked[face_r_dim:MAX_FACE_R_DIM, :FACE_C_DIM] = 0  # Zero part of f0
                masked[MAX_FACE_R_DIM + face_r_dim :, :FACE_C_DIM] = (
                    0  # Zero part of f2
                )
            masked[:MAX_FACE_R_DIM, FACE_C_DIM:] = 0  # Zero f1
            masked[MAX_FACE_R_DIM:, FACE_C_DIM:] = 0  # Zero f3
        else:
            # matrix A (In0/SrcB): keep f0, f1
            masked[face_r_dim:, :] = 0  # Zero part of f0 and f1
    return masked


def generate_face_matmul_data(
    num_faces: int,
    stimuli_format: DataFormat,
    input_dimensions=[DEFAULT_TILE_R_DIM, DEFAULT_TILE_C_DIM],
    is_matrix_A=True,  # True for matrix A (SrcB), False for matrix B (SrcA)
    face_r_dim=MAX_FACE_R_DIM,
) -> torch.Tensor:

    # Validate num_faces
    if num_faces not in [1, 2, MAX_NUM_FACES]:
        raise ValueError(f"num_faces must be 1, 2, or {MAX_NUM_FACES}, got {num_faces}")

    # Validate input_dimensions
    rows, cols = input_dimensions
    if rows % DEFAULT_TILE_R_DIM != 0 or cols % DEFAULT_TILE_C_DIM != 0:
        raise ValueError(
            f"Input dimensions must be multiples of {DEFAULT_TILE_R_DIM}, "
            f"got {input_dimensions}"
        )

    rt, ct = rows // DEFAULT_TILE_R_DIM, cols // DEFAULT_TILE_C_DIM
    dtype = format_dict[stimuli_format]

    out = torch.rand(rt, ct, DEFAULT_TILE_R_DIM, DEFAULT_TILE_C_DIM, dtype=dtype)
    mask = _mask_tile(
        torch.ones(DEFAULT_TILE_R_DIM, DEFAULT_TILE_C_DIM, dtype=dtype),
        num_faces,
        not is_matrix_A,
        face_r_dim,
    )
    out *= mask
    out = out.permute(0, 2, 1, 3).reshape(rows, cols)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# L1 view helpers
# ─────────────────────────────────────────────────────────────────────────────


def convert_to_l1_view(
    tilized_tensor: torch.Tensor,
    input_dimensions: list,
    tile_dimensions: list = None,
) -> torch.Tensor:
    """
    Convert a tilized tensor to its L1 memory view by condensing data based on tile dimensions.

    This function extracts only the data that corresponds to the specified tile dimensions
    and places it at the beginning of each tile, with the remaining space zeroed out.
    The full tile size (1024 elements) is preserved.

    Tilized format: faces are stored sequentially [f0 (256), f1 (256), f2 (256), f3 (256)]
    Within each face, data is stored row-major (16 rows × 16 cols).

    Face layout in a 32×32 tile:
    - f0: rows 0-15, cols 0-15  (top-left)
    - f1: rows 0-15, cols 16-31 (top-right)
    - f2: rows 16-31, cols 0-15 (bottom-left)
    - f3: rows 16-31, cols 16-31 (bottom-right)

    Examples:
    - tile_dimensions=[32, 32]: full tile, no change [f0, f1, f2, f3]
    - tile_dimensions=[16, 32]: top half [f0, f1, 0, 0]
    - tile_dimensions=[32, 16]: left half [f0, f2, 0, 0]
    - tile_dimensions=[16, 16]: top-left only [f0, 0, 0, 0]
    - tile_dimensions=[8, 32]: first 8 rows [f0_rows0-7, f1_rows0-7, 0, ...]

    Args:
        tilized_tensor: Input tensor in tilized format (faces stored sequentially per tile)
        input_dimensions: [rows, cols] of the full input matrix
        tile_dimensions: [rows, cols] to keep per tile (default [32, 32])
                        rows must be one of: 1, 2, 4, 8, 16, 32
                        cols must be one of: 16, 32

    Returns:
        Tensor with condensed data at the beginning (face by face), zeros at the end
    """
    if tile_dimensions is None:
        tile_dimensions = [32, 32]

    tile_rows, tile_cols = tile_dimensions

    valid_rows = {1, 2, 4, 8, 16, 32}
    valid_cols = {16, 32}

    if tile_rows not in valid_rows:
        raise ValueError(
            f"tile_dimensions[0] (rows) must be one of {sorted(valid_rows)}, got {tile_rows}"
        )
    if tile_cols not in valid_cols:
        raise ValueError(
            f"tile_dimensions[1] (cols) must be one of {sorted(valid_cols)}, got {tile_cols}"
        )

    rows, cols = input_dimensions
    if rows % 32 != 0 or cols % 32 != 0:
        raise ValueError(
            f"Input dimensions must be multiples of 32, got {input_dimensions}"
        )

    # If using full tile dimensions, no conversion needed
    if tile_rows == 32 and tile_cols == 32:
        return tilized_tensor.flatten()

    # Calculate number of tiles
    tile_cnt = (rows // 32) * (cols // 32)
    face_rows = 16
    face_cols = 16

    # Reshape to [num_tiles, 4, 16, 16] for easier face/row manipulation
    # Face order in tilized format: [f0, f1, f2, f3]
    tensor_by_tiles = tilized_tensor.flatten().view(tile_cnt, 4, face_rows, face_cols)

    # Create output tensor with same shape, initialized to zeros
    output = torch.zeros_like(tensor_by_tiles)

    # Determine which faces to use and how many rows from each
    # tile_rows <= 16: only top faces (f0, f1), take tile_rows from each
    # tile_rows == 32: all faces, take all 16 rows from each
    # tile_cols == 16: only left faces (f0, f2)
    # tile_cols == 32: both left and right faces
    use_bottom_faces = tile_rows == 32
    use_right_faces = tile_cols == 32
    rows_per_face = tile_rows if tile_rows <= 16 else 16

    # Extract data face by face (not interleaved)
    for tile_idx in range(tile_cnt):
        out_flat = []

        # f0: always used - extract rows_per_face rows
        for row in range(rows_per_face):
            out_flat.extend(tensor_by_tiles[tile_idx, 0, row, :].tolist())

        # f1: used if tile_cols == 32 - extract rows_per_face rows
        if use_right_faces:
            for row in range(rows_per_face):
                out_flat.extend(tensor_by_tiles[tile_idx, 1, row, :].tolist())

        # f2: used if tile_rows == 32 - extract all 16 rows
        if use_bottom_faces:
            for row in range(16):
                out_flat.extend(tensor_by_tiles[tile_idx, 2, row, :].tolist())

        # f3: used if tile_rows == 32 and tile_cols == 32 - extract all 16 rows
        if use_bottom_faces and use_right_faces:
            for row in range(16):
                out_flat.extend(tensor_by_tiles[tile_idx, 3, row, :].tolist())

        # Place condensed data at the beginning of the tile
        out_flat_tensor = torch.tensor(out_flat, dtype=tilized_tensor.dtype)
        output_flat = output[tile_idx].flatten()
        output_flat[: len(out_flat)] = out_flat_tensor
        output[tile_idx] = output_flat.view(4, face_rows, face_cols)

    # Flatten and return
    return output.flatten()


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────


# Set of distribution names that produce a deterministic global sweep and
# should short-circuit the face loop in _generate_source_tensor_v2.
_LINSPACE_DISTRIBUTIONS = frozenset(
    {
        "ramp",
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
    """Generate a deterministic linspace-style tensor (ramp/gaussian_linspace/log_uniform_linspace).

    Args:
        spec: Stimuli specification with distribution type and bounds.
        size: Number of elements to generate.
        dtype: Target torch dtype (computation uses float32, cast at end).

    Returns:
        1-D tensor of *size* elements in the requested dtype.
    """
    dist = spec.distribution

    if dist == "ramp":
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
    """Return the valid integer range (min, max) inclusive for *stimuli_format*."""
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
    """Generate one face of integer-format data according to *spec*.

    Bounds are narrowed to integers within [spec.low, spec.high] intersected
    with the format's representable range.  Degenerate ranges produce constant
    tensors (with a warning if no valid integer exists).

    Args:
        spec: Stimuli specification with distribution type and bounds.
        stimuli_format: Target integer data format.
        size: Number of elements per face.
        dtype: Target torch dtype.
        generator: Optional RNG state for reproducibility.

    Returns:
        1-D integer tensor of *size* elements.
    """
    int_min, int_max = _get_integer_bounds(stimuli_format)
    # Narrow to the tightest set of integers that lie within [spec.low, spec.high]:
    #   ceil(low)  = smallest integer ≥ spec.low
    #   floor(high) = largest integer ≤ spec.high
    # Then clamp to the format's representable range.
    low = max(math.ceil(spec.low), int_min)
    high = min(math.floor(spec.high), int_max)

    if high < low:
        # No integer exists in [spec.low, spec.high] after clamping —
        # fall back to the nearest representable integer (midpoint, rounded).
        fallback = max(int_min, min(round((spec.low + spec.high) / 2), int_max))
        warnings.warn(
            f"No integer exists in [{spec.low}, {spec.high}] for "
            f"{stimuli_format.name} (representable range [{int_min}, {int_max}]). "
            f"Returning constant tensor filled with {fallback}.",
            stacklevel=3,
        )
        return torch.full((size,), fallback, dtype=dtype)

    # Single valid integer — return constant tensor
    if high == low:
        return torch.full((size,), low, dtype=dtype)

    distribution = spec.distribution

    if distribution == "uniform":
        return torch.randint(
            low=low, high=high + 1, size=(size,), dtype=dtype, generator=generator
        )

    if distribution == "saw":
        return (
            torch.linspace(spec.low, spec.high, size, dtype=torch.float32)
            .round()
            .clamp(low, high)
            .to(dtype=dtype)
        )

    if distribution == "gaussian":
        raw = (
            torch.randn(size, dtype=torch.float32, generator=generator) * spec.std
            + spec.mean
        )
        return raw.round().clamp(low, high).to(dtype=dtype)

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
        return raw.round().clamp(low, high).to(dtype=dtype)

    if distribution in _LINSPACE_DISTRIBUTIONS:
        raw = _generate_linspace_tensor(spec, size, torch.float32)
        return raw.round().clamp(low, high).to(dtype=dtype)

    raise ValueError(
        f"Unknown distribution {distribution!r} for integer format. "
        f"Expected one of 'uniform', 'gaussian', 'saw', 'log_uniform', "
        f"'ramp', 'gaussian_linspace', 'log_uniform_linspace', "
        f"'constant', 'sequential', 'face_identity', 'custom', or a callable."
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
    """Generate a single face tensor of shape (face_r_dim * 16,) using *spec*.

    Args:
        spec: Generation specification (distribution, bounds, seed, etc.).
        stimuli_format: Target hardware data format.
        face_r_dim: Rows per face (1-16, default 16).
        generator: External RNG state; when supplied, spec.seed is ignored.

    Returns:
        1-D tensor with face_r_dim * 16 elements.
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

    if distribution == "identity":
        raise ValueError(
            "distribution='identity' is a tensor-level operation and cannot "
            "be used in a per-face context (e.g. inside face_specs). "
            "Use distribution='face_identity' for per-face identity blocks."
        )

    if distribution == "face_identity":
        diag_val = spec.value
        if stimuli_format.is_integer():
            int_min, int_max = _get_integer_bounds(stimuli_format)
            diag_val = max(int_min, min(int(round(diag_val)), int_max))
        face = torch.zeros(face_r_dim, FACE_C_DIM, dtype=dtype)
        face.diagonal()[:] = diag_val
        return face.reshape(-1)

    if distribution == "custom":
        if spec.values is None or len(spec.values) == 0:
            raise ValueError("distribution='custom' requires a non-empty 'values' list")
        if len(spec.values) > size:
            raise ValueError(
                f"custom values list has {len(spec.values)} elements "
                f"but face has only {size} "
                f"({face_r_dim} rows × {FACE_C_DIM} cols)"
            )
        if stimuli_format.is_integer():
            int_min, int_max = _get_integer_bounds(stimuli_format)
            vals = [max(int_min, min(int(round(v)), int_max)) for v in spec.values]
        else:
            vals = list(spec.values)
        tensor = torch.zeros(size, dtype=dtype)
        tensor[: len(vals)] = torch.tensor(vals, dtype=dtype)
        return tensor

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

    if distribution == "saw":
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
        f"Expected one of 'uniform', 'gaussian', 'saw', 'log_uniform', "
        f"'ramp', 'gaussian_linspace', 'log_uniform_linspace', "
        f"'constant', 'sequential', 'face_identity', 'custom', or a callable."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Private: full operand tensor
# ─────────────────────────────────────────────────────────────────────────────


def _generate_source_tensor_v2(
    stimuli_format: DataFormat,
    num_elements: int,
    face_r_dim: int,
    spec: StimuliSpec,
    input_dimensions: Optional[list] = None,
) -> torch.Tensor:
    """Generate a full operand tensor of *num_elements* using *spec*.

    Iterates over faces with a shared RNG, applying face_specs overrides and
    masked_faces zeroing.  Identity, sequential, and linspace distributions
    short-circuit the face loop. BFP4_b tensors are post-quantized.

    Args:
        stimuli_format: Hardware data format.
        num_elements: Total elements to generate.
        face_r_dim: Rows per face (1-16).
        spec: Stimuli specification.
        input_dimensions: [rows, cols], required for identity distribution.

    Returns:
        1-D tensor of *num_elements* elements.
    """
    dtype = _get_dtype_for_format(stimuli_format)

    # ── masked_faces validation for short-circuit distributions ──────────
    _SHORT_CIRCUIT_DISTRIBUTIONS = frozenset(
        {"identity", "sequential"} | _LINSPACE_DISTRIBUTIONS
    )
    if spec.masked_faces and spec.distribution in _SHORT_CIRCUIT_DISTRIBUTIONS:
        raise ValueError(
            f"masked_faces cannot be used with distribution={spec.distribution!r} "
            f"because it bypasses the face loop.  Use a per-face distribution "
            f"(uniform, gaussian, saw, …) instead."
        )

    # ── identity: tensor-level structured pattern (no face loop) ───────
    if spec.distribution == "identity":
        if input_dimensions is None or len(input_dimensions) != 2:
            raise ValueError(
                "distribution='identity' requires input_dimensions=[rows, cols]"
            )
        rows, cols = input_dimensions
        diag_val = spec.value
        if stimuli_format.is_integer():
            int_min, int_max = _get_integer_bounds(stimuli_format)
            diag_val = max(int_min, min(int(round(diag_val)), int_max))
        tensor = torch.zeros(rows, cols, dtype=dtype)
        tensor.diagonal()[:] = diag_val
        tensor = tensor.reshape(-1)
        if stimuli_format == DataFormat.Bfp4_b:
            tensor = bfp4b_to_float16b(tensor)
        return tensor

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
            override = spec.face_specs[face_idx]
            if override is not None:
                face_spec = override

        face_tensor = generate_face_v2(
            spec=face_spec,
            stimuli_format=stimuli_format,
            face_r_dim=face_r_dim,
            generator=gen,
        )

        # ── face masking: zero out selected faces ────────────────────────
        if spec.masked_faces and face_idx in spec.masked_faces:
            face_tensor = torch.zeros_like(face_tensor)

        face_tensors.append(face_tensor)

    tensor = torch.cat(face_tensors)[:num_elements]

    # Simulate the hardware pack/unpack round-trip for BFP4_b so that stimuli
    # already reflect the precision loss introduced by the format.
    if stimuli_format == DataFormat.Bfp4_b:
        tensor = bfp4b_to_float16b(tensor)

    return tensor


# ─────────────────────────────────────────────────────────────────────────────
# Built-in defaults for omitted specs
# ─────────────────────────────────────────────────────────────────────────────


def _default_bfp8b_face(
    size: int, dtype: torch.dtype, gen: Optional[torch.Generator]
) -> torch.Tensor:
    integer_part = torch.randint(0, 3, (size,))
    fraction = torch.randint(0, 16, (size,)).to(dtype=torch.bfloat16) / 16.0
    return integer_part.to(dtype=torch.bfloat16) + fraction


def _default_bfp4b_face(
    size: int, dtype: torch.dtype, gen: Optional[torch.Generator]
) -> torch.Tensor:
    integer_part = torch.randint(0, 3, (size,))
    fraction = torch.randint(0, 8, (size,)).to(dtype=torch.bfloat16) / 8.0
    return integer_part.to(dtype=torch.bfloat16) + fraction


def _default_spec_for_format(stimuli_format: DataFormat) -> StimuliSpec:
    """Return the built-in default StimuliSpec for a given data format.

    Defaults are chosen to give reasonable value ranges and avoid overflows
    (e.g. positive ranges for floats, half-range for integers).
    """
    if stimuli_format == DataFormat.MxFp8R:
        return StimuliSpec.gaussian(mean=0.1, std=0.05 * MXFP8_E5M2_MAX_NORMAL)
    if stimuli_format == DataFormat.MxFp8P:
        return StimuliSpec.gaussian(mean=0.1, std=0.05 * MXFP8_E4M3_MAX_NORMAL)
    if stimuli_format == DataFormat.Bfp8_b:
        return StimuliSpec(distribution=_default_bfp8b_face)
    if stimuli_format == DataFormat.Bfp4_b:
        return StimuliSpec(distribution=_default_bfp4b_face)
    if stimuli_format.is_integer():
        if stimuli_format == DataFormat.UInt32:
            return StimuliSpec.uniform(low=0.0, high=float(2**32 - 2))
        dtype = format_dict[stimuli_format]
        v1_type_max = torch.iinfo(dtype).max // 2
        return StimuliSpec.uniform(low=0.0, high=float(v1_type_max - 1))
    return StimuliSpec.uniform(low=0.1, high=1.1)


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
    """Generate test stimuli for two operands.

    When tile_dimensions is provided, operates in dense mode with derived face layout;
    otherwise uses the standard 32x32 tile path.

    Args:
        stimuli_format_A: Hardware data format for operand A.
        input_dimensions_A: [height, width] in elements (default [32, 32]).
        stimuli_format_B: Hardware data format for operand B.
        input_dimensions_B: [height, width] in elements (default [32, 32]).
        spec_A: Generation spec for operand A (default: format-aware built-in spec).
        spec_B: Generation spec for operand B (default: format-aware built-in spec).
        tile_dimensions: [rows, cols] tile size for dense mode (default None = standard path).
        face_r_dim: Rows per face, 1-16 (ignored in dense mode, default 16).
        num_faces: Faces per tile for partial-face case (ignored in dense mode, default 4).
        output_format: Clamp outputs for mixed MX-format pairs when set.

    Returns:
        (srcA_tensor, tile_cnt_A, srcB_tensor, tile_cnt_B).
    """
    if input_dimensions_A is None:
        input_dimensions_A = [DEFAULT_TILE_R_DIM, DEFAULT_TILE_C_DIM]
    if input_dimensions_B is None:
        input_dimensions_B = [DEFAULT_TILE_R_DIM, DEFAULT_TILE_C_DIM]
    if spec_A is None:
        spec_A = _default_spec_for_format(stimuli_format_A)
    if spec_B is None:
        spec_B = _default_spec_for_format(stimuli_format_B)

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
        input_dimensions=input_dimensions_A,
    )
    srcB_tensor = _generate_source_tensor_v2(
        stimuli_format=stimuli_format_B,
        num_elements=num_elements_B,
        face_r_dim=face_r_dim,
        spec=spec_B,
        input_dimensions=input_dimensions_B,
    )

    srcA_tensor, srcB_tensor = _clamp_mx_tensors(
        srcA_tensor, srcB_tensor, stimuli_format_A, stimuli_format_B, output_format
    )

    return srcA_tensor, tile_cnt_A, srcB_tensor, tile_cnt_B
