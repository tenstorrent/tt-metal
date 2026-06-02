# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

# ─────────────────────────────────────────────────────────────────────────────
# Distribution kinds
# ─────────────────────────────────────────────────────────────────────────────


class DistributionKind(str, Enum):
    UNIFORM = "uniform"
    GAUSSIAN = "gaussian"
    SAW = "saw"
    LOG_UNIFORM = "log_uniform"
    CONSTANT = "constant"
    SEQUENTIAL = "sequential"
    RAMP = "ramp"
    GAUSSIAN_LINSPACE = "gaussian_linspace"
    LOG_UNIFORM_LINSPACE = "log_uniform_linspace"
    IDENTITY = "identity"
    FACE_IDENTITY = "face_identity"
    CUSTOM = "custom"
    ULP_SWEEP = "ulp_sweep"


# ─────────────────────────────────────────────────────────────────────────────
# StimuliSpec
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class StimuliSpec:
    """
    Declarative per-operand specification for LLK test stimuli generation.

    Parameters
    ----------
    distribution : DistributionKind or callable
        How values are sampled. Supported enum members (from DistributionKind):

        "uniform"
            Uniform random in [low, high]. For integer formats the bounds
            are narrowed to the tightest enclosing integers via
            torch.randint(ceil(low), floor(high) + 1), so only integers
            that actually lie within [low, high] are generated.

        "gaussian"
            Normal distribution with *mean* and *std*. For integer formats
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
            Arithmetic progression.  By default generates 1, 2, 3, …,
            size (legacy behavior).  Use the ``sequential()`` factory
            with *low*, *high*, and *step* to customize.  When *high*
            is set, positions beyond it are zero-filled.  Supports
            *intervals* — values outside the union are zeroed.

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

        "ulp_sweep"
            Exhaustive 1-ULP sweep: enumerates every finite representable
            value in [low, high] for the target format (sorted, deduplicated),
            pads with zeros to fill the tensor.  Only Float16_b and
            Float16 formats are supported.  Bypasses the face loop
            (tensor-level operation).  Ignores *seed*, *mean*, *std*,
            *value*, *intervals*, *face_specs*, *masked_faces*.

            **Dimension auto-sizing (operand A only):** when spec_A uses
            ULP_SWEEP, generate_stimuli ignores any caller-supplied
            input_dimensions_A and computes a 32×(32*N) layout large
            enough to hold all representable values.  When spec_B is
            omitted, input_dimensions_B is mirrored from A.  If spec_B
            is provided explicitly, input_dimensions_B is *not*
            auto-sized — the caller must supply compatible dimensions.

        callable
            fn(size: int, dtype: torch.dtype, generator: Optional[torch.Generator]) -> torch.Tensor.
            The *generator* argument carries the per-operand RNG state (or
            "None" when no seed is set), enabling reproducible custom
            distributions.  The caller is responsible for producing a 1-D
            tensor of exactly *size* elements and returning it as the
            requested *dtype*.

    low: float
        Lower bound for "uniform", "saw", "ramp",
        "log_uniform", and "log_uniform_linspace".
        Defaults to 0.0.
    high: float
        Upper bound for "uniform", "saw", "ramp",
        "log_uniform", and "log_uniform_linspace".
        Defaults to 1.0.
    value: float
        Fill value used by "constant" (all elements), "identity"
        (diagonal elements), and "face_identity" (face diagonal
        elements).  Defaults to 1.0.
    values: list[float], optional
        Explicit value list for "custom".  Written at the start of
        the flattened face; remaining elements are zero.  For integer
        formats each value is rounded and clamped to the representable
        range.  Must not be empty when distribution="custom".
    mean: float
        Mean for "gaussian" and "gaussian_linspace".  Defaults to
        0.0.
    std: float
        Standard deviation for "gaussian" and "gaussian_linspace".
        Defaults to 1.0.
    seed: int, optional
        Seed for a per-spec torch.Generator.  "None" uses the global
        torch RNG state.  When an external generator is supplied to
        generate_face function the *seed* field is ignored so the caller
        controls state across faces.
    face_specs: list[StimuliSpec | None], optional
        Per-face overrides. face_specs itself may be "None"
        (no overrides at all).  Individual entries may also be "None",
        meaning "use the outer/base spec for this face."  Face *i* is
        generated with face_specs[i] when that entry is a
        StimuliSpec class, a "None" entry or an index beyond the
        list length falls back to the outer spec.
    masked_faces: set[int], optional
        Set of 0-based face indices whose output should be zeroed.
        Masking is applied *after* face generation (including any
        face_specs overrides), so it always wins.  Not compatible
        with global short-circuit distributions ("identity",
        "sequential", linspace variants) — raises ValueError
        if combined.
    intervals: list[tuple[float, float]], optional
        Union of [low, high] ranges.  When set, *low*/*high* are ignored
        and values are generated from the union of these intervals.
        Supported distributions:

        - **uniform** (random): each interval is selected with
          probability proportional to its length.
        - **log_uniform** (random): each interval is selected with
          probability proportional to its log-space length.
        - **integer uniform**: each interval is selected with
          probability proportional to the number of integer points.
        - **gaussian**: truncated Gaussian via rejection sampling —
          only values inside the union are kept.  For integer formats
          the samples are rounded and clamped to representable integers
          within the intervals.
        - **saw**: piecewise linspace — each interval gets a proportional
          share of the total *size* elements, concatenated in order.
        - **ramp / log_uniform_linspace**: same piecewise linspace
          semantics as *saw*, applied at the tensor level.
        - **sequential**: values outside the union are zeroed.

        Not supported for *gaussian_linspace* (raises ValueError).
        Ignored by *identity*, *face_identity*, *custom*.
        Example::

            StimuliSpec.uniform(intervals=[(-10.0, -1.0), (1.0, 10.0)])
    """

    distribution: Union[DistributionKind, Callable] = DistributionKind.UNIFORM
    low: float = 0.0
    high: float = 1.0
    value: float = 1.0
    values: Optional[List[float]] = None
    mean: float = 0.0
    std: float = 1.0
    seed: Optional[int] = None
    face_specs: Optional[List[Optional["StimuliSpec"]]] = None
    masked_faces: Optional[Set[int]] = None
    intervals: Optional[List[Tuple[float, float]]] = None

    def __post_init__(self) -> None:
        if not (
            callable(self.distribution)
            or isinstance(self.distribution, DistributionKind)
        ):
            raise TypeError(
                f"StimuliSpec.distribution must be DistributionKind or callable, "
                f"got {type(self.distribution).__name__!r}: {self.distribution!r}"
            )

    # ── convenience constructors ──────────────────────────────────────────────

    @classmethod
    def constant(cls, value: float = 1.0, **kwargs) -> "StimuliSpec":
        """All elements equal to *value*."""
        return cls(distribution=DistributionKind.CONSTANT, value=value, **kwargs)

    @classmethod
    def identity(cls, diagonal_value: float = 1.0, **kwargs) -> "StimuliSpec":
        """Identity matrix: *diagonal_value* on the diagonal, zero elsewhere."""
        return cls(
            distribution=DistributionKind.IDENTITY, value=diagonal_value, **kwargs
        )

    @classmethod
    def face_identity(cls, diagonal_value: float = 1.0, **kwargs) -> "StimuliSpec":
        """Per-face identity block: *diagonal_value* on the face diagonal, zero elsewhere."""
        return cls(
            distribution=DistributionKind.FACE_IDENTITY, value=diagonal_value, **kwargs
        )

    @classmethod
    def custom(cls, values: List[float], **kwargs) -> "StimuliSpec":
        """Explicit values at the start of each face, zero-filled remainder."""
        return cls(distribution=DistributionKind.CUSTOM, values=list(values), **kwargs)

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
            face_values:
                Mapping from 0-based face index to the value list for that face.
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
            distribution=DistributionKind.CONSTANT,
            value=0.0,
            face_specs=face_specs,
            **kwargs,
        )

    @classmethod
    def sequential(
        cls,
        low: Optional[float] = None,
        high: Optional[float] = None,
        step: float = 1.0,
        **kwargs,
    ) -> "StimuliSpec":
        """Arithmetic progression, optionally bounded with zero-fill.

        No arguments produces the legacy 1, 2, 3, … sequence.  With
        *low*/*high*/*step*, generates low, low+step, low+2*step, …
        up to *high* (then zero-fills the remainder).
        """
        kw = dict(distribution=DistributionKind.SEQUENTIAL, **kwargs)
        if low is not None:
            kw["low"] = low
        if high is not None:
            kw["high"] = high
        kw["std"] = step
        return cls(**kw)

    @classmethod
    def uniform(cls, low: float = 0.0, high: float = 1.0, **kwargs) -> "StimuliSpec":
        """Uniform random in [low, high]."""
        return cls(distribution=DistributionKind.UNIFORM, low=low, high=high, **kwargs)

    @classmethod
    def gaussian(cls, mean: float = 0.0, std: float = 1.0, **kwargs) -> "StimuliSpec":
        """Normal distribution with *mean* and *std*."""
        return cls(distribution=DistributionKind.GAUSSIAN, mean=mean, std=std, **kwargs)

    @classmethod
    def saw(cls, low: float = 0.0, high: float = 1.0, **kwargs) -> "StimuliSpec":
        """Per-face sawtooth linspace from *low* to *high*, restarting every face."""
        return cls(distribution=DistributionKind.SAW, low=low, high=high, **kwargs)

    @classmethod
    def log_uniform(
        cls, low: float = 1e-4, high: float = 1.0, **kwargs
    ) -> "StimuliSpec":
        """Log-uniform in [low, high]. Both bounds must be strictly positive."""
        if low <= 0 or high <= 0:
            raise ValueError(
                f"log_uniform requires strictly positive low and high, "
                f"got low={low}, high={high}"
            )
        return cls(
            distribution=DistributionKind.LOG_UNIFORM, low=low, high=high, **kwargs
        )

    @classmethod
    def ramp(cls, low: float = 0.0, high: float = 1.0, **kwargs) -> "StimuliSpec":
        """Continuous linear sweep from *low* to *high* across the full tensor."""
        return cls(distribution=DistributionKind.RAMP, low=low, high=high, **kwargs)

    @classmethod
    def gaussian_linspace(
        cls, mean: float = 0.0, std: float = 1.0, **kwargs
    ) -> "StimuliSpec":
        """
        Deterministic sweep through a Gaussian: values spaced by inverting
        the normal CDF (no randomness, fully determined by mean/std/size).
        """
        return cls(
            distribution=DistributionKind.GAUSSIAN_LINSPACE,
            mean=mean,
            std=std,
            **kwargs,
        )

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
        return cls(
            distribution=DistributionKind.LOG_UNIFORM_LINSPACE,
            low=low,
            high=high,
            **kwargs,
        )

    @classmethod
    def ulp_sweep(cls, low: float, high: float, **kwargs) -> "StimuliSpec":
        """Exhaustive 1-ULP sweep of all representable values in [low, high].

        Enumerates every finite representable value for the target format,
        sorted and deduplicated, padding with zeros to fill the tensor.
        Only Float16_b and Float16 formats are supported.

        When used as spec_A, generate_stimuli auto-sizes input_dimensions_A
        and mirrors it to B if spec_B is omitted.  Explicit spec_B is not
        auto-sized — the caller must supply compatible input_dimensions_B.
        """
        return cls(
            distribution=DistributionKind.ULP_SWEEP, low=low, high=high, **kwargs
        )
