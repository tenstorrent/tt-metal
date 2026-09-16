#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""High-precision ground-truth provider.

This module is the *single source of truth* for golden activation values. It
absorbs and extends the legacy ``compute_mpmath_300b_ulp`` mpmath path (which
only covered gelu / tanh / sigmoid / erf) so that **mpmath at 300-bit precision
is the PRIMARY reference for ALL activations** whose mathematics is expressible
in mpmath. For the genuinely piecewise-linear / clamp activations (relu, abs,
hardtanh, selu, …) that mpmath buys nothing for, it falls back to the FP64
golden provided by ``ground_truth.get_activation``.

Design (per the refactor plan):

1. ``reference_value(activation, x)`` returns the high-precision golden as a
   ``numpy.float64`` array, computed via mpmath at 300-bit for every activation
   that has an mpmath expression, and via FP64 otherwise. The path actually used
   is recorded and exposed per-activation through ``reference_source`` /
   ``reference_sources``.

2. A small provenanced cache (``ReferenceCache``) keyed by
   ``(activation, domain, sample-grid-hash)`` so full-domain 300-bit references
   are not recomputed on every call (§5 risk: "mpmath-300b cost"). The cache is
   an in-memory dict plus an optional on-disk ``.npz`` store. It records a
   ``timestamp`` for provenance but never *reads* wall-clock time itself — any
   timestamp is accepted as an argument so the cache stays deterministic.

3. Dependency-light: mpmath + numpy only. ``ground_truth`` is imported solely
   for the FP64 fallback and activation/domain metadata.

The mpmath expression for each activation is derived from the activation's
canonical single-expression ``sollya_expr`` in ``activations/<name>.json``
(the same one-source-of-truth domain/expression file the rest of the library
uses). Smooth activations that also carry a ``piecewise`` fitting decomposition
still expose a top-level ``sollya_expr`` describing the true function — that is
the form evaluated here, so e.g. gelu / sigmoid / silu / mish / softplus all get
the mpmath-300b path even though they are *fit* piecewise.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np

# mpmath is the primary engine. It is a hard dependency of this module (the
# whole point is the 300-bit reference), but we degrade gracefully to the FP64
# fallback for *every* activation if it is somehow unavailable.
try:
    import mpmath as mp

    MPMATH_AVAILABLE = True
except ImportError:  # pragma: no cover - mpmath is expected to be installed
    mp = None
    MPMATH_AVAILABLE = False

# The relocated golden-activation provider (the byte-identical port of the
# legacy ``ground_truth`` module) is imported ONLY for the FP64 fallback +
# activation/domain metadata (do not duplicate the
# activation list). It now lives inside this package, so ttpoly is self-contained.
from . import activations as ground_truth
from .spec_context import activation_spec_root, spec_cache_key


# ---------------------------------------------------------------------------
# Precision configuration
# ---------------------------------------------------------------------------

#: Primary reference precision in bits (mpmath-300b primary).
DEFAULT_MPMATH_PREC_BITS = 300

#: Tag values recorded in ``reference_source``.
SOURCE_MPMATH = "mpmath_300b"
SOURCE_FP64 = "fp64_fallback"


# ---------------------------------------------------------------------------
# 300-bit mpmath namespace for evaluating activation expressions
# ---------------------------------------------------------------------------


def _mp_cbrt(x):
    """Sign-correct real cube root.

    ``activations/cbrt.json`` carries ``sollya_expr == "x^(1/3)"`` which, taken
    literally, is the complex principal root for ``x < 0`` — silicon, kernel
    and the model all produce the real root (measured on Blackhole:
    cbrt(-100) = -4.64, not the principal +2.32). The spec therefore declares
    ``"golden_expr": "cbrt(x)"``, which resolves to this function in the
    mpmath namespace (and to ``np.cbrt`` in the fp64 golden's namespace).
    """
    return mp.sign(x) * mp.power(abs(x), mp.mpf(1) / 3)


def _mp_minimum(left, right):
    """NaN-propagating scalar minimum for declarative piecewise goldens."""
    if mp.isnan(left) or mp.isnan(right):
        return mp.nan
    return left if left <= right else right


def _mp_maximum(left, right):
    """NaN-propagating scalar maximum for declarative piecewise goldens."""
    if mp.isnan(left) or mp.isnan(right):
        return mp.nan
    return left if left >= right else right


# RETIRED name-keyed golden override layer — must stay EMPTY (asserted by
# tests/test_golden_strategy_migration.py). Wrong-branch fixes belong in the
# activation's spec: the config's "golden_expr" is preferred over its
# "sollya_expr" here (see _get_mpmath_expr), e.g. cbrt.json's sign-correct
# "cbrt(x)". The lookup below is kept only as an emergency escape hatch.
_MP_OVERRIDES = {}


def _build_mpmath_namespace():
    """Return the safe ``eval`` namespace mapping sollya names -> mpmath ops.

    Mirrors ``ground_truth._eval_sollya_expr``'s numpy namespace but every
    operation is an mpmath op evaluated at the active (300-bit) precision. The
    ``x`` binding is injected per-call.
    """
    if not MPMATH_AVAILABLE:
        return None

    ns: Dict[str, Callable] = {
        # elementary
        "exp": mp.exp,
        "log": mp.log,
        "sqrt": mp.sqrt,
        "abs": abs,
        "min": _mp_minimum,
        "max": _mp_maximum,
        # trig
        "sin": mp.sin,
        "cos": mp.cos,
        "tan": mp.tan,
        "asin": mp.asin,
        "acos": mp.acos,
        "atan": mp.atan,
        # hyperbolic
        "sinh": mp.sinh,
        "cosh": mp.cosh,
        "tanh": mp.tanh,
        "asinh": mp.asinh,
        "acosh": mp.acosh,
        "atanh": mp.atanh,
        # logs / exps (other bases, accurate variants)
        "log2": lambda v: mp.log(v, 2),
        "log10": lambda v: mp.log10(v),
        "log1p": lambda v: mp.log1p(v),
        "exp2": lambda v: mp.power(2, v),
        "expm1": lambda v: mp.expm1(v),
        # error / special functions
        "erf": mp.erf,
        "erfc": mp.erfc,
        "erfinv": mp.erfinv,
        # bessel (i0 / i1)
        "i0": lambda v: mp.besseli(0, v),
        "i1": lambda v: mp.besseli(1, v),
        # gamma family. loggamma is complex off the positive reals; the real
        # part is ln|Gamma| which is what lgamma denotes.
        "lgamma": lambda v: mp.re(mp.loggamma(v)),
        "digamma": mp.digamma,
        "polygamma": lambda n, v: mp.polygamma(int(n), v),
        "trigamma": lambda v: mp.polygamma(1, v),
        # cube root (sign-correct, see _mp_cbrt)
        "cbrt": _mp_cbrt,
        # constants
        "pi": mp.pi,
        "e": mp.e,
    }
    return ns


def _to_mp_expr(sollya_expr: str) -> str:
    """Translate a sollya expression string into Python-evaluable form."""
    # Sollya uses ``^`` for exponentiation; Python uses ``**``.
    return sollya_expr.replace("^", "**")


def _mp_tanhshrink_odd_series(x: float) -> float:
    """High-precision stable implementation selected by its spec key."""
    if x == 0.0:
        return math.copysign(0.0, x)
    xv = mp.mpf(x)
    return float(xv - mp.tanh(xv))


_MP_GOLDEN_IMPLS = {
    "tanhshrink_odd_series": _mp_tanhshrink_odd_series,
}


def _get_mpmath_expr(activation: str) -> Optional[str]:
    """Return the explicitly selected canonical expression, or ``None``.

    The spec's ``golden_expr`` (the numerically-correct closed form, e.g.
    cbrt.json's sign-correct ``cbrt(x)``) takes precedence; otherwise we
    deliberately read the *top-level* ``sollya_expr`` (the true closed-form of
    the function) rather than any per-piece fitting expression. Piecewise-linear
    / clamp activations have neither and therefore fall through to the FP64
    path.
    """
    config = ground_truth.load_activation_config(activation.lower())
    if not config:
        return None
    strategies = config.get("golden_strategy")
    if strategies:
        for strategy in strategies:
            if strategy == "golden_expr" and config.get("golden_expr"):
                return config["golden_expr"]
            if strategy == "sollya_expr" and config.get("sollya_expr"):
                return config["sollya_expr"]
            if strategy in {"golden_impl", "pytorch", "piecewise"}:
                return None
        return None
    return config.get("golden_expr") or config.get("sollya_expr")


# ---------------------------------------------------------------------------
# Per-activation mpmath function resolution + source classification
# ---------------------------------------------------------------------------

# Memoised: activation -> compiled mpmath evaluator (or None if not expressible).
_MP_FUNC_CACHE: Dict[str, Optional[Callable[[float], float]]] = {}
# Memoised: activation -> source tag.
_SOURCE_CACHE: Dict[str, str] = {}


def _probe_grid(lo: float, hi: float, n: int = 11):
    """A small deterministic probe grid spanning the (closed) domain."""
    if hi <= lo:
        return [float(lo)]
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]


def _make_mpmath_func(activation: str) -> Optional[Callable[[float], float]]:
    """Build a scalar ``float -> float`` mpmath-300b evaluator for *activation*.

    Returns ``None`` if the activation has no mpmath-expressible form or the
    expression fails / yields non-finite values across its domain (in which case
    the caller falls back to FP64).
    """
    if not MPMATH_AVAILABLE:
        return None

    override = _MP_OVERRIDES.get(activation.lower())
    if override is not None:
        return lambda x, _f=override: float(mp.re(_f(mp.mpf(x))))

    config = ground_truth.load_activation_config(activation.lower()) or {}
    strategies = config.get("golden_strategy") or ()
    if strategies and strategies[0] == "golden_impl":
        impl_name = config.get("golden_impl")
        impl = _MP_GOLDEN_IMPLS.get(impl_name)
        if impl is None:
            raise ValueError(
                f"activations/{activation}.json: golden_impl {impl_name!r} has no "
                "high-precision reference implementation"
            )
        return impl

    expr = _get_mpmath_expr(activation)
    if expr is None:
        return None

    py_expr = _to_mp_expr(expr)
    code = compile(py_expr, f"<mpmath:{activation}>", "eval")
    base_ns = _build_mpmath_namespace()

    def _eval_scalar(x: float) -> float:
        # Each call runs at the module-set 300-bit precision (set by the caller
        # before invoking, via mp.workprec). ``x`` is bound as a 300-bit mpf.
        ns = dict(base_ns)
        ns["x"] = mp.mpf(x)
        result = eval(code, {"__builtins__": {}}, ns)  # noqa: S307 - local-controlled expr
        # Imaginary parts can appear from e.g. loggamma; the real part is the
        # intended golden for these (matching ground_truth's torch lgamma).
        return float(mp.re(result))

    # Validate over the domain: the expression must evaluate to finite values on
    # a probe grid, otherwise we are better served by the FP64 fallback (which
    # has bespoke handling for the awkward activations).
    try:
        lo, hi = ground_truth.get_activation_domain(activation.lower())
    except KeyError:
        return None

    with mp.workprec(DEFAULT_MPMATH_PREC_BITS):
        for xv in _probe_grid(lo, hi):
            try:
                val = _eval_scalar(xv)
            except Exception:
                return None
            if not np.isfinite(val):
                return None

    return _eval_scalar


def _resolve(activation: str):
    """Resolve (and memoise) the mpmath evaluator + source tag for *activation*."""
    key = spec_cache_key(activation.lower())
    if key in _SOURCE_CACHE:
        return _MP_FUNC_CACHE[key], _SOURCE_CACHE[key]

    func = _make_mpmath_func(activation.lower())
    if func is not None:
        _MP_FUNC_CACHE[key] = func
        _SOURCE_CACHE[key] = SOURCE_MPMATH
    else:
        _MP_FUNC_CACHE[key] = None
        _SOURCE_CACHE[key] = SOURCE_FP64
    return _MP_FUNC_CACHE[key], _SOURCE_CACHE[key]


def reference_source(activation: str, *, spec_root=None) -> str:
    """Return which reference path is used for *activation*.

    One of ``"mpmath_300b"`` or ``"fp64_fallback"``.
    """
    if spec_root is not None:
        with activation_spec_root(spec_root):
            return reference_source(activation)
    _, source = _resolve(activation)
    return source


def require_mpmath_available() -> None:
    """Certification callers must not silently lose the high-precision provider."""
    if mp is None:
        raise RuntimeError("canonical reference certification requires mpmath")


def reference_sources() -> Dict[str, str]:
    """Return ``{activation: reference_source}`` for every known activation."""
    return {name: reference_source(name) for name in ground_truth.get_all_activations()}


# ---------------------------------------------------------------------------
# Provenanced cache: cache mpmath-300b as provenanced
# artifacts keyed by activation+domain+sample-grid.
# ---------------------------------------------------------------------------


def _grid_hash(x: np.ndarray) -> str:
    """Stable hash of the sample grid (and its dtype/shape)."""
    arr = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    h = hashlib.sha256()
    h.update(str(arr.shape).encode("utf-8"))
    h.update(arr.tobytes())
    return h.hexdigest()[:16]


def _domain_tag(activation: str) -> str:
    """Best-effort ``lo:hi`` domain string for cache provenance."""
    try:
        lo, hi = ground_truth.get_activation_domain(activation.lower())
        return f"{lo!r}:{hi!r}"
    except KeyError:
        return "unknown"


def _reference_identity(activation: str) -> str:
    """Hash every spec fact that can change canonical reference values."""
    config = ground_truth.load_activation_config(activation.lower()) or {}
    payload = {
        "activation": activation.lower(),
        "config": config,
        "precision_bits": DEFAULT_MPMATH_PREC_BITS,
        "source": reference_source(activation),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


class ReferenceCache:
    """In-memory + optional on-disk cache of 300-bit references.

    Keyed by ``(activation, domain, sample-grid-hash)`` so a full-domain 300-bit
    reference is computed once. Determinism: this class never calls wall-clock
    time; a ``timestamp`` may be supplied to :meth:`get` purely for provenance
    and is recorded alongside the cached array.
    """

    def __init__(self, disk_path: Optional[str] = None):
        self._mem: Dict[str, np.ndarray] = {}
        self._prov: Dict[str, dict] = {}
        self._disk_path = Path(disk_path) if disk_path else None
        if self._disk_path is not None:
            self._load_disk()

    @staticmethod
    def make_key(activation: str, x: np.ndarray) -> str:
        return f"{activation.lower()}|{_domain_tag(activation)}|" f"{_reference_identity(activation)}|{_grid_hash(x)}"

    def _load_disk(self):
        if self._disk_path is None or not self._disk_path.exists():
            return
        try:
            with np.load(self._disk_path, allow_pickle=False) as data:
                for key in data.files:
                    self._mem[key] = data[key]
        except Exception:
            # A corrupt cache must never break correctness; recompute instead.
            self._mem.clear()

    def flush(self):
        """Persist the in-memory cache to disk (no-op without a disk path)."""
        if self._disk_path is None:
            return
        self._disk_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self._disk_path, **self._mem)

    def get(
        self,
        activation: str,
        x: np.ndarray,
        compute: Callable[[], np.ndarray],
        timestamp: Optional[str] = None,
    ) -> np.ndarray:
        """Return cached reference for ``(activation, x)`` or compute + store it.

        ``compute`` is a zero-arg callable that produces the reference array;
        it is only invoked on a cache miss. ``timestamp`` (any caller-supplied
        string) is recorded for provenance; it is never derived from the clock.
        """
        key = self.make_key(activation, x)
        if key in self._mem:
            return self._mem[key]
        # np.ascontiguousarray promotes 0-d -> 1-d, which would silently change
        # scalar-input shape; asarray(order="C") preserves 0-d.
        value = np.asarray(compute(), dtype=np.float64, order="C")
        self._mem[key] = value
        self._prov[key] = {
            "activation": activation.lower(),
            "domain": _domain_tag(activation),
            "grid_hash": _grid_hash(x),
            "source": reference_source(activation),
            "reference_identity": _reference_identity(activation),
            "n": int(value.size),
            "timestamp": timestamp,
        }
        return value

    def provenance(self, activation: str, x: np.ndarray) -> Optional[dict]:
        """Return the provenance record for a cached entry, if present."""
        return self._prov.get(self.make_key(activation, x))


# Module-level default cache so repeated full-domain calls are cheap by default.
_DEFAULT_CACHE = ReferenceCache()


# ---------------------------------------------------------------------------
# The public reference_value entry point
# ---------------------------------------------------------------------------


def _compute_mpmath(func: Callable[[float], float], x: np.ndarray) -> np.ndarray:
    """Evaluate *func* (a scalar mpmath evaluator) over array *x* at 300-bit."""
    flat = np.asarray(x, dtype=np.float64).ravel()
    out = np.empty(flat.shape, dtype=np.float64)
    with mp.workprec(DEFAULT_MPMATH_PREC_BITS):
        for i, xv in enumerate(flat):
            out[i] = func(float(xv))
    return out.reshape(np.asarray(x).shape)


def _compute_fp64(activation: str, x: np.ndarray) -> np.ndarray:
    """FP64 fallback via the legacy ground_truth provider."""
    fn = ground_truth.get_activation(activation.lower())
    result = fn(np.asarray(x, dtype=np.float64))
    return np.asarray(result, dtype=np.float64)


from . import valid_domain as _valid_domain


def _apply_declared_signed_zero_policy(activation: str, x: np.ndarray, result: np.ndarray) -> np.ndarray:
    """Apply the spec's structural signed-zero contract to zero inputs.

    This both restores a sign lost by mpmath for odd/preserve functions and
    canonicalizes legacy piecewise expressions whose zero-valued branch was
    built with ``zeros_like(input)``.  An explicit schema-v3 contract takes
    precedence over inferred odd parity.  No activation name participates in
    the repair.
    """
    config = ground_truth.load_activation_config(activation.lower()) or {}
    rational = config.get("rational_parity") or {}
    contract = config.get("contract") or {}
    declared_odd = config.get("polynomial_parity") == "odd" or (
        rational.get("numerator") == "odd" and rational.get("denominator") == "even"
    )
    declared_preserve = contract.get("signed_zero_policy") == "preserve"
    declared_positive = contract.get("signed_zero_policy") == "return_positive_zero"
    if not (declared_odd or declared_preserve or declared_positive):
        return np.asarray(result, dtype=np.float64)
    out = np.asarray(result, dtype=np.float64)
    zero = (np.asarray(x, dtype=np.float64) == 0.0) & (out == 0.0)
    if not np.any(zero):
        return out
    out = np.array(out, copy=True)
    if declared_positive:
        out[zero] = 0.0
    else:
        out[zero] = np.copysign(0.0, np.asarray(x, dtype=np.float64)[zero])
    return out


def reference_value(
    activation: str,
    x,
    *,
    cache: Optional[ReferenceCache] = None,
    use_cache: bool = True,
    timestamp: Optional[str] = None,
    spec_root=None,
) -> np.ndarray:
    """Return the high-precision golden for *activation* evaluated at *x*.

    The reference is computed via mpmath at 300-bit precision for every
    activation that has an mpmath-expressible form, and via FP64 otherwise. Use
    :func:`reference_source` to discover which path a given activation takes.

    Args:
        activation: Activation name (case-insensitive), e.g. ``"gelu"``.
        x: Scalar or array-like of inputs.
        cache: Optional :class:`ReferenceCache`. Defaults to a module-level
            in-memory cache. Pass a cache with a ``disk_path`` for persistence.
        use_cache: Set ``False`` to bypass the cache entirely.
        timestamp: Optional provenance timestamp recorded on a cache miss. Never
            derived from the wall clock (keeps the path deterministic).
        spec_root: Optional packaged activation JSON directory. Missing specs
            fail closed; the repository is never used as a fallback.

    Returns:
        ``numpy.ndarray`` of ``float64`` golden values, same shape as ``x``
        (a 0-d array for scalar input).
    """
    if spec_root is not None:
        with activation_spec_root(spec_root):
            return reference_value(activation, x, cache=cache, use_cache=use_cache, timestamp=timestamp)
    x_arr = np.asarray(x, dtype=np.float64)
    func, source = _resolve(activation)

    def _raw(xs):
        if source == SOURCE_MPMATH and func is not None:
            return _compute_mpmath(func, xs)
        return _compute_fp64(activation, xs)

    def _compute():
        # Outside the real mathematical domain neither backend signals
        # invalidity, and one of them raises. mpmath continues into the complex
        # plane (log(-2) -> Re(ln2 + i*pi) = 0.6931), the fp64 helper returns 0.0
        # (sqrt(-4) -> 0.0), and rsqrt(0) raises ZeroDivisionError outright.
        # Scoring against any of those compares the kernel to an invented target,
        # so invalid inputs are never evaluated: they are NaN by construction.
        # This gives every encoding exactly one disposition
        # (ACTIVATION_DOMAIN_REPAIR.md 0.10 Step 1).
        if not _valid_domain.has_restricted_domain(activation):
            return _apply_declared_signed_zero_policy(activation, x_arr, _raw(x_arr))
        valid = _valid_domain.real_domain_mask(activation, x_arr)
        endpoint_results = _valid_domain.real_domain_endpoint_results(activation, x_arr)
        singularity_results = _valid_domain.real_domain_singularity_results(activation, x_arr)
        out = np.full(x_arr.shape, np.nan, dtype=np.float64)
        ordinary = valid & (endpoint_results == "") & (singularity_results == "")
        if ordinary.any():
            out[ordinary] = np.asarray(_raw(x_arr[ordinary]), dtype=np.float64)
        out[endpoint_results == "neg_inf"] = -np.inf
        out[endpoint_results == "pos_inf"] = np.inf
        signed_inf = endpoint_results == "signed_inf"
        out[signed_inf] = np.copysign(np.inf, x_arr[signed_inf])
        signed_zero = endpoint_results == "signed_zero"
        out[signed_zero] = np.copysign(0.0, x_arr[signed_zero])
        out[singularity_results == "neg_inf"] = -np.inf
        out[singularity_results == "pos_inf"] = np.inf
        return _apply_declared_signed_zero_policy(activation, x_arr, out)

    if not use_cache:
        return np.asarray(_compute(), dtype=np.float64)

    active_cache = cache if cache is not None else _DEFAULT_CACHE
    return active_cache.get(activation, x_arr, _compute, timestamp=timestamp)


__all__ = [
    "reference_value",
    "reference_source",
    "reference_sources",
    "ReferenceCache",
    "DEFAULT_MPMATH_PREC_BITS",
    "SOURCE_MPMATH",
    "SOURCE_FP64",
    "activation_spec_root",
    "require_mpmath_available",
]
