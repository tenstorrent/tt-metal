# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Real mathematical domains — where each activation is defined over the reals.

This is the ``valid_domain`` that ``ACTIVATION_DOMAIN_REPAIR.md`` §0.10 Step 1
asks the spec schema to carry, and it is deliberately distinct from the
``domain`` field in ``activations/*.json``. That field is the **campaign**
domain: the interval the fitter targets and the Table-2 grid scores. This module
is the **mathematical** domain: where a real-valued answer exists at all. It
also interprets contract-declared pole lattices and their typed result classes.

Why it has to exist. The reference silently returns the real part of a complex
continuation, or zero, for inputs outside the real domain — measured:

    sqrt(-4)    -> 0.0                acosh(0.5)  -> 0.0
    log(-2)     -> 0.6931 = Re(ln2 + i*pi)
    atanh(2)    -> 0.5493 = Re(atanh 2)
    logit(-0.5) -> -1.0986            rsqrt(-4)   -> 0.0

Nothing raises and nothing returns NaN, so scoring an op outside its real domain
compares the kernel against an invented target. Marking those inputs NaN is what
lets a scorer give every encoding exactly one disposition.

The contract is activation data: restricted intervals use ``valid_domain``,
while nonpositive-integer pole lattices use the common ``contract`` vocabulary.
This module interprets those shapes without activation-name dispatch.
"""

from __future__ import annotations

import functools
import json
import os
from typing import Optional

import numpy as np
from .spec_context import current_spec_root, scoped_spec_cache


ENDPOINT_RESULTS = frozenset({"neg_inf", "pos_inf", "signed_inf", "signed_zero"})
_NONPOSITIVE_INTEGER_DOMAIN = "all_real_except_nonpositive_integer_poles"
_INTEGER_POLICY_RESULTS = {
    "return_nan_class": "nan",
    "return_pos_inf": "pos_inf",
    "return_neg_inf": "neg_inf",
}
_NONFINITE_POLICY_RESULTS = dict(_INTEGER_POLICY_RESULTS)

_ACTIVATIONS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "activations",
)


@scoped_spec_cache
def _spec_domain(activation: str) -> Optional[dict]:
    """Load the mathematical domain from activation data, never from its name."""
    path = os.path.join(current_spec_root() or _ACTIVATIONS_DIR, f"{activation}.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        domain = json.load(fh).get("valid_domain")
    if domain is None:
        return None
    if not isinstance(domain, dict):
        raise ValueError(f"{activation}: valid_domain must be an object")
    allowed = {
        "min",
        "max",
        "min_exclusive",
        "max_exclusive",
        "min_result",
        "max_result",
    }
    unknown = set(domain) - allowed
    if unknown:
        raise ValueError(f"{activation}: unknown valid_domain fields: {sorted(unknown)}")
    if "min" not in domain and "max" not in domain:
        raise ValueError(f"{activation}: valid_domain must declare min or max")
    lo = float(domain.get("min", -np.inf))
    hi = float(domain.get("max", np.inf))
    if not lo < hi:
        raise ValueError(f"{activation}: valid_domain min must be below max")
    for edge in ("min", "max"):
        result_key = f"{edge}_result"
        if result_key not in domain:
            continue
        if edge not in domain:
            raise ValueError(f"{activation}: {result_key} requires {edge}")
        if domain.get(f"{edge}_exclusive", False):
            raise ValueError(f"{activation}: {result_key} requires an inclusive endpoint")
        if domain[result_key] not in ENDPOINT_RESULTS:
            raise ValueError(f"{activation}: {result_key} must be one of {sorted(ENDPOINT_RESULTS)}")
    return domain


@scoped_spec_cache
def _spec_contract(activation: str) -> dict:
    """Load the declarative mathematical contract for generic pole handling."""
    path = os.path.join(current_spec_root() or _ACTIVATIONS_DIR, f"{activation}.json")
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        contract = json.load(fh).get("contract") or {}
    if not isinstance(contract, dict):
        raise ValueError(f"{activation}: contract must be an object")
    return contract


def _nonpositive_integer_mask(x: np.ndarray) -> np.ndarray:
    return np.isfinite(x) & (x <= 0.0) & (x == np.trunc(x))


def real_domain_mask_from_declaration(declaration: dict, x) -> np.ndarray:
    """Interpret compiler-carried real-domain data without an op-name lookup."""
    if not isinstance(declaration, dict):
        raise ValueError("real-domain declaration must be an object")
    unknown = set(declaration) - {"valid_domain", "finite_domain"}
    if unknown:
        raise ValueError(f"unknown real-domain declaration fields: {sorted(unknown)}")
    x = np.asarray(x, dtype=np.float64)
    mask = np.ones(x.shape, dtype=bool)
    domain = declaration.get("valid_domain")
    if domain is not None:
        if not isinstance(domain, dict):
            raise ValueError("real-domain valid_domain must be an object or null")
        if "min" in domain:
            lo = float(domain["min"])
            mask &= (x > lo) if domain.get("min_exclusive") else (x >= lo)
        if "max" in domain:
            hi = float(domain["max"])
            mask &= (x < hi) if domain.get("max_exclusive") else (x <= hi)
    finite_domain = declaration.get("finite_domain", "all_real")
    if finite_domain == _NONPOSITIVE_INTEGER_DOMAIN:
        mask &= ~_nonpositive_integer_mask(x)
    elif finite_domain == "nonnegative":
        mask &= x >= 0.0
    elif finite_domain != "all_real":
        raise ValueError(f"unsupported finite_domain {finite_domain!r}")
    return mask


@functools.lru_cache(maxsize=None)
def _base_name(activation: str) -> str:
    """Strip lowering/variant suffixes that do not change the math domain."""
    name = (activation or "").strip().lower()
    for suffix in ("_bw", "_accurate", "_fast", "_approx"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def has_restricted_domain(activation: str) -> bool:
    """True when reference evaluation needs typed domain or singularity handling."""
    name = _base_name(activation)
    contract = _spec_contract(name)
    return (
        _spec_domain(name) is not None
        or contract.get("finite_domain") == _NONPOSITIVE_INTEGER_DOMAIN
        or "negative_integer_policy" in contract
    )


def real_domain_mask(activation: str, x) -> np.ndarray:
    """Boolean mask: True where *activation* is defined over the reals at *x*."""
    x = np.asarray(x, dtype=np.float64)
    name = _base_name(activation)

    domain = _spec_domain(name)
    mask = np.ones(x.shape, dtype=bool)
    if domain is not None:
        if "min" in domain:
            lo = float(domain["min"])
            mask &= (x > lo) if domain.get("min_exclusive") else (x >= lo)
        if "max" in domain:
            hi = float(domain["max"])
            mask &= (x < hi) if domain.get("max_exclusive") else (x <= hi)
    contract = _spec_contract(name)
    if contract.get("finite_domain") == _NONPOSITIVE_INTEGER_DOMAIN:
        mask &= ~_nonpositive_integer_mask(x)
    return mask


def real_domain_singularity_results(activation: str, x) -> np.ndarray:
    """Return data-declared pole and nonfinite result classes, or empty strings."""
    x = np.asarray(x, dtype=np.float64)
    contract = _spec_contract(_base_name(activation))
    result = np.full(x.shape, "", dtype="<U7")
    integer_policy = contract.get("negative_integer_policy")
    if integer_policy is not None:
        try:
            kind = _INTEGER_POLICY_RESULTS[str(integer_policy)]
        except KeyError as exc:
            raise ValueError(f"{activation}: unsupported negative_integer_policy {integer_policy!r}") from exc
        negative_integer = _nonpositive_integer_mask(x) & (x < 0.0)
        result[negative_integer] = kind

    for field, mask in (
        ("nan_policy", np.isnan(x)),
        ("pos_inf_policy", np.isposinf(x)),
        ("neg_inf_policy", np.isneginf(x)),
    ):
        policy = contract.get(field)
        if policy in _NONFINITE_POLICY_RESULTS:
            result[mask] = _NONFINITE_POLICY_RESULTS[policy]

    zero = x == 0.0
    zero_policy = contract.get("signed_zero_policy")
    if zero_policy == "return_pos_inf":
        result[zero] = "pos_inf"
    elif zero_policy == "positive_zero_to_neg_inf_negative_zero_to_pos_inf":
        result[zero & ~np.signbit(x)] = "neg_inf"
        result[zero & np.signbit(x)] = "pos_inf"
    return result


def real_domain_endpoint_results(activation: str, x) -> np.ndarray:
    """Return declared exact endpoint result kinds, or ``""`` off endpoints."""
    x = np.asarray(x, dtype=np.float64)
    domain = _spec_domain(_base_name(activation))
    result = np.full(x.shape, "", dtype="<U11")
    if domain is None:
        return result
    for edge in ("min", "max"):
        result_key = f"{edge}_result"
        if result_key in domain:
            result[x == float(domain[edge])] = domain[result_key]
    return result


__all__ = [
    "ENDPOINT_RESULTS",
    "real_domain_mask",
    "real_domain_endpoint_results",
    "real_domain_mask_from_declaration",
    "real_domain_singularity_results",
    "has_restricted_domain",
]
