# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The fused decode kernels by name and the ``QWEN38_FUSED`` / ``QWEN38_FUSED_OFF`` switches.

Every fused kernel stands in for one named chain of existing ttnn ops.  The kernels in ``DEFAULT_ON`` (the one list
below: proven bitwise against their chains and at the model level -- or, for a COMPONENT-class kernel such as
``gdn_step``, against the tt/ oracle at the component gate -- and faster than the record in their own timing
slot) serve by default; ``QWEN38_FUSED_OFF`` (comma-separated names, or ``all``) falls back to the composed chains,
``QWEN38_FUSED`` switches an opt-in kernel on.  A name that is not registered raises in either variable and in
``DEFAULT_ON``, so a typo cannot silently change what runs.  Callers resolve once at construction and keep the choice
through trace capture: ``run = resolve("router_tail")``.  A kernel that declares its input contract (``admits``)
is resolved with ``resolve_admitted``: the fused program serves the calls that satisfy the contract and the composed
chain the others, so a production default never raises on inputs its program was not written for.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping

ENV = "QWEN38_FUSED"
OFF_ENV = "QWEN38_FUSED_OFF"
ALL = "all"
# Tolerance classes of the component gate (fused_component_gate, dev tools): BITWISE where the arithmetic order is preserved (pure data
# movement, integer work, the same ops in the same order); ULP where the same math is re-associated or fused
# (an eltwise chain in one pass, a reduction in another order); COMPONENT where the chain's precision points move
# (fp32 state kept, a rounding removed) and only the tt/ oracle can judge the component.
BITWISE, ULP, COMPONENT = "bitwise", "ulp", "component"
TOLERANCE_CLASSES = (BITWISE, ULP, COMPONENT)
_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
# The kernels that serve by default: bitwise against their chains, the acceptance tables unchanged, and a 200-step
# traced wall under the record in their own slot (the FUSION-DEFAULTS notes).  Flipping a kernel is this one list.
DEFAULT_ON: frozenset[str] = frozenset(
    {
        "gdn_prefill_rows",
        "gdn_rows_wrap",
        "gdn_rows_scan",
        "gdn_step",
        "gr_fold",
        "gr_read",
        "gr_recip_last",
        "gr_write",
        "greedy_tail",
        "moe_combine",
        "moe_post",
        "ple",
        "position_derive",
        "qsa_index_tail",
        "qsa_main_tail",
        "qsa_rows",
        "qsa_score_pages",
        "qsa_rows_post_attention",
        "qsa_post_attention",
        "qsa_score_merge",
        "qsa_selection_row",
        "qsa_widen_partial",
        "router_tail",
        "sampler_tail",
        "candidate_row",
        "shared_expert",
        "moe_dense",
        "sparse_sdpa_tiled",
    }
)


@dataclass(frozen=True)
class GateSpec:
    """How the component gate feeds a kernel captured real inputs (``numerics_audit_device.py`` records).

    ``inputs(mesh, capture, positions, layer)`` returns the keyword arguments both callables take, as device tensors
    built from the capture's positions (one tile row per position); ``output(result)`` returns the call's result as a
    host tensor ``[rows, ...]``; ``reference(oracle, positions, layer)`` returns the oracle's value of the same
    quantity (or None when the oracle keeps none).  ``topk`` names the ranking width when the output is a ranking.
    """

    inputs: Callable[[Any, Mapping[str, Any], tuple[int, ...], int], dict[str, Any]]
    output: Callable[[Any], Any]
    reference: Callable[[Mapping[str, Any], tuple[int, ...], int], Any] | None = None
    layers: tuple[int, ...] = (0,)
    topk: int | None = None


@dataclass(frozen=True)
class FusedKernel:
    name: str
    replaces: str
    tolerance: str
    fused: Callable[..., Any]
    composed: Callable[..., Any]
    gate: GateSpec | None = None
    # A COMPONENT-class kernel may serve by default only with this recorded component-gate proof (the numbers of the
    # probe against the tt/ oracle); BITWISE defaults need none.
    component_proof: str | None = None
    # The kernel's input contract as a predicate over the call's arguments (what its program asserts before it
    # builds): a production default runs the fused program only when it holds and the composed chain otherwise
    # (``resolve_admitted``).  None: the fused callable serves every call.
    admits: Callable[..., bool] | None = None

    @property
    def default_on(self) -> bool:
        return self.name in DEFAULT_ON

    def __post_init__(self) -> None:
        if not _NAME.match(self.name) or self.name == ALL:
            raise ValueError(f"fused kernel name must match {_NAME.pattern} and not be {ALL!r}, got {self.name!r}")
        if self.tolerance not in TOLERANCE_CLASSES:
            raise ValueError(
                f"fused kernel {self.name}: tolerance must be one of {TOLERANCE_CLASSES}, got {self.tolerance!r}"
            )


_REGISTRY: dict[str, FusedKernel] = {}


def register(kernel: FusedKernel) -> FusedKernel:
    if kernel.name in _REGISTRY:
        raise ValueError(f"fused kernel {kernel.name!r} is registered twice")
    _REGISTRY[kernel.name] = kernel
    return kernel


def kernels() -> dict[str, FusedKernel]:
    return dict(_REGISTRY)


def kernel(name: str) -> FusedKernel:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"no fused kernel {name!r}; registered: {sorted(_REGISTRY)}") from None


def _names(environ: Mapping[str, str], variable: str) -> frozenset[str]:
    tokens = [t.strip() for t in environ.get(variable, "").split(",") if t.strip()]
    if ALL in tokens:
        return frozenset(_REGISTRY)
    unknown = sorted(set(tokens) - set(_REGISTRY))
    if unknown:
        raise ValueError(f"{variable} names unregistered fused kernels {unknown}; registered: {sorted(_REGISTRY)}")
    return frozenset(tokens)


def default_names() -> frozenset[str]:
    """``DEFAULT_ON``, every name of which must be registered and BITWISE, or COMPONENT with its component-gate proof."""

    unknown = sorted(DEFAULT_ON - set(_REGISTRY))
    if unknown:
        raise ValueError(f"DEFAULT_ON names unregistered fused kernels {unknown}; registered: {sorted(_REGISTRY)}")
    unproven = sorted(
        name
        for name in DEFAULT_ON
        if _REGISTRY[name].tolerance != BITWISE
        and not (_REGISTRY[name].tolerance == COMPONENT and _REGISTRY[name].component_proof)
    )
    if unproven:
        raise ValueError(f"DEFAULT_ON kernels {unproven} are neither BITWISE nor COMPONENT with a component-gate proof")
    return DEFAULT_ON


def enabled_names(environ: Mapping[str, str] = os.environ) -> frozenset[str]:
    """The kernels that run: ``DEFAULT_ON`` plus ``QWEN38_FUSED``, less ``QWEN38_FUSED_OFF``; every name in either
    variable must be registered."""

    return (default_names() | _names(environ, ENV)) - _names(environ, OFF_ENV)


def enabled(name: str, environ: Mapping[str, str] = os.environ) -> bool:
    kernel(name)
    return name in enabled_names(environ)


def resolve(name: str, environ: Mapping[str, str] = os.environ) -> Callable[..., Any]:
    """The kernel's fused callable when it runs (see ``enabled_names``), else its composed chain."""

    entry = kernel(name)
    return entry.fused if name in enabled_names(environ) else entry.composed


class AdmittedStep:
    """A kernel that runs, dispatched per call: the fused program when ``kernel.admits`` holds for the call's
    arguments, the composed chain otherwise.  The test is a few shape reads, and a traced capture keeps whichever branch
    its real tensors took (the served tensors satisfy the contract; host fakes outside it take the chain)."""

    def __init__(self, entry: FusedKernel) -> None:
        if entry.admits is None:
            raise ValueError(f"fused kernel {entry.name!r} has no admission predicate")
        self.kernel = entry
        self.fused = entry.fused
        self.composed = entry.composed
        self.admits = entry.admits

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.admits(*args, **kwargs):
            return self.fused(*args, **kwargs)
        return self.composed(*args, **kwargs)


def resolve_admitted(name: str, environ: Mapping[str, str] = os.environ) -> Callable[..., Any]:
    """``resolve`` with the kernel's admission: the composed chain when the kernel is off; the fused callable when it
    runs and declares no ``admits``; an ``AdmittedStep`` (fused within the input contract, composed outside it) when it
    does.  The production default's site resolves here."""

    entry = kernel(name)
    if name not in enabled_names(environ):
        return entry.composed
    if entry.admits is None:
        return entry.fused
    return AdmittedStep(entry)
