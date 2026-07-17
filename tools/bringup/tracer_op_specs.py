#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared op specifications for the bringup tracer tooling.

Single source of truth for the op kinds that the Phase-1 tracer records and the
metadata needed to validate and replay them. Both the manifest validator
(``trace_manifest_validation.py``) and the runtime harness
(``tracer_test_harness.py``) import this module, so op knowledge lives in one
place instead of being duplicated across the preflight validator and the device
replay path.

Kept dependency-light on purpose: stdlib only (no ``torch`` / ``ttnn``), so the
validator can import it without pulling heavy runtime dependencies, and so it is
cheap for the harness to import as a sibling module.

The recorded kinds mirror the "interesting" leaf modules hooked by
``phase1_record_ops.py``. ``runnable`` marks the subset the harness can actually
replay on device today (the rest are valid to appear in a manifest but skipped).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class OpSpec:
    """Static description of a traced op kind.

    Attributes:
        kind: The module type name recorded by the tracer (e.g. ``"Conv2d"``).
        required_params: ``params`` keys the harness dereferences to replay the
            op; the validator checks these are present.
        uses_weight: Whether the op carries a weight artifact (``w_path``).
        uses_bias: Whether the op carries a bias artifact (``b_path``).
        runnable: Whether ``tracer_test_harness`` can replay this kind on device.
    """

    kind: str
    required_params: Tuple[str, ...] = ()
    uses_weight: bool = False
    uses_bias: bool = False
    runnable: bool = False


# Params a Conv2d replay dereferences (tracer_test_harness.run_record).
CONV2D_REQUIRED_PARAMS: Tuple[str, ...] = (
    "in_channels",
    "out_channels",
    "kernel_size",
    "stride",
    "padding",
    "dilation",
    "groups",
)


# Registry of all kinds the Phase-1 tracer records (phase1_record_ops.py, the
# "interesting" module tuple). The harness only replays the ``runnable`` subset.
OP_SPECS: Dict[str, OpSpec] = {
    "Conv2d": OpSpec(
        kind="Conv2d",
        required_params=CONV2D_REQUIRED_PARAMS,
        uses_weight=True,
        uses_bias=True,
        runnable=True,
    ),
    "ConvTranspose2d": OpSpec(kind="ConvTranspose2d", uses_weight=True, uses_bias=True),
    "GroupNorm": OpSpec(kind="GroupNorm", uses_weight=True, uses_bias=True),
    "BatchNorm2d": OpSpec(kind="BatchNorm2d", uses_weight=True, uses_bias=True),
    "ReLU": OpSpec(kind="ReLU", runnable=True),
    "MaxPool2d": OpSpec(kind="MaxPool2d"),
    "Upsample": OpSpec(kind="Upsample"),
}

# Kinds valid to appear in a manifest (superset of the runnable kinds).
SUPPORTED_KINDS = frozenset(OP_SPECS)


def get_spec(kind: str) -> Optional[OpSpec]:
    """Return the ``OpSpec`` for ``kind``, or ``None`` if unsupported."""
    return OP_SPECS.get(kind)


def is_supported(kind: str) -> bool:
    """Whether ``kind`` is a recognized (recordable) op kind."""
    return kind in OP_SPECS


def is_runnable(kind: str) -> bool:
    """Whether the harness can replay ``kind`` on device."""
    spec = OP_SPECS.get(kind)
    return bool(spec and spec.runnable)


def required_params(kind: str) -> Tuple[str, ...]:
    """Return the required ``params`` keys for ``kind`` (empty if none/unknown)."""
    spec = OP_SPECS.get(kind)
    return spec.required_params if spec is not None else ()
