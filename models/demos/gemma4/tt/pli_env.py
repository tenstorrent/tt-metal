# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Resolve the Gemma4 per-layer-input mechanism and speculative route."""

import os

from loguru import logger

PLI_ENV = "GEMMA4_PLI"
ROUTE_ENV = "GEMMA4_SPEC_ROUTE"
LEGACY_ROUTE_ENV = "GEMMA4_SPEC_FUSED_PLI_DEV"
ROUTES = ("auto", "host-loop", "fused-packed", "fused-batch-dim")
_warned = set()


def _warn_once(name, replacement):
    if name not in _warned:
        _warned.add(name)
        logger.warning(f"{name} is deprecated; use {replacement}")


def pli_on_device(legacy_env=None):
    """Return the selected mechanism; an explicitly set legacy site wins."""
    if legacy_env is not None and legacy_env in os.environ:
        _warn_once(legacy_env, PLI_ENV)
        return os.environ[legacy_env] == "1"
    value = os.environ.get(PLI_ENV, "device").strip().lower()
    if value not in ("device", "host"):
        raise ValueError(f"{PLI_ENV} must be 'device' or 'host', got {value!r}")
    return value == "device"


def spec_route():
    """Parse canonical and legacy route declarations and detect conflicts."""
    raw = os.environ.get(ROUTE_ENV)
    legacy = os.environ.get(LEGACY_ROUTE_ENV)
    route = "auto" if raw is None else raw.strip().lower()
    if route == "fused-batched":
        route = "fused-packed"
    if route not in ROUTES:
        raise ValueError(f"{ROUTE_ENV} must be one of {ROUTES}, got {raw!r}")
    if legacy is not None:
        _warn_once(LEGACY_ROUTE_ENV, ROUTE_ENV)
        implied = "auto" if legacy == "1" else "host-loop"
        if raw is None:
            return implied
        if route != implied and not (legacy == "1" and route in ("fused-packed", "fused-batch-dim")):
            raise ValueError(f"{ROUTE_ENV}={route} contradicts {LEGACY_ROUTE_ENV}={legacy}")
    return route


def resolve_route(route, target_has_pli, pli_device, trace=True, greedy=True):
    """Return an executable route, rejecting unsupported explicit requests."""
    if route not in ROUTES:
        raise ValueError(f"Unknown speculative route: {route!r}")
    if target_has_pli and route == "fused-batch-dim":
        raise ValueError("PLI targets cannot use fused-batch-dim")
    if route == "auto":
        if not trace or not greedy or (target_has_pli and not pli_device):
            return "host-loop"
        return "fused-packed" if target_has_pli else "fused-batch-dim"
    if route != "host-loop" and not trace:
        raise ValueError(f"{route} requires GEMMA4_SPEC_TRACE=1")
    if route != "host-loop" and not greedy:
        raise ValueError(f"{route} requires greedy decoding")
    if target_has_pli and route != "host-loop" and not pli_device:
        raise ValueError(f"{route} requires device PLI")
    return route
