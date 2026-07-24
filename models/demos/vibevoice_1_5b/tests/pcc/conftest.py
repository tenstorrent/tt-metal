# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared PCC-test harness shims for `coqui/XTTS-v2` bring-up.

The generated per-component tests (`test_<component>.py`) call a handful of
helpers as *bare names* — most notably ``_captured_submodule_path`` — that the
test template expects to exist in scope but never defines. Because the tests
reference these by bare name (not via import), the shim is published into
``builtins`` here so every test module resolves the name uniformly, without
having to hand-patch all ~29 generated tests.

pytest imports the nearest ``conftest.py`` before collecting the test modules,
so these builtins are in place by the time any test body runs.
"""

from __future__ import annotations

import builtins
import json
import os

# tests/pcc/conftest.py  ->  model root (…/models/demos/vibevoice_1_5b) holds
# _captured/<component>/manifest.json produced by the capture-inputs step.
_HERE = os.path.dirname(os.path.abspath(__file__))
_CAPTURED_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", "_captured"))


def _captured_positional_tensor_shapes(component_name):
    """Return the list of captured positional-arg tensor shapes for a component.

    Reads ``_captured/<component>/manifest.json`` and returns one entry per
    positional arg: a ``tuple`` shape for tensor args, ``None`` for non-tensor
    args (or ``[]`` if the manifest is missing). The per-component tests use
    this so their synthetic ``_make_arg_for`` inputs match the module's REAL
    captured ranks/shapes (e.g. channels-first ``(1, C, T)`` conv/norm inputs)
    instead of the generic ``_detect_hidden_shape`` guess, which otherwise
    SKIPs the test on a shape/weight mismatch.
    """
    manifest = os.path.join(_CAPTURED_ROOT, str(component_name), "manifest.json")
    try:
        with open(manifest) as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return []
    items = (data.get("args") or {}).get("items") or []
    shapes = []
    for entry in items:
        if isinstance(entry, dict) and entry.get("kind") == "tensor":
            shape = entry.get("shape")
            shapes.append(tuple(shape) if isinstance(shape, list) and shape else None)
        else:
            shapes.append(None)
    return shapes


def _load_captured_inputs(component_name):
    """Load the real captured `(args, kwargs)` for a component, or `(None, None)`.

    Reads the `args.pt` / `kwargs.pt` that the capture-inputs step recorded next
    to the manifest. Replaying these exact inputs is the robust way to drive a
    module whose forward takes interdependent args that can't be safely
    synthesized (matching token-id ranges/lengths, required non-default flags
    like `return_latent=True`, etc.). Tensors come back on CPU.
    """
    import torch

    base = os.path.join(_CAPTURED_ROOT, str(component_name))

    def _load(path):
        if not os.path.isfile(path):
            return None
        for kw in ({"weights_only": False}, {}):
            try:
                return torch.load(path, map_location="cpu", **kw)
            except Exception:
                continue
        return None

    return _load(os.path.join(base, "args.pt")), _load(os.path.join(base, "kwargs.pt"))


def _captured_submodule_path(component_name):
    """Return the ``submodule_path`` recorded in the capture manifest for
    ``component_name``, or ``None`` if no manifest / field is available.

    Capture and the PCC test must resolve to the SAME torch submodule, or the
    captured args won't fit the test-resolved module's forward signature. The
    generated test tries this path FIRST (BUG-2 fix) and only falls back to
    ``_CANDIDATE_SUBMODULE_PATHS`` when the manifest has nothing usable.
    """
    manifest = os.path.join(_CAPTURED_ROOT, str(component_name), "manifest.json")
    try:
        with open(manifest) as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    path = data.get("submodule_path")
    if isinstance(path, str) and path:
        return path
    return None


# Publish as a builtin so the bare-name references in the generated tests
# resolve. Idempotent: re-importing conftest just re-binds the same helper.
builtins._captured_submodule_path = _captured_submodule_path
builtins._captured_positional_tensor_shapes = _captured_positional_tensor_shapes
builtins._load_captured_inputs = _load_captured_inputs
