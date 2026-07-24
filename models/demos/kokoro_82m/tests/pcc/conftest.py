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

# tests/pcc/conftest.py  ->  model root (…/models/demos/kokoro_82m) holds
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


def _captured_reference_inputs(torch_module, component_name):
    """Build ``(kwargs, primary)`` from the REAL captured inputs, or ``None``.

    Many Kokoro submodules take interdependent positional args (e.g. AdaIN1d's
    ``forward(x, s)`` where ``x`` is a channels-first ``[B, C, T]`` activation
    and ``s`` is a ``[B, style_dim]`` style vector) whose shapes cannot be
    synthesized by the generic ``_detect_hidden_shape`` guess — so the HF
    reference forward raises and the PCC test SKIPs.

    When ``_captured/<component>/`` recorded the real ``args.pt`` / ``kwargs.pt``,
    this maps the captured positional tuple onto ``torch_module.forward``'s
    parameter names (and merges captured kwargs), then picks the first tensor
    arg as the primary. Returns ``None`` when nothing usable was captured so the
    caller falls back to the synthetic ``_make_arg_for`` path.
    """
    import inspect

    import torch

    cap_args, cap_kwargs = _load_captured_inputs(component_name)
    if cap_args is None and not cap_kwargs:
        return None

    if isinstance(cap_args, (list, tuple)):
        cap_args = tuple(cap_args)
    elif cap_args is None:
        cap_args = ()
    else:
        cap_args = (cap_args,)
    cap_kwargs = cap_kwargs if isinstance(cap_kwargs, dict) else {}

    def _named_params(fn):
        try:
            s = inspect.signature(fn)
        except (TypeError, ValueError):
            return []
        return [
            name
            for name, param in s.parameters.items()
            if name != "self" and param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
        ]

    param_names = _named_params(torch_module.forward)
    # Some wrappers define `forward(self, *args, **kwargs)` and delegate to a
    # base class (e.g. CustomAlbert -> AlbertModel). Walk the MRO to recover the
    # real positional arg names so captured positional args can be mapped.
    if not param_names:
        for klass in type(torch_module).__mro__:
            fwd = klass.__dict__.get("forward")
            if fwd is None:
                continue
            names = _named_params(fwd)
            if names:
                param_names = names
                break

    kwargs = {}
    for i, val in enumerate(cap_args):
        if i < len(param_names):
            kwargs[param_names[i]] = val
    for k, v in cap_kwargs.items():
        kwargs[k] = v

    primary = None
    for name in param_names:
        v = kwargs.get(name)
        if isinstance(v, torch.Tensor):
            primary = (name, v)
            break
    if primary is None:
        for name, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                primary = (name, v)
                break
    if primary is None:
        return None
    return kwargs, primary


# Publish as a builtin so the bare-name references in the generated tests
# resolve. Idempotent: re-importing conftest just re-binds the same helper.
builtins._captured_submodule_path = _captured_submodule_path
builtins._captured_positional_tensor_shapes = _captured_positional_tensor_shapes
builtins._load_captured_inputs = _load_captured_inputs
builtins._captured_reference_inputs = _captured_reference_inputs


def pytest_runtest_setup(item):
    """Systemic harness fix: make every per-component PCC test drive its torch
    reference with the REAL captured inputs when they exist.

    The capture-inputs step records the exact ``(args, kwargs)`` each submodule
    was called with (``_captured/<component>/args.pt`` etc.), but the generated
    test template only ever used the captured *submodule path* — it still built
    forward inputs synthetically via ``_make_arg_for`` / ``_detect_hidden_shape``.
    For Kokoro submodules with interdependent, non-standard forward signatures
    (e.g. ``TextEncoder.forward(input_ids, input_lengths, mask)`` where the first
    arg must be a Long token-id tensor, or ``Decoder.forward(asr, F0, N, s)``
    with four differently-shaped tensors), the synthetic guess produces wrong
    dtypes/shapes, so the HF reference forward raises and the test SKIPs — a pure
    TEST-HARNESS defect, not a stub bug.

    Rather than hand-patch each of ~29 duplicated ``_build_torch_reference``
    functions, we wrap the module's builder once here: run the original (which
    still loads the model and resolves the torch submodule), then, if real
    captured inputs are available, replace the synthesized kwargs/primary with
    the captured ones. Genuinely-correct native stubs pass on the real inputs
    just as they did on synthetic ones, so already-graduated components are
    unaffected; components whose forward can't be synthesized now actually RUN.
    """
    mod = getattr(item, "module", None)
    if mod is None:
        return
    orig = getattr(mod, "_build_torch_reference", None)
    comp = getattr(mod, "COMPONENT_NAME", None)
    if orig is None or comp is None or getattr(orig, "_capture_wrapped", False):
        return

    def wrapped(_orig=orig, _comp=comp):
        torch_module, kwargs, primary = _orig()
        try:
            cap = _captured_reference_inputs(torch_module, _comp)
        except Exception:
            cap = None
        if cap is not None:
            cap_kwargs, cap_primary = cap
            print(f"[bringup] driving `{_comp}` torch reference with REAL captured inputs")
            return torch_module, cap_kwargs, cap_primary
        return torch_module, kwargs, primary

    wrapped._capture_wrapped = True
    mod._build_torch_reference = wrapped
