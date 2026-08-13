# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared harness for the generated per-component PCC tests of
`/localdev/lserbedzija/hf_models/voxtral-tts-backbone`.

Everything here is a TEST-HARNESS fix that is common to all five generated
`test_<component>.py` modules, so it lives once in this conftest instead of
being copy-pasted (and drifting) per component:

1. `_captured_submodule_path` / `_maybe_load_captured` — the generated tests
   CALL `_captured_submodule_path(COMPONENT_NAME)` in `_build_torch_reference`
   but the scaffolder never injected its definition
   (`capture_inputs.CAPTURE_LOADER_SOURCE`), so every test died with
   `NameError: name '_captured_submodule_path' is not defined` before it could
   build a torch reference. Re-published here verbatim-equivalent to the
   scaffolder's source.

2. `_make_arg_for` — the template's synthetic side-inputs are degenerate for
   this Mistral decoder and would pin a meaningless golden:
     * `position_embeddings` was `(cos=ones, sin=zeros)`, i.e. RoPE reduced to
       the identity, so a stub that skips rotary entirely would still score
       PCC 1.0. We now build the REAL (cos, sin) from the model's own
       `rotary_emb` at positions `arange(seq_len)`.
     * `position_ids` was all-zeros — same vacuum for the rotary component.
       Now `arange(seq_len)`.
     * `attention_mask` was `ones(1, seq)` of dtype long. HF Mistral ADDS the
       mask to the attention scores, so `ones` is a uniform +1 shift that
       softmax cancels: the golden was silently NON-causal. We now build the
       additive 4D causal mask `(1, 1, seq, seq)` that `MistralModel` really
       feeds its layers, and register `attention_mask` as a well-known input so
       the components whose signature defaults it to `None` get it too.

3. `_marshal_side_inputs` — the test hands the primary input to the stub as a
   ttnn tensor but every OTHER kwarg as a raw `torch.Tensor`. The stub's
   forward runs inside `models/common/native_probe.run_native_probe`, which
   fails a stub that executes ANY torch op (`ttnn.from_torch` alone costs two
   `__dlpack__` calls), so a stub that marshalled its own side-inputs could
   never graduate no matter how native its math is. Marshalling happens here,
   OUTSIDE the probe, which is where input staging belongs.

None of this touches the PCC assertion or its 0.99 target.
"""

from __future__ import annotations

import pytest


# --------------------------------------------------------------------------
# 1. capture-manifest helpers the scaffolder failed to inject
# --------------------------------------------------------------------------
def _captured_submodule_path(component_name):
    """Read the submodule_path the capture step hooked when it saved inputs
    for this component. Returns the path string or ``None``.

    Capture and test MUST resolve the same submodule or the recorded shapes
    don't describe what the test actually exercises."""
    import json as _json
    import re as _re
    from pathlib import Path as _Path

    safe = _re.sub(r"[^A-Za-z0-9_]+", "_", component_name).strip("_").lower() or "component"
    demo_dir = _Path(__file__).resolve().parents[2]
    manifest_p = demo_dir / "_captured" / safe / "manifest.json"
    if not manifest_p.is_file():
        return None
    try:
        data = _json.loads(manifest_p.read_text())
        path = data.get("submodule_path")
        if isinstance(path, str) and path:
            return path
    except Exception:
        pass
    return None


def _maybe_load_captured(component_name):
    """Load `(args, kwargs, output)` from `<demo_dir>/_captured/<safe>/` if the
    planner's capture-inputs step produced them; `None` otherwise."""
    import re as _re
    from pathlib import Path as _Path

    safe = _re.sub(r"[^A-Za-z0-9_]+", "_", component_name).strip("_").lower() or "component"
    comp_dir = _Path(__file__).resolve().parents[2] / "_captured" / safe
    if not comp_dir.is_dir():
        return None
    args_p, kwargs_p, output_p = (comp_dir / "args.pt", comp_dir / "kwargs.pt", comp_dir / "output.pt")
    if not (args_p.is_file() and kwargs_p.is_file() and output_p.is_file()):
        return None
    try:
        import torch as _torch

        args = _torch.load(args_p, map_location="cpu", weights_only=False)
        kwargs = _torch.load(kwargs_p, map_location="cpu", weights_only=False)
        output = _torch.load(output_p, map_location="cpu", weights_only=False)
        print(f"[bringup] using captured inputs from {comp_dir}", flush=True)
        return args, kwargs, output
    except Exception as _e:
        print(f"[bringup] captured-inputs load failed for {component_name}: {_e}", flush=True)
        return None


# --------------------------------------------------------------------------
# 2. non-degenerate synthetic side-inputs
# --------------------------------------------------------------------------
def _seq_len_for(mod, torch_module, model):
    """The sequence length the module's `hidden_states` will be built with, so
    every other per-position input (mask, position_ids, cos/sin) agrees."""
    try:
        shape, kind = mod._detect_hidden_shape(torch_module, model=model)
        if kind == "nlc" and len(shape) == 3:
            return int(shape[1])
    except Exception:
        pass
    return 64


def _find_rotary(model):
    """The model's own RotaryEmbedding module, wherever it hangs."""
    for holder in (model, getattr(model, "model", None)):
        if holder is None:
            continue
        for name in ("rotary_emb", "rotary_embedding", "rope"):
            rot = getattr(holder, name, None)
            if rot is not None and callable(rot):
                return rot
    return None


def _real_cos_sin(model, seq_len, dtype):
    """(cos, sin) from the model's real RoPE at positions 0..seq_len-1.

    Falls back to `(ones, zeros)` — identity RoPE — only if the model exposes
    no rotary module, and says so loudly, because that fallback makes the
    rotary half of an attention PCC test vacuous."""
    import torch

    rot = _find_rotary(model)
    cfg = getattr(model, "config", None)
    h = getattr(cfg, "hidden_size", 4096) if cfg is not None else 4096
    n_heads = getattr(cfg, "num_attention_heads", 32) if cfg is not None else 32
    head_dim = getattr(cfg, "head_dim", None) or (h // n_heads)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    if rot is not None:
        try:
            with torch.no_grad():
                cos, sin = rot(torch.zeros(1, seq_len, head_dim, dtype=dtype), position_ids)
            return cos.to(dtype), sin.to(dtype)
        except Exception as exc:
            print(f"[bringup] real RoPE build failed ({type(exc).__name__}: {exc}); using identity cos/sin", flush=True)
    else:
        print("[bringup] model exposes no rotary module; using identity cos/sin", flush=True)
    return torch.ones(1, seq_len, head_dim, dtype=dtype), torch.zeros(1, seq_len, head_dim, dtype=dtype)


def _causal_mask(seq_len, dtype):
    """The additive causal mask HF Mistral really feeds its decoder layers:
    0 where a key may be attended, a large negative where it may not."""
    import torch

    blocked = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    mask = torch.zeros(seq_len, seq_len, dtype=dtype)
    mask.masked_fill_(blocked, -1.0e9)
    return mask.view(1, 1, seq_len, seq_len)


def _install_make_arg_for(mod):
    """Wrap the test module's `_make_arg_for` so the per-position inputs are
    real. Anything we don't special-case falls through to the template."""
    orig = getattr(mod, "_make_arg_for", None)
    if orig is None or getattr(orig, "_bringup_shared", False):
        return

    def _shared(arg_name, *, model, torch_module):
        import torch

        seq_len = _seq_len_for(mod, torch_module, model)
        dtype = torch.float32
        if arg_name == "position_embeddings":
            return _real_cos_sin(model, seq_len, dtype)
        if arg_name == "attention_mask":
            return _causal_mask(seq_len, dtype)
        if arg_name == "position_ids":
            return torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        return orig(arg_name, model=model, torch_module=torch_module)

    _shared._bringup_shared = True
    mod._make_arg_for = _shared


# --------------------------------------------------------------------------
# 3. side-input marshalling (must run OUTSIDE the native probe)
# --------------------------------------------------------------------------
def _marshal_side_inputs(kwargs, device):
    """Stage every non-primary tensor kwarg onto `device` as a ttnn tensor.

    Float tensors become bfloat16 TILE_LAYOUT ttnn tensors (replicated on a
    mesh); tuples/lists are mapped element-wise so `position_embeddings`
    arrives as a `(cos, sin)` pair of ttnn tensors. Integer tensors are staged
    as float32 — the device has no int64, `position_ids` is exactly
    representable in float32, and its only device-side use is as a numeric
    operand (the RoPE outer product). Plain python values pass through."""
    import torch

    import ttnn

    def _is_mesh(dev):
        try:
            if isinstance(dev, ttnn.MeshDevice):
                return True
        except AttributeError:
            pass
        return hasattr(dev, "get_device_ids") or hasattr(dev, "get_devices")

    def _to_device(t, dtype):
        if _is_mesh(device):
            try:
                return ttnn.from_torch(
                    t,
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(device),
                )
            except (AttributeError, TypeError):
                pass
        return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    def _one(value):
        if isinstance(value, torch.Tensor):
            if value.is_floating_point():
                return _to_device(value.to(torch.bfloat16), ttnn.bfloat16)
            return _to_device(value.to(torch.float32), ttnn.float32)
        if isinstance(value, tuple):
            return tuple(_one(v) for v in value)
        if isinstance(value, list):
            return [_one(v) for v in value]
        return value

    return {k: _one(v) for k, v in (kwargs or {}).items()}


# --------------------------------------------------------------------------
# wiring: publish the helpers into each generated test module
# --------------------------------------------------------------------------
_EXTRA_WELL_KNOWN = ("attention_mask",)


@pytest.hookimpl(trylast=True)
def pytest_runtest_setup(item):
    mod = getattr(item, "module", None)
    if mod is None or not getattr(mod, "COMPONENT_NAME", None):
        return
    for name, fn in (
        ("_captured_submodule_path", _captured_submodule_path),
        ("_maybe_load_captured", _maybe_load_captured),
        ("_marshal_side_inputs", _marshal_side_inputs),
    ):
        if getattr(mod, name, None) is None:
            setattr(mod, name, fn)
    well_known = getattr(mod, "_WELL_KNOWN_INPUTS", None)
    if isinstance(well_known, tuple):
        missing = tuple(n for n in _EXTRA_WELL_KNOWN if n not in well_known)
        if missing:
            mod._WELL_KNOWN_INPUTS = well_known + missing
    _install_make_arg_for(mod)
