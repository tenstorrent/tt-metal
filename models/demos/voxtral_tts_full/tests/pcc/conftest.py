# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Harness repairs for the generated per-component PCC tests of voxtral-tts-full.

Everything here fixes a defect in `tt_hw_planner`'s test TEMPLATE, so it belongs in one
shared conftest rather than in seven generated files (which get re-emitted) and never in a
stub (the stub is not what is broken).

What is repaired
----------------
1. `_captured_submodule_path(...)` is CALLED by every generated test and DEFINED by none of
   them -> `NameError` at stage=build_torch_reference.  Published into `builtins` (an
   unresolved global in a test module falls back there) and reads `submodule_path` out of
   `_captured/<component>/manifest.json`.

2. `_make_arg_for` cannot build valid inputs for this model.  The reference is FUNCTIONAL
   (see modeling_voxtral_tts.py) and its module `forward`s take the reference's own
   arguments, not HF's canonical names:

     * `attention` / `decoder_layer` take `cis` -- a COMPLEX RoPE table [S, head_dim/2],
       required, which the template's introspection fallback would hand a
       `randn(1, 64, 3072)`.
     * `bias` (the additive causal mask) defaults to None and is not "well known", so it was
       dropped -- silently making the golden NON-CAUSAL.  It is added to the well-known set
       and built as the reference's own `causal_bias`.
     * `flow_matching` wants `llm_hidden` [B, 3072] (2-D), not the template's 3-D activation,
       and draws `torch.randn` for `x_0` when it is None -- so golden and port would
       integrate different noise.  `x_0` is added to the well-known set and built from a
       fixed generator; it is also persisted next to the capture so the port can stage the
       SAME tensor at build time (see 3).
     * `codec_decoder` wants integer `codes` [T, 37]; the fallback would hand it a float
       activation.

3. Side inputs must not force a torch op inside the probed forward.
   `models/common/native_probe.py` graduates only on `torch_ops == 0`, and `ttnn.from_torch`
   itself counts (it shows up as `__dlpack__`), so a stub may not marshal `cis` / `bias` /
   `x_0` per call.  All three are deterministic given the sequence length, so the stubs
   rebuild them in `build()` (which is NOT probed) and ignore the passed values.  The one
   input that cannot be recomputed -- flow's Gaussian `x_0` -- is written here to
   `_captured/flow_matching/x_0.pt` for `build()` to load.

4. `_ttnn_from_torch_mesh_safe` stages the primary input as bfloat16 unconditionally.
   For an INDEX tensor that is destructive (bf16 is exact only to 256, so codebook ids round:
   8191 -> 8192), and for an activation it needlessly injects 4e-3 of input error the port
   then gets blamed for.  Replaced with: integer -> uint32/ROW_MAJOR (which is also what
   `ttnn.embedding` wants), float -> float32/TILE.

NOTE the assertion itself is untouched: `PCC_TARGET = 0.99` and the single `assert ok` in each
generated test are exactly as emitted.
"""

from __future__ import annotations

import builtins
import json
import math
import pathlib

import pytest
import torch

import ttnn

_DEMO = pathlib.Path(__file__).resolve().parents[2]
_CAPTURED = _DEMO / "_captured"

# --- model constants (voxtral_common_ref.py; duplicated so the conftest needs no model load)
HEAD_DIM = 128
ROPE_THETA = 1_000_000.0
DIM = 3072
N_ACOUSTIC_CODEBOOK = 36
NUM_CODEBOOKS = 37
ACOUSTIC_CODEBOOK_SIZE = 21
SEMANTIC_CODEBOOK_SIZE = 8192
N_AUDIO_SPECIAL = 2

SEQ_LEN = 64  # what `_detect_hidden_shape` gives every activation; 2 full tiles
CODEC_FRAMES = 8  # matches _captured/codec_decoder (8 frames -> 15360 samples)

_X0_SEED = 12345
_X0_PATH = _CAPTURED / "flow_matching" / "x_0.pt"


# --------------------------------------------------------------------------- defect 1
def _captured_submodule_path(component):
    """`submodule_path` recorded by the capture step, so the test resolves the SAME module."""
    man = _CAPTURED / str(component) / "manifest.json"
    try:
        return json.loads(man.read_text()).get("submodule_path")
    except Exception:  # noqa: BLE001 - a missing manifest just means "no hint"
        return None


builtins._captured_submodule_path = _captured_submodule_path


# --------------------------------------------------------------------------- reference inputs
def rope_cis(seq_len, head_dim=HEAD_DIM, theta=ROPE_THETA, offset=0):
    """`voxtral_common_ref.rope_cis` verbatim -- Mistral-native INTERLEAVED complex table."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(offset, offset + seq_len).float()
    return torch.polar(torch.ones(seq_len, freqs.shape[0]), torch.outer(t, freqs))


def causal_bias(seq_len, dtype=torch.float32):
    """`voxtral_common_ref.causal_bias` verbatim -- additive 0 / -inf mask [1, 1, S, S]."""
    m = torch.full((seq_len, seq_len), float("-inf"), dtype=dtype)
    return torch.triu(m, diagonal=1).view(1, 1, seq_len, seq_len)


def flow_x_0(batch=1, seed=_X0_SEED):
    """The deterministic noise start for Block 2.  Written to disk as well: the port cannot
    regenerate a noise draw inside a probed forward, so `build()` loads this exact tensor."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, N_ACOUSTIC_CODEBOOK, generator=g)


def codec_codes(n_frames=CODEC_FRAMES, seed=7):
    """Block 2-shaped frames [T, 37] WITH the special-token offset, as `codec.forward` expects.

    Codebook 0 is kept clear of [END_AUDIO]=1 so `strip_offset_and_trim` keeps all T frames
    (its cut point is host-side generation control, not arithmetic the port must reproduce)."""
    g = torch.Generator().manual_seed(seed)
    sem = torch.randint(0, SEMANTIC_CODEBOOK_SIZE, (n_frames, 1), generator=g)
    ac = torch.randint(0, ACOUSTIC_CODEBOOK_SIZE, (n_frames, NUM_CODEBOOKS - 1), generator=g)
    return torch.cat([sem, ac], dim=1) + N_AUDIO_SPECIAL


_MISS = object()


def _component_arg(component, arg_name, *, model, torch_module):
    """Model-specific inputs.  `_MISS` falls through to the template's own `_make_arg_for`."""
    if arg_name in ("h", "x", "inputs_embeds", "hidden_states"):
        return torch.randn(1, SEQ_LEN, DIM)
    if arg_name == "cis":
        return rope_cis(SEQ_LEN)
    if arg_name == "bias":
        return causal_bias(SEQ_LEN)
    if arg_name == "llm_hidden":
        # [B, 3072] -- `semantic_code`/`decode_frame` index batch off dim 0 and would raise on
        # the template's 3-D activation.  Same distribution the reference's own
        # `make_synthetic_inputs` uses.
        g = torch.Generator().manual_seed(3)
        return torch.randn(1, DIM, generator=g)
    if arg_name == "x_0":
        return flow_x_0()
    if arg_name == "codes":
        return codec_codes()
    return _MISS


# --------------------------------------------------------------------------- defect 4
def _is_mesh(device):
    try:
        if isinstance(device, ttnn.MeshDevice):
            return True
    except AttributeError:
        pass
    return hasattr(device, "get_device_ids") or hasattr(device, "get_devices")


def _stage(tensor, device, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    """Stage the primary input HONESTLY: indices stay integral, activations stay float32."""
    if isinstance(tensor, torch.Tensor) and not tensor.is_floating_point():
        t, dt, lay = tensor.to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
    else:
        t, dt, lay = tensor.to(torch.float32), ttnn.float32, ttnn.TILE_LAYOUT
    if _is_mesh(device):
        try:
            return ttnn.from_torch(t, dtype=dt, layout=lay, device=device,
                                   mesh_mapper=ttnn.ReplicateTensorToMesh(device))
        except (AttributeError, TypeError):
            pass
    return ttnn.from_torch(t, dtype=dt, layout=lay, device=device)


# --------------------------------------------------------------------------- wiring
_EXTRA_WELL_KNOWN = ("cis", "bias", "x_0", "codes", "llm_hidden", "h")


@pytest.fixture(autouse=True)
def _repair_generated_test(request):
    """Patch the generated module in place.  Runs before the test body calls
    `_build_torch_reference()`, so the replacements are live when inputs are built."""
    mod = getattr(request, "module", None)
    if mod is None or not hasattr(mod, "COMPONENT_NAME"):
        yield
        return

    component = mod.COMPONENT_NAME
    saved = {}

    def _save(name):
        saved[name] = getattr(mod, name, _MISS)

    for name in ("_make_arg_for", "_WELL_KNOWN_INPUTS", "_ttnn_from_torch_mesh_safe"):
        _save(name)

    template_make_arg = saved["_make_arg_for"]

    def _make_arg_for(arg_name, *, model, torch_module):
        value = _component_arg(component, arg_name, model=model, torch_module=torch_module)
        if value is not _MISS:
            return value
        return template_make_arg(arg_name, model=model, torch_module=torch_module)

    mod._make_arg_for = _make_arg_for
    mod._WELL_KNOWN_INPUTS = tuple(saved["_WELL_KNOWN_INPUTS"]) + _EXTRA_WELL_KNOWN
    mod._ttnn_from_torch_mesh_safe = _stage

    # The one input a probed forward cannot rebuild for itself.
    if component == "flow_matching":
        _X0_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(flow_x_0(), _X0_PATH)

    try:
        yield
    finally:
        for name, value in saved.items():
            if value is _MISS:
                delattr(mod, name)
            else:
                setattr(mod, name, value)
