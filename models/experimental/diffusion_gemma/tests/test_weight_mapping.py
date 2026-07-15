# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DiffusionGemma -> Gemma-4 weight remap + self-conditioning loader tests (#47461).

Two tiers:
  * **Index-only** (needs only the two ``model.safetensors.index.json`` files, ~100 KB
    each): proves the prefix remap maps the DiffusionGemma text backbone EXACTLY
    onto the gemma4 ``model.language_model.*`` key set the in-repo loader expects,
    isolates the net-new self-conditioning keys, and ignores encoder/vision. This is
    the "catch missing/renamed weight keys" gate.
  * **Checkpoint** (needs the gated 51 GB DiffusionGemma checkpoint): loads the real
    self-conditioning tensors and validates shapes against the config + a forward.

Both tiers discover files from the HF cache and skip cleanly when absent.
"""

import glob
import json
import os

import pytest
import torch

from models.experimental.diffusion_gemma.config import TextConfig
from models.experimental.diffusion_gemma.reference.self_conditioning import SelfConditioning
from models.experimental.diffusion_gemma.weight_mapping import (
    SELF_CONDITIONING_PREFIX,
    classify_keys,
    expected_self_conditioning_shapes,
    gemma4_key_for,
    remap_state_dict,
)

HF_CACHE = os.path.expanduser("~/.cache/huggingface/hub")


def _index_path(repo_dirname: str):
    hits = glob.glob(os.path.join(HF_CACHE, repo_dirname, "snapshots", "*", "model.safetensors.index.json"))
    return hits[0] if hits else None


def _weight_map(repo_dirname: str):
    p = _index_path(repo_dirname)
    if p is None:
        return None
    return json.load(open(p))["weight_map"]


DG_DIR = "models--google--diffusiongemma-26B-A4B-it"
G4_DIR = "models--google--gemma-4-26B-A4B-it"


# ---------------------------------------------------------------------------
# unit (no files)
# ---------------------------------------------------------------------------
def test_gemma4_key_for_prefix_swap_and_self_cond_is_none():
    assert gemma4_key_for("model.decoder.layers.5.self_attn.q_proj.weight") == (
        "model.language_model.layers.5.self_attn.q_proj.weight"
    )
    assert gemma4_key_for("model.decoder.embed_tokens.weight") == "model.language_model.embed_tokens.weight"
    assert gemma4_key_for("model.decoder.norm.weight") == "model.language_model.norm.weight"
    assert gemma4_key_for("model.decoder.self_conditioning.gate_proj.weight") is None
    assert gemma4_key_for("model.encoder.vision_tower.std_bias") is None


def test_expected_self_conditioning_shapes_use_intermediate_not_moe():
    tc = TextConfig()
    shapes = expected_self_conditioning_shapes(tc.hidden_size, tc.intermediate_size)
    assert shapes["gate_proj.weight"] == (2112, 2816)  # intermediate_size, NOT moe_intermediate_size(704)
    assert shapes["down_proj.weight"] == (2816, 2112)
    assert shapes["pre_norm.weight"] == (2816,)


# ---------------------------------------------------------------------------
# index-only: real key coverage
# ---------------------------------------------------------------------------
@pytest.mark.skipif(_weight_map(DG_DIR) is None, reason="DiffusionGemma index.json not in HF cache")
def test_self_conditioning_keys_are_exactly_the_four():
    dg = _weight_map(DG_DIR)
    sc = sorted(k for k in dg if k.startswith(SELF_CONDITIONING_PREFIX))
    assert sc == sorted(
        SELF_CONDITIONING_PREFIX + w + ".weight" for w in ("down_proj", "gate_proj", "pre_norm", "up_proj")
    )


@pytest.mark.skipif(_weight_map(DG_DIR) is None, reason="DiffusionGemma index.json not in HF cache")
def test_classification_covers_all_keys_no_leftovers():
    dg = _weight_map(DG_DIR)
    res = classify_keys(dg.keys())
    # every key is accounted for exactly once
    assert res.num_backbone + len(res.self_conditioning) + len(res.ignored) == len(dg)
    assert len(res.self_conditioning) == 4
    # the text backbone is the bulk (30 layers; MoE experts are PACKED into one
    # gate_up_proj + one down_proj tensor per layer, so the count is ~657, not ~1000).
    # Exact set-equality vs the gemma4 keyset is asserted in the test below.
    assert res.num_backbone > 600
    # ignored is encoder/vision/multimodal only
    assert all(k.startswith(("model.encoder.", "model.vision_tower.", "model.embed_vision.")) for k in res.ignored)


@pytest.mark.skipif(
    _weight_map(DG_DIR) is None or _weight_map(G4_DIR) is None,
    reason="need BOTH DiffusionGemma and gemma-4-26B-A4B-it index.json in HF cache",
)
def test_remapped_backbone_matches_gemma4_language_model_keyset():
    """The remapped DiffusionGemma backbone keys must EXACTLY equal the gemma4
    backbone keys the in-repo loader expects (model.language_model.*) — no missing,
    no renamed, no extra. This is the #47461 stage-2 weight-mapping gate."""
    dg = _weight_map(DG_DIR)
    g4 = _weight_map(G4_DIR)
    remapped = set(classify_keys(dg.keys()).backbone.values())
    g4_lm = {k for k in g4 if k.startswith("model.language_model.")}
    missing = g4_lm - remapped  # gemma4 expects but DiffusionGemma (remapped) lacks
    extra = remapped - g4_lm  # DiffusionGemma (remapped) has but gemma4 lacks
    assert not missing, f"DiffusionGemma backbone missing gemma4 keys: {sorted(missing)[:10]}"
    assert not extra, f"DiffusionGemma backbone has non-gemma4 keys after remap: {sorted(extra)[:10]}"


# ---------------------------------------------------------------------------
# checkpoint: real self-conditioning tensors
# ---------------------------------------------------------------------------
def _open_self_cond_tensors():
    """Load the 4 self-conditioning tensors from the DiffusionGemma safetensors."""
    idx_path = _index_path(DG_DIR)
    if idx_path is None:
        return None
    snap = os.path.dirname(idx_path)
    wmap = json.load(open(idx_path))["weight_map"]
    sc_keys = [k for k in wmap if k.startswith(SELF_CONDITIONING_PREFIX)]
    shards = {wmap[k] for k in sc_keys}
    if not all(os.path.exists(os.path.join(snap, s)) for s in shards):
        return None  # index present but tensors not downloaded
    from safetensors import safe_open

    out = {}
    handles = {s: safe_open(os.path.join(snap, s), framework="pt") for s in shards}
    for k in sc_keys:
        out[k] = handles[wmap[k]].get_tensor(k)
    return out


@pytest.mark.skipif(_open_self_cond_tensors() is None, reason="DiffusionGemma checkpoint tensors not downloaded")
def test_real_self_conditioning_loads_and_matches_config_shapes():
    sc_state = _open_self_cond_tensors()
    _, sc_short, _ = remap_state_dict(sc_state)  # strips the prefix
    tc = TextConfig()
    expected = expected_self_conditioning_shapes(tc.hidden_size, tc.intermediate_size)
    for name, shape in expected.items():
        assert tuple(sc_short[name].shape) == shape, f"{name}: {tuple(sc_short[name].shape)} != {shape}"

    # load into the reference module and run a forward
    mod = SelfConditioning(tc.hidden_size, intermediate_size=tc.intermediate_size).to(torch.float32)
    mod.load_from_state_dict({k: v.float() for k, v in sc_short.items()})
    emb = torch.randn(1, 4, tc.hidden_size)
    signal = torch.randn(1, 4, tc.hidden_size)
    out = mod(emb, signal)
    assert out.shape == (1, 4, tc.hidden_size) and torch.isfinite(out).all()
