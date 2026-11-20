# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import replace
from pathlib import Path

import huggingface_hub
import pytest

from models.experimental.alpamayo2_edge.reference.config import (
    MODEL_REVISION,
    load_config,
    resolve_model_dir,
    validate_supported,
)
from models.experimental.alpamayo2_edge.reference.weights import load_weights
from models.experimental.alpamayo2_edge.tests._util import full_depth_only


@full_depth_only
def test_config_matches_checkpoint(cfg):
    assert (cfg.text.hidden, cfg.text.layers, cfg.text.mlp) == (2048, 28, 9216)
    assert (cfg.expert.hidden, cfg.expert.layers, cfg.expert.mlp) == (512, 28, 3840)
    assert cfg.vision.layers == 27 and cfg.projector.hidden == 11520
    assert cfg.tokens.future_start == 135084
    # Upstream constants the checkpoint does not carry (`reference/config.py`), and the vision activation it does.
    assert (cfg.projector.norm_eps, cfg.action.in_proj_norm_eps) == (1e-6, 1e-5)
    assert cfg.vision.hidden_act == "gelu_pytorch_tanh"


@pytest.mark.skipif(bool(os.environ.get("ALPAMAYO_REVISION")), reason="ALPAMAYO_REVISION selects another snapshot")
def test_model_dir_is_the_pinned_revision(model_dir):
    assert model_dir.name == MODEL_REVISION
    # The README's download command must fetch the same snapshot.
    assert MODEL_REVISION in (Path(__file__).parents[1] / "README.md").read_text()


@pytest.mark.parametrize("override, expected", [(None, MODEL_REVISION), ("main", "main")])
def test_resolve_model_dir_revision(override, expected, monkeypatch):
    calls = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo, revision: calls.append(revision) or "/x")
    if override is None:
        monkeypatch.delenv("ALPAMAYO_REVISION", raising=False)
    else:
        monkeypatch.setenv("ALPAMAYO_REVISION", override)
    resolve_model_dir()
    assert calls == [expected]


def test_weights_prefixes(model_dir):
    vis = load_weights(model_dir, "vlm.model.visual.")
    assert vis["embeddings.patch_embedding.weight"].shape == (1152, 768)
    assert vis["embeddings.position_embedding.weight"].shape == (256, 1152)
    ex = load_weights(model_dir, "expert.expert.")
    assert ex["layers.0.self_attn.q_proj.weight"].shape == (2048, 512)
    assert ex["layers.0.self_attn.k_proj.weight"].shape == (1024, 512)


def test_num_layers_override(monkeypatch, model_dir):
    monkeypatch.setenv("ALPAMAYO_NUM_LAYERS", "2")
    c = load_config(model_dir)
    assert c.vision.layers == c.text.layers == c.expert.layers == 2


def test_action_config_matches_checkpoint(cfg, model_dir):
    import json

    with open(model_dir / "config.json") as f:
        expert = json.load(f)["expert_config"]
    expected = {k: v for k, v in expert["action_space_cfg"].items() if k != "_target_"}
    expected.update(
        {
            k: v
            for k, v in expert["diffusion_cfg"].items()
            if k in ("int_method", "inference_guidance_weight", "use_classifier_free_guidance")
        }
    )
    in_proj = expert["action_in_proj_cfg"]
    expected.update(
        fourier_feats=in_proj["num_fourier_feats"],
        max_freq=in_proj["max_freq"],
        in_proj_hidden=in_proj["hidden_size"],
        in_proj_layers=in_proj["num_enc_layers"],
        expert_non_causal_attention=expert["expert_non_causal_attention"],
    )
    for name, value in expected.items():
        got = getattr(cfg.action, name)
        assert got == (tuple(value) if isinstance(value, list) else value), name
    assert cfg.action.steps == 10


@pytest.mark.parametrize(
    "part, change, message",
    [
        ("action", {"use_classifier_free_guidance": True}, "unguided Euler"),
        ("action", {"int_method": "rk4"}, "unguided Euler"),
        ("action", {"expert_non_causal_attention": False}, "non-causal"),
        ("expert", {"layers": 27}, r"KV-cache geometry \(layers\)"),
        ("expert", {"kv_heads": 4}, r"KV-cache geometry \(kv_heads\)"),
        ("expert", {"head_dim": 64}, r"KV-cache geometry \(head_dim\)"),
        ("vision", {"hidden_act": "gelu"}, "tanh-approximated GELU"),
    ],
)
def test_unsupported_config_rejected(cfg, part, change, message, expect_error):
    validate_supported(cfg)
    with expect_error(ValueError, message):
        validate_supported(replace(cfg, **{part: replace(getattr(cfg, part), **change)}))
