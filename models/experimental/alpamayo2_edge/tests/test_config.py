# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from models.experimental.alpamayo2_edge.reference.config import load_config
from models.experimental.alpamayo2_edge.reference.weights import load_weights


def test_config_matches_checkpoint(cfg):
    assert (cfg.text.hidden, cfg.text.layers, cfg.text.mlp) == (2048, 28, 9216)
    assert (cfg.expert.hidden, cfg.expert.layers, cfg.expert.mlp) == (512, 28, 3840)
    assert cfg.vision.layers == 27 and cfg.projector.hidden == 11520
    assert cfg.tokens.future_start == 135084


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
    )
    for name, value in expected.items():
        got = getattr(cfg.action, name)
        assert got == (tuple(value) if isinstance(value, list) else value), name
    assert cfg.action.steps == 10
