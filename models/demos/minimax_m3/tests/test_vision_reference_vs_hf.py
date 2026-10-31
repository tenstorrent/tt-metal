# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU check that the vision Torch reference (reference/vision.py) matches Hugging Face minimax_m3_vl.

Builds HF's MiniMaxM3VLVisionModel and MiniMaxM3VLMultiModalProjector at small configs, gives them the
reference's seeded random weights, and compares in fp32: the RoPE cos and sin, the patch embed, the output
of every layer, the tower output and the vision tokens. With HF's eager attention the reference runs the
same ops, so the results must be bit-identical; with HF's default SDPA attention they must agree to fp32
rounding. Two more tests check that the reference's default dimensions are HF's (and the checkpoint's, when
HF_MODEL is set) and that load_hf_state_dict maps HF's names.

No device and no checkpoint. Needs transformers with minimax_m3_vl (tt-metal pins 5.12.1). The test does
not import ttnn, but pytest also loads models/demos/minimax_m3/conftest.py, which does; add --noconftest
where ttnn is not installed.

Run:
    pytest models/demos/minimax_m3/tests/test_vision_reference_vs_hf.py
"""

from __future__ import annotations

import dataclasses
import json
import os

import pytest
import torch

from models.demos.minimax_m3.reference.vision import MiniMaxM3VisionConfig, MiniMaxM3VisionEncoder

hf_vision = pytest.importorskip("transformers.models.minimax_m3_vl.modeling_minimax_m3_vl")
MiniMaxM3VLConfig = hf_vision.MiniMaxM3VLConfig

# Small configs with the checkpoint's patch geometry. head_dim 80 is the checkpoint's (78 dims rotated,
# 2 pass through); head_dim 64 checks the general split of the rotary dims over T, H and W (60 and 4).
CONFIGS = {
    "hd80": dict(hidden_size=160, num_attention_heads=2, intermediate_size=320),
    "hd64": dict(hidden_size=256, num_attention_heads=4, intermediate_size=384, rope_theta=5000.0),
}
NUM_LAYERS = 2
PROJECTOR_HIDDEN, TEXT_HIDDEN = 96, 64

# (t, h, w) patch grid of each image. The images of one call share one attention (no mask).
GRIDS = {
    "h4w4": [(1, 4, 4)],
    "h4w6": [(1, 4, 6)],
    "h6w4": [(1, 6, 4)],
    "t2h4w4": [(2, 4, 4)],
    "two_images": [(1, 4, 4), (1, 2, 6)],
}


def _hf_config(dims: dict) -> MiniMaxM3VLConfig:
    dims = dict(dims)
    theta = dims.pop("rope_theta", 10000.0)
    return MiniMaxM3VLConfig(
        vision_config=dict(
            num_hidden_layers=NUM_LAYERS, rope_parameters={"rope_type": "default", "rope_theta": theta}, **dims
        ),
        text_config=dict(hidden_size=TEXT_HIDDEN),
        projector_hidden_size=PROJECTOR_HIDDEN,
    )


def _strip(state_dict: dict, prefix: str) -> dict:
    return {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}


def _build(config_id: str, seed: int = 0):
    """Reference with seeded random weights, and the HF tower and projector holding the same weights."""
    hf_config = _hf_config(CONFIGS[config_id])
    config = MiniMaxM3VisionConfig.from_hf_config(hf_config.to_dict())
    reference = MiniMaxM3VisionEncoder(config).init_random_weights(seed).eval()

    tower = hf_vision.MiniMaxM3VLVisionModel(hf_config.vision_config).eval()
    projector = hf_vision.MiniMaxM3VLMultiModalProjector(hf_config).eval()
    weights = reference.hf_state_dict()
    tower.load_state_dict(_strip(weights, "vision_tower."), strict=True)
    projector.load_state_dict(_strip(weights, "multi_modal_projector."), strict=True)
    return reference, tower, projector


def _pixel_values(config: MiniMaxM3VisionConfig, grid_thw: torch.Tensor, seed: int = 1) -> torch.Tensor:
    num_patches = int(grid_thw.prod(dim=-1).sum())
    return torch.randn(num_patches, config.patch_dim, generator=torch.Generator().manual_seed(seed))


def _layer_outputs(layers) -> tuple:
    """Forward hooks that record each layer's output, as (L, hidden)."""
    outputs = []

    def hook(module, args, out):
        out = out[0] if isinstance(out, tuple) else out
        outputs.append(out.detach().reshape(-1, out.shape[-1]))

    return outputs, [layer.register_forward_hook(hook) for layer in layers]


def _check(name: str, ours: torch.Tensor, hf: torch.Tensor, exact: bool):
    assert ours.shape == hf.shape, f"{name}: shape {tuple(ours.shape)} != HF {tuple(hf.shape)}"
    if exact:
        max_diff = (ours - hf).abs().max().item()
        assert torch.equal(ours, hf), f"{name}: not bit-identical to HF, max abs diff {max_diff:.3e}"
    else:
        torch.testing.assert_close(ours, hf, rtol=1e-5, atol=1e-5, msg=lambda m: f"{name}: {m}")


@torch.no_grad()
@pytest.mark.parametrize("grid_id", GRIDS)
@pytest.mark.parametrize("config_id", CONFIGS)
def test_vision_rope_vs_hf(config_id, grid_id):
    """reference rope_cos_sin == HF MiniMaxM3VL3DRotaryEmbedding, bit for bit."""
    reference, tower, _ = _build(config_id)
    grid_thw = torch.tensor(GRIDS[grid_id])
    cos, sin = reference.rope(grid_thw)
    hf_cos, hf_sin = tower.rotary_emb(grid_thw, device=torch.device("cpu"), dtype=torch.float32)
    _check("rope cos", cos, hf_cos, exact=True)
    _check("rope sin", sin, hf_sin, exact=True)


@torch.no_grad()
@pytest.mark.parametrize("attn_implementation", ["eager", "sdpa"])
@pytest.mark.parametrize("grid_id", GRIDS)
@pytest.mark.parametrize("config_id", CONFIGS)
def test_vision_reference_vs_hf(config_id, grid_id, attn_implementation):
    """Patch embed, every layer, tower output and vision tokens == HF (bit-identical against eager attention)."""
    reference, tower, projector = _build(config_id)
    tower.set_attn_implementation(attn_implementation)
    exact = attn_implementation == "eager"
    grid_thw = torch.tensor(GRIDS[grid_id])
    pixel_values = _pixel_values(reference.config, grid_thw)

    _check("patch embed", reference.embed(pixel_values), tower.embeddings(pixel_values), exact=True)

    ours_layers, ours_hooks = _layer_outputs(reference.layers)
    hf_layers, hf_hooks = _layer_outputs(tower.layers)
    try:
        ours_tower = reference.forward_tower(pixel_values, grid_thw)
        hf_tower = tower(pixel_values=pixel_values, image_grid_thw=grid_thw).last_hidden_state.squeeze(0)
    finally:
        for handle in ours_hooks + hf_hooks:
            handle.remove()
    assert len(ours_layers) == len(hf_layers) == NUM_LAYERS
    for i, (ours, hf) in enumerate(zip(ours_layers, hf_layers)):
        _check(f"layer {i} output", ours, hf, exact)
    _check("tower output", ours_tower, hf_tower, exact)

    # Projector on the same input on both sides, then the whole encoder.
    hf_tokens = projector(hf_tower)
    _check("projector", reference.projector(hf_tower), hf_tokens, exact=True)
    tokens = reference(pixel_values, grid_thw)
    _check("vision tokens", tokens, hf_tokens, exact)
    assert tokens.shape == (pixel_values.shape[0] // reference.config.spatial_merge_size**2, TEXT_HIDDEN)


def test_vision_config_defaults_match_hf():
    """The reference's default dimensions are HF's MiniMax-M3 defaults (and the checkpoint's, when HF_MODEL is set)."""
    assert MiniMaxM3VisionConfig.from_hf_config(MiniMaxM3VLConfig().to_dict()) == MiniMaxM3VisionConfig()
    checkpoint = os.getenv("HF_MODEL")
    if checkpoint and os.path.isfile(os.path.join(checkpoint, "config.json")):
        with open(os.path.join(checkpoint, "config.json")) as f:
            assert MiniMaxM3VisionConfig.from_hf_config(json.load(f)) == MiniMaxM3VisionConfig()


@torch.no_grad()
def test_vision_load_hf_state_dict():
    """load_hf_state_dict takes HF's names, skips other keys and layers past num_hidden_layers, and accepts
    the patch embed weight flattened to (hidden, patch_dim)."""
    reference, tower, projector = _build("hd80")
    weights = {f"vision_tower.{k}": v for k, v in tower.state_dict().items()}
    weights.update({f"multi_modal_projector.{k}": v for k, v in projector.state_dict().items()})
    weights["vision_tower.embeddings.proj.weight"] = weights["vision_tower.embeddings.proj.weight"].flatten(1)
    weights["language_model.embed_tokens.weight"] = torch.zeros(8, TEXT_HIDDEN)

    config = reference.config
    loaded = MiniMaxM3VisionEncoder(config).load_hf_state_dict(weights).state_dict()
    for name, value in reference.state_dict().items():
        assert torch.equal(loaded[name], value), name

    short = MiniMaxM3VisionEncoder(dataclasses.replace(config, num_hidden_layers=1)).load_hf_state_dict(weights)
    assert torch.equal(short.layers[0].mlp.fc1.weight, reference.layers[0].mlp.fc1.weight)
