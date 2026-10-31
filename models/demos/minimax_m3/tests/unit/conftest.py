# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Fixtures for the vision encoder unit tests (test_vision_*_vs_ref.py).

The tests compare the TT vision modules against the Torch reference in reference/vision.py, at the
checkpoint's dimensions with seeded random weights, so they need no checkpoint and no HF_MODEL. Both
fixtures are session scoped and cache what they build.

  - vision_reference(num_layers=2): the reference tower and projector with num_layers layers.
  - vision_activations(grid_thw, num_layers=2): one forward of that reference on seeded random pixel values
    for one image of grid_thw = (t, h, w) patches. Keys: pixel_values and grid_thw; "<name>.in" and
    "<name>.out" for pre_layrnorm, layer0 and its layer_norm1, self_attn and mlp, each (L, hidden);
    tower.out (L, hidden) and final (L / 4, text hidden).
"""
from __future__ import annotations

import pytest
import torch


@pytest.fixture(scope="session")
def vision_reference():
    from models.demos.minimax_m3.reference.vision import MiniMaxM3VisionConfig, MiniMaxM3VisionEncoder

    cache = {}

    def get(num_layers: int = 2):
        if num_layers not in cache:
            config = MiniMaxM3VisionConfig(num_hidden_layers=num_layers)
            cache[num_layers] = MiniMaxM3VisionEncoder(config).init_random_weights(seed=0).eval()
        return cache[num_layers]

    return get


@pytest.fixture(scope="session")
def vision_activations(vision_reference):
    cache = {}

    def get(grid_thw, num_layers: int = 2):
        key = (tuple(grid_thw), num_layers)
        if key not in cache:
            cache[key] = _record_activations(vision_reference(num_layers), tuple(grid_thw))
        return cache[key]

    return get


@torch.no_grad()
def _record_activations(reference, grid_thw):
    t, h, w = grid_thw
    generator = torch.Generator().manual_seed(1000 * h + w)
    pixel_values = torch.randn(t * h * w, reference.config.patch_dim, generator=generator)
    acts = {"pixel_values": pixel_values, "grid_thw": torch.tensor([grid_thw])}

    def record(name):
        def hook(module, args, out):
            acts[f"{name}.in"] = args[0].reshape(-1, args[0].shape[-1])
            acts[f"{name}.out"] = out.reshape(-1, out.shape[-1])

        return hook

    layer0 = reference.layers[0]
    modules = {
        "pre_layrnorm": reference.pre_layrnorm,
        "layer0": layer0,
        "layer0.layer_norm1": layer0.layer_norm1,
        "layer0.self_attn": layer0.self_attn,
        "layer0.mlp": layer0.mlp,
    }
    handles = [module.register_forward_hook(record(name)) for name, module in modules.items()]
    try:
        acts["tower.out"] = reference.forward_tower(pixel_values, acts["grid_thw"])
    finally:
        for handle in handles:
            handle.remove()
    acts["final"] = reference.projector(acts["tower.out"])
    return acts
