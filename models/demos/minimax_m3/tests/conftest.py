# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Fixtures and a hook of the vision encoder's device tests (unit/test_vision_*_vs_ref.py).

The tests compare the TT vision modules against Hugging Face's vision tower and projector
(vision_helpers.hf_vision_modules: MiniMaxM3VLVisionModel with eager attention, then MiniMaxM3VLMultiModalProjector), at
the checkpoint's dimensions with seeded random weights, so they need no checkpoint and no HF_MODEL. Both fixtures are
session scoped and cache what they build. The weights and pixel values are bf16 values, as the device holds them, and
the module tests run their reference module on the bf16 copy of the input that the device gets, so the tests measure the
device's arithmetic and not the rounding of its inputs.

  - vision_reference(num_layers=2): the reference with num_layers layers. Fields: config, its MiniMaxM3VisionConfig;
    weights, the weights of vision_checkpoint.random_vision_weights under transformers' names, as
    M3VLVisionModel.from_state_dict and TtMiniMaxM3VisionEncoder take them; tower and projector, Hugging Face's modules
    holding them.
  - vision_activations(grid_thw, num_layers=2): one forward of that reference on seeded random pixel values for one
    image of grid_thw = (t, h, w) patches. Keys: pixel_values and grid_thw; rope, the cos and sin that the tower gives
    its layers, each (L, 78); "<name>.in", the input of layer0, layer0.self_attn and layer0.mlp, each (L, hidden);
    tower.out (L, hidden) and final (L / 4, text hidden).

Before each vision test that opens a device (a test_vision_* test with a mesh_device), pytest_runtest_call logs the
device's compute grid: the program configs of the weight matmuls and of SDPA depend on it, and a CI runner's grid can
differ from the one the bounds were measured on.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from loguru import logger


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_call(item):
    """Logs "[<test id>] compute grid <x>x<y>" for a test_vision_* test that has a mesh_device."""
    mesh_device = item.funcargs.get("mesh_device") if item.name.startswith("test_vision_") else None
    if mesh_device is not None:
        grid = mesh_device.compute_with_storage_grid_size()
        logger.info(f"[{item.name}] compute grid {grid.x}x{grid.y}")


@pytest.fixture(scope="session")
def vision_reference():
    from models.demos.minimax_m3.tests.unit.vision_checkpoint import random_vision_weights
    from models.demos.minimax_m3.tests.unit.vision_helpers import hf_vision_modules
    from models.demos.minimax_m3.tt.vision.config import MiniMaxM3VisionConfig

    cache = {}

    def get(num_layers: int = 2):
        if num_layers not in cache:
            config = MiniMaxM3VisionConfig(num_hidden_layers=num_layers)
            weights = {name: value.bfloat16().float() for name, value in random_vision_weights(config).items()}
            tower, projector = hf_vision_modules(config, weights)
            cache[num_layers] = SimpleNamespace(config=config, weights=weights, tower=tower, projector=projector)
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
    pixel_values = torch.randn(t * h * w, reference.config.patch_dim, generator=generator).bfloat16().float()
    acts = {"pixel_values": pixel_values, "grid_thw": torch.tensor([grid_thw])}

    def rows(x):
        return x.reshape(-1, x.shape[-1])

    def record_layer_input(module, args, kwargs):
        acts["layer0.in"], acts["rope"] = rows(args[0]), kwargs["position_embeddings"]

    def record_output(name):
        def hook(module, args, out):
            acts[f"{name}.in"] = rows(out)

        return hook

    # Layer 0's input and position embeddings, and its LayerNorms' outputs: the inputs of its attention and MLP.
    layer0 = reference.tower.layers[0]
    handles = [
        layer0.register_forward_pre_hook(record_layer_input, with_kwargs=True),
        layer0.layer_norm1.register_forward_hook(record_output("layer0.self_attn")),
        layer0.layer_norm2.register_forward_hook(record_output("layer0.mlp")),
    ]
    try:
        tower = reference.tower(pixel_values=pixel_values, image_grid_thw=acts["grid_thw"])
    finally:
        for handle in handles:
            handle.remove()
    acts["tower.out"] = tower.last_hidden_state.squeeze(0)
    acts["final"] = reference.projector(acts["tower.out"])
    return acts
