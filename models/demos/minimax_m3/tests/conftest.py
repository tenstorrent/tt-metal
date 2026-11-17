# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Fixtures and a hook for the vision encoder's device tests (unit/test_vision_*_vs_ref.py).

The reference is Hugging Face's vision tower and projector, with eager attention, at the checkpoint's sizes. Its
weights are seeded random values, so no checkpoint is needed. Weights and inputs are bf16 values, as the chip holds
them, so the tests measure the chip's arithmetic and not the rounding of its inputs. The reference fixtures are
session scoped and cache what they build. ensure_gc replaces models/conftest.py's fixture for every test here.
"""
from __future__ import annotations

import gc
from types import SimpleNamespace

import pytest
import torch
from loguru import logger


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_call(item):
    """Log "[<test id>] compute grid <x>x<y>" before a test_vision_* test that has a mesh_device.

    The matmul and SDPA program configs depend on the grid, and a CI runner's grid may differ from the one the bounds
    were measured on.
    """
    mesh_device = item.funcargs.get("mesh_device") if item.name.startswith("test_vision_") else None
    if mesh_device is not None:
        grid = mesh_device.compute_with_storage_grid_size()
        logger.info(f"[{item.name}] compute grid {grid.x}x{grid.y}")


@pytest.fixture(autouse=True)
def ensure_gc(request):
    """Collect garbage before each test, as models/conftest.py's ensure_gc does, except before a vision host test.

    A vision test without a chip holds no device memory, so a full collection before it only costs time.
    """
    if request.node.path.name.startswith("test_vision_") and "mesh_device" not in request.fixturenames:
        return
    gc.collect()


@pytest.fixture(scope="session")
def vision_reference():
    """Return a function get(num_layers=2) that builds the reference with num_layers layers.

    The reference has these fields:
        config: its MiniMaxM3VisionConfig.
        weights: random_vision_weights rounded to bf16, under transformers 5.12's names.
        tower, projector: Hugging Face's modules holding those weights.
    """
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
    """Return a function get(grid_thw, num_layers=2) that runs the reference on seeded random pixels for one image.

    get returns a dict of tensors:
        pixel_values, grid_thw: the input, for one image of grid_thw = (t, h, w) patches.
        rope: the (cos, sin) that the tower gives its layers, each (L, 78).
        layer0.in, layer0.self_attn.in, layer0.mlp.in: the inputs of layer 0, its attention and its MLP, (L, hidden).
        tower.out: the tower output, (L, hidden).
        final: the vision tokens, (L / 4, text hidden).
    """
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
