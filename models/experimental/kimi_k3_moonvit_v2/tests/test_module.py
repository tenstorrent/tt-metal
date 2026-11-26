# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""CPU: the TT module tree loads the reference state dict by name, with the device upload stubbed.

Catches renamed or missing parameters and wrong pre-load transforms without a device. Also checks
that per-image inputs are written into the same device tensors on every call (`DeviceInput`), with the values
`host_prep.prepare_host_inputs` gives, and that an image of the grid already on the chips writes only its patches.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import ttnn
from models.experimental.kimi_k3_moonvit_v2.tt import host_prep
from models.experimental.kimi_k3_moonvit_v2.tests.common import SYNTHETIC, device_config, random_patches
from models.experimental.kimi_k3_moonvit_v2.tt import module, vision_block, vision_model
from models.experimental.kimi_k3_moonvit_v2.tt.device_input import DeviceInput, tiles_view, tilize_into
from models.experimental.kimi_k3_moonvit_v2.tt.image_prep import pixels_to_patches
from models.experimental.kimi_k3_moonvit_v2.tt.module import LoadingError, Module, Parameter, rename_substate


@pytest.fixture
def host_parameters(monkeypatch):
    """Parameters keep the torch tensor instead of uploading it."""

    def load_torch_tensor(self, value):
        if tuple(value.shape) != self.shape:
            raise LoadingError(f"expected shape {self.shape}, got {tuple(value.shape)}")
        self._data = value

    monkeypatch.setattr(Parameter, "load_torch_tensor", load_torch_tensor)
    monkeypatch.setattr(vision_model, "compute_kernel_config", lambda mesh_device: None)
    monkeypatch.setattr(vision_block, "linear_kernel_config", lambda mesh_device: None)
    return SimpleNamespace(shape=(1, 1))


@pytest.fixture
def host_inputs(monkeypatch):
    """`ttnn.from_torch` keeps the torch tensor, and copies to the chips are recorded, not made: a `DeviceInput`'s
    `host_value()` is what a copy sends. Returns the calls, "allocate" for a device upload and "copy" for an
    in-place write."""
    calls = []

    def from_torch(value, *, device=None, **kwargs):
        tensor = value.clone()
        if device is not None:
            calls.append("allocate")
            tensor.spec = None  # read by `_borrowing_host_tensor`, which the patched `host_tensor_from_bytes` ignores
        return tensor

    monkeypatch.setattr(ttnn, "from_torch", from_torch)
    monkeypatch.setattr(ttnn, "copy_host_to_device_tensor", lambda host, device_tensor: calls.append("copy"))
    monkeypatch.setattr(ttnn, "host_tensor_from_bytes", lambda raw, spec: raw)
    monkeypatch.setattr(ttnn, "ReplicateTensorToMesh", lambda mesh_device: None)
    return calls


@SYNTHETIC
def test_vision_model_loads_reference(host_parameters, config, reference):
    tt = vision_model.TT_MoonViT_Vision_Model(device_config(config, None), mesh_device=host_parameters)
    state = reference.state_dict()
    tt.load_torch_state_dict(state)
    assert tt.is_loaded()

    tower = tt.vision_tower
    block = tower.encoder.blocks[0]
    torch.testing.assert_close(block.attention.wqkv.weight.data, state["vision_tower.encoder.blocks.0.wqkv.weight"].T)
    torch.testing.assert_close(block.norm0.weight.data, state["vision_tower.encoder.blocks.0.norm0.weight"][None])
    proj = tower.patch_embed.proj.weight.data
    assert proj.shape == (config.padded_patch_dim, config.hidden_size)
    torch.testing.assert_close(
        proj[: config.patch_dim],
        state["vision_tower.patch_embed.proj.weight"].reshape(config.hidden_size, -1).T,
    )
    assert not proj[config.patch_dim :].any()
    assert torch.equal(tower.patch_embed.pos_emb_weight, state["vision_tower.patch_embed.pos_emb.weight"])
    torch.testing.assert_close(tt.mm_projector.fc0.weight.data, state["mm_projector.proj.0.weight"].T)
    torch.testing.assert_close(tt.mm_projector.fc1.weight.data, state["mm_projector.proj.2.weight"].T)
    torch.testing.assert_close(tt.mm_projector.post_norm.weight.data, state["mm_projector.post_norm.weight"][None])


@SYNTHETIC
def test_rejects_missing_and_unexpected_keys(host_parameters, config, reference, expect_error):
    state = reference.state_dict()
    state.pop("vision_tower.encoder.final_layernorm.weight")
    state["vision_tower.encoder.extra.weight"] = torch.zeros(1)
    tt = vision_model.TT_MoonViT_Vision_Model(device_config(config, None), mesh_device=host_parameters)
    with expect_error(LoadingError, "final_layernorm"):
        tt.load_torch_state_dict(state)


@SYNTHETIC
def test_rejects_missing_pos_emb(host_parameters, config, reference, expect_error):
    """The position table is kept on the host, not as a `Parameter`, so it has its own missing-key check."""
    state = reference.state_dict()
    state.pop("vision_tower.patch_embed.pos_emb.weight")
    tt = vision_model.TT_MoonViT_Vision_Model(device_config(config, None), mesh_device=host_parameters)
    with expect_error(LoadingError, "pos_emb.weight"):
        tt.load_torch_state_dict(state)


def test_prepare_inputs_needs_loaded_weights(host_parameters, config, expect_error):
    tt = vision_model.TT_MoonViT_Vision_Model(device_config(config, None), mesh_device=host_parameters)
    with expect_error(RuntimeError, "weights not loaded"):
        tt.prepare_inputs(*random_patches(config, [(1, 8, 8)]))


def test_rename_substate():
    state = {"wqkv.weight": 1, "wo.weight": 2, "norm0.weight": 3}
    rename_substate(state, "wqkv", "attention.wqkv")
    assert state == {"attention.wqkv.weight": 1, "wo.weight": 2, "norm0.weight": 3}


def test_load_cached(tmp_path, monkeypatch):
    class Leaf(Module):
        def __init__(self):
            super().__init__()
            self.weight = Parameter(shape=(2,), device=None)

    monkeypatch.setattr(Parameter, "load_torch_tensor", lambda self, v: setattr(self, "_data", v))
    monkeypatch.setattr(Parameter, "save", lambda self, path: torch.save(self.data, path))
    monkeypatch.setattr(Parameter, "load", lambda self, path: setattr(self, "_data", torch.load(path)))

    calls = []
    source = lambda: calls.append(1) or {"weight": torch.tensor([1.0, 2.0])}  # noqa: E731
    module.load_cached(Leaf(), tmp_path, source)
    cached = Leaf()
    module.load_cached(cached, tmp_path, source)
    assert len(calls) == 1
    assert cached.is_loaded() and torch.equal(cached.weight.data, torch.tensor([1.0, 2.0]))


def test_device_input_writes_in_place(host_inputs):
    x = DeviceInput((1, 1, 4, 2), mesh_device=object(), sp_axis=None, layout=ttnn.ROW_MAJOR_LAYOUT)
    first = x.write(torch.zeros(4, 2))
    second = x.write(torch.ones(4, 2))
    assert second is first
    assert torch.equal(x.host_value(), torch.ones(1, 1, 4, 2, dtype=torch.bfloat16))
    assert host_inputs == ["allocate", "copy"]


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "row_major"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
def test_device_input_write_rows_clears_stale_rows(host_inputs, layout, dtype):
    """write_rows fills the rows it is given and leaves every other row zero, also after a longer write."""
    x = DeviceInput((1, 1, 96, 64), mesh_device=object(), sp_axis=None, dtype=dtype, layout=layout)
    torch_dtype = x.storage_dtype
    writes = (70, 5, 33, 33, 0, 96)
    for n in writes:
        rows = torch.randn(n, 64)
        x.write_rows(rows)
        expected = torch.zeros(96, 64, dtype=torch_dtype)
        expected[:n] = rows
        assert torch.equal(x.host_value()[0, 0], expected), f"after writing {n} rows"
    assert host_inputs == ["allocate"] + ["copy"] * (len(writes) - 1)
    if layout == ttnn.ROW_MAJOR_LAYOUT:
        x.fill_rows(3, lambda view: view.fill_(2.0))
        assert torch.equal(x.host_value()[0, 0, :3], torch.full((3, 64), 2.0, dtype=torch_dtype))
        assert not x.host_value()[0, 0, 3:].any()


def test_device_input_rejects(expect_error):
    with expect_error(ValueError, "whole tiles"):
        DeviceInput((1, 1, 4, 2), mesh_device=object(), sp_axis=None)
    with expect_error(ValueError, r"\[1, 1, rows, width\]"):
        DeviceInput((2, 1, 32, 32), mesh_device=object(), sp_axis=None)
    x = DeviceInput((1, 1, 32, 32), mesh_device=object(), sp_axis=None)
    with expect_error(ValueError, "at most 32 rows of width 32"):
        x.write_rows(torch.zeros(33, 32))
    with expect_error(ValueError, "row-major"):
        x.fill_rows(1, lambda view: None)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
def test_staging_tile_layout_is_ttnn_tile_layout(dtype):
    """The staging buffer's TILE bytes, read by ttnn as a TILE tensor, give back the rows: the layout the chips get."""
    rows = torch.randn(96, 64)
    flat = torch.zeros(96 * 64, dtype=torch.float32 if dtype == ttnn.float32 else torch.bfloat16)
    tilize_into(tiles_view(flat, 64), rows)
    spec = ttnn.TensorSpec(ttnn.Shape([1, 1, 96, 64]), dtype, ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG)
    from_bytes = ttnn.to_torch(ttnn.host_tensor_from_bytes(flat.view(torch.uint8).numpy(), spec))
    assert torch.equal(from_bytes[0, 0], rows.to(flat.dtype))


@SYNTHETIC
def test_prepare_inputs_reuses_device_tensors(host_parameters, host_inputs, config, reference):
    """A second image of another size lands in the first image's device tensors, with the values of
    `prepare_host_inputs` cast to the device dtypes, bit for bit, padding rows zero."""
    config = dataclasses.replace(config, max_patches=1024)
    tt = vision_model.TT_MoonViT_Vision_Model(config, mesh_device=host_parameters)
    tt.load_torch_state_dict(reference.state_dict())

    first = tt.prepare_inputs(*random_patches(config, [(1, 8, 8)]))
    pixels, grid_thws = random_patches(config, [(1, 6, 20)], seed=1)
    second = tt.prepare_inputs(pixels, grid_thws)
    assert all(second[key] is first[key] for key in ("patches", "pos_emb", "logical_n"))
    assert all(b is a for a, b in zip(first["rope"], second["rope"]))

    pos_emb_weight = reference.vision_tower.patch_embed.pos_emb.weight
    expected = host_prep.prepare_host_inputs(config, pos_emb_weight, pixels, grid_thws)
    check_host_values(tt, expected)


def check_host_values(tt, expected: dict) -> None:
    for key, x in (("patches", tt.patches), ("pos_emb", tt.pos_emb), ("cos", tt.rope.cos), ("sin", tt.rope.sin)):
        assert torch.equal(x.host_value()[0, 0], expected[key].to(x.storage_dtype)), key
    assert tt.logical_n.host_value().item() == expected["logical_n"]


@SYNTHETIC
def test_prepare_image_writes_only_the_patches_for_the_same_grid(host_parameters, host_inputs, config, reference):
    """prepare_image writes what prepare_inputs(*pixels_to_patches(pixels)) does; an image of the grid the chips
    already hold rewrites only the patches."""
    config = dataclasses.replace(config, max_patches=1024)
    tt = vision_model.TT_MoonViT_Vision_Model(config, mesh_device=host_parameters)
    tt.load_torch_state_dict(reference.state_dict())
    pos_emb_weight = reference.vision_tower.patch_embed.pos_emb.weight
    rng = np.random.default_rng(0)

    tt.prepare_image(rng.integers(0, 256, (20 * 14, 20 * 14, 3), dtype=np.uint8))  # allocates every input
    for h, w, copies in ((6, 20, 5), (6, 20, 1), (20, 6, 5)):
        pixels = rng.integers(0, 256, (h * 14, w * 14, 3), dtype=np.uint8)
        del host_inputs[:]
        tt.prepare_image(pixels)
        assert host_inputs == ["copy"] * copies, f"{h}x{w}"
        check_host_values(tt, host_prep.prepare_host_inputs(config, pos_emb_weight, *pixels_to_patches(pixels, config)))


@SYNTHETIC
def test_prepare_image_rewrites_the_grid_after_a_reload(host_parameters, host_inputs, config, reference):
    """New weights mean new position embeddings, even for the grid the chips already hold."""
    config = dataclasses.replace(config, max_patches=1024)
    tt = vision_model.TT_MoonViT_Vision_Model(config, mesh_device=host_parameters)
    state = reference.state_dict()
    tt.load_torch_state_dict(state)
    pixels = np.random.default_rng(0).integers(0, 256, (6 * 14, 20 * 14, 3), dtype=np.uint8)
    tt.prepare_image(pixels)
    key = "vision_tower.patch_embed.pos_emb.weight"
    state = {**state, key: state[key] + 1.0}
    tt.load_torch_state_dict(state)
    tt.prepare_image(pixels)
    expected = host_prep.prepare_host_inputs(config, state[key], *pixels_to_patches(pixels, config))
    check_host_values(tt, expected)
