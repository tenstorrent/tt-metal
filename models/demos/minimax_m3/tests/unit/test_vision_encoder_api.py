# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Host-side checks of TtMiniMaxM3VisionEncoder and read_embeddings (multimodal/encoder.py), with stand-ins for the chip.
The tests check that:
  - forward raises for pixels that no image resizes to (check_pixels' rules) before any ttnn call;
  - forward runs the model on patch_rows of the pixels, untilizes the model's output on the chip, frees the tiled copy
    and returns the tokens as [n_tokens, out_dim];
  - __init__ and from_weights_path raise for a mesh of more than one chip, a config whose patches are not the image
    code's, and weights under the checkpoint's own names, before any weight goes to the device; the model is built
    from the state_dict's weights, reading no other value of the state_dict, and from_weights_path reads a
    checkpoint's vision weights and, without a config, its config.json, checking the mesh before it reads a weight;
  - vision_config takes a MiniMaxM3VisionConfig, a config.json dict and a transformers config, and raises ValueError
    naming the fields it needs for one without them;
  - read_embeddings raises for an out it cannot fill before it reads, and copies the tokens' bf16 bits into out with
    Tensor.cpu, from a stand-in and, with ttnn itself, from a host tensor.
The encoder's device tests are in test_vision_encoder_vs_ref.py. These tests have no mesh id; the 1-card CI leg selects
them by name (its -k includes "test_vision_").

Run:
    pytest models/demos/minimax_m3/tests/unit/test_vision_encoder_api.py
"""
from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

import ttnn
from models.demos.minimax_m3.multimodal import encoder as encoder_module
from models.demos.minimax_m3.multimodal.encoder import TtMiniMaxM3VisionEncoder, read_embeddings, vision_config
from models.demos.minimax_m3.reference.vision import MiniMaxM3VisionConfig, MiniMaxM3VisionEncoder
from models.demos.minimax_m3.tt.vision.model import M3VLVisionModel
from models.demos.minimax_m3.tt.vision.preprocess import patch_rows

from .test_vision_loader import _checkpoint_name

# The image code's patch geometry (the defaults) with small widths, so the reference is built in milliseconds.
CONFIG = MiniMaxM3VisionConfig(
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=1,
    num_attention_heads=2,
    projector_hidden_size=24,
    text_hidden_size=40,
)
SOURCE = MiniMaxM3VisionEncoder(CONFIG).init_random_weights(seed=0)
# CONFIG in the layout of the checkpoint's config.json (transformers 4.52.4): the merge and temporal sizes under
# img_token_compression_config.
COMPRESSION = {"image_token_compression_method": "patch_merge", "spatial_merge_size": 2, "temporal_patch_size": 2}
CONFIG_JSON = {
    "vision_config": {
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "patch_size": 14,
        "img_token_compression_config": COMPRESSION,
    },
    "projector_hidden_size": 24,
    "text_config": {"hidden_size": 40},
}
PIXELS = np.zeros((56, 84, 3), dtype=np.uint8)  # 2 tokens down and 3 across


class _Mesh:
    """A stand-in for a mesh device of num_devices chips."""

    def __init__(self, num_devices: int = 1):
        self.num_devices = num_devices

    def get_num_devices(self) -> int:
        return self.num_devices


def _device_work(*args, **kwargs):
    raise AssertionError("device work before the input check")


def _build(monkeypatch, model=_device_work, mesh=None, state_dict=None, config=CONFIG):
    """An encoder whose model is the stand-in model, and the calls of M3VLVisionModel.from_torch that built it."""
    built = []

    def from_torch(mesh_device, reference, with_projector, dtype):
        built.append(SimpleNamespace(mesh_device=mesh_device, reference=reference, with_projector=with_projector))
        assert dtype == ttnn.bfloat16, dtype
        return model

    monkeypatch.setattr(M3VLVisionModel, "from_torch", from_torch)
    state_dict = SOURCE.hf_state_dict() if state_dict is None else state_dict
    return TtMiniMaxM3VisionEncoder(_Mesh() if mesh is None else mesh, state_dict, config), built


# id: (pixels, error, message): pixels that no image resizes to.
BAD_PIXELS = {
    "list": (PIXELS.tolist(), TypeError, "numpy array"),
    "float32": (PIXELS.astype(np.float32), TypeError, "uint8"),
    "rgba": (np.zeros((56, 84, 4), dtype=np.uint8), ValueError, r"\(height, width, 3\)"),
    "not_whole_blocks": (np.zeros((56, 80, 3), dtype=np.uint8), ValueError, "multiples of 28"),
    "too_few_tokens": (np.zeros((28, 56, 3), dtype=np.uint8), ValueError, "4 to 576 vision tokens"),
    "too_many_tokens": (np.zeros((700, 700, 3), dtype=np.uint8), ValueError, "4 to 576 vision tokens"),
    "too_stretched": (np.zeros((28, 9520, 3), dtype=np.uint8), ValueError, "aspect ratio must be at most 339"),
}


@pytest.mark.parametrize("case", BAD_PIXELS)
def test_vision_encoder_rejects_bad_pixels(case, expect_error, monkeypatch):
    """forward raises TypeError or ValueError for pixels that no image resizes to, before any ttnn call: the model and
    the ttnn calls are stand-ins that fail the test if they run."""
    pixels, error, message = BAD_PIXELS[case]
    encoder, _ = _build(monkeypatch)
    for name in ("from_torch", "to_layout", "reshape", "deallocate"):
        monkeypatch.setattr(ttnn, name, _device_work)
    with expect_error(error, message):
        encoder.forward(pixels)


def test_vision_encoder_forward_ops(monkeypatch):
    """forward runs the model on patch_rows of the pixels, then untilizes the model's output, frees the tiled copy and
    returns the row-major tokens reshaped to [n_tokens, out_dim]. No device: the model and the ttnn calls are
    stand-ins that log each call."""
    calls = []

    def model(pixel_values, grid_thw):
        calls.append(("model", pixel_values, grid_thw))
        return "tokens_tiled"

    monkeypatch.setattr(ttnn, "to_layout", lambda x, layout: calls.append(("to_layout", x, layout)) or "tokens_rows")
    monkeypatch.setattr(ttnn, "deallocate", lambda x: calls.append(("deallocate", x)))
    monkeypatch.setattr(ttnn, "reshape", lambda x, shape: calls.append(("reshape", x, shape)) or "tokens")
    encoder, _ = _build(monkeypatch, model=model)
    pixels = np.random.default_rng(0).integers(0, 256, (56, 84, 3), dtype=np.uint8)

    assert encoder.forward(pixels) == "tokens" and encoder(pixels) == "tokens"
    pixel_values, grid_thw = patch_rows(pixels)
    (kind, model_rows, model_grid), *ops = calls[: len(calls) // 2]
    assert kind == "model" and torch.equal(model_rows, pixel_values) and torch.equal(model_grid, grid_thw)
    assert grid_thw.tolist() == [[1, 4, 6]]
    assert ops == [
        ("to_layout", "tokens_tiled", ttnn.ROW_MAJOR_LAYOUT),
        ("deallocate", "tokens_tiled"),
        ("reshape", "tokens_rows", (6, CONFIG.text_hidden_size)),
    ], ops
    assert encoder.out_dim == CONFIG.text_hidden_size


def test_vision_encoder_builds_from_state_dict(monkeypatch):
    """The model is built once, with the projector and bf16 weights, from a reference that holds every weight of the
    state_dict under the config; a language model weight in the state_dict is ignored."""
    state_dict = {**SOURCE.hf_state_dict(), "language_model.model.embed_tokens.weight": torch.zeros(8, 40)}
    mesh = _Mesh()
    encoder, built = _build(monkeypatch, mesh=mesh, state_dict=state_dict)
    (call,) = built
    assert call.mesh_device is mesh and call.with_projector and encoder.mesh_device is mesh
    assert call.reference.config == CONFIG == encoder.config and not call.reference.training
    loaded = call.reference.state_dict()
    assert loaded.keys() == SOURCE.state_dict().keys()
    for name, value in SOURCE.state_dict().items():
        assert torch.equal(loaded[name], value), name


class _LazyCheckpoint(Mapping):
    """A mapping that records the keys whose values are read, as a lazy view of a whole checkpoint would read them."""

    def __init__(self, values):
        self.values, self.read = values, []

    def __getitem__(self, key):
        self.read.append(key)
        return self.values[key]

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)


def test_vision_encoder_reads_only_vision_weights(monkeypatch):
    """Given a mapping over a whole checkpoint, the encoder reads the values of the vision weights only, once each, and
    never a language model's."""
    state_dict = _LazyCheckpoint({"language_model.model.embed_tokens.weight": None, **SOURCE.hf_state_dict()})
    _, built = _build(monkeypatch, state_dict=state_dict)
    assert sorted(state_dict.read) == sorted(SOURCE.hf_state_dict()), state_dict.read
    (call,) = built
    for name, value in SOURCE.state_dict().items():
        assert torch.equal(call.reference.state_dict()[name], value), name


def test_vision_encoder_one_chip(monkeypatch, expect_error, tmp_path):
    """__init__ and from_weights_path raise ValueError naming create_submesh for a mesh of more than one chip, before
    any weight is read or goes to the device."""
    monkeypatch.setattr(M3VLVisionModel, "from_torch", _device_work)
    monkeypatch.setattr(encoder_module, "load_vision_state_dict", _device_work)
    (tmp_path / "config.json").write_text(json.dumps(CONFIG_JSON))
    with expect_error(ValueError, r"one chip, got a mesh of 4: .*create_submesh\(ttnn.MeshShape\(1, 1\)\)"):
        TtMiniMaxM3VisionEncoder(_Mesh(4), SOURCE.hf_state_dict(), CONFIG)
    with expect_error(ValueError, "one chip, got a mesh of 32"):
        TtMiniMaxM3VisionEncoder.from_weights_path(_Mesh(32), str(tmp_path))


# id: config field and the value that is not the image code's.
OTHER_GEOMETRY = {
    "patch_size": 16,
    "temporal_patch_size": 1,
    "num_channels": 1,
    "spatial_merge_size": 3,
}


@pytest.mark.parametrize("field", OTHER_GEOMETRY)
def test_vision_encoder_checks_patch_geometry(field, monkeypatch, expect_error):
    """A config whose patches are not those preprocess.py makes (14 x 14 pixels, 2 frames, 3 channels, merged 2 x 2)
    raises ValueError before the model is built."""
    config = dataclasses.replace(CONFIG, **{field: OTHER_GEOMETRY[field]})
    with expect_error(ValueError, f"the image code makes patches of 14 x 14 pixels.*{field} {OTHER_GEOMETRY[field]}"):
        _build(monkeypatch, model=_device_work, config=config)


def test_vision_encoder_rejects_checkpoint_names(monkeypatch, expect_error):
    """Weights under the checkpoint's own (transformers 4.52.4) names raise ValueError naming load_vision_state_dict,
    which renames them."""
    state_dict = {_checkpoint_name(name): value for name, value in SOURCE.hf_state_dict().items()}
    assert any(name.startswith("patch_merge_mlp.") for name in state_dict)
    with expect_error(ValueError, "weights under the checkpoint's own names.*load_vision_state_dict"):
        _build(monkeypatch, state_dict=state_dict)


def test_vision_encoder_config_forms(expect_error):
    """vision_config takes a MiniMaxM3VisionConfig as it is, the checkpoint's config.json as a dict, and a transformers
    MiniMaxM3VLConfig, the form the encoder service has; anything else raises TypeError."""
    from transformers import MiniMaxM3VLConfig

    assert vision_config(CONFIG) is CONFIG
    assert vision_config(CONFIG_JSON) == CONFIG
    two_layers = MiniMaxM3VLConfig(vision_config={"num_hidden_layers": 2})
    assert vision_config(two_layers) == MiniMaxM3VisionConfig(num_hidden_layers=2)
    with expect_error(TypeError, "config must be a transformers config"):
        vision_config("config.json")


@pytest.mark.parametrize("missing", ["vision_config", "projector_hidden_size", "text_config"])
def test_vision_encoder_config_missing_field(missing, expect_error):
    """A config without a field that the vision config is read from raises ValueError naming the fields it needs: the
    checkpoint's config.json without that field, and for vision_config also the vision part of a transformers config
    on its own, a config that is easy to pass by mistake."""
    from transformers import MiniMaxM3VLConfig

    configs = [{key: value for key, value in CONFIG_JSON.items() if key != missing}]
    if missing == "vision_config":
        configs.append(MiniMaxM3VLConfig().vision_config)
    for config in configs:
        with expect_error(ValueError, f"config has no field '{missing}': it needs vision_config"):
            vision_config(config)


def test_vision_encoder_from_weights_path(monkeypatch, expect_error, tmp_path):
    """from_weights_path builds the model from the vision weights of a checkpoint written as MiniMax-M3's is (the
    checkpoint's names, an index, a config.json): with no config, by its config.json; with one, by that config. An
    empty weights_path raises ValueError instead of reading the current directory."""
    weights = {_checkpoint_name(name): value for name, value in SOURCE.hf_state_dict().items()}
    weights["language_model.model.embed_tokens.weight"] = torch.zeros(8, 40)
    shard = "model-00001-of-00001.safetensors"
    save_file(weights, str(tmp_path / shard))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": dict.fromkeys(weights, shard)}))
    (tmp_path / "config.json").write_text(json.dumps(CONFIG_JSON))

    _, built = _build(monkeypatch)
    built.clear()
    TtMiniMaxM3VisionEncoder.from_weights_path(_Mesh(), str(tmp_path))
    no_layers = dataclasses.replace(CONFIG, num_hidden_layers=0)
    TtMiniMaxM3VisionEncoder.from_weights_path(_Mesh(), str(tmp_path), no_layers)
    from_json, from_config = (call.reference for call in built)
    assert from_json.config == CONFIG and from_config.config == no_layers and len(from_config.layers) == 0
    for name, value in SOURCE.state_dict().items():
        assert torch.equal(from_json.state_dict()[name], value), name
    with expect_error(ValueError, "weights_path must be a MiniMax-M3 checkpoint directory"):
        TtMiniMaxM3VisionEncoder.from_weights_path(_Mesh(), "")


TOKENS = SimpleNamespace(shape=(6, 40), dtype=ttnn.bfloat16)  # forward's output for PIXELS, as far as the checks see

# id: (tokens, out, error, message)
BAD_OUTS = {
    "out_list": (TOKENS, [[0] * 40] * 6, TypeError, "numpy array of uint16, int16, bfloat16"),
    "out_float16": (TOKENS, np.zeros((6, 40), dtype=np.float16), TypeError, "got float16"),
    "out_float32": (TOKENS, np.zeros((6, 40), dtype=np.float32), TypeError, "got float32"),
    "out_shape": (TOKENS, np.zeros((6, 41), dtype=np.uint16), ValueError, r"shape \(6, 40\), got \(6, 41\)"),
    "out_strided": (TOKENS, np.zeros((6, 80), dtype=np.uint16)[:, ::2], ValueError, "C-contiguous False"),
    "out_read_only": (TOKENS, np.zeros((6, 40), dtype=np.uint16), ValueError, "writable False"),
    "tokens_3d": (SimpleNamespace(shape=(1, 6, 40), dtype=ttnn.bfloat16), None, ValueError, "2D bf16"),
    "tokens_float32": (SimpleNamespace(shape=(6, 40), dtype=ttnn.float32), None, ValueError, "2D bf16"),
}


@pytest.mark.parametrize("case", BAD_OUTS)
def test_vision_read_embeddings_rejects(case, expect_error, monkeypatch):
    """read_embeddings raises TypeError for an out that does not hold bf16 values and ValueError for one of another
    shape, not C-contiguous or read-only, and for tokens that are not forward's 2D bf16 output, before it reads."""
    tokens, out, error, message = BAD_OUTS[case]
    out = np.zeros((6, 40), dtype=np.uint16) if out is None else out
    if case == "out_read_only":
        out.setflags(write=False)
    monkeypatch.setattr(ttnn, "get_device_tensors", _device_work)
    with expect_error(error, message):
        read_embeddings(tokens, out)


@pytest.mark.parametrize("dtype", ["uint16", "int16", "bfloat16"])
def test_vision_read_embeddings_copies(dtype, monkeypatch):
    """read_embeddings reads the tokens' one chip with Tensor.cpu and copies their bf16 bits into out, a slice of a
    larger buffer, of dtype uint16, int16 or ml_dtypes' bfloat16, leaving the rest of the buffer as it was."""
    np_dtype = np.dtype(dtype) if dtype != "bfloat16" else np.dtype(pytest.importorskip("ml_dtypes").bfloat16)
    values = torch.randn(6, 40, generator=torch.Generator().manual_seed(0)).bfloat16()
    reads = []

    class Chip:
        def cpu(self):
            reads.append("cpu")
            return SimpleNamespace(to_torch=lambda: values.clone())

    monkeypatch.setattr(ttnn, "get_device_tensors", lambda tokens: [Chip()] if tokens is TOKENS else [])
    buffer = np.full(8 * 40, 0x5A5A, dtype=np.uint16).view(np_dtype)
    read_embeddings(TOKENS, buffer[40:280].reshape(6, 40))
    assert reads == ["cpu"]
    bits = buffer.view(np.uint16)
    assert np.array_equal(bits[40:280].reshape(6, 40), values.view(torch.uint16).numpy())
    assert (bits[:40] == 0x5A5A).all() and (bits[280:] == 0x5A5A).all()


def test_vision_read_embeddings_host_tensor():
    """With ttnn itself (not a stub), read_embeddings copies a bf16 row-major host tensor bit for bit: the path of
    forward's output after Tensor.cpu, without a chip."""
    if isinstance(ttnn.from_torch, mock.Mock):
        pytest.skip("needs ttnn itself; ttnn is a stand-in here")
    values = torch.randn(6, 40, generator=torch.Generator().manual_seed(1)).bfloat16()
    tokens = ttnn.from_torch(values, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    out = np.zeros((6, 40), dtype=np.uint16)
    read_embeddings(tokens, out)
    assert np.array_equal(out, values.view(torch.uint16).numpy())
