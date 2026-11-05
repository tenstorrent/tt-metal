# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Host-side checks of M3VLVisionModel.from_state_dict (tt/vision/model.py), which builds the vision model from weights
under transformers 5.12's names, as loader.load_vision_state_dict reads the checkpoint's. No device: ttnn.from_torch is
a stand-in that records what it gets. The tests check that:
  - from_state_dict hands ttnn.from_torch the same tensors, bit for bit, in the same order and with the same arguments,
    and builds the same model, with no torch module, as the sub-builders' from_torch (M3VLPatchEmbed, M3VLLayerNorm,
    M3VLBlock, M3VLProjector) do from Hugging Face's MiniMaxM3VLVisionModel and MiniMaxM3VLMultiModalProjector holding
    the same weights. The state_dict holds fp32 or bf16 weights, which the device gets in bf16 either way, a patch
    embedding in the Conv3d layout or flattened, a layer past the config's and a language model weight. The configs
    have head_dim 80, the checkpoint's, which the device pads to 96, and 64, which it does not pad; the model is built
    with the projector in bf16, and without it in bfloat8_b, also from the tower's weights alone;
  - from_state_dict uploads the same for fp32 weights as for their bf16 rounding, so reading the checkpoint as bf16
    loses nothing;
  - from_state_dict raises ValueError naming the weights, before any upload, for a state_dict with a vision weight that
    the model has no place for, without a weight that it needs, or with a weight of another shape than the config's.

Needs transformers with minimax_m3_vl (tt-metal pins 5.12.1) for Hugging Face's modules. These tests have no mesh id;
the 1-card CI leg selects them by name (its -k includes "test_vision_").

Run:
    pytest models/demos/minimax_m3/tests/unit/test_vision_from_state_dict.py
"""
from __future__ import annotations

import dataclasses
import hashlib

import pytest
import torch

import ttnn
from models.common.tensor_utils import nearest_32
from models.demos.minimax_m3.tt.vision.block import M3VLBlock
from models.demos.minimax_m3.tt.vision.config import MiniMaxM3VisionConfig
from models.demos.minimax_m3.tt.vision.layernorm import M3VLLayerNorm
from models.demos.minimax_m3.tt.vision.model import M3VLVisionModel
from models.demos.minimax_m3.tt.vision.patch_embed import M3VLPatchEmbed
from models.demos.minimax_m3.tt.vision.projector import M3VLProjector

from .vision_checkpoint import random_vision_weights
from .vision_helpers import hf_vision_modules

# Two layers with the checkpoint's patch geometry. head_dim 80 is the checkpoint's: the device pads it to 96 lanes and
# rotates 78 of them. head_dim 64 is not padded and rotates 60; hd64 also has another LayerNorm eps than the default,
# which the model must take from the config.
CONFIGS = {
    "hd80": MiniMaxM3VisionConfig(
        hidden_size=160,
        intermediate_size=320,
        num_hidden_layers=2,
        num_attention_heads=2,
        projector_hidden_size=96,
        text_hidden_size=64,
    ),
    "hd64": MiniMaxM3VisionConfig(
        hidden_size=256,
        intermediate_size=384,
        num_hidden_layers=2,
        num_attention_heads=4,
        layer_norm_eps=1e-6,
        projector_hidden_size=96,
        text_hidden_size=64,
    ),
}
PATCH_EMBED = "vision_tower.embeddings.proj.weight"
MESH = object()  # the mesh device: only the stand-ins get it


def _state_dict(config: MiniMaxM3VisionConfig, dtype: torch.dtype) -> dict:
    """Seeded random weights as dtype under Hugging Face's names, of a tower with one layer more than config's and the
    projector, and a language model weight."""
    longer = dataclasses.replace(config, num_hidden_layers=config.num_hidden_layers + 1)
    state_dict = {name: value.to(dtype) for name, value in random_vision_weights(longer).items()}
    state_dict["language_model.model.embed_tokens.weight"] = torch.zeros(8, config.text_hidden_size, dtype=dtype)
    return state_dict


def _record_uploads(monkeypatch) -> list:
    """Make ttnn.from_torch a stand-in that records the shape, dtype and sha256 of the bytes of each host tensor that it
    gets, with its other arguments. Returns the list it records into."""
    uploads = []

    def from_torch(tensor, **kwargs):
        data = tensor.contiguous().view(torch.uint8).numpy().tobytes()
        uploads.append((tuple(tensor.shape), tensor.dtype, hashlib.sha256(data).hexdigest(), kwargs))
        return len(uploads) - 1

    monkeypatch.setattr(ttnn, "from_torch", from_torch)
    monkeypatch.setattr(ttnn, "ReplicateTensorToMesh", lambda mesh_device: ("replicated over", mesh_device))
    return uploads


def _torch_module(*args, **kwargs):
    raise AssertionError("built a torch module")


def _from_hf_modules(config: MiniMaxM3VisionConfig, state_dict: dict, with_projector: bool, dtype) -> M3VLVisionModel:
    """The model that the sub-builders' from_torch build from Hugging Face's tower and projector holding the weights of
    state_dict (vision_helpers.hf_vision_modules), the projector's only with_projector."""
    tower, projector = hf_vision_modules(config, state_dict)
    padded_head_dim = nearest_32(config.head_dim)
    return M3VLVisionModel(
        mesh_device=MESH,
        config=config,
        padded_head_dim=padded_head_dim,
        patch_embed=M3VLPatchEmbed.from_torch(MESH, tower.embeddings.proj, dtype=dtype),
        pre_layrnorm=M3VLLayerNorm.from_torch(MESH, tower.pre_layrnorm, dtype=dtype),
        blocks=[M3VLBlock.from_torch(MESH, layer, padded_head_dim, dtype=dtype) for layer in tower.layers],
        projector=M3VLProjector.from_torch(MESH, projector, dtype=dtype) if with_projector else None,
    )


def _structure(model: M3VLVisionModel) -> tuple:
    """What the model holds besides its uploads: the config, the padded head size, each LayerNorm's eps, each
    attention's heads and scale, and the projector's merge factor (None without the projector)."""
    norms = [model.pre_layrnorm] + [norm for block in model.blocks for norm in (block.layer_norm1, block.layer_norm2)]
    return (
        model.config,
        model.padded_head_dim,
        [norm.eps for norm in norms],
        [(block.attention.num_heads, block.attention.scale) for block in model.blocks],
        None if model.projector is None else model.projector.merge_factor,
    )


# The model with the projector in bf16, and without it in bfloat8_b.
BUILDS = pytest.mark.parametrize(
    "with_projector, dtype", [(True, ttnn.bfloat16), (False, ttnn.bfloat8_b)], ids=["projector_bf16", "tower_bfp8"]
)


@torch.no_grad()
@BUILDS
@pytest.mark.parametrize("flat_patch_embed", [False, True], ids=["conv3d", "flat"])
@pytest.mark.parametrize("weights_dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("config_id", CONFIGS)
def test_vision_from_state_dict_uploads(config_id, weights_dtype, flat_patch_embed, with_projector, dtype, monkeypatch):
    """from_state_dict uploads what the sub-builders' from_torch upload for Hugging Face's tower and projector holding
    the same weights, bit for bit, in the same order and with the same arguments, and builds the same model, with no
    torch module. Without the projector it needs none of the projector's weights."""
    config = CONFIGS[config_id]
    state_dict = _state_dict(config, weights_dtype)
    uploads = _record_uploads(monkeypatch)
    expected = _from_hf_modules(config, state_dict, with_projector, dtype)
    expected_uploads = list(uploads)
    if flat_patch_embed:
        state_dict[PATCH_EMBED] = state_dict[PATCH_EMBED].flatten(1)

    uploads.clear()
    with monkeypatch.context() as no_module:
        no_module.setattr(torch.nn.Module, "__init__", _torch_module)
        model = M3VLVisionModel.from_state_dict(MESH, state_dict, config, with_projector=with_projector, dtype=dtype)
    assert len(uploads) == len(expected_uploads), f"{len(uploads)} uploads, from_torch made {len(expected_uploads)}"
    for i, (upload, expected_upload) in enumerate(zip(uploads, expected_uploads)):
        assert upload == expected_upload, f"upload {i} of {len(uploads)} differs from from_torch's"
    assert _structure(model) == _structure(expected)

    if not with_projector:
        uploads.clear()
        tower = {name: value for name, value in state_dict.items() if not name.startswith("multi_modal_projector.")}
        M3VLVisionModel.from_state_dict(MESH, tower, config, with_projector=False, dtype=dtype)
        assert uploads == expected_uploads, "a model without the projector differs when built without its weights"


@torch.no_grad()
@BUILDS
@pytest.mark.parametrize("config_id", CONFIGS)
def test_vision_from_state_dict_reads_bf16(config_id, with_projector, dtype, monkeypatch):
    """from_state_dict uploads the same for fp32 weights as for their bf16 rounding: it rounds each weight to bf16
    before any arithmetic, so the encoder can read the checkpoint as bf16 and lose nothing."""
    config = CONFIGS[config_id]
    fp32 = _state_dict(config, torch.float32)
    bf16 = {name: value.bfloat16() for name, value in fp32.items()}
    assert not all(torch.equal(value, value.bfloat16().float()) for value in fp32.values()), "no weight to round"
    uploads = _record_uploads(monkeypatch)
    M3VLVisionModel.from_state_dict(MESH, fp32, config, with_projector=with_projector, dtype=dtype)
    from_fp32 = list(uploads)
    uploads.clear()
    M3VLVisionModel.from_state_dict(MESH, bf16, config, with_projector=with_projector, dtype=dtype)
    assert uploads == from_fp32, "fp32 weights upload otherwise than their bf16 rounding"


def _set(name: str, shape: tuple):
    """A change to a state_dict that sets name to zeros of shape."""
    return lambda state_dict: state_dict.update({name: torch.zeros(shape)})


def _drop(name: str):
    """A change to a state_dict that removes name."""
    return lambda state_dict: state_dict.pop(name)


HIDDEN, INTERMEDIATE = CONFIGS["hd80"].hidden_size, CONFIGS["hd80"].intermediate_size
# id: (change to a state_dict of CONFIGS["hd80"], message)
BAD_STATE_DICTS = {
    # Weights of a tower with a final LayerNorm, a learned position embedding or a patch embedding bias, of a layer
    # with a packed QKV, and of a projector with a norm: this model has none of them.
    "final_layernorm": (
        _set("vision_tower.post_layernorm.weight", (HIDDEN,)),
        "no place for: vision_tower.post_layernorm.weight$",
    ),
    "position_embedding": (
        _set("vision_tower.embeddings.position_embedding.weight", (16, HIDDEN)),
        "no place for: vision_tower.embeddings.position_embedding.weight$",
    ),
    "patch_embed_bias": (
        _set("vision_tower.embeddings.proj.bias", (HIDDEN,)),
        "no place for: vision_tower.embeddings.proj.bias$",
    ),
    "packed_qkv": (
        _set("vision_tower.layers.1.self_attn.qkv.weight", (3 * HIDDEN, HIDDEN)),
        "no place for: vision_tower.layers.1.self_attn.qkv.weight$",
    ),
    "projector_norm": (
        _set("multi_modal_projector.norm.weight", (64,)),
        "no place for: multi_modal_projector.norm.weight$",
    ),
    "missing_bias": (_drop("vision_tower.layers.1.mlp.fc2.bias"), "lacks .*: vision_tower.layers.1.mlp.fc2.bias$"),
    "missing_merge_linear": (
        _drop("multi_modal_projector.merge_linear_2.weight"),
        "lacks .*: multi_modal_projector.merge_linear_2.weight$",
    ),
    "fc1_transposed": (
        _set("vision_tower.layers.0.mlp.fc1.weight", (HIDDEN, INTERMEDIATE)),
        r"other shapes .*: vision_tower.layers.0.mlp.fc1.weight \(160, 320\), not \(320, 160\)$",
    ),
    # The patch embedding of another kernel of as many values, and flattened to another width.
    "patch_embed_kernel": (
        _set(PATCH_EMBED, (HIDDEN, 3, 1, 28, 14)),
        r"embeddings.proj.weight \(160, 3, 1, 28, 14\), not \(160, 3, 2, 14, 14\)$",
    ),
    "patch_embed_flat_width": (
        _set(PATCH_EMBED, (HIDDEN, 1000)),
        r"embeddings.proj.weight \(160, 1000\), not \(160, 3, 2, 14, 14\)$",
    ),
}


def _device_work(*args, **kwargs):
    raise AssertionError("upload before the weights were checked")


@pytest.mark.parametrize("case", BAD_STATE_DICTS)
def test_vision_from_state_dict_rejects(case, expect_error, monkeypatch):
    """from_state_dict raises ValueError naming the weights, before any upload, for a state_dict with a vision weight
    that the model has no place for, without a weight that it needs, or with a weight of another shape than the
    config's. The state_dict also holds a layer past the config's and a language model weight, which it skips.
    ttnn.from_torch is a stand-in that fails the test if it runs."""
    change, message = BAD_STATE_DICTS[case]
    state_dict = _state_dict(CONFIGS["hd80"], torch.float32)
    change(state_dict)
    monkeypatch.setattr(ttnn, "from_torch", _device_work)
    with expect_error(ValueError, message):
        M3VLVisionModel.from_state_dict(MESH, state_dict, CONFIGS["hd80"])
