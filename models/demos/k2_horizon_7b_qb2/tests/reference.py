# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""HF reference pieces shared by the decoder test and the long-context stream check.

Weights come from the pinned checkpoint (`tt/model.py`): one decoder layer's
shards for the HF reference layer, and the embedding table for realistic inputs.
"""

import math

import torch
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

import ttnn

from ..tt.model import HF_MODEL as MODEL
from ..tt.model import HF_REVISION as REVISION
from ..tt.model import Checkpoint


def load_reference(layer_idx=0):
    """Config, the layer's state dict, the HF decoder layer holding it, and the HF RoPE module."""
    config = AutoConfig.from_pretrained(MODEL, revision=REVISION, trust_remote_code=True)
    config._attn_implementation = "sdpa"
    layer_cls = get_class_from_dynamic_module("modeling_k2_horizon.K2HorizonDecoderLayer", MODEL, revision=REVISION)
    rope_cls = get_class_from_dynamic_module("modeling_k2_horizon.K2HorizonRotaryEmbedding", MODEL, revision=REVISION)
    prefix = f"model.layers.{layer_idx}."
    state = Checkpoint().load(prefix)
    with torch.device("meta"):
        layer = layer_cls(config, layer_idx=layer_idx)
    layer.load_state_dict({k.removeprefix(prefix): v for k, v in state.items()}, assign=True)
    layer.eval()
    return config, state, layer, rope_cls(config)


def real_activations(count):
    """Embeddings of a repeated natural-text prompt: reproducible, non-Gaussian layer inputs."""
    checkpoint = Checkpoint()
    text = (
        "A research team studies how light travels through water. They record the measurements, "
        "compare the results with a mathematical model, and explain the uncertainty. "
        "Write a Python function that computes a running average. The history of cities "
        "includes migration, trade, science, and changing forms of government. "
    )
    ids = checkpoint.tokenizer.encode(text, add_special_tokens=False)
    ids = torch.tensor((ids * math.ceil(count / len(ids)))[:count])
    embeddings = checkpoint.load("model.embed_tokens.")["model.embed_tokens.weight"]
    return embeddings[ids].bfloat16()


def pcc(actual, expected):
    a, b = actual.float().reshape(-1), expected.float().reshape(-1)
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def upload(x, mesh, *, shard=None, integer=False):
    return ttnn.from_torch(
        x.contiguous(),
        device=mesh,
        dtype=ttnn.int32 if integer else ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT if integer else ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=shard) if shard is not None else ttnn.ReplicateTensorToMesh(mesh),
    )


def read(x, mesh, shard=None):
    if mesh.get_num_devices() == 1:
        return ttnn.to_torch(x)
    if shard is not None:
        return ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=shard))
    return ttnn.to_torch(ttnn.get_device_tensors(x)[0])


def to_device(x, mesh, integer=False):
    """Hidden states are width-sharded over the mesh; everything else is replicated."""
    return upload(x, mesh, shard=-1 if not integer and x.shape[-1] == 4096 else None, integer=integer)
