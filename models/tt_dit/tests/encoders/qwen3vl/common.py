# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Shared helpers and constants for the Qwen3-VL encoder tests.

import contextlib

import pytest
import torch
import transformers

import ttnn

from ....encoders.qwen3vl.model_qwen3vl import Qwen3VlTextEncoder
from ....parallel.config import EncoderParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....utils.tensor import bf16_tensor

# FABRIC_1D on a 1x1 mesh has no ethernet partner and times out router init, so fabric is per-config, CCL-only.
L1_SMALL = 32768
FABRIC = {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": L1_SMALL}
NO_FABRIC = {"l1_small_size": L1_SMALL}

# MiniMax-H3's Qwen3-VL vision tower (also Qwen3-VL-32B's).
HIDDEN_SIZE = 1152
NUM_HEADS = 16
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS  # 72 -- deliberately not tile-aligned
INTERMEDIATE_SIZE = 4304
SPATIAL_MERGE_SIZE = 2
OUT_HIDDEN_SIZE = 5120
NUM_POSITION_EMBEDDINGS = 2304
NORM_EPS = 1e-6
HIDDEN_ACT = "gelu_pytorch_tanh"


def vision_config(depth, **overrides):
    kwargs = dict(
        depth=depth,
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        intermediate_size=INTERMEDIATE_SIZE,
        in_channels=3,
        patch_size=16,
        temporal_patch_size=2,
        spatial_merge_size=SPATIAL_MERGE_SIZE,
        num_position_embeddings=NUM_POSITION_EMBEDDINGS,
        out_hidden_size=OUT_HIDDEN_SIZE,
        hidden_act=HIDDEN_ACT,
        deepstack_visual_indexes=[0],
        initializer_range=0.02,
    )
    kwargs.update(overrides)
    return transformers.Qwen3VLVisionConfig(**kwargs)


VISION_MESH = [
    pytest.param((1, 1), (1, 1), None, None, 1, NO_FABRIC, id="single"),
    pytest.param((8, 4), (8, 4), 0, None, 2, FABRIC, id="tp8_sp1"),
    pytest.param((8, 4), (8, 4), 0, 1, 2, FABRIC, id="tp8_sp4"),
]
VISION_PARAMS = pytest.mark.parametrize(
    ("mesh_device", "submesh_shape", "tp_axis", "sp_axis", "num_links", "device_params"),
    VISION_MESH,
    indirect=["mesh_device", "device_params"],
)


def resolve_parallel(submesh, tp_axis, sp_axis, num_links):
    if tp_axis is None and sp_axis is None:
        return None, None
    shape = tuple(submesh.shape)
    cfg = EncoderParallelConfig(
        tensor_parallel=ParallelFactor(factor=shape[tp_axis] if tp_axis is not None else 1, mesh_axis=tp_axis),
        sequence_parallel=(ParallelFactor(factor=shape[sp_axis], mesh_axis=sp_axis) if sp_axis is not None else None),
    )
    return cfg, CCLManager(submesh, num_links=num_links, topology=ttnn.Topology.Linear)


def skip_if_sp_misaligned(total, submesh, sp_axis):
    """Ring SDPA needs N_local_q % 32 == 0; misalignment is model geometry, not a port bug, so skip."""
    if sp_axis is None:
        return
    sp = tuple(submesh.shape)[sp_axis]
    if total % (sp * 32) != 0:
        pytest.skip(f"{total} patches do not divide into {sp} tile-aligned shards (needs a multiple of {sp * 32})")


def sp_shard(x, submesh, sp_axis):
    if sp_axis is None:
        return bf16_tensor(x, device=submesh)
    return bf16_tensor(x, device=submesh, mesh_axis=sp_axis, shard_dim=0)


@contextlib.contextmanager
def capture_layer_outputs(lm, layers):
    caps: dict[int, torch.Tensor] = {}
    handles = [
        lm.layers[i].register_forward_hook(
            lambda m, i_, o, i=i: caps.__setitem__(i, (o[0] if isinstance(o, tuple) else o).detach())
        )
        for i in layers
    ]
    try:
        yield caps
    finally:
        for h in handles:
            h.remove()


def hf_rope_params(cfg):
    """transformers>=5 has no top-level cfg.rope_theta, so it must not be an (eagerly evaluated) dict.get default."""
    rope_params = getattr(cfg, "rope_parameters", None) or cfg.rope_scaling
    mrope_section = rope_params["mrope_section"]
    rope_theta = rope_params["rope_theta"] if "rope_theta" in rope_params else cfg.rope_theta
    return rope_theta, mrope_section


def encoder_from_hf_config(cfg, **overrides):
    rope_theta, mrope_section = hf_rope_params(cfg)
    kwargs = dict(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        hidden_act="silu",
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        rms_norm_eps=cfg.rms_norm_eps,
        rope_theta=rope_theta,
        mrope_section=mrope_section,
    )
    kwargs.update(overrides)
    return Qwen3VlTextEncoder(**kwargs)
