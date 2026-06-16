# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Weight loading for ZImageTransformer TTNN model."""

import torch

import ttnn
from models.demos.z_image_turbo.tt.dit.model_pt import EXTRA_DIM, HEAD_DIM, ORIGINAL_HEADS

_ON_DEVICE = frozenset(
    {
        "all_final_layer.2-1.linear.bias",
        "all_final_layer.2-1.linear.weight",
        "cap_pad_token",
        "x_pad_token",
    }
)


def _shard_type(name):
    """Derive tensor-parallel sharding from parameter name."""
    if name.endswith((".to_q.weight", ".to_k.weight", ".to_v.weight")):
        return "col_par_attn"
    if ".attention.to_out.0.weight" in name:
        return "row_par_attn_out"
    if ".feed_forward." in name:
        if name.endswith((".w1.weight", ".w3.weight")):
            return "col_par_mlp"
        if name.endswith(".w2.weight"):
            return "row_par_mlp"
    return "full"


def _mesh_mapper(stype, mesh_device):
    if stype in ("col_par_attn", "col_par_mlp"):
        return ttnn.ShardTensorToMesh(mesh_device, dim=0)
    if stype in ("row_par_attn_out", "row_par_mlp"):
        return ttnn.ShardTensorToMesh(mesh_device, dim=1)
    return ttnn.ReplicateTensorToMesh(mesh_device)


def _pad_attn_heads(pt, stype):
    """Pad attention weights from 30 → 32 heads for 4-way TP."""
    if stype == "col_par_attn" and pt.shape[0] == ORIGINAL_HEADS * HEAD_DIM:
        return torch.cat([pt, torch.zeros(EXTRA_DIM, pt.shape[1], dtype=pt.dtype)], dim=0)
    if stype == "row_par_attn_out" and pt.shape[1] == ORIGINAL_HEADS * HEAD_DIM:
        return torch.cat([pt, torch.zeros(pt.shape[0], EXTRA_DIM, dtype=pt.dtype)], dim=1)
    return pt


def load_weights(mesh_device, transformer):
    """Load all model weights into a dict keyed by state_dict parameter names.

    On-device weights (final layer, pad tokens) go directly to device.
    All other weights stay on host as ROW_MAJOR for consteval to transform.
    RoPE frequency tables are added under ``__rope_freqs_{F,H,W}__`` keys.
    """
    from diffusers.models.transformers.transformer_z_image import RopeEmbedder

    mem = ttnn.DRAM_MEMORY_CONFIG
    state_dict = transformer.state_dict()
    weights = {}

    for name, param in state_dict.items():
        pt = param.bfloat16()
        stype = _shard_type(name)
        on_device = name in _ON_DEVICE
        pt = _pad_attn_heads(pt, stype)

        kwargs = dict(
            dtype=ttnn.DataType.BFLOAT16,
            layout=ttnn.Layout.TILE if on_device else ttnn.Layout.ROW_MAJOR,
            mesh_mapper=_mesh_mapper(stype, mesh_device),
        )
        if on_device:
            kwargs["device"] = mesh_device
            kwargs["memory_config"] = mem
        weights[name] = ttnn.from_torch(pt, **kwargs)

    rope = transformer.rope_embedder
    freqs = RopeEmbedder.precompute_freqs_cis(rope.axes_dims, rope.axes_lens, getattr(rope, "theta", 256.0))
    replicated = ttnn.ReplicateTensorToMesh(mesh_device)
    for key, ft in [
        ("__rope_freqs_F__", freqs[0]),
        ("__rope_freqs_H__", freqs[1]),
        ("__rope_freqs_W__", freqs[2]),
    ]:
        weights[key] = ttnn.from_torch(
            ft.float(),
            dtype=ttnn.DataType.FLOAT32,
            layout=ttnn.Layout.ROW_MAJOR,
            device=mesh_device,
            memory_config=mem,
            mesh_mapper=replicated,
        )

    return weights
