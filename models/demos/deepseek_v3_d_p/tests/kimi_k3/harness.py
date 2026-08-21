# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared device setup for the Kimi-K3 prefill-block tests."""

from __future__ import annotations

from pathlib import Path

import torch

import ttnn
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config, kimi_k3_hf_config
from models.demos.deepseek_v3_d_p.tests.kimi_k3.weights import load_dense_block_state_dict
from models.demos.deepseek_v3_d_p.tt.tt_prefill_block import TtPrefillBlock
from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import init_checker

SP_AXIS = 0
TP_AXIS = 1


def kda_sequence_length(mesh_device: ttnn.MeshDevice) -> int:
    """The shortest sequence the KDA recurrence accepts on this mesh.

    ttKDA groups its chunked recurrence in ``KDA_SUMMARY_GROUP_CHUNKS`` chunks of one tile, and the
    block hands it the production program config, so the per-device sequence must be a whole number
    of groups. Sequence-parallel splitting makes that a constraint on the global length.
    """
    return ttnn.TILE_SIZE * KimiK3Config.KDA_SUMMARY_GROUP_CHUNKS * mesh_device.shape[SP_AXIS]


def shard_activation(mesh_device: ttnn.MeshDevice, activation: torch.Tensor) -> ttnn.Tensor:
    """Place ``[1, 1, T, emb]`` as the block wants it: sequence on SP, embedding on TP."""
    dims: list[int | None] = [None, None]
    dims[SP_AXIS] = 2
    dims[TP_AXIS] = 3
    return ttnn.from_torch(
        activation,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=tuple(dims), mesh_shape=tuple(mesh_device.shape)),
    )


def layer_cache_path(mesh_device: ttnn.MeshDevice, cache_root: Path) -> Path:
    """The weight-cache directory for one mesh placement."""
    mesh_shape = tuple(mesh_device.shape)
    path = cache_root / f"sp{mesh_shape[SP_AXIS]}_tp{mesh_shape[TP_AXIS]}"
    path.mkdir(parents=True, exist_ok=True)
    init_checker(path)
    return path


def build_layer_0(
    mesh_device: ttnn.MeshDevice,
    checkpoint_dir: Path,
    cache_root: Path,
    seq_len: int,
) -> tuple[TtPrefillBlock, dict]:
    """Cache and construct the real-weight layer-0 block, returning it and its torch weights."""
    config = kimi_k3_hf_config(max_seq=seq_len)
    state_dict = load_dense_block_state_dict(checkpoint_dir, layer_idx=0)
    cache_path = layer_cache_path(mesh_device, cache_root)
    if not TtPrefillBlock.check_cache_complete(cache_path, 0, is_dense=True, model_cfg=KimiK3Config):
        TtPrefillBlock.build_ttnn_cache(
            state_dict=state_dict,
            layer_idx=0,
            cache_path=cache_path,
            mesh_device=mesh_device,
            config=config,
            model_cfg=KimiK3Config,
            seq_len=seq_len,
            sp_axis=SP_AXIS,
            tp_axis=TP_AXIS,
        )
    block = TtPrefillBlock(
        mesh_device=mesh_device,
        config=config,
        model_cfg=KimiK3Config,
        state_dict=state_dict,
        layer_idx=0,
        seq_len=seq_len,
        sp_axis=SP_AXIS,
        tp_axis=TP_AXIS,
        weight_cache_path=cache_path,
    )
    return block, state_dict
