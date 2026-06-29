# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Prefill-only KV cache allocation for disaggregated GPT-OSS prefill (`gpt_oss_d_p`).

Use this instead of ``kv_cache.init_kv_cache`` when the runner only performs full-sequence
prefill (``ttnn_prefill_forward``) with:
  * batch size 1 (one user / one slot)
  * non-paged attention (``ttnn.fill_cache``, not ``paged_fill_cache``)
  * replicated KV across the mesh (``ReplicateTensorToMesh``)
  * ``NdShardSpec`` DRAM (32-token chunks round-robin on 8 banks) for migration tables

``gpt_oss_d_p/tt/runners/prefill_runner.py`` follows this profile today:
``create_tt_model(..., create_kv_cache=True)`` with ``paged_attention_config=None``.
"""

import torch

import ttnn
from models.demos.gpt_oss.utils.general_utils import get_cache_file_name
from models.demos.gpt_oss_d_p.utils.kv_cache_table import make_gpt_oss_prefill_kv_memory_config

from .config import AttentionConfig


def init_kv_cache(
    mesh_device,
    config: AttentionConfig,
    cache_dtype=ttnn.bfloat8_b,
    tensor_cache_path=None,
):
    """
    Allocate zero-initialized K and V cache tensors for prefill-only serving.

    Returns ``[k_cache, v_cache]`` with shape per device:
        ``[max_local_batch_size, num_kv_heads // TP, max_seq_len, head_dim]``

    Prefill attention writes via ``ttnn.fill_cache`` (see ``attention/prefill.py``).
    """
    tp = mesh_device.shape[1]
    num_local_kv_heads = config.num_kv_heads // tp
    if config.num_kv_heads % tp != 0:
        raise ValueError(
            f"num_kv_heads ({config.num_kv_heads}) must be divisible by TP ({tp}), "
            f"got num_local_kv_heads={num_local_kv_heads}"
        )

    cache_shape = [
        config.max_local_batch_size,
        num_local_kv_heads,
        config.max_seq_len,
        config.head_dim,
    ]

    mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    kv_mem_config = make_gpt_oss_prefill_kv_memory_config(config.head_dim)

    k_cache = ttnn.as_tensor(
        torch.zeros(cache_shape),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=cache_dtype,
        mesh_mapper=mesh_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, f"k_cache_{cache_shape}"),
        memory_config=kv_mem_config,
    )

    v_cache = ttnn.as_tensor(
        torch.zeros(cache_shape),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=cache_dtype,
        mesh_mapper=mesh_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, f"v_cache_{cache_shape}"),
        memory_config=kv_mem_config,
    )

    return [k_cache, v_cache]
