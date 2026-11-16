# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Kimi-K3.1's PP4 deployment as the K3.1 tests share it: the runner's mesh, context, chunk and stage
split, the measured PP4 user capacity, and PP4 rank 0 built with the K3 manifest's knobs."""

from __future__ import annotations

from pathlib import Path

import pytest
from loguru import logger

from models.demos.common.prefill.adapter import PrefillRunParams, get_adapter
from models.demos.common.prefill.runners.runner_utils import compute_layer_split
from models.demos.deepseek_v3_d_p.reference.kimi_k3_1_config import KimiK31Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params

ADAPTER = get_adapter("kimi_k3_1")
MESH = (8, 4)
NUM_STAGES = 4
# The K3 manifest's context; a multiple of 40960 (allocator 8192 x chunk 5120).
MAX_SEQ_LEN = 1_064_960
CHUNK = 5120
TRACE_REGION_SIZE = 256 * 1024 * 1024  # the runner's PREFILL_TRACE_REGION_SIZE default

# (first layer, layer count) of each PP4 rank: 0-23 / 24-47 / 48-71 / 72-92.
STAGES = compute_layer_split(
    KimiK31Config.NUM_LAYERS, NUM_STAGES, ADAPTER.layer_split_boundaries(KimiK31Config.NUM_LAYERS)
)
# The most users PP4 fits at MAX_SEQ_LEN, measured on bh-glx-110-c10u14 with each stage built and traced as the
# runner builds it (one more fails): stage 0 is the bound at 76; stages 1-3 fit 77 / 77 / 93.
PP4_MAX_USERS = 76

# The runner's mesh: 8x4 torus, the adapter's fabric payload and L1_SMALL, the default trace region.
PLACEMENT = [
    pytest.param(
        MESH,
        torus_xy_device_params(
            fabric_payload_size=ADAPTER.fabric_payload_size(),
            l1_small_size=ADAPTER.l1_small_size,
            trace_region_size=TRACE_REGION_SIZE,
        ),
        marks=pytest.mark.requires_mesh_topology(mesh_shape=MESH, topology="mesh-8x4"),
        id="torus-xy-8x4",
    )
]


def weight_cache_path() -> Path:
    """The K3.1 TTNN weight cache the runner reads; skips when there is none."""
    path = ADAPTER.weight_cache_path(MESH)
    if not any(path.resolve().glob(".layer_*.complete")):
        pytest.skip(f"no K3.1 TTNN weight cache at {path}")
    return path.resolve()


def first_stage_params(max_seq_len: int):
    """(hf_config, PrefillRunParams) for PP4 rank 0 (first, not last rank), one user, eager, with the K3
    manifest's knobs as the runner sets them."""
    first_layer_idx, num_layers = STAGES[0]
    hf_config = ADAPTER.load_hf_config()
    hf_config.max_seq_len = max_seq_len
    params = PrefillRunParams(
        mesh_shape=MESH,
        num_layers=num_layers,
        first_layer_idx=first_layer_idx,
        is_first_rank=True,
        is_last_rank=False,
        max_seq_len=max_seq_len,
        chunk_size=CHUNK,
        num_users=1,
        capacity_factor=8,  # the runner's PREFILL_CAPACITY_FACTOR default
        num_links=2,
        gate_mode_name=ADAPTER.default_gate_mode,
        kv_only_last_layer=False,
        weight_cache_path=weight_cache_path(),
        tp_shard_kv=True,
        sparse_kv_cache_format=ADAPTER.default_sparse_kv_cache_format,
        use_trace=False,
    )
    logger.info(
        f"K3.1 PP4 rank 0 layers [{first_layer_idx}, {first_layer_idx + num_layers}) "
        f"max_seq_len={max_seq_len} kvpe={params.sparse_kv_cache_format} "
        f"cache={params.weight_cache_path}"
    )
    return hf_config, params
