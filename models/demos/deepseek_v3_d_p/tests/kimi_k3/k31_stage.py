# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Kimi-K3.1's deployments as the K3.1 tests share them: the runner's mesh, context, chunk and stage split, the
measured user capacity of each layout, and any stage of either layout built with the K3 manifest's knobs.

Two layouts: PP4 (one 8x4 stage per galaxy, SP=8 on axis 0, 5120-token chunks), and the manifest's default, K3's
4x4 stages (two per galaxy, SP=4 on axis 1, TP=4 on axis 0, 2560-token chunks), eight of them for the 93 layers.
`STAGE_LAYOUTS` holds both; it is the one owner of the layouts in the tests (synthetic.py derives from it)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from loguru import logger

from models.demos.common.prefill.adapter import PrefillRunParams, get_adapter
from models.demos.common.prefill.runners.runner_utils import compute_layer_split
from models.demos.deepseek_v3_d_p.reference.kimi_k3_1_config import KimiK31Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_x_device_params, torus_xy_device_params

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
# The most users the first 4x4 stage (layers 0-11, SP=4 on axis 1) fits at MAX_SEQ_LEN, measured on bh-glx-110-c10u20
# through run_multirank_pcc.sh kimi_k3_1 sc1 (2x[4,4], rank 0 = stage 0 exactly), pass at N and fail at N+1 as K3's
# 4x4 count was (#61481): 96 serves one full 1,049,600-token request; 97 fails in compile, the DSA indexer's score
# buffer (42.6 MB per DRAM bank) finding no contiguous block (largest 38.4 MB of 63.3 MB free): fragmentation, not
# total DRAM. 98 fails there too, 100 at the MoE dispatch buffer, 104 at the KDA slabs, 112 and up at the kvpe cache.
# K3 allocates two slots under its limit (127 of 129); K3.1 allocates its measured maximum, at 4x4 and 8x4.
STAGE0_4X4_MAX_USERS = 96

# The 4x4 stage of kimi_k3_1.json: SP on the tray half's axis-1 ring, TP on axis 0.
MESH_4X4 = (4, 4)
SP_AXIS_4X4, TP_AXIS_4X4 = 1, 0
CHUNK_4X4 = 2560
NUM_STAGES_4X4 = 8
# (first layer, layer count) of each 4x4 stage: twelve layers (one AttnRes block) each, nine on the last.
STAGES_4X4 = compute_layer_split(
    KimiK31Config.NUM_LAYERS, NUM_STAGES_4X4, ADAPTER.layer_split_boundaries(KimiK31Config.NUM_LAYERS)
)


@dataclass(frozen=True)
class StageLayout:
    """One K3.1 layout's stages: the mesh axis SP runs on (TP on the other), the chunk every stage prefills, and
    each stage's (first layer, layer count)."""

    sp_axis: int
    # 640 tokens per chip on both layouts: the KDA recurrence's sequence quantum,
    # TILE(32) * KDA_SUMMARY_GROUP_CHUNKS(20) * SP.
    chunk: int
    stages: tuple[tuple[int, int], ...]

    @property
    def tp_axis(self) -> int:
        return 1 - self.sp_axis


# Mesh shape -> layout: one Galaxy as one 8x4 stage, or one tray half as one 4x4 stage.
STAGE_LAYOUTS = {
    MESH: StageLayout(sp_axis=0, chunk=CHUNK, stages=tuple(STAGES)),
    MESH_4X4: StageLayout(sp_axis=SP_AXIS_4X4, chunk=CHUNK_4X4, stages=tuple(STAGES_4X4)),
}
assert STAGE_LAYOUTS[MESH_4X4].tp_axis == TP_AXIS_4X4

# The runner's mesh for each layout, at the adapter's fabric payload and L1_SMALL and the default trace region:
# 8x4 Ring/Ring; 4x4 Linear/Ring (FABRIC_2D_TORUS_X), which the runner opens for an SP-on-axis-1 rank. The 4x4
# row needs the Galaxy narrowed to one tray half (TT_VISIBLE_DEVICES and the 4x4 descriptor, set by
# tests/run_4x4_stage_tests.sh); unnarrowed, the system mesh is 8x4 and requires_mesh_topology skips it.
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
    ),
    pytest.param(
        MESH_4X4,
        torus_x_device_params(
            fabric_payload_size=ADAPTER.fabric_payload_size(),
            l1_small_size=ADAPTER.l1_small_size,
            trace_region_size=TRACE_REGION_SIZE,
        ),
        marks=pytest.mark.requires_mesh_topology(mesh_shape=MESH_4X4, topology="mesh-4x4"),
        id="4x4-sp1",
    ),
]


def weight_cache_path(mesh_shape: tuple) -> Path:
    """The K3.1 TTNN weight cache the runner reads for this layout; skips when there is none."""
    path = ADAPTER.weight_cache_path(mesh_shape, STAGE_LAYOUTS[mesh_shape].sp_axis)
    if not any(path.resolve().glob(".layer_*.complete")):
        pytest.skip(f"no K3.1 TTNN weight cache at {path}")
    return path.resolve()


def stage_params(mesh_shape: tuple, stage: int, max_seq_len: int):
    """(hf_config, PrefillRunParams) for one stage of the layout on `mesh_shape`, one user, eager, with the K3
    manifest's knobs as the runner sets them; stage 0 is the first rank, the layout's last stage the last."""
    stages = STAGE_LAYOUTS[mesh_shape].stages
    return window_params(
        mesh_shape, *stages[stage], max_seq_len, is_first_rank=stage == 0, is_last_rank=stage == len(stages) - 1
    )


def window_params(
    mesh_shape: tuple,
    first_layer_idx: int,
    num_layers: int,
    max_seq_len: int,
    *,
    is_first_rank: bool,
    is_last_rank: bool,
):
    """`stage_params` for any window of layers on the layout's mesh, e.g. two 4x4 stages' layers as one
    undivided stack (K3's undivided 4x4 ladder)."""
    layout = STAGE_LAYOUTS[mesh_shape]
    hf_config = ADAPTER.load_hf_config()
    hf_config.max_seq_len = max_seq_len
    params = PrefillRunParams(
        mesh_shape=mesh_shape,
        num_layers=num_layers,
        first_layer_idx=first_layer_idx,
        is_first_rank=is_first_rank,
        is_last_rank=is_last_rank,
        max_seq_len=max_seq_len,
        chunk_size=layout.chunk,
        num_users=1,
        capacity_factor=8,  # the runner's PREFILL_CAPACITY_FACTOR default
        num_links=2,
        gate_mode_name=ADAPTER.default_gate_mode,
        kv_only_last_layer=False,
        weight_cache_path=weight_cache_path(mesh_shape),
        sp_axis=layout.sp_axis,
        tp_axis=layout.tp_axis,
        tp_shard_kv=True,
        sparse_kv_cache_format=ADAPTER.default_sparse_kv_cache_format,
        use_trace=False,
    )
    logger.info(
        f"K3.1 {mesh_shape[0]}x{mesh_shape[1]} layers [{first_layer_idx}, {first_layer_idx + num_layers}) first={is_first_rank} last={is_last_rank} "
        f"sp_axis={layout.sp_axis} chunk={layout.chunk} max_seq_len={max_seq_len} "
        f"kvpe={params.sparse_kv_cache_format} cache={params.weight_cache_path}"
    )
    return hf_config, params
