# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Unified dispatch + combine replay test for an 8x1 mesh (LoudBox).

Captures the gate's `indices` tensor on Galaxy 8x4 via the dispatch_module hook
(`TT_DS_CAPTURE_DISPATCH_LAYERS=...`). Then on LB this test:

  1. Loads `L<NN>/indices.pt`.
  2. For each Galaxy column k ∈ {0, 1, 2, 3}, reconstructs from indices:
       - `expert_dispatch_table` = full ExpertMapping table, row k
       - `expert_offsets`, `expert_token_counts`, `expert_region_offsets`
         via `get_gate_outputs(indices, ..., expert_dispatch_table=col_k_table)`
       - Synthesized `x` (random bf16) and `weights` (zeros bf16) — values
         don't drive kernel cycle count
  3. Pushes everything to the 8x1 LB mesh with production-matching mappers.
  4. Constructs TtDispatchModule + TtCombineModule with col-k-equivalent constants
     (num_routed_experts=256, experts_per_chip=8 explicit override).
  5. Runs `warmup + timed` iterations of (dispatch -> layout convert -> combine),
     letting the device profiler capture both DispatchDeviceOperation and
     CombineDeviceOperation rows.

This replaces the older combine-only flow (`test_combine_replay.py`):
  - smaller captures (just `indices`, KB-sized vs MB-GB for metadata)
  - no `src_chip` global-coord remap needed (dispatch on LB writes its own
    correct LB-local src_chip values into metadata)
  - measures both dispatch and combine in one run

## Usage

Single combo (sanity check):
    TT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \\
    TT_METAL_DEVICE_PROFILER=1 \\
    TT_DS_DISPATCH_CAPTURE_DIR=/localdev/nmilicevic/dispatch_captures \\
    python -m pytest -v \\
      models/demos/deepseek_v3_d_p/tests/perf/test_dispatch_combine_replay.py \\
      -k "L05 and col0 and linear-8-2link"

For full sweep, use `run_dispatch_combine_replay_sweep.py`.
"""

import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
    ExpertMapping,
    create_fabric_router_config,
    get_dispatch_input_mesh_mapper,
    get_expert_token_counts_mesh_mapper,
    get_gate_outputs,
    get_max_payload_size,
)
from models.demos.deepseek_v3_d_p.tt.moe.tt_combine import TtCombineModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_dispatch import TtDispatchModule


def _capture_root() -> Path:
    return Path(
        os.getenv(
            "TT_DS_DISPATCH_CAPTURE_DIR",
            str(Path(os.getenv("TT_METAL_HOME", ".")) / "generated" / "dispatch_capture"),
        )
    )


def _list_captures():
    base = _capture_root()
    if not base.exists():
        return []
    return sorted(str(p) for p in base.glob("L*/indices.pt"))


_CAPTURE_FILES = _list_captures()


@pytest.mark.parametrize(
    "capture_file",
    _CAPTURE_FILES or ["NONE"],
    ids=lambda p: f"{Path(p).parent.name}" if p != "NONE" else "NONE",
)
@pytest.mark.parametrize("galaxy_col", [0, 1, 2, 3], ids=["col0", "col1", "col2", "col3"])
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    [
        pytest.param(
            (8, 1),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
            },
            2,
            ttnn.Topology.Linear,
            id="linear-8-2link",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.timeout(0)
def test_dispatch_combine_replay(mesh_device, num_links, topology, capture_file, galaxy_col):
    """Run dispatch + combine for one (layer × galaxy_col) combo. num_links is fixed at 2 (Blackhole)."""
    if capture_file == "NONE":
        pytest.skip(
            f"No capture files found under {_capture_root()}. "
            "Set TT_DS_DISPATCH_CAPTURE_DIR or run the dispatch capture step on Galaxy first."
        )

    blob = torch.load(capture_file, weights_only=False, map_location="cpu")
    indices = blob["indices"]  # (dgs, seq_len, topk) int32 — values in [0, num_routed_experts) Galaxy global IDs.
    #                           Used as-is on LB; the (1, 256) col-k dispatch_table row's -1 entries
    #                           handle skipping out-of-col routings, matching Galaxy col k 1:1.
    cfg = blob["config"]
    layer_idx = cfg["layer_idx"]

    dgs = cfg["dispatch_group_size"]
    ndg_galaxy = cfg["num_dispatch_groups"]
    epc = cfg["experts_per_chip"]
    nre = cfg["num_routed_experts"]  # 256 — kept Galaxy-global on LB; the table's -1 entries skip non-col-k experts.
    nept = cfg["num_experts_per_tok"]  # 8 — same as Galaxy
    sl = cfg["seq_len_per_chip"]
    emb_dim = cfg["emb_dim"]
    mlen = cfg["metadata_len"]
    mdbts = cfg["max_dispatch_buffer_token_size"]

    # LB replay design: TRUE 1:1 with Galaxy col k via skip-by-table mechanism.
    #   - num_routed_experts = 256 (Galaxy global) — the expert-ID INDEXING space, NOT the
    #     number of physical experts hosted on LB. LB physically hosts experts_per_chip * dgs = 64
    #     (8 per chip across 8 chips), same as Galaxy col k.
    #   - experts_per_chip = 8 explicit (overrides compute_constants which would derive 256/8=32).
    #   - dispatch_table = col k's row of ExpertMapping.create_dispatch_table(256, 8, 4) →
    #     (1, 256) with 64 valid chip IDs (in [0, 8)) for experts at global IDs [k*64, (k+1)*64),
    #     and -1 for the other 192 experts. Kernel skips -1 entries — identical to Galaxy col k.
    #   - Indices: captured Galaxy values [0, 256) used as-is. No remap. The same indices that
    #     drove the production gate→dispatch on Galaxy col k drive LB col k replay.
    #
    # Caveat (untested kernel config): on an 8-chip mesh, num_routed_experts=256 +
    # experts_per_chip=8 + dispatch_group_size=8 doesn't appear in any CI parametrize. If the
    # kernel asserts num_routed_experts == experts_per_chip * num_devices_physical anywhere
    # internal, it could fail. First-run risk: hang at synchronize_device.

    assert galaxy_col < ndg_galaxy, f"galaxy_col={galaxy_col} but Galaxy had only {ndg_galaxy} dispatch groups"

    logger.info(f"[replay] file={capture_file}  layer={layer_idx}  galaxy_col={galaxy_col}")
    logger.info(
        f"[replay] config: dgs={dgs} num_routed_experts={nre} topk={nept} experts_per_chip={epc} "
        f"seq_len_per_chip={sl} emb_dim={emb_dim}  num_links={num_links}  topology={topology}"
    )

    # --- 1. Build full Galaxy (4, 256) dispatch_table; take col k's row as LB's (1, 256) table.
    #        Row k has chip IDs [0, 8) only for experts in [k*64, (k+1)*64); all other 192 entries
    #        are -1 (kernel skips). Identical semantics to Galaxy col k.
    full_table = ExpertMapping.create_dispatch_table(
        num_routed_experts=nre,
        dispatch_group_size=dgs,
        num_dispatch_groups=ndg_galaxy,
    )  # (4, 256)
    col_table = full_table[galaxy_col : galaxy_col + 1].contiguous()  # (1, 256)
    n_valid = int((col_table >= 0).sum().item())
    logger.info(
        f"[replay] col{galaxy_col} dispatch_table: {n_valid} valid entries / {nre} total "
        f"({nre - n_valid} entries are -1 → kernel skips those routings)"
    )

    # --- 2. Compute offsets/counts/region_offsets from indices + col-k table.
    #        Routings to experts with table entry == -1 contribute 0 to counts/offsets,
    #        so buffer layout exactly matches what Galaxy col k saw.
    logger.info(f"[replay] computing gate outputs (col{galaxy_col})...")
    col_offsets, col_counts, col_region_offsets, _ = get_gate_outputs(
        indices=indices,
        dispatch_group_size=dgs,
        num_routed_experts=nre,
        experts_per_chip=epc,
        seq_len_per_chip=sl,
        num_experts_per_tok=nept,
        expert_dispatch_table=col_table,
    )  # each (1, dgs, nre)
    col_routings = int(col_counts.sum().item()) // dgs
    total_topk_slots = indices.numel()
    logger.info(
        f"[replay] col{galaxy_col} actual routings = {col_routings} / {total_topk_slots} topk slots "
        f"({100.0 * col_routings / total_topk_slots:.1f}% — matches Galaxy col {galaxy_col} 1:1)"
    )

    # --- 3. Synthesize x and weights (values don't drive kernel cycle count) ---
    logger.info(f"[replay] allocating host x ({dgs}, {sl}, {emb_dim}) bf16 and weights ({dgs}, {sl}, {nept}) bf16...")
    x = torch.randn(dgs, sl, emb_dim, dtype=torch.bfloat16)
    weights = torch.zeros(dgs, sl, nept, dtype=torch.bfloat16)

    # --- Push to LB mesh ---
    sp_axis = 0
    input_mapper = get_dispatch_input_mesh_mapper(mesh_device, sp_axis)
    logger.info(f"[replay] pushing x to device (TILE bf16)...")
    tt_x = ttnn.from_torch(
        x, mesh_mapper=input_mapper, layout=ttnn.TILE_LAYOUT, device=mesh_device, dtype=ttnn.bfloat16
    )
    logger.info(f"[replay] pushing weights to device (ROW_MAJOR bf16)...")
    tt_weights = ttnn.from_torch(
        weights, mesh_mapper=input_mapper, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, dtype=ttnn.bfloat16
    )
    logger.info(f"[replay] pushing captured indices (ROW_MAJOR int32, Galaxy global values in [0, {nre}))...")
    tt_indices = ttnn.from_torch(
        indices, mesh_mapper=input_mapper, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, dtype=ttnn.int32
    )

    logger.info(f"[replay] sharding col{galaxy_col} dispatch_table and expert_offsets...")
    tt_table = TtDispatchModule.shard_expert_dispatch_table(mesh_device, col_table, dispatch_axis=sp_axis)
    tt_offsets = TtDispatchModule.shard_expert_offsets(mesh_device, col_offsets)

    counts_mapper = get_expert_token_counts_mesh_mapper(mesh_device)
    logger.info(f"[replay] pushing counts and region_offsets...")
    tt_counts = ttnn.from_torch(
        col_counts, mesh_mapper=counts_mapper, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, dtype=ttnn.int32
    )
    tt_region_offsets = ttnn.from_torch(
        col_region_offsets,
        mesh_mapper=counts_mapper,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.int32,
    )

    # --- Construct modules ---
    # LB config = true Galaxy col-k mirror:
    #   num_routed_experts=256 (Galaxy global), experts_per_chip=8 explicit override,
    #   dispatch_group_size=8, num_experts_per_tok=8.
    #   Buffer layout per chip = 8 expert regions = same as Galaxy col k.
    logger.info(f"[replay] constructing TtDispatchModule (num_links={num_links}, topology={topology})...")
    dispatch_module = TtDispatchModule(
        mesh_device=mesh_device,
        dispatch_group_size=dgs,
        experts_per_chip=epc,
        num_routed_experts=nre,
        num_experts_per_tok=nept,
        metadata_len=mlen,
        max_dispatch_buffer_token_size=mdbts,
        seq_len_per_chip=sl,
        emb_dim=emb_dim,
        cluster_axis=sp_axis,
        num_links=num_links,
        topology=topology,
        layer_idx=layer_idx,
    )
    logger.info(f"[replay] constructing TtCombineModule...")
    combine_module = TtCombineModule(
        mesh_device=mesh_device,
        dispatch_group_size=dgs,
        num_dispatch_groups=1,
        experts_per_chip=epc,
        num_experts_per_tok=nept,
        seq_len_per_chip=sl,
        cluster_axis=sp_axis,
        num_links=num_links,
        topology=topology,
        init_zeros=True,
        layer_idx=layer_idx,
    )
    logger.info(f"[replay] modules ready")

    n_warmup = int(os.getenv("TT_DS_REPLAY_WARMUP", "2"))
    n_timed = int(os.getenv("TT_DS_REPLAY_TIMED", "5"))

    def _run_one_iter():
        # Dispatch -> layout convert (mirroring production tt_moe.py:468-489) -> combine.
        dispatched_buffer, metadata = dispatch_module(tt_x, tt_weights, tt_indices, tt_offsets, tt_table)
        # In production, the buffer is squeezed to 2D, tile-converted to bfp8, fed to routed FFN
        # (which writes back in-place), then unsqueezed to 4D and fed to combine. For perf replay
        # we skip the FFN but match the layout transforms so combine sees production-shape input.
        buf_2d = ttnn.squeeze(ttnn.squeeze(dispatched_buffer, dim=0), dim=0)
        buf_tiled = ttnn.to_layout(buf_2d, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
        buf_for_combine = ttnn.unsqueeze(ttnn.unsqueeze(buf_tiled, dim=0), dim=0)
        _ = combine_module(buf_for_combine, metadata, tt_counts, tt_region_offsets)

    logger.info(f"[replay] starting {n_warmup} warmup iters...")
    for i in range(n_warmup):
        logger.info(f"[replay]   warmup iter {i + 1}/{n_warmup}: launching dispatch+combine on host...")
        _run_one_iter()
        logger.info(f"[replay]   warmup iter {i + 1}/{n_warmup}: host launches returned")
    logger.info(f"[replay] warmup queued; calling synchronize_device...")
    ttnn.synchronize_device(mesh_device)
    logger.info(f"[replay] warmup sync returned (device drained)")

    logger.info(f"[replay] starting {n_timed} timed iters...")
    for i in range(n_timed):
        logger.info(f"[replay]   timed iter {i + 1}/{n_timed}: launching dispatch+combine on host...")
        _run_one_iter()
        logger.info(f"[replay]   timed iter {i + 1}/{n_timed}: host launches returned")
    logger.info(f"[replay] timed queued; calling synchronize_device...")
    ttnn.synchronize_device(mesh_device)
    logger.info(f"[replay] timed sync returned (device drained)")

    logger.info(f"[replay] L{layer_idx} col{galaxy_col} done ({n_warmup} warmup + {n_timed} timed)")
