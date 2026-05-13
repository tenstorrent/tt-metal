# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Replay captured combine inputs from a Galaxy 8x4 dispatch group on an 8x1 mesh.

This test is meant to run on a standalone 8-chip Blackhole machine (e.g., LoudBox)
to measure the combine kernel's wall-clock duration in isolation, **without** the
inter-column fabric contention present on Galaxy when 4 dispatch groups run the
combine simultaneously. The aim is to isolate the ethernet/fabric component of
combine perf and compare across machines with different ethernet capabilities.

## Workflow

1. On Galaxy 8x4, capture combine inputs for one or more MoE layers:
       TT_DS_CAPTURE_COMBINE_LAYERS=3,30,60 \\
       TT_DS_COMBINE_CAPTURE_DIR=/path/to/captures \\
       TT_METAL_DEVICE_PROFILER=1 ... \\
       python -m pytest models/demos/deepseek_v3_d_p/tests/test_prefill_transformer.py \\
         -k "... 61_layers ... mesh-8x4 ..."

   For each captured layer L this writes:
       /path/to/captures/L<LL>/col<k>.pt   for k in 0..3

2. Copy the capture directory to the target machine (LoudBox).

3. Replay each column on the 8x1 target, varying num_links to sweep ethernet config:
       TT_METAL_DEVICE_PROFILER=1 \\
       TT_DS_COMBINE_CAPTURE_DIR=/path/to/captures \\
       python -m pytest models/demos/deepseek_v3_d_p/tests/perf/test_combine_replay.py
       python -m tracy --process-logs-only

4. Per Galaxy-layer combine time estimate (no-contention LB lower bound):
       layer_L = max over col in 0..3 of (max over 8 chips of combine_kernel_ns)

## Capture file contents

Each `col<k>.pt` is a torch.save dict with:
    "dispatched_buffer"      (1, dispatch_group_size, experts_per_chip,
                              max_dispatched_tokens_per_expert, emb_dim)  bf16
    "dispatched_metadata"    (1, dispatch_group_size, experts_per_chip,
                              max_dispatched_tokens_per_expert, 5)        int32
    "expert_token_counts"    (1, dispatch_group_size, num_routed_experts)  int32
    "expert_region_offsets"  same shape as counts                          int32
    "config"                 dict with dispatch_group_size, num_dispatch_groups=1,
                              experts_per_chip, num_routed_experts (=experts_per_col=64),
                              num_experts_per_tok, seq_len_per_chip, emb_dim,
                              max_dispatched_tokens_per_expert, galaxy_column, layer_idx

The "expert_id" field inside metadata still contains Galaxy global expert IDs
(e.g., 64..127 for column 1). The combine kernel does not consume that field
for routing (it indexes the buffer by chip+local_expert position), so this is
harmless for replay correctness.
"""

import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
    create_fabric_router_config,
    get_ep_mesh_mapper,
    get_expert_token_counts_mesh_mapper,
    get_max_payload_size,
)
from models.demos.deepseek_v3_d_p.tt.moe.tt_combine import TtCombineModule


def _capture_root() -> Path:
    return Path(
        os.getenv(
            "TT_DS_COMBINE_CAPTURE_DIR",
            str(Path(os.getenv("TT_METAL_HOME", ".")) / "generated" / "combine_capture"),
        )
    )


def _list_captures():
    base = _capture_root()
    if not base.exists():
        return []
    return sorted(str(p) for p in base.glob("L*/col*.pt"))


_CAPTURE_FILES = _list_captures()


@pytest.mark.parametrize(
    "capture_file",
    _CAPTURE_FILES or ["NONE"],
    ids=lambda p: f"{Path(p).parent.name}_{Path(p).stem}" if p != "NONE" else "NONE",
)
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    [
        pytest.param(
            (8, 1),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
            },
            1,
            ttnn.Topology.Linear,
            id="linear-8-1link",
        ),
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
# Note: no `requires_mesh_topology` marker. The replay test only needs an 8-chip linear sub-mesh;
# the deepseek conftest's BH-specific skip rule ("only allow tests matching num_devices") would
# otherwise prevent running on Galaxy (32 chips) for verification. The underlying ttnn mesh_device
# fixture opens an 8-chip sub-mesh from whatever physical mesh is available.
@pytest.mark.timeout(0)
def test_combine_replay(mesh_device, num_links, topology, capture_file):
    """Push captured per-column inputs onto an 8x1 mesh and run combine N times for timing."""
    if capture_file == "NONE":
        pytest.skip(
            f"No capture files found under {_capture_root()}. "
            "Set TT_DS_COMBINE_CAPTURE_DIR or run the capture step on Galaxy first."
        )

    blob = torch.load(capture_file, weights_only=False, map_location="cpu")
    cfg = blob["config"]
    layer_idx = cfg["layer_idx"]
    galaxy_col = cfg["galaxy_column"]

    logger.info(f"[replay] file={capture_file}")
    logger.info(
        f"[replay] L{layer_idx} col{galaxy_col}  "
        f"dgs={cfg['dispatch_group_size']}  experts_per_chip={cfg['experts_per_chip']}  "
        f"num_routed_experts={cfg['num_routed_experts']}  topk={cfg['num_experts_per_tok']}  "
        f"isl_per_chip={cfg['seq_len_per_chip']}  emb_dim={cfg['emb_dim']}"
    )
    logger.info(
        f"[replay] meta shape={tuple(blob['dispatched_metadata'].shape)}, "
        f"counts shape={tuple(blob['expert_token_counts'].shape)}, "
        f"num_links={num_links}"
    )

    # Buffer source: if the .pt has real captured bytes (only when TT_DS_CAPTURE_COMBINE_FULL_BUFFER=1
    # at capture time), shard them onto the mesh. Otherwise allocate directly on device with
    # ttnn.empty — zero host RAM, per-chip uninitialized bfloat8_b. Combine kernel only reads
    # buffer bytes to multiply by a (captured) weight and write to a (captured) destination;
    # uninitialized values produce garbage output but the per-iteration cycle count and fabric
    # traffic pattern are identical to a real-data run. This is the same pattern used in
    # test_ttnn_routed_expert.py (the "fast path" branch) — it avoids the 24 GB host allocation
    # that the bf16-on-host route would require at 25K isl.
    if "dispatched_buffer" in blob:
        buffer_source = f"captured real values, host shape {tuple(blob['dispatched_buffer'].shape)} sharded onto mesh"
        tt_buf = ttnn.from_torch(
            blob["dispatched_buffer"],
            mesh_mapper=get_ep_mesh_mapper(mesh_device),
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            dtype=ttnn.bfloat8_b,
        )
    else:
        buf_shape = cfg["buffer_shape"]  # 4D: (1, dgs, max_dispatch_buffer_token_size, emb_dim)
        per_device_shape = [1, 1] + list(buf_shape[2:])  # (1, 1, max_dispatch_buffer_token_size, emb_dim)
        buffer_source = (
            f"ttnn.empty on device, per-device shape {tuple(per_device_shape)} bfloat8_b " f"(no host allocation)"
        )
        tt_buf = ttnn.empty(
            per_device_shape,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
        )
    logger.info(f"[replay] buffer pushed to device: {buffer_source}")

    logger.info(f"[replay] pushing metadata {tuple(blob['dispatched_metadata'].shape)} int32...")
    tt_meta = ttnn.from_torch(
        blob["dispatched_metadata"],
        mesh_mapper=get_ep_mesh_mapper(mesh_device),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.int32,
    )
    logger.info(f"[replay] metadata pushed")

    logger.info(f"[replay] pushing counts {tuple(blob['expert_token_counts'].shape)} int32...")
    tt_counts = ttnn.from_torch(
        blob["expert_token_counts"],
        mesh_mapper=get_expert_token_counts_mesh_mapper(mesh_device),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.int32,
    )
    logger.info(f"[replay] counts pushed")

    logger.info(f"[replay] pushing region_offsets {tuple(blob['expert_region_offsets'].shape)} int32...")
    tt_offsets = ttnn.from_torch(
        blob["expert_region_offsets"],
        mesh_mapper=get_expert_token_counts_mesh_mapper(mesh_device),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.int32,
    )
    logger.info(f"[replay] region_offsets pushed")

    logger.info(
        f"[replay] constructing TtCombineModule (cluster_axis=0, num_links={num_links}, topology={topology})..."
    )
    combine = TtCombineModule(
        mesh_device=mesh_device,
        dispatch_group_size=cfg["dispatch_group_size"],
        num_dispatch_groups=cfg["num_dispatch_groups"],
        experts_per_chip=cfg["experts_per_chip"],
        num_experts_per_tok=cfg["num_experts_per_tok"],
        seq_len_per_chip=cfg["seq_len_per_chip"],
        cluster_axis=0,
        num_links=num_links,
        topology=topology,
        init_zeros=True,
        layer_idx=layer_idx,
    )
    logger.info(f"[replay] TtCombineModule ready")

    n_warmup = int(os.getenv("TT_DS_REPLAY_WARMUP", "3"))
    n_timed = int(os.getenv("TT_DS_REPLAY_TIMED", "10"))

    logger.info(f"[replay] starting {n_warmup} warmup iterations (host launches)...")
    for i in range(n_warmup):
        logger.info(f"[replay]   warmup iter {i + 1}/{n_warmup}: launching combine on host...")
        _ = combine(tt_buf, tt_meta, tt_counts, tt_offsets)
        logger.info(f"[replay]   warmup iter {i + 1}/{n_warmup}: host launch returned")
    logger.info(f"[replay] warmup launches queued; calling synchronize_device (waits for device to finish)...")
    ttnn.synchronize_device(mesh_device)
    logger.info(f"[replay] warmup synchronize_device returned — device drained")

    logger.info(f"[replay] starting {n_timed} timed iterations...")
    for i in range(n_timed):
        logger.info(f"[replay]   timed iter {i + 1}/{n_timed}: launching combine on host...")
        _ = combine(tt_buf, tt_meta, tt_counts, tt_offsets)
        logger.info(f"[replay]   timed iter {i + 1}/{n_timed}: host launch returned")
    logger.info(f"[replay] timed launches queued; calling synchronize_device...")
    ttnn.synchronize_device(mesh_device)
    logger.info(f"[replay] timed synchronize_device returned — device drained")

    logger.info(f"[replay] L{layer_idx} col{galaxy_col} done ({n_warmup} warmup + {n_timed} timed)")
