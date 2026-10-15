# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Device-perf worker for dispatch_fabric2d: one warm-up launch, then ITERATIONS more.

No correctness check here; `op_unit_tests/test_dispatch_fabric2d.py` covers that.

Environment:
    TT_DS_INPUT_LAYOUT     row_major (default) or tile. The model passes tile.
    TT_DS_OUTPUT_DTYPE     bf16 (default) or fp8: the dispatched buffer's dtype. A row_major input is then
                           made FP8_E4M3 too, since the op copies it as is; a tile input stays bf16.
    TT_DS_CAPTURED_LAYER   an integer: use one recorded MoE layer's real routing instead of the
                           synthetic draw.
    TT_DS_CAPTURED_PATH    where that recording lives; defaults to the golden prefill cache.

These are read by this worker's own process, so a harness that launches it must pass them through
its `env=` parameter, not as a prefix on the command.
"""

import os

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.demos.deepseek_v3_d_p.tests.op_unit_tests.test_dispatch_fabric2d import (
    _HALF_TOKEN_PAYLOAD_8X1,
    PRODUCTION_ROUTING,
    _draw_indices,
    _expert_dispatch_table,
)
from models.demos.deepseek_v3_d_p.tests.pcc.mesh_configs import ALL_MESH_CONFIGS
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import compute_constants, extract_mesh_config, get_gate_outputs

# Production prefill geometry: one 5120-token chunk over an 8-chip dispatch group.
CHUNK = 5 * 1024
DISPATCH_GROUP_SIZE = 8
SEQ_LEN_PER_CHIP = CHUNK // DISPATCH_GROUP_SIZE
DISPATCH_BUFFER_CAPACITY_FACTOR = 8
EMB_DIM = 7 * 1024
NUM_ROUTED_EXPERTS = 256
NUM_EXPERTS_PER_TOK = 8

# Launches after the warm-up one.
ITERATIONS = 10

# The production Galaxy row, then the 8x1 LoudBox proxy at a whole-token payload and at the model's half-token
# one, where a bf16 token crosses each hop as two packets and an fp8 one as one.
_MESH_IDS = ("fabric2d-torus-xy-8x4-2link", "fabric2d-torus-y-8x1-2link")
_MESH_CONFIGS = [param for param in ALL_MESH_CONFIGS if param.id in _MESH_IDS] + [_HALF_TOKEN_PAYLOAD_8X1]
assert len(_MESH_CONFIGS) == len(_MESH_IDS) + 1, "dispatch_fabric2d perf mesh configs missing from ALL_MESH_CONFIGS"


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    _MESH_CONFIGS,
    indirect=["mesh_device", "device_params"],
)
# Kept finite: a hang leaves the eth links down, and bringing them back needs admin access.
@pytest.mark.timeout(900)
def test_dispatch_fabric2d_perf_worker(mesh_device, device_params, num_links):
    cfg = extract_mesh_config(mesh_device)
    sp_axis, H, G = cfg.sp_axis, cfg.dispatch_group_size, cfg.num_dispatch_groups
    assert sp_axis == 0, "this op runs on the dispatch axis, which extract_mesh_config puts at 0"
    seq_len_per_chip = SEQ_LEN_PER_CHIP

    experts_per_chip, metadata_len, max_dispatch_buffer_token_size, _ = compute_constants(
        seq_len_per_chip,
        NUM_ROUTED_EXPERTS,
        NUM_EXPERTS_PER_TOK,
        mesh_device.get_num_devices(),
        H,
        DISPATCH_BUFFER_CAPACITY_FACTOR,
        experts_per_chip_override=NUM_ROUTED_EXPERTS // G // H,
        emb_dim=EMB_DIM,
    )

    torch.manual_seed(42)
    table = _expert_dispatch_table(NUM_ROUTED_EXPERTS, H, G)
    in_group_share, hot_weight = PRODUCTION_ROUTING
    if G == 1:
        # One dispatch group has nowhere else to send a pick.
        in_group_share = 1.0
    indices = _draw_indices(G, H, seq_len_per_chip, NUM_EXPERTS_PER_TOK, NUM_ROUTED_EXPERTS, in_group_share, hot_weight)
    routing = "synthetic"
    # Optionally, real routing from one recorded MoE layer: 5120 tokens split as 8 chips x 640. Every
    # dispatch group gets the same picks and keeps only the ones for its own experts.
    captured_layer = os.environ.get("TT_DS_CAPTURED_LAYER")
    if captured_layer is not None:
        from safetensors import safe_open

        path = os.environ.get(
            "TT_DS_CAPTURED_PATH",
            "/mnt/models/deepseek-prefill-cache/golden/longbook_qa_eng_prefill_5120_nopad/expert_routing.safetensors",
        )
        with safe_open(path, "pt") as f:
            ids = f.get_tensor(f"expert_ids_layer_{int(captured_layer)}").to(torch.int64)
        indices = ids.view(H, seq_len_per_chip, NUM_EXPERTS_PER_TOK).unsqueeze(0).expand(G, -1, -1, -1).clone()
        routing = f"captured-L{int(captured_layer)}"
    realized = float((table[torch.arange(G).view(G, 1, 1, 1), indices] != -1).to(torch.float64).mean())

    offs = torch.zeros(G, H, NUM_ROUTED_EXPERTS, dtype=torch.int32)
    counts = torch.zeros(G, H, NUM_ROUTED_EXPERTS, dtype=torch.int32)
    region = torch.zeros(G, H, NUM_ROUTED_EXPERTS, dtype=torch.int32)
    for g in range(G):
        o, c, r, _ = get_gate_outputs(
            indices[g],
            H,
            NUM_ROUTED_EXPERTS,
            experts_per_chip,
            seq_len_per_chip,
            NUM_EXPERTS_PER_TOK,
            expert_dispatch_table=table[g : g + 1],
        )
        offs[g], counts[g], region[g] = o[0].to(torch.int32), c[0].to(torch.int32), r[0].to(torch.int32)

    def shard(t, dims, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
        return ttnn.from_torch(
            t,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims),
            layout=layout,
            device=mesh_device,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    x = torch.randn(H, G, seq_len_per_chip, EMB_DIM, dtype=torch.bfloat16)
    layout = os.environ.get("TT_DS_INPUT_LAYOUT", "row_major")
    assert layout in ("row_major", "tile"), f"TT_DS_INPUT_LAYOUT: expected row_major or tile, got {layout!r}"
    output_dtype = os.environ.get("TT_DS_OUTPUT_DTYPE", "bf16")
    assert output_dtype in ("bf16", "fp8"), f"TT_DS_OUTPUT_DTYPE: expected bf16 or fp8, got {output_dtype!r}"
    fp8_output = output_dtype == "fp8"
    if fp8_output and layout == "row_major":
        # Host conversion to FP8_E4M3 takes a float32 source only; widening bf16 is exact.
        tt_x = ttnn.to_device(
            ttnn.from_torch(
                x.to(torch.float32),
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(0, 1)),
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.fp8_e4m3,
            ),
            mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    else:
        tt_x = shard(x, (0, 1), ttnn.bfloat16, layout=ttnn.TILE_LAYOUT if layout == "tile" else ttnn.ROW_MAJOR_LAYOUT)
    tt_idx = shard(indices.permute(1, 0, 2, 3).to(torch.int32).to(torch.int16), (0, 1), ttnn.uint16)
    tt_table = shard(table.unsqueeze(1), (None, 0), ttnn.int32)
    # Replicated: a relaying chip needs every source chip's row to know how much it forwards.
    tt_offs = shard(offs, (None, 0), ttnn.int32)
    tt_counts = shard(counts[:, 0:1, :], (None, 0), ttnn.int32)
    tt_region = shard(region[:, 0:1, :], (None, 0), ttnn.int32)

    logger.info(
        f"perf worker: mesh={tuple(mesh_device.shape)} seq={seq_len_per_chip} emb={EMB_DIM} "
        f"experts={NUM_ROUTED_EXPERTS} topk={NUM_EXPERTS_PER_TOK} epc={experts_per_chip} "
        f"capacity={max_dispatch_buffer_token_size} links={num_links} iters={ITERATIONS}"
    )
    logger.info(f"routing {routing}: picks landing in this dispatch group {100 * realized:.1f}%")
    logger.info(f"input layout: {layout}, output dtype: {output_dtype}")
    if captured_layer is None:
        # The draw must hit its target share, or the measured traffic silently isn't what we think.
        assert abs(realized - in_group_share) < 0.02, (
            f"the routing draw asked for {100 * in_group_share:.1f}% of picks in this dispatch group "
            f"and drew {100 * realized:.1f}%"
        )

    def fabric2d():
        return ttnn.experimental.deepseek_prefill.dispatch_fabric2d(
            tt_x,
            tt_idx,
            tt_offs,
            tt_table,
            tt_counts,
            tt_region,
            experts_per_chip=experts_per_chip,
            num_routed_experts=NUM_ROUTED_EXPERTS,
            num_experts_per_tok=NUM_EXPERTS_PER_TOK,
            metadata_len=metadata_len,
            max_dispatch_buffer_token_size=max_dispatch_buffer_token_size,
            seq_len_per_chip=seq_len_per_chip,
            cluster_axis=sp_axis,
            num_links=num_links,
            topology=ttnn.Topology.Ring,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            fp8_output=fp8_output,
        )

    fabric2d()  # warm-up, before the signpost
    ttnn.synchronize_device(mesh_device)

    # Synced after every launch so launches never overlap and each chip's device time is its own launch;
    # overlap is the subject of test_dispatch_fabric2d_back_to_back and _overlapped_skew, not this one's.
    signpost("dispatch_fabric2d")
    for _ in range(ITERATIONS):
        fabric2d()
        ttnn.synchronize_device(mesh_device)
    signpost("done")
