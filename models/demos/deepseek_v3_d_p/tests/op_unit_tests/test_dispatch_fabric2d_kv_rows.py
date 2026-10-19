# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""dispatch_fabric2d's transport at KV-row size: is one-hop relaying faster per chip than routed 2D unicast?

Benchmark for the sparse KV fetch design. Each chip sends `seq` tokens to `topk` experts drawn uniformly
inside its 8-chip dispatch group (so every pick crosses the group's fabric ring, 1/8 of them locally), with
rows of 704 B (one scaled-FP8 KVPE row) or the production 14 KB. Reported: device time of one traced launch
(blocking, synced between launches as the op requires) and the bytes each chip injects per second.
"""

import os
import statistics
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tests.op_unit_tests.test_dispatch_fabric2d import (
    _draw_indices,
    _expert_dispatch_table,
)
from models.demos.deepseek_v3_d_p.tests.pcc.mesh_configs import ALL_MESH_CONFIGS
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import compute_constants, extract_mesh_config, get_gate_outputs

NUM_ROUTED_EXPERTS = 256
TOPK = 8
_MESH_CONFIGS = [param for param in ALL_MESH_CONFIGS if param.id == "fabric2d-torus-xy-8x4-2link"]


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="local design benchmark")
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    _MESH_CONFIGS,
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "emb_dim, seq, capacity_factor",
    [(352, 4096, 2), (352, 1024, 2), (7168, 640, 8)],
    ids=["kv704B_4096tok", "kv704B_1024tok", "prod14KB_640tok"],
)
@pytest.mark.timeout(900)
def test_dispatch_fabric2d_kv_rows(mesh_device, device_params, num_links, emb_dim, seq, capacity_factor):
    cfg = extract_mesh_config(mesh_device)
    sp_axis, H, G = cfg.sp_axis, cfg.dispatch_group_size, cfg.num_dispatch_groups
    experts_per_chip, metadata_len, capacity, _ = compute_constants(
        seq,
        NUM_ROUTED_EXPERTS,
        TOPK,
        mesh_device.get_num_devices(),
        H,
        capacity_factor,
        experts_per_chip_override=NUM_ROUTED_EXPERTS // G // H,
        emb_dim=emb_dim,
    )
    torch.manual_seed(0)
    table = _expert_dispatch_table(NUM_ROUTED_EXPERTS, H, G)
    # Every pick stays in the token's own group, spread evenly over its chips: all traffic rides the ring.
    indices = _draw_indices(G, H, seq, TOPK, NUM_ROUTED_EXPERTS, in_group_share=1.0, hot_weight=1.0)

    offs = torch.zeros(G, H, NUM_ROUTED_EXPERTS, dtype=torch.int32)
    counts = torch.zeros(G, H, NUM_ROUTED_EXPERTS, dtype=torch.int32)
    region = torch.zeros(G, H, NUM_ROUTED_EXPERTS, dtype=torch.int32)
    for g in range(G):
        o, c, r, _ = get_gate_outputs(
            indices[g], H, NUM_ROUTED_EXPERTS, experts_per_chip, seq, TOPK, expert_dispatch_table=table[g : g + 1]
        )
        offs[g], counts[g], region[g] = o[0].to(torch.int32), c[0].to(torch.int32), r[0].to(torch.int32)

    def shard(t, dims, dtype):
        return ttnn.from_torch(
            t,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    tt_x = shard(torch.randn(H, G, seq, emb_dim, dtype=torch.bfloat16), (0, 1), ttnn.bfloat16)
    tt_idx = shard(indices.permute(1, 0, 2, 3).to(torch.int32).to(torch.int16), (0, 1), ttnn.uint16)
    tt_table = shard(table.unsqueeze(1), (None, 0), ttnn.int32)
    tt_offs = shard(offs, (None, 0), ttnn.int32)
    tt_counts = shard(counts[:, 0:1, :], (None, 0), ttnn.int32)
    tt_region = shard(region[:, 0:1, :], (None, 0), ttnn.int32)

    def dispatch():
        return ttnn.experimental.deepseek_prefill.dispatch_fabric2d(
            tt_x,
            tt_idx,
            tt_offs,
            tt_table,
            tt_counts,
            tt_region,
            experts_per_chip=experts_per_chip,
            num_routed_experts=NUM_ROUTED_EXPERTS,
            num_experts_per_tok=TOPK,
            metadata_len=metadata_len,
            max_dispatch_buffer_token_size=capacity,
            seq_len_per_chip=seq,
            cluster_axis=sp_axis,
            num_links=num_links,
            topology=ttnn.Topology.Ring,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    dispatch()
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    dispatch()
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    samples = []
    for _ in range(10):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        samples.append(time.perf_counter() - t0)
    ttnn.release_trace(mesh_device, tid)
    t = statistics.median(samples)

    row_bytes = emb_dim * 2
    sent = seq * TOPK * (H - 1) / H  # row copies each chip sends off-chip
    logger.info(
        f"DISPATCH_KV_ROWS row={row_bytes}B seq={seq} topk={TOPK} group={H} links={num_links}: "
        f"{t * 1e6:.0f} us, {sent:.0f} rows off-chip per chip, {sent * row_bytes / t / 1e9:.1f} GB/s injected per chip, "
        f"{t * 1e6 / (sent / (2 * 4)):.3f} us per row per (link, direction)"
    )
