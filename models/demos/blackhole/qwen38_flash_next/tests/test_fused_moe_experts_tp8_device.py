# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""x37: ``moe_experts`` on the 1x8 TP=8 mesh (all dies, real cached layers) against today's chain, bit for bit per die, and
traced timing.  Today's chain = untilize (+ reshape) + fill + moe_compute exactly as ``Qwen38TTNNMoE._routed_partial`` runs it,
with the model's expert mapping ([8, 512], die d's row); the new program takes die d's owner row (moe_post's).
Run: MESH_DEVICE=P150x8 TT_VISIBLE_DEVICES=0..7 (run_dev.sh with DEV=0,1,2,3,4,5,6,7)."""

import json
import os
from pathlib import Path

import pytest
import torch
from loguru import logger
from ttnn.experimental.moe_compute_utils import auto_output_width_shard_dim, effective_matmul_ring_size
from ttnn.operations.ccl import MoEActivationFunction

import ttnn
from models.demos.blackhole.qwen38_flash_next.tests.moe_experts_support import cache_dir, traced_wall_us
from models.demos.blackhole.qwen38_flash_next.tests.tp_harness import DEVICE_PARAMS
from models.demos.blackhole.qwen38_flash_next.ttnn import decode_matmul as dm
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import moe_experts as me
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import moe_post

BF16, U16, DRAM = ttnn.bfloat16, ttnn.uint16, ttnn.DRAM_MEMORY_CONFIG
HIDDEN, TOP_K, EXPERTS, PER = 2560, 10, 512, 64
LAYERS = [int(v) for v in os.environ.get("X37_LAYERS", "0,11,30").split(",")]
PARAMS = {**DEVICE_PARAMS, "trace_region_size": 64 * 1024 * 1024}


def rep(mesh):
    return ttnn.ReplicateTensorToMesh(mesh)


def routing_case(gen, rows, busy_die, n_busy):
    """Top-10 per row: ``n_busy`` ids of ``busy_die`` (row 1 shares the first of them) and the rest uniformly elsewhere."""
    idx = torch.zeros(rows, TOP_K, dtype=torch.int64)
    first = []
    for r in range(rows):
        lo = busy_die * PER
        pick = (first[:1] if r == 1 and first and n_busy else []) + []
        pool = [e for e in range(lo, lo + PER) if e not in pick]
        pick += [pool[i] for i in torch.randperm(len(pool), generator=gen)[: n_busy - len(pick)].tolist()]
        rest_pool = [e for e in range(EXPERTS) if not lo <= e < lo + PER]
        rest = [rest_pool[i] for i in torch.randperm(len(rest_pool), generator=gen)[: TOP_K - n_busy].tolist()]
        ids = pick + rest
        idx[r] = torch.tensor([ids[i] for i in torch.randperm(TOP_K, generator=gen).tolist()])
        if r == 0:
            first = [e for e in idx[0].tolist() if lo <= e < lo + PER]
    scores = (torch.rand(rows, TOP_K, generator=gen) * 0.5 + 0.05).to(torch.bfloat16)
    return idx, scores


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [PARAMS], indirect=True)
def test_tp8_bitwise_and_timing(mesh_device):
    mesh = mesh_device
    ring = effective_matmul_ring_size(mesh)
    d = max(d for d in range(1, 5) if (HIDDEN // 32) % d == 0 and ring % d == 0)
    drain = ttnn.experimental.get_moe_tilize_drain_core(mesh, 1, auto_output_width_shard_dim(HIDDEN), HIDDEN)
    mapping = (torch.arange(EXPERTS, dtype=torch.int32) // PER).unsqueeze(0).repeat(8, 1).contiguous()
    expert_mapping = ttnn.from_torch(
        mapping, device=mesh, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=U16, memory_config=DRAM, mesh_mapper=rep(mesh)
    )
    owner = ttnn.from_torch(
        moe_post.owner_rows(8),
        device=mesh,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=U16,
        memory_config=DRAM,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=MESH_SHAPE, dims=(None, 0)),
    )
    exclude = me.model_exclude(mesh)
    results = Path(os.environ.get("X37_RESULTS", "/tmp/x37_tp8.jsonl"))
    all_ok = True
    for layer in LAYERS:
        root = cache_dir() / "backbone" / f"layer-{layer:02d}"
        w01 = ttnn.load_tensor(root / "w0_w1_dtype_BFLOAT4_B_layout_TILE.tensorbin", device=mesh)
        w2 = ttnn.load_tensor(root / "w2_dtype_BFLOAT4_B_layout_TILE.tensorbin", device=mesh)
        for rows in (1, 2):
            routing_cfg = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(drain, drain)}), [rows, TOP_K], ttnn.ShardOrientation.ROW_MAJOR
                ),
            )
            hidden_cfg, _ = dm.dram_sharded_matmul_configs(mesh, HIDDEN, EXPERTS, num_cores=5)
            hs = -(-rows // (32 * d))
            combine_ref = ttnn.from_torch(
                torch.zeros(TOP_K, rows, HIDDEN, dtype=torch.bfloat16),
                device=mesh,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=BF16,
                memory_config=DRAM,
                mesh_mapper=rep(mesh),
            )
            combine_new = ttnn.from_torch(
                torch.zeros(TOP_K, rows, HIDDEN, dtype=torch.bfloat16),
                device=mesh,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=BF16,
                memory_config=DRAM,
                mesh_mapper=rep(mesh),
            )
            gen = torch.Generator().manual_seed(100 * layer + rows)
            for busy_die, n_busy in [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 2), (6, 4), (7, 1), (3, 0), (5, 4)]:
                idx, sc = routing_case(gen, rows, busy_die, n_busy)
                hidden = (torch.randn(rows, HIDDEN, generator=gen) * 2.0).to(torch.bfloat16)
                h_tile = ttnn.from_torch(
                    hidden.reshape(1, 1, rows, HIDDEN),
                    device=mesh,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=BF16,
                    memory_config=DRAM,
                    mesh_mapper=rep(mesh),
                )
                h_l1 = ttnn.from_torch(
                    hidden.reshape(1, 1, rows, HIDDEN),
                    device=mesh,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=BF16,
                    memory_config=hidden_cfg,
                    mesh_mapper=rep(mesh),
                )
                i3 = ttnn.reshape(
                    ttnn.from_torch(
                        idx.reshape(1, 1, rows, TOP_K).to(torch.int32),
                        device=mesh,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        dtype=U16,
                        memory_config=DRAM,
                        mesh_mapper=rep(mesh),
                    ),
                    (1, rows, TOP_K),
                )
                s3 = ttnn.reshape(
                    ttnn.from_torch(
                        sc.reshape(1, 1, rows, TOP_K),
                        device=mesh,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        dtype=BF16,
                        memory_config=DRAM,
                        mesh_mapper=rep(mesh),
                    ),
                    (1, rows, TOP_K),
                )
                idx_l1, sc_l1 = ttnn.to_memory_config(i3, routing_cfg), ttnn.to_memory_config(s3, routing_cfg)

                def today():
                    s = ttnn.to_layout(h_tile, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
                    if rows != 1:
                        s = ttnn.reshape(s, (1, rows, HIDDEN))
                    ttnn.fill(combine_ref, 0.0, output_tensor=combine_ref)
                    o = ttnn.experimental.moe_compute(
                        s,
                        idx_l1,
                        sc_l1,
                        expert_mapping,
                        w01,
                        w2,
                        layer_id=0,
                        output_height_shard_dim=hs,
                        intermediate_size=640,
                        has_bias=False,
                        cluster_axis=0,
                        topology=None,
                        num_links=None,
                        mux_core_range_set=None,
                        output_memory_config=DRAM,
                        optional_output_tensor=combine_ref,
                        optional_cross_device_semaphore=None,
                        activation_type=MoEActivationFunction.SILU,
                        compute_only=False,
                        num_shared_experts_per_device=0,
                    )
                    for t in (o[0], o[1], o[2], o[4], s):
                        ttnn.deallocate(t)

                def new():
                    me.moe_experts(h_l1, idx_l1, owner, w01, w2, combine_new, exclude=exclude)

                today()
                ttnn.fill(combine_new, 0.0, output_tensor=combine_new)
                new()
                bad = 0
                for a, b in zip(ttnn.get_device_tensors(combine_ref), ttnn.get_device_tensors(combine_new)):
                    bad += int((ttnn.to_torch(a).view(torch.int16) != ttnn.to_torch(b).view(torch.int16)).sum())
                per_die = [int(((idx // PER) == die).sum()) for die in range(8)]
                row = dict(
                    layer=layer,
                    rows=rows,
                    busy_die=busy_die,
                    n_busy=n_busy,
                    per_die_owned_slots=per_die,
                    differing_words=bad,
                )
                if os.environ.get("X37_TIME", "1") == "1" and layer == LAYERS[0]:
                    row["today_chain_us"] = round(traced_wall_us(mesh, today, replays=50)[0], 2)
                    row["moe_experts_us"] = round(traced_wall_us(mesh, new, replays=50)[0], 2)
                logger.info(json.dumps(row))
                with results.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                all_ok &= bad == 0
                for t in (h_tile, h_l1, i3, s3, idx_l1, sc_l1):
                    ttnn.deallocate(t)
        ttnn.deallocate(w01)
        ttnn.deallocate(w2)
    assert all_ok
