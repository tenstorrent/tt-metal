# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MoEBlock (models/demos/moe_block/tt/moe_block.py) end to end on an 8x4 Galaxy mesh: head -> dispatch -> experts +
combine -> tail with the shared arena, on zero / uninitialized weights. Checks that the four ops plan, build and launch
together (the arena spec accepted by the head and the combine, the output's shape) and reports the block's host time a
forward on program-cache hits. No numerics (the ops' own tests check those bit-exact).

Needs an 8x4 Galaxy, or the mock Galaxy (plan, build and launch only):

    TT_METAL_MOCK_CLUSTER_DESC_PATH=$TT_METAL_HOME/tt_metal/third_party/umd/tests/cluster_descriptor_examples/\
blackhole_galaxy.yaml pytest models/demos/moe_block/tests/test_moe_block_smoke.py
"""

import importlib.util
import os
import time
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.moe_block.tt.moe_block import MoEBlock, MoEBlockWeights

R, C = 8, 4
CHUNK = 5120
ACCUMULATORS = 11  # (what the combine planner gives GLM-5.3 at 8x4, chunk 5120: the block derives it)
CALLS = 20
REPO = Path(__file__).resolve().parents[4]
MOCK = "blackhole_galaxy" in os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH", "")


def _flat_weights_module():
    path = REPO / "ttnn/cpp/ttnn/operations/experimental/moe_block/flat_routed_expert/flat_routed_expert_weights.py"
    spec = importlib.util.spec_from_file_location("flat_routed_expert_weights", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _glm53():
    mb = ttnn.experimental.moe_block
    silu = mb.ExpertActivation(kind=mb.ExpertActivationKind.Silu)
    return mb.MoEModelDims(
        num_experts=256,
        top_k=8,
        emb_dim=6144,
        routed_expert_hidden_dim=2048,
        routed_expert_activation=silu,
        router=mb.RouterSpec(score_func=mb.RouterScoreFunc.Sigmoid, route_scale=2.5),
        shared_expert_hidden_dim=2048,
        shared_expert_activation=silu,
    )


@pytest.fixture(scope="module")
def mesh():
    if not MOCK and ttnn.get_num_devices() < R * C:
        pytest.skip(f"needs a {R}x{C} Galaxy or the mock Galaxy")
    # (the combine sends one row of H bf16 + its 32 B trailer a fabric packet)
    rc = ttnn._ttnn.fabric.FabricRouterConfig()
    rc.max_packet_payload_size_bytes = max(7232, (2 * _glm53().emb_dim + 32 + 63) // 64 * 64)
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_2D_TORUS_XY, ttnn.FabricReliabilityMode.RELAXED_INIT, router_config=rc
    )
    m = ttnn.open_mesh_device(ttnn.MeshShape(R, C), l1_small_size=5888)  # (the block's words and semaphores)
    yield m
    ttnn.close_mesh_device(m)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def test_moe_block_smoke(mesh):
    mb = ttnn.experimental.moe_block
    fw = _flat_weights_module()
    dims = _glm53()
    E, H, I, Is, N = dims.num_experts, dims.emb_dim, dims.routed_expert_hidden_dim, 2048, R * C
    dtypes = mb.MoEDtypes()
    res = mb.MoEBlockResources(
        mesh, dims, mb.MoEBlockConfig(chunk_tokens=CHUNK, num_links=2, persist_l1_small=True), dtypes
    )
    try:
        block = MoEBlock(mesh, res, dtypes=dtypes, activation=fw.ACTS["silu"])
        assert block.accumulators == ACCUMULATORS, f"the combine planner's accumulators {block.accumulators}"
        host = lambda *s: ttnn.from_torch(torch.zeros(*s), dtype=ttnn.float32)
        head_w = mb.prepare_fused_head_weights(
            host(Is, H), host(Is, H), host(H, Is), None, host(E, H), host(E),
            resources=res, dtypes=dtypes, accumulators=block.accumulators,
        )  # fmt: skip
        EPC = E // N
        gids = [list(range(c * EPC, (c + 1) * EPC)) for c in range(N)]
        experts = fw.FlatRoutedExpert(
            mesh, "uninitialized", m=block.expert_kw["max_tokens_per_expert"], H=dims.routed_width, I=I, gids=gids,
            n_global=E, act="silu", pin=1, layout=block.expert_kw["layout"],
        )  # fmt: skip
        weights = MoEBlockWeights(head=head_w, expert_ids=experts.gidx, gate_up=experts.w_gu, down=experts.w_d)
        x = ttnn.from_torch(
            torch.zeros(1, 1, CHUNK, H),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=(R, C), dims=(2, 3)),
        )
        y = block(x, weights)  # (a miss on every op)
        assert tuple(y.shape) == (1, 1, CHUNK // R, H // C), y.shape
        ttnn.deallocate(y)
        block(x, weights)
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        for _ in range(CALLS):
            ttnn.deallocate(block(x, weights))
        us = (time.perf_counter() - t0) / CALLS * 1e6
        ttnn.synchronize_device(mesh)
        print(f"MoEBlock glm53 {R}x{C} chunk {CHUNK}: {us:.1f} us of host a forward (program-cache hits, untraced)")
        experts.close()
    finally:
        res.release()
