# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-time profile of MoEBlock (GLM-5.3, 8x4, chunk 5120, untraced) under Tracy: the smoke test's block, then
phases each in a Tracy zone (PHASE <name>) so the per-op zones can be grouped by phase:
  warmup        WARMUP forwards (>= 20), not measured
  steady        STEADY forwards back to back, one sync at the end
  after_sync    AFTER forwards, each right after a device sync (the host idle while it waited)
  after_spin    AFTER forwards, each after a sync and a 50 ms busy spin (the core's clock back up first)
The timed thread is pinned to a quiet core, as the smoke test does."""

import os
import time

import torch

import ttnn
from models.demos.moe_block.tests.test_moe_block_smoke import CHUNK, C, R, _glm53, _quiet_cpu, mesh  # noqa: F401
from models.demos.moe_block.tt.flat_gather import flat_weights_module
from models.demos.moe_block.tt.moe_block import MoEBlock, MoEBlockWeights

WARMUP = int(os.environ.get("PROF_WARMUP", "25"))
STEADY = int(os.environ.get("PROF_STEADY", "50"))
AFTER = int(os.environ.get("PROF_AFTER", "30"))
assert WARMUP >= 20, "at least 20 warm-up forwards before anything is measured"


def _phase(name, body):
    ttnn.start_tracy_zone("host_profile", f"PHASE {name}", 0)
    try:
        body()
    finally:
        ttnn.stop_tracy_zone(f"PHASE {name}")


def test_host_profile(mesh):  # noqa: F811
    mb = ttnn.experimental.moe_block
    fw = flat_weights_module()
    dims = _glm53()
    E, H, I, Is, N = dims.num_experts, dims.emb_dim, dims.routed_expert_hidden_dim, 2048, R * C
    dtypes = mb.MoEDtypes()
    res = mb.MoEBlockResources(
        mesh, dims, mb.MoEBlockConfig(chunk_tokens=CHUNK, num_links=2, persist_l1_small=True), dtypes
    )
    try:
        block = MoEBlock(mesh, res, dtypes=dtypes)
        host = lambda *s: ttnn.from_torch(torch.zeros(*s), dtype=ttnn.float32)  # noqa: E731
        head_w = mb.prepare_moe_head_weights(
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

        def fwd():
            ttnn.deallocate(block(x, weights))

        cpu = _quiet_cpu()
        saved = os.sched_getaffinity(0)
        if cpu is not None:
            os.sched_setaffinity(0, {cpu})
        try:

            def warmup():
                for _ in range(WARMUP):
                    fwd()
                ttnn.synchronize_device(mesh)

            def steady():
                for _ in range(STEADY):
                    fwd()
                ttnn.synchronize_device(mesh)

            def after_sync():
                for _ in range(AFTER):
                    ttnn.synchronize_device(mesh)
                    fwd()
                ttnn.synchronize_device(mesh)

            def after_spin():
                for _ in range(AFTER):
                    ttnn.synchronize_device(mesh)
                    t = time.perf_counter() + 0.05
                    while time.perf_counter() < t:
                        pass
                    fwd()
                ttnn.synchronize_device(mesh)

            for name, body in (("warmup", warmup), ("steady", steady), ("after_sync", after_sync),
                               ("after_spin", after_spin)):  # fmt: skip
                _phase(name, body)
        finally:
            os.sched_setaffinity(0, saved)
        print(f"HOST_PROFILE warmup {WARMUP} steady {STEADY} after {AFTER} (cpu {cpu})")
        experts.close()
    finally:
        res.release()
