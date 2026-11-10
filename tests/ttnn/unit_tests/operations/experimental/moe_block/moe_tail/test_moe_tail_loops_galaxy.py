# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ttnn.experimental.moe_block.moe_tail, Kimi-K3's engine tail, row blocks in one call on an 8x4 Galaxy (torus XY:
a latent model needs a torus axis, so the LoudBox's test_moe_tail_row_blocks_one_call has no K3 case).

test_moe_tail_engine_row_blocks_one_call: row_blocks k (the head's block layout) - the tail over every block's rows in
one call (its kernels loop: the global step's slots, the block's thresholds, the gate a block) is bit-identical to a
one-block call on each block's rows made on resources planned for one block's rows (y_own, the shared expert's output
and the output: block i's rows after block i - 1's), on seeded random owner rows, shared-expert output and W_up;
repeated (the persistent ready words and the cumulative semaphores carried from call to call)."""

import sys
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from head_tail_lb_cases import MODELS, Weights, _payload, mb  # noqa: E402

R, C = 8, 4
ROW_TOKENS = 512  # (a block's tokens a mesh row: a multiple of 32 x TP)


@pytest.fixture(scope="module")
def mesh():
    if ttnn.get_num_devices() < R * C:
        pytest.skip(f"needs a {R}x{C} Galaxy")
    rc = ttnn._ttnn.fabric.FabricRouterConfig()
    rc.max_packet_payload_size_bytes = _payload(MODELS["k3"])
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_2D_TORUS_XY, ttnn.FabricReliabilityMode.RELAXED_INIT, router_config=rc
    )
    m = ttnn.open_mesh_device(ttnn.MeshShape(R, C), l1_small_size=16384)
    yield m
    ttnn.close_mesh_device(m)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _to_owner(mesh, t):
    """[chunk, W] -> [1, 1, S, W] a chip, the owners' rows (chip (r, c) owns tokens (r C + c) S ..), row-major."""
    return ttnn.from_torch(
        t.reshape(1, 1, *t.shape),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=2),
    )


def _to_residual(mesh, t):
    """[chunk, W] -> [1, 1, chunk / R, W / C] a chip (the residual's layout), TILE."""
    return ttnn.from_torch(
        t.reshape(1, 1, *t.shape),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=(R, C), dims=(2, 3)),
    )


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("k", [2, 3])
def test_moe_tail_engine_row_blocks_one_call(mesh, k):
    model = MODELS["k3"]
    chunk = k * R * ROW_TOKENS
    w = Weights(model, seed=3)
    g = torch.Generator().manual_seed(2003)
    y = _to_owner(mesh, torch.randn(chunk, model.routed_width, generator=g).to(torch.bfloat16))
    se = _to_residual(mesh, (torch.randn(chunk, model.H, generator=g) * 0.5).to(torch.bfloat16))
    f32 = lambda t: ttnn.from_torch(t, dtype=ttnn.float32)  # noqa: E731

    def resources(tokens):
        cfg = mb.MoEBlockConfig(chunk_tokens=tokens, num_links=2, persist_l1_small=True)
        return mb.MoEBlockResources(mesh, model.dims(), cfg, mb.MoEDtypes())

    def w_up_of(res, row_blocks):
        return mb.prepare_moe_tail_weights(
            f32(w.latent_up), f32(w.latent_norm), resources=res, dtype=ttnn.bfloat8_b, row_blocks=row_blocks
        )

    def rows(t, i):
        n, wd = int(t.shape[-2]) // k, int(t.shape[-1])
        return ttnn.slice(t, [0, 0, i * n, 0], [1, 1, (i + 1) * n, wd], [1, 1, 1, 1])

    res1 = resources(R * ROW_TOKENS)  # (the reference: one block's rows a call)
    try:
        w_up1 = w_up_of(res1, 1)
        ref = ttnn.concat(
            [mb.moe_tail(rows(y, i), rows(se, i), resources=res1, w_up=w_up1, row_blocks=1) for i in range(k)], dim=2
        )
        ref_host = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(ref)]
    finally:
        res1.release()
    res = resources(chunk)
    try:
        w_up = w_up_of(res, k)
        for call in range(3):  # (repeated: the ready words and the cumulative counts carried from call to call)
            got = mb.moe_tail(y, se, resources=res, w_up=w_up, row_blocks=k)
            diff = sum(int((ttnn.to_torch(a) != b).sum()) for a, b in zip(ttnn.get_device_tensors(got), ref_host))
            logger.info(
                f"MOE_TAIL engine one call x {k} row blocks (call {call}): {diff} elements differ from a call a block"
            )
            assert diff == 0
            ttnn.deallocate(got)
    finally:
        res.release()
