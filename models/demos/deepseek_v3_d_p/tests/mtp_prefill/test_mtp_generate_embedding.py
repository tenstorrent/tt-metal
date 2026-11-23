# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The traced MTP generation reads one LM-head row and still picks the eager path's token.

``TtPrefillTransformer.mtp_generate_embedding_selected`` copies the chunk's last real row into every row of
the LM head's tile and takes the argmax of row 0 alone. The eager-shaped path projects the tile holding that
row and takes the argmax of all 32 rows. Same token, same embedding, on every chip, bit for bit.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import GLM53Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tt.mtp_prefill.trace_geometry import select_rows
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.tt.tt_lm_head import TtLMHead
from models.demos.deepseek_v3_d_p.tt.tt_parallel_embedding import TtParallelEmbedding
from models.demos.deepseek_v3_d_p.tt.tt_prefill_transformer import TtPrefillTransformer

SP_AXIS, TP_AXIS = 0, 1
WINDOW = 640
HIDDEN = 6144
VOCAB = 154880


def _sharded(t: torch.Tensor, mesh_device, dims) -> ttnn.Tensor:
    return ttnn.from_torch(
        t,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_device.shape, dims=dims),
    )


def _per_device(t: ttnn.Tensor) -> list:
    return [ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)]


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(
                fabric_payload_size=GLM53Config.FABRIC_PAYLOAD_SIZE,
                worker_l1_size=ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE,
            ),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="torus-xy-8x4",
        )
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("local_row", [0, 31, 333, 639], ids=lambda r: f"row{r}")
def test_mtp_generate_embedding_one_row_matches_tile(mesh_device, device_params, local_row):
    torch.manual_seed(local_row)
    sp, tp = mesh_device.shape[SP_AXIS], mesh_device.shape[TP_AXIS]
    topology = per_axis_topology(device_params["fabric_config"])
    tile = ttnn.TILE_SIZE

    lm_head = TtLMHead(
        mesh_device=mesh_device,
        emb_dim=HIDDEN,
        vocab_size=VOCAB,
        torch_weight=torch.randn(VOCAB, HIDDEN) * 0.02,
        num_links=2,
        topology=topology[TP_AXIS] if isinstance(topology, tuple) else topology,
        is_column_parallel=True,
    )
    embed = TtParallelEmbedding(
        mesh_device=mesh_device,
        vocab_size=VOCAB,
        emb_dim=HIDDEN,
        torch_weight=torch.randn(VOCAB, HIDDEN),
        sp_axis=SP_AXIS,
        tp_axis=TP_AXIS,
    )
    model = SimpleNamespace(
        lm_head=lm_head,
        tp_factor=tp,
        tp_axis=TP_AXIS,
        sp_factor=sp,
        sp_axis=SP_AXIS,
        num_links=2,
        sp_topology=topology[SP_AXIS] if isinstance(topology, tuple) else topology,
        embed=embed,
    )
    model.mtp_embed_ids = lambda ids: TtPrefillTransformer.mtp_embed_ids(model, ids)
    model._mtp_all_gather_sp = lambda rows: TtPrefillTransformer._mtp_all_gather_sp(model, rows)

    # [sp, 1, W, H] -> each chip holds its own W rows, TP-sharded on H.
    h = _sharded(torch.randn(sp, 1, WINDOW, HIDDEN), mesh_device, (SP_AXIS, -1))

    # Eager-shaped reference: the tile holding local_row, all 32 rows projected and argmaxed.
    tile_start = local_row // tile * tile
    old_pick = torch.zeros(sp, 1, tile, WINDOW)
    old_pick[:, 0, torch.arange(tile), tile_start + torch.arange(tile)] = 1.0
    old_rows = _sharded(old_pick, mesh_device, (SP_AXIS, None))
    new_pick = torch.zeros(sp, 1, tile, WINDOW)
    new_pick[:, 0, :, local_row] = 1.0
    new_rows = _sharded(new_pick, mesh_device, (SP_AXIS, None))

    def old_path():
        sel = select_rows(old_rows, h)
        out = TtPrefillTransformer._mtp_embed_greedy(model, sel, None)
        ttnn.deallocate(sel)
        return out

    def new_path():
        return TtPrefillTransformer.mtp_generate_embedding_selected(model, h, new_rows)

    timings = {}
    for name, fn in (("old", old_path), ("new", new_path)):
        fn()  # compile
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        for _ in range(5):
            out = fn()
            ttnn.synchronize_device(mesh_device)
            ttnn.deallocate(out)
        timings[name] = (time.perf_counter() - t0) / 5 * 1e3
    logger.info(f"generate embedding: old {timings['old']:.3f} ms, new {timings['new']:.3f} ms (host wall, synced)")

    old = _per_device(old_path())
    new = _per_device(new_path())
    r = local_row % tile
    for dev, (o, n) in enumerate(zip(old, new)):
        # Chip c's 32 rows sit at [32c, 32c + 32); every row of the new path must match the reference.
        o = o.reshape(sp * tile, -1)
        n = n.reshape(sp * tile, -1)
        for c in range(sp):
            want = o[c * tile + r]
            got = n[c * tile : (c + 1) * tile]
            assert torch.equal(got, want.expand_as(got)), f"device {dev}, SP block {c}: token embedding differs"
