# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""User-partitioned DSA cache on a (2, 4) block (OPT_SPARSE_1M deliverable 3): column c of a row owns users
``[c*B/4, (c+1)*B/4)``, the latent cache + pooled-key index are allocated for those users (+ the prefill scratch
slot) only, the decode step exchanges heads -> users before the traced sparse attention and users -> heads after
(``GLMMesh.all_to_all_cols``), prefill fills the owner's slot and the scratch slot.

* ``test_partitioned_two_rows_four_users``: layer 3 on ``GLM53_MESH_SHAPE`` (default 2,4), 4 users per row (one per column) prefilled from four
  reference streams (R4097 sparse, R2040 dense -> sparse at 2047, R2042, R2043), 12 lockstep decode steps
  (eager, then the captured trace replayed), every (row, user, step) output vs its HF dump > 0.995 and the two
  rows byte-equal (replicated prefill, identical tokens).
* ``test_partitioned_1m_step``: one user per column at position 1 048 576 (cache 1 M + 64 per chip-slot, unseeded),
  the traced layer step timed on the block.

    TT_VISIBLE_DEVICES=27,26,31,30,23,22,19,18 TT_MESH_GRAPH_DESC_PATH=<p150_x8 descriptor> MESH_DEVICE=P150x8 \
        pytest models/demos/glm53_flash/tests/test_mesh_sparse_partitioned.py -s --timeout 3600
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir, load_layer_raw, shard_headers
from models.demos.glm53_flash.tests.utils import pcc
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.functional_decoder import FunctionalDecoder, host_streams_to_device
from models.demos.glm53_flash.tt.mesh import GLMMesh, mesh_device_params

LAYER = 3
ROWS, COLS = (int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "2,4").split(","))  # (8,4) = the whole Galaxy
B = 4  # users per row = one per column
DUMPS = ["R4097", "R2040", "R2042", "R2043"]
REF = Path("bringup/references/glm53_flash")


def _packed(streams_per_user: list[torch.Tensor]) -> torch.Tensor:
    """[1,1,4,4096] per user -> [1,1,U,16384] fp32 (user-major)."""
    rows = [s.reshape(1, -1).float() for s in streams_per_user]
    return torch.stack(rows, dim=0).reshape(1, 1, len(rows), -1)


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "device_params", [mesh_device_params(ROWS, COLS, trace_region_size=64 * 1024 * 1024)], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(ROWS, COLS)], indirect=True)
def test_partitioned_two_rows_four_users(mesh_device):
    snapshot = default_snapshot_dir()
    raw = load_layer_raw(snapshot, LAYER, shard_headers(snapshot))
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    mesh = GLMMesh.from_device(mesh_device)
    pres = [torch.load(REF / d / f"layer{LAYER}_prefill.pt") for d in DUMPS]
    S = [p["x_in"].shape[1] for p in pres]
    n_steps = 12
    dec = FunctionalDecoder.from_state_dict(
        raw,
        hf_config=cfg,
        layer_idx=LAYER,
        mesh_device=mesh_device,
        max_batch=B,
        max_seq_len=max(S) + 64,
        partition_users=True,
    )
    assert dec.cache.partition and dec.cache.users_per_chip == 1 and dec.cache.slots == 2
    for u, pre in enumerate(pres):
        y = dec.prefill_forward(host_streams_to_device(pre["x_in"], mesh_device, mesh=mesh), user=u, seq_len=S[u])
        out = mesh.to_host(y, rows=2 if mesh.rows > 1 else None).float()[0, 0, : S[u]]
        p = pcc(pre["x_out"].reshape(S[u], -1), out)
        logger.info(f"user {u} ({DUMPS[u]}) prefill S={S[u]}: pcc {p:.6f}")
        assert p > 0.995
        ttnn.deallocate(y)

    steps = [[torch.load(REF / d / f"layer{LAYER}_decode{s}.pt") for d in DUMPS] for s in range(n_steps)]
    U = ROWS * B
    positions = [steps[0][u % B]["position"] for u in range(U)]
    streams_dev = mesh.upload(_packed([steps[0][u % B]["x_in"] for u in range(U)]), ttnn.float32, rows=2)
    dp = dec.make_decode_positions(positions)
    results = []

    def check(out_dev, s, mode):
        shards = [ttnn.to_torch(d).float() for d in mesh.device_tensors(out_dev)]  # [1,1,B,16384] per chip
        for r in range(ROWS):
            for c in range(1, COLS):
                assert torch.equal(shards[r * COLS + c], shards[r * COLS]), (s, r, c, "columns differ")
        for r in range(ROWS):
            for u in range(B):
                d = steps[s][u]
                p = pcc(d["x_out"].reshape(1, -1), shards[r * COLS][0, 0, u : u + 1])
                results.append((s, r, u, d["position"], p))
                logger.info(f"  step {s} row {r} user {u} pos {d['position']} ({mode}): pcc {p:.6f}")

    def load_step(s):
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                _packed([steps[s][u % B]["x_in"] for u in range(U)]),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=mesh.shard_rows(2),
            ),
            streams_dev,
        )
        dec.update_decode_positions(dp, [steps[s][u % B]["position"] for u in range(U)])

    for s in range(2):
        if s:
            load_step(s)
        y = dec.decode_forward(streams_dev, positions=dp)
        check(y, s, "eager")
        ttnn.deallocate(y)
    load_step(2)
    trace_id, out_dev, warm = dec.capture_decode_trace(streams_dev, dp)
    check(warm, 2, "capture warm-up")
    for s in range(3, n_steps):
        load_step(s)
        dec.replay_decode(trace_id)
        check(out_dev, s, "trace replay")
    ttnn.release_trace(mesh_device, trace_id)
    worst = min(r[4] for r in results)
    logger.info(f"partitioned ({ROWS},{COLS}): worst pcc {worst:.6f} over {len(results)} (step, row, user) samples")
    assert all(r[4] > 0.995 for r in results), [r for r in results if r[4] <= 0.995]


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "device_params", [mesh_device_params(ROWS, COLS, trace_region_size=200_000_000)], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(ROWS, COLS)], indirect=True)
def test_partitioned_1m_step(mesh_device):
    snapshot = default_snapshot_dir()
    raw = load_layer_raw(snapshot, LAYER, shard_headers(snapshot))
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    mesh = GLMMesh.from_device(mesh_device)
    N = 1 << 20
    t0 = time.perf_counter()
    dec = FunctionalDecoder.from_state_dict(
        raw,
        hf_config=cfg,
        layer_idx=LAYER,
        mesh_device=mesh_device,
        max_batch=B,
        max_seq_len=N + 64,
        partition_users=True,
    )
    logger.info(
        f"layer {LAYER} on the ({ROWS},{COLS}) mesh, 1 user per column x {N + 64} positions: built in {time.perf_counter() - t0:.0f} s"
    )
    U = ROWS * B
    streams = mesh.upload(torch.randn(1, 1, U, 4 * cfg.hidden_size) * 0.5, ttnn.float32, rows=2)
    for pos in (4096, N):
        dp = dec.make_decode_positions([pos] * U)
        trace_id, out, warm = dec.capture_decode_trace(streams, dp)
        ttnn.deallocate(warm)
        dec.replay_decode(trace_id)
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        for _ in range(20):
            dec.replay_decode(trace_id, blocking=False)
        ttnn.synchronize_device(mesh_device)
        ms = (time.perf_counter() - t0) / 20 * 1e3
        ttnn.release_trace(mesh_device, trace_id)
        logger.info(
            f"layer {LAYER} traced decode step on the (2,4) block, {U} users (1 per column) at position {pos}: {ms:.3f} ms/step"
        )
