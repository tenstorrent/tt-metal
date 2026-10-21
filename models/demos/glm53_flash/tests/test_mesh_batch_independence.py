# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Decode batch independence on a (rows, 4) block with the user-partitioned DSA cache (heads2users layout, B user
slots per row): a slot's decode output must depend only on its own tokens, and rows fed the same prompts must
agree bit-exactly. Motivated by the 1 M x 32-user run (doc/mesh/README.md): the column-3 users' chains diverged
from the other columns' and the rows disagreed even for users 0-2, which a cross-slot leak in the decode step
would explain. Real weights of one layer (3 = DSA, 4 = KDA + MoE), ``S`` prompt tokens fed through the decode step
(positions 0..S-1), eager by default (``GLM53_INDEP_TRACED=1``: through the captured decode trace).

Checks (all ``torch.equal`` on the fp32 stream outputs):
* rows: every row holds the same B prompts -> identical outputs on every row;
* last slot: slot B-1 of every row gets another prompt -> slots 0..B-2 unchanged (and slot B-1 changed);
* first slot: slot 0 gets another prompt -> slots 1..B-1 unchanged.

    TT_VISIBLE_DEVICES=27,26,31,30,23,22,19,18 TT_MESH_GRAPH_DESC_PATH=.../p150_x8_mesh_graph_descriptor.textproto \\
    MESH_DEVICE=P150x8 GLM53_MESH_SHAPE=2,4 pytest models/demos/glm53_flash/tests/test_mesh_batch_independence.py -s
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir, load_layer_raw, shard_headers
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.functional_decoder import (
    FunctionalDecoder,
    device_streams_to_host,
    host_streams_to_device,
)
from models.demos.glm53_flash.tt.mesh import mesh_device_params

DUMP = Path("bringup/references/glm53_flash/R4097")
ROWS, COLS = (int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "2,4").split(","))
B = int(os.environ.get("GLM53_MESH_B", "4"))  # user slots per row (one per column with the partitioned cache)
S = int(os.environ.get("GLM53_MESH_S", "16"))  # prompt tokens fed through the decode step
TRACED = os.environ.get("GLM53_INDEP_TRACED", "0") == "1"
LAYERS = [int(v) for v in os.environ.get("GLM53_INDEP_LAYERS", "3,4").split(",")]


def _first_diff(a: torch.Tensor, b: torch.Tensor) -> str:
    d = (a.float() - b.float()).abs()
    return f"max abs diff {d.max().item():.4g} over {int((d > 0).sum())} of {d.numel()} values"


@pytest.mark.timeout(1500)
@pytest.mark.parametrize("device_params", [mesh_device_params(ROWS, COLS, trace_region_size=60_000_000)], indirect=True)
@pytest.mark.parametrize("mesh_device", [(ROWS, COLS)], indirect=True)
@pytest.mark.parametrize("layer_idx", LAYERS)
def test_mesh_batch_independence(mesh_device, layer_idx):
    snapshot = default_snapshot_dir()
    headers = shard_headers(snapshot)
    raw = load_layer_raw(snapshot, layer_idx, headers)
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    base = torch.load(DUMP / f"layer{layer_idx}_prefill.pt")["x_in"]  # [1, 4097, 4, 4096] random-token streams
    U = ROWS * B
    # B + 2 distinct prompt windows: slots 0..B-1 and two alternates (for the last / first slot)
    offs = [100 + (3800 // (B + 2)) * k for k in range(B + 2)]
    windows = [base[:, o : o + S] for o in offs]

    dec = FunctionalDecoder.from_state_dict(
        raw,
        hf_config=cfg,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        max_batch=B,
        max_seq_len=256,
        partition_users=True,
        prefill_partition="heads2users",
    )
    mesh = dec.mesh
    assert (mesh.rows, mesh.cols) == (ROWS, COLS) and dec.total_users == U
    kind = "KDA" if dec.is_kda else "DSA"
    logger.info(f"layer {layer_idx} ({kind}) on {(ROWS, COLS)}: {B} slots per row, {S} tokens, traced={TRACED}")

    def token(assign: list[int], t: int) -> torch.Tensor:
        """[1, U, 4, 4096]: slot u of every row carries token ``t`` of window ``assign[u]``."""
        return torch.cat([windows[assign[u]][:, t : t + 1] for _ in range(ROWS) for u in range(B)], dim=1)

    def load(x_dev, xin):
        packed = xin.reshape(1, 1, U, -1).float()
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(packed, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, mesh_mapper=mesh.shard_rows(2)), x_dev
        )

    def reset_kda():
        if dec.kda_batched is None:
            return
        for t in (dec.kda_batched.recurrent, dec.kda_batched.convolution):
            z = ttnn.zeros(
                tuple(t.shape),
                dtype=t.dtype,
                layout=t.layout,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.copy(z, t)
            ttnn.deallocate(z)

    base_assign = list(range(B))
    x_dev = host_streams_to_device(token(base_assign, 0), mesh_device, pad_to=U, mesh=mesh)
    dp = dec.make_decode_positions([0] * U)
    trace_id = y_tr = None
    if TRACED:
        trace_id, y_tr, warm = dec.capture_decode_trace(x_dev, dp)
        ttnn.deallocate(warm)
    reset_kda()

    def run(assign: list[int]) -> list[torch.Tensor]:
        """The S decode outputs ([1, U, 4, 4096] fp32 each) of the prompts ``assign`` from a fresh state."""
        reset_kda()
        outs = []
        for t in range(S):
            load(x_dev, token(assign, t))
            dec.update_decode_positions(dp, [t] * U)
            if TRACED:
                dec.replay_decode(trace_id)
                y = y_tr
            else:
                y = dec.decode_forward(x_dev, positions=dp)
            outs.append(device_streams_to_host(y, 4, U, mesh=mesh).clone())
            if not TRACED:
                ttnn.deallocate(y)
        return outs

    failures = []
    ref = run(base_assign)
    # rows: the same B prompts on every row
    for t, out in enumerate(ref):
        for r in range(1, ROWS):
            for u in range(B):
                a, b = out[:, u], out[:, r * B + u]
                if not torch.equal(a, b):
                    failures.append(f"step {t}: row {r} slot {u} differs from row 0 ({_first_diff(a, b)})")
    logger.info(f"rows agree: {'yes' if not failures else 'NO'} ({len(failures)} mismatches)")

    def check_unchanged(name: str, assign: list[int], changed: int):
        got = run(assign)
        n0 = len(failures)
        for t, (a_t, b_t) in enumerate(zip(ref, got)):
            for r in range(ROWS):
                for u in range(B):
                    i = r * B + u
                    if u == changed:
                        if t == 0 and torch.equal(a_t[:, i], b_t[:, i]):
                            failures.append(f"{name}: slot {u} of row {r} unchanged although its prompt changed")
                    elif not torch.equal(a_t[:, i], b_t[:, i]):
                        failures.append(
                            f"{name}: step {t} row {r} slot {u} changed with slot {changed}'s prompt "
                            f"({_first_diff(a_t[:, i], b_t[:, i])})"
                        )
        logger.info(f"{name}: {'independent' if len(failures) == n0 else 'LEAK'} ({len(failures) - n0} mismatches)")

    last = list(base_assign)
    last[B - 1] = B  # another window in the last slot
    check_unchanged("last slot changed", last, B - 1)
    first = list(base_assign)
    first[0] = B + 1
    check_unchanged("first slot changed", first, 0)
    if trace_id is not None:
        ttnn.release_trace(mesh_device, trace_id)
    assert not failures, "\n".join(failures[:40])
