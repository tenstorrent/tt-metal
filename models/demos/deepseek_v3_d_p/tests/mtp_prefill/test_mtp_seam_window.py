# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""``MTPSeam`` on device: a chunk resuming at any tile-aligned start still gets exact MTP windows.

The union holds a random table indexed by global position, so the window every chip must produce is
known in closed form -- the embedding ``d`` positions on -- and the check is bit-exact. No weights.
"""

from __future__ import annotations

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.glm_5_2_config import GLM52Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tt.mla.utils import rotated_chip_positions
from models.demos.deepseek_v3_d_p.tt.mtp_prefill.device_windows import MTPSeam, MTPUnionEmbedding
from models.demos.deepseek_v3_d_p.tt.runners.input_prep import build_sp_chip_index
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology

SP_AXIS = 0
CHUNK = 5 * 1024
HIDDEN = 6144
LOOKAHEAD = 32
NUM_LEVELS = 7

STARTS = (0, 3200, 32, 608, 3296, 4512, 5088, 8000)
"""Resume points: slab- and chip-aligned controls (no seam), the largest and the smallest seam row, a
mid-chunk resume, the seam on the last chip at both extremes (its neighbour wraps to chip 0), a later
slab."""

GENERATED_ROWS = (3, 40, 77, 200)
"""Rows of the gathered generation block that the patch writes, one per generated position."""


def _union_positions(start: int, sp: int, window_len: int) -> list:
    """Per chip, the global position of every union row: the rotated trunk, then the lookahead."""
    return [
        row + list(range(row[-1] + 1, row[-1] + 1 + LOOKAHEAD)) for row in rotated_chip_positions(start, sp, window_len)
    ]


def _upload(t: torch.Tensor, mesh_device, dims) -> ttnn.Tensor:
    return ttnn.from_torch(
        t.contiguous(),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims),
    )


def _download(t: ttnn.Tensor, mesh_device) -> torch.Tensor:
    """``[1, 1, rows, H/tp]`` per chip -> ``[sp, 1, rows, H]``."""
    return ttnn.to_torch(
        t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, -1), mesh_shape=tuple(mesh_device.shape))
    )


def _per_device_scalars(t: ttnn.Tensor) -> list:
    return [ttnn.to_torch(d).flatten()[0].item() for d in ttnn.get_device_tensors(t)]


def _check_windows(union, table, positions, window_len, mesh_device, label, failures) -> None:
    """Every shift's window against ``table[position + d]``, on every chip."""
    for d in range(1, NUM_LEVELS + 1):
        window = union.window(d)
        got = _download(window, mesh_device)
        ttnn.deallocate(window)
        for c, pos in enumerate(positions):
            want = table[torch.tensor(pos[:window_len]) + d]
            bad = (got[c, 0] != want).any(dim=-1).nonzero().flatten().tolist()
            if bad:
                failures.append(f"{label} d={d} chip {c}: {len(bad)} wrong rows, first {bad[:4]}")


_MESH_PARAMS = [
    pytest.param(
        (8, 4),
        torus_xy_device_params(
            fabric_payload_size=GLM52Config.FABRIC_PAYLOAD_SIZE,
            worker_l1_size=ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE,
        ),
        2,
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
        id="torus-xy-8x4",
    ),
]


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links", _MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
@pytest.mark.skipif(not is_blackhole(), reason="deepseek_v3_d_p prefill is Blackhole-only")
@pytest.mark.timeout(1200)
def test_mtp_seam_windows(mesh_device, device_params, num_links):
    sp, tp = tuple(mesh_device.shape)
    window_len = CHUNK // sp
    union_len = window_len + LOOKAHEAD
    sp_topology = per_axis_topology(device_params["fabric_config"])[0]

    def gather(rows):
        return ttnn.all_gather(rows, dim=-2, cluster_axis=SP_AXIS, num_links=num_links, topology=sp_topology)

    chip_index = build_sp_chip_index(mesh_device, sp, (sp, tp), SP_AXIS)
    order = _per_device_scalars(chip_index)
    assert order == [float(i // tp) for i in range(sp * tp)], f"device tensors are not SP-row-major: {order}"
    for seam_chip in range(sp):
        for name, op, value in (("eq", ttnn.eq, 1.0), ("ne", ttnn.ne, 0.0)):
            flag = op(chip_index, float(seam_chip))
            want = [value if i // tp == seam_chip else 1.0 - value for i in range(sp * tp)]
            got = _per_device_scalars(flag)
            assert got == want, f"{name}(chip_index, {seam_chip}) gave {got} ({flag.dtype})"
            ttnn.deallocate(flag)

    g = torch.Generator().manual_seed(0)
    table = torch.randn(3 * CHUNK + 2 * LOOKAHEAD, HIDDEN, generator=g).to(torch.bfloat16)
    generated = torch.randint(-8, 9, (LOOKAHEAD * sp, HIDDEN), generator=g).to(torch.bfloat16)
    generated_t = _upload(generated.view(1, 1, -1, HIDDEN), mesh_device, (None, -1))

    failures = []
    for start in STARTS:
        positions = _union_positions(start, sp, window_len)
        union_host = table[torch.tensor(positions)].unsqueeze(1)
        seam_chip = (start // window_len) % sp
        neighbour = (seam_chip + 1) % sp
        targets = [positions[neighbour][0], positions[neighbour][1], start + CHUNK, positions[(seam_chip + 3) % sp][5]]
        written = table.clone()
        keep = torch.ones(sp, 1, union_len, 1)
        select = torch.zeros(sp, 1, union_len, LOOKAHEAD * sp)
        for p, source_row in zip(targets, GENERATED_ROWS):
            written[p] = generated[source_row]
            for c, upos in enumerate(positions):
                for u, q in enumerate(upos):
                    if q == p:
                        keep[c, 0, u, 0] = 0.0
                        select[c, 0, u, source_row] = 1.0

        for two_blocks in (False, True):
            label = f"start={start} {'two' if two_blocks else 'one'}-block"
            if two_blocks:
                parts = [
                    _upload(union_host[:, :, :window_len], mesh_device, (0, -1)),
                    _upload(union_host[:, :, window_len:], mesh_device, (0, -1)),
                ]
            else:
                parts = [_upload(union_host, mesh_device, (0, -1))]
            union = MTPUnionEmbedding(parts, num_levels=NUM_LEVELS, window_len=window_len)
            seam = MTPSeam.for_chunk(start, window_len, sp, chip_index, gather)
            assert (seam is None) == (start % window_len == 0), f"{label}: seam {seam}"
            union.set_seam(seam)

            t0 = time.perf_counter()
            _check_windows(union, table, positions, window_len, mesh_device, f"{label} pristine", failures)
            t1 = time.perf_counter()
            for d in range(1, NUM_LEVELS + 1):
                ttnn.deallocate(union.window(d))
            ttnn.synchronize_device(mesh_device)
            t2 = time.perf_counter()

            keep_t = _upload(keep.expand(-1, -1, -1, HIDDEN), mesh_device, (0, -1))
            select_t = _upload(select, mesh_device, (0, None))
            union.clear_rows(keep_t)
            union.add_patch(select_t, generated_t)
            _check_windows(union, written, positions, window_len, mesh_device, f"{label} patched", failures)
            logger.info(
                f"[seam] {label}: seam row {seam.row if seam else None}, first {NUM_LEVELS} windows + readback "
                f"{t1 - t0:.2f}s, warm {NUM_LEVELS} windows {1e3 * (t2 - t1):.1f}ms"
            )

            if seam is not None:
                union.clear_seam()
                seam.deallocate()
            union.deallocate()
            ttnn.deallocate(keep_t)
            ttnn.deallocate(select_t)

    ttnn.deallocate(generated_t)
    ttnn.deallocate(chip_index)
    assert not failures, f"{len(failures)} wrong windows:\n" + "\n".join(failures[:40])
