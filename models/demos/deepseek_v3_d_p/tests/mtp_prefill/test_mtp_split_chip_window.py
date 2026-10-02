# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""``MTPSplitChipLookahead`` on device: a chunk resuming at any tile-aligned start still gets exact MTP windows.
The union holds a random table row per global position, its lookahead slots laid out as the inference server
sends them, so every window is known exactly and checked bit-exact. No weights."""

from __future__ import annotations

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.glm_5_2_config import GLM52Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tt.mla.utils import mtp_lookahead_positions, rotated_chip_positions
from models.demos.deepseek_v3_d_p.tt.mtp_prefill.device_windows import MTPSplitChipLookahead, MTPUnionEmbedding
from models.demos.deepseek_v3_d_p.tt.runners.input_prep import build_sp_rank_tensor
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology

SP_AXIS = 0
CHUNK = 5 * 1024
HIDDEN = 6144
LOOKAHEAD = 32
NUM_LEVELS = 7
PAD_ROW = 3 * CHUNK + 2 * LOOKAHEAD
"""The table row every pad slot holds: no real position's embedding."""

STARTS = (0, 3200, 32, 608, 3296, 4512, 5088, 8000)
"""Resume points: chunk- and chip-aligned controls (no split chip), the largest and smallest split rows, a
mid-chunk resume, the last chip as the split chip at both extremes (next chip wraps to 0), a later chunk."""

GENERATED_ROWS = (3, 40, 77, 200)
"""Rows of the gathered generation block that the patch writes, one per generated position."""


def _union_positions(start: int, sp: int, window_len: int, chunk_end: int) -> list:
    """Per chip, the table row of every union row: the rotated trunk, then the lookahead slots."""
    slots = mtp_lookahead_positions(start, sp, window_len, chunk_end, NUM_LEVELS)
    return [
        row + look + [PAD_ROW] * (LOOKAHEAD - len(look))
        for row, look in zip(rotated_chip_positions(start, sp, window_len), slots)
    ]


def _upload(t: torch.Tensor, mesh_device, dims) -> ttnn.Tensor:
    """Shard a host tensor over the mesh along ``dims``, as bf16 TILE in DRAM."""
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
    """The first element of ``t`` on every device, in device order."""
    return [ttnn.to_torch(d).flatten()[0].item() for d in ttnn.get_device_tensors(t)]


def _check_windows(union, table, positions, window_len, chunk_end, mesh_device, label, failures) -> None:
    """Every shift's window against ``table[position + d]``, on every chip's rows below ``chunk_end``."""
    for d in range(1, NUM_LEVELS + 1):
        window = union.window(d)
        got = _download(window, mesh_device)
        ttnn.deallocate(window)
        for c, pos in enumerate(positions):
            trunk = torch.tensor(pos[:window_len])
            wrong = (got[c, 0] != table[trunk + d]).any(dim=-1) & (trunk < chunk_end)
            bad = wrong.nonzero().flatten().tolist()
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
def test_mtp_split_chip_windows(mesh_device, device_params, num_links):
    """Every MTP window of every ``STARTS`` resume, for both lookahead sources and one- and two-block unions, before
    and after a generation patch; also the SP-rank masks and where ``for_chunk_start`` turns the lookahead on."""
    sp, tp = tuple(mesh_device.shape)
    window_len = CHUNK // sp
    union_len = window_len + LOOKAHEAD
    sp_topology = per_axis_topology(device_params["fabric_config"])[0]

    def all_gather_sp(rows):
        return ttnn.all_gather(rows, dim=-2, cluster_axis=SP_AXIS, num_links=num_links, topology=sp_topology)

    sp_rank = build_sp_rank_tensor(mesh_device, sp, (sp, tp), SP_AXIS)
    order = _per_device_scalars(sp_rank)
    assert order == [float(i // tp) for i in range(sp * tp)], f"device tensors are not SP-row-major: {order}"
    for split_chip in range(sp):
        for name, op, value in (("eq", ttnn.eq, 1.0), ("ne", ttnn.ne, 0.0)):
            flag = op(sp_rank, float(split_chip))
            want = [value if i // tp == split_chip else 1.0 - value for i in range(sp * tp)]
            got = _per_device_scalars(flag)
            assert got == want, f"{name}(sp_rank, {split_chip}) gave {got} ({flag.dtype})"
            ttnn.deallocate(flag)

    g = torch.Generator().manual_seed(0)
    table = torch.randn(PAD_ROW + 1, HIDDEN, generator=g).to(torch.bfloat16)
    generated = torch.randint(-8, 9, (LOOKAHEAD * sp, HIDDEN), generator=g).to(torch.bfloat16)
    generated_t = _upload(generated.view(1, 1, -1, HIDDEN), mesh_device, (None, -1))

    failures = []
    for start in STARTS:
        offset = start % window_len
        split_chip = (start // window_len) % sp
        next_chip = (split_chip + 1) % sp
        if offset:
            last_plain_end = start + window_len - offset - NUM_LEVELS
            for chunk_end, wants_lookahead in ((last_plain_end, False), (last_plain_end + 1, True)):
                lookahead = MTPSplitChipLookahead.for_chunk_start(
                    start, window_len, sp, sp_rank, all_gather_sp, chunk_end=chunk_end, num_levels=NUM_LEVELS
                )
                assert (lookahead is not None) == wants_lookahead, f"start={start} chunk_end={chunk_end}: {lookahead}"
                if lookahead is not None:
                    lookahead.deallocate()

        # Both lookahead sources: an end one past the next chip's first position (own slots), a full chunk (SP).
        chunk_ends = (start + CHUNK, start + window_len - offset + 1) if offset else (start + CHUNK,)
        for chunk_end in chunk_ends:
            positions = _union_positions(start, sp, window_len, chunk_end)
            union_host = table[torch.tensor(positions)].unsqueeze(1)
            targets = [
                positions[next_chip][0],
                positions[next_chip][1],
                start + CHUNK,
                positions[(split_chip + 3) % sp][5],
            ]
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
                label = f"start={start} end={chunk_end} {'two' if two_blocks else 'one'}-block"
                if two_blocks:
                    parts = [
                        _upload(union_host[:, :, :window_len], mesh_device, (0, -1)),
                        _upload(union_host[:, :, window_len:], mesh_device, (0, -1)),
                    ]
                else:
                    parts = [_upload(union_host, mesh_device, (0, -1))]
                union = MTPUnionEmbedding(parts, num_levels=NUM_LEVELS, window_len=window_len)
                split_lookahead = MTPSplitChipLookahead.for_chunk_start(
                    start, window_len, sp, sp_rank, all_gather_sp, chunk_end=chunk_end, num_levels=NUM_LEVELS
                )
                assert (split_lookahead is None) == (offset == 0), f"{label}: split-chip lookahead {split_lookahead}"
                if split_lookahead is not None:
                    want_next = next_chip if chunk_end == start + CHUNK else None
                    assert (
                        split_lookahead.next_chip == want_next
                    ), f"{label}: next_chip {split_lookahead.next_chip}, expected {want_next} (None: slots)"
                union.set_split_chip_lookahead(split_lookahead)

                t0 = time.perf_counter()
                _check_windows(
                    union, table, positions, window_len, chunk_end, mesh_device, f"{label} pristine", failures
                )
                t1 = time.perf_counter()
                for d in range(1, NUM_LEVELS + 1):
                    ttnn.deallocate(union.window(d))
                ttnn.synchronize_device(mesh_device)
                t2 = time.perf_counter()

                keep_t = _upload(keep.expand(-1, -1, -1, HIDDEN), mesh_device, (0, -1))
                select_t = _upload(select, mesh_device, (0, None))
                union.clear_rows(keep_t)
                union.add_patch(select_t, generated_t)
                _check_windows(
                    union, written, positions, window_len, chunk_end, mesh_device, f"{label} patched", failures
                )
                logger.info(
                    f"[split chip] {label}: split row {split_lookahead.split_row if split_lookahead else None}, "
                    f"first {NUM_LEVELS} windows + readback {t1 - t0:.2f}s, warm {NUM_LEVELS} windows "
                    f"{1e3 * (t2 - t1):.1f}ms"
                )

                if split_lookahead is not None:
                    union.clear_split_chip_lookahead()
                    split_lookahead.deallocate()
                union.deallocate()
                ttnn.deallocate(keep_t)
                ttnn.deallocate(select_t)

    ttnn.deallocate(generated_t)
    ttnn.deallocate(sp_rank)
    assert not failures, f"{len(failures)} wrong windows:\n" + "\n".join(failures[:40])
