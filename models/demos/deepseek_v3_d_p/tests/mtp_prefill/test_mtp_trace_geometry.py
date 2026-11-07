# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The traced MTP geometry on device matches its host build, chunk after chunk.

``MTPTraceGeometry.write`` skips every tensor whose plan key did not change and, on a provided chunk,
every generate-only tensor. A wrong key leaves a previous chunk's tensor on device, which still
correlates with the right one, so the check is bit-exact and runs over a sequence of chunks.
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.demos.common.prefill.runners.runner_utils import num_mtp_tokens
from models.demos.deepseek_v3_d_p.tt.mtp_prefill.trace_geometry import MTPTraceGeometry, _generate_only

SP_AXIS = 0
CHUNK = 5120


def _provided_levels(start: int, end: int, prompt_end: int, k_levels: int) -> int:
    """Same rule as the runner: only a full chunk takes levels from the prompt, up to the tokens left."""
    if end < start + CHUNK:
        return 0
    return min(k_levels, prompt_end - end)


def _chunks(start: int, prompt_end: int) -> list:
    out = []
    while start < prompt_end:
        end = min(start + CHUNK, prompt_end)
        out.append((start, end, prompt_end))
        start = end
    return out


def _turns() -> list:
    """Chunk sequences replayed back to back on one geometry, as consecutive requests would."""
    window = CHUNK // 8
    return (
        # Ragged end after provided chunks.
        _chunks(0, 4 * CHUNK + 1777)
        # A full final chunk, then one with only 3 prompt tokens after it (partially provided).
        + _chunks(0, 2 * CHUNK)
        + _chunks(0, 3 * CHUNK + 3)
        # A prefix-cache hit that resumes mid-chip, so a chip is split, with a ragged end.
        + _chunks(CHUNK + window // 2, 3 * CHUNK + 1000)
        # The same ragged end again: every key is unchanged, so nothing may be rewritten wrongly.
        + _chunks(0, 4 * CHUNK + 1777)
        # A mid-chip resume ending a few rows into the next chunk.
        + _chunks(2 * CHUNK + 3 * window + 64, 3 * CHUNK + 3 * window + 64 + 40)
    )


def _device_shards(geometry: MTPTraceGeometry, name: str, mesh_device) -> list:
    tensor = geometry.selects[int(name[len("select") :])] if name.startswith("select") else geometry._tensors[name]
    cols = mesh_device.shape[1]
    out = []
    for i, shard_t in enumerate(ttnn.get_device_tensors(tensor)):
        shard = i // cols if SP_AXIS == 0 else i % cols
        out.append((i, shard, ttnn.to_torch(shard_t).float()))
    return out


@pytest.mark.parametrize("mesh_device", [(8, 4)], ids=["8x4"], indirect=True)
@pytest.mark.parametrize("num_levels", [7, 3], ids=["levels7", "levels3"])
def test_mtp_trace_geometry_multi_chunk_bit_exact(mesh_device, num_levels):
    sp = mesh_device.shape[SP_AXIS]
    geometry = MTPTraceGeometry(
        mesh_device,
        sp_factor=sp,
        chunk_size=CHUNK,
        mesh_shape=tuple(mesh_device.shape),
        sp_axis=SP_AXIS,
        num_mtp_tokens=num_mtp_tokens(num_levels),
        num_levels=num_levels,
    )
    kinds = set()
    try:
        for start, end, prompt_end in _turns():
            provided = _provided_levels(start, end, prompt_end, num_levels)
            generates = provided < num_levels
            kinds.add((generates, end - start < CHUNK, start % (CHUNK // sp) != 0))
            geometry.write(start, end, provided)
            ref = geometry._host(start, end, provided)
            for name, host in ref.items():
                # A provided replay never reads these, so write() leaves them stale on purpose.
                if not generates and (_generate_only(name) or name.startswith("select")):
                    continue
                for dev, shard, got in _device_shards(geometry, name, mesh_device):
                    want = host[shard]
                    got = got.reshape(want.shape)
                    assert torch.equal(got, want), (
                        f"{name} on device {dev} differs from the host build in {int((got != want).sum())} "
                        f"elements: chunk [{start}, {end}) of a {prompt_end}-token prompt, provided={provided}"
                    )
    finally:
        geometry.deallocate()
    # The sequence covers provided and generate chunks, ragged ends and split-chip starts.
    assert (False, False, False) in kinds and (True, True, False) in kinds and (True, False, False) in kinds
    assert any(split for _, _, split in kinds)
