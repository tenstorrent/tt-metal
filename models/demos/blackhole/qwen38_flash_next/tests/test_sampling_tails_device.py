# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The one-program sampler versus the composite and the host reference, with its presence history, its guard row and
trace reuse."""

import dataclasses
import os

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.tests.helpers.tp_harness import (
    DEVICE_PARAMS,
    assert_program_cache_stable,
    assert_tensors_equal,
    download_replica,
    uint32_bits,
    upload,
)
from models.demos.blackhole.qwen38_flash_next.ttnn import device_sampler as ds
from models.demos.blackhole.qwen38_flash_next.ttnn import fused
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import Qwen38MeshContract
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import sampler_tail as st

pytestmark = pytest.mark.skipif(os.environ.get("QWEN38_FUSED_DEVICE_TEST") != "1", reason="requires held four-die mesh")

SHARDS = 4
SHARD_LANES = st.LANES // SHARDS
SHARD_VOCAB = st.VOCAB_SIZE // SHARDS
POLICY = ds.Qwen38DeviceSamplerPolicy(temperature=0.8, top_k=20, top_p=0.9, min_p=0.02, presence_penalty=0.5)
# Every read lane kept: with one shard's lanes above all others, the kept minimum is the shard floor and the guard sets.
GUARD_POLICY = ds.Qwen38DeviceSamplerPolicy(temperature=1.0, top_k=32, top_p=1.0, min_p=0.0, presence_penalty=2.0)


def _candidate_rows(rows, rng):
    """Per row and shard 32 bf16-exact values (a four-way tie at the maximum) then 32 distinct ids of its slice."""

    host = torch.empty(1, 1, rows, st.ROW_LANES)
    for row in range(rows):
        values = (torch.randn(SHARDS, SHARD_LANES, generator=rng) * 2).to(torch.bfloat16).float()
        values[:, 0] = values.max()
        ids = [
            torch.randperm(SHARD_VOCAB, generator=rng)[:SHARD_LANES] + shard * SHARD_VOCAB for shard in range(SHARDS)
        ]
        host[0, 0, row] = torch.cat([torch.cat([values[shard], ids[shard].float()]) for shard in range(SHARDS)])
    return host


def _dominant_shard_rows(rows, rng):
    """Candidate rows whose shard ``row % 4`` holds the row's 32 largest values, so the kept minimum under
    ``GUARD_POLICY`` is the shard floor: the guard row's bit is set for every row."""

    host = _candidate_rows(rows, rng)
    for row in range(rows):
        shard = row % SHARDS
        start = shard * 2 * SHARD_LANES
        values = host[0, 0, row, start : start + SHARD_LANES]
        host[0, 0, row, start : start + SHARD_LANES] = (values + 16.0).to(torch.bfloat16).float()  # still bf16-exact
    return host


def _draws(rows, rng):
    return [
        int(numerator) * 2.0**-ds.UNIFORM_BITS
        for numerator in torch.randint(0, 1 << ds.UNIFORM_BITS, (rows,), generator=rng)
    ]


def _expected(row_host, policy, draws, emitted):
    samples = []
    for row, uniform in enumerate(draws):
        values, ids = ds.candidate_row_lanes(row_host[0, 0, row])
        history = set(emitted[row])
        seen = torch.tensor([int(token_id) in history for token_id in ids])
        samples.append(ds.device_sampler_reference(values, ids, policy, uniform, seen=seen))
    return samples


def _assert_drawn(token, constants, row_host, policy, draws, emitted, label):
    """The token tile's draws and the guard row's mask (bit r: row r), asserted equal to the host reference's for the
    same uniforms and history.  Returns the draws and the mask."""

    rows = len(draws)
    lanes = download_replica(token).reshape(-1)
    assert_tensors_equal(lanes[rows:], torch.zeros_like(lanes[rows:]), f"{label}: token lanes past the {rows} rows")
    drawn = lanes[:rows].to(torch.int64).tolist()
    samples = _expected(row_host, policy, draws, emitted)
    expected = [sample.token_id for sample in samples]
    assert drawn == expected, f"{label}: drew {drawn}, expected {expected}"
    mask = int(download_replica(constants.guard).reshape(-1)[0])
    expected_mask = sum(1 << row for row, sample in enumerate(samples) if sample.guard)
    assert mask == expected_mask, f"{label}: guard mask {mask:#x}, reference {expected_mask:#x}"
    return drawn, mask


def _assert_history(constants, rows, emitted, label):
    assert_tensors_equal(
        uint32_bits(download_replica(constants.history)),
        uint32_bits(st.history_image(rows, emitted)),
        f"{label}: presence history",
    )


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [{**DEVICE_PARAMS, "trace_region_size": 2_000_000}], indirect=True)
@pytest.mark.parametrize("rows", [1, st.MAX_ROWS])
def test_sampler_tail_reference_composite_and_reuse(mesh_device, rows):
    entry = fused.kernel(st.NAME)
    assert entry.tolerance == fused.BITWISE, f"{entry.name}: tolerance {entry.tolerance}"
    contract = Qwen38MeshContract(tuple(int(device_id) for device_id in mesh_device.get_device_ids()))
    cached = None
    for seed in (311, 312):
        rng = torch.Generator().manual_seed(seed)
        owned, released = [], []
        try:
            row_host = _candidate_rows(rows, rng)
            greedy_host = torch.zeros(st.TOKEN_ROW_SHAPE)
            greedy_host[..., :rows] = torch.randint(0, st.VOCAB_SIZE, (rows,), generator=rng).float()
            row = upload(mesh_device, row_host, ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)
            greedy = upload(mesh_device, greedy_host, ttnn.float32)
            owned += [row, greedy]
            constants = st.Qwen38TTNNSamplerTailConstants.build(mesh_device, contract, rows=rows)
            released.append(constants)
            # Some of each row's own candidates already emitted, so the presence penalty reorders the row.
            emitted = [
                ds.candidate_row_lanes(row_lanes)[1][torch.randperm(st.LANES, generator=rng)[:3]].tolist()
                for row_lanes in row_host[0, 0]
            ]
            constants.write_policy(POLICY)
            constants.rewrite_history(emitted)
            # Two eager draws: the second sees the bits the first set on the device.
            for step in range(2):
                draws = _draws(rows, rng)
                constants.write_uniforms(draws)
                token = entry.fused(row, greedy, constants)
                owned.append(token)
                label = f"eager draw {step}, seed={seed}"
                drawn, _ = _assert_drawn(token, constants, row_host, POLICY, draws, emitted, label)
                emitted = [tokens + [token_id] for tokens, token_id in zip(emitted, drawn)]
                _assert_history(constants, rows, emitted, label)
            # A draw the read lanes cannot prove: every row's guard bit set, the draw still the reference's.
            guard_host = _dominant_shard_rows(rows, rng)
            guard_row = upload(mesh_device, guard_host, ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)
            owned.append(guard_row)
            draws = _draws(rows, rng)
            constants.write_policy(GUARD_POLICY)
            constants.write_uniforms(draws)
            token = entry.fused(guard_row, greedy, constants)
            owned.append(token)
            label = f"guarded draw, seed={seed}"
            drawn, mask = _assert_drawn(token, constants, guard_host, GUARD_POLICY, draws, emitted, label)
            assert mask == (1 << rows) - 1, f"{label}: guard mask {mask:#x}, every one of the {rows} rows expected"
            emitted = [tokens + [token_id] for tokens, token_id in zip(emitted, drawn)]
            _assert_history(constants, rows, emitted, label)
            constants.write_policy(POLICY)
            if rows == 1:
                # The composite takes no presence penalty and one row; the greedy flag copies the greedy tile.
                composite = ds.Qwen38TTNNDeviceSamplerConstants.build(mesh_device, contract)
                released.append(composite)
                for policy in (dataclasses.replace(POLICY, presence_penalty=0.0), POLICY.greedy_policy()):
                    draws = _draws(rows, rng)
                    constants.write_policy(policy)
                    constants.write_uniforms(draws)
                    composite.write_policy(policy)
                    composite.write_uniform(draws[0])
                    token = entry.fused(row, greedy, constants)
                    reference = entry.composed(row, greedy, composite)
                    owned += [token, reference]
                    label = f"{policy}, seed={seed}"
                    token_host = download_replica(token)
                    assert_tensors_equal(token_host, download_replica(reference), f"{label}: fused versus composite")
                    if policy.greedy:
                        assert_tensors_equal(token_host, greedy_host, f"{label}: greedy token tile")
                    else:
                        _assert_drawn(token, constants, row_host, policy, draws, emitted, label)
                constants.write_policy(POLICY)
            cached = assert_program_cache_stable(mesh_device, cached)
            # The served loop: the host rewrites the history and the draws between replays of the captured TAIL.
            trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            try:
                token = entry.fused(row, greedy, constants)
            finally:
                ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
            owned.append(token)
            try:
                for replay in range(3):
                    draws = _draws(rows, rng)
                    constants.rewrite_history(emitted)
                    constants.write_uniforms(draws)
                    ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
                    label = f"replay {replay}, seed={seed}"
                    drawn, _ = _assert_drawn(token, constants, row_host, POLICY, draws, emitted, label)
                    _assert_history(
                        constants, rows, [tokens + [token_id] for tokens, token_id in zip(emitted, drawn)], label
                    )
                count = mesh_device.num_program_cache_entries()
                assert count == cached, f"program cache {count} entries after the replays, {cached} before the capture"
            finally:
                ttnn.release_trace(mesh_device, trace)
        finally:
            for tensor in owned:
                ttnn.deallocate(tensor)
            for holder in released:
                holder.release()
