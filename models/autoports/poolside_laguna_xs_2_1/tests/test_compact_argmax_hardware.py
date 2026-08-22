# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Explicit P150x2 correctness gate for Sampling1D's experimental compact argmax.

Normal Laguna/CPU suites skip this module. Run it only on an explicitly selected
two-ASIC P150 mesh with ``TT_LAGUNA_RUN_COMPACT_ARGMAX_HW=1``.

This is deliberately not a serving-performance qualification. The 2026-08-22
P150x2 gate measured compact B=1/V=100352 at 25.334 ms versus 10.755 ms for the
existing generic k=1 path, and B=32/V=1024 at 284.6 us versus 160.3 us. Compact
argmax therefore remains default-off experimental evidence only.
"""

from __future__ import annotations

import os

import pytest
import torch

import ttnn
from models.autoports.poolside_laguna_xs_2_1.tests.laguna_test_utils import close_mesh, open_mesh, resolve_profile
from models.common.modules.sampling.sampling_1d import Sampling1D

HARDWARE_GATE_ENV = "TT_LAGUNA_RUN_COMPACT_ARGMAX_HW"
VOCAB_SIZE = 100_352
PER_DEVICE_VOCAB = VOCAB_SIZE // 2
TRACE_REGION_SIZE = 128 << 20

pytestmark = pytest.mark.skipif(
    os.environ.get(HARDWARE_GATE_ENV) != "1",
    reason=f"set {HARDWARE_GATE_ENV}=1 to run the explicit P150x2 compact-argmax hardware gate",
)


@pytest.fixture(scope="module")
def p150x2_mesh():
    profile = resolve_profile("p150x2", trace_region_size=TRACE_REGION_SIZE)
    mesh = open_mesh(ttnn, profile)
    try:
        assert mesh.get_num_devices() == 2
        yield mesh
    finally:
        close_mesh(ttnn, mesh)


def _vocab_sharded_logits(logits: torch.Tensor, mesh):
    return ttnn.from_torch(
        logits,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(None, -1), mesh_shape=tuple(mesh.shape)),
    )


def _persistent_output(batch: int, mesh):
    return ttnn.from_torch(
        torch.zeros((1, 1, 1, batch), dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _output_source(values: torch.Tensor, mesh):
    return ttnn.from_torch(
        values.to(torch.int32).reshape(1, 1, 1, -1),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _assert_replicated_tokens(output, expected: torch.Tensor):
    expected = expected.reshape(-1).to(torch.int64)
    device_outputs = ttnn.get_device_tensors(output)
    assert len(device_outputs) == 2
    for device_index, device_output in enumerate(device_outputs):
        actual = ttnn.to_torch(device_output).reshape(-1)[: expected.numel()].to(torch.int64)
        assert torch.equal(actual, expected), (
            f"device {device_index} compact argmax mismatch:\n"
            f"  expected={expected.tolist()}\n"
            f"  actual={actual.tolist()}"
        )


def _controlled_logits(winners: list[int]) -> torch.Tensor:
    batch = len(winners)
    logits = torch.full((1, 1, batch, VOCAB_SIZE), -100.0, dtype=torch.bfloat16)
    for row, token in enumerate(winners):
        logits[0, 0, row, token] = 20.0
    return logits


def _sampler(batch: int, mesh) -> Sampling1D:
    sampler = Sampling1D(
        vocab_size=VOCAB_SIZE,
        mesh_device=mesh,
        max_batch_size=batch,
        max_top_k=32,
        allow_force_argmax=True,
        use_compact_argmax=True,
    )
    sampler.load_device_buffers()
    return sampler


@pytest.mark.parametrize(
    "winner",
    [
        pytest.param(17, id="winner-on-device-0"),
        pytest.param(PER_DEVICE_VOCAB + 23, id="winner-on-device-1"),
    ],
)
def test_compact_argmax_b1_exact_on_both_output_devices(p150x2_mesh, winner):
    logits = _vocab_sharded_logits(_controlled_logits([winner]), p150x2_mesh)
    output = _persistent_output(1, p150x2_mesh)

    returned, logprobs = _sampler(1, p150x2_mesh).decode_forward(logits, tt_out_tok=output)
    ttnn.synchronize_device(p150x2_mesh)

    assert logprobs is None
    _assert_replicated_tokens(returned, torch.tensor([winner]))
    _assert_replicated_tokens(output, torch.tensor([winner]))


def test_compact_argmax_b32_exact_with_winners_on_both_shards(p150x2_mesh):
    winners = [(17 + 37 * row) if row % 2 == 0 else (PER_DEVICE_VOCAB + 23 + 41 * row) for row in range(32)]
    logits = _vocab_sharded_logits(_controlled_logits(winners), p150x2_mesh)
    output = _persistent_output(32, p150x2_mesh)

    _sampler(32, p150x2_mesh).decode_forward(logits, tt_out_tok=output)
    ttnn.synchronize_device(p150x2_mesh)

    _assert_replicated_tokens(output, torch.tensor(winners))


def test_compact_argmax_cross_shard_tie_chooses_lowest_global_index(p150x2_mesh):
    lower = 123
    higher = PER_DEVICE_VOCAB + 456
    logits_host = torch.full((1, 1, 1, VOCAB_SIZE), -100.0, dtype=torch.bfloat16)
    logits_host[0, 0, 0, lower] = 20.0
    logits_host[0, 0, 0, higher] = 20.0
    logits = _vocab_sharded_logits(logits_host, p150x2_mesh)
    output = _persistent_output(1, p150x2_mesh)

    _sampler(1, p150x2_mesh).decode_forward(logits, tt_out_tok=output)
    ttnn.synchronize_device(p150x2_mesh)

    _assert_replicated_tokens(output, torch.tensor([lower]))


def test_compact_argmax_persistent_output_trace_replay(p150x2_mesh):
    winners = [(31 + 43 * row) if row % 2 == 0 else (PER_DEVICE_VOCAB + 29 + 47 * row) for row in range(32)]
    expected = torch.tensor(winners, dtype=torch.int64)
    logits = _vocab_sharded_logits(_controlled_logits(winners), p150x2_mesh)
    output = _persistent_output(32, p150x2_mesh)
    sampler = _sampler(32, p150x2_mesh)

    # Compile every operation and materialize all sampler buffers before capture.
    sampler.decode_forward(logits, tt_out_tok=output)
    ttnn.synchronize_device(p150x2_mesh)
    _assert_replicated_tokens(output, expected)

    trace_id = None
    try:
        trace_id = ttnn.begin_trace_capture(p150x2_mesh, cq_id=0)
        try:
            sampler.decode_forward(logits, tt_out_tok=output)
        finally:
            ttnn.end_trace_capture(p150x2_mesh, trace_id, cq_id=0)
            ttnn.synchronize_device(p150x2_mesh)

        zeros = _output_source(torch.zeros(32, dtype=torch.int32), p150x2_mesh)
        for _ in range(3):
            ttnn.copy_host_to_device_tensor(zeros, output)
            ttnn.execute_trace(p150x2_mesh, trace_id, cq_id=0, blocking=True)
            _assert_replicated_tokens(output, expected)
    finally:
        if trace_id is not None:
            ttnn.release_trace(p150x2_mesh, trace_id)
