# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""On-device next-token sampling (:class:`DeepSeekV41DeviceSampler`) on one 1x4 stage, traced as in decode.

The logits are synthetic, column-sharded over the stage as ``lm_head``'s are, and uploaded in place between
replays of one captured trace, so these also check that the sampler's PRNG advances from replay to replay.

* greedy: every pick is the exact argmax, wherever its rank;
* temperature: the empirical distribution of many replays against ``softmax(logits / T)`` over the union of the
  ranks' top-``SAMPLE_K`` (the sampler's target), per token and per rank -- the per-rank masses are what the
  gathered log-partitions decide -- and nothing outside that union is ever drawn;
* the same seed replays the same draws.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.deepseek_v41_flash.tt.lm_head import SAMPLE_K, DeepSeekV41DeviceSampler

PARENT_MESH = (8, 4)
TP_SIZE = 4
VOCAB = 129280
LOCAL = VOCAB // TP_SIZE
PCIE_ALIGNMENT = 64
DRAWS = 4000
# Logit of every token outside the support: its total mass, ~1e5 * e^-30, never shows in DRAWS draws.
_FLOOR = -30.0


def _sharded(logits: torch.Tensor, device) -> ttnn.Tensor:
    return ttnn.from_torch(
        logits.reshape(1, 1, 1, VOCAB),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        mesh_mapper=ttnn.ShardTensorToMesh(device, dim=-1),
    )


def _upload(logits: torch.Tensor, dst: ttnn.Tensor) -> None:
    host = ttnn.from_torch(
        logits.reshape(1, 1, 1, VOCAB),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(dst.device(), dim=-1),
    )
    ttnn.copy_host_to_device_tensor(host, dst)


def _capture(device, sampler: DeepSeekV41DeviceSampler, logits: ttnn.Tensor) -> tuple[int, ttnn.Tensor]:
    """Compile the sampler once, then capture it: ``(trace id, its output)``."""
    sampler(logits)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    out = sampler(logits)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    return tid, out


def _draw(device, tid: int, out: ttnn.Tensor) -> int:
    """Replay once and read the token rank 0 -- the D2H socket's chip -- picked."""
    ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    return int(ttnn.to_torch(ttnn.get_device_tensors(out)[0]).reshape(-1)[0])


def _target(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """The sampler's distribution over ``VOCAB``: ``softmax`` at ``temperature`` of the bf16 logits, scaled in bf16
    as on device, over the union of each rank's top-``SAMPLE_K`` and zero elsewhere."""
    inv = torch.tensor(1.0 / temperature).bfloat16()
    scaled = (logits.bfloat16() * inv).float()
    keep = torch.zeros(VOCAB, dtype=torch.bool)
    for r in range(TP_SIZE):
        top = scaled[r * LOCAL : (r + 1) * LOCAL].topk(SAMPLE_K).indices + r * LOCAL
        keep[top] = True
    return torch.where(keep, torch.softmax(torch.where(keep, scaled, -torch.inf), dim=-1), 0.0)


def _support_logits(generator: torch.Generator) -> torch.Tensor:
    """Logits with unequal rank masses: rank 0 holds more support tokens than its top-``SAMPLE_K`` keeps (its 8
    weakest must never be drawn), ranks 1-3 ten each at falling offsets."""
    logits = torch.full((VOCAB,), _FLOOR)
    counts, offsets = (SAMPLE_K + 8, 10, 10, 10), (0.5, 0.0, -0.75, -1.5)
    for r, (n, offset) in enumerate(zip(counts, offsets)):
        cols = torch.randperm(LOCAL, generator=generator)[:n] + r * LOCAL
        values = offset + 2.0 * torch.rand(n, generator=generator)
        if n > SAMPLE_K:
            values = values.sort(descending=True).values
            values[SAMPLE_K:] = offset - 2.0 - torch.rand(n - SAMPLE_K, generator=generator)
        logits[cols] = values
    return logits


@pytest.fixture
def stage(mesh_device):
    return mesh_device.create_submesh(ttnn.MeshShape(1, TP_SIZE), ttnn.MeshCoordinate(0, 0))


_DEVICE_PARAMS = [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY, "trace_region_size": 16 << 20}]


@pytest.mark.parametrize("device_params", _DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [PARENT_MESH], indirect=True, ids=["8x4"])
def test_device_sampling_greedy(stage):
    generator = torch.Generator().manual_seed(0)
    logits = 4.0 * torch.randn(VOCAB, generator=generator)
    logits_dev = _sharded(logits, stage)
    sampler = DeepSeekV41DeviceSampler(stage, LOCAL, PCIE_ALIGNMENT)
    tid, out = _capture(stage, sampler, logits_dev)
    try:
        for trial in range(8):
            logits = 4.0 * torch.randn(VOCAB, generator=generator)
            # One clear winner, on each rank in turn: bf16 ties at the top are left to the hardware's order.
            target = trial % TP_SIZE * LOCAL + int(torch.randint(LOCAL, (1,), generator=generator))
            logits[target] = logits.max() + 1.0
            _upload(logits, logits_dev)
            got = _draw(stage, tid, out)
            logger.info(f"greedy trial {trial}: device {got}, argmax {target} (rank {target // LOCAL})")
            assert got == target
    finally:
        ttnn.release_trace(stage, tid)


@pytest.mark.parametrize("device_params", _DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [PARENT_MESH], indirect=True, ids=["8x4"])
@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_device_sampling_distribution(stage, temperature):
    logits = _support_logits(torch.Generator().manual_seed(1))
    want = _target(logits, temperature)
    full = torch.softmax(logits.bfloat16().float() / temperature, dim=-1)
    logits_dev = _sharded(logits, stage)
    sampler = DeepSeekV41DeviceSampler(stage, LOCAL, PCIE_ALIGNMENT)
    sampler.set_sampling(temperature, seed=1234)
    tid, out = _capture(stage, sampler, logits_dev)
    try:
        counts = torch.zeros(VOCAB)
        for _ in range(DRAWS):
            counts[_draw(stage, tid, out)] += 1
        again = []
        for seed in (99, 99):
            sampler.set_sampling(temperature, seed=seed)
            again.append([_draw(stage, tid, out) for _ in range(64)])
    finally:
        ttnn.release_trace(stage, tid)

    got = counts / DRAWS
    tv = 0.5 * float((got - want).abs().sum())
    ranks_got = got.reshape(TP_SIZE, LOCAL).sum(-1)
    ranks_want = want.reshape(TP_SIZE, LOCAL).sum(-1)
    outside = int(counts[want == 0].sum())
    logger.info(
        f"T={temperature}: TV to target {tv:.4f} over {DRAWS} draws; rank mass device "
        f"{[round(float(x), 3) for x in ranks_got]} vs target {[round(float(x), 3) for x in ranks_want]}; "
        f"{outside} draws outside the ranks' top-{SAMPLE_K}; the truncation drops "
        f"{float(full[want == 0].sum()):.4f} of the exact softmax"
    )
    assert outside == 0, f"{outside} draws outside the union of the ranks' top-{SAMPLE_K}"
    # ~60 support tokens: a correct sampler's expected TV at 4000 draws is ~0.04.
    assert tv < 0.07, f"TV {tv:.4f} between device draws and the target distribution"
    # A rank's mass has a standard error under 0.008 at 4000 draws.
    assert (ranks_got - ranks_want).abs().max() < 0.03, "per-rank mass off: the gathered log-partitions"
    assert again[0] == again[1], "the same seed drew different tokens"
    assert len(set(again[0])) > 1, "64 replays drew one token: the PRNG did not advance"
