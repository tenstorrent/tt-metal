# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""``decode_sample_candidates`` for a DSpark verify block: logits ``[1, 1, R, V_local]`` (TILE, rows 0 .. R - 1 of
the tiles, what ``ttnn.matmul`` with M = R gives) -> per row the top-k values ``[1, 1, R, 2k]`` (TILE) and indices
``[1, 1, R, 2k]`` (INT32 ROW_MAJOR), as ``ttnn.sampling`` takes R users. Then the whole device sampler
(:class:`DeepSeekV41DeviceSampler` with ``rows=4``: candidates, ``ttnn.sampling``, the TP4 cross-rank merge) on
one 1x4 stage.

* ``test_sample_candidates_rows``: per row, values (times inv_temp) and indices exactly vs a host top-k;
* ``test_sample_candidates_rows_perf``: R = 1 vs R = 4 at the TP4 shard width (129280 / 4). Profile::

    python -m tracy -r -p -v --no-web-server -t 8123 -o <dir> -m pytest <this file>::test_sample_candidates_rows_perf -s
    python3 dv_prof.py <dir>/reports/<ts>/ops_perf_results_*.csv SampleCandidates

* ``test_device_sampler_rows_greedy``: each row's token is that row's exact argmax over the full vocab;
* ``test_device_sampler_rows_temperature``: each row's draws stay in that row's union of the ranks' top-k, and
  their empirical distribution matches the per-row target.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.deepseek_v41_flash.tt.lm_head import (
    SAMPLE_K,
    SAMPLE_WORKERS,
    SAMPLE_WORKERS_ROWS,
    DeepSeekV41DeviceSampler,
)

FILL = -1.0e9
VOCAB = 129280
TP = 4
LOCAL = VOCAB // TP


def _reference(row: torch.Tensor, k: int, inv_temp: float, offset: int):
    vals = row.reshape(-1).bfloat16()
    order = sorted(range(vals.numel()), key=lambda i: (-float(vals[i]), i))[:k]
    scaled = (vals[order].float() * torch.tensor(inv_temp).bfloat16().float()).bfloat16()
    want_vals = torch.cat([scaled.float(), torch.full((k,), FILL).bfloat16().float()])
    want_idx = torch.tensor([i + offset for i in order] + [0] * k, dtype=torch.int64)
    return want_vals, want_idx


def _inputs(device, logits: torch.Tensor, inv_temp: float, offset: int):
    tt_logits = ttnn.from_torch(
        logits, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_temp = ttnn.from_torch(
        torch.full((1, 1, 1, 1), inv_temp), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    tt_offset = ttnn.from_torch(
        torch.full((1, 1, 1, 1), offset, dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    return tt_logits, tt_temp, tt_offset


@pytest.mark.parametrize("rows", (4, 1, 2, 3))
@pytest.mark.parametrize("v", (LOCAL, 1000, 64, 70016))
@pytest.mark.parametrize("k", (32, 16))
@pytest.mark.parametrize("inv_temp", (1.0, 1.4375))
@pytest.mark.parametrize("ties", (False, True))
def test_sample_candidates_rows(device, rows, v, k, inv_temp, ties, workers=64):
    torch.manual_seed(v + k + rows)
    logits = torch.randn(1, 1, rows, v) * 4
    if ties:
        logits = torch.round(logits)
    # Rows differ in where their maxima sit: row j's winner near column j * v / rows.
    for j in range(rows):
        logits[0, 0, j, (j * v) // rows] = 30.0 + j
    logits = logits.bfloat16().float()
    offset = 3 * LOCAL
    tt_logits, tt_temp, tt_offset = _inputs(device, logits, inv_temp, offset)
    for _ in range(2):  # the second call runs from the program cache
        vals, idx = ttnn.experimental.deepseek.decode_sample_candidates(
            tt_logits, tt_temp, tt_offset, k=k, fill_value=FILL, num_workers=workers
        )
        assert vals.layout == ttnn.TILE_LAYOUT and vals.dtype == ttnn.bfloat16
        assert list(vals.shape) == [1, 1, rows, 2 * k]
        assert idx.layout == ttnn.ROW_MAJOR_LAYOUT and idx.dtype == ttnn.int32
        assert list(idx.shape) == [1, 1, rows, 2 * k]
        got_vals = ttnn.to_torch(vals).float().reshape(rows, 2 * k)
        got_idx = ttnn.to_torch(idx).to(torch.int64).reshape(rows, 2 * k)
        for j in range(rows):
            want_vals, want_idx = _reference(logits[0, 0, j], k, inv_temp, offset)
            assert torch.equal(got_idx[j], want_idx), (j, got_idx[j, :k], want_idx[:k])
            assert torch.equal(got_vals[j], want_vals), (j, got_vals[j, :k], want_vals[:k])


@pytest.mark.parametrize("rows", (4, 1))
@pytest.mark.parametrize("v", (LOCAL, 70016))
@pytest.mark.parametrize("ties", (False, True))
def test_sample_candidates_rows_all_cores(device, rows, v, ties):
    """The core count the device sampler uses for a verify block (the whole grid)."""
    test_sample_candidates_rows(device, rows, v, SAMPLE_K, 1.0, ties, workers=SAMPLE_WORKERS_ROWS)


@pytest.mark.parametrize("rows", (1, 4))
def test_sample_candidates_rows_perf(device, rows):
    from tracy import signpost

    torch.manual_seed(0)
    tt_logits, tt_temp, tt_offset = _inputs(device, torch.randn(1, 1, rows, LOCAL) * 4, 1.0, 0)

    def run():
        # The core counts the device sampler uses.
        return ttnn.experimental.deepseek.decode_sample_candidates(
            tt_logits,
            tt_temp,
            tt_offset,
            k=SAMPLE_K,
            fill_value=FILL,
            num_workers=SAMPLE_WORKERS.get(rows, SAMPLE_WORKERS_ROWS),
        )

    run()
    ttnn.synchronize_device(device)
    signpost(f"sample_candidates-r{rows}")
    for _ in range(20):
        run()
    ttnn.synchronize_device(device)


# ---- the device sampler chain on one 1x4 stage (needs the Galaxy's fabric) ----

PARENT_MESH = (8, 4)
PCIE_ALIGNMENT = 64
_DEVICE_PARAMS = [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY, "trace_region_size": 16 << 20}]
_FLOOR = -30.0


@pytest.fixture
def stage(mesh_device):
    return mesh_device.create_submesh(ttnn.MeshShape(1, TP), ttnn.MeshCoordinate(0, 0))


def _host_sharded(logits: torch.Tensor, mesh, on_device: bool = False):
    return ttnn.from_torch(
        logits.reshape(1, 1, -1, VOCAB),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh if on_device else None,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
    )


def _capture(device, sampler, logits_dev):
    sampler(logits_dev)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    out = sampler(logits_dev)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    return tid, out


def _draw(device, tid, out, rows: int) -> list[int]:
    ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    return ttnn.to_torch(ttnn.get_device_tensors(out)[0]).reshape(-1)[:rows].tolist()


@pytest.mark.parametrize("device_params", _DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [PARENT_MESH], indirect=True, ids=["8x4"])
def test_device_sampler_rows_greedy(stage):
    rows = 4
    generator = torch.Generator().manual_seed(0)
    logits_dev = _host_sharded(torch.zeros(rows, VOCAB), stage, on_device=True)
    sampler = DeepSeekV41DeviceSampler(stage, LOCAL, PCIE_ALIGNMENT, rows=rows)
    tid, out = _capture(stage, sampler, logits_dev)
    assert list(out.shape) == [1, 1, 1, PCIE_ALIGNMENT // 4]
    try:
        for trial in range(6):
            logits = 4.0 * torch.randn(rows, VOCAB, generator=generator)
            targets = []
            for j in range(rows):
                # One clear winner per row, the rows' winners on different ranks.
                t = (trial + j) % TP * LOCAL + int(torch.randint(LOCAL, (1,), generator=generator))
                logits[j, t] = logits[j].max() + 1.0
                targets.append(t)
            ttnn.copy_host_to_device_tensor(_host_sharded(logits, stage), logits_dev)
            got = _draw(stage, tid, out, rows)
            logger.info(f"greedy trial {trial}: device {got}, argmax {targets}")
            assert got == targets
    finally:
        ttnn.release_trace(stage, tid)


def _support_logits(generator: torch.Generator, shift: int) -> torch.Tensor:
    """A row with ~60 support tokens spread unevenly over the ranks (rank ``shift`` holds the most)."""
    logits = torch.full((VOCAB,), _FLOOR)
    counts, offsets = (SAMPLE_K + 8, 10, 10, 10), (0.5, 0.0, -0.75, -1.5)
    for i, (n, offset) in enumerate(zip(counts, offsets)):
        r = (i + shift) % TP
        cols = torch.randperm(LOCAL, generator=generator)[:n] + r * LOCAL
        values = offset + 2.0 * torch.rand(n, generator=generator)
        if n > SAMPLE_K:
            values = values.sort(descending=True).values
            values[SAMPLE_K:] = offset - 2.0 - torch.rand(n - SAMPLE_K, generator=generator)
        logits[cols] = values
    return logits


def _target(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    inv = torch.tensor(1.0 / temperature).bfloat16()
    scaled = (logits.bfloat16() * inv).float()
    keep = torch.zeros(VOCAB, dtype=torch.bool)
    for r in range(TP):
        keep[scaled[r * LOCAL : (r + 1) * LOCAL].topk(SAMPLE_K).indices + r * LOCAL] = True
    return torch.where(keep, torch.softmax(torch.where(keep, scaled, -torch.inf), dim=-1), 0.0)


@pytest.mark.parametrize("device_params", _DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [PARENT_MESH], indirect=True, ids=["8x4"])
def test_device_sampler_rows_temperature(stage):
    rows, temperature, draws = 4, 0.8, 2000
    generator = torch.Generator().manual_seed(1)
    logits = torch.stack([_support_logits(generator, j) for j in range(rows)])
    want = [_target(logits[j], temperature) for j in range(rows)]
    logits_dev = _host_sharded(logits, stage, on_device=True)
    sampler = DeepSeekV41DeviceSampler(stage, LOCAL, PCIE_ALIGNMENT, rows=rows)
    sampler.set_sampling(temperature, seed=1234)
    tid, out = _capture(stage, sampler, logits_dev)
    try:
        counts = torch.zeros(rows, VOCAB)
        for _ in range(draws):
            for j, t in enumerate(_draw(stage, tid, out, rows)):
                counts[j, t] += 1
    finally:
        ttnn.release_trace(stage, tid)
    for j in range(rows):
        got = counts[j] / draws
        tv = 0.5 * float((got - want[j]).abs().sum())
        outside = int(counts[j][want[j] == 0].sum())
        ranks_got = got.reshape(TP, LOCAL).sum(-1)
        ranks_want = want[j].reshape(TP, LOCAL).sum(-1)
        logger.info(
            f"row {j}: TV {tv:.4f}, rank mass {[round(float(x), 3) for x in ranks_got]} vs "
            f"{[round(float(x), 3) for x in ranks_want]}, {outside} outside"
        )
        assert outside == 0, (j, outside)
        # ~60 support tokens: a correct sampler's expected TV at 2000 draws is ~0.06.
        assert tv < 0.1, (j, tv)
        assert (ranks_got - ranks_want).abs().max() < 0.04, j
