# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The standard TTSampling stack on the held one-by-eight mesh under the model's fabric config: greedy equals the
argmax over the true vocabulary (lowest id on ties), one captured trace samples the top-k/top-p distribution, seeds
reproduce, and the token row of lane 0 matches the embedding's token-row format."""

import os
import time
import traceback

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.tests.tp_harness import DEVICE_PARAMS
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_ttsampler import (
    SAMPLER_BATCH,
    SAMPLER_PADDED_VOCAB_SIZE,
    SAMPLER_VOCAB_SIZE,
    Qwen38TTSampler,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE
from models.demos.blackhole.qwen38_flash_next.ttnn.embedding import TOKEN_ROW_SHAPE

pytestmark = pytest.mark.skipif(
    os.environ.get("QWEN38_FUSED_DEVICE_TEST") != "1", reason="requires held one-by-eight mesh"
)
MESH = pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
PARAMS = pytest.mark.parametrize("device_params", [{**DEVICE_PARAMS, "trace_region_size": 16_000_000}], indirect=True)

TEMPERATURE = 0.8
TOP_K = 20
TOP_P = 0.9
REPLAYS = 200
TIMED_REPLAYS = 100
SEED_REPLAYS = 12


def upload_logits(mesh, logits):
    return ttnn.from_torch(
        logits,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3),
    )


def read_tokens(out_tok):
    """Lane ids of device 0, asserting every device holds the same row."""

    rows = [
        ttnn.to_torch(local).reshape(-1)[:SAMPLER_BATCH].to(torch.int64) for local in ttnn.get_device_tensors(out_tok)
    ]
    assert all(torch.equal(rows[0], other) for other in rows[1:]), "devices disagree on the sampled ids"
    return rows[0]


def true_argmax(row):
    values = row[:SAMPLER_VOCAB_SIZE].float()
    return int((values == values.max()).nonzero()[0])


def greedy_logits(generator):
    logits = (torch.randn(1, 1, SAMPLER_BATCH, SAMPLER_PADDED_VOCAB_SIZE, generator=generator) * 2).bfloat16()
    logits[..., SAMPLER_VOCAB_SIZE:] = 30.0  # padded ids beat every real logit unless the sampler masks them
    per_device = SAMPLER_PADDED_VOCAB_SIZE // MESH_SHAPE[1]
    peak = 12.0
    for row, ids in enumerate(
        (
            (5, 100_000),  # tie across devices
            (3 * per_device + 7, 3 * per_device + 9),  # tie inside one device
            (200_000, SAMPLER_VOCAB_SIZE - 1),  # tie against the last real id
        )
    ):
        logits[0, 0, row, :SAMPLER_VOCAB_SIZE] = -5.0
        for column in ids:
            logits[0, 0, row, column] = peak
    return logits


def case_greedy(mesh, sampler):
    generator = torch.Generator().manual_seed(7)
    sampler.set_slot_params(range(SAMPLER_BATCH), temperature=1.0, top_k=1, top_p=1.0)
    out_tok = sampler.make_out_tok()
    for round_index in range(4):
        logits = greedy_logits(generator)
        device_logits = upload_logits(mesh, logits)
        sampler.write_step_seeds()
        sampler.sample(device_logits, out_tok)
        tokens = read_tokens(out_tok)
        expected = torch.tensor([true_argmax(logits[0, 0, row]) for row in range(SAMPLER_BATCH)])
        bad = (tokens != expected).nonzero().flatten().tolist()
        assert not bad, f"round {round_index}: rows {bad}: got {tokens[bad].tolist()} want {expected[bad].tolist()}"
        ttnn.deallocate(device_logits)
    return out_tok, tokens


def case_token_row(mesh, sampler, out_tok, tokens):
    row = sampler.token_row_from_out(out_tok)
    assert tuple(row.shape) == TOKEN_ROW_SHAPE, row.shape
    assert row.dtype == ttnn.float32 and row.layout == ttnn.TILE_LAYOUT
    for local in ttnn.get_device_tensors(row):
        values = ttnn.to_torch(local).reshape(-1)
        assert int(values[0]) == int(tokens[0]) and float(values[0]) == float(tokens[0]), (values[:4], tokens[0])
        assert not values[1:].any()


def expected_distribution(values):
    """Kept ids and probabilities as the kernel computes them: bf16 softmax over the top-k, the smallest prefix whose
    cumulative bf16 mass exceeds the bf16 top-p, renormalised over that prefix."""

    top = torch.topk(values.float(), TOP_K)
    scaled = (top.values * torch.tensor(1 / TEMPERATURE).bfloat16().float()).bfloat16().float()
    probabilities = torch.softmax(scaled, dim=0).bfloat16().float()
    cumulative = torch.cumsum(probabilities, dim=0)
    top_p = float(torch.tensor(TOP_P).bfloat16())
    kept = int((cumulative > top_p).nonzero()[0]) + 1
    return top.indices[:kept], probabilities[:kept] / probabilities[:kept].sum(), cumulative


def sampling_row():
    """One row: forty planted logits with the spacing whose cumulative masses stay farthest from top-p."""

    generator = torch.Generator().manual_seed(11)
    best = None
    for spacing in (0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32):
        planted = (10 - spacing * torch.arange(40)).bfloat16()
        scaled = (planted[:TOP_K].float() * 1.25).bfloat16().float()
        cumulative = torch.cumsum(torch.softmax(scaled, dim=0).bfloat16().float(), dim=0)
        margin = float((cumulative - float(torch.tensor(TOP_P).bfloat16())).abs().min())
        if best is None or margin > best[0]:
            best = (margin, spacing, planted)
    margin, spacing, planted = best
    row = (torch.randn(SAMPLER_PADDED_VOCAB_SIZE, generator=generator) * 0.5 - 30).bfloat16()
    row[SAMPLER_VOCAB_SIZE:] = 30.0
    ids = torch.randperm(SAMPLER_VOCAB_SIZE, generator=generator)[:40]
    row[ids] = planted
    print(f"sampling row: spacing {spacing}, min |cumulative - top_p| {margin:.4f}")
    return row, ids


def capture(mesh, sampler, device_logits, out_tok):
    sampler.sample(device_logits, out_tok)
    ttnn.synchronize_device(mesh)
    trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
    sampler.sample(device_logits, out_tok)
    ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh)
    return trace_id


def replay(mesh, sampler, trace_id, out_tok):
    sampler.write_step_seeds()
    ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
    return read_tokens(out_tok)


def case_statistics(mesh, sampler, trace_id, out_tok, expected_ids, expected_probabilities, allowed):
    from scipy.stats import chisquare

    sampler.set_slot_params(range(SAMPLER_BATCH), temperature=TEMPERATURE, top_k=TOP_K, top_p=TOP_P)
    drawn = torch.cat([replay(mesh, sampler, trace_id, out_tok) for _ in range(REPLAYS)])
    outside = [int(t) for t in drawn if int(t) not in allowed]
    assert not outside, f"{len(outside)} draws outside the top-k/top-p set, e.g. {outside[:5]}"
    counts = torch.stack([(drawn == int(i)).sum() for i in expected_ids]).double()
    expected = expected_probabilities.double() * drawn.numel()
    assert float(expected.min()) >= 5
    statistic, p_value = chisquare(counts.numpy(), expected.numpy())
    print(f"chi-square over {drawn.numel()} draws, {len(expected_ids)} bins: stat {statistic:.2f} p {p_value:.4f}")
    print("counts  ", counts.int().tolist())
    print("expected", [round(float(e), 1) for e in expected])
    assert p_value > 1e-4, f"chi-square p {p_value}"
    return p_value


def case_seeds(mesh, sampler, trace_id, out_tok):
    def run(seed):
        sampler.set_slot_params(range(SAMPLER_BATCH), temperature=TEMPERATURE, top_k=TOP_K, top_p=TOP_P, seed=seed)
        assert sampler.seeds_active
        return torch.stack([replay(mesh, sampler, trace_id, out_tok) for _ in range(SEED_REPLAYS)])

    first, second, other = run(1234), run(1234), run(4321)
    assert torch.equal(first, second), "same seed, different sequences"
    assert not torch.equal(first, other), "different seeds, same sequences"
    assert len({tuple(first[:, lane].tolist()) for lane in range(SAMPLER_BATCH)}) > 1, "all lanes drew the same stream"
    print(
        f"seed repro: identical across runs; seed 4321 differs in {(first != other).sum().item()}/{first.numel()} draws"
    )


def case_timing(mesh, sampler, trace_id):
    sampler.set_slot_params(range(SAMPLER_BATCH), temperature=TEMPERATURE, top_k=TOP_K, top_p=TOP_P)
    sampler.write_step_seeds()
    ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    start = time.perf_counter()
    for _ in range(TIMED_REPLAYS):
        ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    per_call = (time.perf_counter() - start) / TIMED_REPLAYS
    print(f"sample trace: {per_call * 1e3:.3f} ms per call (mean of {TIMED_REPLAYS} replays)")


@MESH
@PARAMS
def test_ttsampler(mesh_device):
    from models.tt_transformers.tt.ccl import TT_CCL

    mesh = mesh_device
    sampler = Qwen38TTSampler(mesh, TT_CCL(mesh))
    failures = {}

    def run(name, function, *args):
        try:
            result = function(*args)
            print(f"CASE {name}: pass")
            return result
        except BaseException:
            failures[name] = traceback.format_exc()
            print(f"CASE {name}: FAIL\n{failures[name]}")
            return None

    greedy = run("greedy", case_greedy, mesh, sampler)
    if greedy is not None:
        run("token_row", case_token_row, mesh, sampler, *greedy)

    row, planted_ids = sampling_row()
    kept, probabilities, _ = expected_distribution(row[planted_ids])
    expected_ids = planted_ids[kept]
    allowed = {int(i) for i in expected_ids}
    device_logits = upload_logits(mesh, row.reshape(1, 1, 1, -1).repeat(1, 1, SAMPLER_BATCH, 1))
    out_tok = sampler.make_out_tok()
    sampler.set_slot_params(range(SAMPLER_BATCH), temperature=TEMPERATURE, top_k=TOP_K, top_p=TOP_P)
    trace_id = capture(mesh, sampler, device_logits, out_tok)
    run("statistics", case_statistics, mesh, sampler, trace_id, out_tok, expected_ids, probabilities, allowed)
    run("seeds", case_seeds, mesh, sampler, trace_id, out_tok)
    run("timing", case_timing, mesh, sampler, trace_id)
    ttnn.release_trace(mesh, trace_id)

    assert not failures, f"failed cases: {sorted(failures)}"
