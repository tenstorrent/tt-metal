# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Sol-Attn schedule (which denoise steps and blocks stay dense) and the per-request route constants. Host only,
no device."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from ....models.transformers.minimax_h3.sol_prepass import SolGeometry, build_route_constants, route_factors
from ....pipelines.minimax_h3.packing import video_latent_num_frames
from ....pipelines.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from ....pipelines.minimax_h3.pipeline_minimax_h3_turbo import MiniMaxH3TurboPipeline
from ....pipelines.minimax_h3.policy import SolSchedule

_SOL_ENV = ("H3_SOL_ATTN", "H3_SOL_TAU", "H3_SOL_DENSE_STEPS")


_EROS = SolSchedule(tau=1.13, dense_steps=3, dense_layers=0, sinks=True)  # the 10Eros-Max card and a third dense step
_EROS_DEFAULT = replace(_EROS, max_frames=158)  # with no switch set: clips up to 6 s
_HF = SolSchedule(tau=1.0, dense_steps=2, dense_layers=2, sinks=False)  # NVIDIA's HyperFlow recipe


@pytest.mark.parametrize(
    ("file", "task", "env", "expected"),
    [
        ("/models/v1/v1t.safetensors", "t2va", {}, _EROS_DEFAULT),
        ("/models/v1/v1t.safetensors", "t2va", {"H3_SOL_ATTN": "0"}, None),  # the off switch
        ("/models/v1/v1t.safetensors", "ref2va", {}, None),
        ("/models/v1/v1t.safetensors", "ref2va", {"H3_SOL_ATTN": "1"}, SolSchedule()),
        ("/models/v1/v1n.safetensors", "t2va", {}, None),  # the non-Turbo build
        (None, "t2va", {}, None),
    ],
)
def test_the_10eros_max_turbo_file_runs_sol_attn_by_default(monkeypatch, file, task, env, expected):
    pipeline = _pipeline(monkeypatch, env)
    pipeline.task, pipeline.transformer_file = task, file and Path(file)
    assert pipeline._requested_sol_schedule() == expected


@pytest.mark.parametrize(("hyperflow", "expected"), [(object(), _HF), (None, SolSchedule())])
def test_turbo_pipeline_names_the_hyperflow_preset(monkeypatch, hyperflow, expected):
    pipeline = _pipeline(monkeypatch, {"H3_SOL_ATTN": "1"}, MiniMaxH3TurboPipeline)
    pipeline.hyperflow = hyperflow
    assert pipeline._requested_sol_schedule() == expected


def _pipeline(monkeypatch, env, cls=MiniMaxH3Pipeline):
    """A pipeline without `__init__`, holding only what the Sol-Attn hookup reads, under `env`."""
    for name in _SOL_ENV:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    pipeline = cls.__new__(cls)
    pipeline.task, pipeline.transformer_file = "t2va", None
    pipeline.dynamic_length = False
    pipeline._sol_schedule = None
    pipeline._warming = False
    return pipeline


@pytest.mark.parametrize(
    ("env", "warming", "num_frames", "dense"),
    [
        ({}, False, 158, [True] * 3 + [False] * 5),
        ({}, False, 175, [True] * 8),
        ({}, False, 243, [True] * 8),
        ({"H3_SOL_ATTN": "1"}, False, 243, [True] * 3 + [False] * 5),
        ({}, True, 243, [True] + [False] * 7),  # a rung's warm-up clip still captures the sparse trace
        # and the dense one, which a clip over the default's length replays, for a schedule with no dense step too
        ({"H3_SOL_DENSE_STEPS": "0"}, True, 158, [True] + [False] * 7),
    ],
)
def test_the_default_runs_a_clip_over_its_length_dense(monkeypatch, env, warming, num_frames, dense):
    pipeline = _pipeline(monkeypatch, env)
    pipeline.transformer_file, pipeline._warming = Path("/models/v1/v1t.safetensors"), warming
    pipeline._sol_schedule = pipeline._requested_sol_schedule()
    latent_frames = video_latent_num_frames(num_frames)
    assert [pipeline._is_sol_dense_step(step, num_latent_frames=latent_frames) for step in range(8)] == dense


_RULE = {"band": 1, "force_first_block": True, "dense_prefix_queries": True}


def _mask(hit, const_or, valid_blocks):
    """The route kernel's block mask for a score that passes (hit 1) or misses (hit 0) everywhere."""
    return ((hit + const_or) > 0.5).float() * valid_blocks


def _reference_masks(geo, sink_tokens, rule=_RULE):
    c = build_route_constants(geo, sink_tokens=sink_tokens, scale=0.125, **rule)
    return [_mask(hit, c["const_or"], c["valid_blocks"]) for hit in (0.0, 1.0)], c["n_valid_blocks"]


@pytest.mark.parametrize(
    "shards, rows, logical_n, sink_tokens",
    [
        (8, 9600, 73419, 843),  # 6 s rung
        (8, 9600, 73419, 0),
        (8, 9600, 76800, 843),  # no padding
        (8, 10336, 77467, 4607),  # ragged last block per shard, sink longer than a few blocks
        (8, 6592, 51983, 559),
        (8, 6592, 20000, 7000),  # whole shards of padding, sink past the first shard
        (8, 14400, 110219, 843),  # 10 s rung
        (4, 9600, 38000, 300),
    ],
)
@pytest.mark.parametrize("rule", [_RULE, {"band": 0, "force_first_block": False, "dense_prefix_queries": False}])
def test_route_factors_compose_to_the_route_constants(shards, rows, logical_n, sink_tokens, rule):
    geo = SolGeometry(shards=shards, rows=rows, logical_n=logical_n)
    f = route_factors(geo, sink_tokens=sink_tokens, **rule)
    const_or = f["base"] + f["sink_cols"] + f["prefix_rows"]
    valid_blocks = f["q_full"] * f["full_cols"]
    masks, n_valid_blocks = _reference_masks(geo, sink_tokens, rule)
    assert f["n_valid_blocks"] == n_valid_blocks
    for hit, expected in zip((0.0, 1.0), masks):
        assert torch.equal(_mask(hit, const_or, valid_blocks), expected)


@pytest.mark.parametrize(("rows", "logical_n"), [(9600, 73419), (14400, 110219)])  # the 6 s and 10 s rungs
def test_no_score_drops_a_pair_with_a_prefix_token(rows, logical_n):
    """The text, condition and audio rows are the sink prefix: only video-to-video pairs can go sparse."""
    geo, sink_tokens = SolGeometry(shards=8, rows=rows, logical_n=logical_n), 843
    c = build_route_constants(geo, sink_tokens=sink_tokens, scale=0.125, **_RULE)
    miss, valid = _mask(0.0, c["const_or"], c["valid_blocks"]), c["valid_blocks"][0, 0]
    prefix = [(s, b) for s in range(geo.shards) for b in range(geo.nb) if s * geo.rows + b * geo.block < sink_tokens]
    assert len(prefix) == -(-sink_tokens // geo.block)
    for s, b in prefix:
        column = s * geo.nb_pad + b
        assert valid[: geo.nb, column].all()  # a full block: computed exactly, not pooled
        assert torch.equal(miss[s, 0, b], valid[b])  # as a query: every key block
        assert (miss[:, 0, :, column] == valid[:, column]).all()  # as a key: for every query block
