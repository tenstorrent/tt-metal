# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Sol-Attn schedule (which denoise steps and blocks stay dense) and the per-request route constants. Host only,
no device."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import torch

import ttnn

from ....models.transformers.minimax_h3.attention_minimax_h3 import MiniMaxH3Attention
from ....models.transformers.minimax_h3.sol_prepass import SolGeometry, build_route_constants, route_factors
from ....pipelines.minimax_h3.packing import video_latent_num_frames
from ....pipelines.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from ....pipelines.minimax_h3.pipeline_minimax_h3_turbo import MiniMaxH3TurboPipeline
from ....pipelines.minimax_h3.policy import SolSchedule, sol_schedule
from ....pipelines.minimax_h3.scheduler import MiniMaxH3Scheduler

_SOL_ENV = ("H3_SOL_ATTN", "H3_SOL_TAU", "H3_SOL_DENSE_STEPS", "H3_SOL_POOLED")


@pytest.mark.parametrize("env", [{}, {"H3_SOL_ATTN": "0"}, {"H3_SOL_ATTN": "0", "H3_SOL_DENSE_STEPS": "0"}])
def test_off_switch_is_dense(env):
    assert sol_schedule(env) is None


_EROS = SolSchedule(tau=1.13, dense_steps=3, dense_layers=0, sinks=True)  # the 10Eros-Max card and a third dense step
_EROS_DEFAULT = replace(_EROS, max_frames=158)  # with no switch set: clips up to 6 s
_HF = SolSchedule(tau=1.0, dense_steps=2, dense_layers=2, sinks=False)  # the HyperFlow arm of the Sol sheet


@pytest.mark.parametrize(
    ("file", "task", "env", "expected"),
    [
        ("/models/v1/v1t.safetensors", "t2va", {}, _EROS_DEFAULT),
        ("/models/v1/v1t.safetensors", "t2va", {"H3_SOL_DENSE_STEPS": "2"}, replace(_EROS_DEFAULT, dense_steps=2)),
        ("/models/v1/v1t.safetensors", "t2va", {"H3_SOL_ATTN": "0"}, None),
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
    pipeline.bucket_denoise = True
    pipeline.sp_factor = 8
    pipeline._sol_key = pipeline._sol_schedule = None
    pipeline._warming = False
    pipeline._buckets = {76800: SimpleNamespace(warm=True)}
    pipeline._request_state = SimpleNamespace(warm=True)
    pipeline._release_denoise_traces = Mock()
    pipeline._log = lambda message: None
    return pipeline


def _states(pipeline):
    return [state.warm for state in (*pipeline._buckets.values(), pipeline._request_state)]


@pytest.mark.parametrize("env", [{}, {"H3_SOL_ATTN": "0", "H3_SOL_DENSE_STEPS": "2"}])
def test_pipeline_off_switch_touches_nothing(monkeypatch, env):
    pipeline, transformer = _pipeline(monkeypatch, env), Mock()
    pipeline._set_sol_attention(transformer, 76800, 73419, 843)
    assert (pipeline._sol_key, pipeline._sol_schedule) == (None, None)
    assert _states(pipeline) == [True, True]
    pipeline._release_denoise_traces.assert_not_called()
    transformer.set_sol_request.assert_not_called()


@pytest.mark.parametrize(
    "file, env, dense",
    [
        (None, {}, [False]),
        (None, {"H3_SOL_ATTN": "0", "H3_SOL_DENSE_STEPS": "2"}, [False]),
        (None, {"H3_SOL_ATTN": "1"}, [True, False]),
        (None, {"H3_SOL_ATTN": "1", "H3_SOL_DENSE_STEPS": "1"}, [True, False]),
        (None, {"H3_SOL_ATTN": "1", "H3_SOL_DENSE_STEPS": "0"}, [False, False]),
        # On by default, a clip over the preset's length runs dense: warm-up captures that trace too.
        ("/models/v1/v1t.safetensors", {"H3_SOL_DENSE_STEPS": "0"}, [True, False]),
    ],
)
def test_warmup_runs_a_dense_and_a_sparse_step(monkeypatch, file, env, dense):
    """The dense flag of each forward of a warm-up generation. `num_inference_steps` counts the schedule's points,
    one more than its forwards."""
    pipeline, calls = _pipeline(monkeypatch, env), []
    pipeline.transformer_file = file and Path(file)
    pipeline._warming = True
    monkeypatch.setattr(MiniMaxH3Pipeline, "__call__", lambda self, prompt, **kwargs: calls.append(kwargs))
    pipeline._run_forced(76800, "a prompt", {"num_frames": 158})
    assert pipeline._force_bucket is None
    assert calls == [{"num_inference_steps": len(dense) + 1, "num_frames": 158}]
    scheduler = MiniMaxH3Scheduler(shift=6.0)
    scheduler.set_timesteps(len(dense) + 1)
    pipeline._sol_schedule = pipeline._requested_sol_schedule()
    assert [pipeline._is_sol_dense_step(step, num_latent_frames=1) for step in range(len(scheduler.timesteps))] == dense


def test_a_request_keeps_the_schedules_dense_steps(monkeypatch):
    pipeline = _pipeline(monkeypatch, {"H3_SOL_ATTN": "1"})
    # Sol-Attn not set up: the off switch
    assert not any(pipeline._is_sol_dense_step(step, num_latent_frames=1) for step in range(4))
    pipeline._sol_schedule = pipeline._requested_sol_schedule()
    assert [pipeline._is_sol_dense_step(step, num_latent_frames=1) for step in range(4)] == [True, True, False, False]


def test_pooled_off_reaches_the_transformer(monkeypatch):
    pipeline = _pipeline(monkeypatch, {"H3_SOL_ATTN": "1", "H3_SOL_POOLED": "0"})
    pipeline._release_denoise_traces, transformer = Mock(), Mock()
    pipeline._set_sol_attention(transformer, 76800, 73419, 843)
    assert transformer.set_sol_request.call_args.kwargs["pooled"] is False


@pytest.mark.parametrize(
    ("env", "warming", "num_frames", "dense"),
    [
        ({}, False, 158, [True] * 3 + [False] * 5),
        ({}, False, 175, [True] * 8),
        ({}, False, 243, [True] * 8),
        ({"H3_SOL_ATTN": "1"}, False, 243, [True] * 3 + [False] * 5),
        ({}, True, 243, [True] + [False] * 7),  # a rung's warm-up clip still captures the sparse trace
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


@pytest.mark.parametrize("pooled", [True, False])
def test_attention_hands_the_op_the_pooled_term_only_when_on(monkeypatch, pooled):
    """Default: the op gets the pre-pass's pooled K and V. Off: neither (its kept-blocks-only mode), and only the
    mask is there to free."""
    routed = {"block_mask": "mask", "pooled_k": "pk", "pooled_v": "pv"} if pooled else {"block_mask": "mask"}
    calls, freed = [], []
    monkeypatch.setattr(ttnn.transformer, "sparse_ring_sdpa", lambda *args, **kwargs: calls.append(args) or "out")
    monkeypatch.setattr(ttnn, "deallocate", freed.append)
    attn = MagicMock(_sol={"pooled": pooled}, sdpa_scale=0.125)
    attn.parallel_config.sequence_parallel.factor = 8
    q = k = v = SimpleNamespace(shape=(1, 14, 512, 64), dtype=ttnn.bfloat16)

    def prepass(q, k, v, *, scale, pooled):
        assert pooled is attn._sol["pooled"]
        return routed

    assert MiniMaxH3Attention._sol_attention(attn, q, k, v, prepass, "logical_n") == "out"
    assert calls == [(q, k, v, "mask", "pk" if pooled else None, "pv" if pooled else None)]
    assert freed == list(routed.values())
