# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Sol-Attn schedule (which denoise steps and blocks stay dense) and the per-request route constants. Host only,
no device."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from ....models.transformers.minimax_h3 import sol_prepass
from ....models.transformers.minimax_h3.sol_prepass import (
    SolGeometry,
    SolRouteConstants,
    build_route_constants,
    route_factors,
)
from ....models.transformers.minimax_h3.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
from ....pipelines.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from ....pipelines.minimax_h3.pipeline_minimax_h3_turbo import MiniMaxH3TurboPipeline
from ....pipelines.minimax_h3.policy import SolSchedule, sol_schedule
from ....pipelines.minimax_h3.scheduler import MiniMaxH3Scheduler

_SOL_ENV = ("H3_SOL_ATTN", "H3_SOL_PRESET", "H3_SOL_TAU", "H3_SOL_DENSE_STEPS", "H3_SOL_DENSE_LAYERS", "H3_SOL_SINKS")


@pytest.mark.parametrize("env", [{}, {"H3_SOL_ATTN": "0"}, {"H3_SOL_ATTN": "0", "H3_SOL_DENSE_STEPS": "0"}])
def test_off_switch_is_dense(env):
    assert sol_schedule(env) is None


def test_defaults_keep_two_steps_and_two_layers_dense():
    schedule = sol_schedule({"H3_SOL_ATTN": "1"})
    assert schedule == SolSchedule(tau=1.0, dense_steps=2, dense_layers=2, sinks=True)
    assert [schedule.is_dense_step(i) for i in range(4)] == [True, True, False, False]


def test_env_sets_the_schedule():
    env = {"H3_SOL_ATTN": "1", "H3_SOL_TAU": "0.5", "H3_SOL_DENSE_STEPS": "1", "H3_SOL_DENSE_LAYERS": "0"}
    schedule = sol_schedule(env)
    assert schedule == SolSchedule(tau=0.5, dense_steps=1, dense_layers=0)
    assert [schedule.is_dense_step(i) for i in range(4)] == [True, False, False, False]
    assert not any(SolSchedule(dense_steps=0).is_dense_step(i) for i in range(4))


def test_sinks_switch_drops_the_dense_prefix():
    assert not sol_schedule({"H3_SOL_ATTN": "1", "H3_SOL_SINKS": "0"}).sinks
    assert sol_schedule({"H3_SOL_ATTN": "1", "H3_SOL_SINKS": "1"}).sinks


_CARD13 = SolSchedule(tau=1.13, dense_steps=2, dense_layers=0, sinks=True)  # the 10Eros-Max arm of the Sol sheet
_HF = SolSchedule(tau=1.0, dense_steps=2, dense_layers=2, sinks=False)  # its HyperFlow arm


@pytest.mark.parametrize(
    ("env", "preset", "expected"),
    [
        ({"H3_SOL_PRESET": "10eros-max"}, None, _CARD13),
        ({}, "hyperflow", _HF),
        ({"H3_SOL_PRESET": "hyperflow"}, None, _HF),
        ({"H3_SOL_PRESET": "10eros-max"}, "hyperflow", _CARD13),
        ({}, None, SolSchedule()),
        (
            {"H3_SOL_PRESET": "10eros-max", "H3_SOL_TAU": "1.97", "H3_SOL_SINKS": "0"},
            None,
            SolSchedule(1.97, 2, 0, False),
        ),
        ({"H3_SOL_DENSE_LAYERS": "0", "H3_SOL_SINKS": "1"}, "hyperflow", replace(_HF, dense_layers=0, sinks=True)),
    ],
)
def test_model_preset_sets_the_schedule_behind_the_switch(env, preset, expected):
    assert sol_schedule({"H3_SOL_ATTN": "1", **env}, preset) == expected
    assert sol_schedule(env, preset) is None
    assert sol_schedule({"H3_SOL_ATTN": "0", **env}, preset) is None


def test_unknown_preset_is_refused(expect_error):
    with expect_error(ValueError, "H3_SOL_PRESET must be one of"):
        sol_schedule({"H3_SOL_ATTN": "1", "H3_SOL_PRESET": "10eros"})


@pytest.mark.parametrize(("hyperflow", "expected"), [(object(), _HF), (None, SolSchedule())])
def test_turbo_pipeline_names_the_hyperflow_preset(monkeypatch, hyperflow, expected):
    pipeline = _pipeline(monkeypatch, {"H3_SOL_ATTN": "1"}, MiniMaxH3TurboPipeline)
    pipeline.hyperflow = hyperflow
    assert pipeline._requested_sol_schedule() == expected


@pytest.mark.parametrize("name", ["H3_SOL_DENSE_STEPS", "H3_SOL_DENSE_LAYERS"])
def test_negative_counts_are_refused(name, expect_error):
    with expect_error(ValueError, "0 or more"):
        sol_schedule({"H3_SOL_ATTN": "1", name: "-1"})


def test_transformer_keeps_leading_layers_and_flagged_steps_dense():
    seen = []

    def attn(index):
        return SimpleNamespace(
            _sol={"dense_step": False},
            set_sol_request=lambda request, tau, constants: seen.append((index, request, tau, constants)),
        )

    model = SimpleNamespace(
        transformer_blocks=[SimpleNamespace(attn=attn(i)) for i in range(4)],
        _sol_route_constants=lambda request: ("shared", request),
    )
    request = (512, 4000, 300)
    MiniMaxH3Transformer3DModel.set_sol_request(model, request, tau=0.5, dense_layers=2)
    shared = ("shared", request)  # one set of route constants for every block
    assert seen == [
        (0, None, 0.5, shared),
        (1, None, 0.5, shared),
        (2, request, 0.5, shared),
        (3, request, 0.5, shared),
    ]

    for dense in (True, False):
        MiniMaxH3Transformer3DModel.set_sol_dense_step(model, dense)
        assert model._sol_dense_step is dense
        assert all(block.attn._sol["dense_step"] is dense for block in model.transformer_blocks)


def _pipeline(monkeypatch, env, cls=MiniMaxH3Pipeline):
    """A pipeline without `__init__`, holding only what the Sol-Attn hookup reads, under `env`."""
    for name in _SOL_ENV:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    pipeline = cls.__new__(cls)
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


def test_pipeline_releases_traces_only_on_a_schedule_change(monkeypatch):
    pipeline, transformer = _pipeline(monkeypatch, {"H3_SOL_ATTN": "1", "H3_SOL_SINKS": "0"}), Mock()
    pipeline._set_sol_attention(transformer, 76800, 73419, 843)
    assert pipeline._sol_key == (9600, 73419, 0)
    assert pipeline._sol_schedule == SolSchedule(sinks=False)
    assert _states(pipeline) == [False, False]
    transformer.set_sol_request.assert_called_once_with((9600, 73419, 0), tau=1.0, dense_layers=2)
    pipeline._release_denoise_traces.assert_called_once()

    # Another prompt length, and another rung: the request moves, the traces and the warm buckets stay.
    for state in (*pipeline._buckets.values(), pipeline._request_state):
        state.warm = True
    for rung, logical_n in ((76800, 71002), (115200, 110219)):
        pipeline._set_sol_attention(transformer, rung, logical_n, 843)
        assert pipeline._sol_key == (rung // 8, logical_n, 0)
        transformer.set_sol_request.assert_called_with((rung // 8, logical_n, 0), tau=1.0, dense_layers=2)
    assert _states(pipeline) == [True, True]
    pipeline._release_denoise_traces.assert_called_once()

    # A new schedule: the traces go and the buckets turn cold again.
    monkeypatch.setenv("H3_SOL_DENSE_LAYERS", "3")
    pipeline._set_sol_attention(transformer, 76800, 73419, 843)
    assert pipeline._sol_schedule.dense_layers == 3
    assert _states(pipeline) == [False, False]
    assert pipeline._release_denoise_traces.call_count == 2

    # Sinks on: the prefix length enters the request.
    monkeypatch.setenv("H3_SOL_SINKS", "1")
    pipeline._set_sol_attention(transformer, 76800, 73419, 843)
    assert pipeline._sol_key == (9600, 73419, 843)

    # Off again: dense, and the traces captured with Sol go.
    monkeypatch.setenv("H3_SOL_ATTN", "0")
    pipeline._set_sol_attention(transformer, 76800, 73419, 843)
    assert (pipeline._sol_key, pipeline._sol_schedule) == (None, None)
    transformer.set_sol_request.assert_called_with(None)
    assert pipeline._release_denoise_traces.call_count == 4


@pytest.mark.parametrize(
    "env, dense",
    [
        ({}, [False]),
        ({"H3_SOL_ATTN": "0", "H3_SOL_DENSE_STEPS": "2"}, [False]),
        ({"H3_SOL_ATTN": "1"}, [True, False]),
        ({"H3_SOL_ATTN": "1", "H3_SOL_DENSE_STEPS": "1"}, [True, False]),
        ({"H3_SOL_ATTN": "1", "H3_SOL_DENSE_STEPS": "0"}, [False, False]),
    ],
)
def test_warmup_runs_a_dense_and_a_sparse_step(monkeypatch, env, dense):
    """The dense flag of each forward of a warm-up generation. `num_inference_steps` counts the schedule's points,
    one more than its forwards."""
    pipeline, calls = _pipeline(monkeypatch, env), []
    pipeline._warming = True
    monkeypatch.setattr(MiniMaxH3Pipeline, "__call__", lambda self, prompt, **kwargs: calls.append(kwargs))
    pipeline._run_forced(76800, "a prompt", {"num_frames": 158})
    assert pipeline._force_bucket is None
    assert calls == [{"num_inference_steps": len(dense) + 1, "num_frames": 158}]
    scheduler = MiniMaxH3Scheduler(shift=6.0)
    scheduler.set_timesteps(len(dense) + 1)
    pipeline._sol_schedule = pipeline._requested_sol_schedule()
    assert [pipeline._is_sol_dense_step(step) for step in range(len(scheduler.timesteps))] == dense


def test_a_request_keeps_the_schedules_dense_steps(monkeypatch):
    pipeline = _pipeline(monkeypatch, {"H3_SOL_ATTN": "1"})
    assert not any(pipeline._is_sol_dense_step(step) for step in range(4))  # Sol-Attn not set up: the off switch
    pipeline._sol_schedule = pipeline._requested_sol_schedule()
    assert [pipeline._is_sol_dense_step(step) for step in range(4)] == [True, True, False, False]


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


def _host_constants(monkeypatch, geo, *, trace_captured=False):
    """SolRouteConstants with host tensors standing in for the device ones, and the names it uploads."""
    monkeypatch.setattr(sol_prepass, "live_tracers", lambda: [SimpleNamespace(trace_captured=trace_captured)])
    constants, uploads = SolRouteConstants(None, 1, geo), []

    def upload(name, t, **_):
        uploads.append(name)
        if name in constants.tensors:
            constants.tensors[name].copy_(t)
        else:
            constants.tensors[name] = t.clone()

    def compose(op):
        return lambda a, b, output_tensor: output_tensor.copy_(op(a, b))

    constants._upload = upload
    monkeypatch.setattr(sol_prepass.ttnn, "add", compose(torch.add))
    monkeypatch.setattr(sol_prepass.ttnn, "multiply", compose(torch.mul))
    return constants, uploads


def _assert_set_to(constants, geo, sink_tokens):
    t = constants.tensors
    masks, n_valid_blocks = _reference_masks(geo, sink_tokens)
    for hit, expected in zip((0.0, 1.0), masks):
        assert torch.equal(_mask(hit, t["const_or"], t["valid_blocks"]), expected)
    valid_tokens = [max(0, min(geo.rows, geo.logical_n - s * geo.rows)) for s in range(geo.shards)]
    assert t["geo"][:, 0, 0, 0].tolist() == valid_tokens
    assert set(t["geo"][:, 0, 0, 1].tolist()) == {sol_prepass._f32_bits(1.0 / n_valid_blocks)}


def test_route_constants_move_in_place_to_a_new_length_and_sink(monkeypatch, expect_error):
    def geo(logical_n):
        return SolGeometry(shards=8, rows=6592, logical_n=logical_n)

    constants, uploads = _host_constants(monkeypatch, geo(51983))
    constants.set(geo(51983), 559)
    assert {"base", "q_full", "ones", "shsel", "const_or", "valid_blocks", "geo"} <= set(uploads)
    _assert_set_to(constants, geo(51983), 559)
    buffers = {name: id(t) for name, t in constants.tensors.items()}

    del uploads[:]
    constants.set(geo(51983), 559)
    assert uploads == []  # the same request costs nothing

    constants.set(geo(40011), 1300)
    assert uploads == ["geo", "full_cols", "sink_cols", "prefix_rows"]
    _assert_set_to(constants, geo(40011), 1300)

    del uploads[:]
    constants.set(geo(40010), 1300)  # the same full blocks: only the geometry page changes
    assert uploads == ["geo"]
    _assert_set_to(constants, geo(40010), 1300)
    assert {name: id(t) for name, t in constants.tensors.items()} == buffers  # no buffer was replaced

    with expect_error(AssertionError, "rows=9600"):  # another padded geometry needs its own set
        constants.set(SolGeometry(shards=8, rows=9600, logical_n=73419), 559)


def test_route_constants_allocated_after_a_trace_are_rewritten_at_every_set(monkeypatch):
    geo = SolGeometry(shards=8, rows=6592, logical_n=51983)
    constants, uploads = _host_constants(monkeypatch, geo, trace_captured=True)
    constants.set(geo, 559)
    for tensor in constants.tensors.values():
        tensor.fill_(7)  # a trace replay overwrote them
    del uploads[:]
    constants.set(geo, 559)
    assert {"base", "q_full", "ones", "shsel", "geo", "full_cols", "sink_cols", "prefix_rows"} == set(uploads)
    _assert_set_to(constants, geo, 559)
