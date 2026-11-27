# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Sol-Attn schedule: which denoise steps and blocks stay dense. Host only, no device."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ....models.transformers.minimax_h3.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
from ....pipelines.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from ....pipelines.minimax_h3.policy import SolSchedule, sol_schedule

_SOL_ENV = ("H3_SOL_ATTN", "H3_SOL_TAU", "H3_SOL_DENSE_STEPS", "H3_SOL_DENSE_LAYERS", "H3_SOL_SINKS")


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


@pytest.mark.parametrize("name", ["H3_SOL_DENSE_STEPS", "H3_SOL_DENSE_LAYERS"])
def test_negative_counts_are_refused(name, expect_error):
    with expect_error(ValueError, "0 or more"):
        sol_schedule({"H3_SOL_ATTN": "1", name: "-1"})


def test_transformer_keeps_leading_layers_and_flagged_steps_dense():
    seen = []

    def attn(index):
        return SimpleNamespace(
            _sol={"dense_step": False}, set_sol_request=lambda request, tau: seen.append((index, request, tau))
        )

    model = SimpleNamespace(transformer_blocks=[SimpleNamespace(attn=attn(i)) for i in range(4)])
    request = (512, 4000, 300)
    MiniMaxH3Transformer3DModel.set_sol_request(model, request, tau=0.5, dense_layers=2)
    assert seen == [(0, None, 0.5), (1, None, 0.5), (2, request, 0.5), (3, request, 0.5)]

    for dense in (True, False):
        MiniMaxH3Transformer3DModel.set_sol_dense_step(model, dense)
        assert model._sol_dense_step is dense
        assert all(block.attn._sol["dense_step"] is dense for block in model.transformer_blocks)


def _pipeline(monkeypatch, env):
    """A pipeline without `__init__`, holding only what the Sol-Attn hookup reads, under `env`."""
    for name in _SOL_ENV:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    pipeline = MiniMaxH3Pipeline.__new__(MiniMaxH3Pipeline)
    pipeline.dynamic_length = False
    pipeline.bucket_denoise = True
    pipeline.sp_factor = 8
    pipeline._sol_key = pipeline._sol_schedule = None
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


def test_pipeline_sets_the_request_once_and_marks_buckets_cold(monkeypatch):
    pipeline, transformer = _pipeline(monkeypatch, {"H3_SOL_ATTN": "1", "H3_SOL_SINKS": "0"}), Mock()
    pipeline._set_sol_attention(transformer, 76800, 73419, 843)
    assert pipeline._sol_key == (9600, 73419, 0)
    assert pipeline._sol_schedule == SolSchedule(sinks=False)
    assert _states(pipeline) == [False, False]
    transformer.set_sol_request.assert_called_once_with((9600, 73419, 0), tau=1.0, dense_layers=2)

    pipeline._set_sol_attention(transformer, 76800, 73419, 843)
    pipeline._release_denoise_traces.assert_called_once()

    # A new schedule on the same request is a new key: the traces go and the buckets turn cold again.
    monkeypatch.setenv("H3_SOL_DENSE_STEPS", "1")
    pipeline._set_sol_attention(transformer, 76800, 73419, 843)
    assert pipeline._sol_schedule.dense_steps == 1
    assert pipeline._release_denoise_traces.call_count == 2

    # Sinks on: the prefix length enters the request.
    monkeypatch.setenv("H3_SOL_SINKS", "1")
    pipeline._set_sol_attention(transformer, 76800, 73419, 843)
    assert pipeline._sol_key == (9600, 73419, 843)


@pytest.mark.parametrize(
    "env, steps",
    [
        ({}, 2),
        ({"H3_SOL_ATTN": "1"}, 3),
        ({"H3_SOL_ATTN": "1", "H3_SOL_DENSE_STEPS": "0"}, 2),
        ({"H3_SOL_ATTN": "1", "H3_SOL_DENSE_STEPS": "1"}, 2),
        ({"H3_SOL_ATTN": "0", "H3_SOL_DENSE_STEPS": "2"}, 2),
    ],
)
def test_warmup_runs_one_step_past_the_dense_steps(monkeypatch, env, steps):
    pipeline, calls = _pipeline(monkeypatch, env), []
    monkeypatch.setattr(MiniMaxH3Pipeline, "__call__", lambda self, prompt, **kwargs: calls.append(kwargs))
    pipeline._run_forced(76800, "a prompt", {"num_frames": 158})
    assert calls == [{"num_inference_steps": steps, "num_frames": 158}]
    assert pipeline._force_bucket is None
