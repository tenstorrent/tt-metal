# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Sol-Attn schedule: which denoise steps and blocks stay dense. Host only, no device."""

from types import SimpleNamespace

import pytest

from ....models.transformers.minimax_h3.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
from ....pipelines.minimax_h3.policy import SolSchedule, sol_schedule


@pytest.mark.parametrize("env", [{}, {"H3_SOL_ATTN": "0"}, {"H3_SOL_ATTN": "0", "H3_SOL_DENSE_STEPS": "0"}])
def test_off_switch_is_dense(env):
    assert sol_schedule(env) is None


def test_defaults_keep_first_step_and_two_layers_dense():
    schedule = sol_schedule({"H3_SOL_ATTN": "1"})
    assert schedule == SolSchedule(tau=1.0, dense_steps=1, dense_layers=2, sinks=True)
    assert [schedule.is_dense_step(i) for i in range(4)] == [True, False, False, False]


def test_env_sets_the_schedule():
    env = {"H3_SOL_ATTN": "1", "H3_SOL_TAU": "0.5", "H3_SOL_DENSE_STEPS": "2", "H3_SOL_DENSE_LAYERS": "0"}
    schedule = sol_schedule(env)
    assert schedule == SolSchedule(tau=0.5, dense_steps=2, dense_layers=0)
    assert [schedule.is_dense_step(i) for i in range(4)] == [True, True, False, False]
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
