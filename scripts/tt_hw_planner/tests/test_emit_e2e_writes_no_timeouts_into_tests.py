# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""emit-e2e never gives a bring-up agent a reason to type a timeout into a model's test.

Its gates ran the model tests as a bare `pytest`, so the repo's pytest.ini (`timeout = 300`) applied,
and a Galaxy model cannot even load in 5 minutes. The bring-up agent for Qwen-Image-Edit
(2026-09-25) wrote `@pytest.mark.timeout(2 * 3600)` into its tests to survive; that marker beats
any command-line value, rode into the perf test, and killed a healthy per-op profile at 7,204 s
(2026-09-28). The gates now switch pytest's timeout off, and the prompt says not to add one.
"""

import inspect
from pathlib import Path

from models.experimental.perf_automation.agent import probes
from scripts.tt_hw_planner._cli_helpers import e2e_synthesizer
from scripts.tt_hw_planner.commands import emit_e2e as E

_FLAGS = " ".join(probes.PYTEST_NO_TIMEOUT)


def test_the_agent_is_told_how_to_run_and_not_to_add_a_timeout():
    prompt = E._build_agent_prompt(model_id="org/some-model", demo_dir=Path("/tmp/x"), pcc=0.99)
    assert f"-m pytest {_FLAGS} <file>" in prompt
    assert "Do NOT add `@pytest.mark.timeout`" in prompt


def test_the_gate_runs_with_pytests_timeout_off():
    src = inspect.getsource(E._run_deterministic_gates)
    assert "*_pr.PYTEST_NO_TIMEOUT" in src


def test_the_synthesizers_demo_runs_have_it_off_too():
    for fn in (e2e_synthesizer._make_default_pytest_runner, e2e_synthesizer._default_pytest_runner):
        assert "*PYTEST_NO_TIMEOUT" in inspect.getsource(fn), fn.__name__


def test_the_emitted_tests_usage_line_suggests_no_timeout():
    from scripts.tt_hw_planner import e2e_emitter

    src = inspect.getsource(e2e_emitter)
    assert "--timeout=" not in src
