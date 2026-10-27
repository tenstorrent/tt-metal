# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F200: a runner test that tt-d-gen's engine drives does not gate, and starts no agent, while serving.require_engine
holds and no build is found; independent tasks go on, and the task runs once the build appears (DeepSeek-V4.1 8x4
K.1, 2026-10-27: the engine was built alongside the run)."""

from models.demos.common.bringup import orchestrator as O
from models.demos.common.bringup.orchestrator import DONE, HUMAN
from models.demos.common.bringup.selftest.conftest import record_cmd
from models.demos.common.bringup.selftest.test_orchestrator import orch  # noqa: F401 (fixture)
from models.demos.common.bringup.testing import dgen_engine


def _tasks(sandbox):
    (sandbox.repo / "tests/test_runner_smoke.py").write_text("from runner_case import use_engine\n")
    runner = {
        "id": "K.1",
        "title": "runner",
        "gate": {"cmd": record_cmd(ok=1) + " && test -f tests/test_runner_smoke.py", "metrics": {"ok": "== 1"}},
    }
    other = {"id": "X.1", "title": "other", "gate": {"cmd": record_cmd(n=1), "metrics": {"n": "== 1"}}}
    return [runner, other]


def test_a_runner_test_waits_for_its_engine_and_independent_work_goes_on(
    orch, sandbox, monkeypatch, tmp_path
):  # noqa: F811
    monkeypatch.setenv("BRINGUP_SERVER_REPO", str(tmp_path / "no-dgen"))
    monkeypatch.delenv("BRINGUP_DGEN", raising=False)
    o = orch(_tasks(sandbox), {})
    assert o.run() == HUMAN
    st = o.led.state()
    assert st["X.1"]["status"] == "PASS"
    assert st["K.1"]["waiting"].startswith(O.ENGINE_WAIT) and "no tt-d-gen checkout" in st["K.1"]["waiting"]
    assert st["K.1"].get("status") not in ("PASS", "FAIL", "STOPPED") and not orch.calls()  # no gate, no agent

    monkeypatch.setattr(dgen_engine, "find_build", lambda spec=None: (object(), ""))
    assert o.run() == DONE
    assert o.led.state()["K.1"]["status"] == "PASS" and not orch.calls()


def test_no_wait_when_the_engine_is_not_required_or_the_gate_runs_no_runner_test(
    orch, sandbox, monkeypatch, tmp_path
):  # noqa: F811
    monkeypatch.setenv("BRINGUP_SERVER_REPO", str(tmp_path / "no-dgen"))
    o = orch(_tasks(sandbox), {})
    assert O.engine_tests(sandbox.repo, o.led.task("X.1")["gate"]["cmd"]) == []
    assert O.engine_missing(o.spec, o.led.task("X.1")["gate"]["cmd"]) is None
    monkeypatch.setenv("BRINGUP_DGEN", "0")  # the producer stands in (spec serving.require_engine: false likewise)
    assert O.engine_missing(o.spec, o.led.task("K.1")["gate"]["cmd"]) is None
    assert o.run() == DONE


def test_the_specs_server_repo_expands_like_its_other_paths(monkeypatch, tmp_path):
    """The 8x4 spec's `/scratch/$USER/tt-d-gen` was taken literally: a built engine was never found."""

    class _Spec:
        def get(self, key):
            return {"serving.server_repo": "$BRINGUP_SELFTEST_DIR/tt-d-gen"}.get(key)

    monkeypatch.delenv("BRINGUP_SERVER_REPO", raising=False)
    monkeypatch.setenv("BRINGUP_SELFTEST_DIR", str(tmp_path))
    assert dgen_engine.server_repo(_Spec()) == tmp_path / "tt-d-gen"
    assert "$" not in str(dgen_engine.server_repo(None))  # the defaults.yaml template path


def test_the_engine_driver_names_its_pipeline_arguments():
    """tt-d-gen 1cf343887 dropped device_prefill_pipeline's layers_per_chunk: positional arguments shifted the timeout
    into sp_factor. The driver passes keywords, layers_per_chunk only to a build that declares it."""
    from models.demos.common.bringup.testing.dgen_prefill_driver import device_pipeline_args

    new = "device_prefill_pipeline(service_id: str, ack_shm_name: str, chunk_size: int, connect_timeout_ms: int = 30000, sp_factor: int = 1, num_mtp_tokens: int = 0) -> PipelineConfig"
    old = "device_prefill_pipeline(service_id: str, ack_shm_name: str, chunk_size: int, layers_per_chunk: int, connect_timeout_ms: int = 30000, sp_factor: int = 1, num_mtp_tokens: int = 0) -> PipelineConfig"
    want = dict(service_id="s", ack_shm_name="a", chunk_size=5120, connect_timeout_ms=1800000, sp_factor=8)
    assert device_pipeline_args(new + "\n\nOn-device", "s", "a", 5120, 40, 1800000, 8) == want
    assert device_pipeline_args(old, "s", "a", 5120, 40, 1800000, 8) == {**want, "layers_per_chunk": 40}
