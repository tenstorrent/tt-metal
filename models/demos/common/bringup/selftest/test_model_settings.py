# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""A model's switches in one place: the Settings helper, and the lint that keeps environment reads out of the model's
code and out of the forks (diagnostics aside)."""

from models.demos.common.bringup.core.model_settings import Setting, Settings
from models.demos.common.bringup.selftest.test_device_side import gspec, noise  # noqa: F401 (fixtures)
from models.demos.common.bringup.testing.settings_lint import violations

MD = "models/demos/toy"


def test_settings_default_override_and_choices(monkeypatch):
    s = Settings("TOY_", {"KV": Setting("bfp8", ("bfp8", "bf16"), "owner"), "Q": Setting(64)})
    monkeypatch.delenv("TOY_KV", raising=False)
    assert s.get("KV") == "bfp8" and s.all() == {"KV": "bfp8", "Q": 64}
    monkeypatch.setenv("TOY_KV", "bf16")
    monkeypatch.setenv("TOY_Q", "32")
    assert s.get("KV") == "bf16" and s.get("Q") == 32
    monkeypatch.setenv("TOY_KV", "fp16")
    try:
        s.get("KV")
    except ValueError as e:
        assert "TOY_KV" in str(e)
    else:
        raise AssertionError("an unknown value was accepted")


def test_a_test_value_applies_only_under_pytest_and_the_environment_still_wins(monkeypatch):
    """F222: DeepSeek-V4.1 ENGRAM_TABLE: the served default builds the engram tables in RAM at start (~1 min a layer);
    tests read them from disk instead (rows bit-identical). A setting's ``test`` value is what a pytest process gets;
    a process without pytest gets the default; the environment variable overrides both."""
    import sys

    S = Settings("XT_", {"TABLE": Setting("resident", ("resident", "random"), "why", test="random"), "N": Setting(8)})
    monkeypatch.delenv("XT_TABLE", raising=False)
    assert S.get("TABLE") == "random" and S.get("N") == 8  # this process runs under pytest
    monkeypatch.delitem(sys.modules, "pytest")
    assert S.get("TABLE") == "resident"
    monkeypatch.setenv("XT_TABLE", "resident")
    import pytest as _  # noqa: F401  (back under pytest)

    assert S.get("TABLE") == "resident"


def test_lint_flags_env_reads_outside_settings(tmp_path):
    def put(rel, text):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        return rel

    files = [
        put(f"{MD}/tt/settings.py", "import os\nx = os.environ.get('TOY_KV')\n"),
        put(f"{MD}/tt/attention.py", "import os\nq = int(os.environ.get('TOY_Q', '64'))\n"),
        put(f"{MD}/bringup/hooks.py", "import os\nif os.getenv('TOY_X'):\n    pass\n"),
        put(
            "ttnn/ttnn/bringup/op/op.py",
            "import os\nn = os.environ.get('OP_ITERS')\nt = os.environ.get('OP_TRACE')  # diagnostic\n",
        ),
        put("ttnn/ttnn/bringup/op/tests/test_op.py", "import os\nos.environ.get('X')\n"),
        put("models/demos/other/tt/a.py", "import os\nos.environ.get('Y')\n"),
    ]
    got = violations(tmp_path, MD, files)
    assert any("tt/attention.py:2" in v for v in got) and any("hooks.py:2" in v for v in got)
    assert any("op/op.py:2" in v for v in got) and not any("op/op.py:3" in v for v in got)
    assert not any("settings.py" in v.split(":")[0] or "tests/" in v or "other/" in v for v in got)
    assert len(got) == 3


def test_there_is_no_max_precision_setting():
    """F137 (owner): every gated device test runs the shipped defaults; no setting table names another value."""
    import ast
    from pathlib import Path

    import pytest

    with pytest.raises(TypeError):
        Settings("TOY_", {"KV": Setting("bfp8")}, max_precision={"KV": "bf16"})
    root = Path(__file__).resolve().parents[1]
    for f in ("testing/component.py", "orchestrator.py", "testing/model_precision.py", "core/model_settings.py"):
        names = {getattr(n, "attr", getattr(n, "id", None)) for n in ast.walk(ast.parse((root / f).read_text()))}
        assert not names & {"max_precision", "max_precision_env", "gate_tests_side", "switches_pinned"}, f
    from models.demos.common.bringup.core import defaults

    assert defaults.get("tests.component_max_precision", None) is None


def test_the_lint_refuses_an_emulation_default_and_an_undeclared_emulation():
    from models.demos.common.bringup.testing.settings_lint import emulation_violations

    ok = Settings("TOY_", {"Q": Setting("skip", ("skip", "qdq"), "owner", emulation=("qdq",)), "F": Setting("hifi2")})
    assert emulation_violations([ok]) == []
    bad = Settings(
        "TOY_", {"Q": Setting("qdq", ("qdq", "skip"), emulation=("qdq",)), "X": Setting("bf16", ("bf16", "round_tf32"))}
    )
    got = emulation_violations([bad])
    assert any("TOY_Q" in v and "default" in v for v in got) and any("TOY_X" in v and "round_tf32" in v for v in got)


def test_lint_flags_only_the_env_reads_a_change_adds_to_a_fork(tmp_path):
    """F106: an imported fork's own knobs (committed) are not the editing agent's violations; a knob it adds is; model
    code stays strict even when the read was committed (DeepSeek-V4.1 P.3: 173 flags on flat_routed_expert's MIMO_FL_*
    reads after an edit, attempt 2 started to convert them)."""
    import subprocess

    def git(*a):
        subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True)

    fork, model = "ttnn/ttnn/bringup/op/op.py", f"{MD}/tt/attention.py"
    for rel, text in ((fork, "import os\nn = os.environ.get('OP_ITERS')\n"), (model, "import os\nos.getenv('Q')\n")):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)
    git("init", "-q")
    git("-c", "user.email=t@t", "-c", "user.name=t", "add", ".")
    git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "import")
    (tmp_path / fork).write_text("import os\nn = os.environ.get('OP_ITERS')\nm = os.environ.get('OP_NEW')\n")
    got = violations(tmp_path, MD, [fork, model])
    assert sorted(v.split(": ")[0] for v in got) == sorted([f"{model}:2", f"{fork}:3"])
    assert len(violations(tmp_path, MD, [fork], base=None)) == 2  # no base: every read counts (the audit's view)


def test_a_failing_step_is_diagnosed_on_the_cpu_never_by_a_device_rerun(gspec, noise, monkeypatch):  # noqa: F811
    """F137: logic or precision is told by the step's CPU reference at the device's formats on the same inputs,
    recorded, never gated: a wrong module is 'logic'; one that is the device-format step itself is 'precision'."""
    import torch

    from models.demos.common.bringup.reference.interface import Ctx
    from models.demos.common.bringup.selftest.conftest import got
    from models.demos.common.bringup.testing import devfmt
    from models.demos.common.bringup.testing.component import run_component_test

    s = gspec()
    noise(0.5)
    assert not run_component_test(s, "mlp", 0, checks="auto")
    m = got()
    assert m["err_mlp_L00_devfmt_precision"] == 0 and m["err_mlp_L00_devfmt_dev"] > m["err_mlp_L00_devfmt_cpu"]

    class Case:
        inputs = [torch.randn(64, 32)]

        def rctx(self):
            return Ctx(0, 0, 64, None, {})

    class Ref:
        def component(self, layer, step):
            return lambda ctx, x: x * 3.0

    cpu = Case.inputs[0] * 3.0
    dev = devfmt.step_fn(s, Ref(), 0, "x")[0](Case().rctx(), Case.inputs[0])  # the device-format step's own output
    assert devfmt.diagnose(s, Ref(), 0, "x", Case(), cpu, dev, "t") == "precision"


def test_lint_counts_an_imported_branchs_knobs_as_the_forks_own(tmp_path):
    """F231 (DeepSeek-V4.1 P.17): a task taking another branch as is (tasks.yaml imports) brought new environment
    knobs in a fork; they are the branch's, not the agent's. A knob in neither HEAD nor the import is still flagged."""
    import subprocess

    def git(*a):
        subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True)

    rel = "ttnn/ttnn/bringup/op/op.py"
    f = tmp_path / rel
    f.parent.mkdir(parents=True)
    git("init", "-q", "-b", "main")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    f.write_text("import os\n")
    git("add", ".")
    git("commit", "-q", "-m", "base")
    git("checkout", "-q", "-b", "drop")
    f.write_text("import os\nk = os.environ.get('DROP_KNOB')\n")
    git("commit", "-q", "-am", "drop")
    git("checkout", "-q", "main")
    f.write_text("import os\nk = os.environ.get('DROP_KNOB')\nm = os.environ.get('AGENT_KNOB')\n")
    assert len(violations(tmp_path, MD, [rel])) == 2  # without the import both count
    got = violations(tmp_path, MD, [rel], imports=["drop"])
    assert len(got) == 1 and "op.py:3" in got[0], got
