# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""A model's switches in one place: the Settings helper, and the lint that keeps environment reads out of the model's
code and out of the forks (diagnostics aside)."""

import os

from models.demos.common.bringup.core.model_settings import Setting, Settings
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


def test_component_tests_get_the_max_precision_overrides(monkeypatch, tmp_path):
    import sys
    import types

    from models.demos.common.bringup.testing import model_precision

    s = Settings(
        "TOY_",
        {"KV": Setting("bfp8"), "FID": Setting("hifi2"), "Q": Setting(64)},
        max_precision={"KV": "bf16", "FID": "hifi4"},
    )
    assert s.max_precision_env() == {"TOY_KV": "bf16", "TOY_FID": "hifi4"}
    mod = types.ModuleType("models.demos.toy.tt.settings")
    mod.settings = s
    monkeypatch.setitem(sys.modules, "models.demos.toy.tt.settings", mod)

    class Spec:
        repo = tmp_path
        model_dir = tmp_path / "models/demos/toy"

        def get(self, key, default=None):
            return {"tests.component_max_precision": True}.get(key, default)

    monkeypatch.delenv("TOY_KV", raising=False)
    monkeypatch.setenv("TOY_FID", "hifi2")  # set by the caller: kept
    assert model_precision.apply(Spec()) == {"TOY_KV": "bf16"}
    assert s.get("KV") == "bf16" and s.get("FID") == "hifi2"
    os.environ.pop("TOY_KV", None)  # apply() sets the environment directly
