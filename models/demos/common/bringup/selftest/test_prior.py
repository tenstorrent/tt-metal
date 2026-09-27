# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""A bring-up with a prior (spec ``prior``): the same checkpoint on another mesh shares the prior's hf/ and goldens,
its briefs list the prior's files, and ``new --prior`` scaffolds it."""

import json
import types

import pytest
import yaml

from models.demos.common.bringup.core.spec import Spec
from models.demos.common.bringup.selftest.test_orchestrator import impl_task, orch  # noqa: F401 (fixture)


def _prior(repo, name="priorfix", mesh=(1, 4), **over):
    d = repo / "models" / "demos" / name
    (d / "bringup").mkdir(parents=True, exist_ok=True)
    data = {
        "model": name,
        "hf_id": "none/fixture",
        "model_dir": f"models/demos/{name}",
        "num_layers": 3,
        "box": {"mesh": list(mesh)},
        "paths": {"repo": str(repo), "art": str(repo.parent / "art")},
        **over,
    }
    (d / "bringup" / "spec.yaml").write_text(yaml.safe_dump(data))
    return d


def test_the_prior_shares_its_checkpoint_and_goldens(fx, tmp_path):
    s = Spec.load(fx())
    _prior(s.repo)
    s2 = Spec.load(fx(prior="models/demos/priorfix", box={"mesh": [2, 2]}))
    p = s2.prior_spec()
    assert s2.prior == s.repo / "models/demos/priorfix"
    assert s2.hf_dir == p.hf_dir and s2.golden_root == p.golden_root and s2.golden_root != s.golden_root
    assert (
        Spec.load(fx(prior="models/demos/priorfix", paths={**s.data["paths"], "golden": "/x/g"})).golden_root.as_posix()
        == "/x/g"
    )
    assert s2.validate() == [], s2.validate()


def test_a_prior_must_be_the_same_checkpoint(fx):
    s = Spec.load(fx())
    _prior(s.repo, hf_id="other/model")
    errs = Spec.load(fx(prior="models/demos/priorfix")).validate()
    assert any("hf_id" in e for e in errs)
    assert any("no bringup/spec.yaml" in e for e in Spec.load(fx(prior="models/demos/nope")).validate())


def test_briefs_list_the_priors_files_per_role(orch, sandbox):  # noqa: F811
    d = _prior(sandbox.repo)
    (d / "bringup" / "plan.md").write_text("# plan\n")
    (d / "tt").mkdir()
    sandbox.write_spec(prior="models/demos/priorfix", box={"mesh": [2, 2]})
    o = orch([impl_task()], {})
    plan = o.brief(o.led.task("C.1"), "plan", 1).read_text()
    assert "## Prior bring-up" in plan and "mesh 1x4" in plan and "re-plan for this mesh (2x2" in plan
    assert (
        "`models/demos/priorfix/bringup/plan.md`" in plan and "priorfix/bringup/components.yaml" not in plan
    )  # only existing files
    impl = o.brief(o.led.task("C.1"), "implement", 1).read_text()
    assert "`models/demos/priorfix/tt/`" in impl and "Do not import from the prior" in impl
    sandbox.write_spec(prior=None)
    assert "## Prior bring-up" not in orch([impl_task()], {}).brief(o.led.task("C.1"), "plan", 2).read_text()


def test_new_from_prior_scaffolds_the_spec_and_hooks(tmp_path, monkeypatch):
    from models.demos.common.bringup import __main__ as cli
    from models.demos.common.bringup.core import spec as S

    monkeypatch.setattr(S, "CODE_ROOT", tmp_path)
    _prior(tmp_path, name="mdl", contract={"adapter": "mdl"}, hooks="models.demos.mdl.bringup.hooks")
    assert cli._new_from_prior(types.SimpleNamespace(prior="mdl", model=None, mesh="2,2")) == 0
    b = tmp_path / "models/demos/mdl_2x2/bringup"
    data = yaml.safe_load((b / "spec.yaml").read_text())
    assert data["prior"] == "models/demos/mdl" and data["box"]["mesh"] == [2, 2] and data["model"] == "mdl_2x2"
    assert data["hooks"] == "models.demos.mdl_2x2.bringup.hooks" and data["contract"]["adapter"] == "mdl_2x2"
    assert "from models.demos.mdl.bringup import hooks as _prior" in (b / "hooks.py").read_text()
    with pytest.raises(SystemExit, match="exists"):
        cli._new_from_prior(types.SimpleNamespace(prior="mdl", model=None, mesh="2,2"))


def test_a_reused_golden_must_match_the_rung(fx, tmp_path):
    from models.demos.common.bringup.reference import generate_golden as G

    s = Spec.load(fx())
    out = tmp_path / "g"
    out.mkdir()
    (out / "manifest.json").write_text(
        json.dumps({"seq": 999, "chunk": 64, "layers": s.layers(), "model": "none/fixture"})
    )
    with pytest.raises(SystemExit, match="does not match this rung"):
        G.reuse(s, s.rung("s256"), out)
