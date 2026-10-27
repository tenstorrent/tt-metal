# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""A run grown from a seed (F183-F186): its own ledger, the seeded task graph, imported goldens."""

import json
import shutil
import subprocess

import pytest
import yaml

from models.demos.common.bringup.core.ledger import Ledger
from models.demos.common.bringup.core.spec import Spec
from models.demos.common.bringup.plan.ledger_gen import generate
from models.demos.common.bringup.reference import generate_golden, golden_import, prompt
from models.demos.common.bringup.reference.drift import drift_path
from models.demos.common.bringup.reference.golden import INDEX, rung_dir, verify_copy
from models.demos.common.bringup.selftest.conftest import got

BOXES = {
    "big": {"mesh": [8, 4], "chip_dram_gb": 32},
    "small": {"mesh": [2, 2], "chip_dram_gb": 32},
}
SEED_LAYERS = [0, 2]


def seed_and_grown(fx, **over):
    """The seed (layers 0 and 2 on the small box, the model's own ledger) and the full run grown from it."""
    seed = Spec.load(fx(layers=SEED_LAYERS))
    b = seed.repo / "models/demos/fixture/bringup"
    b.mkdir(parents=True, exist_ok=True)
    shutil.copy(seed.path, b / "spec.yaml")
    seed = Spec.load(b / "spec.yaml")
    data = dict(seed.raw)
    data.pop("box")
    data.update(
        layers="all",
        deploy={"box": "big", "seq": 4096, "chunk": 64},
        boxes=BOXES,
        run={"box": "big", "mode": "full", "seed": "models/demos/fixture/bringup/spec.yaml"},
        ladder=[
            {"name": "s256", "seq": 256, "chunk": 64, "full_dumps": True},
            {"name": "seed", "seq": 256, "chunk": 64, "golden": "s256", "layers": SEED_LAYERS},
            {"name": "s512", "seq": 512, "chunk": 64},
            {"name": "last", "seq": 512, "chunk": 64, "golden": "s512", "prefix_from_golden": True},
        ],
    )
    data.update(over)
    p = b / "ledgers" / "big-full" / "spec.yaml"
    p.parent.mkdir(parents=True)
    p.write_text(yaml.safe_dump(data))
    return seed, Spec.load(p)


def test_a_seeded_run_has_its_own_ledger_and_run_dirs(fx):
    seed, grown = seed_and_grown(fx)
    assert grown.validate() == [], grown.validate()
    assert seed.ledger_dir == seed.bringup_dir and grown.bringup_dir == seed.bringup_dir
    assert grown.ledger_dir == seed.bringup_dir / "ledgers" / "big-full"
    assert grown.run_dir("run1") == seed.art / "runs" / "big-full" / "run1" != seed.run_dir("run1")
    Ledger(seed.ledger_dir).write_tasks({"tasks": [{"id": "R.1", "title": "x", "step": "intake", "deps": []}]})
    Ledger(grown.ledger_dir).write_tasks({"tasks": []})
    Ledger(grown.ledger_dir).update("CK.1", status="PASS")
    assert list(Ledger(seed.ledger_dir).tasks()) == ["R.1"] and Ledger(seed.ledger_dir).status("CK.1") == "TODO"


@pytest.mark.parametrize(
    "over, needle",
    [
        ({"hf_id": "other/model"}, "a seed is the same model"),
        ({"ladder": [{"name": "s256", "seq": 256, "chunk": 64}]}, "has a ladder rung 'seed'"),
        (
            {
                "ladder": [
                    {"name": "s256", "seq": 256, "chunk": 64},
                    {"name": "seed", "seq": 256, "chunk": 64, "golden": "s256", "layers": [0, 1]},
                ]
            },
            "must be the seed's",
        ),
        ({"prior": "models/demos/fixture"}, "exclude each other"),
    ],
)
def test_a_seed_is_the_same_model_and_its_layers_are_the_port_rung(fx, over, needle):
    _, grown = seed_and_grown(fx, **over)
    assert any(needle in e for e in grown.validate()), grown.validate()


def test_a_seeded_spec_lives_in_its_ledger(fx):
    _, grown = seed_and_grown(fx)
    other = grown.repo / "elsewhere.yaml"
    shutil.copy(grown.path, other)
    assert any("lives in its ledger" in e for e in Spec.load(other).validate())


def test_the_seeded_graph_carries_the_seed_and_ports_its_layers(fx):
    seed, grown = seed_and_grown(fx)
    led = Ledger(seed.ledger_dir)
    carried = ["R.1", "R.2", "R.3", "SC.1", "PL.0", "PL.1"]
    led.write_tasks({"tasks": [{"id": t, "title": t, "step": "x", "deps": []} for t in carried + ["C.blk.attention"]]})
    for t in carried[:-1]:
        led.update(t, status="PASS")
    git = lambda *a: subprocess.run(
        ["git", "-c", "user.email=t@e", "-c", "user.name=t", *a], cwd=grown.repo, check=True
    )  # noqa: E731
    git("add", "-f", str(led.state_path))
    git("commit", "-q", "-m", "seed")
    out = generate(grown)
    ids = [t["id"] for t in out["tasks"]]
    assert ids[:6] == ["CK.1", "G.s256", "G.s512", "B.1", "W.1", "M.1"]
    assert not [i for i in ids if i.split(".")[0] in ("R", "SC", "PL", "C", "S", "SG")]
    assert "L.seed" not in ids and ids[-4:] == ["X.3", "O.1", "Z.1", "F.1"]
    t = {x["id"]: x for x in out["tasks"]}
    assert t["W.1"]["role"] == "port" and t["W.1"]["deps"] == ["B.1", "CK.1", "G.s256"]
    assert "BRINGUP_RUNG=seed " in t["W.1"]["gate"]["cmd"] and t["W.1"]["gate"]["metrics"]["err_layer_L*"] == "<= limit"
    assert t["M.1"]["deps"] == ["W.1"] and t["M.1"]["role"] == "port"
    assert "err_final_hidden" in t["M.1"]["gate"]["metrics"] and "BRINGUP_RUNG=s256 " in t["M.1"]["gate"]["cmd"]
    assert "golden_import --rung s256" in t["G.s256"]["gate"]["cmd"] and "G.last" not in t
    assert t["K.1"]["deps"][0] == "L.s256" and "check_hf_sanity" not in t["CK.1"]["gate"]["cmd"]
    assert out["seed"]["carried"] == carried[:-1] and out["seed"]["not_passed"] == ["PL.1"]
    assert len(out["seed"]["commit"]) == 40
    Ledger(grown.ledger_dir).write_tasks(out)
    assert Ledger(grown.ledger_dir).validate() == []


def test_approvals_are_the_runs_except_the_plan(fx):
    from models.demos.common.bringup.plan import approvals

    seed, grown = seed_and_grown(fx)
    (seed.bringup_dir / "plan.yaml").write_text("x: 1\n")
    assert approvals._files(grown, "intake") == [str(grown.path.relative_to(grown.repo))]
    assert approvals._files(grown, "plan") == [str((seed.bringup_dir / "plan.yaml").relative_to(grown.repo))]
    approvals.approve(grown, "intake")
    assert (grown.ledger_dir / "approvals.yaml").exists() and not (seed.bringup_dir / "approvals.yaml").exists()


def test_relative_artifact_paths_are_under_the_artifact_root(fx):
    _, grown = seed_and_grown(fx, paths={"art": "/a", "repo": str(Spec.load(fx()).repo), "golden": "g/x"})
    assert grown.golden_root.as_posix() == "/a/fixture/g/x" and grown.hf_dir.as_posix() == "/a/fixture/hf"


def fake_drift(d, rung):
    drift_path(d, rung).write_text(json.dumps({"layer_L00": {"rel": 0.01, "rel_res": 0.001}}))


def test_an_imported_golden_verifies_as_a_partial_copy(fx):
    _, grown = seed_and_grown(fx)
    grown.raw["paths"]["golden"] = "copied"
    grown.path.write_text(yaml.safe_dump(grown.raw))
    grown = Spec.load(grown.path)
    seed_spec = Spec.load(fx())  # every layer, the same checkpoint: the golden is made here, "elsewhere"
    prompt.build(seed_spec)
    generate_golden.main(["--spec", str(seed_spec.path), "--rung", "s256"])
    made = rung_dir(seed_spec, seed_spec.rung("s256"))
    golden_import.main(["--spec", str(seed_spec.path), "--rung", "s256", "--index"])
    d = rung_dir(grown, grown.rung("s256"))
    with pytest.raises(SystemExit, match="not on this host"):
        golden_import.main(["--spec", str(grown.path), "--rung", "s256"])
    shutil.copytree(made, d)
    (d / "chunk_00" / "layer_01.safetensors").unlink()  # a copy without what the run's tests never read
    fake_drift(d, "s256")
    with pytest.raises(SystemExit, match=r"no drift for rung\(s\) \['seed'\]"):
        golden_import.main(["--spec", str(grown.path), "--rung", "s256"])
    fake_drift(d, "seed")
    golden_import.main(["--spec", str(grown.path), "--rung", "s256"])
    m = got()
    assert m["golden_hash_ok"] == 1 and m["golden_files_missing"] == 1 and m["golden_layers"] == 3
    assert m["drift_metrics"] == 2 and m["drift_layers_above_res"] == 1
    assert verify_copy(d) == (True, ["chunk_00/layer_01.safetensors"])
    (d / "chunk_01" / "model.safetensors").write_bytes(b"tampered")
    assert not verify_copy(d)[0]
    idx = json.loads((d / INDEX).read_text())
    idx["extra"] = "0" * 64
    (d / INDEX).write_text(json.dumps(idx))
    assert not verify_copy(d)[0]  # an index that does not give the content_hash


def test_the_port_brief_names_the_seed(fx, monkeypatch, tmp_path):
    from models.demos.common.bringup.core import runs
    from models.demos.common.bringup.orchestrator import Orchestrator

    seed, grown = seed_and_grown(fx)
    out = generate(grown)
    Ledger(grown.ledger_dir).write_tasks(out)
    runs.init_run(grown, Ledger(grown.ledger_dir), "run1")
    o = Orchestrator(grown, echo=lambda *_: None)
    text = o.brief(o.led.task("W.1"), "port", 1).read_text()
    assert "## Seed" in text and "Role: port" in text and "models/demos/fixture/bringup/spec.yaml" in text
    assert "mesh 2x2" not in text or "8x4" in text
    assert str(grown.ledger_dir.relative_to(grown.repo)) + "/results/knowledge_use" in text


def test_the_skill_and_readme_name_the_seeded_mode():
    from models.demos.common.bringup.core.spec import FRAMEWORK

    for f in ("skill/bringup/SKILL.md", "README.md"):
        text = (FRAMEWORK / f).read_text()
        assert "Full bring-up from a seed" in text and "golden_import" in text, f
