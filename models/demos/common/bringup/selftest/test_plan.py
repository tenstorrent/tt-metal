# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F5: plan memory check, components mapping, approvals, ledger generator, intake check, opportunity list, knowledge."""

import pytest
import torch
import yaml
from safetensors.torch import save_file

from models.demos.common.bringup.core.ledger import Ledger
from models.demos.common.bringup.core.spec import Spec
from models.demos.common.bringup.intake.check_checkpoint import check as check_ckpt
from models.demos.common.bringup.knowledge import check as kcheck
from models.demos.common.bringup.plan import approvals, components
from models.demos.common.bringup.plan.ledger_gen import generate
from models.demos.common.bringup.plan.memory import GB, check_plan, checkpoint_tensors
from models.demos.common.bringup.plan.opportunities import build
from models.demos.common.bringup.selftest.fixture_model import Reference

TENSORS = {
    "model.embed.weight": [1000, 64],
    "model.layers.0.q.weight": [64, 64],
    "model.layers.1.q.weight": [64, 64],
    "model.layers.0.experts.w": [8, 128, 64],
    "model.layers.1.experts.w": [8, 128, 64],
    "vision.patch.weight": [16, 16],
}


def plan(**over):
    p = {
        "placements": [
            {"pattern": "model.embed.weight", "placement": "replicate", "dtype": "bf16"},
            {"pattern": "model.layers.*.q.weight", "placement": "shard", "dtype": "bf16"},
            {"pattern": "model.layers.*.experts.*", "placement": "expert", "dtype": "bfp8"},
            {"pattern": "vision.*", "placement": "skip"},
        ],
        "state": [{"layers": "0-2", "heads_per_chip": 1, "head_dim": 64, "tensors": 2, "dtype": "bf16"}],
        "activations_gb_per_chip": 0.001,
    }
    p.update(over)
    return p


@pytest.fixture
def mspec(fx):
    return Spec.load(fx(box={"mesh": [1, 4], "chip_dram_gb": 0.01}))


def test_memory_from_checkpoint_shapes(mspec):
    r = check_plan(mspec, plan(), TENSORS, {"num_key_value_heads": 4})
    assert r["errors"] == [] and r["fits"]
    by = {x["group"]: x["gb"] * GB for x in r["rows"]}
    assert by["model.embed.weight"] == 1000 * 64 * 2
    assert by["model.layers.*.q.weight"] == 2 * 64 * 64 * 2 / 4
    assert by["model.layers.*.experts.*"] == pytest.approx(2 * 8 * 128 * 64 * 1088 / 1024 / 4)
    assert by["state: state layers 0-2"] == 3 * 64 * 2 * 2 * 256  # target seq 256, 1 user


@pytest.mark.parametrize(
    "over,needle",
    [
        ({"placements": plan()["placements"][:3]}, "match no placement"),
        (
            {"placements": plan()["placements"] + [{"pattern": "nothing.*", "placement": "skip"}]},
            "matches no checkpoint",
        ),
        (
            {"state": [{"layers": "0-1", "heads_per_chip": 1, "head_dim": 64, "dtype": "bf16"}]},
            "no state entry covers layers [2]",
        ),
        ({"state": [{"layers": "0-2", "heads_per_chip": 0, "head_dim": 64, "dtype": "bf16"}]}, "KV heads"),
        ({"activations_gb_per_chip": 0}, "positive estimate"),
        ({"placements": [{"pattern": "*", "placement": "shard", "dtype": "bf17"}]}, "unknown dtype"),
    ],
)
def test_memory_errors(mspec, over, needle):
    r = check_plan(mspec, plan(**over), TENSORS, {"num_key_value_heads": 4})
    assert not r["fits"] and any(needle in e for e in r["errors"]), r["errors"]


def test_memory_over_budget(mspec):
    r = check_plan(mspec, plan(activations_gb_per_chip=0.009), TENSORS, {"num_key_value_heads": 4})
    assert not r["fits"] and any("exceeds" in e for e in r["errors"])


def test_sliding_window_caps_state(mspec):
    st = [{"layers": "0-2", "heads_per_chip": 1, "head_dim": 64, "dtype": "bf16", "window": 32}]
    r = check_plan(mspec, plan(state=st), TENSORS, {"num_key_value_heads": 4})
    assert {x["group"]: x["gb"] * GB for x in r["rows"]}["state: state layers 0-2"] == 3 * 64 * 2 * 2 * 32


def test_checkpoint_headers(tmp_path):
    save_file({"a": torch.zeros(3, 5), "b": torch.zeros(2)}, str(tmp_path / "m-1.safetensors"))
    save_file({"c": torch.zeros(7, 1, 2)}, str(tmp_path / "m-2.safetensors"))
    assert checkpoint_tensors(tmp_path) == {"a": [3, 5], "b": [2], "c": [7, 1, 2]}


def test_intake_check(fx):
    s = Spec.load(
        fx(
            checkpoint={
                "expect": {"model.layers.*.q.weight": [64, "*"], "missing.*": [1]},
                "count": {"model.layers.*.q.weight": 3},
            }
        )
    )
    r = check_ckpt(s, TENSORS, {"num_hidden_layers": 3})
    assert r["missing"] == ["missing.*"] and r["shape"] == [] and len(r["count"]) == 1 and r["config"] == []
    assert check_ckpt(s, TENSORS, {"num_hidden_layers": 2})["config"]


def comps(ref, skip=None, **edit):
    out = []
    for st in ref.block_graph(0):
        if st.name != skip:
            out.append({"block_type": "blk", "step": st.name, "tag": "NATIVE", "ttnn": "ttnn.x", "reuse": "none"})
    out += [
        {"block_type": "model", "step": m, "tag": "NATIVE", "ttnn": "ttnn.y", "reuse": "none"}
        for m in components.MODEL_STEPS
    ]
    out[0].update(edit)
    return {"components": out}


def write(s, name, data):
    s.bringup_dir.mkdir(parents=True, exist_ok=True)
    (s.bringup_dir / name).write_text(yaml.safe_dump(data) if not isinstance(data, str) else data)


def test_components_validation(fx):
    s, ref = Spec.load(fx()), Reference()
    write(s, "components.yaml", comps(ref))
    assert components.validate(s, ref) == []
    write(s, "components.yaml", comps(ref, skip="mlp", tag="COMPOSED"))
    errs = components.validate(s, ref)
    assert any("(blk, mlp) has no component entry" in e for e in errs) and any("'searched'" in e for e in errs)


def test_ledger_generator(fx):
    s = Spec.load(fx())
    out = generate(s, Reference())
    ids = [t["id"] for t in out["tasks"]]
    assert ids[:8] == ["R.1", "R.2", "R.3", "G.s256", "G.s512", "B.1", "SC.1", "PL.0"]
    assert [t["id"] for t in generate(s, early=True)["tasks"]] == ids[:8]
    assert [i for i in ids if i.startswith("C.")] == [f"C.blk.{st.name}" for st in Reference().block_graph(0)]
    assert [i for i in ids if i.startswith("S.")] == [f"S.blk.{n:02d}" for n in range(1, 7)]
    assert ids[-9:] == ["L.s256", "L.s512", "L.last", "K.1", "X.1", "X.2", "X.3", "O.1", "Z.1"]
    z = next(t for t in out["tasks"] if t["id"] == "Z.1")
    assert z["role"] == "settings" and z["deps"] == ["O.1"] and "settings_lint" in z["gate"]["cmd"]
    o1 = next(t for t in out["tasks"] if t["id"] == "O.1")
    assert o1["role"] == "optests" and o1["deps"] == ["X.3"] and o1["paths"] == ["ttnn/ttnn/bringup"]
    rung = next(r for r in s.data["ladder"] if r["chunk"] == s.data["target"]["chunk"])  # the target-size chunk
    assert f"BRINGUP_RUNG={rung['name']} " in o1["gate"]["cmd"] and "fork_capture" in o1["gate"]["cmd"]
    assert o1["gate"]["metrics"]["fork_calls_uncovered"] == "== 0"
    t = {x["id"]: x for x in out["tasks"]}
    assert t["S.blk.03"]["deps"] == ["C.blk.attn_residual", "S.blk.02"]
    assert t["L.last"]["deps"] == ["L.s512", "G.s512"]
    assert t["C.blk.attention"]["tests"] == ["models/demos/fixture/tests/bringup/test_c_blk_attention.py"]
    assert t["C.blk.attention"]["freeze_extra"][0].endswith("golden/s256_c64/manifest.json")
    assert all(x.get("device") for x in out["tasks"] if x["step"] in ("implement", "integrate", "contract", "box"))
    Ledger(s.bringup_dir).write_tasks(out)
    assert Ledger(s.bringup_dir).validate() == []


def test_approvals_follow_the_approved_bytes(fx):
    s = Spec.load(fx())
    s.repo.mkdir(exist_ok=True)
    led = Ledger(s.bringup_dir)
    led.write_tasks(generate(s, Reference()))
    write(s, "plan.yaml", plan())
    write(s, "plan.md", "# plan\n")
    write(s, "components.yaml", comps(Reference()))
    assert not approvals.is_approved(s, "plan")
    approvals.approve(s, "plan", by="reviewer")
    assert approvals.is_approved(s, "plan")
    led.update_task_def("C.blk.mlp", frozen={"files": {"x": "y"}})  # freezing keeps the approval
    tasks = led.load_spec()
    tasks["tasks"].append(
        {"id": "P.1", "title": "picked", "step": "perf", "role": "perf", "deps": ["X.2"], "gate": {"cmd": "true"}}
    )
    led.write_tasks(tasks)  # a picked perf item keeps it too
    assert approvals.is_approved(s, "plan")
    tasks["tasks"][5]["gate"]["metrics"] = {"x": ">= 0"}  # a changed gate does not
    led.write_tasks(tasks)
    assert not approvals.is_approved(s, "plan")
    approvals.approve(s, "plan")
    write(s, "plan.yaml", plan(activations_gb_per_chip=5))
    assert not approvals.is_approved(s, "plan")


def test_opportunities():
    prof = {
        "rung": "last",
        "chunk": [4096, 8192],
        "layers": [0, 1],
        "wall_ms": 120.0,
        "sections_ms": {"attn.sdpa": 80.0, "moe.experts": 30.0, "moe.all_reduce": 10.0},
        "sections_ms_per_chip": {"attn.sdpa": {"0": 80.0, "1": 78.0}},
    }
    text, n = build(prof, 10)
    assert n == 3
    rows = [line for line in text.splitlines() if line.startswith("| [ ]")]
    assert "`attn.sdpa`" in rows[0] and "66.7%" in rows[0] and "2.0 ms" in rows[0]
    assert "fp32 accumulate disables streaming SDPA" in rows[0] and "Chunked causal attention" in rows[0]
    assert "Collectives on 1xN" in rows[2]


def test_knowledge_files_are_well_formed(tmp_path):
    errs, _ = kcheck.check_known_issues(kcheck.HERE / "known_issues.md")
    assert errs == []
    assert kcheck.check_repo_map(kcheck.HERE / "repo_map.md") == []
    bad = tmp_path / "ki.md"
    bad.write_text("## API behavior\n- **x.** Symptom: a. Cause: b.\n")
    errs, _ = kcheck.check_known_issues(bad)
    assert any("lacks" in e for e in errs) and any("missing section" in e for e in errs)


def test_hf_sanity_records_revision_and_accuracy(fx, tmp_path):
    from models.demos.common.bringup.intake import check_hf_sanity as H
    from models.demos.common.bringup.reference import prompt as P
    from models.demos.common.bringup.selftest.conftest import got

    s = Spec.load(fx())
    P.build(s)
    H.main(["--spec", str(s.path)])
    m = got()
    assert m["revision_ok"] == 1 and m["revision_pinned"] == 0 and 0 <= m["text_top1_acc"] <= 1
    pinned = Spec.load(fx(hf={"revision": "a" * 40}))
    H.main(["--spec", str(pinned.path)])
    assert got()["revision_ok"] == 0  # pinned, but the local checkout's revision is unknown or different
    d = tmp_path / "ckpt" / ".cache" / "huggingface" / "download"
    d.mkdir(parents=True)
    (d / "config.json.metadata").write_text("4d7ae4984b7db7de8f8457170b3f1a419ee76d52\nabc\n")
    assert H.local_revision(tmp_path / "ckpt") == "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"


def test_ledger_gates_the_intake_sanity(fx):
    s = Spec.load(fx(intake={"smoke": {"prompt": "capital of France?", "expect": "Paris"}}, text={"min_top1": 0.5}))
    r1 = generate(s, Reference())["tasks"][0]
    assert "check_hf_sanity" in r1["gate"]["cmd"]
    assert r1["gate"]["metrics"]["text_top1_acc"] == ">= 0.5" and r1["gate"]["metrics"]["smoke_ok"] == "== 1"


def test_custom_loader_moves_the_sanity_into_r2(fx):
    """F39: a checkpoint the stock HF loader cannot run here gets its sanity checks after the reference agent's loader."""
    s = Spec.load(fx(intake={"smoke": {"prompt": "capital?", "expect": "Paris"}}, hf={"custom_loader": True}))
    r1, r2 = generate(s, Reference())["tasks"][:2]
    assert "check_hf_sanity" not in r1["gate"]["cmd"] and "smoke_ok" not in r1["gate"]["metrics"]
    assert "revision_ok" not in r1["gate"]["metrics"] and r1["gate"]["metrics"]["prompt_hash_ok"] == "== 1"
    assert r2["gate"]["cmd"].index("check_hf_sanity") < r2["gate"]["cmd"].index("check_hf ")
    m = r2["gate"]["metrics"]
    assert m["smoke_ok"] == "== 1" and m["revision_ok"] == "== 1" and "pcc_logits" in m and "text_top1_acc" in m


def test_profile_gates_size_the_profiler_for_large_models():
    """F52: the timeline dropped programs past the profiler's default 1000 per core (GLM: 1319 per chunk)."""
    from models.demos.common.bringup.plan.ledger_gen import PROFILE_ENV

    counts = [
        int(w.split("=")[1]) for w in PROFILE_ENV.split() if w.startswith("TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=")
    ]
    assert counts and counts[0] >= 4000


SERVING_MD = """# Serving contract for fixture (tt-d-gen @ 24c381bb6d39)

## Input
Tokens arrive rotated for any start.

## KV cache
One region per slot.

## Attention and cache writes
Write the KV at any 32-aligned start.

## Acks
One device-to-host ack per layer.

## Adapter and table
build_kv_chunk_table(kv, path, first_layer_idx=..., num_my_layers=..., stage_layout=...).

## Deployment
PREFILL_FABRIC_MODE=2d.

## Questions for the owner
KV dtype for the decode side? (default bf16)
"""


def test_serving_contract_gates_the_steps_it_names(fx):
    """F58: SC.1 comes before the plan; its tests join the gates of the steps they name and the runner test joins
    K.1; a step's brief carries its how-to section; the SC.1 gate checks the files agree."""
    from models.demos.common.bringup.testing import serving as SV

    s = Spec.load(fx())
    s.repo.mkdir(exist_ok=True)
    tdir = s.repo / "models/demos/fixture/tests/bringup/contract"
    tdir.mkdir(parents=True)
    for n in ("test_contract_kv_write.py", "test_contract_runner.py"):
        (tdir / n).write_text("def test_x():\n    assert False, 'not built'\n")
    write(s, "serving_contract.md", SERVING_MD)
    tests = [
        {
            "test": "models/demos/fixture/tests/bringup/contract/test_contract_kv_write.py",
            "checks": "unaligned starts",
            "section": "Attention and cache writes",
            "gates": "attention",
        },
        {
            "test": "models/demos/fixture/tests/bringup/contract/test_contract_runner.py",
            "checks": "real runner",
            "section": "Adapter and table",
            "gates": "adapter",
        },
    ]
    write(s, "contract_tests.yaml", {"tests": tests})
    assert SV.check(s, s.repo) == []
    t = {x["id"]: x for x in generate(s, Reference())["tasks"]}
    assert t["SC.1"]["role"] == "serving" and "SC.1" in t["PL.1"]["deps"]
    assert "test_contract_kv_write.py" in t["C.blk.attention"]["gate"]["cmd"]
    assert "test_contract_kv_write.py" not in t["C.blk.mlp"]["gate"]["cmd"]
    assert "test_contract_runner.py" in t["K.1"]["gate"]["cmd"]
    text = SV.brief_text(s, "attention")
    assert "Write the KV at any 32-aligned start." in text and "test_contract_kv_write.py" in text
    assert SV.brief_text(s, "mlp") == ""
    write(s, "serving_contract.md", SERVING_MD.replace("## Acks\n", "## Ackz\n"))
    assert any("'## Acks' missing" in e for e in SV.check(s, s.repo))
    write(s, "contract_tests.yaml", {"tests": tests[:1]})
    assert any("no runner test" in e for e in SV.check(s, s.repo))


def test_serving_run_tests_runs_the_named_gate(fx):
    from models.demos.common.bringup.testing import serving as SV

    s = Spec.load(fx())
    ts = [{"test": "a.py", "gates": "attention", "section": "x"}, {"test": "b.py", "gates": "adapter", "section": "x"}]
    write(s, "contract_tests.yaml", {"tests": ts})
    seen = []
    fail_b = lambda cmd: seen.append(cmd) or (1 if cmd.endswith("b.py") else 0)  # noqa: E731
    assert SV.run_tests(s, s.repo, "all", fail_b) == ["b.py"] and len(seen) == 2
    assert SV.run_tests(s, s.repo, "attention", fail_b) == [] and seen[-1].endswith("a.py")


def test_serving_gate_commit_stages_its_outputs(fx):
    """F58: SC.1's gate commit carries the how-to, the test list and the contract tests (it missed them on Xing)."""
    from models.demos.common.bringup.core.gate import stage_paths

    s = Spec.load(fx())
    s.repo.mkdir(exist_ok=True)
    write(s, "serving_contract.md", SERVING_MD)
    write(s, "contract_tests.yaml", {"tests": []})
    tdir = s.model_dir / "tests/bringup/contract"
    tdir.mkdir(parents=True)
    (tdir / "test_x.py").write_text("x = 1\n")
    led = Ledger(s.bringup_dir)
    led.write_tasks(generate(s, Reference()))
    sc = next(t for t in generate(s, Reference())["tasks"] if t["id"] == "SC.1")
    got = stage_paths(s, led, sc)
    assert any(p.endswith("serving_contract.md") for p in got) and any(p.endswith("contract_tests.yaml") for p in got)
    assert any(p.endswith("tests/bringup/contract") for p in got)


def test_served_kv_dtype_is_required_on_every_ladder_rung(fx):
    s = Spec.load(fx())
    s.data.setdefault("serving", {})["kv_dtype"] = "bfp8"
    t = {x["id"]: x for x in generate(s, Reference())["tasks"]}
    names = s.get("state.tensors") or []
    assert names and all(
        t[f"L.{r['name']}"]["gate"]["metrics"][f"state_bits_{names[0]}"] == "== 8" for r in s.data["ladder"]
    )
    s.data["serving"].pop("kv_dtype")
    t = {x["id"]: x for x in generate(s, Reference())["tasks"]}
    assert not any(k.startswith("state_bits_") for k in t["L.s256"]["gate"]["metrics"])


def test_settings_gate_commit_stages_the_model_code(fx):
    """Z.1's gate commit carries tt/ and hooks.py (it missed them on Xing: the role's paths are computed, not listed)."""
    from models.demos.common.bringup.core.gate import stage_paths

    s = Spec.load(fx())
    s.repo.mkdir(exist_ok=True)
    (s.model_dir / "tt").mkdir(parents=True)
    (s.model_dir / "tt/settings.py").write_text("x = 1\n")
    (s.bringup_dir).mkdir(parents=True, exist_ok=True)
    (s.bringup_dir / "hooks.py").write_text("x = 1\n")
    led = Ledger(s.bringup_dir)
    led.write_tasks(generate(s, Reference()))
    z = next(t for t in generate(s, Reference())["tasks"] if t["id"] == "Z.1")
    got = stage_paths(s, led, z)
    assert any(p.endswith("/tt") for p in got) and any(p.endswith("bringup/hooks.py") for p in got)
