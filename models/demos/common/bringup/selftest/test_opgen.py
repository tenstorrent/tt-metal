# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F46: deferring a component step to op-gen. The DEFERRED status, the op request (generator and checker), the CPU
bridge and its metrics, the harness, the orchestrator's defer path. CPU only; the fixture model is the reference."""

import json
import types

import pytest
import torch
import yaml

from models.demos.common.bringup.core.ledger import Ledger
from models.demos.common.bringup.core.runs import IMPL_ENV, rerun_from
from models.demos.common.bringup.core.spec import Spec
from models.demos.common.bringup.plan import components
from models.demos.common.bringup.plan import op_request as OR
from models.demos.common.bringup.plan.ledger_gen import generate
from models.demos.common.bringup.reference import generate_golden, prompt
from models.demos.common.bringup.selftest import fixture_model
from models.demos.common.bringup.selftest.conftest import PY, got
from models.demos.common.bringup.testing import cpu_bridge
from models.demos.common.bringup.testing.host_transfers import HostTransfers

# ---------------------------------------------------------------- a request the agent would finish
NORM_REF = '''"""Pure-torch RMS norm of the fixture's attn_norm step."""

import torch


def pytorch_{op}(in_):
    x = in_.float()
    return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)).to(in_.dtype)
'''
MLP_REF = '''"""Pure-torch SwiGLU-less MLP of the fixture's mlp step."""

import torch


def pytorch_{op}(ffn_norm, w1, w2):
    x = ffn_norm.float()
    return (torch.nn.functional.silu(x @ w1.float().T) @ w2.float().T).to(ffn_norm.dtype)
'''
MLP_BIND = """def arguments(ref, layer, ctx):
    w = ref.w["layers"][layer]
    return [w["w1"], w["w2"]], {}
"""
EVIDENCE = {
    "searched": ["repo_map: norms", "grep -rn rms_norm ttnn/ttnn/operations"],
    "tried": [{"what": "ttnn.rms_norm", "outcome": "C.blk.x gate FAIL: pcc_x_L00 = 0.52 (need >= 0.99)"}],
    "why_not_fork": "the fixture's op has no TTNN counterpart whose kernels could be changed behind an option",
}


def finish(d, ref=NORM_REF, bind=None, weights=(), evidence=EVIDENCE):
    """Fill the generator's markers the way an agent would."""
    req = OR.load(d)
    req["evidence"] = evidence
    for name, shape in weights:
        req["interface"]["inputs"].append(
            {
                "name": name,
                "kind": "weight",
                "shape": shape,
                "dtype": "bfloat16",
                "layout": "TILE",
                "placement": "replicate",
            }
        )
    OR.save(d, req)
    op = req["op"]
    (d / "reference.py").write_text(ref.format(op=op))
    (d / "bind.py").write_text(bind or "def arguments(ref, layer, ctx):\n    return [], {}\n")
    text = (d / "op_prompt.txt").read_text().splitlines()
    out = []
    for line in text:
        if OR.MARK in line:
            out.append("- When the input is TILE: MUST NOT untilize on the host." if line.startswith("- ") else "Math.")
        elif "delete this line" in line or line.startswith("  the host") or "the host.>>" in line:
            continue
        else:
            out.append(line)
    (d / "op_prompt.txt").write_text("\n".join(l for l in out if OR.MARK not in l and ">>" not in l) + "\n")


@pytest.fixture
def gspec(fx, monkeypatch):
    """Fixture spec with its goldens and the generated ledger (C.blk.* tasks)."""

    def make(**over):
        p = fx(**over)
        monkeypatch.setenv("BRINGUP_SPEC", p)
        s = Spec.load(p)
        prompt.build(s)
        generate_golden.main(["--spec", p, "--rung", "s256"])
        Ledger(s.bringup_dir).write_tasks(generate(s, fixture_model.Reference()))
        return Spec.load(p)

    return make


def request(s, op="fixture_norm", tid="C.blk.attn_norm", **kw):
    d = OR.generate(s, tid, op)
    finish(d, **kw)
    OR.refresh(s, d)
    return d


# ---------------------------------------------------------------- ledger
def test_deferred_counts_like_pass_for_dependents(sandbox):
    led = sandbox.tasks(
        {"id": "A", "title": "a", "gate": {"cmd": "true"}},
        {"id": "B", "title": "b", "deps": ["A"], "gate": {"cmd": "true"}},
        {"id": "C", "title": "c", "deps": ["B"], "gate": {"cmd": "true"}},
    )
    led.update("A", status="DEFERRED")
    assert led.runnable() == ["B"] and led.first_unpassed() == "B" and led.deferred() == ["A"]
    led.update("B", status="PASS")
    assert rerun_from(led, "A") == ["A", "B", "C"]  # named with --from: redone
    led.update("A", status="DEFERRED")
    led.update("B", status="PASS")
    led.update("C", status="DEFERRED")
    assert rerun_from(led, "B") == ["B"] and led.status("C") == "DEFERRED"  # downstream deferral kept


def test_a_gate_runs_after_a_deferred_dep(sandbox):
    from models.demos.common.bringup.core.gate import run_gate

    led = sandbox.tasks(
        {"id": "A", "title": "a", "gate": {"cmd": "true"}},
        {"id": "B", "title": "b", "deps": ["A"], "gate": {"cmd": "true"}},
    )
    assert run_gate(sandbox.spec, led, "B").verdict == "BLOCKED"
    led.update("A", status="DEFERRED")
    assert run_gate(sandbox.spec, led, "B").verdict == "PASS"


def test_opgen_tag_needs_searched_and_block_steps_only(fx):
    s = Spec.load(fx())
    ref = fixture_model.Reference()
    from models.demos.common.bringup.selftest.test_plan import comps

    doc = comps(ref)
    for c in doc["components"]:
        if c["step"] == "mlp":
            c.update(tag="OPGEN", ttnn=None, searched=["repo_map: mlp"], op="fixture_mlp")
        if c["step"] == "embed":
            c.update(tag="OPGEN", ttnn=None)
    s.bringup_dir.mkdir(parents=True, exist_ok=True)
    (s.bringup_dir / "components.yaml").write_text(yaml.safe_dump(doc))
    errs = components.validate(s, ref)
    assert errs and all("model" in e for e in errs), errs  # embed: model-level, and no searched
    assert any("needs 'searched'" in e for e in errs) and any("OPGEN is for block steps" in e for e in errs)


# ---------------------------------------------------------------- generator and checker
def test_generator_fills_the_mechanical_parts(gspec):
    s = gspec(box={"mesh": [1, 4]})
    d = OR.generate(s, "C.blk.attn_norm", "fixture_norm")
    req = OR.load(d)
    assert req["model"] == "fixture" and req["tasks"] == ["C.blk.attn_norm"] and req["layers"] == [0, 1, 2]
    assert req["component"] == {"block_type": "blk", "step": "attn_norm", "kind": "norm", "stateful": False}
    x = req["interface"]["inputs"][0]
    assert x["name"] == "in" and x["shape"] == ["S", 64] and x["dtype"] == "bfloat16" and x["placement"] == "replicate"
    assert req["interface"]["chunks"] == [64, 128] and req["tolerance"] == {"pcc": 0.99, "rel_rms": 0.141}
    assert req["acceptance"]["test"].endswith("test_c_blk_attn_norm.py") and req["acceptance"]["golden"].endswith(
        "manifest.json"
    )
    assert (d / "op_prompt.txt").read_text().splitlines()[0] == "# golden: fixture_norm"
    assert "def pytorch_fixture_norm(in_):" in (d / "reference.py").read_text()
    fs = OR._import_file(d / "feature_spec.py", "fs_gen")
    assert fs.TARGET == {"dtype": ["bfloat16"], "layout": ["TILE"], "memory_layout": ["INTERLEAVED"]}
    assert fs.INPUTS == [((64, 64),), ((128, 64),)] and fs.INVALID == []
    # a sharded placement changes the per-device shapes
    req["interface"]["inputs"][0]["placement"] = "shard:1"
    OR.save(d, req)
    OR.refresh(s, d)
    assert OR._import_file(d / "feature_spec.py", "fs_gen2").INPUTS == [((64, 16),), ((128, 16),)]
    assert OR.per_device(["S", 64, 8], "shard2d:0,2", [2, 4]) == ["S/2", 64, 2]
    assert any("markers" in e for e in OR.check(s, d))  # a fresh request is not valid yet
    with pytest.raises(FileExistsError):
        OR.generate(s, "C.blk.attn_norm", "fixture_norm")
    with pytest.raises(ValueError, match="component task"):
        OR.generate(s, "S.blk.01", "other_op")


def test_a_finished_request_is_valid(gspec):
    s = gspec()
    assert OR.check(s, request(s)) == []
    mlp = request(
        s, "fixture_mlp", "C.blk.mlp", ref=MLP_REF, bind=MLP_BIND, weights=[("w1", [128, 64]), ("w2", [64, 128])]
    )
    assert OR.check(s, mlp) == []
    assert OR.for_task(s, "C.blk.mlp") == [mlp]
    assert OR._import_file(mlp / "feature_spec.py", "fs_mlp").INPUTS[0] == ((64, 64), (128, 64), (64, 128))


def _edit(d, fn):
    req = OR.load(d)
    fn(req)
    OR.save(d, req)


@pytest.mark.parametrize(
    "breaks,needle",
    [
        (lambda d: _edit(d, lambda r: r["evidence"].update(tried=[])), "evidence.tried is empty"),
        (lambda d: _edit(d, lambda r: r["evidence"].update(searched=[])), "evidence.searched is empty"),
        (lambda d: _edit(d, lambda r: r["evidence"].update(why_not_fork="no")), "why_not_fork"),
        (lambda d: _edit(d, lambda r: r["evidence"]["tried"][0].update(outcome="failed")), "quote the gate"),
        (lambda d: _edit(d, lambda r: r.update(status="done")), "status 'done'"),
        (lambda d: _edit(d, lambda r: r.update(tasks=["S.blk.01"])), "not a component task"),
        (lambda d: _edit(d, lambda r: r.update(tasks=["C.blk.mlp"])), "not the request's component"),
        (lambda d: (d / "op_prompt.txt").write_text("rms norm\n## Rules\n"), "first line"),
        (lambda d: (d / "reference.py").write_text("x = 1  # <<AGENT: later>>\n"), "markers"),
        (
            lambda d: (d / "feature_spec.py").write_text(
                (d / "feature_spec.py").read_text().replace("['bfloat16']", "['bfloat16', 'float32']")
            ),
            "exactly one value",
        ),
        (lambda d: (d / "feature_spec.py").write_text("TARGET = {}\nINPUTS = []\nINVALID = []\n"), "no case"),
        (
            lambda d: (d / "reference.py").write_text(NORM_REF.format(op="fixture_norm").replace("1e-6", "1e-1")),
            "does not reproduce",
        ),
        (
            lambda d: (d / "reference.py").write_text(
                "from models.demos.common.bringup.selftest.fixture_model import norm\n\n"
                "def pytorch_fixture_norm(in_):\n    return norm(in_)\n"
            ),
            "pure torch only",
        ),
        (lambda d: (d / "bind.py").unlink(), "missing bind.py"),
    ],
)
def test_the_checker_rejects(gspec, breaks, needle):
    s = gspec()
    d = request(s)
    breaks(d)
    errs = OR.check(s, d)
    assert any(needle in e for e in errs), errs


def test_the_checker_rejects_a_name_ttnn_already_has(gspec):
    s = gspec()
    d = request(s)
    (s.repo / "ttnn/ttnn/operations/fixture_norm").mkdir(parents=True)
    assert any("already in TTNN" in e for e in OR.check(s, d))
    assert any("differs from its folder" in e for e in OR.check(s, d.rename(d.parent / "Bad-Op")))
    _edit(d.parent / "Bad-Op", lambda r: r.update(op="Bad-Op"))
    assert any("snake_case" in e for e in OR.check(s, d.parent / "Bad-Op"))


def test_the_checker_cli(gspec, capsys):
    s = gspec()
    d = request(s)
    assert OR.main(["check", str(d), "--spec", str(s.path)]) == 0 and "valid" in capsys.readouterr().out
    (d / "reference.py").write_text("x = 1\n")
    assert OR.main(["check", "fixture_norm", "--spec", str(s.path)]) == 1
    assert "REJECTED" in capsys.readouterr().out


# ---------------------------------------------------------------- the bridge and its metrics
class FakeT:
    """A fake mesh tensor: one torch tensor per device (shape = the per-device shape, as in ttnn)."""

    def __init__(self, *parts, dtype="bf16", layout="TILE"):
        self.parts, self.shape, self.dtype, self.layout = list(parts), tuple(parts[0].shape), dtype, layout

    @property
    def t(self):
        return self.parts[0]


def fake_ttnn(n_dev=1):
    calls = []

    def from_torch(y, **k):
        calls.append(("from_torch", k.get("mesh_mapper")))
        m = k.get("mesh_mapper")
        if m and m[0] == "shard":
            return FakeT(*y.chunk(n_dev, dim=m[1]))
        return FakeT(*[y.clone() for _ in range(n_dev)])

    ns = types.SimpleNamespace(
        DRAM_MEMORY_CONFIG="dram",
        get_device_tensors=lambda x: [FakeT(p) for p in x.parts],
        to_torch=lambda t: t.t,
        from_torch=from_torch,
        synchronize_device=lambda d: None,
        ReplicateTensorToMesh=lambda mesh: ("replicate",),
        ShardTensorToMesh=lambda mesh, dim: ("shard", dim),
        ShardTensor2dMesh=lambda mesh, mesh_shape, dims: ("shard2d", dims),
    )
    ns.calls = calls
    return ns


def test_the_bridge_gathers_runs_and_places_and_is_not_a_host_transfer(fx):
    s = Spec.load(fx(box={"mesh": [1, 4]}))
    tt = fake_ttnn(4)
    x = torch.randn(1, 1, 8, 16)
    dev = FakeT(*x.chunk(4, dim=3))  # sharded on the hidden dim
    seen = {}

    def step(ctx, h):
        seen["shape"] = tuple(h.shape)
        return h * 2

    b = cpu_bridge.CpuBridge(None, s, "pool", step, inputs=["shard:3"], output="replicate", ttnn_module=tt)
    cpu_bridge.STATS.reset()
    with HostTransfers(tt) as ht:
        out = b(types.SimpleNamespace(layer=1), dev)
    assert seen["shape"] == (8, 16)  # gathered to the reference's [S, H]
    assert len(out.parts) == 4 and torch.equal(out.t, x * 2) and out.shape == (1, 1, 8, 16)
    assert ht.total == 0 and ht.bridge_calls == {"to_torch": 4, "from_torch": 1}  # to_torch per shard
    assert cpu_bridge.STATS.steps == {(1, "pool")} and cpu_bridge.STATS.ms >= 0 and b.cpu_bridge
    with HostTransfers(tt) as ht:
        tt.to_torch(dev)  # outside the bridge: counted
    assert ht.total == 1 and ht.bridge_total == 0
    grid = cpu_bridge.CpuBridge(None, Spec.load(fx(box={"mesh": [2, 2]})), "p", step, ttnn_module=tt)
    parts = FakeT(*[torch.full((2, 3), float(k)) for k in range(4)])
    assert grid.to_host(parts, "shard2d:0,1").shape == (4, 6) and grid.to_host(parts, "shard2d:none,1").shape == (2, 6)


def test_the_ladder_records_the_bridge_and_keeps_host_transfers_at_zero(fx, monkeypatch):
    from models.demos.common.bringup.testing import ladder

    p = fx()
    monkeypatch.setenv("BRINGUP_SPEC", p)
    s = Spec.load(p)
    prompt.build(s)
    generate_golden.main(["--spec", p, "--rung", "s256"])
    tt = fake_ttnn(1)

    class BridgedModel(fixture_model.FakeDeviceModel):
        """The fixture model with its mlp step run through the CPU bridge (fake single-device ttnn)."""

        def layer(self, i, h, start, state):
            from models.demos.common.bringup.reference.interface import run_block

            ctx = self.ref.chunk_context(i, start, h.shape[0], state.s)
            br = cpu_bridge.CpuBridge(None, s, "mlp", self.ref.component(i, "mlp"), ttnn_module=tt)
            ov = {"mlp": lambda c, x: br(c, FakeT(x)).t}
            return run_block(self.ref.block_graph(i), lambda n: self.ref.component(i, n), ctx, h, overrides=ov)

    monkeypatch.setattr(
        fixture_model, "device_model", lambda mesh, spec, layers, lm_head=True: BridgedModel(spec, layers)
    )
    monkeypatch.setattr(ladder, "HostTransfers", lambda: HostTransfers(tt))
    out = ladder.run_ladder(s, "s256", None)
    m = got()
    assert not out["failed"] and m["host_transfers_per_layer"] == 0 and m["device_model_hybrid"] == 0
    assert m["deferred_cpu_steps"] == 3 and m["deferred_cpu_ms"] >= 0  # mlp in each of the 3 layers


# ---------------------------------------------------------------- the harness
def test_swap_runs_a_deferred_step_on_the_reference_and_its_component_test_fails(gspec, monkeypatch):
    from models.demos.common.bringup.testing.component import run_component_test, run_swap_test

    s = gspec()
    monkeypatch.setitem(fixture_model.NOISE, "value", 0.5)  # the "device" mlp is wrong
    assert not run_swap_test(s, "blk", ["attn_norm", "attention", "attn_residual", "ffn_norm", "mlp"])
    Ledger(s.bringup_dir).update("C.blk.mlp", status="DEFERRED")
    monkeypatch.setitem(fixture_model.NOISE, "value", 1e-3)
    assert OR.deferred_steps(s) == {("blk", "mlp")}
    assert run_swap_test(s, "blk", ["attn_norm", "attention", "attn_residual", "ffn_norm", "mlp"])
    assert not run_component_test(s, "mlp", layer=0)  # never PASS while deferred
    monkeypatch.setenv(IMPL_ENV, "reference")
    assert run_component_test(s, "mlp", layer=0)  # freeze checks are unchanged


def test_a_component_test_refuses_a_cpu_bridge(gspec, monkeypatch):
    from models.demos.common.bringup.testing.component import run_component_test

    s = gspec()
    real = fixture_model.device_component

    def bridged(mesh, spec, layer, name):
        fn = real(mesh, spec, layer, name)
        fn.cpu_bridge = True
        return fn

    monkeypatch.setattr(fixture_model, "device_component", bridged)
    assert not run_component_test(s, "attn_norm", layer=0)


# ---------------------------------------------------------------- the orchestrator
MOCK = f"{PY} -m models.demos.common.bringup.selftest.mock_agent"
CHECK = """
import os
from pathlib import Path
from models.demos.common.bringup.core import metrics as M
impl = os.environ.get("BRINGUP_IMPL", "device")
dev = float(Path("src/impl.txt").read_text()) if Path("src/impl.txt").exists() else 0.1
M.record("pcc_out", {"reference": 1.0, "stub": 0.0, "device": dev}[impl])
"""


def request_files(op="fixture_norm", tid="C.1", evidence=EVIDENCE):
    """A finished op request for the fixture's attn_norm (layer 0), as the mock agent's files."""
    d = f"bringup/op_requests/{op}"
    req = {
        "op": op,
        "model": "toyspec",
        "tasks": [tid],
        "component": {"block_type": "blk", "step": "attn_norm", "kind": "norm", "stateful": False},
        "layers": [0, 1, 2],
        "representative_layer": 0,
        "status": "draft",
        "evidence": evidence,
        "interface": {
            "inputs": [
                {
                    "name": "in",
                    "kind": "activation",
                    "shape": ["S", 64],
                    "dtype": "bfloat16",
                    "layout": "TILE",
                    "placement": "replicate",
                    "per_device_shape": ["S", 64],
                }
            ],
            "output": {
                "name": "attn_norm",
                "kind": "output",
                "shape": ["S", 64],
                "dtype": "bfloat16",
                "layout": "TILE",
                "placement": "replicate",
                "per_device_shape": ["S", 64],
            },
            "chunks": [64],
            "params": {},
            "memory_layout": "INTERLEAVED",
        },
        "tolerance": {"pcc": 0.99, "rel_rms": 0.141},
        "acceptance": {"test": "tests/check.py", "golden": None, "metric": "pcc_out", "threshold": ">= 0.99"},
    }
    prompt_text = (
        f"# golden: {op}\nRMS norm over the last dim.\n\n## Rules\n\n- When TILE: MUST NOT untilize on the host.\n\n"
        f"Import path: from ttnn.operations.{op} import {op}\n"
    )
    fs = "TARGET = {'dtype': ['bfloat16'], 'layout': ['TILE'], 'memory_layout': ['INTERLEAVED']}\nINPUTS = [((64, 64),)]\nINVALID = []\nLOOSE_CASES = []\n"
    return {
        f"{d}/request.yaml": yaml.safe_dump(req, sort_keys=False),
        f"{d}/op_prompt.txt": prompt_text,
        f"{d}/feature_spec.py": fs,
        f"{d}/reference.py": NORM_REF.format(op=op),
        f"{d}/bind.py": "def arguments(ref, layer, ctx):\n    return [], {}\n",
    }


@pytest.fixture
def orch(sandbox, monkeypatch, tmp_path):
    from models.demos.common.bringup.orchestrator import Orchestrator

    script = tmp_path / "script.json"
    monkeypatch.setenv("BRINGUP_AGENT_CMD", MOCK)
    monkeypatch.setenv("MOCK_AGENT_SCRIPT", str(script))
    (sandbox.repo / "tests").mkdir()
    (sandbox.repo / "tests/check.py").write_text(CHECK)
    (sandbox.repo / "src").mkdir()
    sandbox.write_spec(
        hooks="models.demos.common.bringup.selftest.fixture_model",
        num_layers=3,
        block_types={"blk": {"layers": "0-2"}},
        ladder=[{"name": "s256", "seq": 256, "chunk": 64}],
    )
    lines = []

    def make(tasks, actions):
        script.write_text(json.dumps(actions))
        sandbox.tasks(*tasks)
        return Orchestrator(sandbox.spec, echo=lines.append)

    make.lines = lines
    make.calls = (
        lambda: script.with_suffix(".calls").read_text().split("\n")[:-1]
        if script.with_suffix(".calls").exists()
        else []
    )
    return make


def comp_task(**kw):
    t = {
        "id": "C.1",
        "title": "attn norm",
        "step": "implement",
        "deps": [],
        "tests": ["tests/check.py"],
        "paths": ["src"],
        "gate": {"cmd": f"{PY} tests/check.py", "metrics": {"pcc_out": ">= 0.99"}},
        "brief": {"block_type": "blk", "step": "attn_norm", "kind": "norm", "layer": 0},
    }
    t.update(kw)
    return t


AFTER = {"id": "S.1", "title": "swap", "step": "implement", "deps": ["C.1"], "gate": {"cmd": "true"}}


def test_a_valid_request_defers_the_task_and_the_run_continues(orch, sandbox):
    o = orch([comp_task(), AFTER], {"C.1.implement.1.md": {"write": request_files()}})
    assert o.run() == 0, orch.lines
    st = o.led.state()["C.1"]
    assert st["status"] == "DEFERRED" and st["deferred"]["op"] == "fixture_norm" and o.led.status("S.1") == "PASS"
    assert "deferred to op-gen: fixture_norm" in st["reason"][0] and "searched 2 places" in st["reason"][0]
    assert any("complete with 1 deferred: C.1 (fixture_norm)" in x for x in orch.lines)
    log = sandbox.git("log", "--format=%s")
    assert "[toy][C.1] attn norm (deferred to op-gen: fixture_norm)" in log
    assert "bringup/op_requests/fixture_norm/request.yaml" in sandbox.git("ls-files", "bringup/op_requests")
    brief = (o.run_dir / "briefs" / "C.1.implement.1.md").read_text()
    assert "## Deferring this step to op-gen" in brief and "op_request new <op> --task C.1" in brief
    assert "bringup/op_requests" in brief  # an allowed path for this role
    from models.demos.common.bringup.orchestrator import main

    assert main(["resume", "--spec", str(o.spec.path)]) == 0 and o.led.status("C.1") == "DEFERRED"  # resume keeps it


def test_thin_evidence_is_rejected_and_counts_as_a_failed_attempt(orch):
    thin = dict(EVIDENCE, why_not_fork="hard")
    o = orch(
        [comp_task()],
        {
            "C.1.implement.1.md": {"write": request_files(evidence=thin)},
            "C.1.implement.2.md": {"write": {"src/impl.txt": "0.999"}},
        },
    )
    assert o.run() == 0
    st = o.led.state()["C.1"]
    assert st["status"] == "PASS" and any(h["status"] == "DEFER_REJECTED" for h in st["history"])
    assert any("deferral rejected" in x and "why_not_fork" in x for x in orch.lines)
    brief2 = (o.run_dir / "briefs" / "C.1.implement.2.md").read_text()
    assert "rejected by the checker" in brief2 and "why_not_fork" in brief2


def test_after_the_debugger_one_last_attempt_may_defer(orch):
    o = orch([comp_task(), AFTER], {"C.1.implement.201.md": {"write": request_files()}})
    assert o.run() == 0, orch.lines
    calls = orch.calls()
    c1 = [c for c in calls if c.startswith("C.1.implement")]
    assert c1[-4:] == [f"C.1.implement.{100 + n}.md ttnn-expert-debugger" for n in (1, 2, 3)] + [
        "C.1.implement.201.md bringup-engineer"
    ]
    assert o.led.status("C.1") == "DEFERRED" and o.led.status("S.1") == "PASS"
    assert "## Deferring this step" not in (o.run_dir / "briefs" / "C.1.implement.101.md").read_text()  # the debugger's


def test_no_deferral_after_the_debugger_stops(orch):
    o = orch([comp_task()], {})
    assert o.run() == 1 and o.led.status("C.1") == "STOPPED"
    assert orch.calls()[-1] == "C.1.implement.201.md bringup-engineer"


def test_only_the_implement_role_of_a_component_task_may_write_a_request(orch):
    o = orch([comp_task(), AFTER], {})
    root = "bringup/op_requests"
    assert root in o.allowed_paths(o.led.task("C.1"), "implement")
    for role in ("test", "fix", "debug", "assemble"):
        assert root not in o.allowed_paths(o.led.task("C.1"), role), role
    assert root not in o.allowed_paths(o.led.task("S.1"), "implement")  # a swap task cannot defer


def test_an_opgen_tag_makes_the_task_defer_from_the_start(orch, sandbox):
    o = orch([comp_task()], {})
    (sandbox.repo / "bringup").mkdir(exist_ok=True)
    comp = {"block_type": "blk", "step": "attn_norm", "tag": "OPGEN", "reuse": "none", "searched": ["repo_map: norm"]}
    (sandbox.repo / "bringup/components.yaml").write_text(yaml.safe_dump({"components": [comp]}))
    text = o.brief(o.led.task("C.1"), "implement", 1).read_text()
    assert "The plan tagged this step OPGEN" in text and "defer it from the start" in text
    o.led.update(
        "C.1", status="DEFERRED", deferred={"op": "fixture_norm", "request": "bringup/op_requests/fixture_norm"}
    )
    text = o.brief(dict(o.led.task("C.1"), id="M.1", step="assemble", brief={}), "assemble", 1).read_text()
    assert "## Steps deferred to op-gen" in text and "- C.1: `blk.attn_norm`" in text and "CpuBridge" in text
