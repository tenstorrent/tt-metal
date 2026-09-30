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


# ---------------------------------------------------------------- approval, export, delivery
def _mlp(s):
    return request(
        s, "fixture_mlp", "C.blk.mlp", ref=MLP_REF, bind=MLP_BIND, weights=[("w1", [128, 64]), ("w2", [64, 128])]
    )


def _fake_ttnn_module():
    ttnn = types.ModuleType("ttnn")
    for k in ("bfloat16", "float32", "bfloat8_b", "uint32", "int32", "TILE_LAYOUT", "ROW_MAJOR_LAYOUT"):
        setattr(ttnn, k, f"<{k}>")
    ttnn.TensorMemoryLayout = types.SimpleNamespace(INTERLEAVED="<INTERLEAVED>")
    return ttnn


def test_approval_is_voided_by_an_edit_and_kept_by_the_lifecycle(gspec):
    from models.demos.common.bringup.plan import approvals

    s = gspec()
    d = _mlp(s)
    assert not approvals.op_request_approved(s, "fixture_mlp")
    approvals.approve_op_request(s, "fixture_mlp", by="owner")
    assert approvals.op_request_approved(s, "fixture_mlp") and OR.load(d)["status"] == "approved"
    OR.set_status(d, "exported", exported={"at": "now"})  # lifecycle fields do not void it
    assert approvals.op_request_approved(s, "fixture_mlp")
    (d / "op_prompt.txt").write_text((d / "op_prompt.txt").read_text() + "\nMore.\n")
    assert not approvals.op_request_approved(s, "fixture_mlp")


def test_op_export_writes_the_prompt_and_a_suite_whose_reference_is_the_requests(gspec, tmp_path, monkeypatch, capsys):
    import ast
    import sys

    from models.demos.common.bringup.__main__ import main
    from models.demos.common.bringup.plan import approvals
    from models.demos.common.bringup.plan.op_export import export

    s = gspec()
    _mlp(s)
    root = tmp_path / "codegen"
    (root / "eval").mkdir(parents=True)
    with pytest.raises(PermissionError, match="not approved"):
        export(s, "fixture_mlp", root)
    approvals.approve_op_request(s, "fixture_mlp", by="owner")
    assert main(["op-export", "fixture_mlp", "--spec", str(s.path), "--codegen-root", str(root)]) == 0
    out = capsys.readouterr().out
    assert (
        "Nothing was committed, pushed or launched" in out and "run_eval.py .claude/eval/prompts/fixture_mlp.txt" in out
    )
    assert OR.load(OR.request_dir(s, "fixture_mlp"))["status"] == "exported"
    g = root / "eval/golden_tests/fixture_mlp"
    names = {
        "__init__.py",
        "feature_spec.py",
        "helpers.py",
        "axes.py",
        "test_golden.py",
        "conftest.py",
        "test_regression.py",
    }
    assert {p.name for p in g.iterdir()} == names
    for p in g.iterdir():
        ast.parse(p.read_text())  # every file parses
    assert (root / "eval/prompts/fixture_mlp.txt").read_text().splitlines()[0] == "# golden: fixture_mlp"
    monkeypatch.setitem(sys.modules, "ttnn", _fake_ttnn_module())
    fs = OR._import_file(g / "feature_spec.py", "exported_fs")
    assert fs.TARGET == {"dtype": ["<bfloat16>"], "layout": ["<TILE_LAYOUT>"], "memory_layout": ["<INTERLEAVED>"]}
    assert fs.INPUTS == [((64, 64), (128, 64), (64, 128)), ((128, 64), (128, 64), (64, 128))] and fs.INVALID == []
    helpers = (g / "helpers.py").read_text()
    assert "TOLERANCES = {ttnn.bfloat16: (0.99, 0.141)}" in helpers and "from eval.metrics import" in helpers
    assert "import models" not in helpers and "from models" not in helpers
    # the suite's reference is the request's reference.py, verbatim
    fn = next(n for n in ast.parse(helpers).body if isinstance(n, ast.FunctionDef) and n.name == "pytorch_fixture_mlp")
    ns = {"torch": torch}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "helpers", "exec"), ns)
    ref = OR._import_file(OR.request_dir(s, "fixture_mlp") / "reference.py", "req_ref").pytorch_fixture_mlp
    x, w1, w2 = torch.randn(8, 64), torch.randn(128, 64), torch.randn(64, 128)
    assert torch.equal(ns["pytorch_fixture_mlp"](x, w1, w2), ref(x, w1, w2))
    assert "run_fixture_mlp(inputs, device=device, **axes)" in (g / "test_golden.py").read_text()
    assert "from ttnn.operations.fixture_mlp import fixture_mlp as _raw, INPUT_TAGGERS" in (g / "axes.py").read_text()


def _generated_op(root, op):
    d = root / "ttnn/ttnn/operations" / op
    (d / "kernels").mkdir(parents=True)
    (d / "__init__.py").write_text(f"from ttnn.operations.{op}.{op} import {op}\n")
    (d / f"{op}.py").write_text(
        f'KERNEL = "ttnn/ttnn/operations/{op}/kernels/compute.cpp"\n\ndef {op}(x):\n    return x\n'
    )
    (d / "kernels/compute.cpp").write_text("// kernel\n")
    return d


def test_op_ready_registers_the_ops_and_resets_the_deferred_tasks_and_their_dependents(gspec, tmp_path, capsys):
    import ast
    import shutil as sh

    from models.demos.common.bringup.__main__ import main
    from models.demos.common.bringup.core.spec import CODE_ROOT
    from models.demos.common.bringup.plan import approvals
    from models.demos.common.bringup.plan.op_export import export

    s = gspec()
    bops = s.repo / "ttnn/ttnn/bringup"  # a copy of the real registry; the real tree is never touched
    bops.mkdir(parents=True)
    sh.copy(CODE_ROOT / "ttnn/ttnn/bringup/__init__.py", bops / "__init__.py")
    sh.copy(CODE_ROOT / "ttnn/ttnn/bringup/INDEX.md", bops / "INDEX.md")
    before_init = (bops / "__init__.py").read_text()
    codegen = tmp_path / "codegen"
    (codegen / "eval").mkdir(parents=True)
    led = Ledger(s.bringup_dir)
    for op, tid, kw in (("fixture_norm", "C.blk.attn_norm", {}), ("fixture_mlp", "C.blk.mlp", None)):
        request(s, op, tid, **kw) if kw is not None else _mlp(s)
        approvals.approve_op_request(s, op, by="owner")
        export(s, op, codegen)
        led.update(tid, status="DEFERRED", deferred={"op": op})
    for tid in led.topo_order():
        if led.status(tid) != "DEFERRED":
            led.update(tid, status="PASS")
    clone = tmp_path / "clone"
    _generated_op(clone, "fixture_norm")
    _generated_op(clone, "fixture_mlp")
    args = [
        "op-ready",
        "fixture_norm",
        "fixture_mlp",
        "--from",
        str(clone / "ttnn/ttnn/operations"),
        "--spec",
        str(s.path),
    ]
    assert main(args + ["--no-commit"]) == 0
    out = capsys.readouterr().out
    assert "delivered: ttnn.bringup.fixture_norm, ttnn.bringup.fixture_mlp" in out and "orchestrator resume" in out

    def python_ops(text):
        tree = ast.parse(text)
        return ast.literal_eval(
            next(n for n in tree.body if isinstance(n, ast.Assign) and n.targets[0].id == "PYTHON_OPS").value
        )

    assert python_ops((bops / "__init__.py").read_text()) == {
        **python_ops(before_init),  # the ops already registered stay (mhc_pre / mhc_post)
        "fixture_norm": ("fixture_norm", "fixture_norm"),
        "fixture_mlp": ("fixture_mlp", "fixture_mlp"),
    }
    assert "def __getattr__(name):" in (bops / "__init__.py").read_text()
    assert before_init.split("PYTHON_OPS")[0] == (bops / "__init__.py").read_text().split("PYTHON_OPS")[0]
    code = (bops / "fixture_mlp/fixture_mlp.py").read_text()
    assert "ttnn/ttnn/bringup/fixture_mlp/kernels" in code and "operations" not in code
    assert (
        bops / "fixture_mlp/__init__.py"
    ).read_text() == "from ttnn.bringup.fixture_mlp.fixture_mlp import fixture_mlp\n"
    cl = (bops / "fixture_mlp/CHANGELOG.md").read_text()
    assert "generated by op-gen from op request fixture/fixture_mlp" in cl and "## Changes" in cl
    assert "| `fixture_mlp` | op-gen, op request `fixture/fixture_mlp`" in (bops / "INDEX.md").read_text()
    assert OR.load(OR.request_dir(s, "fixture_mlp"))["status"] == "delivered"
    st = led.state()
    reset = set(led.downstream("C.blk.attn_norm")) | set(led.downstream("C.blk.mlp"))
    assert {"S.blk.01", "S.blk.05", "M.1", "L.s256", "K.1", "X.1", "X.3", "O.1"} <= reset
    assert all(st[t]["status"] == "TODO" for t in reset)
    assert st["C.blk.attention"]["status"] == "PASS" and st["R.3"]["status"] == "PASS"  # upstream / unrelated keep
    assert led.task("C.blk.mlp")["brief"]["details"].startswith("use ttnn.bringup.fixture_mlp (op-gen, request ")
    with pytest.raises(ValueError, match="delivered"):
        main(args + ["--no-commit"])


# ---------------------------------------------------------------- dashboard
def test_dashboard_shows_deferred_steps(gspec, tmp_path, monkeypatch):
    import shutil as sh
    import subprocess

    from models.demos.common.bringup.core import metrics as M
    from models.demos.common.bringup.dashboard.export import build, main
    from models.demos.common.bringup.reference.check_reference import write_graphs
    from models.demos.common.bringup.selftest.test_dashboard import SHIM
    from models.demos.common.bringup.selftest.test_dashboard_prior import TT_TEXT

    s = gspec()
    _mlp(s)
    led = Ledger(s.bringup_dir)
    led.update("C.blk.mlp", status="DEFERRED", deferred={"op": "fixture_mlp"})
    monkeypatch.setenv(M.RESULTS_ENV, str(led.results_dir))
    M.record("deferred_cpu_steps", 3, task="X.1")
    for k, v in dict(
        chunk_wall_ms=800, chunk_start=192, chunk_len=64, device_model_hybrid=0, deferred_cpu_ms=12
    ).items():
        M.record(k, v, task="X.1")
    write_graphs(fixture_model.Reference(), {"blk": 0})
    prof = {
        "chunk": [192, 256],
        "layers": [0],
        "wall_ms": 9.0,
        "sections_ms": {"mlp": 1.0, "attention": 4.0},
        "sections_ms_per_chip": {},
        "programs": {},
        "bridged_steps": ["mlp"],
        "deferred_cpu_ms": 5.0,
    }
    (led.results_dir / "X.1_profile.json").write_text(json.dumps(prof))
    d = build(s)
    assert [(r["task"], r["op"], r["status"], r["layers"]) for r in d["deferred"]] == [
        ("C.blk.mlp", "fixture_mlp", "draft", "L0–2")
    ]
    assert d["deferred"][0]["tried"][0].startswith("ttnn.rms_norm: C.blk.x gate FAIL")
    assert next(st for st in d["graphs"]["blk"] if st["name"] == "mlp")["deferred"]
    assert "3 CPU steps (deferred to op-gen" in d["timing"][0]["model"]
    steps = {x["key"]: x for x in d["profile"]["steps"]}
    assert steps["mlp"]["cpu_bridge"] and "(CPU bridge" in steps["mlp"]["name"] and not steps["attention"]["cpu_bridge"]
    assert "deferred to op-gen (mlp)" in d["profile"]["note"]
    if not sh.which("node"):
        pytest.skip("node not installed")
    outs = main(["--spec", str(s.path), "--style", "both", "--out", str(tmp_path)])
    r = subprocess.run(["node", str(SHIM), str(tmp_path / "index.html")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    els = json.loads(r.stdout)
    assert not els["#def-sec"]["hidden"] and els["#deferred tbody"]["html"] > 0
    assert not els["#def-banner"]["hidden"] and "include 1 step on the CPU" in els["#def-banner"]["text"]
    r = subprocess.run(
        ["node", "-e", TT_TEXT.replace("p107: (B[107]", "p107: (B[108]"), str(tmp_path / "teletext.html")],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    tt = json.loads(r.stdout)
    assert 108 in tt["pages"] and "DEFERRED TO OP-GEN" in tt["p107"][0] and "PAGE ERROR" not in " ".join(tt["p107"])
    assert len(outs) == 2


def test_status_and_op_requests_show_deferrals(gspec, capsys):
    from models.demos.common.bringup.__main__ import main

    s = gspec()
    _mlp(s)
    Ledger(s.bringup_dir).update("C.blk.mlp", status="DEFERRED", deferred={"op": "fixture_mlp", "request": "x"})
    assert main(["status", "--spec", str(s.path)]) == 0
    out = capsys.readouterr().out
    assert "C.blk.mlp DEFERRED" in out and "1 deferred to op-gen" in out and "C.blk.mlp: fixture_mlp (x)" in out
    assert main(["op-requests", "--spec", str(s.path)]) == 0
    out = capsys.readouterr().out
    assert (
        "fixture_mlp" in out and "draft" in out and "blk.mlp [C.blk.mlp DEFERRED]" in out and "searched 2 places" in out
    )
