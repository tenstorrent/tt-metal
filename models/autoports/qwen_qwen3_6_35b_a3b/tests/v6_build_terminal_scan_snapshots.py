# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Build static decode-only terminal-scan diagnostics, alone and with norm-tail."""

import argparse
import ast
import copy
import hashlib
import json
import sys
import types
from pathlib import Path

from models.autoports.qwen_qwen3_6_35b_a3b.tests.frozen_source_compatibility import raw_sha256, verify_source

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "doc/fused_decoder/endurance"
BASES = {
    "terminal_scan": (
        "v6_corrected_fma_explicit_norm_full_v5.py.snapshot",
        "52df1b042d445630ef9bdce0d100245fa2faad9dd801042dc638b24bf3fe50d9",
    ),
    "terminal_scan_norm_tail": (
        "v6_corrected_fma_full_v5_norm_tail.py.snapshot",
        "9362bfa3984efda2a63f383d12b419783a996831e2cff97aeb05a28ba25eaff1",
    ),
}
HELPER = ROOT / "tests/v6_terminal_projection_scan_probe.py"
HELPER_SHA = "8c9eeb05d3cd388fdffe581c459b2178fd2c526c6117f630a779154d39736b64"
ROUTER = ROOT / "tests/v6_ordered_router_fusion_probe.py"
ROUTER_SHA = "067595f30f1b7e29c65795748dddd9b594725d40190c8541c0863a9cf8dd5314"
RENAMES = {
    "READER": "TERMINAL_SCAN_READER",
    "COMPUTE": "TERMINAL_SCAN_COMPUTE",
    "WRITER": "TERMINAL_SCAN_WRITER",
    "schedule": "terminal_projection_schedule",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(node):
    return hashlib.sha256(ast.dump(node).encode()).hexdigest()


def named(scope, name):
    return next(n for n in scope.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)


class Rename(ast.NodeTransformer):
    def __init__(self, names):
        self.names = names

    def visit_Name(self, node):
        node.id = self.names.get(node.id, node.id)
        return node

    def visit_FunctionDef(self, node):
        node.name = self.names.get(node.name, node.name)
        return self.generic_visit(node)


def helper_source():
    assert verify_source(HELPER, HELPER_SHA) and verify_source(ROUTER, ROUTER_SHA)
    evidence_path = E / "v6_terminal_projection_scan_device.json"
    evidence = json.loads(evidence_path.read_text())
    assert evidence["status"] == "component_pass" and evidence["device_closed"]
    assert evidence["script_sha256"] == HELPER_SHA and evidence["frozen_router_sha256"] == ROUTER_SHA
    assert evidence["fixture_cases"] == len(evidence["cases"]) == 32
    assert all(all(v["exact"] for v in c["outputs"].values()) for c in evidence["cases"])
    partial_cases = [c for c in evidence["cases"] if "partial_fp32" in c]
    assert len(partial_cases) == 30 and all(c["partial_fp32"]["exact"] for c in partial_cases)
    assert evidence["multiple_leading_axes_fp32"]["exact"]
    original = ast.parse(HELPER.read_text())
    router_tree = ast.parse(ROUTER.read_text())
    strings = {
        n.targets[0].id: ast.literal_eval(n.value)
        for n in router_tree.body
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id in ("READER", "COMPUTE", "WRITER")
    }
    scope = {"router": types.SimpleNamespace(**strings)}
    assignments = [
        copy.deepcopy(n)
        for n in original.body
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in strings
    ]
    exec(compile(ast.Module(assignments, []), "<terminal-source-build-only>", "exec"), scope)
    literals = [ast.Assign([ast.Name(RENAMES[k], ast.Store())], ast.Constant(scope[k])) for k in strings]
    helpers = [copy.deepcopy(named(original, k)) for k in ("schedule", "TerminalProjectionScan")]
    before = copy.deepcopy(helpers)
    helpers = [Rename(RENAMES).visit(n) for n in helpers]
    restored = [Rename({v: k for k, v in RENAMES.items()}).visit(copy.deepcopy(n)) for n in helpers]
    assert ast.dump(ast.Module(restored, [])) == ast.dump(ast.Module(before, []))
    return literals + helpers, {k: scope[k] for k in strings}, evidence, evidence_path


def add_flag(call, expression):
    assert not any(k.arg == "terminal_scan" for k in call.keywords)
    call.keywords.append(ast.keyword("terminal_scan", ast.parse(expression, mode="eval").body))


def build(mode, embedded, strings, evidence, evidence_path):
    base_name, base_sha = BASES[mode]
    base = E / base_name
    assert sha(base) == base_sha
    tree = ast.parse(base.read_text())
    original = copy.deepcopy(tree)
    cls = named(tree, "FusedDecoder")
    setup = named(cls, "from_state_dict")
    addition = ast.parse(
        "if self.kind == 'linear_attention':\n    self.terminal_projection_scan = TerminalProjectionScan(ttnn, mesh_device)"
    ).body[0]
    assert isinstance(setup.body[-1], ast.Return)
    setup.body.insert(len(setup.body) - 1, addition)
    linear = named(cls, "linear")
    linear.args.kwonlyargs.append(ast.arg("terminal_scan"))
    linear.args.kw_defaults.append(ast.Constant(False))
    changes = []
    for branch in linear.body[1:4]:
        assert isinstance(branch, ast.If)
        i = next(
            i
            for i, n in enumerate(branch.body)
            if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "prefixes"
        )
        old = copy.deepcopy(branch.body[i : i + 6])
        assert [ast.unparse(n).split(" = ")[0] for n in old] == [
            "prefixes",
            "del products",
            "tail",
            "del prefixes",
            "partial",
            "del tail",
        ]
        assert "disable_compensated_sum=True" in ast.unparse(old[0])
        replacement = ast.If(
            ast.Name("terminal_scan", ast.Load()),
            ast.parse("partial = self.terminal_projection_scan(products)\ndel products").body,
            copy.deepcopy(old),
        )
        branch.body[i : i + 6] = [replacement]
        changes.append((branch, i, old))
    attention = named(cls, "_linear_attention")
    calls = []
    for call in ast.walk(attention):
        if not isinstance(call, ast.Call) or ast.unparse(call.func) != "self.linear":
            continue
        key = ast.literal_eval(call.args[1])
        if key in ("attention_packed", "output_only_packed"):
            add_flag(call, "decode and cache_boundary")
            calls.append(call)
        elif any(k.arg == "ordered_out" for k in call.keywords):
            add_flag(call, "decode and cache_boundary")
            calls.append(call)
    assert len(calls) == 3
    moe = named(cls, "_moe")
    router_calls = [
        n
        for n in ast.walk(moe)
        if isinstance(n, ast.Call)
        and ast.unparse(n.func) == "self.linear"
        and any(k.arg == "ordered_router" for k in n.keywords)
    ]
    assert len(router_calls) == 1
    add_flag(router_calls[0], "ordered_router")
    calls += router_calls
    # Public decode is the only source of ordered_router=True. The full-kind
    # early return still has its literal V5 body; all prefill passes False.
    forward = named(cls, "_forward")
    ordered_values = [
        ast.unparse(k.value)
        for n in ast.walk(forward)
        if isinstance(n, ast.Call) and ast.unparse(n.func) == "self._moe"
        for k in n.keywords
        if k.arg == "ordered_router"
    ]
    assert ordered_values == ["decode and cache_boundary"]
    candidate = copy.deepcopy(tree)
    # Exact module restoration proves all unlisted paths/setup stay unchanged.
    setup.body.remove(addition)
    linear.args.kwonlyargs.pop()
    linear.args.kw_defaults.pop()
    for branch, i, old in changes:
        assert ast.dump(ast.Module(branch.body[i].orelse, [])) == ast.dump(ast.Module(old, []))
        branch.body[i : i + 1] = old
    for call in calls:
        assert call.keywords[-1].arg == "terminal_scan"
        call.keywords.pop()
    assert ast.dump(tree) == ast.dump(original)
    candidate_cls = named(candidate, "FusedDecoder")
    full_guards = {}
    for method in ("from_state_dict", "linear", "norm", "_moe", "_forward", "prepare_prefill", "prefill_forward"):
        old_method, new_method = named(named(original, "FusedDecoder"), method), named(candidate_cls, method)
        old_guard = next(n for n in old_method.body if isinstance(n, ast.If))
        new_guard = next(n for n in new_method.body if isinstance(n, ast.If))
        assert digest(old_guard) == digest(new_guard)
        full_guards[method] = digest(new_guard)
    candidate.body += copy.deepcopy(embedded)
    ast.fix_missing_locations(candidate)
    assert not any(
        isinstance(n, (ast.Import, ast.ImportFrom)) and "tests" in ast.unparse(n) for n in ast.walk(candidate)
    )
    assert not any(
        isinstance(n, ast.Call) and ast.unparse(n.func) in ("exec", "eval", "compile") for n in ast.walk(candidate)
    )
    runtime = named(named(candidate, "TerminalProjectionScan"), "__call__")
    function_calls = [ast.unparse(n.func) for n in ast.walk(runtime) if isinstance(n, ast.Call)]
    assert not any(c.startswith("torch.") or c.endswith((".from_torch", ".to_torch")) for c in function_calls)
    code = (
        "# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC\n# SPDX-License-Identifier: Apache-2.0\n"
        + ast.unparse(candidate)
        + "\n"
    )
    compile(code, "<terminal-scan-static-candidate>", "exec")
    assert "ttnn" not in sys.modules and "torch" not in sys.modules
    report = dict(
        classification="Standalone decode-only terminal projection scan diagnostic; integrated accuracy/performance unmeasured",
        mode=mode,
        base=str(base),
        base_sha256=base_sha,
        helper=str(HELPER),
        helper_sha256=raw_sha256(HELPER),
        helper_historical_sha256=HELPER_SHA,
        router_helper=str(ROUTER),
        router_helper_sha256=raw_sha256(ROUTER),
        router_helper_historical_sha256=ROUTER_SHA,
        component_evidence=str(evidence_path),
        component_evidence_sha256=sha(evidence_path),
        fixture_cases=32,
        component_timing=evidence["timing"],
        full_module_restoration=True,
        full_attention_v5_guard_ast_sha256=full_guards,
        changed_existing_methods={
            name: dict(
                before=digest(named(named(original, "FusedDecoder"), name)), after=digest(named(candidate_cls, name))
            )
            for name in ("from_state_dict", "linear", "_linear_attention", "_moe")
        },
        changed_scan_blocks=3,
        changed_call_sites=4,
        runtime_helper_alpha_equivalence=True,
        runtime_host_audit="TTNN generic_op only; constants literal; no test imports/dynamic code/host tensor conversions",
        gating="Explicit terminal_scan=False default; attention/output/out-projection pass decode and cache_boundary; router passes ordered_router derived from decode and cache_boundary",
        preserved="All prefill including public S=1/K512 and internal tails/K1024; all full-kind V5 branches; native product multiplication/block reduction/dtypes/layouts; all cache arithmetic",
        kernel_sha256={k: hashlib.sha256(v.encode()).hexdigest() for k, v in strings.items()},
        snapshot_sha256=hashlib.sha256(code.encode()).hexdigest(),
        device_executed=False,
        acceptance_eligible=False,
    )
    return code, report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=E)
    args = p.parse_args()
    embedded, strings, evidence, evidence_path = helper_source()
    outputs = []
    for mode in BASES:
        output = args.output_dir / f"v6_corrected_fma_full_v5_{mode}.py.snapshot"
        report_path = args.output_dir / f"v6_corrected_fma_full_v5_{mode}.provenance.json"
        builder_path = Path(str(output) + ".builder.py.snapshot")
        assert not any(p.exists() for p in (output, report_path, builder_path)), "Preserve frozen artifacts"
        code, report = build(mode, embedded, strings, evidence, evidence_path)
        outputs.append((output, report_path, builder_path, code, report))
    for output, report_path, builder_path, code, report in outputs:
        output.write_text(code)
        builder_path.write_bytes(Path(__file__).read_bytes())
        kernels = {}
        for name, source in strings.items():
            path = Path(str(output) + f".{name.lower()}.cpp.snapshot")
            assert not path.exists()
            path.write_text(source)
            kernels[name] = str(path)
        report.update(
            snapshot=str(output),
            builder_source=str(Path(__file__).resolve()),
            builder_sha256=sha(__file__),
            builder_snapshot=str(builder_path),
            kernel_snapshots=kernels,
        )
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print("TERMINAL_SCAN_STATIC_BUILD_PASS", output.name, sha(output))


if __name__ == "__main__":
    main()
