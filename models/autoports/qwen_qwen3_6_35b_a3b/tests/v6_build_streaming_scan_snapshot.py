# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Build a static decode-only streaming projection snapshot with exact base restoration."""

import argparse
import ast
import copy
import hashlib
import json
import sys
from pathlib import Path

from models.autoports.qwen_qwen3_6_35b_a3b.tests.frozen_source_compatibility import raw_sha256, verify_source

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "doc/fused_decoder/endurance"
BASE = E / "v6_corrected_fma_full_v5_terminal_scan_norm_tail.py.snapshot"
BASE_SHA = "99d2680b696bfbba57ceca9bd6597ae06daa217c062bf9289a25c9a10e53e5bd"
HELPER = ROOT / "tests/v6_streaming_projection_scan_probe.py"
HELPER_SHA = "b555f5caae383fb2cf1976860e92b9af3f1f2e2e39ec560e4c43d4d05bb23354"
TERMINAL = ROOT / "tests/v6_terminal_projection_scan_probe.py"
TERMINAL_SHA = "8c9eeb05d3cd388fdffe581c459b2178fd2c526c6117f630a779154d39736b64"
ROUTER = ROOT / "tests/v6_ordered_router_fusion_probe.py"
ROUTER_SHA = "067595f30f1b7e29c65795748dddd9b594725d40190c8541c0863a9cf8dd5314"
EVIDENCE = E / "v6_streaming_projection_scan_device.json"
RENAMES = {
    "READER": "STREAMING_SCAN_READER",
    "COMPUTE": "STREAMING_SCAN_COMPUTE",
    "WRITER": "STREAMING_SCAN_WRITER",
    "PRODUCT_WRITER": "STREAMING_SCAN_PRODUCT_WRITER",
    "schedule": "terminal_projection_schedule",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(node):
    return hashlib.sha256(ast.dump(node).encode()).hexdigest()


def named(scope, name):
    return next(n for n in scope.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)


class Rename(ast.NodeTransformer):
    def __init__(self, mapping):
        self.mapping = mapping

    def visit_Name(self, node):
        node.id = self.mapping.get(node.id, node.id)
        return node


def validate_evidence():
    for path, expected in [(BASE, BASE_SHA), (HELPER, HELPER_SHA), (TERMINAL, TERMINAL_SHA), (ROUTER, ROUTER_SHA)]:
        assert verify_source(path, expected), path
    evidence = json.loads(EVIDENCE.read_text())
    assert evidence["status"] == "component_pass" and evidence["device_closed"]
    assert evidence["script_sha256"] == HELPER_SHA
    assert evidence["frozen_terminal_sha256"] == TERMINAL_SHA and evidence["frozen_router_sha256"] == ROUTER_SHA
    assert evidence["fixture_cases"] == len(evidence["cases"]) == 32
    singletons = [c for c in evidence["cases"] if "partial_fp32" in c]
    assert len(singletons) == 30
    assert all(c["partial_fp32"]["exact"] and c["products_fp32"]["exact"] for c in singletons)
    assert all(all(o["exact"] for o in c["outputs"].values()) for c in evidence["cases"])
    assert evidence["multiple_leading_axes_fp32"]["exact"] and evidence["multiple_leading_axes_products_fp32"]["exact"]
    assert len(evidence["timing"]) == 16
    assert all(t["after_100_replays"]["exact"] for t in evidence["timing"])
    return evidence


def embed(original):
    helper = ast.parse(HELPER.read_text())
    strings = {
        n.targets[0].id: ast.literal_eval(n.value)
        for n in helper.body
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id in ("READER", "COMPUTE", "PRODUCT_WRITER")
    }
    strings["WRITER"] = next(
        ast.literal_eval(n.value)
        for n in original.body
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == "TERMINAL_SCAN_WRITER"
    )
    assert len(strings) == 4
    kernel_evidence = {}
    for name, source in strings.items():
        path = Path(str(EVIDENCE) + f".{name.lower()}.snapshot")
        assert path.read_text() == source, (name, path)
        kernel_evidence[name] = dict(path=str(path), sha256=sha(path))
    # Reuse the already-embedded scheduling function only after proving its exact body.
    schedule = copy.deepcopy(named(ast.parse(TERMINAL.read_text()), "schedule"))
    schedule.name = "terminal_projection_schedule"
    assert ast.dump(schedule) == ast.dump(named(original, "terminal_projection_schedule"))
    runtime = copy.deepcopy(named(helper, "StreamingProjectionScan"))
    renamed = Rename(RENAMES).visit(copy.deepcopy(runtime))
    restored = Rename({v: k for k, v in RENAMES.items()}).visit(copy.deepcopy(renamed))
    assert ast.dump(restored) == ast.dump(runtime)
    literals = [ast.Assign([ast.Name(RENAMES[k], ast.Store())], ast.Constant(v)) for k, v in strings.items()]
    return literals + [renamed], strings, kernel_evidence


def build():
    evidence = validate_evidence()
    original = ast.parse(BASE.read_text())
    embedded, strings, kernel_evidence = embed(original)
    candidate = copy.deepcopy(original)
    cls = named(candidate, "FusedDecoder")
    setup = named(cls, "from_state_dict")
    assignments = [
        n
        for n in ast.walk(setup)
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "self.terminal_projection_scan"
    ]
    assert len(assignments) == 1
    setup_assignment = assignments[0]
    original_assignment = copy.deepcopy(setup_assignment)
    setup_assignment.targets[0].attr = "streaming_projection_scan"
    assert ast.unparse(setup_assignment.value) == "TerminalProjectionScan(ttnn, mesh_device)"
    setup_assignment.value.func.id = "StreamingProjectionScan"
    linear = named(cls, "linear")
    restorations = []
    for branch in linear.body[1:4]:
        assert isinstance(branch, ast.If)
        i = next(
            i
            for i, n in enumerate(branch.body)
            if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "products"
        )
        product, guard = branch.body[i : i + 2]
        assert isinstance(guard, ast.If) and ast.unparse(guard.test) == "terminal_scan"
        assert ast.unparse(guard.body[0]) == "partial = self.terminal_projection_scan(products)"
        assert ast.unparse(guard.body[1]) == "del products" and len(guard.body) == 2
        old = copy.deepcopy(branch.body[i : i + 2])
        replacement = ast.If(
            copy.deepcopy(guard.test),
            ast.parse("partial = self.streaming_projection_scan(weight, coefficients)").body,
            [copy.deepcopy(product)] + copy.deepcopy(guard.orelse),
        )
        # False branch is precisely the previously executed native multiply/cumsum path.
        assert ast.dump(ast.Module(replacement.orelse, [])) == ast.dump(ast.Module([product] + guard.orelse, []))
        branch.body[i : i + 2] = [replacement]
        restorations.append((branch, i, old))
    assert len(restorations) == 3
    modified = copy.deepcopy(candidate)
    setup_assignment.targets = copy.deepcopy(original_assignment.targets)
    setup_assignment.value = copy.deepcopy(original_assignment.value)
    for branch, i, old in restorations:
        branch.body[i : i + 1] = old
    assert ast.dump(candidate) == ast.dump(original), "Exact base module restoration failed"
    candidate = modified
    old_cls, cls = named(original, "FusedDecoder"), named(candidate, "FusedDecoder")
    changed = [
        n.name for n in cls.body if isinstance(n, ast.FunctionDef) and ast.dump(n) != ast.dump(named(old_cls, n.name))
    ]
    assert changed == ["from_state_dict", "linear"]
    old_linear, new_linear = named(old_cls, "linear"), named(cls, "linear")
    assert ast.dump(old_linear.args) == ast.dump(new_linear.args)
    assert ast.unparse(new_linear.args.kwonlyargs[-1]) == "terminal_scan"
    assert ast.literal_eval(new_linear.args.kw_defaults[-1]) is False
    guard_checks = {}
    for method in ("from_state_dict", "linear", "norm", "_moe", "_forward", "prepare_prefill", "prefill_forward"):
        before = next(n for n in named(old_cls, method).body if isinstance(n, ast.If))
        after = next(n for n in named(cls, method).body if isinstance(n, ast.If))
        assert ast.dump(before) == ast.dump(after), method
        guard_checks[method] = digest(after)
    flag_calls = []
    for method in ("_linear_attention", "_moe"):
        assert ast.dump(named(old_cls, method)) == ast.dump(named(cls, method))
        for call in ast.walk(named(cls, method)):
            if isinstance(call, ast.Call) and ast.unparse(call.func) == "self.linear":
                flags = [ast.unparse(k.value) for k in call.keywords if k.arg == "terminal_scan"]
                if flags:
                    flag_calls.append(dict(method=method, key=ast.literal_eval(call.args[1]), gate=flags[0]))
    assert [c["gate"] for c in flag_calls] == ["decode and cache_boundary"] * 3 + ["ordered_router"]
    forward = named(cls, "_forward")
    assert [
        ast.unparse(k.value)
        for n in ast.walk(forward)
        if isinstance(n, ast.Call) and ast.unparse(n.func) == "self._moe"
        for k in n.keywords
        if k.arg == "ordered_router"
    ] == ["decode and cache_boundary"]
    candidate.body += embedded
    ast.fix_missing_locations(candidate)
    assert not any(
        isinstance(n, (ast.Import, ast.ImportFrom)) and "tests" in ast.unparse(n) for n in ast.walk(candidate)
    )
    assert not any(
        isinstance(n, ast.Call) and ast.unparse(n.func) in ("exec", "eval", "compile") for n in ast.walk(candidate)
    )
    runtime = named(named(candidate, "StreamingProjectionScan"), "__call__")
    runtime_calls = [ast.unparse(n.func) for n in ast.walk(runtime) if isinstance(n, ast.Call)]
    assert not any(c.startswith("torch.") or c.endswith((".from_torch", ".to_torch")) for c in runtime_calls)
    # Existing static class remains as inert source. Its previous setup instance is replaced.
    assert ast.dump(named(candidate, "TerminalProjectionScan")) == ast.dump(named(original, "TerminalProjectionScan"))
    code = (
        "# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC\n# SPDX-License-Identifier: Apache-2.0\n"
        + ast.unparse(candidate)
        + "\n"
    )
    compile(code, "<static-streaming-projection-integration>", "exec")
    assert "ttnn" not in sys.modules and "torch" not in sys.modules
    report = dict(
        classification="Standalone decode-only streaming projection candidate; integrated correctness/performance unmeasured",
        base=str(BASE),
        base_sha256=BASE_SHA,
        helper=str(HELPER),
        helper_sha256=raw_sha256(HELPER),
        helper_historical_sha256=HELPER_SHA,
        terminal_helper=str(TERMINAL),
        terminal_helper_sha256=raw_sha256(TERMINAL),
        terminal_helper_historical_sha256=TERMINAL_SHA,
        router_helper=str(ROUTER),
        router_helper_sha256=raw_sha256(ROUTER),
        router_helper_historical_sha256=ROUTER_SHA,
        component_evidence=str(EVIDENCE),
        component_evidence_sha256=sha(EVIDENCE),
        component_cases=32,
        component_timing=evidence["timing"],
        component_kernel_evidence=kernel_evidence,
        full_module_restoration=True,
        changed_existing_methods=changed,
        changed_scan_blocks=3,
        from_state_dict_change="Replace only the existing linear-kind terminal helper instance with the streaming helper",
        decode_calls_unchanged=flag_calls,
        linear_signature_and_false_default_unchanged=True,
        prefill_false_branches_exact=True,
        full_attention_v5_guard_ast_sha256=guard_checks,
        helper_alpha_equivalence=True,
        existing_schedule_exact=True,
        preserved="Public singleton/multirow prefill and internal K1024 tails; full-attention V5; all cache/norm/state arithmetic; output reductions/dtypes/layouts",
        runtime_audit="Literal static kernels and class; no test imports, dynamic AST/exec, Torch or tensor host conversions in runtime helper",
        kernel_sha256={k: hashlib.sha256(v.encode()).hexdigest() for k, v in strings.items()},
        snapshot_sha256=hashlib.sha256(code.encode()).hexdigest(),
        device_executed=False,
        acceptance_eligible=False,
    )
    return code, report, strings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=E)
    args = parser.parse_args()
    stem = args.output_dir / "v6_corrected_fma_full_v5_streaming_scan_norm_tail_integration"
    snapshot = Path(str(stem) + ".py.snapshot")
    provenance = Path(str(stem) + ".provenance.json")
    builder = Path(str(stem) + ".builder.py.snapshot")
    assert not any(p.exists() for p in (snapshot, provenance, builder)), "Preserve existing evidence"
    code, report, strings = build()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    kernel_paths = {k: Path(str(stem) + f".{k.lower()}.cpp.snapshot") for k in strings}
    assert not any(p.exists() for p in kernel_paths.values())
    snapshot.write_text(code)
    builder.write_bytes(Path(__file__).read_bytes())
    for k, path in kernel_paths.items():
        path.write_text(strings[k])
    report.update(
        snapshot=str(snapshot),
        builder_source=str(Path(__file__).resolve()),
        builder_sha256=sha(__file__),
        builder_snapshot=str(builder),
        kernel_snapshots={k: str(p) for k, p in kernel_paths.items()},
    )
    provenance.write_text(json.dumps(report, indent=2) + "\n")
    print("STREAMING_SCAN_STATIC_INTEGRATION_BUILD_PASS", snapshot.name, sha(snapshot))


if __name__ == "__main__":
    main()
