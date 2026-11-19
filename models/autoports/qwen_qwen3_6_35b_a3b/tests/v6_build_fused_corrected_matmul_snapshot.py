# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Static single-dispatch GEMM integration from frozen streaming/norm-tail base."""
import argparse
import ast
import copy
import hashlib
import json
import sys
from pathlib import Path

from models.autoports.qwen_qwen3_6_35b_a3b.tests import v6_fused_corrected_matmul_probe as helper
from models.autoports.qwen_qwen3_6_35b_a3b.tests.frozen_source_compatibility import raw_sha256, verify_source

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "doc/fused_decoder/endurance"
BASE = E / "v6_corrected_fma_full_v5_streaming_scan_norm_tail_integration.py.snapshot"
BASE_SHA = "700c10db90b23fac1d40427c9bbcd82a6924c7704699d54768a0160c2b641e05"
HELPER_SHA = "845c959110a713f808d1475eb22ded5ffa4e6ddd8209f02170fb90ed02b353dc"
WRAPPER = ROOT / "tests/v6_chunk64_fused_corrected_output_probe.py"
WRAPPER_SHA = "ae0e11d3c683be1cede6ee18bea429c1b2cb6d06a7da6ac1917fd24c3d75713d"
RENAMES = {
    "READER": "FUSED_CORRECTED_MM_READER",
    "WRITER": "FUSED_CORRECTED_MM_WRITER",
    "shape_plan": "fused_corrected_mm_shape_plan",
    "schedule": "fused_corrected_mm_schedule",
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


def gates(sources):
    matrix = E / "v6_fused_corrected_matmul_all_device.json"
    jm = json.loads(matrix.read_text())
    assert jm["status"] == "component_adapter_exact" and jm["device_closed"]
    assert jm["source_sha256"] == HELPER_SHA and len(jm["cases"]) == 11
    assert all(c["vs_frozen_adapter"]["fp32_bits_different"] == 0 for c in jm["cases"])
    paths = [matrix]
    cases = []
    for filename, state, lengths in [
        ("v6_chunk64_fused_corrected_output_zero_device.json", "zero", [128]),
        ("v6_chunk64_fused_corrected_output_nonzero_device.json", "prefix64", [127, 65, 1]),
    ]:
        path = E / filename
        j = json.loads(path.read_text())
        paths.append(path)
        assert j["status"] == "completed_diagnostic" and j["device_closed"]
        assert j["variant"] == "single_dispatch_corrected_fma" and j["fused_factory_installed"]
        assert j["fused_wrapper_sha256"] == WRAPPER_SHA
        assert j["initial_state"] == state and [c["length"] for c in j["cases"]] == lengths
        for name, source in sources.items():
            want = hashlib.sha256(source.encode()).hexdigest()
            assert want == jm[name + "_sha256"] == j["fused_matmul_kernels"][name]["sha256"]
        for c in j["cases"]:
            exact = {k: c[k] for k in ("device_output_bf16_vs_hf", "device_output_fp32_vs_hf", "device_state_vs_hf")}
            assert all(v["fp32_code_differences"] == v["bf16_code_differences"] == 0 for v in exact.values())
            cases.append(
                dict(initial_state=state, length=c["length"], internal_padding=c["internal_padding"], comparison=exact)
            )
    return dict(component_evidence={str(path): sha(path) for path in paths}, matrix_cases=11, propagated_cases=cases)


def build():
    assert (
        verify_source(BASE, BASE_SHA)
        and verify_source(helper.__file__, HELPER_SHA)
        and verify_source(WRAPPER, WRAPPER_SHA)
    )
    sources = {"reader": helper.READER, "compute": helper.compute_source(), "writer": helper.WRITER}
    evidence = gates(sources)
    checks = helper.source_checks()
    helper_tree = ast.parse(Path(helper.__file__).read_text())
    original_helpers = [
        copy.deepcopy(named(helper_tree, name)) for name in ("shape_plan", "schedule", "FusedCorrectedMatmul")
    ]
    embedded = [Rename(RENAMES).visit(copy.deepcopy(n)) for n in original_helpers]
    cls = embedded[-1]
    setup = named(cls, "__init__")
    generated = next(n for n in setup.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "self.source")
    assert ast.unparse(generated.value) == "compute_source()"
    generated.value = ast.Name("FUSED_CORRECTED_MM_COMPUTE", ast.Load())
    # Restore only global renames and setup literal, proving measured helper body identity.
    restored = [Rename({v: k for k, v in RENAMES.items()}).visit(copy.deepcopy(n)) for n in embedded]
    restored_setup = named(restored[-1], "__init__")
    next(
        n for n in restored_setup.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "self.source"
    ).value = ast.Call(ast.Name("compute_source", ast.Load()), [], [])
    assert ast.dump(ast.Module(restored, [])) == ast.dump(ast.Module(original_helpers, []))
    literals = [
        ast.Assign([ast.Name("FUSED_CORRECTED_MM_" + name.upper(), ast.Store())], ast.Constant(source))
        for name, source in sources.items()
    ]
    tree = ast.parse(BASE.read_text())
    original = copy.deepcopy(tree)
    graph = named(tree, "Chunk64StateGraph")
    init = named(graph, "__init__")
    start = next(
        i
        for i, n in enumerate(init.body)
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "self.corrected_fma"
    )
    old = copy.deepcopy(init.body[start : start + 2])
    assert [ast.unparse(n) for n in old] == [
        "self.corrected_fma = CorrectedFma(ttnn, mesh)",
        "self.corrected_mm = CorrectedMatmul(self.corrected_fma)",
    ]
    init.body[start : start + 2] = ast.parse("self.corrected_mm = FusedCorrectedMatmul(ttnn, mesh, max_cores=64)").body
    removed = []
    for i, n in enumerate(tree.body):
        if (
            isinstance(n, ast.ClassDef)
            and n.name in ("CorrectedFma", "CorrectedMatmul")
            or isinstance(n, ast.Assign)
            and ast.unparse(n.targets[0]) == "CORRECTED_FMA_COMPUTE_SOURCE"
        ):
            removed.append((i, copy.deepcopy(n)))
    assert len(removed) == 3
    for i, _ in reversed(removed):
        tree.body.pop(i)
    # Old helpers and retained input references are now unreachable and removed.
    assert not any(
        isinstance(n, ast.Name) and n.id in ("CorrectedFma", "CorrectedMatmul", "CORRECTED_FMA_COMPUTE_SOURCE")
        for n in ast.walk(tree)
    )
    assert not any(isinstance(n, ast.Attribute) and n.attr == "corrected_fma" for n in ast.walk(tree))
    candidate = copy.deepcopy(tree)
    candidate.body += literals + embedded
    ast.fix_missing_locations(candidate)
    init.body[start : start + 1] = old
    for i, n in removed:
        tree.body.insert(i, n)
    assert ast.dump(tree) == ast.dump(original), "All other base AST must remain identical"
    assert digest(named(candidate, "FusedDecoder")) == digest(named(original, "FusedDecoder"))
    assert digest(named(candidate, "Chunk64OutputGraph")) == digest(named(original, "Chunk64OutputGraph"))
    assert digest(named(named(candidate, "Chunk64StateGraph"), "mm")) == digest(
        named(named(original, "Chunk64StateGraph"), "mm")
    )
    assert not any(
        isinstance(n, (ast.Import, ast.ImportFrom)) and "tests" in ast.unparse(n) for n in ast.walk(candidate)
    )
    assert not any(
        isinstance(n, ast.Call)
        and ast.unparse(n.func) in ("exec", "eval", "compile", "compute_source", "arithmetic", "adapter_source")
        for n in ast.walk(candidate)
    )
    runtime = named(named(candidate, "FusedCorrectedMatmul"), "__call__")
    calls = [ast.unparse(n.func) for n in ast.walk(runtime) if isinstance(n, ast.Call)]
    assert calls.count("t.generic_op") == 1
    assert not any(c.startswith("torch.") or c.endswith((".from_torch", ".to_torch")) for c in calls)
    assert "last_inputs" not in ast.unparse(named(candidate, "FusedCorrectedMatmul"))
    code = (
        "# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC\n# SPDX-License-Identifier: Apache-2.0\n"
        + ast.unparse(candidate)
        + "\n"
    )
    compile(code, "<static-fused-matmul-candidate>", "exec")
    assert "ttnn" not in sys.modules and "torch" not in sys.modules
    return (
        code,
        sources,
        dict(
            classification="Static diagnostic integration after exact component output/state gates; integrated correctness/latency pending",
            base=str(BASE),
            base_sha256=BASE_SHA,
            helper=str(Path(helper.__file__).resolve()),
            helper_sha256=raw_sha256(helper.__file__),
            helper_historical_sha256=HELPER_SHA,
            wrapper=str(WRAPPER),
            wrapper_sha256=raw_sha256(WRAPPER),
            wrapper_historical_sha256=WRAPPER_SHA,
            **evidence,
            source_checks=checks,
            full_module_restoration=True,
            helper_alpha_equivalence=True,
            helper_setup_delta="Only source generation replaced by literal compute source",
            removed_unreachable_nodes=[
                dict(
                    index=i,
                    name=getattr(n, "name", ast.unparse(n.targets[0]) if isinstance(n, ast.Assign) else ""),
                    ast_sha256=digest(n),
                )
                for i, n in removed
            ],
            removed_retained_inputs="Old CorrectedFma object, its last_inputs tuple and shape/WHERE caches no longer constructed; new last_plan holds only metadata",
            unchanged_fused_decoder_ast_sha256=digest(named(candidate, "FusedDecoder")),
            unchanged_output_graph_ast_sha256=digest(named(candidate, "Chunk64OutputGraph")),
            changed_state_init=dict(
                before=digest(named(named(original, "Chunk64StateGraph"), "__init__")),
                after=digest(named(named(candidate, "Chunk64StateGraph"), "__init__")),
            ),
            runtime="One device generic_op per matrix; frozen increasing-K corrected FMA and FP32 feedback pack/unpack; no runtime source generation/test imports/Torch conversion",
            preserved="All decode/public singleton prefill/full-attention V5 paths, streaming projection scans/norm tail, cache/padding/output/state equations and context contract",
            snapshot_sha256=hashlib.sha256(code.encode()).hexdigest(),
            device_executed=False,
            acceptance_eligible=False,
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()
    builder = Path(str(args.output) + ".builder.py.snapshot")
    paths = [args.output, args.report, builder] + [
        Path(str(args.output) + f".{name}.cpp.snapshot") for name in ("reader", "compute", "writer")
    ]
    assert not any(path.exists() for path in paths), "Preserve evidence"
    code, sources, report = build()
    args.output.write_text(code)
    builder.write_bytes(Path(__file__).read_bytes())
    kernels = {}
    for name, source in sources.items():
        path = Path(str(args.output) + f".{name}.cpp.snapshot")
        path.write_text(source)
        kernels[name] = dict(path=str(path), sha256=sha(path))
    report.update(
        snapshot=str(args.output),
        builder_source=str(Path(__file__).resolve()),
        builder_snapshot=str(builder),
        builder_sha256=sha(__file__),
        kernels=kernels,
    )
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print("STATIC_FUSED_MATMUL_BUILD_PASS", sha(args.output))


if __name__ == "__main__":
    main()
