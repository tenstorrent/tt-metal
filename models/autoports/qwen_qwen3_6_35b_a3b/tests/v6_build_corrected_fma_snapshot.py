# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Embed the measured corrected-FMA adapter into frozen decoder snapshots.

AST/code generation is builder-only. Emitted decoder runtime has static Python
classes and a literal compute-source string, without test imports or exec.
"""

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
BASES = {
    "native": (
        "v6_inline_router_silu_chunk64_output_mkl_native_norm_v2.py.snapshot",
        "1446f3bbb5d00aa69976be0021b534d895b48fc0824d5120d2db67738ec7f2fa",
    ),
    "explicit": (
        "v6_inline_router_silu_chunk64_output_mkl_explicit_norm_v2.py.snapshot",
        "d57f87095b9c3d8705386b1c7c2f54d9541f3389bb33d6bf9d1821e7f6fd507f",
    ),
}
HELPERS = {
    "v6_corrected_fma_matmul_probe.py": "facef89f3f760fcd86c54e31e63ac938657f2ce02c639715699eb3c94ab90442",
    "v6_fma_sfpu_corrected_probe.py": "185217ed00091eddcdf15db9494071901edaa472d3935b986b4b512834f6832f",
    "v6_fma_sfpu_primitive_probe.py": "8938937a8bdf385a2c908242f296a745bd0ce498dad5b834a93759cd092ca3bc",
    "v6_chunk64_corrected_fma_output_probe.py": "8568bee017b255a738b1034bb97e1fa4ff975be734307c4a1e6a93c7a594906c",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(node):
    return hashlib.sha256(ast.dump(node).encode()).hexdigest()


def named(scope, name):
    return next(n for n in scope.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)


def build(norm, evidence):
    base_name, base_hash = BASES[norm]
    base_path = E / base_name
    assert sha(base_path) == base_hash
    for name, expected in HELPERS.items():
        assert verify_source(ROOT / "tests" / name, expected), name
    measured = json.loads(evidence.read_text())
    assert measured["status"] == "completed_diagnostic" and measured["device_closed"]
    for row in measured["cases"]:
        for key in ("device_output_bf16_vs_hf", "device_output_fp32_vs_hf", "device_state_vs_hf"):
            assert row[key]["fp32_code_differences"] == row[key]["bf16_code_differences"] == 0, (row["length"], key)

    adapter_tree = ast.parse((ROOT / "tests/v6_corrected_fma_matmul_probe.py").read_text())
    primitive_tree = ast.parse((ROOT / "tests/v6_fma_sfpu_corrected_probe.py").read_text())
    # Execute only pure source-generation definitions at build time; neither
    # source module is imported and Torch/TTNN never enter this process.
    graph = copy.deepcopy(named(primitive_tree, "Graph"))
    source_fn = copy.deepcopy(named(adapter_tree, "adapter_source"))
    scope = {}
    exec(compile(ast.Module(body=[graph, source_fn], type_ignores=[]), "<builder-source-generation>", "exec"), scope)
    compute_source = scope["adapter_source"]()
    arithmetic_cpp = scope["Graph"]().build().cpp()
    compute_hash = hashlib.sha256(compute_source.encode()).hexdigest()
    arithmetic_hash = hashlib.sha256(arithmetic_cpp.encode()).hexdigest()
    assert compute_hash == measured["corrected_fma_compute_sha256"]
    assert arithmetic_hash == measured["corrected_fma_arithmetic_cpp_sha256"]

    tree = ast.parse(base_path.read_text())
    before = copy.deepcopy(tree)
    state = named(tree, "Chunk64StateGraph")
    named(state, "__init__").body += ast.parse(
        "self.corrected_fma = CorrectedFma(ttnn, mesh)\n" "self.corrected_mm = CorrectedMatmul(self.corrected_fma)"
    ).body
    named(state, "mm").body = ast.parse("return self.corrected_mm(a, b)").body
    fma = copy.deepcopy(named(adapter_tree, "CorrectedFma"))
    matmul = copy.deepcopy(named(adapter_tree, "CorrectedMatmul"))
    fma_original = copy.deepcopy(fma)
    changed = []
    for node in ast.walk(named(fma, "__init__")):
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "self.source":
            assert ast.unparse(node.value) == "adapter_source()"
            changed.append((node, node.value))
            node.value = ast.Name(id="CORRECTED_FMA_COMPUTE_SOURCE", ctx=ast.Load())
    assert len(changed) == 1
    fma_embedded_hash = digest(fma)
    for node, original in changed:
        node.value = original
    assert ast.dump(fma) == ast.dump(fma_original)
    for node, _ in changed:
        node.value = ast.Name(id="CORRECTED_FMA_COMPUTE_SOURCE", ctx=ast.Load())
    assert digest(matmul) == digest(named(adapter_tree, "CorrectedMatmul"))
    tree.body += [
        ast.Import(names=[ast.alias(name="math")]),
        ast.Assign(
            targets=[ast.Name(id="CORRECTED_FMA_COMPUTE_SOURCE", ctx=ast.Store())],
            value=ast.Constant(value=compute_source),
        ),
        fma,
        matmul,
    ]
    modified = {}
    for old_cls in [n for n in before.body if isinstance(n, ast.ClassDef)]:
        new_cls = named(tree, old_cls.name)
        for old in [n for n in old_cls.body if isinstance(n, ast.FunctionDef)]:
            new = named(new_cls, old.name)
            if digest(old) != digest(new):
                modified[f"{old_cls.name}.{old.name}"] = dict(before=digest(old), after=digest(new))
    assert set(modified) == {"Chunk64StateGraph.__init__", "Chunk64StateGraph.mm"}
    for cls in (fma, matmul):
        for node in ast.walk(cls):
            if isinstance(node, ast.Call):
                call = ast.unparse(node.func)
                assert not call.startswith("torch.") and not call.endswith((".from_torch", ".to_torch")), call
    assert not any(isinstance(n, (ast.Import, ast.ImportFrom)) and "tests" in ast.unparse(n) for n in ast.walk(tree))
    assert not any(
        isinstance(n, ast.Call) and ast.unparse(n.func) in ("exec", "compile", "eval", "adapter_source")
        for n in ast.walk(tree)
    )
    ast.fix_missing_locations(tree)
    source = (
        "# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC\n# SPDX-License-Identifier: Apache-2.0\n"
        + ast.unparse(tree)
        + "\n"
    )
    compile(source, "<standalone-corrected-fma-decoder>", "exec")
    assert "torch" not in sys.modules and "ttnn" not in sys.modules
    return (
        source,
        compute_source,
        dict(
            base_source=str(base_path),
            base_sha256=base_hash,
            helper_sources={str(ROOT / "tests" / name): raw_sha256(ROOT / "tests" / name) for name in HELPERS},
            helper_sources_historical_sha256={str(ROOT / "tests" / name): value for name, value in HELPERS.items()},
            component_result=str(evidence),
            component_result_sha256=sha(evidence),
            component_lengths=[row["length"] for row in measured["cases"]],
            component_claim="Full output/state FP32 and BF16 codes exact on recorded component cases only",
            modified_existing_methods=modified,
            corrected_fma_compute_sha256=compute_hash,
            corrected_fma_arithmetic_cpp_sha256=arithmetic_hash,
            embedded_corrected_fma_ast_sha256=fma_embedded_hash,
            corrected_fma_ast_restoration="Only setup source-generation call replaced by static string constant; restored AST equals original",
            unchanged_corrected_matmul_ast_sha256=digest(matmul),
            norm_policy=norm,
            public_singleton_policy="unchanged baseline recurrence at every start",
            runtime_audit="Static embedded classes and compute string; no test imports/dynamic exec/compile/eval or corrected-adapter Torch/host conversions",
            mathematical_fallback="none: every chunk64 mm uses corrected_fma increasing-K recurrence",
            precision_limitation="Bounded measured FMA correction; no universal subnormal claim; full decoder stress remains required",
            performance_claim="none; diagnostic per-K dispatch and broadcasting remain",
            snapshot_sha256=hashlib.sha256(source.encode()).hexdigest(),
            acceptance_eligible=False,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--norm", choices=list(BASES), required=True)
    parser.add_argument("--component-result", type=Path, default=E / "v6_chunk64_corrected_fma_output_device.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    compute_path = Path(str(args.output) + ".compute.cpp.snapshot")
    builder_snapshot = Path(str(args.output) + ".builder.py.snapshot")
    assert not any(
        path.exists() for path in (args.output, args.report, compute_path, builder_snapshot)
    ), "Preserve evidence"
    source, compute, report = build(args.norm, args.component_result)
    args.output.write_text(source)
    compute_path.write_text(compute)
    builder_snapshot.write_bytes(Path(__file__).read_bytes())
    report.update(
        snapshot=str(args.output),
        compute_snapshot=str(compute_path),
        builder_source=str(Path(__file__).resolve()),
        builder_sha256=sha(__file__),
        builder_snapshot=str(builder_snapshot),
    )
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print("CORRECTED_FMA_STATIC_BUILD_PASS", args.norm, sha(args.output))


if __name__ == "__main__":
    main()
