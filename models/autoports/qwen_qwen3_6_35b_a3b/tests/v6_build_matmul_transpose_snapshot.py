# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Embed exact transpose-address matmul reader in a standalone 44f87 candidate."""

import argparse
import ast
import copy
import json
import subprocess
from pathlib import Path

from models.autoports.qwen_qwen3_6_35b_a3b.tests import v6_corrected_matmul_transpose_probe as component

E = component.E
BASE = E / "v6_transport_conv_norm_decay_lookup_integration.py.snapshot"
BASE_SHA = "44f87c04349836a4ad8b09c7a6c573fe3afe1e558af285e0482cce2c0fe5b2b3"
COMPONENT_SHA = "e48c3ba24c24be0099c7b45eb2fc0ffeb98a0c90d4181b015e110c8236b9e767"
BLACK = "/home/ttuser/.cache/pre-commit/repoefelr92y/py_env-python3/bin/black"
AUTOFLAKE = "/home/ttuser/.cache/pre-commit/repodje5jyzf/py_env-python3/bin/autoflake"
ISORT = "/home/ttuser/.cache/pre-commit/repodrn1fio9/py_env-python3/bin/isort"


def named(tree, name):
    return next(n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)


def digest(node):
    return component.digest(ast.dump(node))


def build(args):
    assert not args.output.exists() and not args.report.exists()
    assert component.sha(BASE) == BASE_SHA and component.sha(component.__file__) == COMPONENT_SHA
    baseline, candidate_scope, source_proof, _, candidate_class = component.source_bundle()
    original = ast.parse(BASE.read_text())
    previous = ast.parse(component.SNAPSHOT.read_text())
    # All addressed arithmetic and methods must be identical to measured803 component base.
    for name in component.NAMES:
        if name.startswith("FUSED_"):
            old = next(n for n in previous.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == name)
            new = next(n for n in original.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == name)
        else:
            old, new = named(previous, name), named(original, name)
        assert digest(old) == digest(new), name
    for name in ("Chunk64StateGraph", "Chunk64OutputGraph"):
        assert digest(named(named(previous, name), "__call__")) == digest(named(named(original, name), "__call__"))
    gate_paths = [
        E / "v6_corrected_matmul_transpose_device.json",
        E / "v6_corrected_matmul_transpose_matched_timing_device.json",
    ]
    gate, timing = [json.loads(p.read_text()) for p in gate_paths]
    assert gate["status"] == "pass" and gate["device_closed"] and gate["script_sha256"] == COMPONENT_SHA
    assert len(gate["device_cases"]) == 22 and all(c["output"]["exact"] for c in gate["device_cases"])
    assert timing["status"] == "matched_timing_exact" and timing["device_closed"]
    assert len(timing["cases"]) == 6
    assert all(all(v < 0 for v in row["paired_trace_candidate_minus_baseline_ms"]) for row in timing["cases"])
    edits = [
        ("self.mm(kb, t.transpose(k, -1, -2))", "self.transpose_mm(kb, k, transpose_b=True)"),
        ("self.mm(t.transpose(weighted_k, -1, -2), v_new)", "self.transpose_mm(weighted_k, v_new, transpose_a=True)"),
        ("self.mm(q, t.transpose(k, -1, -2))", "self.transpose_mm(q, k, transpose_b=True)"),
    ]

    class Replace(ast.NodeTransformer):
        def __init__(self, replacements):
            self.replacements = [
                (ast.parse(a, mode="eval").body, ast.parse(b, mode="eval").body) for a, b in replacements
            ]
            self.counts = [0] * len(replacements)

        def visit_Call(self, node):
            for index, (old, new) in enumerate(self.replacements):
                if ast.dump(node) == ast.dump(old):
                    self.counts[index] += 1
                    return copy.deepcopy(new)
            return self.generic_visit(node)

    change = Replace(edits)
    candidate = change.visit(copy.deepcopy(original))
    assert change.counts == [1, 1, 1]
    init = named(named(candidate, "Chunk64StateGraph"), "__init__")
    setup = ast.parse("self.transpose_mm = TransposeCorrectedMatmul(ttnn, mesh)").body[0]
    init.body.append(copy.deepcopy(setup))
    # Static helper and kernel only: no runtime imports or AST source generation.
    helper_source = component.replace_once(
        candidate_class, "kernel(FUSED_CORRECTED_MM_READER,", "kernel(TRANSPOSE_CORRECTED_MM_READER,"
    )
    helper = ast.parse(helper_source).body[0]
    inverse_helper = component.replace_once(
        helper_source, "kernel(TRANSPOSE_CORRECTED_MM_READER,", "kernel(FUSED_CORRECTED_MM_READER,"
    )
    assert ast.dump(ast.parse(inverse_helper)) == ast.dump(ast.parse(candidate_class))
    helper_init = named(helper, "__init__")
    assert [ast.unparse(n.func) for n in ast.walk(helper_init) if isinstance(n, ast.Call)] == [
        "mesh.compute_with_storage_grid_size"
    ]
    reader = ast.Assign(
        targets=[ast.Name(id="TRANSPOSE_CORRECTED_MM_READER", ctx=ast.Store())],
        value=ast.Constant(candidate_scope["FUSED_CORRECTED_MM_READER"]),
    )
    candidate.body.extend([reader, helper])
    candidate = ast.fix_missing_locations(candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC\n# SPDX-License-Identifier: Apache-2.0\n"
        + ast.unparse(candidate)
        + "\n"
    )
    for cmd in ([AUTOFLAKE, "--remove-all-unused-imports", "--in-place"], [ISORT], [BLACK, "--line-length", "120"]):
        subprocess.run(cmd + [str(args.output)], check=True, capture_output=True)
    emitted = ast.parse(args.output.read_text())
    assert digest(emitted) == digest(candidate), "Formatting must preserve exact AST"
    for cmd in (
        [AUTOFLAKE, "--remove-all-unused-imports", "--check"],
        [ISORT, "--check-only"],
        [BLACK, "--check", "--line-length", "120"],
    ):
        subprocess.run(cmd + [str(args.output)], check=True, capture_output=True)
    restored = copy.deepcopy(emitted)
    assert [digest(n) for n in restored.body[-2:]] == [digest(reader), digest(helper)]
    del restored.body[-2:]
    restored_init = named(named(restored, "Chunk64StateGraph"), "__init__")
    assert digest(restored_init.body[-1]) == digest(setup)
    del restored_init.body[-1]
    undo = Replace([(b, a) for a, b in edits])
    restored = undo.visit(restored)
    assert undo.counts == [1, 1, 1] and digest(restored) == digest(original)
    assert digest(named(emitted, "FusedDecoder")) == digest(named(original, "FusedDecoder"))
    assert digest(named(emitted, "FusedCorrectedMatmul")) == digest(named(original, "FusedCorrectedMatmul"))
    artifacts = {}
    for suffix, text in (
        ("reader.cpp.snapshot", candidate_scope["FUSED_CORRECTED_MM_READER"]),
        ("compute.cpp.snapshot", baseline["FUSED_CORRECTED_MM_COMPUTE"]),
        ("writer.cpp.snapshot", baseline["FUSED_CORRECTED_MM_WRITER"]),
        ("builder.py.snapshot", Path(__file__).read_text()),
    ):
        path = Path(str(args.output) + "." + suffix)
        assert not path.exists()
        path.write_text(text)
        artifacts[str(path)] = component.sha(path)
    report = dict(
        status="standalone_source_candidate_ready_not_selected",
        source_sha256=component.sha(__file__),
        base=str(BASE),
        base_sha256=BASE_SHA,
        output=str(args.output),
        output_sha256=component.sha(args.output),
        component_sha256=COMPONENT_SHA,
        source_proof=source_proof,
        gate_evidence={str(p): component.sha(p) for p in gate_paths},
        precise_call_substitutions=edits,
        full_module_inverse_ast_exact=True,
        formatting_ast_exact=True,
        all_pinned_hooks_pass=True,
        fused_decoder_ast_unchanged=True,
        original_corrected_mm_ast_unchanged=True,
        helper_ast_sha256=digest(helper),
        helper_init_ast_sha256=digest(helper_init),
        helper_kernel_rename_only=True,
        persistent_l1_added_bytes=0,
        persistent_dram_added_bytes=0,
        constructor_nonallocating=True,
        cb_bytes_per_core=40960,
        artifacts=artifacts,
        qualification="Requires fullchunk zero/nonzero/nonaligned, wholedecoder correctness and measured wholedecoder speed before selection",
        device_executed=False,
        production_mutated=False,
    )
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(report["status"], report["output_sha256"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
