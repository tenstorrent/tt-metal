# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Build a standalone candidate with only the chunk64 triangular loop fused.

Source-only builder. It embeds the measured literal kernels and runtime class,
checks exact restoration to the selected input AST, and applies pinned Black.
No production file, previous snapshot, decoder contract, or hardware is changed.
"""
import argparse
import ast
import copy
import hashlib
import json
import subprocess
from pathlib import Path

from models.autoports.qwen_qwen3_6_35b_a3b.tests import v6_triangular_recurrence_fusion_probe as component
from models.autoports.qwen_qwen3_6_35b_a3b.tests.frozen_source_compatibility import raw_sha256, verify_source

DEFAULT_BASE = component.E.parent / "selected_fused_decoder_v6.py.snapshot"
DEFAULT_SHA = "c71ea01a4b38458f0f53808d288d9a39ca5563a13cb5e97f71e701391670c3f2"
COMPONENT_SHA = "8229068839e3e06750353eb66fe720c45445e197b981d982dba1328ab5de4853"
BLACK = "/home/ttuser/.cache/pre-commit/repoefelr92y/py_env-python3/bin/black"
NAMES = dict(
    TriangularRecurrence="Chunk64TriangularRecurrence",
    READER="CHUNK64_TRIANGULAR_READER",
    COMPUTE="CHUNK64_TRIANGULAR_COMPUTE",
    EMPTY_WRITER="CHUNK64_TRIANGULAR_EMPTY_WRITER",
)


def digest(tree):
    return hashlib.sha256(ast.dump(tree).encode()).hexdigest()


def build(args):
    assert not args.output.exists() and not args.report.exists()
    assert component.sha(args.base) == args.base_sha
    assert verify_source(component.__file__, COMPONENT_SHA)
    original = ast.parse(args.base.read_text())
    candidate = copy.deepcopy(original)
    cls = next(n for n in candidate.body if isinstance(n, ast.ClassDef) and n.name == "Chunk64StateGraph")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    call = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__call__")
    original_init, original_call = copy.deepcopy(init), copy.deepcopy(call)
    init.body.extend(ast.parse("self.triangular_recurrence = Chunk64TriangularRecurrence(ttnn, mesh)").body)
    replacement = ast.parse('a = self.triangular_recurrence(a)\nif observer:\n observer(prefix + "A_row63",a)\n').body
    old = []

    class Replace(ast.NodeTransformer):
        def visit_For(self, node):
            if ast.unparse(node.target) == "i" and ast.unparse(node.iter) == "range(1, 64)":
                old.append(copy.deepcopy(node))
                return copy.deepcopy(replacement)
            return self.generic_visit(node)

    Replace().visit(call)
    assert len(old) == 1
    added = []

    class Rename(ast.NodeTransformer):
        def visit_Name(self, node):
            node.id = NAMES.get(node.id, node.id)
            return node

        def visit_ClassDef(self, node):
            node.name = NAMES.get(node.name, node.name)
            return self.generic_visit(node)

    source = ast.parse(Path(component.__file__).read_text())
    for node in source.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in ("READER", "COMPUTE", "EMPTY_WRITER")
        ):
            assert isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
            added.append(Rename().visit(copy.deepcopy(node)))
        if isinstance(node, ast.ClassDef) and node.name == "TriangularRecurrence":
            added.append(Rename().visit(copy.deepcopy(node)))
    assert len(added) == 4
    candidate.body.extend(added)
    ast.fix_missing_locations(candidate)
    text = (
        "# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC\n# SPDX-License-Identifier: Apache-2.0\n"
        + ast.unparse(candidate)
        + "\n"
    )
    args.output.write_text(text)
    subprocess.run([BLACK, "--line-length", "120", str(args.output)], check=True, capture_output=True)
    emitted = ast.parse(args.output.read_text())
    assert digest(emitted) == digest(candidate)
    # Remove only new helpers and restore both precise edited methods.
    restored = copy.deepcopy(emitted)
    restored.body = restored.body[: -len(added)]
    restore_cls = next(n for n in restored.body if isinstance(n, ast.ClassDef) and n.name == "Chunk64StateGraph")
    for index, node in enumerate(restore_cls.body):
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            restore_cls.body[index] = original_init
        if isinstance(node, ast.FunctionDef) and node.name == "__call__":
            restore_cls.body[index] = original_call
    assert digest(restored) == digest(original)
    # Independently ensure there were no non-loop changes within the call.
    restore_call = copy.deepcopy(call)

    class RestoreLoop(ast.NodeTransformer):
        def visit_Assign(self, node):
            if ast.dump(node) == ast.dump(replacement[0]):
                return copy.deepcopy(old[0])
            return node

        def visit_If(self, node):
            if ast.dump(node) == ast.dump(replacement[1]):
                return None
            return self.generic_visit(node)

    restore_call = RestoreLoop().visit(restore_call)
    assert digest(restore_call) == digest(original_call)
    assert all(ast.dump(a) == ast.dump(b) for a, b in zip(init.body[:-1], original_init.body))
    assert len(init.body) == len(original_init.body) + 1
    component_evidence = component.E / "v6_triangular_recurrence_fusion_all_device.json"
    measured = json.loads(component_evidence.read_text())
    assert measured["status"] == "component_exact" and measured["device_closed"]
    assert len(measured["cases"]) == 48 and all(c["bits_different"] == 0 for c in measured["cases"])
    report = dict(
        status="source_candidate_ready",
        source_sha256=component.sha(__file__),
        base=str(args.base),
        base_sha256=args.base_sha,
        component_source=str(Path(component.__file__)),
        component_sha256=raw_sha256(component.__file__),
        component_historical_sha256=COMPONENT_SHA,
        output=str(args.output),
        output_sha256=component.sha(args.output),
        base_ast_sha256=digest(original),
        candidate_ast_sha256=digest(emitted),
        restoration_exact=True,
        method_changes=[
            "Chunk64StateGraph.__init__: initialize nonallocating helper",
            "Chunk64StateGraph.__call__: replace63row triangular loop only",
        ],
        added_names=list(NAMES.values()),
        kernel_literal_sha256={
            name: hashlib.sha256(getattr(component, name).encode()).hexdigest()
            for name in ("READER", "COMPUTE", "EMPTY_WRITER")
        },
        component_device_evidence=dict(
            path=str(component_evidence), sha256=component.sha(component_evidence), cases=48
        ),
        whole_adapter_gate="Required separately; builder does not claim adapter or decoder acceptance",
        public_context_change=False,
        persistent_l1_bytes=0,
        temporary_cb_bytes_per_active_head=40960,
        formatter=BLACK,
    )
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print("TRIANGULAR_SNAPSHOT_READY", report["output_sha256"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, default=DEFAULT_BASE)
    p.add_argument("--base-sha", default=DEFAULT_SHA)
    p.add_argument(
        "--output",
        type=Path,
        default=component.E / "v6_corrected_fma_full_v5_triangular_compact_silu_integration.py.snapshot",
    )
    p.add_argument("--report", type=Path, default=component.E / "v6_triangular_integration_build.json")
    build(p.parse_args())


if __name__ == "__main__":
    main()
