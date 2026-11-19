# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Static direct-reader weighted tail candidates; independent44 and composed85 bases."""

import argparse
import ast
import copy
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "doc/fused_decoder/endurance"
COMPONENT = ROOT / "tests/v6_weighted_norm_interleaved_probe.py"
COMPONENT_SHA = "f1dc330d9f59e15dac3fc95ef795eb625a76526062bbc6d1648337dc5dbc2283"
BASE = E / "v6_transport_conv_norm_decay_lookup_integration.py.snapshot"
BASE_SHA = "44f87c04349836a4ad8b09c7a6c573fe3afe1e558af285e0482cce2c0fe5b2b3"
COMPOSED = E / "v6_transport_conv_norm_decay_transpose_integration.py.snapshot"
COMPOSED_SHA = "85adacf5df348f815e684a6ad1b1a36ecf12e862b72ef156681bfabb86dccd92"
GATE = E / "v6_weighted_norm_interleaved_device.json"
BLACK = "/home/ttuser/.cache/pre-commit/repoefelr92y/py_env-python3/bin/black"
AUTOFLAKE = "/home/ttuser/.cache/pre-commit/repodje5jyzf/py_env-python3/bin/autoflake"
ISORT = "/home/ttuser/.cache/pre-commit/repodrn1fio9/py_env-python3/bin/isort"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(node):
    return hashlib.sha256(ast.dump(node).encode()).hexdigest()


def named(tree, name):
    return next(n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)


def assignment(tree, name):
    return next(
        n for n in tree.body if isinstance(n, ast.Assign) and len(n.targets) == 1 and ast.unparse(n.targets[0]) == name
    )


def build(args):
    assert not args.output.exists() and not args.report.exists()
    for path, expected in ((BASE, BASE_SHA), (COMPOSED, COMPOSED_SHA), (COMPONENT, COMPONENT_SHA)):
        assert sha(path) == expected, path
    gate = json.loads(GATE.read_text())
    assert (
        gate["source_sha256"] == COMPONENT_SHA
        and gate["status"] == "all_exact_complete_adapter"
        and gate["device_closed"]
    )
    assert len(gate["cases"]) == 16 and len(gate["transport_controls"]) == 2
    assert all(
        all(c["exact"] and c["bit_differences"] == 0 for c in row["checks"].values())
        for row in gate["cases"] + gate["transport_controls"]
    )
    assert all(row["all_bit_exact"] and row["repeated_executions"] == 100 for row in gate["timing"].values())
    base, base_sha = (COMPOSED, COMPOSED_SHA) if args.compose else (BASE, BASE_SHA)
    original = ast.parse(base.read_text())
    canonical = ast.parse(BASE.read_text())
    assert digest(named(original, "WeightedNormTail")) == digest(named(canonical, "WeightedNormTail"))
    assert digest(assignment(original, "WEIGHTED_NORM_COMPUTE_SOURCE")) == digest(
        assignment(canonical, "WEIGHTED_NORM_COMPUTE_SOURCE")
    )
    candidate = copy.deepcopy(original)
    source_tree = ast.parse(COMPONENT.read_text())
    source_method = named(named(source_tree, "DirectWeightedNormTail"), "__call__")
    method = copy.deepcopy(source_method)
    assert ast.dump(method.body[0]) == ast.dump(ast.parse("t, base = self.t, self.base").body[0])
    method.body[0] = ast.parse("t = self.t").body[0]
    transformed_attributes = []

    class InlineBase(ast.NodeTransformer):
        def visit_Attribute(self, node):
            if isinstance(node.value, ast.Name) and node.value.id == "base":
                transformed_attributes.append(node.attr)
                node.value.id = "self"
            return self.generic_visit(node)

        def visit_Name(self, node):
            if node.id == "READER":
                node.id = "WEIGHTED_NORM_INTERLEAVED_READER"
            return node

    method = InlineBase().visit(method)
    assert not any(isinstance(n, ast.Name) and n.id == "base" for n in ast.walk(method))
    # Proof of transfer: reverse only reviewed attributes, reader rename and local alias.
    # Original method has self.t/self.mesh/self.source, distinct from baseline-owned fields.
    assert set(transformed_attributes) == {"memory", "gamma", "debug", "cores", "config"}

    class RestoreBase(ast.NodeTransformer):
        def visit_Attribute(self, node):
            if (
                isinstance(node.value, ast.Name)
                and node.value.id == "self"
                and node.attr in set(transformed_attributes)
            ):
                node.value.id = "base"
            return self.generic_visit(node)

        def visit_Name(self, node):
            if node.id == "WEIGHTED_NORM_INTERLEAVED_READER":
                node.id = "READER"
            return node

    transferred = RestoreBase().visit(copy.deepcopy(method))
    transferred.body[0] = copy.deepcopy(source_method.body[0])
    assert digest(transferred) == digest(source_method)
    cls = named(candidate, "WeightedNormTail")
    old_method = copy.deepcopy(named(cls, "__call__"))
    method_index = next(i for i, n in enumerate(cls.body) if isinstance(n, ast.FunctionDef) and n.name == "__call__")
    cls.body[method_index] = method
    compute_node = assignment(candidate, "WEIGHTED_NORM_COMPUTE_SOURCE")
    old_compute = copy.deepcopy(compute_node)
    original_compute = ast.literal_eval(compute_node.value)
    publication = " core.reserve_back(4); core.push_back(4);\n sums.reserve_back(1); sums.push_back(1);\n"
    assert original_compute.count(publication) == 1
    compute = original_compute.replace(publication, "")
    assert hashlib.sha256(compute.encode()).hexdigest() == gate["source_proof"]["candidate_compute_sha256"]
    compute_node.value = ast.Constant(compute)
    reader_source = ast.literal_eval(assignment(source_tree, "READER").value)
    assert hashlib.sha256(reader_source.encode()).hexdigest() == gate["source_proof"]["reader_sha256"]
    reader = ast.Assign(
        targets=[ast.Name(id="WEIGHTED_NORM_INTERLEAVED_READER", ctx=ast.Store())], value=ast.Constant(reader_source)
    )
    candidate.body.append(reader)
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
    assert digest(emitted) == digest(candidate), "Formatter changed executable AST"
    for cmd in (
        [AUTOFLAKE, "--remove-all-unused-imports", "--check"],
        [ISORT, "--check-only"],
        [BLACK, "--check", "--line-length", "120"],
    ):
        subprocess.run(cmd + [str(args.output)], check=True, capture_output=True)
    restored = copy.deepcopy(emitted)
    assert digest(restored.body[-1]) == digest(reader)
    restored.body.pop()
    restored_cls = named(restored, "WeightedNormTail")
    assert digest(restored_cls.body[method_index]) == digest(method)
    restored_cls.body[method_index] = old_method
    restored_compute = assignment(restored, "WEIGHTED_NORM_COMPUTE_SOURCE")
    assert ast.literal_eval(restored_compute.value) == compute
    restored_compute.value = old_compute.value
    assert digest(restored) == digest(original)
    for name in ("FusedDecoder", "LocalSiluMultiply"):
        assert digest(named(emitted, name)) == digest(named(original, name)), name
    assert digest(named(named(emitted, "WeightedNormTail"), "__init__")) == digest(
        named(named(original, "WeightedNormTail"), "__init__")
    )
    artifacts = {}
    for suffix, text in (
        ("reader.cpp.snapshot", reader_source),
        ("compute.cpp.snapshot", compute),
        ("builder.py.snapshot", Path(__file__).read_text()),
    ):
        path = Path(str(args.output) + "." + suffix)
        assert not path.exists()
        path.write_text(text)
        artifacts[str(path)] = sha(path)
    report = dict(
        status="standalone_source_candidate_ready_not_selected",
        source_sha256=sha(__file__),
        base=str(base),
        base_sha256=base_sha,
        output=str(args.output),
        output_sha256=sha(args.output),
        component=str(COMPONENT),
        component_sha256=COMPONENT_SHA,
        gate=dict(path=str(GATE), sha256=sha(GATE)),
        component_debug_full_adapter_and_control_bits_exact=True,
        component_timing=gate["timing"],
        component_transfer_exact=True,
        aliases_inlined=transformed_attributes,
        full_module_inverse_ast_exact=True,
        formatting_ast_exact=True,
        pinned_hook_checks=True,
        fused_decoder_ast_unchanged=True,
        compact_silu_ast_unchanged=True,
        weighted_constructor_ast_unchanged=True,
        weighted_helper_ast_sha256=digest(named(emitted, "WeightedNormTail")),
        weighted_call_ast_sha256=digest(named(named(emitted, "WeightedNormTail"), "__call__")),
        weighted_init_ast_sha256=digest(named(named(emitted, "WeightedNormTail"), "__init__")),
        reader_sha256=hashlib.sha256(reader_source.encode()).hexdigest(),
        compute_sha256=hashlib.sha256(compute.encode()).hexdigest(),
        footprint=dict(
            old_input_shared_tensor_bytes=20480,
            new_input_local_cb_bytes=20480,
            persistent_l1_added_bytes=0,
            persistent_dram_added_bytes=0,
            core_and_sum_physical_tiles=[4, 1],
            gamma_output_debug_shapes_unchanged=True,
            compact_table_unchanged=True,
            new_reader_instruction_runtime_metadata=True,
        ),
        qualification="Wholedecoder correctness/performance/capacity remain required; eager component trends recorded separately",
        artifacts=artifacts,
        device_executed=False,
        production_mutated=False,
    )
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(report["status"], report["output_sha256"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--compose", action="store_true")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
