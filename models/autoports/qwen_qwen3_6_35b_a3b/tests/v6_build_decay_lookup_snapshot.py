# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Embed the measured direct decay reader into new decode-only snapshots.

Precise AST reversal proves all prefill code, table generation, full-kind paths,
and unrelated helpers unchanged. This builder never accesses a device.
"""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

T = Path(__file__).resolve().parent
E = T.parent / "doc/fused_decoder/endurance"
COMPONENT = T / "v6_decay_direct_lookup_probe.py"
COMPONENT_SHA = "591cad1578e85497c788f7e72b11077878014e1d1981199e1d7d16052feba0f0"
BASE = E / "v6_corrected_fma_full_v5_triangular_compact_silu_formatted_integration.py.snapshot"
BASE_SHA = "ab28a91bb3cb5e8b88e5371da0defe3b0d731e86a131c8523483e4b6b65b0334"
COMPOSED = E / "v6_transport_conv_norm_terminal_scan_integration.py.snapshot"
COMPOSED_SHA = "803685e105d7ef1d46a6f9a8ceafb1758d9b1131e1c8c35aed1ab05cbf1475a5"
BLACK = "/home/ttuser/.cache/pre-commit/repoefelr92y/py_env-python3/bin/black"
ISORT = "/home/ttuser/.cache/pre-commit/repodrn1fio9/py_env-python3/bin/isort"
AUTOFLAKE = "/home/ttuser/.cache/pre-commit/repodje5jyzf/py_env-python3/bin/autoflake"
NAMES = dict(DirectDecayLookup="DecodeDecayLookup", READER="DECAY_LOOKUP_READER", EMPTY="DECAY_LOOKUP_EMPTY")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(node):
    return hashlib.sha256(ast.dump(node).encode()).hexdigest()


def decoder(tree):
    return next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FusedDecoder")


def method(cls, name):
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)


def lookup_region(method_node):
    start = next(
        i
        for i, n in enumerate(method_node.body)
        if isinstance(n, ast.Assign) and any(isinstance(x, ast.Name) and x.id == "raw_indices" for x in n.targets)
    )
    end = next(
        i
        for i, n in enumerate(method_node.body[start:], start)
        if isinstance(n, ast.If)
        and any(
            isinstance(x, ast.Assign) and any(isinstance(y, ast.Name) and y.id == "decay_value" for y in x.targets)
            for x in n.body
        )
    )
    return start, end


def build(args):
    assert not args.output.exists() and not args.report.exists()
    assert sha(BASE) == BASE_SHA and sha(COMPOSED) == COMPOSED_SHA and sha(COMPONENT) == COMPONENT_SHA
    base, base_sha = (COMPOSED, COMPOSED_SHA) if args.composed else (BASE, BASE_SHA)
    spec = importlib.util.spec_from_file_location("frozen_direct_decay", COMPONENT)
    component = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(component)
    gate_path = E / "v6_decay_direct_lookup_all_device.json"
    gate = json.loads(gate_path.read_text())
    assert (
        gate["source_sha256"] == COMPONENT_SHA
        and gate["status"] == "completed_microbenchmark"
        and gate["device_closed"]
    )
    assert len(gate["cases"]) == 43
    assert all(
        r["native_vs_compact"]["bits_different"] == r["native_vs_full"]["bits_different"] == 0 for r in gate["cases"]
    )
    original = ast.parse(base.read_text())
    candidate = copy.deepcopy(original)
    cls = decoder(candidate)
    setup, linear = method(cls, "from_state_dict"), method(cls, "_linear_attention")
    old_setup, old_linear = copy.deepcopy(setup), copy.deepcopy(linear)
    added_setup = ast.parse(
        """if self.kind == "linear_attention":
 self.decode_decay_lookup = DecodeDecayLookup(
  ttnn, mesh_device,
  self.compact_decay_table if self.compact_decay_table is not None else self.decay_table,
  lo if self.compact_decay_table is not None else None,
  hi if self.compact_decay_table is not None else None)
"""
    ).body[0]
    assert isinstance(setup.body[-1], ast.Return) and ast.unparse(setup.body[-1].value) == "self"
    setup.body.insert(len(setup.body) - 1, added_setup)
    start, end = lookup_region(linear)
    original_block = copy.deepcopy(linear.body[start : end + 1])
    _, native_sha = component.native_function(None)
    assert digest(ast.Module(original_block, type_ignores=[])) == native_sha
    branch = ast.parse(
        """if decode and cache_boundary:
 g = None
 decay_value = self.decode_decay_lookup(projection_a)
"""
    ).body[0]
    branch.orelse = copy.deepcopy(original_block)
    linear.body[start : end + 1] = [branch]
    # The per-head table construction and its full-domain compaction assertion
    # remain exactly the source used by the native component oracle.
    canonical_setup = method(decoder(ast.parse(BASE.read_text())), "from_state_dict")

    def table_block(setup_node):
        return next(
            n
            for n in setup_node.body
            if isinstance(n, ast.If)
            and any(
                isinstance(x, ast.Assign) and any(isinstance(y, ast.Name) and y.id == "gate_values" for y in x.targets)
                for x in n.body
            )
        )

    assert digest(table_block(old_setup)) == digest(table_block(canonical_setup))

    class Rename(ast.NodeTransformer):
        def visit_Name(self, node):
            node.id = NAMES.get(node.id, node.id)
            return node

        def visit_ClassDef(self, node):
            node.name = NAMES.get(node.name, node.name)
            return self.generic_visit(node)

    helpers = []
    for node in ast.parse(COMPONENT.read_text()).body:
        name = (
            node.name
            if isinstance(node, ast.ClassDef)
            else (
                node.targets[0].id
                if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                else None
            )
        )
        if name in NAMES:
            helpers.append(Rename().visit(copy.deepcopy(node)))
    assert len(helpers) == 3
    candidate.body.extend(helpers)
    candidate = ast.fix_missing_locations(candidate)
    args.output.write_text(
        "# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC\n# SPDX-License-Identifier: Apache-2.0\n"
        + ast.unparse(candidate)
        + "\n"
    )
    for command in ([AUTOFLAKE, "--remove-all-unused-imports", "--in-place"], [ISORT], [BLACK, "--line-length", "120"]):
        subprocess.run(command + [str(args.output)], check=True, capture_output=True)
    emitted = ast.parse(args.output.read_text())
    assert digest(emitted) == digest(candidate)
    for command in (
        [AUTOFLAKE, "--remove-all-unused-imports", "--check"],
        [ISORT, "--check-only"],
        [BLACK, "--check", "--line-length", "120"],
    ):
        subprocess.run(command + [str(args.output)], check=True, capture_output=True)
    restored = copy.deepcopy(emitted)
    assert [digest(n) for n in restored.body[-3:]] == [digest(n) for n in helpers]
    restored.body = restored.body[:-3]
    restore_setup = method(decoder(restored), "from_state_dict")
    assert digest(restore_setup.body[-2]) == digest(added_setup)
    del restore_setup.body[-2]
    restore_linear = method(decoder(restored), "_linear_attention")
    assert digest(restore_linear.body[start]) == digest(branch)
    restore_linear.body[start : start + 1] = copy.deepcopy(restore_linear.body[start].orelse)
    assert digest(restore_setup) == digest(old_setup) and digest(restore_linear) == digest(old_linear)
    assert digest(restored) == digest(original)
    assert digest(method(decoder(emitted), "from_state_dict").body[0]) == digest(old_setup.body[0])
    report = dict(
        status="source_candidate_ready",
        device_executed=False,
        acceptance_eligible=False,
        source=str(Path(__file__)),
        source_sha256=sha(__file__),
        base=str(base),
        base_sha256=base_sha,
        component_source=str(COMPONENT),
        component_sha256=COMPONENT_SHA,
        component_gate=dict(path=str(gate_path), sha256=sha(gate_path), cases=43),
        output=str(args.output),
        output_sha256=sha(args.output),
        composed=args.composed,
        native_adapter_ast_sha256=native_sha,
        table_setup_ast_sha256=digest(table_block(old_setup)),
        exact_region_restoration=True,
        full_module_restoration=True,
        formatting_ast_exact=True,
        pinned_hook_checks=True,
        changed_methods=["from_state_dict", "_linear_attention"],
        setup_change="Append linear-kind helper referencing existing compact/full table and existing computed lo/hi; no tensor allocation or table mutation.",
        runtime_change="if decode and cache_boundary: direct table lookup; else original index/gather/layout block AST.",
        public_singleton_prefill_unchanged=True,
        prefill_all_logical_lengths_unchanged=True,
        full_attention_setup_branch_unchanged=True,
        table_setup_and_assertions_unchanged=True,
        direct_helper_diagnostic_support="Positive logical S including nonaligned33/65; decoder integration restricted to public decode by cache_boundary guard.",
        added_names=list(NAMES.values()),
        kernel_sha256={
            name: hashlib.sha256(getattr(component, name).encode()).hexdigest() for name in ("READER", "EMPTY")
        },
        unchanged_method_ast_sha256={
            n.name: digest(n)
            for n in decoder(original).body
            if isinstance(n, ast.FunctionDef) and n.name not in ("from_state_dict", "_linear_attention")
        },
        base_ast_sha256=digest(original),
        candidate_ast_sha256=digest(emitted),
        persistent_l1_added_bytes=0,
        temporary_cb_bytes_per_active_core=12288,
        singleton_active_cores=1,
        context_contract_changed=False,
        remaining_gates="Supplemental explicit raw-code upload coverage plus independent/composed complete-decoder accuracy, traced performance, repeated-run/capacity/watcher gates.",
    )
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print("DECAY_LOOKUP_SNAPSHOT_READY", report["output_sha256"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--composed", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    build(p.parse_args())


if __name__ == "__main__":
    main()
