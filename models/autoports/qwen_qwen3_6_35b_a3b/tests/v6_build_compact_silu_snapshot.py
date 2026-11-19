# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Build a reversible decode-only weighted-tail/local-SiLU standalone candidate."""

import argparse
import ast
import copy
import hashlib
import json
import sys
from pathlib import Path

from models.autoports.qwen_qwen3_6_35b_a3b.tests import v6_silu_compact_lookup_probe as local
from models.autoports.qwen_qwen3_6_35b_a3b.tests import v6_weighted_norm_compact_silu_probe as weighted
from models.autoports.qwen_qwen3_6_35b_a3b.tests.frozen_source_compatibility import raw_sha256, verify_source

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "doc/fused_decoder/endurance"
BASE = E / "v6_corrected_fma_full_v5_streaming_scan_norm_tail_fused_matmul.py.snapshot"
BASE_SHA = "8cda999c18f9245c872e40d7f67e027838b56396056ede43e162b7a8386f3faa"
WEIGHTED_SHA = "a4b77e7f22e8f0e7bd4122ca1458400ea1ef5efd02b135945b0e5d304b975cd6"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def named(scope, name):
    return next(n for n in scope.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)


def assigned(scope, name):
    return next(n for n in scope.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == name)


def digest(node):
    return hashlib.sha256(ast.dump(node).encode()).hexdigest()


class Rename(ast.NodeTransformer):
    def __init__(self, names):
        self.names = names

    def visit_Name(self, node):
        return ast.copy_location(ast.Name(id=self.names.get(node.id, node.id), ctx=node.ctx), node)


def build(base=BASE, base_sha=BASE_SHA):
    BASE, BASE_SHA = base, base_sha
    assert sha(BASE) == BASE_SHA
    assert verify_source(local.__file__, weighted.LOCAL_SHA)
    assert verify_source(weighted.__file__, WEIGHTED_SHA)
    correctness_path = E / "v6_silu_compact_lookup_device.json"
    timing_path = E / "v6_silu_compact_lookup_device.json"
    for path in (correctness_path, timing_path):
        evidence = json.loads(path.read_text())
        assert evidence["source_sha256"] == weighted.LOCAL_SHA
        assert evidence["device_closed"] and evidence["status"] == "completed_diagnostic"
        assert len(evidence["cases"]) == 30
        assert all(
            row[key]["bit_differences"] == 0
            for row in evidence["cases"]
            for key in ("output_vs_native", "gate_vs_table")
        )
    tree = ast.parse(BASE.read_text())
    original = copy.deepcopy(tree)
    cls = named(tree, "FusedDecoder")
    setup = named(cls, "from_state_dict")
    setup_index = next(
        i for i, n in enumerate(setup.body) if isinstance(n, ast.If) and "self.decode_norm_tail =" in ast.unparse(n)
    )
    old_setup = copy.deepcopy(setup.body[setup_index])
    assert ast.unparse(old_setup.test) == "self.kind == 'linear_attention'"
    assert len(old_setup.body) == 2 and isinstance(old_setup.body[0], ast.Assert)
    setup.body[setup_index] = ast.parse(
        "if self.kind == 'linear_attention':\n"
        "    assert self.config.rms_norm_eps == 1e-6, 'Measured norm-tail epsilon contract'\n"
        "    self.decode_weighted_norm_tail = WeightedNormTail(ttnn, mesh_device, state_dict['linear_attn.norm.weight'], debug=False)\n"
        "    self.decode_local_silu = LocalSiluMultiply(ttnn, mesh_device, debug=False)\n"
    ).body[0]

    attention = named(cls, "_linear_attention")
    branch_index = next(
        i
        for i, n in enumerate(attention.body)
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "cache_boundary"
        and "self.decode_norm_tail" in ast.unparse(n)
    )
    branch = attention.body[branch_index]
    old_branch_body = copy.deepcopy(branch.body)
    assert len(branch.body) == 5 and ast.unparse(branch.body[-1]).endswith(
        "self.decode_norm_tail(wide, sums, gate_rows)"
    )
    old_singleton = digest(ast.Module(body=branch.orelse, type_ignores=[]))
    branch.body = ast.parse(
        "weighted, _ = self.decode_weighted_norm_tail(wide, sums)\n"
        "merged, _ = self.decode_local_silu(weighted, projection_z)\n"
        "return self.linear(merged, 'linear_attn.out_proj.weight', ordered_out=cache_boundary, terminal_scan=decode and cache_boundary)\n"
    ).body
    assert digest(ast.Module(body=branch.orelse, type_ignores=[])) == old_singleton

    helper_tree = ast.parse(Path(weighted.__file__).read_text())
    weighted_class = copy.deepcopy(named(helper_tree, "WeightedNormTail"))
    old_kernel = assigned(tree, "GATED_NORM_TAIL_COMPUTE_SOURCE")
    old_class = named(tree, "GatedNormTail")
    kernel_index, class_index = tree.body.index(old_kernel), tree.body.index(old_class)
    tree.body[kernel_index] = ast.Assign(
        targets=[ast.Name(id="WEIGHTED_NORM_COMPUTE_SOURCE", ctx=ast.Store())],
        value=ast.Constant(value=weighted.WEIGHTED_NORM_COMPUTE_SOURCE),
    )
    tree.body[class_index] = weighted_class
    local_tree = ast.parse(Path(local.__file__).read_text())
    local_class = copy.deepcopy(named(local_tree, "LocalSiluMultiply"))
    local_names = {
        "READER": "LOCAL_SILU_READER_SOURCE",
        "COMPUTE": "LOCAL_SILU_COMPUTE_SOURCE",
        "make_table": "local_silu_math_table",
        "compact_table": "compact_silu_math_table",
    }
    compact_names = ("POS_LO", "POS_HI", "NEG_LO", "NEG_HI", "POS_COUNT", "NEG_COUNT", "COMPACT_ENTRIES")
    local_names.update({name: "SILU_" + name for name in compact_names})
    local_class = Rename(local_names).visit(local_class)
    table_function = copy.deepcopy(named(local_tree, "make_table"))
    table_function.name = "local_silu_math_table"
    # Only the setup function owns torch; runtime methods have no global torch.
    table_function.body.insert(0, ast.Import(names=[ast.alias(name="torch")]))
    compact_function = Rename(local_names).visit(copy.deepcopy(named(local_tree, "compact_table")))
    compact_function.name = "compact_silu_math_table"
    compact_function.body.insert(0, ast.Import(names=[ast.alias(name="torch")]))
    compact_constants = []
    for node in local_tree.body:
        if isinstance(node, ast.Assign):
            stored = {n.id for target in node.targets for n in ast.walk(target) if isinstance(n, ast.Name)}
            if stored and stored.issubset(set(compact_names)):
                compact_constants.append(Rename(local_names).visit(copy.deepcopy(node)))
    assert len(compact_constants) == 3
    compact = local.compact_table(local.make_table())
    assert compact.numel() * compact.element_size() == 29952
    added = compact_constants + [
        ast.Assign(
            targets=[ast.Name(id="LOCAL_SILU_READER_SOURCE", ctx=ast.Store())], value=ast.Constant(value=local.READER)
        ),
        ast.Assign(
            targets=[ast.Name(id="LOCAL_SILU_COMPUTE_SOURCE", ctx=ast.Store())], value=ast.Constant(value=local.COMPUTE)
        ),
        table_function,
        compact_function,
        local_class,
    ]
    tree.body.extend(added)
    ast.fix_missing_locations(tree)

    # Restore exactly the changed setup region, decode branch, two helper nodes,
    # and appended compact-lookup nodes. Every other AST node must be intact.
    restored = copy.deepcopy(tree)
    restored_cls = named(restored, "FusedDecoder")
    named(restored_cls, "from_state_dict").body[setup_index] = old_setup
    named(restored_cls, "_linear_attention").body[branch_index].body = old_branch_body
    restored.body[kernel_index] = copy.deepcopy(old_kernel)
    restored.body[class_index] = copy.deepcopy(old_class)
    restored.body = restored.body[: -len(added)]
    assert ast.dump(restored) == ast.dump(original)
    # The sole local-class differences are names of its frozen static globals.
    restored_local = Rename({v: k for k, v in local_names.items()}).visit(copy.deepcopy(local_class))
    assert ast.dump(restored_local) == ast.dump(named(local_tree, "LocalSiluMultiply"))
    assert digest(weighted_class) == digest(named(helper_tree, "WeightedNormTail"))

    runtime_audit = {}
    for helper in (weighted_class, local_class):
        method = named(helper, "__call__")
        calls = [ast.unparse(n.func) for n in ast.walk(method) if isinstance(n, ast.Call)]
        assert not any(c.startswith("torch.") or c.endswith((".from_torch", ".to_torch")) for c in calls)
        runtime_audit[helper.name] = dict(ast_sha256=digest(method), host_conversion_calls=[])
    text = (
        "# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC\n# SPDX-License-Identifier: Apache-2.0\n"
        + ast.unparse(tree)
        + "\n"
    )
    compile(text, "<local-silu-standalone>", "exec")
    imports = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert not any("tests" in item or "endurance" in item for item in imports)
    assert "GatedNormTail" not in text and "GATED_NORM_TAIL_COMPUTE_SOURCE" not in text
    assert "self.decode_norm_tail" not in text
    # This DRAM table is retained for multi-row prefill, not consumed in new decode.
    assert "ttnn.gather(self.silu_table, 1, silu_indices)" in text
    report = dict(
        classification="Standalone decode-only candidate; not production promotion or decoder acceptance",
        acceptance_eligible=False,
        oracle_assisted=False,
        status="static_build_pass",
        base=str(BASE),
        base_sha256=BASE_SHA,
        weighted_helper=str(Path(weighted.__file__)),
        weighted_helper_sha256=raw_sha256(weighted.__file__),
        weighted_helper_historical_sha256=WEIGHTED_SHA,
        local_helper=str(Path(local.__file__)),
        local_helper_sha256=raw_sha256(local.__file__),
        local_helper_historical_sha256=weighted.LOCAL_SHA,
        evidence={str(path): sha(path) for path in (correctness_path, timing_path)},
        ast_restoration_exact=True,
        public_singleton_else_ast_unchanged=True,
        prefill_and_full_v5_unchanged=True,
        static_kernel_constants=True,
        compact_setup_verifies_all_65536_codes=True,
        compact_table_bytes_per_worker=29952,
        compact_table_sha256=local.digest(compact),
        runtime_audit=runtime_audit,
        table_ownership="Existing DRAM table remains used by prefill; additional 958464-byte aggregate compact L1 table used by cache-boundary decode; full-attention early-return setup unchanged",
        pending="Weighted-tail component, combined-adapter timing, full decoder correctness/stress/watcher and traced performance, full L1 capacity remain mandatory; no hardware executed by builder",
    )
    return text, report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, default=BASE)
    p.add_argument("--base-sha", default=BASE_SHA)
    p.add_argument(
        "--snapshot",
        type=Path,
        default=E / "v6_corrected_fma_full_v5_compact_silu_fused_matmul_integration.py.snapshot",
    )
    p.add_argument("--output", type=Path, default=E / "v6_compact_silu_integration_build.json")
    args = p.parse_args()
    assert not args.snapshot.exists() and not args.output.exists(), "Preserve earlier evidence"
    source, result = build(args.base, args.base_sha)
    args.snapshot.write_text(source)
    result.update(
        snapshot=str(args.snapshot), snapshot_sha256=sha(args.snapshot), builder_sha256=sha(__file__), command=sys.argv
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    assert "ttnn" not in sys.modules
    print("STATIC_COMPACT_SILU_INTEGRATION_PASS", result["snapshot_sha256"])


if __name__ == "__main__":
    main()
