# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Supplement unchanged V6 audit with exact transport-site restoration proof."""

import argparse
import ast
import copy
import json
import sys
from pathlib import Path

from models.autoports.qwen_qwen3_6_35b_a3b.tests import audit_fused_snapshot_v6 as base_audit

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "doc/fused_decoder/endurance"
AUDITOR_SHA = "ab9bca957f8b10215aaa2b4681b6c10874e7941c1d3279e2cbb44e4cfdaf80e8"
BASE = E / "v6_corrected_fma_full_v5_triangular_compact_silu_formatted_integration.py.snapshot"
BASE_SHA = "ab28a91bb3cb5e8b88e5371da0defe3b0d731e86a131c8523483e4b6b65b0334"
CANDIDATES = {
    "placement": "5d8b71b7661a926d2da1749c00752cb3414802a74e592e42c6ff5e1459bf88c8",
    "indices": "215deb3b4b96298c7f56b661cf2bd8a61f5e885164c83a6672df9c8ba2be283d",
    "combined": "13f78cd31929b49bd82bcf12723bccc54e9cbad19544370daafafb93e6de77e4",
}


def method(tree, name):
    return base_audit.named(base_audit.named(tree, "FusedDecoder"), name)


def inverse_transport(tree, reference, variant):
    """Restore individual statements only, never replace or exempt a method/class."""
    assert variant in {"base", *CANDIDATES}
    restored = copy.deepcopy(tree)
    changes = []
    if variant in ("placement", "combined"):
        fn = method(restored, "_forward")
        reference_fn = method(reference, "_forward")
        assert base_audit.digest(fn.body[0]) == base_audit.digest(reference_fn.body[0])
        sites = [i for i, node in enumerate(fn.body[1:], 1) if ast.unparse(node) == "mlp_input = x"]
        assert len(sites) == 1
        original_input = next(
            node
            for node in reference_fn.body[1:]
            if ast.unparse(node) == "mlp_input = ttnn.to_memory_config(x, self.norm_memory) if decode else x"
        )
        changes.append(
            dict(method="_forward", current=ast.unparse(fn.body[sites[0]]), restored=ast.unparse(original_input))
        )
        fn.body[sites[0]] = copy.deepcopy(original_input)
        original_output = next(
            node
            for node in reference_fn.body[1:]
            if ast.unparse(node) == "if decode:\n    mlp_input = ttnn.to_memory_config(mlp_input, DRAM)"
        )
        assert isinstance(fn.body[-1], ast.Return)
        assert ast.unparse(fn.body[-2]) == "mlp_input = self.norm(mlp_input, 'post_attention_layernorm.weight')"
        changes.append(dict(method="_forward", current=None, restored=ast.unparse(original_output)))
        fn.body.insert(-1, copy.deepcopy(original_output))
    if variant in ("indices", "combined"):
        fn = method(restored, "_moe")
        guard_end = fn.body[0].end_lineno
        sites = [
            node
            for node in ast.walk(fn)
            if isinstance(node, ast.Assign) and ast.unparse(node) == "indices = ttnn.reshape(ids, (1, 8))"
        ]
        assert len(sites) == 1 and sites[0].lineno > guard_end
        original = next(
            node
            for node in ast.walk(method(reference, "_moe"))
            if isinstance(node, ast.Assign) and ast.unparse(node) == "indices = reshape(ids, (1, 8))"
        )
        changes.append(dict(method="_moe", current=ast.unparse(sites[0]), restored=ast.unparse(original)))
        sites[0].value = copy.deepcopy(original.value)
    assert base_audit.digest(restored) == base_audit.digest(
        reference
    ), "Unreviewed change outside exact transport sites"
    return dict(
        variant=variant,
        inverse_statements=changes,
        restored_module_ast_sha256=base_audit.digest(restored),
        reference_module_ast_sha256=base_audit.digest(reference),
        full_module_exact_after_only_site_inverse=True,
        whole_method_or_class_exemption=False,
        additional_setup_exemptions=[],
        ignored_executable_strings_or_constants=False,
        arithmetic_boundary_changes=False,
    )


def negative_controls(candidate, reference):
    outcomes = {}
    for label, owner, name, statement in (
        ("forward_runtime_host", "FusedDecoder", "_forward", "ttnn.to_torch(hidden_states)"),
        ("moe_runtime_upload", "FusedDecoder", "_moe", "ttnn.from_torch(x)"),
        ("norm_arithmetic_change", "FusedDecoder", "norm", "x = ttnn.multiply(x, 2.0)"),
        ("helper_runtime_host", "Chunk64TriangularRecurrence", "__call__", "self.t.to_torch(matrix)"),
        ("setup_extra_code", "FusedDecoder", "from_state_dict", "extra = 1"),
        (
            "public_alignment_restriction",
            "FusedDecoder",
            "prepare_prefill",
            "if seq_len % 64: raise ValueError('aligned only')",
        ),
    ):
        changed = copy.deepcopy(candidate)
        function = base_audit.named(base_audit.named(changed, owner), name)
        function.body.insert(0, ast.parse(statement).body[0])
        try:
            inverse_transport(changed, reference, "combined")
        except (AssertionError, StopIteration):
            outcomes[label] = "rejected"
        else:
            raise AssertionError((label, "Unexpected acceptance"))
    for label in ("kernel_string_change", "full_kind_guard_change", "indices_precision_change"):
        changed = copy.deepcopy(candidate)
        if label == "kernel_string_change":
            node = next(
                node
                for node in changed.body
                if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "CHUNK64_TRIANGULAR_COMPUTE"
            )
            node.value.value += "\n// unexpected executable source change\n"
        elif label == "full_kind_guard_change":
            method(changed, "_forward").body[0].body.insert(0, ast.parse("unexpected = 1").body[0])
        else:
            node = next(
                node
                for node in ast.walk(method(changed, "_moe"))
                if isinstance(node, ast.Assign) and ast.unparse(node) == "indices = ttnn.reshape(ids, (1, 8))"
            )
            node.value.args[0] = ast.parse("ttnn.typecast(ids, ttnn.uint32)", mode="eval").body
        try:
            inverse_transport(changed, reference, "combined")
        except (AssertionError, StopIteration):
            outcomes[label] = "rejected"
        else:
            raise AssertionError((label, "Unexpected acceptance"))
    return outcomes


def audit(source):
    assert base_audit.sha(base_audit.__file__) == AUDITOR_SHA, "Review upstream auditor changes explicitly"
    assert base_audit.sha(BASE) == BASE_SHA
    archive = E / "v6_transport_auditor_base.py.snapshot"
    assert base_audit.sha(archive) == AUDITOR_SHA
    reference = ast.parse(BASE.read_text())
    references = {"base": reference}
    reference_hashes = {str(BASE): BASE_SHA, str(archive): AUDITOR_SHA}
    for name, expected in CANDIDATES.items():
        path = E / f"v6_ab28_transport_{name}.py.snapshot"
        assert base_audit.sha(path) == expected
        references[name] = ast.parse(path.read_text())
        reference_hashes[str(path)] = expected
        inverse_transport(references[name], reference, name)
    tree = ast.parse(Path(source).read_text())
    matching = [name for name, ref in references.items() if base_audit.digest(tree) == base_audit.digest(ref)]
    assert len(matching) == 1, "Source must match one explicitly reviewed transport or base execution AST"
    restoration = inverse_transport(tree, reference, matching[0])
    # Run the unchanged auditor on ACTUAL source, retaining all runtime-helper,
    # setup, planner, KV, full-kind, context and mutation gates.
    result = base_audit.audit(source)
    result["transport_restoration"] = restoration
    result["transport_negative_controls"] = negative_controls(references["combined"], reference)
    result["transport_reference_hashes"] = reference_hashes
    result["upstream_audit_source_sha256"] = AUDITOR_SHA
    result["audit_source_sha256"] = base_audit.sha(__file__)
    result[
        "supplemental_audit"
    ] = "Exact reviewed site inverse plus unchanged upstream audit on actual source; no exemptions"
    assert "ttnn" not in sys.modules and "torch" not in sys.modules
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=ROOT / "tt/fused_decoder.py")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "Preserve evidence"
    result = audit(args.snapshot)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("FUSED_TRANSPORT_AUDIT_PASS", result["source_sha256"], result["transport_restoration"]["variant"])


if __name__ == "__main__":
    main()
