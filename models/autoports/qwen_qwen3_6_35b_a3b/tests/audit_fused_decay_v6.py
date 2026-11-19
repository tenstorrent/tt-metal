# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Supplement the frozen 803 source audit with the exact decode-decay delta."""

import argparse
import ast
import copy
import hashlib
import json
import sys
from pathlib import Path

from models.autoports.qwen_qwen3_6_35b_a3b.tests import audit_fused_composed_v6 as composed

base = composed.base
ROOT, E = composed.ROOT, composed.E
AUDITOR_SHA = "6ba4ffe724503feb0475625bc9293d0aa73d6cddbfaa4d73efd84bf4de7e4729"
REFERENCE = E / "v6_transport_conv_norm_terminal_scan_integration.py.snapshot"
REFERENCE_SHA = "803685e105d7ef1d46a6f9a8ceafb1758d9b1131e1c8c35aed1ab05cbf1475a5"
CANDIDATE = E / "v6_transport_conv_norm_decay_lookup_integration.py.snapshot"
CANDIDATE_SHA = "44f87c04349836a4ad8b09c7a6c573fe3afe1e558af285e0482cce2c0fe5b2b3"
HELPER_SHA = "2210cc0b50b60e04700cd32a5c7439228554f5e36f5f52bd7e9e1c1ec5ae2750"
KERNEL_PINS = {
    "DECAY_LOOKUP_READER": "0ac170360f48e92ba07dd7f2e5ab2dec4e6a72d46eb8ea22a3c5c92bc587f61c",
    "DECAY_LOOKUP_EMPTY": "d27764b9c0ea71d9af98000e585fd00262ec4a1cd1f323c0bee21e56a6f7a9fa",
}
SETUP = """if self.kind == 'linear_attention':
 self.decode_decay_lookup = DecodeDecayLookup(ttnn, mesh_device,
  self.compact_decay_table if self.compact_decay_table is not None else self.decay_table,
  lo if self.compact_decay_table is not None else None,
  hi if self.compact_decay_table is not None else None)
"""
BRANCH = """if decode and cache_boundary:
 g = None
 decay_value = self.decode_decay_lookup(projection_a)
"""


def inverse_decay(tree, reference):
    helper = base.named(tree, "DecodeDecayLookup")
    assert composed.exact(helper) == HELPER_SHA
    init = base.named(helper, "__init__")
    assert [ast.unparse(n.func) for n in ast.walk(init) if isinstance(n, ast.Call)] == [
        "tuple",
        "table.memory_config",
    ]
    assert "DecodeDecayLookup.__init__" not in base.SETUP
    assert {composed.node_name(n) for n in tree.body[-3:]} == {"DecodeDecayLookup", *KERNEL_PINS}
    for name, expected in KERNEL_PINS.items():
        nodes = [n for n in tree.body if composed.node_name(n) == name]
        assert len(nodes) == 1 and isinstance(nodes[0].value, ast.Constant)
        assert hashlib.sha256(nodes[0].value.value.encode()).hexdigest() == expected
    restored = copy.deepcopy(tree)
    del restored.body[-3:]
    setup = composed.transport.method(restored, "from_state_dict")
    assert composed.exact(setup.body[-2]) == composed.exact(ast.parse(SETUP).body[0])
    del setup.body[-2]
    linear = composed.transport.method(restored, "_linear_attention")
    sites = [
        i
        for i, n in enumerate(linear.body)
        if isinstance(n, ast.If)
        and any(
            isinstance(c, ast.Attribute) and c.attr == "decode_decay_lookup"
            for statement in n.body
            for c in ast.walk(statement)
        )
    ]
    assert len(sites) == 1
    i = sites[0]
    branch, expected = linear.body[i], ast.parse(BRANCH).body[0]
    assert composed.exact(branch.test) == composed.exact(expected.test)
    assert [composed.exact(n) for n in branch.body] == [composed.exact(n) for n in expected.body]
    assert branch.orelse
    linear.body[i : i + 1] = copy.deepcopy(branch.orelse)
    assert composed.exact(restored) == composed.exact(reference), "Unexpected change outside exact decay regions"
    return dict(
        restored_module_ast_sha256=composed.exact(restored),
        full_module_exact_after_only_region_inverse=True,
        helper_ast_sha256=HELPER_SHA,
        constructor_ast_sha256=composed.exact(init),
        constructor_calls=["tuple", "table.memory_config"],
        kernel_string_sha256=KERNEL_PINS,
        new_setup_exemptions=[],
        whole_class_or_method_exemption=False,
        prefill_including_public_singleton_unchanged=True,
        full_attention_unchanged=True,
        table_construction_and_compaction_proof_unchanged=True,
    )


def negative_controls(tree, reference, v5, reviewed, extra):
    outcomes = {}
    for label, owner, method, statement in (
        ("decay_runtime_readback", "DecodeDecayLookup", "__call__", "self.t.to_torch(projection_a)"),
        ("decay_constructor_upload", "DecodeDecayLookup", "__init__", "t.from_torch(table)"),
        ("decay_runtime_dynamic_exec", "DecodeDecayLookup", "__call__", "exec('pass')"),
        ("decay_constructor_allocation", "DecodeDecayLookup", "__init__", "t.empty((32, 32))"),
        ("unreviewed_setup_call", "FusedDecoder", "from_state_dict", "unexpected()"),
    ):
        changed = copy.deepcopy(tree)
        base.named(base.named(changed, owner), method).body.insert(0, ast.parse(statement).body[0])
        try:
            inverse_decay(changed, reference)
        except AssertionError:
            outcomes[label] = "rejected_by_exact_inverse_proof"
        else:
            raise AssertionError(label)
        if label in {"decay_runtime_readback", "decay_constructor_upload", "decay_runtime_dynamic_exec"}:
            try:
                base.syntax_audit(changed, v5, reviewed, extra)
            except AssertionError:
                outcomes[label] += "_and_independent_runtime_syntax_audit"
            else:
                raise AssertionError((label, "Runtime checker failed"))
    for name in KERNEL_PINS:
        changed = copy.deepcopy(tree)
        next(n for n in changed.body if composed.node_name(n) == name).value.value += "\n// changed\n"
        try:
            inverse_decay(changed, reference)
        except AssertionError:
            outcomes[name + "_mutation"] = "rejected"
        else:
            raise AssertionError(name)
    changed = copy.deepcopy(tree)
    branch = next(
        n
        for n in composed.transport.method(changed, "_linear_attention").body
        if isinstance(n, ast.If) and "decode_decay_lookup" in ast.unparse(n)
    )
    branch.test = ast.parse("decode", mode="eval").body
    try:
        inverse_decay(changed, reference)
    except AssertionError:
        outcomes["singleton_guard_change"] = "rejected"
    else:
        raise AssertionError("Guard mutation accepted")
    return outcomes


def audit(source):
    assert base.sha(composed.__file__) == AUDITOR_SHA
    assert base.sha(E / "v6_composed_803_auditor.py.snapshot") == AUDITOR_SHA
    assert base.sha(REFERENCE) == REFERENCE_SHA and base.sha(CANDIDATE) == CANDIDATE_SHA
    reference = ast.parse(REFERENCE.read_text())
    candidate = ast.parse(CANDIDATE.read_text())
    tree = ast.parse(Path(source).read_text())
    restoration = inverse_decay(tree, reference)
    assert base.digest(tree) == base.digest(candidate)
    # Execute unchanged 803 audit and all its controls. Its planner/dispatch
    # methods are bit-for-bit identical AST in the actual candidate; record that
    # transfer explicitly, without claiming device execution or rerunning kernels.
    result = composed.audit(REFERENCE)
    v5, reviewed, functional = base.reviewed_sources()
    extra = []
    for filename, expected in base.EXTRA_REVIEWED.items():
        assert base.sha(E / filename) == expected
        extra.append(ast.parse((E / filename).read_text()))
    extra.extend([reference, candidate])
    result["syntax"] = base.syntax_audit(tree, v5, reviewed, extra)
    result["contract"] = base.contract_audit(tree, v5, functional)
    unchanged = {}
    for name in base.PUBLIC:
        if name == "from_state_dict":
            continue
        actual = composed.transport.method(tree, name)
        original = composed.transport.method(reference, name)
        assert composed.exact(actual) == composed.exact(original)
        unchanged[name] = composed.exact(actual)
    result["planner_dispatch_transfer"] = dict(
        executed_reference=str(REFERENCE),
        reference_sha256=REFERENCE_SHA,
        exact_unchanged_public_method_ast_sha256=unchanged,
        actual_source_contract_audit_reexecuted=True,
        actual_source_runtime_syntax_audit_reexecuted=True,
    )
    result["decay_negative_controls"] = negative_controls(tree, reference, v5, reviewed, extra)
    result["decay_restoration"] = restoration
    result["source"] = str(Path(source).resolve())
    result["source_sha256"] = base.sha(source)
    result["upstream_composed_audit_source_sha256"] = AUDITOR_SHA
    result["audit_source_sha256"] = base.sha(__file__)
    result["reference_hashes"][str(CANDIDATE)] = CANDIDATE_SHA
    result["reviewed_extension_hashes"][str(CANDIDATE)] = CANDIDATE_SHA
    result["description_normalization"] = base.description_delta(tree, [(str(CANDIDATE), candidate)])
    context = json.loads((ROOT / "doc/context_contract.json").read_text())
    result["context_contract_evidence_matches_current_source"] = context.get("fused_decoder_validation", {}).get(
        "decoder_source_sha256"
    ) == base.sha(source)
    assert "ttnn" not in sys.modules and "torch" not in sys.modules
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=ROOT / "tt/fused_decoder.py")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "Preserve prior evidence"
    result = audit(args.snapshot)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("FUSED_DECAY_AUDIT_PASS", result["source_sha256"])


if __name__ == "__main__":
    main()
