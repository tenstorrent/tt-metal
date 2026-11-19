# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only exact source audit of the three corrected-matmul transpose sites."""

import argparse
import ast
import copy
import hashlib
import json
import sys
from pathlib import Path

from models.autoports.qwen_qwen3_6_35b_a3b.tests import audit_fused_decay_v6 as decay

base, composed = decay.base, decay.composed
ROOT, E = decay.ROOT, decay.E
AUDITOR_SHA = "11c88ceb2dd3ea2855e0e6ce5c913e1f4ec875c12159e25309b3b737b606466b"
REFERENCE = decay.CANDIDATE
REFERENCE_SHA = decay.CANDIDATE_SHA
CANDIDATE = E / "v6_transport_conv_norm_decay_transpose_integration.py.snapshot"
CANDIDATE_SHA = "85adacf5df348f815e684a6ad1b1a36ecf12e862b72ef156681bfabb86dccd92"
HELPER_SHA = "ad92b3e06964b793f0daa48c6259c0f465315b75c2d5cc88cf63a417ef923b01"
READER_SHA = "06c4ade37b4af14aa1acac5f48fdf284ce0318a1f581084b102981f5898593ec"
SITES = (
    ("Chunk64StateGraph", "self.transpose_mm(kb, k, transpose_b=True)", "self.mm(kb, t.transpose(k, -1, -2))"),
    (
        "Chunk64StateGraph",
        "self.transpose_mm(weighted_k, v_new, transpose_a=True)",
        "self.mm(t.transpose(weighted_k, -1, -2), v_new)",
    ),
    ("Chunk64OutputGraph", "self.transpose_mm(q, k, transpose_b=True)", "self.mm(q, t.transpose(k, -1, -2))"),
)


class ReplaceCall(ast.NodeTransformer):
    def __init__(self, current, previous):
        self.current = ast.parse(current, mode="eval").body
        self.previous = ast.parse(previous, mode="eval").body
        self.count = 0

    def visit_Call(self, node):
        if composed.exact(node) == composed.exact(self.current):
            self.count += 1
            return copy.deepcopy(self.previous)
        return self.generic_visit(node)


def inverse_transpose(tree, reference):
    helper = base.named(tree, "TransposeCorrectedMatmul")
    assert composed.exact(helper) == HELPER_SHA
    init = base.named(helper, "__init__")
    assert [ast.unparse(n.func) for n in ast.walk(init) if isinstance(n, ast.Call)] == [
        "mesh.compute_with_storage_grid_size"
    ]
    assert "TransposeCorrectedMatmul.__init__" not in base.SETUP
    assert [composed.node_name(n) for n in tree.body[-2:]] == [
        "TRANSPOSE_CORRECTED_MM_READER",
        "TransposeCorrectedMatmul",
    ]
    reader = tree.body[-2]
    assert hashlib.sha256(ast.literal_eval(reader.value).encode()).hexdigest() == READER_SHA
    restored = copy.deepcopy(tree)
    del restored.body[-2:]
    graph_init = base.named(base.named(restored, "Chunk64StateGraph"), "__init__")
    added = ast.parse("self.transpose_mm = TransposeCorrectedMatmul(ttnn, mesh)").body[0]
    assert composed.exact(graph_init.body[-1]) == composed.exact(added)
    del graph_init.body[-1]
    sites = []
    for owner, current, previous in SITES:
        function = base.named(base.named(restored, owner), "__call__")
        change = ReplaceCall(current, previous)
        change.visit(function)
        assert change.count == 1, (owner, current, change.count)
        sites.append(dict(owner=owner + ".__call__", current=current, restored=previous, occurrences=1))
    assert composed.exact(restored) == composed.exact(reference), "Change outside three exact calls/setup/helper"
    assert composed.exact(base.named(tree, "FusedDecoder")) == composed.exact(base.named(reference, "FusedDecoder"))
    return dict(
        restored_module_ast_sha256=composed.exact(restored),
        inverse_sites=sites,
        full_module_exact_after_only_region_inverse=True,
        helper_ast_sha256=HELPER_SHA,
        constructor_ast_sha256=composed.exact(init),
        constructor_calls=["mesh.compute_with_storage_grid_size"],
        reader_string_sha256=READER_SHA,
        original_compute_writer_and_matmul_helper_unchanged=True,
        fused_decoder_ast_unchanged=True,
        new_setup_exemptions=[],
        whole_method_or_class_exemption=False,
    )


def negative_controls(tree, reference, v5, reviewed, extra):
    outcomes = {}
    for label, owner, name, statement in (
        ("transpose_runtime_readback", "TransposeCorrectedMatmul", "__call__", "self.t.to_torch(a)"),
        ("transpose_constructor_upload", "TransposeCorrectedMatmul", "__init__", "t.from_torch(mesh)"),
        ("transpose_constructor_allocation", "TransposeCorrectedMatmul", "__init__", "t.empty((32, 32))"),
        ("graph_setup_extra_call", "Chunk64StateGraph", "__init__", "unexpected()"),
        ("unreviewed_decoder_change", "FusedDecoder", "_linear_attention", "unexpected()"),
    ):
        changed = copy.deepcopy(tree)
        base.named(base.named(changed, owner), name).body.insert(0, ast.parse(statement).body[0])
        try:
            inverse_transpose(changed, reference)
        except AssertionError:
            outcomes[label] = "rejected_by_exact_inverse_proof"
        else:
            raise AssertionError(label)
        if label in {"transpose_runtime_readback", "transpose_constructor_upload"}:
            try:
                base.syntax_audit(changed, v5, reviewed, extra)
            except AssertionError:
                outcomes[label] += "_and_independent_runtime_syntax_audit"
            else:
                raise AssertionError((label, "Runtime syntax gate failed"))
    for index, (owner, current, _) in enumerate(SITES):
        changed = copy.deepcopy(tree)
        fn = base.named(base.named(changed, owner), "__call__")
        change = ReplaceCall(current, current.replace("=True", "=False"))
        change.visit(fn)
        assert change.count == 1
        try:
            inverse_transpose(changed, reference)
        except AssertionError:
            outcomes[f"site_{index}_transpose_flag"] = "rejected"
        else:
            raise AssertionError(index)
    for label in ("reader_kernel_change", "compute_kernel_change", "wrong_call_site"):
        changed = copy.deepcopy(tree)
        if label == "wrong_call_site":
            cls = base.named(changed, "Chunk64OutputGraph")
            fn = base.named(cls, "__call__")
            fn.body.insert(0, ast.parse("self.transpose_mm(q, k, transpose_b=True)").body[0])
        else:
            name = "TRANSPOSE_CORRECTED_MM_READER" if label == "reader_kernel_change" else "FUSED_CORRECTED_MM_COMPUTE"
            next(n for n in changed.body if composed.node_name(n) == name).value.value += "\n// changed\n"
        try:
            inverse_transpose(changed, reference)
        except AssertionError:
            outcomes[label] = "rejected"
        else:
            raise AssertionError(label)
    return outcomes


def audit(source):
    assert base.sha(decay.__file__) == AUDITOR_SHA
    assert base.sha(E / "v6_decay_44f87_auditor.py.snapshot") == AUDITOR_SHA
    assert base.sha(REFERENCE) == REFERENCE_SHA and base.sha(CANDIDATE) == CANDIDATE_SHA
    reference, candidate = [ast.parse(p.read_text()) for p in (REFERENCE, CANDIDATE)]
    tree = ast.parse(Path(source).read_text())
    restoration = inverse_transpose(tree, reference)
    assert base.digest(tree) == base.digest(candidate)
    result = decay.audit(REFERENCE)
    v5, reviewed, functional = base.reviewed_sources()
    extra = []
    for filename, expected in base.EXTRA_REVIEWED.items():
        assert base.sha(E / filename) == expected
        extra.append(ast.parse((E / filename).read_text()))
    extra.extend([reference, candidate])
    result["syntax"] = base.syntax_audit(tree, v5, reviewed, extra)
    result["contract"] = base.contract_audit(tree, v5, functional)
    # Entire FusedDecoder execution AST is identical, so all planner/dispatch/KV
    # methods and every full-kind branch retain the inherited mock proof.
    result["transpose_restoration"] = restoration
    result["transpose_negative_controls"] = negative_controls(tree, reference, v5, reviewed, extra)
    result["source"] = str(Path(source).resolve())
    result["source_sha256"] = base.sha(source)
    result["upstream_decay_audit_source_sha256"] = AUDITOR_SHA
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
    assert not args.output.exists(), "Preserve earlier evidence"
    result = audit(args.snapshot)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("FUSED_TRANSPOSE_AUDIT_PASS", result["source_sha256"])


if __name__ == "__main__":
    main()
