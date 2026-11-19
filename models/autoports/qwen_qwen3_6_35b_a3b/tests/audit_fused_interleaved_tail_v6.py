# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only exact inverse audit of the interleaved weighted-norm input reader."""

import argparse
import ast
import copy
import hashlib
import json
import sys
from pathlib import Path

from models.autoports.qwen_qwen3_6_35b_a3b.tests import audit_fused_transpose_v6 as transpose

base, composed = transpose.base, transpose.composed
ROOT, E = transpose.ROOT, transpose.E
AUDITOR_SHA = "e6e1de653bba87d2554011a82ca53b613c58aa5df92a8473f9d29067a7b0119d"
REFERENCE, REFERENCE_SHA = transpose.CANDIDATE, transpose.CANDIDATE_SHA
CANDIDATE = E / "v6_transport_conv_norm_decay_transpose_interleaved_tail_integration.py.snapshot"
CANDIDATE_SHA = "d8106611f65edccdf069baab44acc9c0088baa709090b017a6a6df75f0a7bb87"
HELPER_SHA = "ea3a9bd0d5f1c673822f6454a0535f025af17659d1ac64871a6c9a4ac893d2f4"
CALL_SHA = "0bb1cae71b4aed64e5d8311ce6b080070c0e29e2bca3c3e5faddb875ed03bfdf"
INIT_SHA = "582b69dfbed36c930b713be87cf143df1f0f4f57bf2c0fe50cbb127278ea6045"
READER_SHA = "fdfe6321306b121e20bcde3b4cc868ac81121e83add1f6786cc11e0956651d53"
COMPUTE_SHA = "348f1d9a71ca559c08875853bfe07ab4c889ccd05eeb9753720eb812bd43f7a7"
READER = "WEIGHTED_NORM_INTERLEAVED_READER"
COMPUTE = "WEIGHTED_NORM_COMPUTE_SOURCE"
PUBLICATION = " core.reserve_back(4); core.push_back(4);\n sums.reserve_back(1); sums.push_back(1);\n"


def assignment(tree, name):
    nodes = [n for n in tree.body if composed.node_name(n) == name]
    assert len(nodes) == 1, (name, len(nodes))
    assert isinstance(nodes[0], ast.Assign)
    return nodes[0]


def inverse_interleaved_tail(tree, reference):
    helper = base.named(tree, "WeightedNormTail")
    previous = base.named(reference, "WeightedNormTail")
    assert composed.exact(helper) == HELPER_SHA
    call, init = base.named(helper, "__call__"), base.named(helper, "__init__")
    assert composed.exact(call) == CALL_SHA
    assert composed.exact(init) == INIT_SHA == composed.exact(base.named(previous, "__init__"))
    assert "WeightedNormTail.__call__" not in base.SETUP
    assert composed.node_name(tree.body[-1]) == READER
    reader, compute = assignment(tree, READER), assignment(tree, COMPUTE)
    assert hashlib.sha256(ast.literal_eval(reader.value).encode()).hexdigest() == READER_SHA
    assert hashlib.sha256(ast.literal_eval(compute.value).encode()).hexdigest() == COMPUTE_SHA
    old_compute = ast.literal_eval(assignment(reference, COMPUTE).value)
    assert old_compute.count(PUBLICATION) == 1
    assert ast.literal_eval(compute.value) == old_compute.replace(PUBLICATION, "", 1)
    restored = copy.deepcopy(tree)
    del restored.body[-1]
    restored_helper = base.named(restored, "WeightedNormTail")
    restored_helper.body[restored_helper.body.index(base.named(restored_helper, "__call__"))] = copy.deepcopy(
        base.named(previous, "__call__")
    )
    assignment(restored, COMPUTE).value = copy.deepcopy(assignment(reference, COMPUTE).value)
    assert composed.exact(restored) == composed.exact(reference), "Change outside exact weighted call/compute/reader"
    for name in ("FusedDecoder", "LocalSiluMultiply"):
        assert composed.exact(base.named(tree, name)) == composed.exact(base.named(reference, name)), name
    return dict(
        restored_module_ast_sha256=composed.exact(restored),
        full_module_exact_after_only_region_inverse=True,
        inverse_regions=["WeightedNormTail.__call__", COMPUTE, "appended " + READER],
        helper_ast_sha256=HELPER_SHA,
        runtime_call_ast_sha256=CALL_SHA,
        unchanged_constructor_ast_sha256=INIT_SHA,
        reader_string_sha256=READER_SHA,
        compute_string_sha256=COMPUTE_SHA,
        compute_arithmetic_unchanged_after_only_two_input_publications_removed=True,
        fused_decoder_and_local_silu_ast_unchanged=True,
        public_planner_kv_paged_batch_and_full_kind_contracts_inherited_exact=True,
        new_setup_exemptions=[],
        whole_method_or_class_exemption=False,
    )


def negative_controls(tree, reference, v5, reviewed, extra):
    outcomes = {}

    def reject(label, changed, syntax=False):
        try:
            inverse_interleaved_tail(changed, reference)
        except AssertionError:
            outcomes[label] = "rejected_by_exact_inverse_proof"
        else:
            raise AssertionError(label)
        if syntax:
            try:
                base.syntax_audit(changed, v5, reviewed, extra)
            except AssertionError:
                outcomes[label] += "_and_independent_runtime_syntax_audit"
            else:
                raise AssertionError((label, "Runtime syntax gate failed"))

    for label, owner, method, statement, syntax in (
        ("runtime_readback", "WeightedNormTail", "__call__", "self.t.to_torch(core)", True),
        ("runtime_upload", "WeightedNormTail", "__call__", "self.t.from_torch(core)", True),
        ("runtime_dynamic_exec", "WeightedNormTail", "__call__", "exec('pass')", True),
        ("constructor_extra_upload", "WeightedNormTail", "__init__", "t.from_torch(gamma)", True),
        ("constructor_extra_allocation", "WeightedNormTail", "__init__", "t.empty((32, 32))", True),
        ("decoder_change", "FusedDecoder", "_linear_attention", "unexpected()", False),
    ):
        changed = copy.deepcopy(tree)
        base.named(base.named(changed, owner), method).body.insert(0, ast.parse(statement).body[0])
        reject(label, changed, syntax)
    for label, name, old, new in (
        ("reader_page_count", READER, "tile<4", "tile<3"),
        ("reader_byte_stride", READER, "tile*4096", "tile*2048"),
        ("reader_missing_barrier", READER, "noc.async_read_barrier();", ""),
        ("compute_rounding_change", COMPUTE, "bits+0x7fff", "bits+0x7ffe"),
        ("compute_duplicate_publication", COMPUTE, " gamma.reserve_back(4);", PUBLICATION + " gamma.reserve_back(4);"),
    ):
        changed = copy.deepcopy(tree)
        node = assignment(changed, name)
        assert node.value.value.count(old) == 1
        node.value.value = node.value.value.replace(old, new, 1)
        reject(label, changed)
    for label, old, new in (
        (
            "wrong_runtime_buffers",
            "[core.buffer_address(), sums.buffer_address()]",
            "[sums.buffer_address(), core.buffer_address()]",
        ),
        ("wrong_cb_capacity", "pages * 4096", "pages * 2048"),
        (
            "removed_input_memory_guard",
            "assert sums.memory_config() in (t.L1_MEMORY_CONFIG, t.DRAM_MEMORY_CONFIG)",
            "pass",
        ),
    ):
        changed = copy.deepcopy(tree)
        helper = base.named(changed, "WeightedNormTail")
        method = base.named(helper, "__call__")
        text = ast.unparse(method)
        assert text.count(old) == 1
        helper.body[helper.body.index(method)] = ast.parse(text.replace(old, new, 1)).body[0]
        reject(label, changed)
    return outcomes


def audit(source):
    assert base.sha(transpose.__file__) == AUDITOR_SHA
    assert base.sha(E / "v6_transpose_85ad_auditor.py.snapshot") == AUDITOR_SHA
    assert base.sha(REFERENCE) == REFERENCE_SHA and base.sha(CANDIDATE) == CANDIDATE_SHA
    reference, candidate = [ast.parse(p.read_text()) for p in (REFERENCE, CANDIDATE)]
    tree = ast.parse(Path(source).read_text())
    restoration = inverse_interleaved_tail(tree, reference)
    assert base.digest(tree) == base.digest(candidate)
    result = transpose.audit(REFERENCE)
    v5, reviewed, functional = base.reviewed_sources()
    extra = []
    for filename, expected in base.EXTRA_REVIEWED.items():
        assert base.sha(E / filename) == expected
        extra.append(ast.parse((E / filename).read_text()))
    extra.extend([reference, candidate])
    result["syntax"] = base.syntax_audit(tree, v5, reviewed, extra)
    result["contract"] = base.contract_audit(tree, v5, functional)
    result["interleaved_tail_restoration"] = restoration
    result["interleaved_tail_negative_controls"] = negative_controls(tree, reference, v5, reviewed, extra)
    result["source"] = str(Path(source).resolve())
    result["source_sha256"] = base.sha(source)
    result["upstream_transpose_audit_source_sha256"] = AUDITOR_SHA
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
    print("FUSED_INTERLEAVED_TAIL_AUDIT_PASS", result["source_sha256"])


if __name__ == "__main__":
    main()
