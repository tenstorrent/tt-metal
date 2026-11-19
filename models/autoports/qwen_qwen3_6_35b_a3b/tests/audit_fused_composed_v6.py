# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only supplemental audit of the reviewed transport/convolution/norm graph.

No production entrypoint or upstream audit configuration is modified. New helpers
remain subject to the original runtime checker, including their constructors.
"""

import argparse
import ast
import copy
import hashlib
import json
import sys
from pathlib import Path

from models.autoports.qwen_qwen3_6_35b_a3b.tests import audit_fused_snapshot_v6 as base
from models.autoports.qwen_qwen3_6_35b_a3b.tests import audit_fused_transport_v6 as transport

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "doc/fused_decoder/endurance"
PINS = {
    "v6_corrected_fma_full_v5_triangular_compact_silu_formatted_integration.py.snapshot": "ab28a91bb3cb5e8b88e5371da0defe3b0d731e86a131c8523483e4b6b65b0334",
    "v6_ab28_transport_combined.py.snapshot": "13f78cd31929b49bd82bcf12723bccc54e9cbad19544370daafafb93e6de77e4",
    "v6_transport_decode_conv_complete_integration.py.snapshot": "b253be4403326047c86d94b5e460f6a6e32bfb43373ba85af6f491cd05df1257",
    "v6_transport_conv_norm_terminal_scan_integration.py.snapshot": "803685e105d7ef1d46a6f9a8ceafb1758d9b1131e1c8c35aed1ab05cbf1475a5",
}
AUDITOR_PINS = {
    "audit_fused_snapshot_v6.py": "ab9bca957f8b10215aaa2b4681b6c10874e7941c1d3279e2cbb44e4cfdaf80e8",
    "audit_fused_transport_v6.py": "159bf733d9dd6586eadb07010f4173522538b0c6a7eeb65aee3c1690c1ef5029",
}
COMPAT_SHA = "fe5a3db52da5678a8d8041cd5078ff89f55838256bd9ec3c93fae220e87a085e"
CLASS_PINS = {
    "DecodeConvolution": "cda79c378b273363ffcd687819f48963304d3c3a7fe21d782197c67a9d544cb9",
    "NormTerminalScan": "dda73a29d720375872db2cc1187d4361f18bf3545233c520d99d14919589635f",
}
KERNEL_PINS = {
    "DECODE_CONV_READER": "bc0139bd897d23395acede25c4b6ed90331ef76fbb0cccf229bd6d7ea0018373",
    "DECODE_CONV_COMPUTE": "6aa628373e13c6e6477e9983e225b4a0eb3120acaa02765daab6f0c820568aeb",
    "DECODE_CONV_WRITER": "77ad412be12218ce73d1001a47067aa73e1182307550709df175e21169d45cbf",
    "NORM_TERMINAL_READER": "2ae38c4615e02eacd05bbddf36bf15e53e4507cd3cb4d18acadd318c9c3ee886",
    "NORM_TERMINAL_COMPUTE": "657b8cdd1652325ea668d536b6ebffb906c7bce351f55663fb00676c1f1cf291",
    "NORM_TERMINAL_WRITER": "26f341f43924580294128d62646bc849db5742289e845236bc5d4338c0d4a232",
}
SETUP_STATEMENTS = {
    "conv": "if self.kind == 'linear_attention':\n self.decode_convolution = DecodeConvolution(ttnn, mesh_device, self.taps_fp32)",
    "norm": "if self.kind == 'linear_attention':\n self.norm_terminal_scan = NormTerminalScan(ttnn, mesh_device)",
}


def exact(node):
    return hashlib.sha256(ast.dump(node).encode()).hexdigest()


def node_name(node):
    if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
        return node.name
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    return None


def verify_helpers(tree):
    found = {}
    for name, expected in CLASS_PINS.items():
        cls = base.named(tree, name)
        assert exact(cls) == expected, ("Unreviewed helper", name)
        init = base.named(cls, "__init__")
        calls = [ast.unparse(n.func) for n in ast.walk(init) if isinstance(n, ast.Call)]
        assert calls == ([] if name == "DecodeConvolution" else ["mesh.compute_with_storage_grid_size"])
        assert name + ".__init__" not in base.SETUP
        found[name] = dict(ast_sha256=exact(cls), constructor_ast_sha256=exact(init), constructor_calls=calls)
    for name, expected in KERNEL_PINS.items():
        nodes = [n for n in tree.body if node_name(n) == name]
        assert len(nodes) == 1
        value = ast.literal_eval(nodes[0].value)
        assert isinstance(value, str) and hashlib.sha256(value.encode()).hexdigest() == expected, name
    return dict(classes=found, kernel_string_sha256=KERNEL_PINS, additional_setup_exemptions=[])


def remove_tail(tree, names):
    assert {node_name(n) for n in tree.body[-4:]} == set(names)
    del tree.body[-4:]


def remove_setup(tree, kind):
    setup = transport.method(tree, "from_state_dict")
    expected = ast.parse(SETUP_STATEMENTS[kind]).body[0]
    assert exact(setup.body[-2]) == exact(expected)
    del setup.body[-2]


def inverse_chain(tree, refs):
    """Invert only identified regions; any other executable AST delta rejects."""
    ab28, t13, conv, combined = refs
    verify_helpers(tree)
    restored = copy.deepcopy(tree)
    remove_tail(restored, ["NormTerminalScan", *[n for n in KERNEL_PINS if n.startswith("NORM_")]])
    remove_setup(restored, "norm")
    norm = transport.method(restored, "norm")
    branch = next(n for n in norm.body if isinstance(n, ast.If) and ast.unparse(n.test).startswith("key in ("))
    original_branch = next(
        n
        for n in transport.method(conv, "norm").body
        if isinstance(n, ast.If) and ast.unparse(n.test).startswith("key in (")
    )
    sites = [i for i, n in enumerate(branch.body) if ast.unparse(n) == "total = self.norm_terminal_scan(groups)"]
    assert len(sites) == 1
    i = sites[0]
    originals = original_branch.body[i : i + 2]
    assert [ast.unparse(n.targets[0]) for n in originals] == ["lanes", "total"]
    branch.body[i : i + 1] = copy.deepcopy(originals)
    assert exact(restored) == exact(conv), "Outside exact norm region"
    norm_restoration = exact(restored)
    remove_tail(restored, ["DecodeConvolution", *[n for n in KERNEL_PINS if n.startswith("DECODE_")]])
    remove_setup(restored, "conv")
    linear = transport.method(restored, "_linear_attention")
    sites = [
        i
        for i, n in enumerate(linear.body)
        if isinstance(n, ast.If)
        and any(
            isinstance(c, ast.Attribute) and c.attr == "decode_convolution"
            for c in ast.walk(ast.Module(body=n.body, type_ignores=[]))
        )
    ]
    assert len(sites) == 1
    i = sites[0]
    branch = linear.body[i]
    expected = ast.parse(
        "if decode and cache_boundary:\n conv, _conv_partial = self.decode_convolution(qkv, state['conv'], debug=False)\n q, k, v = (conv[:, :, :2048], conv[:, :, 2048:4096], conv[:, :, 4096:])"
    ).body[0]
    assert exact(branch.test) == exact(expected.test)
    assert [exact(n) for n in branch.body] == [exact(n) for n in expected.body]
    assert branch.orelse
    linear.body[i : i + 1] = copy.deepcopy(branch.orelse)
    assert exact(restored) == exact(t13), "Outside exact convolution region"
    transport_proof = transport.inverse_transport(restored, ab28, "combined")
    return dict(
        norm_inverse_module_ast_sha256=norm_restoration,
        convolution_inverse_module_ast_sha256=exact(restored),
        transport=transport_proof,
        full_module_exact_after_only_region_inverse=True,
        additional_setup_exemptions=[],
        whole_method_or_class_exemption=False,
    )


def negative_controls(tree, refs, v5, reviewed, extra):
    outcomes = {}
    for label, owner, method, statement in (
        ("conv_runtime_readback", "DecodeConvolution", "__call__", "self.t.to_torch(qkv)"),
        ("conv_constructor_upload", "DecodeConvolution", "__init__", "t.from_torch(taps)"),
        ("norm_runtime_upload", "NormTerminalScan", "__call__", "self.t.from_torch(groups)"),
        ("norm_constructor_readback", "NormTerminalScan", "__init__", "ttnn.to_torch(mesh)"),
        ("norm_arithmetic_change", "FusedDecoder", "norm", "x = ttnn.multiply(x, 2.0)"),
        ("setup_extra_call", "FusedDecoder", "from_state_dict", "unexpected()"),
        (
            "public_alignment_restriction",
            "FusedDecoder",
            "prepare_prefill",
            "if seq_len % 64: raise ValueError('aligned only')",
        ),
    ):
        changed = copy.deepcopy(tree)
        base.named(base.named(changed, owner), method).body.insert(0, ast.parse(statement).body[0])
        try:
            inverse_chain(changed, refs)
        except (AssertionError, StopIteration):
            outcomes[label] = "rejected_by_exact_region_proof"
        else:
            raise AssertionError(label)
        if owner in CLASS_PINS:
            try:
                base.syntax_audit(changed, v5, reviewed, extra)
            except AssertionError:
                outcomes[label] += "_and_independent_runtime_syntax_audit"
            else:
                raise AssertionError((label, "Runtime checker failed"))
    for name in KERNEL_PINS:
        changed = copy.deepcopy(tree)
        next(n for n in changed.body if node_name(n) == name).value.value += "\n// altered kernel\n"
        try:
            inverse_chain(changed, refs)
        except AssertionError:
            outcomes[name + "_mutation"] = "rejected"
        else:
            raise AssertionError(name)
    changed = copy.deepcopy(tree)
    branch = next(
        n
        for n in transport.method(changed, "_linear_attention").body
        if isinstance(n, ast.If) and "decode_convolution" in ast.unparse(n)
    )
    branch.test = ast.parse("decode", mode="eval").body
    try:
        inverse_chain(changed, refs)
    except AssertionError:
        outcomes["singleton_prefill_guard_change"] = "rejected"
    else:
        raise AssertionError("Guard mutation accepted")
    return outcomes


def audit(source):
    for filename, expected in AUDITOR_PINS.items():
        assert base.sha(ROOT / "tests" / filename) == expected
    assert base.sha(E / "source_format_compatibility/manifest.json") == COMPAT_SHA
    refs = []
    for filename, expected in PINS.items():
        assert base.sha(E / filename) == expected
        refs.append(ast.parse((E / filename).read_text()))
    tree = ast.parse(Path(source).read_text())
    restoration = inverse_chain(tree, refs)
    assert base.digest(tree) == base.digest(refs[-1]), "Source differs from reviewed composed execution AST"
    v5, reviewed, functional = base.reviewed_sources()
    extra, extra_hashes = [], {}
    for filename, expected in base.EXTRA_REVIEWED.items():
        path = E / filename
        assert base.sha(path) == expected
        extra.append(ast.parse(path.read_text()))
        extra_hashes[str(path)] = expected
    # The only newly reviewed setup-method hash is from_state_dict, after exact
    # inverse proof of its two nonallocating linear-only constructor additions.
    # No new name enters SETUP; both new constructors are runtime-audited.
    extra.append(refs[-1])
    extra_hashes[str(E / list(PINS)[-1])] = list(PINS.values())[-1]
    syntax = base.syntax_audit(tree, v5, reviewed, extra)
    contract = base.contract_audit(tree, v5, functional)
    context_path = ROOT / "doc/context_contract.json"
    context = json.loads(context_path.read_text())
    assert context["hf_advertised_context"] == context["supported_context"] == 262144
    assert context["capability_reduction"] is None
    planners = base.planner_audit(tree, v5, contract["v6"], context["supported_context"])
    dispatch = base.dispatch_audit(tree, v5, reviewed, context["supported_context"])
    negative = base.negative_audit(v5, reviewed, functional, extra)
    extra_negative = negative_controls(tree, refs, v5, reviewed, extra)
    transport_negative = transport.negative_controls(refs[1], refs[0])
    assert "ttnn" not in sys.modules and "torch" not in sys.modules
    return dict(
        status="static_audit_pass",
        python_audit_pass=True,
        acceptance_eligible=False,
        source=str(Path(source).resolve()),
        source_sha256=base.sha(source),
        syntax=syntax,
        contract=contract,
        planner=planners,
        mock_dispatch=dispatch,
        negative_controls=negative,
        composed_negative_controls=extra_negative,
        transport_negative_controls=transport_negative,
        composed_restoration=restoration,
        composed_helpers=verify_helpers(tree),
        audit_source_sha256=base.sha(__file__),
        upstream_auditor_sha256=AUDITOR_PINS,
        compatibility_manifest_sha256=COMPAT_SHA,
        reviewed_extension_hashes=extra_hashes,
        reviewed_extension=base.extension_audit(tree, extra),
        description_normalization=base.description_delta(tree, [(str(E / list(PINS)[-1]), refs[-1])]),
        reference_hashes={str(E / k): v for k, v in PINS.items()},
        context_contract_evidence_matches_current_source=context.get("fused_decoder_validation", {}).get(
            "decoder_source_sha256"
        )
        == base.sha(source),
        context_contract_sha256=base.sha(context_path),
        limitations=[
            "Host-only source/AST and mock planner checks do not execute TTNN, kernels or allocations.",
            "Unchanged PCC, stress/determinism, traced performance, watcher and physical context-capacity device gates remain required.",
            "Python syntax audit excludes known host fallbacks; arbitrary native-library behavior is not proven by AST inspection.",
            "No setup allowance added: only the reviewed from_state_dict hash changes after exact inverse proof; new helper constructors remain runtime-audited.",
        ],
        device_executed=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=ROOT / "tt/fused_decoder.py")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "Preserve earlier evidence"
    result = audit(args.snapshot)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("FUSED_COMPOSED_AUDIT_PASS", result["source_sha256"], result["planner"]["valid_plan_cases"])


if __name__ == "__main__":
    main()
