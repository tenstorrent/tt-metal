# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Embed the exact mixed-transport convolution in a new standalone snapshot.

Only public decode's convolution region changes. The original branch is retained
verbatim in AST for every prefill call, including singleton/internal decode calls
with cache_boundary=False. No production file or hardware is accessed.
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
COMPONENT = T / "v6_decode_conv_mixed_transport_probe.py"
COMPONENT_SHA = "4f1e2a409a889c40f2f15d0edbe8492f505049058f3152d9f7becab61f59d44f"
BASE = E / "v6_corrected_fma_full_v5_triangular_compact_silu_formatted_integration.py.snapshot"
BASE_SHA = "ab28a91bb3cb5e8b88e5371da0defe3b0d731e86a131c8523483e4b6b65b0334"
TRANSPORT = E / "v6_ab28_transport_combined.py.snapshot"
TRANSPORT_SHA = "13f78cd31929b49bd82bcf12723bccc54e9cbad19544370daafafb93e6de77e4"
BLACK = "/home/ttuser/.cache/pre-commit/repoefelr92y/py_env-python3/bin/black"
ISORT = "/home/ttuser/.cache/pre-commit/repodrn1fio9/py_env-python3/bin/isort"
AUTOFLAKE = "/home/ttuser/.cache/pre-commit/repodje5jyzf/py_env-python3/bin/autoflake"
NAMES = dict(
    DecodeConv="DecodeConvolution",
    READER="DECODE_CONV_READER",
    COMPUTE="DECODE_CONV_COMPUTE",
    WRITER="DECODE_CONV_WRITER",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(tree):
    return hashlib.sha256(ast.dump(tree).encode()).hexdigest()


def decoder(tree):
    return next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FusedDecoder")


def method(cls, name):
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)


def build(args):
    assert not args.output.exists() and not args.report.exists()
    assert sha(BASE) == BASE_SHA and sha(TRANSPORT) == TRANSPORT_SHA and sha(COMPONENT) == COMPONENT_SHA
    base, base_sha = (TRANSPORT, TRANSPORT_SHA) if args.transport else (BASE, BASE_SHA)
    spec = importlib.util.spec_from_file_location("frozen_decode_conv", COMPONENT)
    component = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(component)
    gate_path = E / "v6_decode_conv_mixed_transport_all_device.json"
    gate = json.loads(gate_path.read_text())
    assert gate["source_sha256"] == COMPONENT_SHA and gate["status"] == "component_exact" and gate["device_closed"]
    assert len(gate["cases"]) == 12
    for row in gate["cases"]:
        assert row["output"]["bits_different"] == row["carry"]["bits_different"] == 0
        assert len(row["partial"]) == 4 and all(p["bits_different"] == 0 for p in row["partial"])
    original = ast.parse(base.read_text())
    canonical = ast.parse(BASE.read_text())
    # Transport composition changes only disjoint methods; convolution/setup are
    # identical in both input snapshots before this builder touches either one.
    for name in ("from_state_dict", "_linear_attention"):
        assert digest(method(decoder(original), name)) == digest(method(decoder(canonical), name))
    candidate = copy.deepcopy(original)
    cls = decoder(candidate)
    setup = method(cls, "from_state_dict")
    linear = method(cls, "_linear_attention")
    old_setup, old_linear = copy.deepcopy(setup), copy.deepcopy(linear)
    added_setup = ast.parse(
        'if self.kind == "linear_attention":\n self.decode_convolution = DecodeConvolution(ttnn, mesh_device, self.taps_fp32)\n'
    ).body[0]
    assert isinstance(setup.body[-1], ast.Return) and ast.unparse(setup.body[-1].value) == "self"
    setup.body.insert(len(setup.body) - 1, added_setup)
    start = next(
        i
        for i, n in enumerate(linear.body)
        if isinstance(n, ast.Assign) and any(isinstance(x, ast.Name) and x.id == "history" for x in n.targets)
    )
    end = next(
        i
        for i, n in enumerate(linear.body[start:], start)
        if isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Call)
        and ast.unparse(n.value.func) == "ttnn.slice"
        and any(k.arg == "output_tensor" for k in n.value.keywords)
    )
    original_block = copy.deepcopy(linear.body[start : end + 1])
    _, component_native_sha = component.native_function(None)
    assert digest(ast.Module(original_block, type_ignores=[])) == component_native_sha
    branch = ast.parse(
        'if decode and cache_boundary:\n conv, _conv_partial = self.decode_convolution(qkv, state["conv"], debug=False)\n q, k, v = (conv[:, :, :2048], conv[:, :, 2048:4096], conv[:, :, 4096:])\n'
    ).body[0]
    branch.orelse = copy.deepcopy(original_block)
    original_split = next(
        n for n in original_block if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "(q, k, v)"
    )
    assert digest(branch.body[1]) == digest(original_split)
    linear.body[start : end + 1] = [branch]

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
    assert len(helpers) == 4
    helper_class = next(n for n in helpers if isinstance(n, ast.ClassDef))
    assert not any(isinstance(n, ast.Call) for n in ast.walk(method(helper_class, "__init__")))
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
    assert digest(emitted) == digest(candidate), "Formatting changed executable AST"
    for command in (
        [AUTOFLAKE, "--remove-all-unused-imports", "--check"],
        [ISORT, "--check-only"],
        [BLACK, "--check", "--line-length", "120"],
    ):
        subprocess.run(command + [str(args.output)], check=True, capture_output=True)
    # Restore precise inserted regions, without replacing the whole methods.
    restored = copy.deepcopy(emitted)
    assert [digest(n) for n in restored.body[-4:]] == [digest(n) for n in helpers]
    restored.body = restored.body[:-4]
    restored_setup = method(decoder(restored), "from_state_dict")
    assert digest(restored_setup.body[-2]) == digest(added_setup)
    del restored_setup.body[-2]
    restored_linear = method(decoder(restored), "_linear_attention")
    assert digest(restored_linear.body[start]) == digest(branch)
    restored_linear.body[start : start + 1] = copy.deepcopy(restored_linear.body[start].orelse)
    assert digest(restored_setup) == digest(old_setup)
    assert digest(restored_linear) == digest(old_linear)
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
        component_gate=dict(path=str(gate_path), sha256=sha(gate_path), cases=12),
        output=str(args.output),
        output_sha256=sha(args.output),
        transport_composed=args.transport,
        baseline_component_ast_sha256=component_native_sha,
        exact_region_restoration=True,
        full_module_restoration=True,
        formatting_ast_exact=True,
        pinned_hook_checks=True,
        changed_methods=["from_state_dict", "_linear_attention"],
        setup_change="Append a nonallocating DecodeConvolution object only for linear_attention; all existing setup is identical.",
        runtime_change="if decode and cache_boundary: one measured generic operation plus identical Q/K/V slices; else original convolution block AST.",
        public_singleton_prefill_unchanged=True,
        full_attention_setup_branch_unchanged=True,
        component_setup_and_linear_transfer_from_ab28_exact=True,
        added_names=list(NAMES.values()),
        kernel_sha256={
            name: hashlib.sha256(getattr(component, name).encode()).hexdigest()
            for name in ("READER", "COMPUTE", "WRITER")
        },
        unchanged_method_ast_sha256={
            n.name: digest(n)
            for n in decoder(original).body
            if isinstance(n, ast.FunctionDef) and n.name not in ("from_state_dict", "_linear_attention")
        },
        base_ast_sha256=digest(original),
        candidate_ast_sha256=digest(emitted),
        persistent_l1_added_bytes=0,
        temporary_cb_bytes_per_core=32768,
        context_contract_changed=False,
        gate_scope="Component exactness only; independent/combined decoder accuracy, trace/performance and capacity gates remain required.",
    )
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print("DECODE_CONV_SNAPSHOT_READY", report["output_sha256"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--transport", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    build(p.parse_args())


if __name__ == "__main__":
    main()
