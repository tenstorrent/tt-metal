# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Embed the measured exact terminal norm scans in a fresh standalone candidate.

Host/source only. No production edits or hardware. The independent ab28 candidate
and transport/convolution composition both preserve every full-attention branch.
"""

import argparse
import ast
import copy
import hashlib
import json
import subprocess
from pathlib import Path

T = Path(__file__).resolve().parent
E = T.parent / "doc/fused_decoder/endurance"
COMPONENT = T / "v6_norm_terminal_scan_probe.py"
COMPONENT_SHA = "85d00325f34ab3ac803fbc9b4acb917b9bd6d3decacb4b3d15c0c52c8104408c"
BASE = E / "v6_corrected_fma_full_v5_triangular_compact_silu_formatted_integration.py.snapshot"
BASE_SHA = "ab28a91bb3cb5e8b88e5371da0defe3b0d731e86a131c8523483e4b6b65b0334"
COMPOSED = E / "v6_transport_decode_conv_complete_integration.py.snapshot"
COMPOSED_SHA = "b253be4403326047c86d94b5e460f6a6e32bfb43373ba85af6f491cd05df1257"
GATE = E / "v6_norm_terminal_scan_device.json"
BLACK = "/home/ttuser/.cache/pre-commit/repoefelr92y/py_env-python3/bin/black"
AUTOFLAKE = "/home/ttuser/.cache/pre-commit/repodje5jyzf/py_env-python3/bin/autoflake"
ISORT = "/home/ttuser/.cache/pre-commit/repodrn1fio9/py_env-python3/bin/isort"
NAMES = dict(
    NormTerminalScan="NormTerminalScan",
    READER="NORM_TERMINAL_READER",
    COMPUTE="NORM_TERMINAL_COMPUTE",
    WRITER="NORM_TERMINAL_WRITER",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(node):
    return hashlib.sha256(ast.dump(node).encode()).hexdigest()


def named(tree, name):
    return next(node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name)


class Rename(ast.NodeTransformer):
    def __init__(self, mapping):
        self.mapping = mapping

    def visit_Name(self, node):
        node.id = self.mapping.get(node.id, node.id)
        return node


def build(args):
    assert not args.output.exists() and not args.report.exists(), "Preserve earlier evidence"
    for path, expected in [(BASE, BASE_SHA), (COMPOSED, COMPOSED_SHA), (COMPONENT, COMPONENT_SHA)]:
        assert sha(path) == expected, (path, "Frozen source mismatch")
    gate = json.loads(GATE.read_text())
    assert gate["source_sha256"] == COMPONENT_SHA and gate["snapshot_sha256"] == BASE_SHA
    assert gate["status"] == "all_exact_component_pass" and gate["device_closed"]
    assert len(gate["cases"]) == 19 and len(gate["synthetic_cases"]) == 2
    for row in gate["cases"]:
        assert all(
            row[key]["exact"] and row[key]["differing_elements"] == 0
            for key in ("lanes_fp32", "total_fp32", "complete_norm_bf16")
        )
    for row in gate["synthetic_cases"]:
        assert all(
            row[key]["exact"] and row[key]["differing_elements"] == 0
            for key in ("lanes_fp32", "total_fp32", "total_bf16")
        )
    for case in gate["timing"].values():
        for data in case.values():
            assert data["replays"] == 100 and data["after_replay_bitdiff"]["exact"]
    base, base_sha = (COMPOSED, COMPOSED_SHA) if args.compose else (BASE, BASE_SHA)
    original = ast.parse(base.read_text())
    canonical = ast.parse(BASE.read_text())
    assert digest(named(named(original, "FusedDecoder"), "norm")) == digest(
        named(named(canonical, "FusedDecoder"), "norm")
    )
    candidate = copy.deepcopy(original)
    cls = named(candidate, "FusedDecoder")
    setup, norm = named(cls, "from_state_dict"), named(cls, "norm")
    old_setup, old_norm = copy.deepcopy(setup), copy.deepcopy(norm)
    branch_index = next(
        i for i, n in enumerate(norm.body) if isinstance(n, ast.If) and ast.unparse(n.test).startswith("key in (")
    )
    branch = norm.body[branch_index]
    first = next(
        i for i, n in enumerate(branch.body) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "lanes"
    )
    removed = copy.deepcopy(branch.body[first : first + 2])
    assert [ast.unparse(n.targets[0]) for n in removed] == ["lanes", "total"]
    assert all("disable_compensated_sum=True" in ast.unparse(n) for n in removed)
    replacement = ast.parse("total = self.norm_terminal_scan(groups)").body[0]
    branch.body[first : first + 2] = [replacement]
    added_setup = ast.parse(
        'if self.kind == "linear_attention":\n self.norm_terminal_scan = NormTerminalScan(ttnn, mesh_device)\n'
    ).body[0]
    assert isinstance(setup.body[-1], ast.Return) and ast.unparse(setup.body[-1].value) == "self"
    setup.body.insert(len(setup.body) - 1, added_setup)
    helpers = []
    kernels = {}
    original_helper = None
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
            if isinstance(node, ast.ClassDef):
                original_helper = copy.deepcopy(node)
            else:
                kernels[name] = ast.literal_eval(node.value)
            helpers.append(Rename(NAMES).visit(copy.deepcopy(node)))
    assert len(helpers) == 4 and original_helper is not None
    helper = next(node for node in helpers if isinstance(node, ast.ClassDef))
    assert digest(Rename({v: k for k, v in NAMES.items()}).visit(copy.deepcopy(helper))) == digest(original_helper)
    init = named(helper, "__init__")
    calls = [ast.unparse(node.func) for node in ast.walk(init) if isinstance(node, ast.Call)]
    assert calls == ["mesh.compute_with_storage_grid_size"], calls
    assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(helper))
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
    restored = copy.deepcopy(emitted)
    assert [digest(node) for node in restored.body[-4:]] == [digest(node) for node in helpers]
    restored.body = restored.body[:-4]
    restored_setup = named(named(restored, "FusedDecoder"), "from_state_dict")
    assert digest(restored_setup.body[-2]) == digest(added_setup)
    del restored_setup.body[-2]
    restored_norm = named(named(restored, "FusedDecoder"), "norm")
    assert digest(restored_norm.body[branch_index].body[first]) == digest(replacement)
    restored_norm.body[branch_index].body[first : first + 1] = removed
    assert digest(restored_setup) == digest(old_setup) and digest(restored_norm) == digest(old_norm)
    assert digest(restored) == digest(original), "Precise delta restoration changed an unrelated node"
    assert digest(norm.body[0]) == digest(old_norm.body[0]), "Full-attention norm early branch changed"
    assert digest(setup.body[0]) == digest(old_setup.body[0]), "Full-attention setup early branch changed"
    kernel_paths = {}
    for name, code in kernels.items():
        path = Path(str(args.output) + "." + name.lower() + ".cpp.snapshot")
        assert not path.exists()
        path.write_text(code)
        kernel_paths[name] = str(path)
    builder = Path(str(args.output) + ".builder.py.snapshot")
    assert not builder.exists()
    builder.write_bytes(Path(__file__).read_bytes())
    report = dict(
        status="source_candidate_ready",
        device_executed=False,
        acceptance_eligible=False,
        source=str(Path(__file__)),
        source_sha256=sha(__file__),
        builder_snapshot=str(builder),
        base=str(base),
        base_sha256=base_sha,
        component_source=str(COMPONENT),
        component_sha256=COMPONENT_SHA,
        component_gate=dict(path=str(GATE), sha256=sha(GATE), cases=19, synthetic_cases=2, all_bit_exact=True),
        component_timing=gate["timing"],
        output=str(args.output),
        output_sha256=sha(args.output),
        transport_convolution_composed=args.compose,
        exact_region_restoration=True,
        full_module_restoration=True,
        formatting_ast_exact=True,
        pinned_hook_checks=True,
        changed_methods=["from_state_dict", "norm"],
        helper_alpha_equivalence=True,
        helper_ast_sha256=digest(helper),
        helper_init_ast_sha256=digest(init),
        setup_change="Append a linear-only nonallocating adapter; metadata grid query only, no tensor upload/creation.",
        runtime_change="Both tiny norm terminal cumsums/crops replaced for linear input and postnorm, all logical row counts; all other norm arithmetic and placement unchanged.",
        full_attention_norm_and_setup_early_branches_unchanged=True,
        public_prefill_semantics="All logical row counts remain valid; component covered rows1/3/33/65. Whole prefill timing and capacity must be requalified.",
        persistent_l1_added_bytes=0,
        persistent_dram_added_bytes=0,
        temporary_cb_bytes_per_active_core=7 * 4096,
        temporary_output_bytes="4096*rows FP32 TILE; same terminal tensor padded layout as baseline; no new persistent cache/state",
        context_contract_changed=False,
        added_names=list(NAMES.values()),
        kernel_sha256={name: hashlib.sha256(code.encode()).hexdigest() for name, code in kernels.items()},
        kernel_snapshots=kernel_paths,
        unchanged_method_ast_sha256={
            node.name: digest(node)
            for node in named(original, "FusedDecoder").body
            if isinstance(node, ast.FunctionDef) and node.name not in ("from_state_dict", "norm")
        },
        base_ast_sha256=digest(original),
        candidate_ast_sha256=digest(emitted),
        quality_gates_unchanged=True,
        limitations="Component traced norm improves ~13us, eager postnorm regresses; final whole decoder performance/prefill/capacity required. No runtime auditor exemptions requested.",
    )
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print("NORM_TERMINAL_SNAPSHOT_READY", report["output_sha256"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compose", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
