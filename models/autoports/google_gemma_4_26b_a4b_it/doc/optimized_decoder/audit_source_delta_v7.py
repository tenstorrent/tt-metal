# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only exact AST/source and setup-policy proof for full-prefill HiFi2."""

import ast
import copy
import difflib
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
OLD = ROOT / "minimal_pairs_v6/runtime_snapshot.py.txt"
NEW = ROOT.parent.parent / "tt/optimized_decoder.py"
OLD_HASH = "b585a21f0b66144f69a823fa2d1088b130e34928a65fc91a38fd1bc5c2526846"
NEW_HASH = "daa82a4a5197a007ccd5d29d912f082e695fd37a596578f25fba96a6694e625b"


def digest(value):
    return hashlib.sha256(value).hexdigest()


def methods(tree):
    return {
        f"{cls.name}.{method.name}": method
        for cls in tree.body
        if isinstance(cls, ast.ClassDef)
        for method in cls.body
        if isinstance(method, ast.FunctionDef)
    }


def dumped(node):
    return ast.dump(node, include_attributes=False)


def remove_keyword_argument(node, name):
    pairs = list(zip(node.args.kwonlyargs, node.args.kw_defaults))
    assert sum(argument.arg == name for argument, _ in pairs) == 1
    node.args.kwonlyargs = [argument for argument, _ in pairs if argument.arg != name]
    node.args.kw_defaults = [default for argument, default in pairs if argument.arg != name]


def main():
    old_source, new_source = OLD.read_bytes(), NEW.read_bytes()
    assert digest(old_source) == OLD_HASH and digest(new_source) == NEW_HASH
    old, new = methods(ast.parse(old_source)), methods(ast.parse(new_source))
    assert old.keys() == new.keys() and len(old) == 40
    changed = [name for name in old if dumped(old[name]) != dumped(new[name])]
    assert changed == ["OptimizedDecoder.from_state_dict", "MinimalPrefillQKV.__init__"]
    factory = copy.deepcopy(new[changed[0]])
    remove_keyword_argument(factory, "prefill_qkv_fidelity")
    auto = next(
        statement
        for statement in factory.body
        if isinstance(statement, ast.If)
        and isinstance(statement.test, ast.Compare)
        and isinstance(statement.test.left, ast.Name)
        and statement.test.left.id == "prefill_qkv_fidelity"
    )
    factory.body.remove(auto)
    projection_calls = [
        node
        for node in ast.walk(factory)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "MinimalPrefillQKV"
    ]
    assert len(projection_calls) == 1
    projection_calls[0].keywords = [keyword for keyword in projection_calls[0].keywords if keyword.arg != "fidelity"]
    assert dumped(factory) == dumped(old[changed[0]])
    constructor = copy.deepcopy(new[changed[1]])
    remove_keyword_argument(constructor, "fidelity")
    compute_assignment = next(
        statement
        for statement in constructor.body
        if isinstance(statement, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "compute" for target in statement.targets)
    )
    compute_branch = next(
        statement
        for statement in constructor.body
        if isinstance(statement, ast.If)
        and isinstance(statement.test, ast.Compare)
        and isinstance(statement.test.left, ast.Name)
        and statement.test.left.id == "fidelity"
    )
    constructor.body.remove(compute_assignment)
    constructor.body.remove(compute_branch)
    projection = next(
        node
        for node in ast.walk(constructor)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "MinimalPrefillProjection"
    )
    assert isinstance(projection.args[1], ast.Name) and projection.args[1].id == "compute"
    projection.args[1] = ast.Attribute(value=ast.Name(id="source", ctx=ast.Load()), attr="compute", ctx=ast.Load())
    assert dumped(constructor) == dumped(old[changed[1]])
    # Execute only the exact scalar auto branch and constructor with CPU stand-ins.
    enums = SimpleNamespace(HiFi4="HiFi4", HiFi2="HiFi2", LoFi="LoFi")
    clone_calls = []

    def clone(architecture, **fields):
        clone_calls.append(dict(architecture=architecture, fields=fields))
        return SimpleNamespace(**fields)

    ttnn = SimpleNamespace(MathFidelity=enums, init_device_compute_kernel_config=clone)
    auto_checks = []
    for sliding in (True, False):
        for setting in ("auto", None, "HiFi4", "HiFi2", "LoFi"):
            scope = dict(ttnn=ttnn, sliding=sliding, prefill_qkv_fidelity=setting)
            exec(
                compile(
                    ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(auto)], type_ignores=[])),
                    "source_auto_branch",
                    "exec",
                ),
                scope,
            )
            expected = ("HiFi4" if sliding else "HiFi2") if setting == "auto" else setting
            assert scope["prefill_qkv_fidelity"] == expected
            auto_checks.append(dict(sliding=sliding, setting=setting, resolved=expected, passed=True))
    classes = {
        name: type(name, (), {})
        for name in ("BroadcastQKV", "TiedQKV", "DirectQKV", "LanePartitionQKV", "CompensatedQKV")
    }
    scope = dict(ttnn=ttnn, **classes)
    scope["MinimalPrefillProjection"] = lambda weight, compute, mesh, block_w: SimpleNamespace(
        weight=weight, compute=compute, mesh=mesh, block_w=block_w
    )
    init = copy.deepcopy(new[changed[1]])
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[init], type_ignores=[])), "source_constructor", "exec"),
        scope,
    )
    checks = []
    for source_type in ("BroadcastQKV", "TiedQKV"):
        for fidelity in (None, "HiFi4", "HiFi2", "LoFi"):
            source = classes[source_type]()
            flags = dict(
                math_fidelity="HiFi4",
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
                dst_full_sync_en=False,
                throttle_level="NO_THROTTLE",
            )
            source.compute = SimpleNamespace(**flags)
            source.weight = object()
            target = SimpleNamespace()
            mesh = SimpleNamespace(arch=lambda: "cpu_stand_in")
            scope["__init__"](target, source, mesh, block_w=16, fidelity=fidelity)
            expected = dict(flags, math_fidelity=fidelity or "HiFi4")
            assert vars(target.projection.compute) == expected
            assert (
                target.source is source and target.decode_source is source and target.projection.weight is source.weight
            )
            assert vars(source.compute) == flags
            assert (target.projection.compute is source.compute) == (fidelity is None)
            checks.append(
                dict(
                    source_type=source_type,
                    fidelity=fidelity,
                    effective_compute=expected,
                    original_compute_unchanged=True,
                    weight_alias_preserved=True,
                    decode_source_preserved=True,
                    passed=True,
                )
            )
    full = json.loads((ROOT / "minimal_hifi2_acceptance_v6/long_layer5.json").read_text())
    sliding = json.loads((ROOT / "minimal_hifi2_acceptance_v6/long_layer0.json").read_text())
    assert (
        full["prefill_sampled_rows_passed"]
        and len(full["sampled_row_diagnostics"]) == 291
        and all(row["passed"] and row["pcc"] >= 0.995 for row in full["sampled_row_diagnostics"])
    )
    rejected = [row for row in sliding["sampled_row_diagnostics"] if not row["passed"]]
    assert rejected == [dict(position=32, pcc=0.9949930133358956, passed=False)]
    report = dict(
        current_runtime_sha256=NEW_HASH,
        ancestor_runtime_sha256=OLD_HASH,
        ancestor_source=str(OLD.relative_to(ROOT)),
        generator_sha256=digest(Path(__file__).read_bytes()),
        unified_diff="".join(
            difflib.unified_diff(
                old_source.decode().splitlines(True), new_source.decode().splitlines(True), fromfile="v6", tofile="v7"
            )
        ),
        method_count=40,
        changed_methods=changed,
        exact_delta_reversal_proof=[dict(method=name, remaining_ast_exactly_matches_v6=True) for name in changed],
        unchanged_methods={name: digest(dumped(node).encode()) for name, node in new.items() if name not in changed},
        auto_resolution_checks=auto_checks,
        constructor_clone_checks=checks,
        source_claim="Only full minimal-prefill QKV default fidelity changes numerically; sliding resolves to the same complete compute-field values. All forward methods, decode source, projections/weights/layouts, cache/tail/geometry, router and expert logic remain AST-identical. Source equality does not prove full-prefill numerical equivalence.",
        selected_policy=dict(
            sliding_attention="HiFi4",
            full_attention="HiFi2",
            explicit_none="original source compute object; no override",
            minimal_disabled="factory does not construct or change the minimal QKV path",
        ),
        candidate_gate=dict(
            runtime_sha256=OLD_HASH,
            paired_summary="minimal_pairs_v6_summary.json",
            paired_summary_sha256=digest((ROOT / "minimal_pairs_v6_summary.json").read_bytes()),
            full_long="minimal_hifi2_acceptance_v6/long_layer5.json",
            full_long_sha256=digest((ROOT / "minimal_hifi2_acceptance_v6/long_layer5.json").read_bytes()),
            full_minimum_sampled_row_pcc=min(row["pcc"] for row in full["sampled_row_diagnostics"]),
            sliding_long="minimal_hifi2_acceptance_v6/long_layer0.json",
            sliding_rejected_rows=rejected,
        ),
        tensor_payload_delta_bytes=0,
        new_setup_tensors=False,
        current_validation="Integrated v7 campaign is separate; see validated_v7_validation_summary.json when complete. Probe controls retain v6 plus setup-override attribution.",
        inheritance=dict(
            v6_summary="validated_v6_validation_summary.json",
            v6_summary_sha256=digest((ROOT / "validated_v6_validation_summary.json").read_bytes()),
            v6_source_delta="source_delta_v6.json",
            v6_source_delta_sha256=digest((ROOT / "source_delta_v6.json").read_bytes()),
            unchanged_sliding_contracts=[
                "B32",
                "prefix continuation",
                "BF16 cache compatibility",
                "maximum/near-maximum sampled context",
                "tail/capacity lifecycle",
            ],
        ),
    )
    (ROOT / "source_delta_v7.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        "PASS: 38 unchanged methods;2 exact reversed deltas;10 auto/override and8 constructor checks;zero tensor payload delta"
    )


if __name__ == "__main__":
    main()
