# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only exact source and setup/producer proof for full-prefill M2/L1."""

import ast
import copy
import difflib
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
OLD = ROOT / "runtime_v7_before_placement.py.txt"
NEW = ROOT.parent.parent / "tt/optimized_decoder.py"
OLD_HASH = "daa82a4a5197a007ccd5d29d912f082e695fd37a596578f25fba96a6694e625b"
NEW_HASH = "5ff391a2efb6096e7d9eaa499c6d19a8548cf9a60d108425e773d4ded72ee898"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def dumped(node):
    return ast.dump(node, include_attributes=False)


def methods(tree):
    return {
        f"{cls.name}.{method.name}": method
        for cls in tree.body
        if isinstance(cls, ast.ClassDef)
        for method in cls.body
        if isinstance(method, ast.FunctionDef)
    }


def remove_arg(node, name):
    pairs = list(zip(node.args.kwonlyargs, node.args.kw_defaults))
    assert sum(arg.arg == name for arg, _ in pairs) == 1
    node.args.kwonlyargs = [arg for arg, _ in pairs if arg.arg != name]
    node.args.kw_defaults = [default for arg, default in pairs if arg.arg != name]


def run_nodes(nodes, scope):
    module = ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes), type_ignores=[]))
    exec(compile(module, "source_cpu_proof", "exec"), scope)


def main():
    old_source, new_source = OLD.read_bytes(), NEW.read_bytes()
    assert digest(old_source) == OLD_HASH and digest(new_source) == NEW_HASH
    old, new = methods(ast.parse(old_source)), methods(ast.parse(new_source))
    assert old.keys() == new.keys() and len(old) == 40
    changed = [key for key in old if dumped(old[key]) != dumped(new[key])]
    assert changed == [
        "OptimizedDecoder.from_state_dict",
        "OptimizedDecoder.normalize",
        "MinimalPrefillProjection.__init__",
        "MinimalPrefillQKV.__init__",
    ]
    factory = copy.deepcopy(new[changed[0]])
    for name in ("prefill_qkv_minimal_block_h", "prefill_qkv_input_l1"):
        remove_arg(factory, name)
    auto = [
        node
        for node in factory.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id in ("prefill_qkv_minimal_block_h", "prefill_qkv_input_l1")
    ]
    assert len(auto) == 2
    factory.body = [node for node in factory.body if node not in auto]
    assign = next(
        node
        for node in factory.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Attribute) and target.attr == "prefill_qkv_input_l1" for target in node.targets)
    )
    factory.body.remove(assign)
    for node in ast.walk(factory):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "MinimalPrefillQKV":
            node.keywords = [kw for kw in node.keywords if kw.arg != "block_h"]
    projection_branch = next(
        node
        for node in factory.body
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "prefill_qkv_minimal"
    )
    metadata = next(
        node
        for node in projection_branch.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Subscript)
            and isinstance(target.slice, ast.Constant)
            and target.slice.value == "input_memory"
            for target in node.targets
        )
    )
    projection_branch.body.remove(metadata)
    assert dumped(factory) == dumped(old[changed[0]])
    norm = copy.deepcopy(new[changed[1]])
    branch = norm.body[-2]
    assert isinstance(branch, ast.If) and "prefill_qkv_input_l1" in ast.unparse(branch.test)
    norm.body.remove(branch)
    assert dumped(norm) == dumped(old[changed[1]])
    init = copy.deepcopy(new[changed[2]])
    remove_arg(init, "block_h")
    validation = init.body[0]
    assert isinstance(validation, ast.If) and ast.unparse(validation.test) == "block_h not in (1, 2, 3, 4)"
    init.body.remove(validation)
    for node in ast.walk(init):
        if isinstance(node, ast.keyword) and node.arg == "M_block_size":
            assert ast.unparse(node.value) == "min(rows, block_h)"
            node.value = ast.Name(id="rows", ctx=ast.Load())
        if isinstance(node, ast.keyword) and node.arg == "m_block":
            assert isinstance(node.value, ast.JoinedStr)
            node.value = ast.Constant(value="min(4, padded_M_tiles)")
    assert dumped(init) == dumped(old[changed[2]])
    qkv = copy.deepcopy(new[changed[3]])
    remove_arg(qkv, "block_h")
    for node in ast.walk(qkv):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "MinimalPrefillProjection"
        ):
            node.keywords = [kw for kw in node.keywords if kw.arg != "block_h"]
    assert dumped(qkv) == dumped(old[changed[3]])

    auto_checks = []
    for sliding in (False, True):
        for minimal in (False, True):
            for block_h in ("auto", 1, 2, 4):
                for input_l1 in ("auto", False, True):
                    scope = dict(
                        sliding=sliding,
                        prefill_qkv_minimal=minimal,
                        prefill_qkv_minimal_block_h=block_h,
                        prefill_qkv_input_l1=input_l1,
                    )
                    run_nodes(auto, scope)
                    expected_h = (4 if sliding else 2) if block_h == "auto" else block_h
                    expected_l1 = bool(minimal and not sliding) if input_l1 == "auto" else input_l1
                    assert scope["prefill_qkv_minimal_block_h"] == expected_h
                    assert scope["prefill_qkv_input_l1"] == expected_l1
                    auto_checks.append(
                        dict(
                            sliding=sliding,
                            minimal=minimal,
                            block_h=block_h,
                            input_l1=input_l1,
                            resolved_h=expected_h,
                            resolved_l1=expected_l1,
                            passed=True,
                        )
                    )

    ttnn = SimpleNamespace(MinimalMatmulConfig=lambda **kw: SimpleNamespace(**kw), CoreCoord=lambda x, y: (x, y))
    scope = dict(ttnn=ttnn)
    run_nodes([new[changed[2]]], scope)
    setup_checks = []
    for block_h in (1, 2, 3, 4):
        for block_w in (8, 16):
            target = SimpleNamespace()
            weight = SimpleNamespace(dtype="BFP8", shape=[1, 1, 2816, 9216])
            compute = SimpleNamespace(
                math_fidelity="HiFi2", fp32_dest_acc_en=True, math_approx_mode=False, packer_l1_acc=False
            )
            mesh = SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=11, y=10))
            scope["__init__"](target, weight, compute, mesh, block_h=block_h, block_w=block_w)
            assert target.weight is weight and target.compute is compute
            assert [p.M_block_size for p in target.programs] == [min(rows, block_h) for rows in (1, 2, 3, 4)]
            assert all(
                (p.K_block_size, p.N_block_size, p.subblock_h, p.subblock_w, p.compute_with_storage_grid_size)
                == (block_w, 8, 1, 4, (11, 8))
                for p in target.programs
            )
            setup_checks.append(
                dict(
                    block_h=block_h,
                    block_w=block_w,
                    same_compute_and_weight=True,
                    effective_M=[p.M_block_size for p in target.programs],
                    passed=True,
                )
            )

    calls = []

    def mul(value, weight, **kwargs):
        calls.append(("mul", value, weight, kwargs))
        return "weighted_output"

    class Base:
        def normalize(self, value, epsilon, weight=None):
            calls.append(("normalize_unweighted", value, epsilon))
            return "norm_output" if weight is None else mul("norm_output", weight)

    norm_scope = dict(Base=Base, ttnn=SimpleNamespace(mul=mul, L1_MEMORY_CONFIG="L1"))
    for label, node in (("Old", old[changed[1]]), ("New", new[changed[1]])):
        cls = ast.ClassDef(
            name=label,
            bases=[ast.Name(id="Base", ctx=ast.Load())],
            keywords=[],
            body=[copy.deepcopy(node)],
            decorator_list=[],
        )
        run_nodes([cls], norm_scope)
    norm_checks = []
    for enabled in (False, True):
        for rows in (1, 32, 1024):
            for site in ("input", "post", "common"):
                weight = object() if site != "common" else None
                value = SimpleNamespace(shape=[1, 1, rows, 2816], is_sharded=lambda: False)
                records = []
                for label in ("Old", "New"):
                    obj = norm_scope[label]()
                    obj.prefill_qkv_input_l1 = enabled
                    obj.use_sharded_norms = False
                    obj.input_norm_weight = weight if site == "input" else object()
                    obj.post_attention_norm_weight = weight if site == "post" else object()
                    obj.config = SimpleNamespace(hidden_size=2816)
                    calls.clear()
                    obj.normalize(value, 1e-6, weight)
                    records.append(list(calls))
                expected = copy.deepcopy(records[0])
                changed_memory = enabled and rows > 1 and site == "input"
                # Compare operation order/object identity; only final weighted output placement differs.
                assert len(records[0]) == len(records[1])
                for index, (left, right) in enumerate(zip(records[0], records[1])):
                    if changed_memory and index == len(records[0]) - 1:
                        assert left[:-1] == right[:-1] and left[-1] == {} and right[-1] == {"memory_config": "L1"}
                    else:
                        assert left == right
                norm_checks.append(
                    dict(
                        enabled=enabled,
                        rows=rows,
                        site=site,
                        final_mul_memory_only=changed_memory,
                        arithmetic_call_order_preserved=True,
                        passed=True,
                    )
                )
    report = dict(
        current_runtime_sha256=NEW_HASH,
        ancestor_runtime_sha256=OLD_HASH,
        ancestor_source=OLD.name,
        ancestor_manifest="validated_v7_validation_summary.json",
        ancestor_manifest_sha256=digest((ROOT / "validated_v7_validation_summary.json").read_bytes()),
        generator_sha256=digest(Path(__file__).read_bytes()),
        method_count=40,
        changed_methods=changed,
        exact_delta_reversal_proof=[dict(method=name, remaining_ast_exactly_matches_v7=True) for name in changed],
        unchanged_methods={name: digest(dumped(node).encode()) for name, node in new.items() if name not in changed},
        unified_diff="".join(
            difflib.unified_diff(
                old_source.decode().splitlines(True), new_source.decode().splitlines(True), fromfile="v7", tofile="v8"
            )
        ),
        auto_resolution_checks=auto_checks,
        projection_setup_checks=setup_checks,
        normalization_checks=norm_checks,
        source_claim="Full minimal QKV uses M2 and input normalization writes its final weighted result directly to L1. Sliding keeps M4 and skips the new normalization branch. All decode branches, QKV compute values/weight aliases, full public lifecycle and cache-capacity methods remain unchanged. M2 may change numerical evaluation and requires fresh full-path gates.",
        resident_tensor_payload_delta_bytes=0,
        new_setup_tensors=False,
        temporary_full_input=dict(
            shape=[1, 1, 1024, 2816],
            dtype="FP32",
            payload_bytes=11534336,
            movement="existing final weighted-normalization tensor goes to L1 instead of DRAM; no extra copy",
        ),
        downstream_temporary_placement=dict(
            output_spec_source="ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_device_operation.cpp:297",
            output_spec_rule="output_mem_config.value_or(in0_input_tensor.memory_config())",
            minimal_output=dict(shape=[1, 1, 1024, 9216], dtype="FP32", memory="L1", payload_bytes=37748736),
            tied_kv_slice=dict(shape=[1, 1, 1024, 1024], dtype="FP32", memory="L1", payload_bytes=4194304),
            tied_concat=dict(shape=[1, 1, 1024, 10240], dtype="FP32", memory="DRAM", payload_bytes=41943040),
            scope="Individual temporary payloads, not summed live allocations or allocator peaks. None output request inherits input memory, so full QKV output and slice also move into L1.",
        ),
        workspace=dict(
            scope="Per compute core QKV CB payload, excludes runtime allocator/other kernels and metadata",
            prior_M4_bytes=1196032,
            selected_M2_bytes=737280,
            delta_bytes=-458752,
        ),
        source_proof_only=True,
    )
    (ROOT / "source_delta_v8.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        "PASS:36 unchanged methods;4 exact reversed deltas;48 auto/override,8 setup,18 norm checks;zero resident payload delta"
    )


if __name__ == "__main__":
    main()
