# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only V5/V6 fused source contract audit. Does not establish device acceptance."""
import argparse
import ast
import copy
import hashlib
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
V5 = ROOT / "doc/fused_decoder/selected_fused_decoder_v5.py.snapshot"
V5_SHA = "8ac733159befba919abf705603f3aada9d105195c7b495d8cd25aa65e531769f"
REVIEWED = (
    ROOT / "doc/fused_decoder/endurance/v6_corrected_fma_full_v5_streaming_scan_norm_tail_fused_matmul.py.snapshot"
)
REVIEWED_SHA = "8cda999c18f9245c872e40d7f67e027838b56396056ede43e162b7a8386f3faa"
EXTRA_REVIEWED = {
    "v6_corrected_fma_full_v5_triangular_compact_silu_integration.py.snapshot": "85644b73bc1effb516c1fc7a5092cc443ef7d8a30a67a6635767a06f3eb7f858",
    "v6_corrected_fma_full_v5_triangular_compact_silu_formatted_integration.py.snapshot": "ab28a91bb3cb5e8b88e5371da0defe3b0d731e86a131c8523483e4b6b65b0334",
    "v6_corrected_fma_full_v5_compact_silu_fused_matmul_integration.py.snapshot": "951391e44ed08d1646f857091d74c011b3c93e0c8735f7a0d941890291fd5579",
    "v6_corrected_fma_full_v5_direct_coeff_compact_silu_fused_matmul_integration.py.snapshot": "5ff93fd27348d9e49b6c09da6fd77a362243c7c575b4b92431232638247de242",
    "v6_corrected_fma_full_v5_direct_coeff_cast_placement_fused_matmul_integration.py.snapshot": "c823e6f6f8480a9550276856a7aa660b9308ff598c6565a38c2bc54460440ece",
}
FUNCTIONAL = ROOT / "tt/functional_decoder.py"
FUNCTIONAL_SHA = "ff3307ea3a3a4dcc93400fdd2a79a47a32e216d9f1bc47626e875dc7b561f99b"
PUBLIC = (
    "from_state_dict",
    "new_recurrent_state",
    "new_kv_state",
    "prepare_positions",
    "prepare_prefill",
    "prefill_forward",
    "prefill_batch",
    "decode_forward",
    "decode_batch",
)
UNCHANGED = ("new_kv_state", "prepare_positions", "prefill_batch", "decode_forward", "decode_batch")
FULL_GUARDS = ("from_state_dict", "linear", "norm", "_moe", "_forward", "prepare_prefill", "prefill_forward")
SETUP = {
    "FusedDecoder.from_state_dict": "Explicit model construction and CPU weight packing/upload before execution",
    "FusedDecoder.new_recurrent_state": "Explicit state allocation before execution",
    "FusedDecoder.new_kv_state": "Explicit paged pool/table construction before execution",
    "FusedDecoder.prepare_positions": "Explicit absolute-position/RoPE preparation before execution",
    "FusedDecoder.prepare_prefill": "Explicit host plan construction; permitted to invoke prepare_positions",
    "InlineSelector.__init__": "Reviewed fixed selector setup constants only",
    "Chunk64StateGraph.__init__": "Reviewed masks/constants and device-adapter construction only",
    "chunk64_mathematical_exp_table": "Reviewed mathematical exponential table construction only",
    "Chunk64MKLNegativeExp.__init__": "Reviewed fixed exponential tables/scalars only",
    "GatedNormTail.__init__": "Reviewed norm-weight upload and program setup only",
    "WeightedNormTail.__init__": "Reviewed norm-weight expansion/upload and static program configuration only",
    "LocalSiluMultiply.__init__": "Reviewed exhaustive mathematical compact-LUT verification/upload and static placement only",
    "local_silu_math_table": "Reviewed exhaustive 65536 BF16-code mathematical SiLU setup table, independent of model data",
    "compact_silu_math_table": "Reviewed setup-only compact packing with exhaustive raw FP32 reconstruction assertion before upload",
}
HOST_ATTRS = {"from_torch", "to_torch", "cpu", "numpy", "to_numpy", "tolist", "item", "from_dlpack", "to_dlpack"}
DYNAMIC = {"exec", "eval", "compile", "__import__", "globals", "locals", "vars"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize_legacy_descriptions(node):
    """Only trim trailing whitespace in two reviewed legacy descriptions.

    Assigned kernel strings and all other executable constants remain exact.
    V5 stores these as method docstrings; V6 copies them into full-kind guards.
    """
    result = copy.deepcopy(node)
    for method in ast.walk(result):
        if not isinstance(method, ast.FunctionDef) or method.name not in {"prepare_prefill", "prefill_forward"}:
            continue
        blocks = [method.body]
        for statement in method.body:
            if isinstance(statement, ast.If) and ast.unparse(statement.test) == "self.kind == 'full_attention'":
                blocks.append(statement.body)
                break
        for body in blocks:
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                body[0].value.value = body[0].value.value.rstrip()
    return result


def digest(node):
    return hashlib.sha256(ast.dump(normalize_legacy_descriptions(node)).encode()).hexdigest()


def named(scope, name):
    return next(n for n in scope.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)


def functions(tree):
    result = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            result[node.name] = node
        elif isinstance(node, ast.ClassDef):
            for method in node.body:
                if isinstance(method, ast.FunctionDef):
                    result[node.name + "." + method.name] = method
    return result


def reviewed_sources():
    assert sha(V5) == V5_SHA and sha(REVIEWED) == REVIEWED_SHA and sha(FUNCTIONAL) == FUNCTIONAL_SHA
    return ast.parse(V5.read_text()), ast.parse(REVIEWED.read_text()), ast.parse(FUNCTIONAL.read_text())


def syntax_audit(tree, v5, reviewed, extra=()):
    fs = functions(tree)
    references = [functions(ref) for ref in (v5, reviewed, *extra)]
    reviewed_setup = {name: {digest(ref[name]) for ref in references if name in ref} for name in SETUP}
    found_setup = {}
    runtime = {}
    violations = []
    classes = {n.name for n in tree.body if isinstance(n, ast.ClassDef)}
    setup_classes = {name.rsplit(".", 1)[0] for name in SETUP if name.endswith(".__init__")}
    setup_functions = {name for name in SETUP if "." not in name}
    setup_methods = {name.split(".")[-1] for name in SETUP if name.startswith("FusedDecoder.")}
    # Module/class initialization cannot hide unreviewed executable side effects.
    for scope in [tree] + [n for n in tree.body if isinstance(n, ast.ClassDef)]:
        if isinstance(scope, ast.ClassDef):
            assert not scope.decorator_list and not scope.keywords
            assert all(
                isinstance(b, ast.Name) and b.id in {"LightweightModule", "Chunk64StateGraph"} for b in scope.bases
            )
        for node in scope.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.FunctionDef):
                    assert all(ast.unparse(d) == "classmethod" for d in node.decorator_list)
                continue
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                continue
            assert isinstance(node, ast.Assign) and all(
                isinstance(t, ast.Name) or (isinstance(t, ast.Tuple) and all(isinstance(x, ast.Name) for x in t.elts))
                for t in node.targets
            ), ast.unparse(node)
            if ast.unparse(node) not in {"DRAM = ttnn.DRAM_MEMORY_CONFIG", "state_call = Chunk64StateGraph.__call__"}:
                ast.literal_eval(node.value)
    # Imports are restricted globally as well as prohibited in measured helpers.
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            assert ast.unparse(node) in (
                "import ttnn",
                "import math",
                "from models.common.lightweightmodule import LightweightModule",
            ), ast.unparse(node)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            text = ast.unparse(node)
            assert "tests" not in text and "functional_decoder" not in text, text
        if isinstance(node, ast.Name):
            assert node.id not in {"FunctionalDecoder", "functional_decoder"} | DYNAMIC, node.id
        if isinstance(node, ast.Attribute):
            assert node.attr not in DYNAMIC, ast.unparse(node)
        if isinstance(node, ast.Call):
            assert ast.unparse(node.func) not in DYNAMIC, ast.unparse(node)
    for name, method in fs.items():
        if name in SETUP:
            assert digest(method) in reviewed_setup[name], (
                "Setup method changed; explicit source review required",
                name,
            )
            found_setup[name] = dict(reason=SETUP[name], ast_sha256=digest(method))
            continue
        runtime[name] = digest(method)
        for node in ast.walk(method):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                violations.append((name, ast.unparse(node)))
            if isinstance(node, ast.Name) and node.id in {"torch", "numpy", "np"} | setup_functions | setup_classes:
                violations.append((name, node.id))
            if isinstance(node, ast.Attribute) and node.attr in HOST_ATTRS | setup_methods | {
                "__init__",
                "getattr",
                "setattr",
            }:
                violations.append((name, ast.unparse(node)))
            if isinstance(node, ast.Call):
                call = ast.unparse(node.func)
                if isinstance(node.func, ast.Name) and node.func.id in setup_functions | setup_classes:
                    violations.append((name, "Runtime reaches setup-only constructor/function: " + call))
                if isinstance(node.func, ast.Attribute) and node.func.attr in setup_methods | {
                    "__init__",
                    "getattr",
                    "setattr",
                }:
                    violations.append((name, "Runtime reaches setup/dynamic method: " + call))
                if call in ("getattr", "setattr", "open"):
                    violations.append((name, "Runtime dynamic/host I/O: " + call))
    assert not violations, violations
    cls = named(tree, "FusedDecoder")
    assert [ast.unparse(n) for n in cls.bases] == ["LightweightModule"]
    # Audit every non-setup function/method, a strict superset of reachable helpers.
    return dict(
        runtime_function_ast_sha256=runtime,
        reviewed_setup_methods=found_setup,
        forbidden_runtime_constructs=violations,
        classes=sorted(classes),
        coverage="All top-level functions and every method outside individually hash-reviewed setup scopes; nested functions inspected within their enclosing method; no class-wide exemption",
    )


def contract_audit(tree, v5, functional):
    cls, vc, fc = named(tree, "FusedDecoder"), named(v5, "FusedDecoder"), named(functional, "FunctionalDecoder")
    signatures = {}
    unchanged = {}
    guards = {}
    for name in PUBLIC:
        method, base = named(cls, name), named(fc, name)
        if name == "prepare_prefill":
            # Existing reviewed V5 optimization; only the hint default changed.
            assert ast.dump(method.args) == ast.dump(named(vc, name).args)
            normalized = copy.deepcopy(method.args)
            index = [a.arg for a in normalized.kwonlyargs].index("chunk_size")
            assert ast.literal_eval(normalized.kw_defaults[index]) == 4096
            assert ast.literal_eval(base.args.kw_defaults[index]) == 256
            normalized.kw_defaults[index] = copy.deepcopy(base.args.kw_defaults[index])
            assert ast.dump(normalized) == ast.dump(base.args), ("Public signature", name)
        else:
            assert ast.dump(method.args) == ast.dump(base.args), ("Public signature", name)
        signatures[name] = digest(method.args)
    for name in UNCHANGED:
        assert digest(named(cls, name)) == digest(named(fc, name)), ("Changed functional contract body", name)
        unchanged[name] = digest(named(cls, name))
    direct_v5 = digest(cls) == digest(vc)
    if not direct_v5:
        for name in FULL_GUARDS:
            method, base = named(cls, name), named(vc, name)
            i = int(
                isinstance(method.body[0], ast.Expr)
                and isinstance(method.body[0].value, ast.Constant)
                and isinstance(method.body[0].value.value, str)
            )
            guard = method.body[i]
            condition = (
                "hf_config.layer_types[layer_idx] == 'full_attention'"
                if name == "from_state_dict"
                else "self.kind == 'full_attention'"
            )
            assert isinstance(guard, ast.If) and ast.unparse(guard.test) == condition and not guard.orelse, (
                "Missing exact full-kind guard",
                name,
            )
            assert ast.dump(ast.Module(normalize_legacy_descriptions(method).body[i].body, [])) == ast.dump(
                ast.Module(normalize_legacy_descriptions(base).body, [])
            ), (
                "Changed V5 full-kind branch",
                name,
            )
            assert isinstance(guard.body[-1], ast.Return), ("Full guard must return before linear repair", name)
            guards[name] = digest(guard)
        # All remaining shared methods except linear recurrence must retain V5 bodies.
        for method in vc.body:
            if (
                isinstance(method, ast.FunctionDef)
                and method.name not in FULL_GUARDS
                and method.name != "_linear_attention"
            ):
                assert digest(named(cls, method.name)) == digest(method), ("Changed V5 shared method", method.name)
    assert digest(named(cls, "_full_attention")) == digest(named(vc, "_full_attention"))
    return dict(
        public_signature_ast_sha256=signatures,
        reviewed_signature_default_delta={"prepare_prefill.chunk_size": {"functional": 256, "V5_and_V6": 4096}},
        unchanged_functional_contract_ast_sha256=unchanged,
        full_kind_guards_ast_sha256=guards,
        full_kind_mode="Exact whole V5 decoder class" if direct_v5 else "Exact original V5 early-return guarded bodies",
        v6=not direct_v5,
    )


def pure_planner(method):
    # Execute only this reviewed host-plan method, never import the snapshot.
    allowed_calls = {"ValueError", "range", "min", "max", "self.prepare_positions", "plan.append"}
    allowed_attrs = {
        "self.kind",
        "self.config",
        "self.config.max_position_embeddings",
        "self.prepare_positions",
        "plan.append",
    }
    for node in ast.walk(method):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Lambda)):
            raise AssertionError("Unsafe planner construct")
        if isinstance(node, ast.Call):
            assert ast.unparse(node.func) in allowed_calls, ast.unparse(node)
        if isinstance(node, ast.Attribute):
            assert ast.unparse(node) in allowed_attrs, ast.unparse(node)
    scope = {}
    exec(compile(ast.Module([copy.deepcopy(method)], []), "<isolated-host-plan>", "exec"), scope)
    return scope[method.name]


def planner_audit(tree, v5, is_v6, capacity):
    actual = pure_planner(named(named(tree, "FusedDecoder"), "prepare_prefill"))
    reference = pure_planner(named(named(v5, "FusedDecoder"), "prepare_prefill"))
    cases = {}
    invalid = {}
    nonaligned = {}
    starts = list(range(66)) + [
        127,
        128,
        255,
        256,
        257,
        4095,
        capacity - 257,
        capacity - 65,
        capacity - 33,
        capacity - 1,
    ]
    lengths = list(range(1, 130)) + [255, 256, 257, 511, 512, 513, 4095, 4096, capacity - 1, capacity]
    for kind in ("linear_attention", "full_attention"):
        obj = SimpleNamespace(
            kind=kind,
            config=SimpleNamespace(max_position_embeddings=capacity),
            prepare_positions=lambda **kw: (kw["start_pos"], kw["seq_len"]),
        )
        total = unaligned = 0
        for start, length, chunk in itertools.product(starts, lengths, (32, 64, 128, 4096)):
            if start + length > capacity:
                continue
            plan = actual(obj, seq_len=length, start_pos=start, chunk_size=chunk)
            assert plan and plan[0][0] == 0
            offset = 0
            for item in plan:
                o, n, absolute, positions = item
                assert o == offset and n > 0 and absolute == start + o and absolute + n <= capacity
                if kind == "full_attention":
                    assert positions == (absolute, n)
                    assert n == 1 or (absolute % 256 == 0 and n % 32 == 0 and n <= min(chunk, 256))
                else:
                    assert positions is None
                    if is_v6:
                        assert n == min(64, length - o)
                offset += n
            assert offset == length
            if kind == "full_attention" or not is_v6:
                assert plan == reference(obj, seq_len=length, start_pos=start, chunk_size=chunk)
            total += 1
            unaligned += int(start % 32 != 0 or length % 32 != 0)
        bad = [
            dict(seq_len=0, start_pos=0, chunk_size=64),
            dict(seq_len=-1, start_pos=0, chunk_size=64),
            dict(seq_len=1, start_pos=-1, chunk_size=64),
            dict(seq_len=1, start_pos=capacity, chunk_size=64),
            dict(seq_len=2, start_pos=capacity - 1, chunk_size=64),
            dict(seq_len=1, start_pos=0, chunk_size=0),
            dict(seq_len=1, start_pos=0, chunk_size=-32),
        ]
        for args in bad:
            try:
                actual(obj, **args)
            except ValueError:
                pass
            else:
                raise AssertionError(("Accepted invalid plan", kind, args))
        # Small positive hints are valid only for the V6 fixed64 internal planner.
        for chunk in (1, 7, 31, 33, 63, 65):
            if kind == "linear_attention" and is_v6:
                plan = actual(obj, seq_len=65, start_pos=1, chunk_size=chunk)
                assert plan == [(0, 64, 1, None), (64, 1, 65, None)]
            else:
                try:
                    actual(obj, seq_len=65, start_pos=1, chunk_size=chunk)
                except ValueError:
                    pass
                else:
                    raise AssertionError("Unexpected non-multiple32 physical chunk accepted")
        cases[kind] = total
        nonaligned[kind] = unaligned
        invalid[kind] = len(bad)
    return dict(
        valid_plan_cases=cases,
        unaligned_start_or_length_cases=nonaligned,
        invalid_cases_rejected=invalid,
        capacity_bound=capacity,
        full_kind_plans_exact_v5=True,
        linear_grouping="64 relative to public call; positive chunk_size accepted as hint"
        if is_v6
        else "exact V5 planner",
        physical_device_allocation_exercised=False,
    )


def dispatch_audit(tree, v5, reviewed, capacity):
    """Exercise only hash-reviewed Python dispatch with tensor/TTNN stand-ins."""
    method = named(named(tree, "FusedDecoder"), "prefill_forward")
    assert digest(method) in {
        digest(named(named(r, "FusedDecoder"), "prefill_forward")) for r in (v5, reviewed)
    }, "Changed dispatcher requires explicit review before mock execution"
    planner = pure_planner(named(named(tree, "FusedDecoder"), "prepare_prefill"))

    class Tensor:
        def __init__(self, length, offset=0):
            self.shape = (1, length, 2048)
            self.offset = offset

        def __getitem__(self, key):
            part = key[1]
            return Tensor(part.stop - part.start, self.offset + part.start)

    casts, copies = [], []

    def cast(x, dtype):
        casts.append(dtype)
        return x

    def concatenate(outputs, dim):
        assert dim == 1
        return Tensor(sum(o.shape[1] for o in outputs))

    fake = SimpleNamespace(
        typecast=cast, copy=lambda a, b: copies.append((a, b)), concat=concatenate, bfloat16="bf16", float32="fp32"
    )
    scope = {"ttnn": fake}
    exec(compile(ast.Module([copy.deepcopy(method)], []), "<hash-reviewed-mock-dispatch>", "exec"), scope)
    forward = scope[method.name]
    total = internal_singletons = public_singletons = 0
    for kind, length in itertools.product(
        ("linear_attention", "full_attention"), (1, 2, 31, 32, 33, 63, 64, 65, 127, 128, 129, 257)
    ):
        for start in (0, 1, 31, 32, 255, 256, capacity - length):
            records = []

            def record(x, **kw):
                records.append((x, kw))
                return x

            obj = SimpleNamespace(
                kind=kind,
                config=SimpleNamespace(max_position_embeddings=capacity),
                prepare_positions=lambda **kw: (kw["start_pos"], kw["seq_len"]),
                _forward=record,
            )
            plan = planner(obj, seq_len=length, start_pos=start)
            state = {"recurrent": object(), "context_length": capacity}
            casts.clear()
            copies.clear()
            result = forward(obj, Tensor(length), state=state, plan=plan)
            assert result.shape == (1, length, 2048) and len(records) == len(plan)
            assert casts == (["bf16", "fp32"] if kind == "linear_attention" else [])
            assert len(copies) == int(kind == "linear_attention")
            for (tensor, kw), (offset, n, absolute, positions) in zip(records, plan):
                assert tensor.shape == (1, n, 2048) and tensor.offset == offset
                assert kw["state"] is state and kw["start_pos"] == absolute and kw["positions"] == positions
                assert kw["cache_boundary"] is False and kw["projection_k"] == (1024 if length > 1 else 512)
                v6 = digest(method) == digest(named(named(reviewed, "FusedDecoder"), "prefill_forward"))
                assert kw["decode"] == (n == 1 and (kind == "full_attention" or length == 1 or not v6))
                internal_singletons += int(kind == "linear_attention" and n == 1 and length > 1)
                public_singletons += int(kind == "linear_attention" and length == 1)
            for bad_plan, bad_state in [
                (plan[:-1] + [(plan[-1][0], plan[-1][1] + 1, plan[-1][2], plan[-1][3])], state)
            ]:
                try:
                    forward(obj, Tensor(length), state=bad_state, plan=bad_plan)
                except ValueError:
                    pass
                else:
                    raise AssertionError("Mismatched plan accepted")
            if kind == "full_attention":
                try:
                    forward(obj, Tensor(length), state={"context_length": start + length - 1}, plan=plan)
                except ValueError:
                    pass
                else:
                    raise AssertionError("Allocated KV context exceeded")
            total += 1
    return dict(
        cases=total,
        internal_linear_singleton_dispatches=internal_singletons,
        public_linear_singleton_dispatches=public_singletons,
        tensor_arithmetic_executed=False,
        tested="Absolute positions, input slicing/concatenation, public singleton recurrence, internal singleton chunk graph, initial cache boundary cast, projection K, invalid plan and allocated KV bounds",
    )


def negative_audit(v5, reviewed, functional, extra=()):
    """Mutation controls: each targeted defect must be rejected independently."""
    outcomes = {}

    def reject(label, change, checker):
        tree = copy.deepcopy(reviewed)
        change(tree)
        try:
            checker(tree)
        except (AssertionError, ValueError):
            outcomes[label] = "rejected"
        else:
            raise AssertionError(("Negative audit accepted mutation", label))

    def prepend(tree, owner, name, statement):
        named(named(tree, owner), name).body.insert(0, ast.parse(statement).body[0])

    syntax = lambda t: syntax_audit(t, v5, reviewed, extra)
    contract = lambda t: contract_audit(t, v5, functional)
    reject("helper_host_readback", lambda t: prepend(t, "FusedCorrectedMatmul", "__call__", "ttnn.to_torch(a)"), syntax)
    reject(
        "helper_torch_alias", lambda t: prepend(t, "FusedCorrectedMatmul", "__call__", "torch_alias = torch"), syntax
    )
    reject(
        "runtime_setup_constructor",
        lambda t: prepend(t, "FusedCorrectedMatmul", "__call__", "InlineSelector(None, None)"),
        syntax,
    )
    reject("dynamic_exec", lambda t: prepend(t, "FusedCorrectedMatmul", "__call__", "exec('pass')"), syntax)
    reject(
        "test_import",
        lambda t: prepend(
            t, "FusedCorrectedMatmul", "__call__", "from models.autoports.qwen_qwen3_6_35b_a3b.tests import helpers"
        ),
        syntax,
    )
    reject("changed_setup_whitelist", lambda t: prepend(t, "GatedNormTail", "__init__", "extra = 1"), syntax)
    reject("module_host_call", lambda t: t.body.append(ast.parse("hidden = ttnn.to_torch(None)").body[0]), syntax)
    reject("full_kind_guard", lambda t: prepend(t, "FusedDecoder", "_forward", "changed = 1"), contract)
    reject("functional_kv_body", lambda t: prepend(t, "FusedDecoder", "new_kv_state", "changed = 1"), contract)
    reject(
        "public_signature",
        lambda t: named(named(t, "FusedDecoder"), "decode_forward").args.kwonlyargs.append(ast.arg(arg="extra")),
        contract,
    )
    reject(
        "planner_alignment_restriction",
        lambda t: prepend(t, "FusedDecoder", "prepare_prefill", "if seq_len % 64: raise ValueError('aligned only')"),
        syntax,
    )
    # The new setup exemptions must never mask host work in the new runtime helpers.
    compact = next((t for t in extra if "LocalSiluMultiply.__call__" in functions(t)), None)
    direct = next((t for t in extra if "DirectCoefficientStreamingScan.__call__" in functions(t)), None)
    for label, ref, owner, statement in (
        ("compact_runtime_readback", compact, "LocalSiluMultiply", "t.to_torch(weighted)"),
        ("compact_runtime_aliased_setup", compact, "LocalSiluMultiply", "builder = compact_silu_math_table"),
        ("compact_runtime_aliased_exec", compact, "LocalSiluMultiply", "runner = exec"),
        ("compact_runtime_table_rebuild", compact, "LocalSiluMultiply", "compact_silu_math_table(weighted)"),
        ("weighted_runtime_upload", compact, "WeightedNormTail", "t.from_torch(core)"),
        ("direct_coefficient_runtime_cpu", direct, "DirectCoefficientStreamingScan", "activation.cpu()"),
    ):
        if ref is not None:
            tree = copy.deepcopy(ref)
            prepend(tree, owner, "__call__", statement)
            try:
                syntax(tree)
            except AssertionError:
                outcomes[label] = "rejected"
            else:
                raise AssertionError(("Negative audit accepted mutation", label))
    if compact is not None:
        tree = copy.deepcopy(compact)
        named(tree, "compact_silu_math_table").body = [
            n for n in named(tree, "compact_silu_math_table").body if not isinstance(n, ast.Assert)
        ]
        try:
            syntax(tree)
        except AssertionError:
            outcomes["compact_setup_proof_removed"] = "rejected"
        else:
            raise AssertionError("Removed compact table proof accepted")
    triangular = next((t for t in extra if "Chunk64TriangularRecurrence.__call__" in functions(t)), None)
    if triangular is not None:
        for label, method, statement in (
            ("triangular_runtime_readback", "__call__", "self.t.to_torch(matrix)"),
            ("triangular_constructor_upload", "__init__", "t.from_torch(None)"),
        ):
            tree = copy.deepcopy(triangular)
            prepend(tree, "Chunk64TriangularRecurrence", method, statement)
            try:
                syntax(tree)
            except AssertionError:
                outcomes[label] = "rejected"
            else:
                raise AssertionError(("Negative audit accepted triangular host operation", label))
        tree = copy.deepcopy(triangular)
        kernel = next(
            n
            for n in tree.body
            if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "CHUNK64_TRIANGULAR_COMPUTE"
        )
        kernel.value.value += "\n// unreviewed kernel change\n"
        try:
            extension_audit(tree, extra)
        except AssertionError:
            outcomes["triangular_kernel_change"] = "rejected"
        else:
            raise AssertionError("Changed triangular kernel accepted")
    return outcomes


def extension_audit(tree, extra):
    new_names = {
        "Chunk64TriangularRecurrence",
        "WeightedNormTail",
        "LocalSiluMultiply",
        "DirectCoefficientStreamingScan",
        "local_silu_math_table",
        "compact_silu_math_table",
    }
    globals_to_pin = {
        "CHUNK64_TRIANGULAR_READER",
        "CHUNK64_TRIANGULAR_COMPUTE",
        "CHUNK64_TRIANGULAR_EMPTY_WRITER",
        "WEIGHTED_NORM_COMPUTE_SOURCE",
        "LOCAL_SILU_READER_SOURCE",
        "LOCAL_SILU_COMPUTE_SOURCE",
        "DIRECT_COEFFICIENT_READER",
        "SILU_POS_LO",
        "SILU_POS_HI",
        "SILU_NEG_LO",
        "SILU_NEG_HI",
        "SILU_POS_COUNT",
        "SILU_NEG_COUNT",
        "SILU_COMPACT_ENTRIES",
    }

    def relevant(t):
        result = {}
        for n in t.body:
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in new_names:
                result[n.name] = digest(n)
            if isinstance(n, ast.Assign):
                names = {x.id for target in n.targets for x in ast.walk(target) if isinstance(x, ast.Name)}
                if names & globals_to_pin:
                    result[ast.unparse(n.targets[0])] = digest(n)
        return result

    present = relevant(tree)
    allowed = [relevant(ref) for ref in extra]
    for name, value in present.items():
        assert any(ref.get(name) == value for ref in allowed), ("Changed reviewed extension/helper/table/kernel", name)
    has_compact = "LocalSiluMultiply" in present
    return dict(
        pinned_extension_ast_sha256=present,
        compact_table_proof="Pinned setup derives finite active ranges and verifies all65536 raw FP32 codes including signed zeros and nonfinite payloads before upload; no data-dependent reference values"
        if has_compact
        else None,
        compact_entries=7488 if has_compact else None,
        compact_bytes_per_worker=29952 if has_compact else None,
        runtime_helper_host_exemption=False,
        host_table_proof_reexecuted=False,
    )


def description_delta(tree, references):
    matched = next(((label, ref) for label, ref in references if digest(tree) == digest(ref)), None)
    deltas = []
    if matched:
        label, reference = matched
        for before, after in zip(ast.walk(reference), ast.walk(tree)):
            if (
                isinstance(before, ast.Constant)
                and isinstance(after, ast.Constant)
                and isinstance(before.value, str)
                and isinstance(after.value, str)
                and before.value != after.value
            ):
                assert (
                    before.value.rstrip() == after.value.rstrip()
                ), "Non-whitespace executable change cannot be normalized"
                deltas.append(
                    dict(reference_line=before.lineno, source_line=after.lineno, before=before.value, after=after.value)
                )
    else:
        label = None
    # Negative controls ensure assigned strings and ordinary executable constants stay exact.
    a, b = ast.parse("KERNEL = 'code\\n        '"), ast.parse("KERNEL = 'code\\n'")
    assert digest(a) != digest(b)
    assert digest(ast.parse("x = 1")) != digest(ast.parse("x = 2"))
    return dict(
        policy="Trailing whitespace only in leading descriptions of prepare_prefill/prefill_forward and their explicit full_attention guarded V5 copies; no general string/constant removal",
        normalized_execution_ast_sha256=digest(tree),
        identical_reference=label,
        exact_description_deltas=deltas,
        assigned_kernel_and_executable_constant_negative_controls_pass=True,
    )


def audit(source):
    v5, reviewed, functional = reviewed_sources()
    extra = []
    extra_hashes = {}
    for filename, expected in EXTRA_REVIEWED.items():
        path = REVIEWED.parent / filename
        assert sha(path) == expected, ("Reviewed extension source changed", path)
        extra.append(ast.parse(path.read_text()))
        extra_hashes[str(path)] = expected
    text = Path(source).read_text()
    tree = ast.parse(text)
    syntax = syntax_audit(tree, v5, reviewed, extra)
    contract = contract_audit(tree, v5, functional)
    context_path = ROOT / "doc/context_contract.json"
    context = json.loads(context_path.read_text())
    assert context["hf_advertised_context"] == context["supported_context"] == 262144
    assert context["capability_reduction"] is None
    planners = planner_audit(tree, v5, contract["v6"], context["supported_context"])
    dispatch = dispatch_audit(tree, v5, reviewed, context["supported_context"])
    negative = negative_audit(v5, reviewed, functional, extra)
    assert "ttnn" not in sys.modules and "torch" not in sys.modules
    return dict(
        status="static_audit_pass",
        python_audit_pass=True,
        acceptance_eligible=False,
        source=str(Path(source).resolve()),
        source_sha256=sha(source),
        syntax=syntax,
        contract=contract,
        planner=planners,
        mock_dispatch=dispatch,
        negative_controls=negative,
        audit_source_sha256=sha(__file__),
        reviewed_extension_hashes=extra_hashes,
        reviewed_extension=extension_audit(tree, extra),
        description_normalization=description_delta(
            tree,
            [
                (str(V5), v5),
                (str(REVIEWED), reviewed),
                *[(str(REVIEWED.parent / name), ref) for name, ref in zip(EXTRA_REVIEWED, extra)],
            ],
        ),
        reference_hashes={
            str(V5): V5_SHA,
            str(REVIEWED): REVIEWED_SHA,
            str(FUNCTIONAL): FUNCTIONAL_SHA,
            str(context_path): sha(context_path),
        },
        context_contract_evidence_matches_current_source=context.get("fused_decoder_validation", {}).get(
            "decoder_source_sha256"
        )
        == sha(source),
        limitations=[
            "Does not execute TTNN, tensor arithmetic, custom kernels or device allocations.",
            "Exact PCC, long-stream determinism, traced latency, watcher cleanliness and physical context capacity still require unchanged device gates.",
            "AST checks reject known host transfers and dynamic code but do not prove absence of native-library fallback or arbitrary side effects of external TTNN calls.",
            "Optional observer callbacks in generic graph helpers are not executed; delivered decoder call sites remain subject to source review and runtime device_only audit.",
            "Setup allowances are per-method hashes of explicitly reviewed V5/8cda/compact/direct-coefficient source; any new setup change requires review, not a class-wide exemption.",
        ],
        device_executed=False,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, default=ROOT / "tt/fused_decoder.py")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    result = audit(args.snapshot)
    if args.output:
        assert not args.output.exists(), "Preserve prior audit evidence"
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("FUSED_SNAPSHOT_V6_STATIC_AUDIT_PASS", result["source_sha256"], result["planner"]["valid_plan_cases"])


if __name__ == "__main__":
    main()
