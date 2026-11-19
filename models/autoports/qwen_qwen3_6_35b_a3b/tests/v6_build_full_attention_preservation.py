# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Embed qualified V5 full-attention bodies without altering corrected linear code.

Pure source construction/audit; no model imports, TTNN, Torch, or hardware.
"""

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "doc/fused_decoder/endurance"
BASE = E / "v6_inline_router_silu_chunk64_corrected_fma_explicit_norm.py.snapshot"
BASE_SHA = "8391717476ae224ee9b82c203d7dec21ceab69632c5edd0abd89bf8cc4daeb03"
V5 = ROOT / "doc/fused_decoder/selected_fused_decoder_v5.py.snapshot"
V5_SHA = "8ac733159befba919abf705603f3aada9d105195c7b495d8cd25aa65e531769f"
GUARDED = {"from_state_dict", "linear", "norm", "_moe", "_forward", "prepare_prefill", "prefill_forward"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(node):
    return hashlib.sha256(ast.dump(node).encode()).hexdigest()


def named(scope, name):
    return next(n for n in scope.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)


def planner_function(method):
    method = copy.deepcopy(method)
    scope = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "<isolated-source-planner>", "exec"), scope)
    return scope[method.name]


def build():
    assert sha(BASE) == BASE_SHA and sha(V5) == V5_SHA
    tree = ast.parse(BASE.read_text())
    original = copy.deepcopy(tree)
    old_tree = ast.parse(V5.read_text())
    cls, old_cls = named(tree, "FusedDecoder"), named(old_tree, "FusedDecoder")
    assert digest(named(tree, "reshape")) == digest(named(old_tree, "reshape"))
    guards = []
    proofs = {}
    for name in sorted(GUARDED):
        method, old = named(cls, name), named(old_cls, name)
        original_method = copy.deepcopy(method)
        condition = (
            "hf_config.layer_types[layer_idx] == 'full_attention'"
            if name == "from_state_dict"
            else "self.kind == 'full_attention'"
        )
        guard = ast.If(test=ast.parse(condition, mode="eval").body, body=copy.deepcopy(old.body), orelse=[])
        # Preserve the corrected method's actual docstring and place the guard
        # before its first executable statement. Removing it restores exact AST.
        index = int(
            bool(
                method.body
                and isinstance(method.body[0], ast.Expr)
                and isinstance(method.body[0].value, ast.Constant)
                and isinstance(method.body[0].value.value, str)
            )
        )
        method.body.insert(index, guard)
        guards.append((method, index, guard))
        selected_full = copy.deepcopy(method)
        selected_full.body = copy.deepcopy(guard.body)
        # V6 adds optional ordered flags to shared methods. Full V5 call sites
        # never supply them; retain the superset signature for linear callers.
        current_args, old_args = copy.deepcopy(selected_full.args), copy.deepcopy(old.args)
        extras = len(current_args.kwonlyargs) - len(old_args.kwonlyargs)
        assert extras >= 0
        if extras:
            assert name in {"linear", "_moe"}
            assert all(isinstance(x, ast.Constant) and x.value is False for x in current_args.kw_defaults[-extras:])
            del current_args.kwonlyargs[-extras:]
            del current_args.kw_defaults[-extras:]
        assert ast.dump(current_args) == ast.dump(old_args)
        selected_full.args = old_args
        assert ast.dump(selected_full) == ast.dump(old)
        selected_linear = copy.deepcopy(method)
        selected_linear.body.pop(index)
        assert ast.dump(selected_linear) == ast.dump(original_method)
        proofs[name] = dict(
            full_kind_guard=condition,
            original_v5_method_ast_sha256=digest(old),
            full_branch_normalized_method_ast_sha256=digest(selected_full),
            original_linear_method_ast_sha256=digest(original_method),
            linear_branch_method_ast_sha256=digest(selected_linear),
            extra_optional_false_keywords_for_linear_compatibility=extras,
        )
    # Every unguarded shared method must already match V5, except the linear-only
    # recurrence implementation, which full attention never calls.
    shared_unchanged = {}
    for old in [n for n in old_cls.body if isinstance(n, ast.FunctionDef)]:
        if old.name in GUARDED or old.name == "_linear_attention":
            continue
        current = named(cls, old.name)
        assert ast.dump(current) == ast.dump(old), old.name
        shared_unchanged[old.name] = digest(old)
    candidate = copy.deepcopy(tree)
    for method, index, guard in guards:
        assert method.body[index] is guard
        method.body.pop(index)
    assert ast.dump(tree) == ast.dump(original), "Full module linear specialization must restore 839 exactly"
    ast.fix_missing_locations(candidate)
    code = (
        "# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC\n# SPDX-License-Identifier: Apache-2.0\n"
        + ast.unparse(candidate)
        + "\n"
    )
    compile(code, "<full-attention-preservation-candidate>", "exec")
    assert not any(
        isinstance(n, (ast.Import, ast.ImportFrom))
        and ("tests" in ast.unparse(n) or "functional_decoder" in ast.unparse(n))
        for n in ast.walk(candidate)
    )
    assert not any(
        isinstance(n, ast.Call) and ast.unparse(n.func) in ("exec", "compile", "eval") for n in ast.walk(candidate)
    )

    actual = planner_function(named(named(candidate, "FusedDecoder"), "prepare_prefill"))
    references = {
        "linear_attention": planner_function(named(named(original, "FusedDecoder"), "prepare_prefill")),
        "full_attention": planner_function(named(old_cls, "prepare_prefill")),
    }
    counts = {}
    for kind, reference in references.items():
        self = SimpleNamespace(
            kind=kind,
            config=SimpleNamespace(max_position_embeddings=262144),
            prepare_positions=lambda **kw: tuple(sorted(kw.items())),
        )
        count = 0
        for start in [0, 1, 31, 32, 63, 64, 65, 127, 255, 256]:
            for length in [1, 31, 32, 33, 63, 64, 65, 127, 128, 256, 257]:
                for chunk in [32, 64, 256, 4096]:
                    assert actual(self, seq_len=length, start_pos=start, chunk_size=chunk) == reference(
                        self, seq_len=length, start_pos=start, chunk_size=chunk
                    )
                    count += 1
        counts[kind] = count
    return code, dict(
        classification="full-kind scope preservation diagnostic; not selected or device validated",
        corrected_linear_source=str(BASE),
        corrected_linear_source_sha256=BASE_SHA,
        qualified_full_source=str(V5),
        qualified_full_source_sha256=V5_SHA,
        guarded_method_proofs=proofs,
        shared_unchanged_method_ast_sha256=shared_unchanged,
        entire_module_linear_specialization_ast_restoration=True,
        planner_cpu_exact_comparisons=counts,
        setup_policy="Full kind returns from exact V5 setup before any V6-only allocation; linear setup unchanged",
        full_runtime_policy="Embedded V5 fused setup and shared bodies; original generalized gate/residual/planner; no functional fallback/source import",
        linear_runtime_policy="839 method bodies and all helpers unchanged after false full-kind guards",
        snapshot_sha256=hashlib.sha256(code.encode()).hexdigest(),
        acceptance_eligible=False,
        device_executed=False,
        performance_claim="none; full-kind stress and latency still required",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()
    builder_snapshot = Path(str(args.output) + ".builder.py.snapshot")
    assert not any(path.exists() for path in (args.output, args.report, builder_snapshot)), "Preserve evidence"
    code, report = build()
    args.output.write_text(code)
    builder_snapshot.write_bytes(Path(__file__).read_bytes())
    report.update(
        snapshot=str(args.output),
        builder_source=str(Path(__file__).resolve()),
        builder_sha256=sha(__file__),
        builder_snapshot=str(builder_snapshot),
    )
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print("FULL_ATTENTION_PRESERVATION_STATIC_PASS", sha(args.output))


if __name__ == "__main__":
    main()
