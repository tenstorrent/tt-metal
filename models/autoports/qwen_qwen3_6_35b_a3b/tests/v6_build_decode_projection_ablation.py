# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Source-only decode projection ablations; all prefill paths remain unchanged."""

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "doc/fused_decoder/endurance"
BASE = E / "v6_inline_router_silu_chunk64_corrected_fma_explicit_norm.py.snapshot"
BASE_SHA = "8391717476ae224ee9b82c203d7dec21ceab69632c5edd0abd89bf8cc4daeb03"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(node):
    return hashlib.sha256(ast.dump(node).encode()).hexdigest()


def named(scope, name):
    return next(n for n in scope.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)


def build(variant):
    assert sha(BASE) == BASE_SHA
    tree = ast.parse(BASE.read_text())
    original = copy.deepcopy(tree)
    cls = named(tree, "FusedDecoder")
    linear, attention = named(cls, "linear"), named(cls, "_linear_attention")
    undo = []
    if variant in ("native_qz", "native_qz_out"):
        linear.args.kwonlyargs.append(ast.arg(arg="ordered_output_only"))
        linear.args.kw_defaults.append(ast.Constant(value=True))
        for node in linear.body:
            if not isinstance(node, ast.If):
                continue
            condition = ast.unparse(node.test)
            if "key in ('attention_packed', 'output_only_packed')" in condition:
                undo.append((node, "test", copy.deepcopy(node.test)))
                node.test = ast.BoolOp(
                    op=ast.And(),
                    values=[
                        node.test,
                        ast.parse("key != 'output_only_packed' or ordered_output_only", mode="eval").body,
                    ],
                )
        assert len(undo) == 2
        output = next(
            n for n in attention.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "output_projected"
        )
        assert ast.unparse(output.value.func) == "self.linear" and output.value.args[1].value == "output_only_packed"
        output.value.keywords.append(
            ast.keyword(arg="ordered_output_only", value=ast.parse("not (decode and cache_boundary)", mode="eval").body)
        )
    if variant in ("native_out", "native_qz_out"):
        final = attention.body[-1]
        assert isinstance(final, ast.Return) and ast.unparse(final.value.func) == "self.linear"
        assert final.value.args[1].value == "linear_attn.out_proj.weight"
        target = next(n for n in final.value.keywords if n.arg == "ordered_out")
        assert ast.unparse(target.value) == "cache_boundary"
        undo.append((target, "value", copy.deepcopy(target.value)))
        target.value = ast.Constant(value=False)

    changed = {}
    original_cls = named(original, "FusedDecoder")
    for old in [n for n in original_cls.body if isinstance(n, ast.FunctionDef)]:
        new = named(cls, old.name)
        if digest(old) != digest(new):
            changed[old.name] = dict(before=digest(old), after=digest(new))
    assert set(changed) == ({"_linear_attention"} if variant == "native_out" else {"linear", "_linear_attention"})
    candidate = copy.deepcopy(tree)
    # Restore every intentional modification and prove complete module identity.
    for node, field, value in undo:
        setattr(node, field, value)
    if variant in ("native_qz", "native_qz_out"):
        linear.args.kwonlyargs.pop()
        linear.args.kw_defaults.pop()
        output.value.keywords.pop()
    assert ast.dump(tree) == ast.dump(original)
    ast.fix_missing_locations(candidate)
    source = (
        "# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC\n# SPDX-License-Identifier: Apache-2.0\n"
        + ast.unparse(candidate)
        + "\n"
    )
    compile(source, "<decode-only-projection-ablation>", "exec")

    # The only new selector is false exactly at public decode boundaries.
    truth = []
    for decode in (False, True):
        for cache_boundary in (False, True):
            ordered_qz = not (decode and cache_boundary)
            original_out = cache_boundary if decode else None
            ablated_out = False if decode else None
            if not decode or not cache_boundary:
                assert ordered_qz and original_out == ablated_out
            truth.append(
                dict(
                    decode=decode,
                    cache_boundary=cache_boundary,
                    ordered_qz=ordered_qz,
                    original_ordered_out=original_out,
                    ablated_ordered_out=ablated_out,
                )
            )
    return source, dict(
        classification="decode-only precision-cost ablation; no correctness or speed claim",
        base=str(BASE),
        base_sha256=BASE_SHA,
        variant=variant,
        modified_methods=changed,
        full_module_ast_restoration=True,
        selector_truth_table=truth,
        prefill_policy="unchanged, including public S=1 and internal tails (cache_boundary=False)",
        ordered_attention_packed="unchanged",
        ordered_router="unchanged",
        native_projection_route="existing x.shape[-2]==1 and key in self.decode_linear branch; existing weights/config/placement",
        native_qz_route="both ordered output_only_packed branches bypassed only at decode and cache_boundary; attention_packed remains enabled",
        native_out_route="only final decode return disables ordered_out; prefill returns earlier and original false boundary cases remain false",
        snapshot_sha256=hashlib.sha256(source.encode()).hexdigest(),
        acceptance_eligible=False,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=["native_qz", "native_out", "native_qz_out"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()
    snapshot = Path(str(args.output) + ".builder.py.snapshot")
    assert not any(path.exists() for path in (args.output, args.report, snapshot)), "Preserve evidence"
    source, report = build(args.variant)
    args.output.write_text(source)
    snapshot.write_bytes(Path(__file__).read_bytes())
    report.update(
        snapshot=str(args.output),
        builder_source=str(Path(__file__).resolve()),
        builder_sha256=sha(__file__),
        builder_snapshot=str(snapshot),
    )
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print("DECODE_ABLATION_STATIC_PASS", args.variant, sha(args.output))


if __name__ == "__main__":
    main()
