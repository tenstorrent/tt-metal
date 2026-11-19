# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Static decode-only post-sum norm fusion from the full-kind-preserved source."""

import argparse
import ast
import copy
import hashlib
import json
import sys
from pathlib import Path

from models.autoports.qwen_qwen3_6_35b_a3b.tests.frozen_source_compatibility import raw_sha256, verify_source

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "doc/fused_decoder/endurance"
BASE = E / "v6_corrected_fma_explicit_norm_full_v5.py.snapshot"
BASE_SHA = "52df1b042d445630ef9bdce0d100245fa2faad9dd801042dc638b24bf3fe50d9"
HELPER = ROOT / "tests/v6_gated_norm_tail_fusion_probe.py"
HELPER_SHA = "47d50c370b8038bc1b10f24f86c5e58f61222e21b6cc7a1bb22d6318055961bd"
TIMING = ROOT / "tests/v6_gated_norm_tail_timing_probe.py"
TIMING_SHA = "c0af8685423e6688a3362aa04a99c6db4d66999bebe1a5f9a890dc000f75994c"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(node):
    return hashlib.sha256(ast.dump(node).encode()).hexdigest()


def named(scope, name):
    return next(n for n in scope.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)


def module_hash(nodes):
    return digest(ast.Module(body=nodes, type_ignores=[]))


def build():
    assert verify_source(BASE, BASE_SHA) and verify_source(HELPER, HELPER_SHA) and verify_source(TIMING, TIMING_SHA)
    correctness_path = E / "v6_gated_norm_tail_fusion_device.json"
    timing_path = E / "v6_gated_norm_tail_timing_device.json"
    correctness, timing = [json.loads(path.read_text()) for path in (correctness_path, timing_path)]
    assert correctness["status"] == "completed_diagnostic"
    assert timing["status"] == "completed_microbenchmark"
    assert correctness["exact_native_tail"] and correctness["device_closed"] and timing["device_closed"]
    assert correctness["script_sha256"] == HELPER_SHA and timing["helper_sha256"] == HELPER_SHA
    assert timing["source_sha256"] == TIMING_SHA and timing["fused_debug"] is False
    assert len(timing["correctness"]) == 14
    assert all(row["fused_vs_explicit"]["bf16_code_differences"] == 0 for row in timing["correctness"])

    helper = ast.parse(HELPER.read_text())
    epsilon = next(n for n in helper.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "EPSILON")
    assert ast.literal_eval(epsilon.value) == 1e-6
    scope = {}
    exec(
        compile(
            ast.Module(body=[copy.deepcopy(epsilon), copy.deepcopy(named(helper, "kernel_source"))], type_ignores=[]),
            "<norm-tail-build-source>",
            "exec",
        ),
        scope,
    )
    compute = scope["kernel_source"]()
    compute_hash = hashlib.sha256(compute.encode()).hexdigest()
    assert compute_hash == correctness["kernel_sha256"]
    tail = copy.deepcopy(named(helper, "GatedNormTail"))
    tail_before = copy.deepcopy(tail)
    source = next(
        n
        for n in named(tail, "__init__").body
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "self.source"
    )
    assert ast.unparse(source.value) == "kernel_source()"
    source.value = ast.Name(id="GATED_NORM_TAIL_COMPUTE_SOURCE", ctx=ast.Load())
    assert digest(named(tail, "__call__")) == digest(named(tail_before, "__call__"))

    tree = ast.parse(BASE.read_text())
    original = copy.deepcopy(tree)
    cls = named(tree, "FusedDecoder")
    setup = named(cls, "from_state_dict")
    assert isinstance(setup.body[-1], ast.Return)
    addition = ast.parse(
        "if self.kind == 'linear_attention':\n"
        "    assert self.config.rms_norm_eps == 1e-6, 'Measured norm-tail epsilon contract'\n"
        "    self.decode_norm_tail = GatedNormTail(ttnn, mesh_device, state_dict['linear_attn.norm.weight'], debug=False)"
    ).body[0]
    setup.body.insert(len(setup.body) - 1, addition)
    attention = named(cls, "_linear_attention")
    start = next(
        i for i, n in enumerate(attention.body) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "variance"
    )
    end = next(
        i
        for i, n in enumerate(attention.body)
        if i > start and isinstance(n, ast.If) and ast.unparse(n.test) == "cache_boundary"
    )
    old_block = copy.deepcopy(attention.body[start : end + 1])
    common = copy.deepcopy(attention.body[start:end])
    gate_branch = copy.deepcopy(attention.body[end])
    assert ast.unparse(common[-1].targets[0]) == "weighted"
    assert isinstance(gate_branch.body[-1], ast.Assign) and ast.unparse(gate_branch.body[-1].targets[0]) == "gated"
    lookup = copy.deepcopy(gate_branch.body[:-1])
    assert [ast.unparse(n.targets[0]) for n in lookup] == ["silu_indices", "silu_indices", "gate_rows", "gate_rows"]
    fused_call = ast.parse("gated, _ = self.decode_norm_tail(wide, sums, gate_rows)").body[0]
    replacement = ast.If(
        test=ast.Name(id="cache_boundary", ctx=ast.Load()),
        body=lookup + [fused_call],
        orelse=common + copy.deepcopy(gate_branch.orelse),
    )
    attention.body[start : end + 1] = [replacement]
    # All decode=False work returns before this region. A public S=1 prefill
    # enters decode=True/cache_boundary=False; its arithmetic remains identical.
    assert module_hash(replacement.orelse) == module_hash(old_block[:-1] + old_block[-1].orelse)
    assert module_hash(replacement.body[:-1]) == module_hash(old_block[-1].body[:-1])
    candidate = copy.deepcopy(tree)
    attention.body[start : start + 1] = old_block
    setup.body.remove(addition)
    assert ast.dump(tree) == ast.dump(original), "Only setup and decode norm-tail region may change"
    literal = ast.Assign(
        targets=[ast.Name(id="GATED_NORM_TAIL_COMPUTE_SOURCE", ctx=ast.Store())], value=ast.Constant(value=compute)
    )
    candidate.body += [literal, tail]
    ast.fix_missing_locations(candidate)
    calls = [ast.unparse(n.func) for n in ast.walk(named(tail, "__call__")) if isinstance(n, ast.Call)]
    assert not any(c.startswith("torch.") or c.endswith((".from_torch", ".to_torch")) for c in calls)
    assert not any(
        isinstance(n, (ast.Import, ast.ImportFrom)) and "tests" in ast.unparse(n) for n in ast.walk(candidate)
    )
    assert not any(
        isinstance(n, ast.Call) and ast.unparse(n.func) in ("exec", "compile", "eval", "kernel_source")
        for n in ast.walk(candidate)
    )
    code = (
        "# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC\n# SPDX-License-Identifier: Apache-2.0\n"
        + ast.unparse(candidate)
        + "\n"
    )
    compile(code, "<norm-tail-static-candidate>", "exec")
    assert "ttnn" not in sys.modules and "torch" not in sys.modules
    candidate_cls = named(candidate, "FusedDecoder")
    return (
        code,
        compute,
        dict(
            classification="Decode post-sum norm-tail fusion diagnostic; full decoder correctness/speed unmeasured",
            base=str(BASE),
            base_sha256=BASE_SHA,
            helper=str(HELPER),
            helper_sha256=raw_sha256(HELPER),
            helper_historical_sha256=HELPER_SHA,
            timing_probe=str(TIMING),
            timing_probe_sha256=raw_sha256(TIMING),
            timing_probe_historical_sha256=TIMING_SHA,
            component_evidence={str(path): sha(path) for path in (correctness_path, timing_path)},
            component_correctness_cases=14,
            component_timing=timing["timing"],
            component_timing_scope=timing["timing_scope"],
            kernel_sha256=compute_hash,
            full_module_restoration=True,
            changed_existing_methods={
                name: dict(
                    before=digest(named(named(original, "FusedDecoder"), name)),
                    after=digest(named(candidate_cls, name)),
                )
                for name in ("from_state_dict", "_linear_attention")
            },
            unchanged_tail_runtime_ast_sha256=digest(named(tail, "__call__")),
            tail_setup_change="Only source-generation call replaced with static compute-string constant",
            unchanged_prefill_singleton_tail_ast_sha256=module_hash(replacement.orelse),
            unchanged_silu_lookup_ast_sha256=module_hash(lookup),
            runtime_scope="linear_attention decode=True/cache_boundary=True after unchanged native FP32 square/sum",
            preserved="All prefill incl public S=1, full-kind V5 branches/setup, native sum, SiLU lookup, head packing, output projection",
            setup_only_constants="Gamma from real layer state_dict expanded to FP32[1,32,128]; epsilon fixed to measured1e-6; debugFalse",
            runtime_host_audit="No Torch/from_torch/to_torch in tail; no source imports or dynamic Python compilation",
            snapshot_sha256=hashlib.sha256(code.encode()).hexdigest(),
            acceptance_eligible=False,
            device_executed=False,
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()
    compute_path = Path(str(args.output) + ".compute.cpp.snapshot")
    builder_path = Path(str(args.output) + ".builder.py.snapshot")
    assert not any(p.exists() for p in (args.output, args.report, compute_path, builder_path)), "Preserve evidence"
    code, compute, report = build()
    args.output.write_text(code)
    compute_path.write_text(compute)
    builder_path.write_bytes(Path(__file__).read_bytes())
    report.update(
        snapshot=str(args.output),
        compute_snapshot=str(compute_path),
        builder_source=str(Path(__file__).resolve()),
        builder_sha256=sha(__file__),
        builder_snapshot=str(builder_path),
    )
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print("NORM_TAIL_STATIC_BUILD_PASS", sha(args.output))


if __name__ == "__main__":
    main()
