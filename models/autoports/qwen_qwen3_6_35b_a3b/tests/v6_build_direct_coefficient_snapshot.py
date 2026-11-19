# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Build a static decode-only direct-coefficient plus native cast-placement candidate."""

import argparse
import ast
import copy
import hashlib
import json
import sys
from pathlib import Path

from models.autoports.qwen_qwen3_6_35b_a3b.tests.frozen_source_compatibility import raw_sha256, verify_source

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
E = ROOT / "doc/fused_decoder/endurance"
BASE = E / "v6_corrected_fma_full_v5_streaming_scan_norm_tail_fused_matmul.py.snapshot"
BASE_SHA = "8cda999c18f9245c872e40d7f67e027838b56396056ede43e162b7a8386f3faa"
HELPER = ROOT / "tests/v6_direct_coefficient_stream_probe.py"
HELPER_SHA = "94e8be48e0fdd447310af7a1d6974c0e51aee6d2f721acfcad10fdbe463121e7"
CAST = ROOT / "tests/v6_projection_cast_placement_probe.py"
CAST_SHA = "58182574be57b313d6ae236001465e41b45fbdc50b0907397240f37c6f78f6eb"
STREAM = ROOT / "tests/v6_streaming_projection_scan_probe.py"
STREAM_SHA = "b555f5caae383fb2cf1976860e92b9af3f1f2e2e39ec560e4c43d4d05bb23354"
EVIDENCE = E / "v6_direct_coefficient_stream_device.json"
CAST_EVIDENCE = E / "v6_projection_cast_placement_device.json"
RENAMES = {
    "READER": "DIRECT_COEFFICIENT_READER",
    "stream.schedule": "terminal_projection_schedule",
    "stream.COMPUTE": "STREAMING_SCAN_COMPUTE",
    "stream.WRITER": "STREAMING_SCAN_WRITER",
    "stream.PRODUCT_WRITER": "STREAMING_SCAN_PRODUCT_WRITER",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def named(scope, name):
    return next(n for n in scope.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)


def constant(scope, name):
    return next(
        ast.literal_eval(n.value)
        for n in scope.body
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == name
    )


class Rename(ast.NodeTransformer):
    def __init__(self, mapping):
        self.mapping = mapping

    def visit_Attribute(self, node):
        text = ast.unparse(node)
        return ast.parse(self.mapping[text], mode="eval").body if text in self.mapping else self.generic_visit(node)

    def visit_Name(self, node):
        return ast.parse(self.mapping[node.id], mode="eval").body if node.id in self.mapping else node


def validate_evidence():
    for path, expected in [(BASE, BASE_SHA), (HELPER, HELPER_SHA), (CAST, CAST_SHA), (STREAM, STREAM_SHA)]:
        assert verify_source(path, expected), path
    reports = []
    for path, helper_sha, count in [(EVIDENCE, HELPER_SHA, 40), (CAST_EVIDENCE, CAST_SHA, 32)]:
        d = json.loads(path.read_text())
        assert d["status"] == "component_pass" and d["device_closed"] and d["device_executed"]
        assert d["script_sha256"] == helper_sha and d["streaming_helper_sha256"] == STREAM_SHA
        assert d["fixture_cases"] == len(d["cases"]) == count
        assert all(all(o["exact"] for o in c["outputs"].values()) for c in d["cases"])
        assert all(t["after_100_replays"]["exact"] for t in d["timing"])
        for dep in d["dependencies"]:
            assert verify_source(dep["path"], dep["sha256"]), dep
        reports.append(d)
    direct, cast = reports
    singletons = [c for c in direct["cases"] if "coefficient_bits" in c]
    assert len(singletons) == 38
    assert all(all(c[k]["exact"] for k in ("coefficient_bits", "product_fp32", "partial_fp32")) for c in singletons)
    assert cast["rounding_midpoints_bf16"]["exact"]
    return direct, cast


def embed(original):
    helper = ast.parse(HELPER.read_text())
    strings = {"READER": constant(helper, "READER")}
    for name in ("COMPUTE", "WRITER", "PRODUCT_WRITER"):
        strings[name] = constant(original, RENAMES["stream." + name])
    evidence = {}
    for name, source in strings.items():
        path = Path(str(EVIDENCE) + f".{name.lower()}.cpp.snapshot")
        assert path.read_text() == source, path
        evidence[name] = dict(path=str(path), sha256=sha(path))
    runtime = named(helper, "DirectCoefficientStreamingScan")
    renamed = Rename(RENAMES).visit(copy.deepcopy(runtime))
    restored = Rename({v: k for k, v in RENAMES.items()}).visit(copy.deepcopy(renamed))
    assert ast.dump(restored) == ast.dump(runtime), "Measured helper alpha equivalence"
    terminal = ast.parse((ROOT / "tests/v6_terminal_projection_scan_probe.py").read_text())
    schedule = copy.deepcopy(named(terminal, "schedule"))
    schedule.name = "terminal_projection_schedule"
    assert ast.dump(schedule) == ast.dump(named(original, schedule.name))
    return (
        [ast.Assign([ast.Name(RENAMES["READER"], ast.Store())], ast.Constant(strings["READER"])), renamed],
        strings,
        evidence,
    )


def layout_evidence(original):
    cls = named(original, "FusedDecoder")
    norm = ast.unparse(named(cls, "norm"))
    forward = ast.unparse(named(cls, "_forward"))
    attn = ast.unparse(named(cls, "_linear_attention"))
    assert "original_memory = x.memory_config()" in norm
    assert "return ttnn.to_memory_config(result, original_memory)" in norm
    assert "mlp_input = ttnn.to_memory_config(mlp_input, DRAM)" in forward
    assert "ttnn.experimental.nlp_concat_heads_decode(gated, num_heads=32)" in attn
    paths = [
        ROOT / "tests/run_decoder.py",
        ROOT / "tests/v5_layer_stream_probe.py",
        REPO / "ttnn/ttnn/operations/core.py",
        REPO / "ttnn/cpp/ttnn-nanobind/pytensor.cpp",
        REPO / "tt_metal/api/tt-metalium/tensor/spec/memory_config/memory_config.hpp",
        REPO
        / "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode/device/nlp_concat_heads_decode_device_operation.cpp",
    ]
    assert "mem_config=memory_config" in paths[2].read_text()
    assert "mem_config.value_or(MemoryConfig())" in paths[3].read_text()
    assert "MemoryConfig();  // Interleaved DRAM" in paths[4].read_text()
    return dict(
        sources=[dict(path=str(p), sha256=sha(p)) for p in paths],
        attention_qz="Input normalization restores caller memory; standard benchmark and stream harness uploads omit memory_config, resolving through Tensor mem_config.value_or(MemoryConfig()) to interleaved DRAM. L1 component inputs are conditional, not this harness path.",
        router="Decode _forward explicitly places post-attention normalized mlp_input in DRAM before _moe; original producer placement is retained.",
        output_projection="nlp_concat_heads_decode(num_heads=32) returns WIDTH_SHARDED L1 on 32 row-major cores with shard shape (32,128); following reshape preserves its memory configuration.",
        producer_placement_changes=False,
    )


def build():
    direct_evidence, cast_evidence = validate_evidence()
    original = ast.parse(BASE.read_text())
    embedded, strings, kernel_evidence = embed(original)
    candidate = copy.deepcopy(original)
    cls = named(candidate, "FusedDecoder")
    setup = named(cls, "from_state_dict")
    owners = [
        (n.body, i)
        for n in ast.walk(setup)
        if hasattr(n, "body") and isinstance(n.body, list)
        for i, child in enumerate(n.body)
        if isinstance(child, ast.Assign) and ast.unparse(child.targets[0]) == "self.streaming_projection_scan"
    ]
    assert len(owners) == 1
    owner, index = owners[0]
    owner.insert(
        index + 1, ast.parse("self.direct_coefficient_scan = DirectCoefficientStreamingScan(ttnn, mesh_device)").body[0]
    )
    linear = named(cls, "linear")
    restorations = []
    for index, branch in enumerate(linear.body[1:4]):
        assert isinstance(branch, ast.If)
        selected = [
            copy.deepcopy(n)
            for n in branch.body
            if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) in ("projection_k", "blocks", "width", "weight")
        ]
        assert [ast.unparse(n.targets[0]) for n in selected] == (["projection_k"] if index < 2 else []) + [
            "blocks",
            "width",
            "weight",
        ]
        selected += ast.parse("partial = self.direct_coefficient_scan(weight, x)").body
        selected += copy.deepcopy(branch.body[-3:])
        # Only the already-proven placement rewrite; explicit native RTNE typecast remains.
        if index < 2:
            old = selected[-1]
            assert isinstance(old, ast.Return) and ast.unparse(old.value.func) == "ttnn.to_memory_config"
            assert ast.unparse(old.value.args[1]) == "DRAM"
            direct = ast.Return(copy.deepcopy(old.value.args[0]))
            assert ast.unparse(direct.value.func) == "ttnn.reshape"
            cast = direct.value.args[0]
            assert ast.unparse(cast.func) == "ttnn.typecast" and not cast.keywords
            cast.keywords.append(ast.keyword("memory_config", ast.Name("DRAM", ast.Load())))
            selected[-1] = direct
        branch.body.insert(0, ast.If(ast.Name("terminal_scan", ast.Load()), selected, []))
        restorations.append(branch)
    modified = copy.deepcopy(candidate)
    del owner[owners[0][1] + 1]
    for branch in restorations:
        del branch.body[0]
    assert ast.dump(candidate) == ast.dump(original), "Full 8cda module AST restoration"
    candidate = modified
    old_cls, cls = named(original, "FusedDecoder"), named(candidate, "FusedDecoder")
    changed = [
        n.name for n in cls.body if isinstance(n, ast.FunctionDef) and ast.dump(n) != ast.dump(named(old_cls, n.name))
    ]
    assert changed == ["from_state_dict", "linear"]
    old_linear, new_linear = named(old_cls, "linear"), named(cls, "linear")
    assert ast.dump(old_linear.args) == ast.dump(new_linear.args)
    assert ast.unparse(new_linear.args.kwonlyargs[-1]) == "terminal_scan"
    assert ast.literal_eval(new_linear.args.kw_defaults[-1]) is False
    guard_checks = {}
    for method in ("from_state_dict", "linear", "norm", "_moe", "_forward", "prepare_prefill", "prefill_forward"):
        before = next(n for n in named(old_cls, method).body if isinstance(n, ast.If))
        after = next(n for n in named(cls, method).body if isinstance(n, ast.If))
        assert ast.dump(before) == ast.dump(after), method
        guard_checks[method] = hashlib.sha256(ast.dump(after).encode()).hexdigest()
    flags = []
    for method in ("_linear_attention", "_moe"):
        assert ast.dump(named(old_cls, method)) == ast.dump(named(cls, method))
        for call in ast.walk(named(cls, method)):
            if isinstance(call, ast.Call) and ast.unparse(call.func) == "self.linear":
                for kw in call.keywords:
                    if kw.arg == "terminal_scan":
                        flags.append(
                            dict(method=method, key=ast.literal_eval(call.args[1]), gate=ast.unparse(kw.value))
                        )
    assert [c["gate"] for c in flags] == ["decode and cache_boundary"] * 3 + ["ordered_router"]
    assert [
        ast.unparse(k.value)
        for n in ast.walk(named(cls, "_forward"))
        if isinstance(n, ast.Call) and ast.unparse(n.func) == "self._moe"
        for k in n.keywords
        if k.arg == "ordered_router"
    ] == ["decode and cache_boundary"]
    candidate.body += embedded
    ast.fix_missing_locations(candidate)
    assert not any(
        isinstance(n, (ast.Import, ast.ImportFrom)) and "tests" in ast.unparse(n) for n in ast.walk(candidate)
    )
    assert not any(
        isinstance(n, ast.Call) and ast.unparse(n.func) in ("exec", "eval", "compile") for n in ast.walk(candidate)
    )
    runtime = named(named(candidate, "DirectCoefficientStreamingScan"), "__call__")
    calls = [ast.unparse(n.func) for n in ast.walk(runtime) if isinstance(n, ast.Call)]
    assert not any(
        c.startswith("torch.") or c.endswith((".from_torch", ".to_torch", ".to_layout", ".to_memory_config", ".gather"))
        for c in calls
    )
    for n in original.body:
        if isinstance(n, ast.ClassDef) and n.name != "FusedDecoder":
            assert ast.dump(n) == ast.dump(named(candidate, n.name)), n.name
    code = (
        "# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC\n# SPDX-License-Identifier: Apache-2.0\n"
        + ast.unparse(candidate)
        + "\n"
    )
    compile(code, "<static-direct-coefficient-cast-placement>", "exec")
    assert "ttnn" not in sys.modules and "torch" not in sys.modules
    report = dict(
        classification="Standalone decode-only direct coefficient plus native cast-placement candidate; integrated correctness/performance unmeasured",
        base=str(BASE),
        base_sha256=BASE_SHA,
        helper=str(HELPER),
        helper_sha256=raw_sha256(HELPER),
        helper_historical_sha256=HELPER_SHA,
        cast_helper=str(CAST),
        cast_helper_sha256=raw_sha256(CAST),
        cast_helper_historical_sha256=CAST_SHA,
        streaming_helper=str(STREAM),
        streaming_helper_sha256=raw_sha256(STREAM),
        streaming_helper_historical_sha256=STREAM_SHA,
        component_evidence=[dict(path=str(p), sha256=sha(p)) for p in (EVIDENCE, CAST_EVIDENCE)],
        component_direct_cases=40,
        component_cast_cases=32,
        component_midpoint_gate="256 signed below/at/above BF16 midpoint controls exact; explicit RTNE typecast retained",
        component_direct_timing=direct_evidence["timing"],
        component_cast_timing=cast_evidence["timing"],
        component_kernel_evidence=kernel_evidence,
        full_module_restoration=True,
        restoration="Remove exactly three leading terminal_scan early-return guards and one direct helper setup assignment; remove appended reader literal/helper class: complete original 8cda AST restored",
        changed_existing_methods=changed,
        decode_calls_unchanged=flags,
        linear_signature_and_false_default_unchanged=True,
        prefill_false_branches_exact=True,
        full_attention_v5_guard_ast_sha256=guard_checks,
        helper_alpha_equivalence=True,
        existing_schedule_exact=True,
        all_existing_static_classes_exact=True,
        producer_layout_evidence=layout_evidence(original),
        measured_regressions="Direct coefficient DRAM attention K512: 0.179442204535 -> 0.181002705358 ms (+0.87%); K1024 0.192631967366 -> 0.195371196605 ms (+1.42%). K1024 remains prefill-only and is not changed by this integration. L1 gains cannot be substituted for actual DRAM input timings.",
        preserved="All prefill including singleton/internal K1024 tails, full-attention V5, fused corrected matmul, explicit normalization, state/cache arithmetic, native ordered FP32 block reduction, native explicit typecast; context capacity unchanged",
        runtime_audit="Static measured class/kernel literals; no test imports/runtime AST, Torch/host tensor conversion, untilize/gather/tilize or input reshard in new direct helper",
        kernel_sha256={k: hashlib.sha256(v.encode()).hexdigest() for k, v in strings.items()},
        snapshot_sha256=hashlib.sha256(code.encode()).hexdigest(),
        device_executed=False,
        acceptance_eligible=False,
    )
    return code, report, strings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=E)
    args = parser.parse_args()
    stem = args.output_dir / "v6_corrected_fma_full_v5_direct_coeff_cast_placement_fused_matmul_integration"
    snapshot = Path(str(stem) + ".py.snapshot")
    provenance = Path(str(stem) + ".provenance.json")
    builder = Path(str(stem) + ".builder.py.snapshot")
    assert not any(p.exists() for p in (snapshot, provenance, builder)), "Preserve existing evidence"
    code, report, strings = build()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    kernel_paths = {k: Path(str(stem) + f".{k.lower()}.cpp.snapshot") for k in strings}
    assert not any(p.exists() for p in kernel_paths.values())
    snapshot.write_text(code)
    builder.write_bytes(Path(__file__).read_bytes())
    for k, path in kernel_paths.items():
        path.write_text(strings[k])
    report.update(
        snapshot=str(snapshot),
        builder_source=str(Path(__file__).resolve()),
        builder_sha256=sha(__file__),
        builder_snapshot=str(builder),
        kernel_snapshots={k: str(p) for k, p in kernel_paths.items()},
    )
    provenance.write_text(json.dumps(report, indent=2) + "\n")
    print("DIRECT_COEFFICIENT_CAST_PLACEMENT_STATIC_BUILD_PASS", snapshot.name, sha(snapshot))


if __name__ == "__main__":
    main()
