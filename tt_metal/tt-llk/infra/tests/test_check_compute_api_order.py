#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the compute-kernel API ordering checker.

Run with:
    python3 -m pytest tt_metal/tt-llk/infra/tests/test_check_compute_api_order.py
or directly:
    python3 tt_metal/tt-llk/infra/tests/test_check_compute_api_order.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import clang.cindex as ci
import pytest
import yaml

HERE = Path(__file__).resolve().parent
INFRA = HERE.parent
REPO = HERE.parents[3]
CHECKER = INFRA / "check_compute_api_order.py"
RULES = INFRA / "compute_api_order_rules.yaml"
SFPU_RULES = INFRA / "compute_api_order_sfpu.yaml"
FIXTURES = HERE / "compute_api_order_fixtures"
SFPI_CANDIDATES = [
    REPO / "runtime" / "sfpi",
    Path("/opt/tenstorrent/sfpi"),
    Path("/opt/sfpi"),
]


def _load_checker():
    spec = importlib.util.spec_from_file_location("compute_api_order", CHECKER)
    module = importlib.util.module_from_spec(spec)
    sys.modules["compute_api_order"] = module
    spec.loader.exec_module(module)
    module._configure_libclang(None)
    return module


def _sfpi():
    for p in SFPI_CANDIDATES:
        if (p / "include" / "sfpi.h").is_file():
            return p
    pytest.skip("SFPI headers not installed; set TT_METAL_SFPI_ROOT")


@pytest.fixture(scope="session")
def checker():
    return _load_checker()


@pytest.fixture(scope="session")
def tables(checker):
    return {arch: checker.RuleTable.load(arch) for arch in ("wormhole_b0", "blackhole")}


def analyze(checker, fixture: Path, arch: str, thread: str) -> list:
    table = checker.RuleTable.load(arch)
    args = checker.synth_args(str(fixture), arch, thread, _sfpi())
    analyzer = checker.Analyzer(args, thread, arch, table, str(fixture))
    analyzer.analyze_paths()
    return analyzer.findings


CASES = [
    # (fixture, arch, thread, expected rule counts)
    ("ok_binary.cpp", "wormhole_b0", "MATH", {}),
    ("ok_binary.cpp", "wormhole_b0", "UNPACK", {}),
    ("missing_binary_init.cpp", "wormhole_b0", "MATH", {"missing-init": 1}),
    ("missing_binary_init.cpp", "wormhole_b0", "UNPACK", {"missing-init": 1}),
    ("mismatch_math_init.cpp", "wormhole_b0", "MATH", {"init-arg-mismatch": 1}),
    ("wrong_init_math.cpp", "wormhole_b0", "MATH", {"wrong-init": 1}),
    ("unknown_args.cpp", "wormhole_b0", "MATH", {}),
    ("if_constexpr.cpp", "wormhole_b0", "MATH", {}),
    ("ok_sfpu.cpp", "wormhole_b0", "MATH", {}),
    ("missing_sfpu_init.cpp", "wormhole_b0", "MATH", {"missing-init": 1}),
    ("missing_sfpu_init.cpp", "wormhole_b0", "UNPACK", {}),
    ("ok_tilize.cpp", "wormhole_b0", "UNPACK", {}),
    ("missing_tilize_uninit.cpp", "wormhole_b0", "UNPACK", {"missing-uninit": 1}),
    ("missing_tilize_uninit.cpp", "wormhole_b0", "PACK", {"missing-uninit": 1}),
    ("reconfig_after_tilize_init.cpp", "wormhole_b0", "PACK", {"wrong-init": 1}),
    ("ok_bcast.cpp", "wormhole_b0", "UNPACK", {}),
    ("ok_bcast.cpp", "wormhole_b0", "MATH", {}),
    ("missing_bcast_uninit.cpp", "wormhole_b0", "UNPACK", {"missing-uninit": 1}),
    ("missing_bcast_uninit.cpp", "wormhole_b0", "MATH", {"missing-uninit": 1}),
    ("missing_pack_untilize_uninit.cpp", "wormhole_b0", "PACK", {"missing-uninit": 1}),
    ("deprecated_untilize_init.cpp", "wormhole_b0", "MATH", {"deprecated-init": 1, "deprecated-api": 1}),
    ("legacy_reg_api.cpp", "wormhole_b0", "MATH", {"deprecated-api": 2}),
    ("unpack_config_mismatch.cpp", "wormhole_b0", "UNPACK", {"init-arg-mismatch": 1}),
    ("loop_invariant_init.cpp", "wormhole_b0", "MATH", {"loop-invariant-init": 1}),
    # Blackhole uses the *_impl LLK entry points; run the same contract there.
    ("ok_binary.cpp", "blackhole", "MATH", {}),
    ("ok_binary.cpp", "blackhole", "UNPACK", {}),
    ("missing_binary_init.cpp", "blackhole", "MATH", {"missing-init": 1}),
    ("mismatch_math_init.cpp", "blackhole", "MATH", {"init-arg-mismatch": 1}),
    ("wrong_init_math.cpp", "blackhole", "MATH", {"wrong-init": 1}),
    ("ok_tilize.cpp", "blackhole", "UNPACK", {}),
    ("missing_tilize_uninit.cpp", "blackhole", "UNPACK", {"missing-uninit": 1}),
    ("missing_tilize_uninit.cpp", "blackhole", "PACK", {"missing-uninit": 1}),
    ("reconfig_after_tilize_init.cpp", "blackhole", "PACK", {"wrong-init": 1}),
    ("ok_bcast.cpp", "blackhole", "UNPACK", {}),
    ("missing_bcast_uninit.cpp", "blackhole", "MATH", {"missing-uninit": 1}),
    ("missing_pack_untilize_uninit.cpp", "blackhole", "PACK", {"missing-uninit": 1}),
    ("deprecated_untilize_init.cpp", "blackhole", "MATH", {"deprecated-init": 1, "deprecated-api": 1}),
    ("legacy_reg_api.cpp", "blackhole", "MATH", {"deprecated-api": 2}),
    ("unpack_config_mismatch.cpp", "blackhole", "UNPACK", {"init-arg-mismatch": 1}),
    ("loop_invariant_init.cpp", "blackhole", "MATH", {"loop-invariant-init": 1}),
]


@pytest.mark.parametrize("fixture,arch,thread,expected", CASES)
def test_fixture(checker, fixture, arch, thread, expected):
    findings = analyze(checker, FIXTURES / fixture, arch, thread)
    got = {}
    for f in findings:
        got[f.rule] = got.get(f.rule, 0) + 1
    assert got == expected, "\n".join(str(f) for f in findings)


def test_blackhole_fast_tilize_fallback_has_no_false_positives(checker):
    """BH fast_tilize_init/uninit route through tilize_init/uninit in a runtime branch."""
    kernel = (
        REPO
        / "models/demos/deepseek_v3/tests/fused_op_unit_tests/moe/tilize_16x32/kernels/tilize_16x32_kernel.cpp"
    )
    if not kernel.is_file():
        pytest.skip("tilize_16x32 test kernel is not present")
    for thread in ("UNPACK", "MATH", "PACK"):
        findings = analyze(checker, kernel, "blackhole", thread)
        assert not findings, "\n".join(str(f) for f in findings)


def test_legacy_deprecated_calls_are_reported(checker):
    """acquire_dst/release_dst are deprecated but are not init functions."""
    kernel = REPO / "tests/tt_metal/tt_metal/test_kernels/compute/dfb_c2_pipeline_2_0.cpp"
    if not kernel.is_file():
        pytest.skip("dfb legacy test kernel is not present")
    findings = analyze(checker, kernel, "wormhole_b0", "MATH")
    deprecated = [f for f in findings if f.rule == "deprecated-api"]
    assert {f.op for f in deprecated} == {"acquire_dst", "release_dst"}
    assert len(deprecated) >= 4, "\n".join(str(f) for f in findings)


def _find_call(cursor, name):
    if cursor.kind == ci.CursorKind.CALL_EXPR and cursor.referenced is not None:
        if cursor.referenced.spelling == name:
            return cursor
    for child in cursor.get_children():
        found = _find_call(child, name)
        if found is not None:
            return found
    return None


def test_sequence_rule_reports_unseated_packer(checker):
    """A tilize_init that does not freshly init the packer is a wrong-init (#52175)."""
    kernel = FIXTURES / "ok_tilize.cpp"
    table = checker.RuleTable.load("wormhole_b0")
    args = checker.synth_args(str(kernel), "wormhole_b0", "PACK", _sfpi())
    analyzer = checker.Analyzer(args, "PACK", "wormhole_b0", table, str(kernel))
    call = _find_call(analyzer.tu.cursor, "tilize_init")
    assert call is not None, "fixture does not call tilize_init"
    analyzer._check_sequence_rules(table.sequence_rules["tilize_init"], call, checker.State(), 0)
    assert any(f.rule == "wrong-init" for f in analyzer.findings)


def test_helper_local_constexpr_reduce_path_has_no_false_positive(checker):
    """groupnorm_sharded_v2 selects the FPU reduce through a helper-local constexpr."""
    kernel = (
        REPO
        / "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/compute/groupnorm_sharded_v2.cpp"
    )
    if not kernel.is_file():
        pytest.skip("groupnorm kernel is not present")
    for arch in ("wormhole_b0", "blackhole"):
        findings = analyze(checker, kernel, arch, "MATH")
        wrong = [f for f in findings if f.rule == "wrong-init"]
        assert not wrong, "\n".join(str(f) for f in wrong)


def test_every_finding_names_the_earlier_call(checker):
    for f in analyze(checker, FIXTURES / "mismatch_math_init.cpp", "wormhole_b0", "MATH"):
        assert f.earlier or f.rule in (
            "missing-init",
            "deprecated-init",
            "deprecated-api",
            "loop-invariant-init",
        )
        assert f.line > 0 and f.file


def test_all_rules_have_thread_and_citation(checker):
    raw = yaml.safe_load(RULES.read_text())
    valid_threads = {"UNPACK", "MATH", "PACK"}
    for entry in raw["operations"]:
        assert entry["thread"] in valid_threads, entry["id"]
        assert entry.get("citation"), entry["id"]
        assert entry.get("exu") in {"Unpack", "Fpu", "Sfpu", "Pack"}, entry["id"]
    for entry in raw.get("sfpu_extra_groups", []):
        assert entry.get("citation"), entry["id"]
    for entry in raw.get("sequence_rules", []):
        assert entry["thread"] in valid_threads, entry["id"]
        assert entry.get("citation"), entry["id"]
        assert entry["require_init"] in {op["id"] for op in raw["operations"]}, entry["id"]


def test_sfpu_yaml_is_current():
    spec = importlib.util.spec_from_file_location(
        "gen_compute_api_order_sfpu", INFRA / "gen_compute_api_order_sfpu.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["gen_compute_api_order_sfpu"] = module
    spec.loader.exec_module(module)
    assert SFPU_RULES.read_text() == module.build_yaml(), (
        "compute_api_order_sfpu.yaml is stale; run "
        "python3 tt_metal/tt-llk/infra/gen_compute_api_order_sfpu.py"
    )


def _hook_model(arch: str) -> dict[str, dict[str, set[str]]]:
    """Parse SAN_HOOKs: operation -> {kind: {function names}, fields: ...}."""
    root = REPO / "tt_metal" / "hw" / "ckernels" / arch / "metal" / "llk_api"
    model: dict[str, dict[str, set[str]]] = {}
    hook_re = re.compile(r"SAN_HOOK\(\s*(init|execute|uninit|configure|reconfigure)\s*<\s*([A-Za-z0-9_]+)\s*>")
    for path in root.rglob("*.h"):
        text = path.read_text(errors="replace")
        for m in hook_re.finditer(text):
            kind, op = m.groups()
            # Walk back to the enclosing function name.
            head = text[: m.start()]
            idx = head.rfind("inline")
            seg = head[idx:] if idx >= 0 else head[-3000:]
            fm = re.search(r"inline\s+[^\n{;]*?\b([A-Za-z_]\w*)\s*\(", seg)
            fn = fm.group(1) if fm else ""
            entry = model.setdefault(op, {"init": set(), "execute": set(), "uninit": set(), "fields": set()})
            if kind in ("init", "execute", "uninit"):
                entry[kind].add(fn)
            body = text[m.start() : text.find(";", m.start())]
            entry["fields"].update(re.findall(r"StateVal<\s*(?:[A-Za-z0-9_]+::)*([A-Za-z0-9_]+)\s*>", body))
    return model


def test_sanitizer_operation_model_agrees(checker):
    """The table and the LLK Sanitizer must cover the same operations/functions."""
    raw = yaml.safe_load(RULES.read_text())
    by_id = {e["id"]: e for e in raw["operations"]}
    for arch in ("wormhole_b0", "blackhole"):
        model = _hook_model(arch)
        table = checker.RuleTable.load(arch)
        covered = set()
        for op in table.operations.values():
            entry = by_id.get(op.id)
            if entry is None:
                continue
            sanitizer_op = op.sanitizer_operation
            if not sanitizer_op:
                continue
            assert sanitizer_op in model, f"{arch}/{op.id}: {sanitizer_op} missing from SAN_HOOKs"
            hook = model[sanitizer_op]
            seen = {"init": set(), "execute": set(), "uninit": set()}
            for tr in op.transitions:
                seen["init"].add(tr.init_fn)
                seen["execute"].update(tr.execute_fns)
                seen["uninit"].update(tr.uninit_fns)
            for fn, evs in table.extra_events.items():
                for ev in evs:
                    if ev.op_id == op.id:
                        seen[ev.kind].add(fn)
            for kind in ("init", "execute", "uninit"):
                missing = hook[kind] - seen[kind]
                assert not missing, (
                    f"{arch}/{sanitizer_op}: table for {op.id} does not cover "
                    f"{kind} hook(s) {sorted(missing)}"
                )
                extra = seen[kind] - hook[kind]
                assert not extra, (
                    f"{arch}/{sanitizer_op}: table for {op.id} names {kind} "
                    f"function(s) {sorted(extra)} that the sanitizer does not hook"
                )
            covered.add(sanitizer_op)
        # Every sanitizer operation with init+execute hooks on this arch is
        # covered by some table entry or explicitly exempt (sanitizer_operation
        # null: SFPU, unpack-untilize, transpose_dest).
        for sanitizer_op, hooks in model.items():
            if not hooks["init"] or not hooks["execute"]:
                continue
            assert sanitizer_op in covered, f"{arch}: {sanitizer_op} is hooked but not in the rule table"


def test_deprecated_ids_exist_in_deprecations_json():
    raw = yaml.safe_load(RULES.read_text())
    deprecations = json.loads((REPO / ".github" / "deprecations.json").read_text())
    known = {d["id"] for d in deprecations.get("deprecations", [])}
    for dep in raw.get("deprecated_init_apis", []):
        assert dep["id"] in known, dep["id"]


def test_cli_compile_db(checker, tmp_path):
    sfpi = _sfpi()
    fixture = FIXTURES / "missing_binary_init.cpp"
    args = checker.synth_args(str(fixture), "wormhole_b0", "MATH", sfpi)
    db = tmp_path / "compile_commands.json"
    db.write_text(
        json.dumps([{"directory": str(REPO), "file": str(fixture), "arguments": args}])
    )
    out = tmp_path / "report.json"
    # Exercise the CLI argument path in-process: no OS command is constructed
    # from the compile database or any other external input.
    rc = checker.main(
        [
            "--compile-db",
            str(db),
            "--format",
            "json",
            "--output",
            str(out),
            "--sfpi",
            str(sfpi),
        ]
    )
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["summary"]["by_rule"] == {"missing-init": 1}
    assert payload["units"] and payload["units"][0]["thread"] == "MATH"


def test_cli_log_entry_translation(checker, tmp_path):
    """A captured SFPI g++ line becomes a clang analysis command."""
    sfpi = _sfpi()
    compiler = sfpi / "compiler" / "bin" / "riscv-tt-elf-g++"
    if not compiler.is_file():
        pytest.skip("SFPI g++ not present")
    fixture = FIXTURES / "missing_sfpu_init.cpp"
    include_flags = []
    for d in checker._arch_include_flags("wormhole_b0"):
        include_flags.append(d)
    argv = [
        str(compiler),
        "-mcpu=tt-wh-tensix",
        "-std=c++17",
        "-DTENSIX_FIRMWARE",
        "-DTRISC_MATH=1",
        "-DUCK_CHLKC_MATH",
        "-DCOMPILE_FOR_TRISC=1",
        *include_flags,
        "-c",
        "-o",
        str(tmp_path / "out.o"),
        str(fixture),
    ]
    log = tmp_path / "build.log"
    log.write_text("    g++ compile cmd: " + " ".join(argv) + " (build.cpp:1)\n")
    entries = checker.entries_from_log(log)
    assert entries, "log entry was not translated"
    assert any("-include" in " ".join(e["arguments"]) for e in entries) or entries[0]["file"].endswith(
        fixture.name
    )


def test_corpus_report_separates_advisory_warnings(checker):
    """Warning-only rules must not dilute the error table's total or FP rate."""
    spec = importlib.util.spec_from_file_location(
        "run_compute_api_order_corpus", INFRA / "run_compute_api_order_corpus.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    findings = [
        checker.Finding(
            rule="missing-init",
            severity="error",
            file="/repo/kernel.cpp",
            line=1,
            column=1,
            thread="MATH",
            arch="wormhole_b0",
            message="missing init",
        ),
        checker.Finding(
            rule="loop-invariant-init",
            severity="warning",
            file="/repo/kernel.cpp",
            line=2,
            column=1,
            thread="MATH",
            arch="wormhole_b0",
            message="loop init",
        ),
    ]
    args = argparse.Namespace(
        threads=["MATH"],
        discover=True,
        roots=["ttnn"],
        kernel_files=1,
        from_json=None,
        compile_db=None,
        log=None,
    )
    triage = {
        "missing-init": {"true_positive": True},
        "loop-invariant-init": {"true_positive": False},
    }
    report = runner.render_report(
        ["wormhole_b0"],
        args,
        dt.datetime.now(dt.timezone.utc),
        findings,
        [],
        [{"file": "/repo/kernel.cpp"}],
        triage,
        1.0,
    )
    assert "## Error findings by rule" in report
    assert "| **Total** | **1** | **1** | **0** | **0** |" in report
    assert "## Advisory warnings (excluded from error totals)" in report
    assert "| `loop-invariant-init` | 1 | 0 | 1 | 0 |" in report


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
