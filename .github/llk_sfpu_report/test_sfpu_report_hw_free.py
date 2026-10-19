# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Hardware-free tests of the LLK SFPU report (``.github/llk_sfpu_report``).

Everything here runs on the host: which changed files count as device code, which
ops a run measures first, how special-input results are compared, and what the
rendered comment says for a synthetic run.
"""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import accuracy  # noqa: E402
import cli  # noqa: E402
import detect  # noqa: E402
import overlay  # noqa: E402
import perf  # noqa: E402
import report  # noqa: E402
import runner  # noqa: E402

NAN = float("nan")
INF = float("inf")


@pytest.mark.parametrize(
    "path, device",
    [
        (
            "tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_recip.h",
            True,
        ),
        ("tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_log.h", True),
        ("tt_metal/tt-llk/tests/helpers/include/sfpu_operations.h", True),
        ("tt_metal/tt-llk/tests/sources/eltwise_unary_sfpu_test.cpp", True),
        # Host-side code never comes from the PR.
        ("tt_metal/tt-llk/tests/python_tests/helpers/golden_generators.py", False),
        ("tt_metal/tt-llk/tests/python_tests/conftest.py", False),
        (".github/workflows/llk-sfpu-report.yaml", False),
        ("tests/ttnn/unit_tests/operations/eltwise/test_unary.py", False),
    ],
)
def test_only_device_code_comes_from_the_pr(path, device):
    assert overlay.is_device_file(path) is device


WH, BH = "wormhole", "blackhole"


@pytest.mark.parametrize(
    "paths, archs",
    [
        (["tt_metal/tt-llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_exp.h"], [WH]),
        (["tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_recip.h"], [WH]),
        (["tt_metal/hw/inc/internal/tt-1xx/wormhole/cfg_defines.h"], [WH]),
        (["tt_metal/hw/firmware/src/tt-1xx/blackhole/noc.c"], [BH]),
        (["tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_log.h"], [BH]),
        (["tt_metal/tt-llk/tt_llk_blackhole/a.h", "tt_metal/tt-llk/tt_llk_wormhole_b0/a.h"], [WH, BH]),
        # Device code in no one architecture's tree builds for both chips.
        (["tt_metal/tt-llk/common/ckernel_sfpu_welfords_common.h"], [WH, BH]),
        (["tt_metal/hw/inc/api/compute/eltwise_unary.h"], [WH, BH]),
        (["tt_metal/tt-llk/tests/helpers/include/sfpu_operations.h"], [WH, BH]),
        (["tt_metal/tt-llk/tests/sources/eltwise_unary_sfpu_test.cpp"], [WH, BH]),
        (["tt_metal/tt-llk/tt_llk_wormhole_b0/a.h", "tt_metal/tt-llk/common/b.h"], [WH, BH]),
        # Quasar is not measured, even under a shared directory.
        (["tt_metal/tt-llk/tt_llk_quasar/common/inc/sfpu/ckernel_sfpu_sqrt.h"], []),
        (["tt_metal/tt-llk/tests/sources/quasar/sfpu_test.cpp"], []),
        (["tt_metal/hw/inc/internal/tt-2xx/quasar/cfg_defines.h"], []),
        (["tt_metal/tt-llk/tt_llk_quasar/a.h", "tt_metal/tt-llk/tt_llk_blackhole/b.h"], [BH]),
        # Host code runs from main, so the PR's never reaches a chip.
        (["README.md", "tt_metal/tt-llk/tests/python_tests/test_eltwise_binary_sfpu.py"], []),
        ([], []),
    ],
)
def test_the_chips_run_are_those_whose_device_code_changed(paths, archs):
    assert overlay.archs_built(paths) == archs


def test_the_plan_job_reads_the_chips_from_the_classifier():
    paths = "tt_metal/tt-llk/tt_llk_blackhole/a.h\nREADME.md\n"
    script = Path(overlay.__file__)
    out = subprocess.run([sys.executable, str(script), "archs"], input=paths, capture_output=True, text=True)
    assert (out.returncode, out.stdout) == (0, "blackhole\n")


def test_ops_whose_own_kernel_changed_come_first():
    ops = ["Atan", "Erf", "Gelu", "Reciprocal", "Sigmoid"]
    changed = ["tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_recip.h"]
    assert cli._prioritize(ops, changed) == [
        "Reciprocal",
        "Atan",
        "Erf",
        "Gelu",
        "Sigmoid",
    ]


def _spec(values):
    """{bits: (class, input, result)} as accuracy._specials returns it."""
    return {i: v for i, v in enumerate(values)}


def test_specials_diff_reports_changes_and_nan_propagation():
    base = _spec([("nan", NAN, NAN), ("inf", -INF, -1.0), ("zero", -0.0, -0.0)])
    head = _spec([("nan", NAN, 0.0), ("inf", -INF, NAN), ("zero", -0.0, 0.0)])
    diff = accuracy.specials_diff(base, head)
    assert [(c["class"], c["new"] != c["new"]) for c in diff["changed"]] == [
        ("nan", False),
        ("inf", True),
        ("zero", False),
    ]
    # -0 -> +0 is a change: the sign of a zero is part of the result.
    assert any(c["class"] == "zero" for c in diff["changed"])
    assert diff["nan_propagates"] == {"base": True, "head": False}


def test_specials_diff_ignores_nan_payloads():
    base = _spec([("nan", NAN, NAN)])
    head = _spec([("nan", NAN, -NAN)])
    assert accuracy.specials_diff(base, head)["changed"] == []


def _summary():
    perf_row = {
        "op": "Square",
        "formats": "Float16_b->Float16_b",
        "dest_acc": "No",
        "approx": "No",
        "fast_mode": "No",
        "text_base": 2447,
        "text_head": 2519,
        "MATH_ISOLATE": {
            "base": 191.0,
            "head": 479.0,
            "delta": 1.5,
            "regression": True,
            "improvement": False,
        },
        "L1_TO_L1": {
            "base": 207.0,
            "head": 495.0,
            "delta": 1.4,
            "regression": True,
            "improvement": False,
        },
    }
    stats = lambda mx, mean: {  # noqa: E731
        "lanes": 65279,
        "max": mx,
        "mean": mean,
        "p99": 1.0,
        "exact": 0.5,
        "le1": 0.99,
        "nonfinite": 0,
        "nonfinite_examples": [],
        "worst_input": 1.0,
    }
    return {
        "arch": "wormhole",
        "host": "host",
        "host_board": "n150",
        "run_url": None,
        "mode": "merge-base",
        "tool_sha": "a" * 40,
        "base_sha": "b" * 40,
        "head_sha": "c" * 40,
        "head_moved_to": None,
        "merge_base_age_days": 30,
        "applied": [],
        "not_applied": [],
        "ops": {
            "measured": ["Square"],
            "not_covered": [],
            "why": "requested in the command",
        },
        "iterations": 3,
        "thresholds": cli.THRESHOLDS,
        "perf": {"unary": {"loadmacro": [perf_row], "no-loadmacro": [dict(perf_row)]}},
        "accuracy": [
            {
                "key": ["Square", "Float16", "Float16", "No", "No"],
                "base": stats(1, 0.3),
                "head": stats(1024, 1.4),
                "worse": 22920,
                "better": 2010,
                "bit_identical": False,
                "head_digest": "x",
                "specials": accuracy.specials_diff(_spec([("inf", INF, INF)]), _spec([("inf", INF, NAN)])),
            }
        ],
        "notes": [],
        "commands": ["python3 cli.py ..."],
    }


def test_report_flags_the_regressions():
    text = report.render([_summary()])
    assert text.startswith(report.COMMENT_MARKER)
    assert "⚠️ 479" in text  # perf
    assert "⚠️ 1,024" in text  # accuracy
    assert "⚠️ `nan`" in text  # edge case changed kind
    assert "0.40x" in text
    assert "merge-base is 30 days old" in text


def test_report_stays_under_the_comment_limit():
    summary = _summary()
    rec = summary["accuracy"][0]
    summary["accuracy"] = [dict(rec, key=[f"Op{i}", *rec["key"][1:]]) for i in range(600)]
    text = report.render([summary])
    assert len(text) < 65536 and "cut to fit a comment" in text
    assert "git fetch origin" in text and "--ops Square --check" in text  # Reproduce is never folded


def test_binary_ops_are_their_own_family():
    assert cli._family_of("Tanh") == "unary"
    assert cli._family_of("Typecast") == "typecast"
    assert cli._family_of("SfpuLogsigmoid") == "binary"
    assert cli._family_of("SfpuDivInt32") == "binary"


@pytest.mark.parametrize("family", sorted(detect.MODULES))
def test_detection_compiles_modules_and_kernels_that_exist(family):
    module = runner.PYTHON_TESTS / detect.MODULES[family]
    assert module.is_file()
    for source in detect.SOURCES[family]:
        assert (runner.TOOL_LLK / "tests" / "sources" / source).is_file()
        assert (
            f"sources/{source}"
            in module.read_text() + (runner.PYTHON_TESTS / "test_eltwise_binary_sfpu.py").read_text()
        )


def test_the_accuracy_driver_is_in_the_harness_only_while_it_runs(monkeypatch, tmp_path):
    installed = runner.PYTHON_TESTS / accuracy.DRIVER
    assert accuracy.DRIVER_SOURCE.is_file() and not installed.exists()
    stale = tmp_path / "out" / "Tanh__Float16_b-Float16_b__No__No__sweep.pt"
    stale.parent.mkdir()
    stale.write_text("an earlier run's result")
    seen = []
    monkeypatch.setattr(runner, "produce_consume", lambda *args, **kwargs: seen.append(installed.is_file()))
    accuracy.measure(None, "wormhole", ["Tanh"], tmp_path / "out", log=None)
    assert seen == [True]
    assert not installed.exists()
    assert not stale.exists()  # compare() would read it as this run's


def test_the_binary_accuracy_driver_hooks_names_that_exist():
    """The driver replaces functions of test_eltwise_binary_sfpu.py by name; a rename
    there (#57137 dropped ``_assert_against_contract``) would otherwise only show up
    on hardware, as an accuracy run with no results."""
    import ast

    def module_names(path):
        names = set()
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                names.update((a.asname or a.name).split(".")[0] for a in node.names)
            elif isinstance(node, ast.Assign):
                names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        return names

    hooked = {
        call.args[1].value
        for call in ast.walk(ast.parse(accuracy.DRIVER_SOURCE.read_text()))
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "setattr"
        and len(call.args) >= 2
        and isinstance(call.args[1], ast.Constant)
    }
    assert hooked == {"assert_against_contract", "generate_stimuli"}
    # ... and calls the functional test's stimuli builders by name.
    builders = {
        node.value
        for node in ast.walk(ast.parse(accuracy.DRIVER_SOURCE.read_text()))
        if isinstance(node, ast.Constant) and str(node.value).endswith("_stimuli_specs")
    }
    assert builders
    assert hooked | builders <= module_names(runner.PYTHON_TESTS / "test_eltwise_binary_sfpu.py")


@pytest.mark.parametrize("exit_code, builds", [(0, True), (5, True), (1, False)])
def test_the_skew_probe_reads_the_compile_exit_code(monkeypatch, exit_code, builds):
    monkeypatch.setattr(detect, "compile_all", lambda *args, **kwargs: exit_code)
    assert cli._harness_compiles(None, "wormhole", None, 1) is builds


@pytest.mark.parametrize(
    "builds, mode, built", [(True, "merge-base", ["merge-base"]), (False, "rebase", ["merge-base", "rebase"])]
)
def test_an_old_merge_base_falls_back_to_rebase_mode(monkeypatch, builds, mode, built):
    seen = []
    monkeypatch.setattr(cli, "_sides", lambda plan, work, m: (seen.append(m), ("base", "head"))[1])
    monkeypatch.setattr(cli, "_harness_compiles", lambda *args: builds)
    notes = []
    assert cli._choose_sides(None, None, "merge-base", "wormhole", None, 1, notes) == (mode, "base", "head")
    assert seen == built
    assert bool(notes) is not builds


#: What ``git apply --3way`` printed on #57421, whose square kernels main rewrote.
CONFLICT = """Applied patch to 'tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_square.h' with conflicts.
Applied patch to 'tt_metal/tt-llk/tests/helpers/include/sfpu_operations.h' cleanly.
U tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_square.h
U tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_square.h"""
SQUARE = [
    "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_square.h",
    "tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_square.h",
]
PLAN = SimpleNamespace(tool_sha="c" * 40, base_sha="b" * 40, head_sha="a" * 40, applied=SQUARE, not_applied=[])


@pytest.fixture
def one_commit_repo(tmp_path):
    def git(*args):
        return subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True, text=True).stdout

    git("init", "-q")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "c")
    return tmp_path, git("rev-parse", "HEAD").strip()


@pytest.mark.parametrize("ref", ["--output=/tmp/x", "-p", "HEAD --output=/tmp/x", "no-such-ref", ""])
def test_a_ref_that_is_not_a_commit_never_reaches_git_as_an_option(one_commit_repo, ref):
    repo, _ = one_commit_repo
    with pytest.raises(SystemExit):
        overlay.commit_sha(repo, ref)


def test_refs_resolve_to_full_shas(one_commit_repo):
    repo, sha = one_commit_repo
    assert overlay.commit_sha(repo, "HEAD") == sha
    assert overlay.commit_sha(repo, sha[:10]) == sha


def test_a_patch_conflict_names_the_conflicted_files():
    assert overlay.PatchConflict(CONFLICT).files == SQUARE


@pytest.mark.parametrize("mode, cause", [("rebase", "This PR's"), ("merge-base", "does not build")])
def test_a_pr_that_conflicts_with_main_is_not_measured(monkeypatch, mode, cause):
    def sides(plan, work, m):
        if m == "rebase":
            raise overlay.PatchConflict(CONFLICT)
        return "base", "head"

    monkeypatch.setattr(cli, "_sides", sides)
    monkeypatch.setattr(cli, "_harness_compiles", lambda *args: False)
    notes = []
    with pytest.raises(cli.NotMeasured) as raised:
        cli._choose_sides(PLAN, None, mode, "wormhole", None, 1, notes)
    assert cause in raised.value.reason
    assert "`cccccccccc`" in raised.value.reason and "Rebase the PR onto main" in raised.value.reason
    assert raised.value.files == SQUARE
    assert notes == []  # no "this report compares main ..." note for a report that compares nothing


def test_a_run_that_cannot_measure_writes_a_report_that_says_why(monkeypatch, tmp_path):
    monkeypatch.setattr(overlay, "make_plan", lambda *args: PLAN)
    monkeypatch.setattr(cli, "_board", lambda: "n150")
    monkeypatch.setattr(cli, "_merge_base_age", lambda plan: 18)

    def conflict(*args):
        raise cli.NotMeasured("This PR's kernel changes do not apply cleanly to current main.", SQUARE)

    monkeypatch.setattr(cli, "_choose_sides", conflict)
    with pytest.raises(SystemExit) as exited:
        cli.main(["--arch", "wormhole", "--head", "x", "--work", str(tmp_path), "run", "--pr", "57421", "--check"])
    assert exited.value.code == 2  # --check: not a pass
    summary = json.loads((tmp_path / "summary-wormhole.json").read_text())
    assert summary["conflicts"] == SQUARE and summary["ops"]["measured"] == []
    text = (tmp_path / "report-wormhole.md").read_text()
    assert text.startswith(report.COMMENT_MARKER)
    assert "**Not measured** on Wormhole. This PR's kernel changes do not apply" in text
    assert f"- `{SQUARE[1]}`" in text
    assert "No regressions" not in text


def _nothing_measured(arch, *notes):
    return dict(
        _summary(),
        arch=arch,
        ops={"measured": [], "not_covered": [], "why": "auto"},
        perf={},
        accuracy=[],
        detection={},
        notes=list(notes),
    )


def test_a_run_that_measured_nothing_says_why_without_empty_tables():
    fell_back = "The merge-base does not build with today's test harness, so ..."
    text = report.render(
        [
            _nothing_measured("wormhole", fell_back, "This PR changes SFPU kernels of other architectures only."),
            _nothing_measured("blackhole", fell_back, "The changed kernels may be ternary or structural SFPU ops."),
        ]
    )
    assert "**Ops measured:** none. No op this report covers compiles to different code" in text
    assert "No regressions" not in text and "### Performance" not in text and "### Accuracy" not in text
    assert f"- {fell_back}" in text  # both archs wrote it: once, unprefixed
    assert "- Wormhole: This PR changes SFPU kernels of other architectures only." in text
    assert "- Blackhole: The changed kernels may be ternary or structural SFPU ops." in text


def test_a_blackhole_only_change_lists_what_blackhole_measured():
    blackhole = dict(_summary(), arch="blackhole")
    text = report.render([_nothing_measured("wormhole"), blackhole])
    assert "**Ops measured:** `Square`" in text
    assert "⚠️ 479" in text


def test_one_ci_run_is_linked_once():
    wormhole = dict(_summary(), run_url="https://github.com/tenstorrent/tt-metal/actions/runs/1")
    blackhole = dict(wormhole, arch="blackhole")
    assert report.render([wormhole, blackhole]).count("actions/runs/1") == 1


def test_an_arch_that_was_not_measured_is_a_note_beside_the_one_that_was():
    blackhole = dict(_summary(), arch="blackhole", not_measured="This PR's kernel changes do not apply.")
    text = report.render([_summary(), blackhole])
    assert "on Wormhole (" in text and "Blackhole (" not in text
    assert "- Blackhole was not measured. This PR's kernel changes do not apply." in text


def test_rebase_mode_says_what_it_compared():
    summary = _summary()
    summary["mode"] = "rebase"
    text = report.render([summary])
    assert "on top of main (`aaaaaaaaaa`) against main itself" in text
    assert "(merge-base)" not in text
    assert "days old" not in text  # the stale merge-base warning is about merge-base mode
    assert "--mode rebase" in text


def test_requested_ops_take_any_case_and_skip_unknown_names():
    ops, unknown = cli._requested_ops("tanh,SFPULOGSIGMOID,typecast,tanhh,Tanh")
    assert ops == ["Tanh", "SfpuLogsigmoid", "Typecast"]
    assert unknown == ["tanhh"]


def _perf_csv(path, variants):
    """A raw perf CSV as the harness writes it: a TILE_LOOP row per (op, fast_mode, cycles)."""
    import pandas as pd

    common = {
        "marker": "TILE_LOOP",
        "formats.input_A": "Float16_b",
        "formats.output": "Float16_b",
        "dest_acc": "DestAccumulation.No",
        "approx_mode": "ApproximationMode.No",
        "tile_cnt": 4,
        "loop_factor": 16,
    }
    rows = [
        dict(
            common,
            mathop=f"MathOperation.{op}",
            fast_mode=f"FastMode.{fast}",
            **{"mean(L1_TO_L1)": cycles + 1000, "mean(MATH_ISOLATE)": cycles, "TEXT_SIZE(MATH_ISOLATE)": 2048},
        )
        for op, fast, cycles in variants
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _perf_runs(tmp_path, base, head):
    """``perf.sweep``'s result, one CSV per run and side: ``{"loadmacro": {side: [csv...]}}``."""
    return {
        "loadmacro": {
            side: [
                _perf_csv(tmp_path / side / f"run_{i}" / "perf_eltwise_unary_sfpu.csv", variants)
                for i, variants in enumerate(runs, start=1)
            ]
            for side, runs in (("base", base), ("head", head))
        }
    }


def test_perf_runs_pool_into_one_median_per_variant(monkeypatch, tmp_path):
    """Through sweep() and measure(), with the device run faked: the head side's third
    run is slow, and the median of three outvotes it."""
    head_runs = iter([10_000, 10_000, 30_000])

    def device_run(side, arch, args, **kwargs):
        if "--compile-consumer" in args:
            stem = Path(detect.MODULES["unary"]).stem
            cycles = next(head_runs) if side.name == "head" else 10_000
            _perf_csv(side.llk / "perf_data" / "latest" / stem / f"{stem}.csv", [("Exp", "No", cycles)])
        return 0

    monkeypatch.setattr(runner, "pytest", device_run)
    base, head = (runner.Side(n, n, tmp_path / n, tmp_path / f"build-{n}") for n in ("base", "head"))
    runs = perf.sweep(base, head, "wormhole", "unary", ["Exp"], tmp_path / "perf", None, schedules=["loadmacro"])
    verdicts = perf.compare(runs, cli.THRESHOLDS["wormhole"], cli.THRESHOLDS["min_cycles"])
    assert len([r for r in verdicts["loadmacro"]["records"] if r["run_type"] == "MATH_ISOLATE"]) == 1
    (row,) = perf.rows(runs, verdicts)["loadmacro"]
    assert row["MATH_ISOLATE"]["head"] == row["MATH_ISOLATE"]["base"]  # the slow third run is outvoted
    assert not row["MATH_ISOLATE"]["regression"]
    assert (row["text_base"], row["text_head"]) == (2048, 2048)


def test_a_variant_that_stops_running_on_the_head_side_is_a_regression(tmp_path):
    both = ("Exp", "No", 10_000)
    runs = _perf_runs(tmp_path, base=[[both, ("Exp", "Yes", 8_000)]], head=[[both]])
    rows = perf.rows(runs, perf.compare(runs, 0.08, 30))["loadmacro"]
    assert {(r["op"], r.get("missing")) for r in rows} == {("Exp", None), ("Exp (fast_mode Yes)", "head")}
    summary = dict(_summary(), perf={"unary": {"loadmacro": rows}}, accuracy=[])
    text = report.render([summary])
    assert "perf: Wormhole `Exp (fast_mode Yes)` bf16 dest_acc=No approx=No: ran on base, no result on head" in text
    assert "| ⚠️ no result |" in text
    assert "Measured for perf, not for accuracy: `Exp` fast mode." in text


def test_variants_are_labelled_by_the_switches_they_have_on():
    common = dict(
        mathop="SfpuElwadd",
        **{"formats.input_A": "Float16_b", "formats.output": "Float16_b"},
        dest_acc="No",
        approx_mode="No",
        broadcast_type="None_",
        sfpu_bcast_dim="None_",
    )
    plain = dict(common, tile_cnt="32")
    unpack = dict(plain, broadcast_type="Row")
    dest = dict(common, sfpu_bcast_dim="Row", tile_cnt="3")  # test_perf_eltwise_binary_sfpu_bcast
    bigger = dict(plain, tile_cnt="64")  # reads like `plain`: named by what differs
    idents = [tuple(sorted(c.items())) for c in (plain, unpack, dest, bigger)]
    labels = perf._labels(idents)
    assert [labels[i] for i in idents] == [
        "SfpuElwadd (tile_cnt 32)",
        "SfpuElwadd (unpack bcast Row)",
        "SfpuElwadd (Dest bcast Row)",
        "SfpuElwadd (tile_cnt 64)",
    ]


def test_a_csv_cell_and_the_comparers_key_are_one_variant():
    import numpy as np

    assert perf._identity([("loop_factor", np.float64(16.0)), ("fast_mode", "FastMode.Yes")]) == perf._identity(
        [("loop_factor", 16), ("fast_mode", "Yes"), ("test_module", "run_1"), ("stable_sort", NAN)]
    )


#: Perf test ids as the harness builds them (helpers/param_config.build_param_id).
PERF_IDS = {
    "float_add_bf16": "formats:Float16_b->Float16_b-dest_acc:No-mathop:SfpuElwadd-bcast_dim:None_",
    "float_add_fp32": "formats:Float32->Float32-dest_acc:Yes-mathop:SfpuElwadd-bcast_dim:None_",
    "float_sub_bf16": "formats:Float16_b->Float16_b-dest_acc:No-mathop:SfpuElwsub-bcast_dim:None_",
    "gcd": "mathop:SfpuGcd-dest_acc:Yes",
}


def _selected(tmp_path, family, ops, formats):
    """The test ids ``perf._args`` keeps, by pytest's own ``-k`` matching."""
    (tmp_path / "test_ids.py").write_text(
        "import pytest\n"
        f"@pytest.mark.parametrize('case', {sorted(PERF_IDS.values())!r}, ids=str)\n"
        "def test_perf_eltwise_binary_sfpu_float(case): pass\n"
        "def test_perf_eltwise_binary_sfpu_div(): pass\n"
    )
    args = perf._args(family, ops, "loadmacro", formats)
    k = args[args.index("-k") + 1] if "-k" in args else ""
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--co",
            "-q",
            "-p",
            "no:cacheprovider",
            "-o",
            "addopts=",
            "-k",
            k,
            "test_ids.py",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    ).stdout
    found = [line.split("::", 1)[1] for line in out.splitlines() if "::" in line]
    return {name for name, case in PERF_IDS.items() if f"[{case}]" in "".join(found)} | (
        {"div"} if "test_perf_eltwise_binary_sfpu_div" in found else set()
    )


@pytest.mark.parametrize(
    "ops, formats, selected",
    [
        (["SfpuElwadd"], [], {"float_add_bf16", "float_add_fp32"}),
        (["SfpuElwdiv"], [], {"div"}),  # no mathop parameter: picked by its test name
        (["SfpuGcd"], ["Int32"], {"gcd"}),  # no formats parameter: runs in its only format
        (["SfpuElwadd", "SfpuElwdiv", "SfpuGcd"], ["Float16_b"], {"float_add_bf16", "div", "gcd"}),
    ],
)
def test_perf_selects_exactly_the_requested_ops(tmp_path, ops, formats, selected):
    assert _selected(tmp_path, "binary", ops, formats) == selected


def test_typecast_perf_is_not_narrowed_by_op():
    assert "-k" not in perf._args("typecast", ["Typecast"], "loadmacro")


def test_reproduce_commands_drop_the_broadcast_label():
    summary = _summary()
    for rows in summary["perf"]["unary"].values():
        for r in rows:
            r["op"] = "Square (bcast Row)"
    text = report.render([summary])
    assert "| Square (bcast Row) |" in text
    assert "--ops Square --formats Float16_b,Float16 --check" in text


def test_sfpu_types_map_to_math_operations_by_enum_value():
    # Spelled differently (SfpuType::exponential is MathOperation.Exp), or the same.
    assert cli._sfpu_type_to_op(["exponential", "sine", "negative", "tanh", "no_such_type"]) == (
        ["Exp", "Sin", "Neg", "Tanh"],
        ["no_such_type"],
    )


def test_ops_built_on_one_side_only_are_measured():
    found = {
        "unary": {"changed": ["tanh"], "unchanged": ["sine"], "unmatched": ["exponential"]},
        "binary": {"changed": [], "unchanged": [], "unmatched": ["DIV"]},
        "typecast": {"changed": [], "unchanged": ["typecast"], "unmatched": []},
    }
    ops, not_covered, one_sided = cli._detected_ops(found, [])
    assert ops == ["Tanh", "Exp", "SfpuElwdiv"]
    assert not_covered == [] and one_sided == ["Exp", "SfpuElwdiv"]


def test_binary_enum_names_map_to_math_operations():
    names = cli._binary_enum_to_op()
    assert names["LOGSIGMOID"] == "SfpuLogsigmoid"
    assert names["ATAN2"] == "SfpuAtan2"
    assert names["DIV_INT32"] == "SfpuDivInt32"


def test_exact_ops_count_wrong_lanes():
    import torch

    d = {
        "src": torch.tensor([1.0, 2.0, 3.0, 4.0]),
        "src_b": torch.tensor([2.0, 2.0, 2.0, NAN]),
        "golden": torch.tensor([1.0, 0.0, 0.0, 0.0]),
        "result": torch.tensor([1.0, 1.0, 0.0, 0.0]),
    }
    stats = accuracy._exact_stats(d)
    assert (stats["lanes"], stats["wrong"]) == (4, 1)
    assert stats["wrong_examples"][0][:2] == (2.0, 2.0)


def test_binary_specials_are_keyed_by_the_operand_pair():
    import torch

    d = {
        "binary": True,
        "src": torch.tensor([INF, INF, 1.0]),
        "src_b": torch.tensor([INF, -INF, 1.0]),
        "result": torch.tensor([INF, NAN, 2.0]),
    }
    spec = accuracy._specials(d)
    assert len(spec) == 3
    assert {v[1] for v in spec.values()} == {(INF, INF), (INF, -INF), (1.0, 1.0)}


def test_report_renders_exact_ops():
    summary = _summary()
    summary["accuracy"] = [
        {
            "key": ["SfpuElwLt", "Float16_b", "Float16_b", "No", "No"],
            "binary": True,
            "base": {
                "metric": "exact",
                "lanes": 16384,
                "wrong": 0,
                "wrong_examples": [],
            },
            "head": {
                "metric": "exact",
                "lanes": 16384,
                "wrong": 12,
                "wrong_examples": [],
            },
            "worse": 12,
            "better": 0,
            "bit_identical": False,
            "specials": {
                "changed": [{"class": "pair", "input": [INF, INF], "old": 0.0, "new": NAN}],
                "nan_propagates": {"base": True, "head": True},
            },
        }
    ]
    text = report.render([summary])
    assert "a lane is right or wrong" in text
    assert "⚠️ 12" in text
    assert "`(inf, inf)`" in text


def test_an_accuracy_result_missing_on_the_head_side_is_a_regression(tmp_path):
    import torch

    result = {"op": "Tanh", "in": "Float16_b", "out": "Float16_b", "approx": "No", "dest_acc": "No"}
    for side in ("base", "head"):
        (tmp_path / side).mkdir()
    torch.save(dict(result, kind="sweep"), tmp_path / "base" / "Tanh__Float16_b-Float16_b__No__No__sweep.pt")
    (rec,) = accuracy.compare(tmp_path / "base", tmp_path / "head")
    assert rec["missing"] == ["head sweep"]
    summary = dict(_summary(), perf={}, accuracy=[rec])
    assert [f["text"] for f in report.findings([summary])] == [
        "accuracy: Wormhole `Tanh` bf16 dest_acc=No approx=No: sweep: ran on base, no result on head "
        "(did not build or run; see run.log)"
    ]


def test_each_arch_reproduces_its_own_mode_and_ops():
    wormhole = dict(_nothing_measured("wormhole"), mode="rebase")
    text = report.render([wormhole, dict(_summary(), arch="blackhole")])
    assert (
        "main itself on Wormhole (n150), and `cccccccccc` (PR head) against `bbbbbbbbbb` (merge-base) on Blackhole"
        in text
    )
    assert (
        "--arch wormhole --head " + "c" * 40 + " --base " + "b" * 40 + " --mode rebase run --ops <op> --check" in text
    )
    assert "--arch blackhole --head " + "c" * 40 + " --base " + "b" * 40 + " run --ops Square --check" in text
    assert "mode `rebase` (Wormhole), `merge-base` (Blackhole)" in text


def test_file_names_from_the_pr_cannot_write_markdown():
    assert report.code("tt_metal/hw/ckernel_sfpu_exp.h") == "`tt_metal/hw/ckernel_sfpu_exp.h`"
    assert report.code("a`@tenstorrent/x\nb.h") == "`a??tenstorrent/x?b.h`"
    summary = dict(_summary(), not_applied=["tt_metal/tt-llk/tests/python_tests/x` @org/team `.py"])
    assert "@org" not in report.render([summary])


def test_an_arch_that_did_not_finish_is_named():
    url = "https://github.com/tenstorrent/tt-metal/actions/runs/7"
    clean = dict(_summary(), perf={}, accuracy=[])
    text = report.render([clean], expected=["wormhole", "blackhole"], run_url=url)
    assert "**No regressions on Wormhole.** Blackhole was not measured: see the notes." in text
    assert f"- Blackhole was not measured. Its measurement did not finish; see the [run]({url})." in text
    text = report.render([dict(_summary(), not_measured="Conflicts.")], expected=["wormhole", "blackhole"])
    assert "**Not measured** on Wormhole. Conflicts." in text
    assert "**Not measured** on Blackhole. Its measurement did not finish; see the run." in text


def test_a_chip_the_pr_does_not_build_for_is_named_as_skipped():
    text = report.render([_summary()], expected=["wormhole"], skipped=["blackhole"])
    assert "- Blackhole was not measured: this PR changes no device code Blackhole builds" in text
    assert "Its measurement did not finish" not in text  # skipped, not failed
    url = "https://github.com/tenstorrent/tt-metal/actions/runs/7"
    text = report.render([], run_url=url, skipped=["wormhole", "blackhole"])
    assert text.startswith(report.COMMENT_MARKER)
    assert "**Not measured.** This PR changes no device code Wormhole or Blackhole builds." in text
    assert f"Run: {url}." in text
    assert "### Performance" not in text and "No regressions" not in text
    # The chip that was run failed: both reasons are in the comment.
    text = report.render([], expected=["wormhole"], skipped=["blackhole"])
    assert "**Not measured** on Wormhole. Its measurement did not finish" in text
    assert "Blackhole was not measured: this PR changes no device code Blackhole builds" in text


def test_a_moved_device_file_leaves_no_copy_behind(tmp_path):
    def git(*args):
        return subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True, text=True).stdout

    old, new = "tt_metal/hw/inc/ckernel_sfpu_old.h", "tt_metal/hw/inc/sfpu/ckernel_sfpu_new.h"
    git("init", "-q")
    (tmp_path / old).parent.mkdir(parents=True)
    (tmp_path / old).write_text("".join(f"line {i}\n" for i in range(40)))
    commit = ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "c"]
    git("add", "-A")
    git(*commit)
    base = git("rev-parse", "HEAD").strip()
    (tmp_path / new).parent.mkdir(parents=True)
    git("mv", old, new)
    (tmp_path / new).write_text((tmp_path / new).read_text() + "edited\n")
    git("add", "-A")
    git(*commit)
    plan = overlay.make_plan(tmp_path, "HEAD", base)
    assert sorted(plan.applied) == sorted([old, new])
    assert b"deleted file mode" in overlay.device_diff(plan)


def test_findings_list_every_regression_once():
    found = report.findings([_summary()])
    kinds = sorted(f["kind"] for f in found)
    # The no-loadmacro twin of the perf row has the same code: listed once.
    assert kinds == ["accuracy", "edge", "perf"]
    text = report.render([_summary()])
    assert "**⚠️ 3 regression(s)**" in text
    assert "--ops Square --formats Float16_b,Float16 --check" in text


def test_a_special_result_that_changes_to_the_reference_is_not_a_regression():
    import torch

    # #54080: logsigmoid(+inf) went from -0 to +0, which is what torch returns.
    pairs = [(INF, 0.0), (3.4e38, 0.0), (NAN, 1.0), (1.0, 0.36)]
    golden = [0.0, 0.0, INF, -0.31]  # the harness maps a NaN input to inf

    def run(result):
        return accuracy._specials(
            {
                "binary": True,
                "in": "Float32",
                "src": torch.tensor([a for a, _ in pairs], dtype=torch.float64),
                "src_b": torch.tensor([b for _, b in pairs], dtype=torch.float64),
                "golden": torch.tensor(golden, dtype=torch.float64),
                "result": torch.tensor(result, dtype=torch.float64),
            }
        )

    summary = _summary()
    summary["accuracy"][0]["specials"] = accuracy.specials_diff(
        run([-0.0, -0.0, NAN, -0.31]), run([0.0, 0.0, 0.0, 0.31])
    )
    rows = {
        line.split("|")[6].strip(): line for line in report.edge_section([summary]) if line.startswith("| Wormhole")
    }
    assert rows["`(inf, 0)`"].endswith("| **`0`** |")  # to the reference's kind
    assert rows[f"`{report._val((3.4e38, 0.0))}`"].endswith("| **`0`** |")
    assert rows["`(nan, 1)`"].endswith("| ⚠️ `0` |")  # a NaN input has no IEEE reference
    assert rows["`(1, 0.36)`"].endswith("| ⚠️ `0.31` |")  # away from the reference
    assert "`f(nan, 1)`: `nan` → `0` (+1 more special input(s))" in report.render([summary])

    # -0 and subnormal inputs have no IEEE reference either.
    unary = accuracy._specials(
        {
            "in": "Float16_b",
            "src": torch.tensor([-0.0, 1e-40, 2.0], dtype=torch.float64),
            "classes": ["zero", "subnormal", "normal"],
            "golden": torch.tensor([0.0, 0.0, 4.0], dtype=torch.float64),
            "result": torch.tensor([0.0, 0.0, 4.0], dtype=torch.float64),
        }
    )
    assert [v[3] for v in unary.values()] == [None, None, 4.0]


def test_logsigmoid_perf_says_what_mains_cycles_leave_out():
    summary = _summary()
    for rows in summary["perf"]["unary"].values():
        rows[0]["op"] = "SfpuLogsigmoid"
    text = report.render([summary])
    assert "math 191 → 479 cycles/tile (0.40x); the old cycles leave out exp(-x), see Performance" in text
    assert "`SfpuLogsigmoid`: the test feeds exp(-x) as an input." in text
    assert "feeds" not in report.render([_summary()])


def test_a_clean_report_says_so():
    summary = _summary()
    summary["perf"] = {}
    rec = summary["accuracy"][0]
    rec["head"], rec["worse"], rec["better"] = dict(rec["base"]), 0, 0
    rec["specials"] = {"changed": [], "nan_propagates": {"base": True, "head": True}}
    assert report.findings([summary]) == []
    text = report.render([summary])
    assert "**No regressions.** Nothing is slower beyond the thresholds, and no accuracy or edge-case" in text
    assert "Not measured for" not in text


def test_an_op_without_accuracy_results_is_not_reported_as_unchanged():
    # #59869: no accuracy driver covers SfpuMask, and both sections read as measured.
    summary = _summary()
    summary["accuracy"] = []
    summary["notes"] = ["No accuracy driver covers `Square` (not an elementwise function of its inputs, ...)."]
    text = report.render([summary])
    assert text.count("Not measured for `Square`: see the notes.") == 2  # accuracy, edge cases
    assert "No accuracy change worth a row" not in text and "No special-input result changed" not in text
    summary["perf"] = {}
    assert (
        "**No regressions.** Nothing is slower beyond the thresholds. "
        "Accuracy was not measured for `Square`: see the notes." in report.render([summary])
    )


def test_ops_without_accuracy_results_are_named_next_to_the_measured_ones():
    wormhole = _summary()
    wormhole["ops"] = dict(wormhole["ops"], measured=["Square", "Typecast"])
    blackhole = dict(_summary(), arch="blackhole", accuracy=[])
    text = report.render([wormhole, blackhole])
    accuracy_part, edge_part = text.split("### Accuracy")[1].split("### Edge cases")
    for part in (accuracy_part, edge_part):
        assert "Not measured for `Typecast` (Wormhole), `Square` (Blackhole): see the notes." in part
    assert "| Wormhole | Square |" in accuracy_part and "| Wormhole | Square |" in edge_part


def test_a_variant_with_a_result_on_one_side_only_is_not_called_unmeasured():
    summary = _summary()
    summary["accuracy"] = [
        {"key": ["Square", "Float16", "Float16", "No", "No"], "missing": ["head sweep", "head specials"]}
    ]
    text = report.render([summary])
    assert "ran on base, no result on head" in text
    assert "Not measured for" not in text
    assert "No variant has accuracy results on both sides; see run.log." in text
    assert "No variant has special-input results on both sides; see run.log." in text


def test_a_small_drift_is_not_a_regression():
    rec = _summary()["accuracy"][0]
    rec["head"] = dict(rec["base"], max=rec["base"]["max"])
    rec["worse"], rec["better"] = 3, 1  # 2 net of 65,279 lanes
    rec["specials"] = {"changed": [], "nan_propagates": {"base": True, "head": True}}
    assert not report._acc_regressed(rec)
