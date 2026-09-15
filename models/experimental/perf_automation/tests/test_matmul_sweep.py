# SPDX-License-Identifier: Apache-2.0
"""matmul_sweep pre-pass: a SEPARATE, standalone step run before the optimize loop. These tests cover
the device-free logic — matmul enumeration from generic op-sigs, the bounded fidelity x dtype candidate
grid, the PCC-gated best pick, and the summary/speedup roll-up. The on-device sweep itself needs a
board and is exercised via run_matmul_sweep.sh, not here."""

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "matmul_sweep", str(Path(__file__).resolve().parents[1] / "cc_optimize" / "matmul_sweep.py")
)
MS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(MS)


def test_parse_matmul_sigs_extracts_shapes_and_dedups():
    sigs = [
        "ttnn.matmul(((1, 32, 2688), 'BFLOAT16'), ((2688, 2048), 'BFLOAT16'))",
        "ttnn.matmul(((1, 32, 2688), 'BFLOAT16'), ((2688, 2048), 'BFLOAT16'))",  # exact dup -> one entry
    ]
    mm = MS.parse_matmul_sigs(sigs)
    assert len(mm) == 1
    assert mm[0] == {"m": 32, "k": 2688, "n": 2048, "in_dtype": "BFLOAT16", "w_dtype": "BFLOAT16"}


def test_parse_handles_transposed_weight_and_2d():
    # a (N, K) stored/transposed weight (ttnn.linear) must still resolve N by matching the shared K dim
    sigs = ["ttnn.linear(((32, 2048), 'BFLOAT16'), ((8192, 2048), 'BFLOAT16'))"]
    mm = MS.parse_matmul_sigs(sigs)
    assert mm == [{"m": 32, "k": 2048, "n": 8192, "in_dtype": "BFLOAT16", "w_dtype": "BFLOAT16"}]


def test_parse_skips_non_matmul_and_malformed():
    sigs = [
        "ttnn.layer_norm(((1, 32, 2048), 'BFLOAT16'))",  # not a matmul
        "ttnn.matmul(garbled(((",  # unparseable
        "ttnn.matmul(((1, 32, 2048), 'BFLOAT16'))",  # only one operand -> can't form M,K,N
        42,  # not even a string
    ]
    assert MS.parse_matmul_sigs(sigs) == []


def test_candidate_grid_is_bounded_and_covers_top_two_knobs():
    cfgs = MS.candidate_configs(32, 2688, 2048)
    assert len(cfgs) == 6  # 3 fidelities x 2 dtypes
    assert {c["fidelity"] for c in cfgs} == {"LoFi", "HiFi2", "HiFi4"}
    assert {c["dtype"] for c in cfgs} == {"bfloat16", "bfloat8_b"}


def test_pick_best_is_fastest_that_passes_pcc():
    results = [
        {"fidelity": "HiFi4", "dtype": "bfloat16", "ms": 1.0, "pcc": 0.999},
        {"fidelity": "HiFi2", "dtype": "bfloat8_b", "ms": 0.4, "pcc": 0.995},
        {"fidelity": "LoFi", "dtype": "bfloat8_b", "ms": 0.2, "pcc": 0.80},  # fastest but FAILS pcc
        {"fidelity": "LoFi", "dtype": "bfloat16", "ms": None, "pcc": 0.0},  # crashed
    ]
    best = MS.pick_best(results, pcc_threshold=0.99)
    assert best["fidelity"] == "HiFi2" and best["dtype"] == "bfloat8_b" and best["ms"] == 0.4


def test_pick_best_none_when_nothing_passes():
    results = [
        {"fidelity": "LoFi", "dtype": "bfloat8_b", "ms": 0.2, "pcc": 0.80},
        {"fidelity": "HiFi4", "dtype": "bfloat16", "ms": None, "pcc": 0.0},
    ]
    assert MS.pick_best(results, pcc_threshold=0.99) is None


def test_summarize_reports_speedup_vs_full_precision_baseline():
    results = [
        {"fidelity": "HiFi4", "dtype": "bfloat16", "ms": 1.0, "pcc": 0.999},  # baseline
        {"fidelity": "HiFi2", "dtype": "bfloat8_b", "ms": 0.4, "pcc": 0.995},  # best
    ]
    table = [{"m": 32, "k": 2688, "n": 2048, "candidates": results, "best": MS.pick_best(results, 0.99)}]
    s = MS.summarize(table)
    assert s["shapes"] == 1 and s["seeded"] == 1 and s["improved"] == 1
    seed = s["seeds"][0]
    assert seed["baseline_ms"] == 1.0 and seed["best_ms"] == 0.4 and seed["speedup"] == 2.5


def test_optimize_prompt_warm_starts_from_sweep_table():
    # #1 integration: the cc optimize agent must be told to consult matmul_sweep.json as a warm-start
    # for a matmul's fidelity/dtype rung, and that the eager guess STILL passes the normal verify.
    run_path = Path(__file__).resolve().parents[1] / "cc_optimize" / "run.py"
    spec = importlib.util.spec_from_file_location("cc_optimize_run", str(run_path))
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)  # run.py is stdlib-only, safe to import
    prompt = run._PROMPT
    assert "matmul_sweep.json" in prompt
    assert "fidelity" in prompt and "dtype" in prompt
    # must preserve the verify gate (a warm-start guess is not a free commit)
    assert "check_pcc" in prompt and "measure_candidate" in prompt
    # the hitl prompt inherits it too
    assert "matmul_sweep.json" in run._HITL_PROMPT


def test_standalone_has_cli_and_does_not_import_perf_mcp():
    # the pre-pass must be SELF-CONTAINED: it exposes a CLI (main/run_prepass) and must NOT reach into
    # the optimize tool's internals (perf_mcp / the loop) -- that is the whole point of a separate flag.
    import inspect

    assert callable(MS.main) and callable(MS.run_prepass) and callable(MS.enumerate_matmul_sigs)
    import_lines = [
        ln.strip() for ln in inspect.getsource(MS).splitlines() if ln.strip().startswith(("import ", "from "))
    ]
    joined = " ".join(import_lines)
    assert "perf_mcp" not in joined, f"pre-pass must not import the optimize tool internals: {import_lines}"
    assert "import run" not in joined and "cc_optimize.run" not in joined
