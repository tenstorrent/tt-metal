# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Offline tests of python/run_summary.py on synthetic run folders."""
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "python"))
import run_summary  # noqa: E402

LINE_OK = (
    "======== 1660 passed, 20 skipped, 2800 deselected in 201.47s (0:03:21) ========="
)
LINE_FAIL = (
    "======== 412 failed, 1248 passed, 20 skipped, 2800 deselected in 300.00s ========="
)
TEST = "test_eltwise_binary"


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def make_run(
    root,
    name="r1",
    dirty_line=LINE_OK,
    postcheck=True,
    agent_exit=0,
    static=True,
    perf=True,
):
    state = os.path.join(root, "runs", name)
    run_dir = os.path.join(root, "hal-runs", f"20260101_000000_B_c0e38f1f_m_{name}")
    logs = os.path.join(run_dir, "logs")
    write(
        os.path.join(state, "run.env"),
        f'PROFILE="run8"\nREFERENCE_SERIES="8"\nTESTS="{TEST}:add"\n',
    )
    write(os.path.join(state, "run_dir.txt"), run_dir + "\n")
    write(os.path.join(state, "stages.tsv"), "harness\tok\t0\tt0\tt1\n")
    meta = [
        "mode=B",
        "base_commit=c0e38f1ff70b7ee70c526fb44cbbf16736a341ed",
        "hal_commit=b9e522d39ae534f0281499dae11b425574d89d0d",
        "docs_commit=d8c6fa7c22fb89e109178778a537e5860fae0ac6",
        "model=claude-opus-5-5",
        f"tests={TEST}:add",
        f"expect_passed={TEST}:add=1660",
        "prompt_file=/x/prompts/run8.md",
        "prompt_md5=3dbea124246f22cae0a3ce1d444d19b7",
        "claude_version=2.1.287 (Claude Code)",
        "dry_run=0",
        f"claude_exit_code={agent_exit}",
        "claude_seconds=6000",
        f"hal_test_exit_code_{TEST}=0",
        f"dirty_exit_code_{TEST}={0 if dirty_line == LINE_OK else 1}",
    ]
    write(os.path.join(logs, "run_meta.txt"), "\n".join(meta) + "\n")
    write(os.path.join(logs, "SUMMARY.txt"), "==== SUMMARY ====\n")
    for f, line in (
        (f"baseline_{TEST}.txt", LINE_OK),
        (f"hal_test_{TEST}.txt", LINE_OK),
        (f"dirty_{TEST}.txt", dirty_line),
    ):
        write(os.path.join(logs, f), "....\n" + line + "\n")
    metrics = {
        "cost_usd": 22.8,
        "num_turns": 200,
        "pytest_runs_total": 51,
        "pytest_runs_by_kind": {"pass": 23, "fail": 5},
        "process": {
            "iterations": 40,
            "first_full_pass": {f"{TEST}_hal": {"iteration": 37, "elapsed_min": 64.99}},
        },
        "counts": {"denied": 16},
    }
    write(os.path.join(logs, "metrics.json"), json.dumps(metrics))
    write(
        os.path.join(logs, "codestats.json"),
        json.dumps(
            {
                "total": {
                    "loc": 1134,
                    "hex_literals": 6,
                    "numeric_shifts": 0,
                    "hal_uses": 170,
                }
            }
        ),
    )
    write(
        os.path.join(logs, "raw_check.txt"),
        "# Raw TT instructions\n(none)\n\n# NOT_IN_HAL marks\na.h:1: // NOT_IN_HAL: x\n",
    )
    write(os.path.join(logs, "forbidden_access.txt"), "\n# total attempts: 0\n")
    write(os.path.join(logs, "mcp_calls.txt"), "\n# total MCP calls: 0\n")
    write(
        os.path.join(logs, "probe_check.txt"),
        "# HAL probing: 0 traces (has_include 0)\n",
    )
    if postcheck:
        pc = os.path.join(state, "postcheck")
        lines = []
        for step, side in (("1", "hal"), ("2", "original"), ("3", "hal"), ("4", "hal")):
            f = f"step{step}_{side}_{TEST}.txt"
            write(os.path.join(pc, f), LINE_OK + "\n")
            lines.append(f"{f}: exit 0, {LINE_OK}")
        write(os.path.join(pc, "checks.txt"), "== steps\n" + "\n".join(lines) + "\n")
    if static:
        summ = {
            "original": {
                "sum": {
                    "builds": 10,
                    "avg": {
                        "text_bytes": 5800.8,
                        "tti": 78.0,
                        "tt_all_paths": 41.0,
                        "unresolved": 6.8,
                    },
                }
            },
            "hal": {
                "sum": {
                    "builds": 10,
                    "avg": {
                        "text_bytes": 4191.2,
                        "tti": 167.0,
                        "tt_all_paths": 55.2,
                        "unresolved": 2.0,
                    },
                }
            },
        }
        write(
            os.path.join(state, "static", "assert_off", "tt_tti_counts.json"),
            json.dumps({"runs": [{"summary": summ}]}),
        )
    if perf:
        p = {
            "reps": 3,
            "l1": {
                "n": 16,
                "orig": 32.8,
                "new": 41.98,
                "ratio_median": 1.324,
                "ratio_min": 1.226,
                "ratio_max": 1.449,
                "faster_variants": 0,
            },
            "rep_spread_max_pct": 0.2,
        }
        write(os.path.join(state, "perf", "perf_summary.json"), json.dumps(p))
    return state


class RunSummaryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_pytest_counts(self):
        self.assertEqual(
            run_summary.pytest_counts(LINE_FAIL),
            {"failed": 412, "passed": 1248, "skipped": 20, "deselected": 2800},
        )
        self.assertIsNone(run_summary.pytest_counts("collecting ..."))
        self.assertEqual(
            run_summary.pytest_counts("=== 2 errors, 3 passed in 1.0s ==="),
            {"errors": 2, "passed": 3},
        )

    def test_pass(self):
        s = run_summary.collect(make_run(self.root))
        self.assertEqual(s["correctness"]["verdict"], "PASS")
        self.assertEqual(
            (s["correctness"]["checks_ok"], s["correctness"]["checks_total"]), (7, 7)
        )
        self.assertEqual(s["agent"]["first_full_pass_iteration"], 37)
        self.assertEqual(s["code"]["not_in_hal_marks"], 1)
        self.assertEqual(s["code"]["raw_instruction_lines"], 0)
        self.assertEqual(s["static"]["assert_off"]["hal"]["tti"], 167.0)
        text = run_summary.render_run(s)
        self.assertIn("PASS", text)
        self.assertIn("+114.1%", text)  # TTI 78 -> 167
        self.assertIn("ratio 1.324", text)

    def test_dirty_failure_is_fail(self):
        s = run_summary.collect(make_run(self.root, dirty_line=LINE_FAIL))
        self.assertEqual(s["correctness"]["verdict"], "FAIL")
        bad = [c for c in s["checks"] if not c["ok"]]
        self.assertEqual(
            [c["step"] for c in bad], ["original right after HAL, no reset"]
        )

    def test_wrong_count_is_fail(self):
        state = make_run(self.root)
        with open(os.path.join(state, "postcheck", f"step4_hal_{TEST}.txt"), "w") as f:
            f.write("=== 1659 passed, 20 skipped in 1.0s ===\n")
        self.assertEqual(run_summary.collect(state)["correctness"]["verdict"], "FAIL")

    def test_without_postcheck(self):
        s = run_summary.collect(make_run(self.root, postcheck=False))
        self.assertEqual(s["correctness"]["verdict"], "PASS (harness checks only)")
        self.assertEqual(s["correctness"]["checks_total"], 3)

    def test_agent_exit(self):
        s = run_summary.collect(make_run(self.root, agent_exit=124))
        self.assertEqual(s["correctness"]["verdict"], "AGENT DID NOT FINISH (exit 124)")

    def test_incomplete(self):
        state = os.path.join(self.root, "runs", "x")
        write(os.path.join(state, "run.env"), 'PROFILE="run8"\n')
        s = run_summary.collect(state)
        self.assertTrue(s["correctness"]["verdict"].startswith("INCOMPLETE"))
        run_summary.render_run(s)

    def test_compare(self):
        a = run_summary.collect(make_run(self.root, "a"))
        b = run_summary.collect(make_run(self.root, "b", static=False, perf=False))
        text = run_summary.render_compare([a, b])
        self.assertIn("median (2)", text)
        self.assertIn("original LLK", text)
        self.assertIn("run 8 median", text)
        md = run_summary.render_compare([a, b], markdown=True)
        self.assertTrue(md.startswith("| run |"))


if __name__ == "__main__":
    unittest.main()
