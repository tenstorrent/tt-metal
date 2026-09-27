"""Deselect every collected case whose nodeid is not listed in $NODEID_SUBSET_FILE.

Why it exists.  `eval/golden_tests/rms_norm_ttnn` collects 121 864 cases, of which ~23 300
actually run; a whole-suite `eval_test_runner.sh` pass measured ~49 cases/minute (the
profiler and metrics plugins dominate), i.e. about eight hours.  This plugin lets a
regression check run an explicit node-id list instead, so the same ~23 300 live cells can be
covered in two ~40-minute passes.  Collection still walks all 121 864 items -- that part is
a fixed ~3 minutes -- but nothing outside the list executes.

    python3 - <<'EOF'      # build the list from a previous run's results
    import json
    d = json.load(open("generated/verifier_results/test_results.json"))
    ids = [r["nodeid"] for r in d if r["status"] == "passed"
           and r["test_file"].endswith("test_golden")]
    open("/tmp/ids.txt", "w").write("\n".join(ids) + "\n")
    EOF

    PYTHONPATH=tests/ttnn/unit_tests/operations/rms_norm_ttnn \
    NODEID_SUBSET_FILE=/tmp/ids.txt \
    scripts/run_safe_pytest.sh --run-all -p nodeid_subset_plugin \
        eval/golden_tests/rms_norm_ttnn/test_golden.py

It is a NO-OP when NODEID_SUBSET_FILE is unset, so `-p nodeid_subset_plugin` is safe to
leave on a command line.  It never changes what a case asserts -- only which cases run.
"""
import os


def pytest_collection_modifyitems(config, items):
    path = os.environ.get("NODEID_SUBSET_FILE")
    if not path:
        return
    wanted = {line.strip() for line in open(path) if line.strip()}
    keep, drop = [], []
    for it in items:
        (keep if it.nodeid in wanted else drop).append(it)
    items[:] = keep
    config.hook.pytest_deselected(items=drop)
