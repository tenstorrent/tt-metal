# /// script
# requires-python = ">=3.11"
# dependencies = ["swebench==4.0.3", "datasets==2.16.1", "fastcore<1.11"]
# ///

import json
import os
import sys
from types import SimpleNamespace

from swebench.harness.constants import (
    END_TEST_OUTPUT,
    FAIL_ONLY_REPOS,
    FAIL_TO_PASS,
    KEY_INSTANCE_ID,
    PASS_TO_PASS,
    START_TEST_OUTPUT,
    EvalType,
    ResolvedStatus,
)
from swebench.harness.grading import get_eval_tests_report, get_logs_eval, get_resolution_status

with open("/tests/config.json", "r") as file:
    datum = json.load(file)


# Grading existing logs needs identity and test expectations, not image build
# scripts. make_test_spec also downloads environment metadata from GitHub.
def expected_tests(key):
    value = datum.get(key, [])
    return json.loads(value) if isinstance(value, str) else value


test_spec = SimpleNamespace(
    instance_id=datum[KEY_INSTANCE_ID],
    repo=datum["repo"],
    version=datum.get("version"),
    FAIL_TO_PASS=expected_tests(FAIL_TO_PASS),
    PASS_TO_PASS=expected_tests(PASS_TO_PASS),
)
report_map = {}
instance_id = datum[KEY_INSTANCE_ID]
report_map[instance_id] = {
    "patch_is_None": False,
    "patch_exists": False,
    "patch_successfully_applied": False,
    "resolved": False,
}
report_map[instance_id]["patch_exists"] = True
test_log_path = os.environ["LOG_FILE"]
with open(test_log_path, "r+") as f:
    content = f.read()
    f.seek(0)
    f.write(f"{START_TEST_OUTPUT}\n{content}\n{END_TEST_OUTPUT}")
    f.truncate()

eval_status_map, found = get_logs_eval(test_spec, test_log_path)
if found:
    report_map[instance_id]["patch_successfully_applied"] = True

    eval_ref = {
        KEY_INSTANCE_ID: test_spec.instance_id,
        FAIL_TO_PASS: test_spec.FAIL_TO_PASS,
        PASS_TO_PASS: test_spec.PASS_TO_PASS,
    }

    eval_type = EvalType.FAIL_ONLY if test_spec.repo in FAIL_ONLY_REPOS else EvalType.PASS_AND_FAIL

    report = get_eval_tests_report(eval_status_map, eval_ref, eval_type=eval_type)
    if get_resolution_status(report) == ResolvedStatus.FULL.value:
        report_map[instance_id]["resolved"] = True
    report_map[instance_id]["tests_status"] = report

json.dump(report_map, open("/logs/verifier/report.json", "w"), indent=4)

print(f"SWEBench results starts here")
if report_map[instance_id]["resolved"]:
    print("PASSED")
else:
    print("FAILED")
print("SWEBench results ends here")
sys.exit(0 if report_map[instance_id]["resolved"] else 1)
