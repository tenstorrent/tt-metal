"""Onboard Qwen3.6-27B to the LLM spec-test (API conformance) suite.

Data-only, two files, mirroring the qwen3_32b entry that already exists:

  test_module/server_tests_config.json   add model_configs."qwen36_27b"
  test_module/test_suites/llm.json       add "qwen36_27b" to the models list of the
                                         VLLMParamConformanceTest matrix

Why this is needed: `run.py --workflow spec_tests --model Qwen3.6-27B` exits rc=1 with
"No blocks accumulated — cannot generate report", because the model appears in no
spec-test matrix. test_filter.filter_by_model matches `model_name in suite["weights"]`,
and `weights` comes from the model_configs entry, so a model absent from model_configs can
never be selected. That is why the fleet coverage report shows this check as never run,
against Gemma-4-31B's 21/21.

Only the generic VLLMParamConformanceTest is added. The suite also defines
VLLMQwen3StreamingParamConformanceTest, described as "Qwen3-32B streaming reasoning/tool-call
regressions" and currently scoped to qwen3_32b alone. That template is highly relevant to this
model -- it is a Qwen3-family reasoning model configured with reasoning_parser qwen3 AND
tool_call_parser qwen3_coder -- but it may carry 32B-specific assertions, so adding it is left
as a deliberate follow-up rather than assumed safe.
"""

import json
import sys

SERVER_CFG = "test_module/server_tests_config.json"
SUITES = "test_module/test_suites/llm.json"
KEY = "qwen36_27b"
MODEL_NAME = "Qwen3.6-27B"


def patch_server_cfg():
    with open(SERVER_CFG) as f:
        d = json.load(f)
    mc = d["model_configs"]
    if KEY in mc:
        print(f"  {SERVER_CFG}: {KEY} already present")
        return False
    ref = mc["qwen3_32b"]
    mc[KEY] = {
        "id_name": "qwen3.6-27b",
        "weights": [MODEL_NAME],
        "category": ref["category"],
        # this model's specs declare P300X2 and P150X8
        "compatible_devices": ["p300x2", "p150x8"],
    }
    with open(SERVER_CFG, "w") as f:
        json.dump(d, f, indent=4)
        f.write("\n")
    print(f"  {SERVER_CFG}: added model_configs.{KEY} (weights={[MODEL_NAME]})")
    return True


def patch_suites():
    with open(SUITES) as f:
        d = json.load(f)
    changed = False
    for m in d["test_matrices"]:
        templates = [tc.get("template") for tc in m.get("test_cases", [])]
        if "VLLMParamConformanceTest" not in templates:
            continue
        if "p300x2" not in m.get("devices", []):
            print("  matrix has VLLMParamConformanceTest but no p300x2 device; skipping")
            continue
        if KEY in m["models"]:
            print(f"  {SUITES}: {KEY} already in the conformance matrix")
            continue
        m["models"].append(KEY)
        changed = True
        print(f"  {SUITES}: added {KEY} -> {templates}")
    if changed:
        with open(SUITES, "w") as f:
            json.dump(d, f, indent=4)
            f.write("\n")
    return changed


a = patch_server_cfg()
b = patch_suites()

# re-read both to prove they are still valid JSON
for p in (SERVER_CFG, SUITES):
    with open(p) as f:
        json.load(f)
    print(f"  {p}: valid JSON")

if not (a or b):
    print("  nothing changed")
    sys.exit(0)
print("  done")
