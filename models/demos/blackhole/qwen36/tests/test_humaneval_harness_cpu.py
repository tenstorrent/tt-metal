# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only gate for the HumanEval scoring harness (campaign/humaneval_common.py).

The pass@1 numbers are only as trustworthy as reply->code extraction and the
subprocess check() runner; pin both without a device or ttnn:

    pytest models/demos/blackhole/qwen36/tests/test_humaneval_harness_cpu.py --noconftest -v
"""

from models.demos.blackhole.qwen36.campaign.humaneval_common import extract_code, run_candidate

_PROMPT = 'def add(a, b):\n    """Return a + b."""\n'
_DOC = {
    "task_id": "Synthetic/0",
    "prompt": _PROMPT,
    "entry_point": "add",
    "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n    assert candidate(-1, 1) == 0\n",
}
_GOOD = "def add(a, b):\n    return a + b\n"


def test_extracts_fenced_block_after_think():
    reply = (
        "<think>chain of thought with a ```python\ndef add(a, b): return 0\n``` decoy</think>\n"
        "Here you go:\n```python\n" + _GOOD + "```\nHope that helps!"
    )
    assert extract_code(reply, "add", _PROMPT) == _GOOD


def test_prefers_block_naming_the_entry_point():
    reply = "```python\n# helper only\ndef helper():\n    pass\n```\ntext\n```python\n" + _GOOD + "```"
    assert extract_code(reply, "add", _PROMPT) == _GOOD


def test_unfenced_reply_falls_back_to_raw_text():
    assert extract_code(_GOOD, "add", _PROMPT) == _GOOD


def test_bare_body_completion_grafts_official_prompt():
    code = extract_code("    return a + b\n", "add", _PROMPT)
    assert code.startswith("def add(a, b):")
    assert code.endswith("    return a + b\n")
    ok, detail = run_candidate(code, _DOC, timeout=10)
    assert ok, detail


def test_run_candidate_pass_and_fail():
    ok, detail = run_candidate(_GOOD, _DOC, timeout=10)
    assert ok and detail == "pass"
    ok, detail = run_candidate("def add(a, b):\n    return a - b\n", _DOC, timeout=10)
    assert not ok and "AssertionError" in detail
    ok, detail = run_candidate("def add(a, b:\n", _DOC, timeout=10)  # syntax error
    assert not ok


def test_run_candidate_timeout():
    ok, detail = run_candidate("import time\ndef add(a, b):\n    time.sleep(60)\n", _DOC, timeout=2)
    assert not ok and detail == "timeout"
