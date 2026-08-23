# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device-free half of the HumanEval gate: reply -> code extraction and the
official check() execution. Kept ttnn-free so the scoring logic itself is
gated by a CPU-only test (tests/test_humaneval_harness_cpu.py)."""

import os
import re
import subprocess
import sys
import tempfile


def he_user_message(he_prompt):
    """The chat message the gate sends for one HumanEval problem."""
    return (
        "Complete the following Python function. Reply with the complete function "
        "definition (repeat the signature shown, include any imports it needs) in "
        "one ```python code block.\n\n```python\n" + he_prompt + "```"
    )


def extract_code(text, entry_point, he_prompt):
    """Candidate function source from a reply; reasoning (<think>) is skipped first."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, flags=re.DOTALL)
    named = [b for b in blocks if f"def {entry_point}" in b]
    code = named[-1] if named else (blocks[-1] if blocks else text)
    if f"def {entry_point}" not in code:
        # Bare-body completion: graft onto the official prompt (signature + docstring).
        code = he_prompt + code
    return code


def run_candidate(code, doc, timeout):
    """Official HumanEval check() in an isolated subprocess. True = pass@1 hit."""
    program = code + "\n\n" + doc["test"] + f"\n\ncheck({doc['entry_point']})\n"
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "candidate.py")
        with open(path, "w") as f:
            f.write(program)
        try:
            proc = subprocess.run([sys.executable, "-I", path], cwd=td, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, "timeout"
    if proc.returncode == 0:
        return True, "pass"
    out = (proc.stderr or proc.stdout).strip()
    return False, out.splitlines()[-1][:200] if out else "fail"
