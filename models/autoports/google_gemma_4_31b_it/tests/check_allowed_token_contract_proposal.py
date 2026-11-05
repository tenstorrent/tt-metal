# SPDX-License-Identifier: Apache-2.0
"""CPU-only validation of the unapplied PC-02 assertion helper and counterexamples."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

from transformers import AutoTokenizer

from ..tt.weights import HF_MODEL, HF_REVISION


def main():
    root = Path(__file__).parents[1]
    patch = (root / "doc/vllm_integration/PC-02-allowed-token-ids.patch").read_text()
    lines = patch.splitlines()
    start = lines.index("+def _assert_allowed_token_response(response, allowed_ids):")
    block = []
    for line in lines[start:]:
        if not line.startswith("+"):
            break
        block.append(line[1:])
    tree = ast.parse("\n".join(block))
    namespace = {}
    exec(compile(tree, "PC-02 proposed helper", "exec"), namespace)
    check = namespace["_assert_allowed_token_response"]
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL, revision=HF_REVISION, local_files_only=True)
    ids = [1, 2, 3]
    text = tokenizer.decode(ids, skip_special_tokens=True)
    assert text == "" and all(token in tokenizer.all_special_ids for token in ids)

    def response(tokens, visible="", count=None):
        return SimpleNamespace(
            choices=[SimpleNamespace(token_ids=tokens, text=visible)],
            usage=SimpleNamespace(completion_tokens=len(tokens) if count is None else count),
        )

    check(response(ids, text), ids)
    failures = []
    for name, result in (
        ("disallowed_visible", response([4], "visible")),
        ("missing_tokens", response([])),
        ("wrong_usage", response([1], count=2)),
    ):
        try:
            check(result, ids)
        except AssertionError:
            failures.append(name)
        else:
            raise AssertionError(f"Proposed assertion missed {name}")
    report = dict(
        scope="CPU-only unapplied patch assertion regression",
        hf_revision=HF_REVISION,
        original_nonempty_text_assertion_would_fail=True,
        special_ids=ids,
        suppressed_text=text,
        proposed_special_token_check_passed=True,
        invalid_responses_rejected=failures,
        plugin_patch_applied=False,
    )
    (root / "readiness_vllm/pc02_host_regression.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
