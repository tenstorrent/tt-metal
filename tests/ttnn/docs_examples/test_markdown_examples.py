# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

MARKDOWN_DOCS = [
    "ttnn/cpp/ttnn/operations/experimental/isin/docs/IsIn.md",
]


def python_blocks(doc):
    text = (REPO_ROOT / doc).read_text()
    return re.findall(r"^```python\n(.*?)^```", text, flags=re.DOTALL | re.MULTILINE)


@pytest.mark.parametrize("doc", MARKDOWN_DOCS, ids=lambda doc: Path(doc).name)
def test_markdown_python_examples(doc):
    blocks = python_blocks(doc)
    assert blocks, f"{doc} has no python example"
    namespace = {}
    for block in blocks:
        exec(compile(block, doc, "exec"), namespace)
