# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""pytest plugin: render every DiffVAE decode tree recorded during a test, for tests OUTSIDE
models/tt_dit/tests/models/vae (whose conftest owns the `decode_tree` fixture).

The pipeline test in tests/models/ltx sets DIFFVAE_STAGE_TIMING=1 but never asks for that fixture,
so the spans are recorded and then dropped. Load this with

    PYTHONPATH=models/tt_dit/experimental/scripts python -m pytest -p decode_tree_plugin ...

Every root (warm-up gen #0 and the timed gens #1/#2 each decode once, so a pipeline run has three)
is printed at teardown and, when DIFFVAE_TREE_OUT is set, also appended to that file so the trees
survive the thousands of loguru lines above them.
"""
import os

import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    from models.tt_dit.utils import decode_tree as tree

    first = tree.root_count()
    yield
    new = tree.roots()[first:]
    if not new:
        return
    out_path = os.environ.get("DIFFVAE_TREE_OUT")
    blocks = []
    for i, root in enumerate(new):
        tag = "TIMED" if i == len(new) - 1 else f"gen #{i}"
        blocks.append(tree.render(root, title=f"{item.name} · decode pass {i + 1} of {len(new)} ({tag})"))
    text = "\n\n".join(blocks)
    print("\n" + text)
    if out_path:
        with open(out_path, "a") as f:
            f.write(text + "\n\n")
