# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Guards docs/MODEL_ANALYSIS.md against drifting from the reference model.

About two thirds of that document is machine-derived: the operator inventory, the per-module
shape table, the parameter groups and the module tree. If the reference model changes and the
document is not regenerated, those become quietly wrong. This asserts they cannot.

Separate module rather than folded into test_reference_modules.py, so the 475M-parameter
synthetic model this needs is allocated and released on its own instead of stacking on that
module's fixture. No weights and no network.
"""

from models.experimental.nomic_embed_text_v2_moe.scripts.analysis import OUTPUT_PATH, generate_markdown

REGENERATE_COMMAND = "python -m models.experimental.nomic_embed_text_v2_moe.scripts.analysis"


def test_committed_analysis_matches_regeneration():
    committed = OUTPUT_PATH.read_text()
    regenerated = generate_markdown()

    assert committed == regenerated, (
        f"{OUTPUT_PATH.name} no longer matches what the generator produces. Regenerate with:\n"
        f"    {REGENERATE_COMMAND}\n"
        "Then read the diff before committing it. Usually this means the reference model's "
        "operators, shapes, parameters or module structure moved. It can also mean the torch "
        "version changed which operators a high-level call decomposes into, which is a real "
        "change to what the port must implement and is worth seeing either way."
    )
