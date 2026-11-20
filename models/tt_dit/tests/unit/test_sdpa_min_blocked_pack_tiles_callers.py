# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Who sets `SDPAProgramConfig.min_blocked_pack_tiles`: every tt_dit attention, and nothing outside tt_dit.

Unset, the SDPA streaming compute kernel keeps the architecture default (8 tiles on Wormhole, 2 on Blackhole), so a
model that does not set the field compiles the same kernel it always has. Host only: these tests read source files.
"""

import ast
from pathlib import Path

MODELS = Path(__file__).resolve().parents[3]
TT_DIT = MODELS / "tt_dit"

# DiT attention: every SDPAProgramConfig built under these paths sets the field.
DIT_ATTENTION = [
    TT_DIT / "blocks",
    TT_DIT / "models" / "transformers",
    TT_DIT / "models" / "vae" / "minimax_h3" / "decoder_minimax_h3.py",
]


def _sources(path: Path) -> list[Path]:
    return sorted(path.rglob("*.py")) if path.is_dir() else [path]


def _sdpa_program_configs(path: Path) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path)))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "SDPAProgramConfig"
    ]


def test_dit_attention_sets_min_blocked_pack_tiles():
    found = 0
    missing = []
    for path in (source for root in DIT_ATTENTION for source in _sources(root)):
        for call in _sdpa_program_configs(path):
            found += 1
            value = next((kw.value for kw in call.keywords if kw.arg == "min_blocked_pack_tiles"), None)
            if not (isinstance(value, ast.Constant) and value.value == 2):
                missing.append(f"models/{path.relative_to(MODELS)}:{call.lineno}")
    assert found > 0, "no SDPAProgramConfig found: the paths above moved"
    assert not missing, f"tt_dit attention SDPAProgramConfig without min_blocked_pack_tiles=2: {missing}"


def test_only_tt_dit_sets_min_blocked_pack_tiles():
    outside = [
        f"models/{path.relative_to(MODELS)}"
        for path in MODELS.rglob("*.py")
        if TT_DIT not in path.parents and "min_blocked_pack_tiles" in path.read_text(errors="ignore")
    ]
    assert not outside, f"min_blocked_pack_tiles is set outside models/tt_dit: {outside}"
