# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F141: the text agents and the overseer read says what the framework does (audit section 3), and the owner's
precision rules stay in the skill."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT = [
    *sorted((ROOT / "agents").glob("*.md")),
    *sorted((ROOT / "skill").rglob("*.md")),
    *sorted((ROOT / "briefs").glob("*")),
    ROOT / "README.md",
    *sorted((ROOT / "templates").glob("*")),
    *sorted((ROOT / "knowledge").glob("*.md")),
    ROOT / "orchestrator.py",
    ROOT / "testing" / "templates.py",
]
STALE = re.compile(
    r"max_precision|maximum precision|goldens stay official|bit-identical or explained|thresholds\.(component|block)"
    r"|component_max_precision|low_precision_kinds|drift\.json|change a threshold|PCC trail|spec thresholds"
    r"|a good starting point|vs the bf16 golden|^thresholds:",
    re.I | re.M,
)
OWNER_RULES = (  # skill/bringup/SKILL.md, accuracy section (owner, 2026-10-21)
    "No device run at max precision, ever",
    "Every device test runs the shipped realistic settings",
    "Bit-identity with an older device path is never a criterion",
    "Bring-up ops before stock TTNN ops",
    "Packer L1 accumulation adds in the packed format",
    "A deferred part of an owner rule is queued at once",
)


def test_no_stale_precision_text():
    bad = [
        f"{p.relative_to(ROOT)}: {m.group(0)!r}"
        for p in TEXT
        if p.is_file()
        for m in STALE.finditer(p.read_text(errors="replace"))
    ]
    assert not bad, bad


def test_the_owners_precision_rules_stay_in_the_skill():
    text = " ".join((ROOT / "skill" / "bringup" / "SKILL.md").read_text().split())
    missing = [r for r in OWNER_RULES if r not in text]
    assert not missing, missing
