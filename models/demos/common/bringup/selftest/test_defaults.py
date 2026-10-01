# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""defaults.yaml is the one place the framework's switches are set: a spec overrides them under the same key, and
no framework module carries its own default for a key that is there."""

import re
from pathlib import Path

from models.demos.common.bringup.core import defaults
from models.demos.common.bringup.core.spec import Spec

ROOT = Path(__file__).resolve().parents[1]
GET = re.compile(r"""\.get\(\s*f?["']([a-z_]+(?:\.[a-z_]+)+)["']\s*,""")


def test_no_module_keeps_its_own_default_for_a_framework_switch():
    bad = []
    for f in ROOT.rglob("*.py"):
        if "selftest" in f.parts or f.name == "defaults.py":
            continue
        for n, line in enumerate(f.read_text().splitlines(), 1):
            for key in GET.findall(line):
                if defaults.has(key):
                    bad.append(f"{f.relative_to(ROOT)}:{n}: {key} has an inline default; defaults.yaml owns it")
    assert not bad, "\n".join(bad)


def test_the_spec_overrides_a_default_and_falls_back_to_it():
    def spec(data):
        s = Spec.__new__(Spec)
        s.data = data
        return s

    assert spec({"agents": {"component_review": "all"}}).get("agents.component_review") == "all"
    assert spec({}).get("agents.component_review") == defaults.get("agents.component_review")
    assert spec({}).threshold("layer", 0.0) == defaults.get("thresholds.layer")
    assert spec({}).get("not.a.switch", 7) == 7
