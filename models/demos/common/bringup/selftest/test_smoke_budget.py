# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F195: the device smoke decodes as many greedy tokens as the CPU smoke (DeepSeek-V4.1 L.smoke, 2026-10-27: the
device stopped at a fixed 8 tokens mid-thought, 'We need answer one word. Capital of', where the CPU smoke's 24 reached
'...</think>Paris')."""

import inspect

from models.demos.common.bringup.intake import check_hf_sanity
from models.demos.common.bringup.testing import smoke as device_smoke


class _Spec:
    def __init__(self, smoke):
        self.data = {"intake": {"smoke": smoke}}

    def get(self, key, default=None):
        node = self.data
        for k in key.split("."):
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node


def test_the_device_smoke_has_the_cpu_smokes_budget():
    cpu_default = inspect.signature(check_hf_sanity.smoke).parameters["n"].default
    assert check_hf_sanity.smoke_tokens(_Spec({"prompt": "p", "expect": "e"})) == cpu_default
    src = inspect.getsource(device_smoke.run_smoke)
    assert "smoke_tokens(s)" in src and "max_new: int = 8" not in src


def test_the_spec_sets_the_budget_for_both():
    assert check_hf_sanity.smoke_tokens(_Spec({"prompt": "p", "expect": "e", "max_new": 40})) == 40
