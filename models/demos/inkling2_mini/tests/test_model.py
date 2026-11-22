# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Full model vs the CPU golden (needs the weight cache $INKLING_CACHE and the golden dir $INKLING_GOLDEN).

Hidden-forced (each application gets its golden input): per-layer and logits bars.
Token-only (golden tokens, no hidden forcing): reported, gated on logits PCC only.
"""

import pytest

from models.demos.inkling2_mini.convert import cache as wcache
from models.demos.inkling2_mini.demo.golden_check import GOLDEN, run_prompt
from models.demos.inkling2_mini.tests.hf_ref import finite_min
from models.demos.inkling2_mini.tt.model import Inkling2Model

pytestmark = pytest.mark.skipif(
    GOLDEN is None
    or wcache.DEFAULT_ROOT is None
    or not (GOLDEN / "prompt0").exists()
    or not (wcache.DEFAULT_ROOT / "manifest.json").exists(),
    reason="golden or weight cache missing (set INKLING_GOLDEN and INKLING_CACHE)",
)


def test_model_vs_golden(device):
    model = Inkling2Model(device, max_ctx=2048, log=lambda s: None)
    for prompt in (0, 1):
        r = run_prompt(model, prompt, force=True)
        assert finite_min(r["app_pcc"]) >= 0.99, r["app_pcc"]
        assert r["logits_pcc"] >= 0.98, r
        assert r["top5_has_ref_top1"] >= 0.98, r
        assert r["device_argmax_eq_host"] == 1.0, r
        r = run_prompt(model, prompt, force=False)
        print(prompt, r["logits_pcc"], r["top1_agree"], r["top5_has_ref_top1"])
        assert r["logits_pcc"] >= 0.98, r
