# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""GLM-4.7-REAP e2e accuracy (PCC-style) test.

Greedy top-1 token agreement against a golden decode sequence. A host 218B HF
reference is impractical, so the golden is bootstrapped from the current (accuracy-
verified) config on first run and stored; subsequent runs — including tt_hw_planner
`optimize` reruns under a changed model — must reproduce it within a top-1 threshold.
This is the regression guard that catches accuracy-losing optimizations.

Full production env config (incl. the decode-opt knobs) is defaulted in conftest.py.

Run:
  export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
  ./python_env/bin/python -m pytest -svq \
    models/experimental/glm4_moe/tests/test_glm4_moe_pcc_e2e.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from loguru import logger

# Accuracy target: fraction of greedy top-1 tokens that must match the golden.
# 1.0 = exact greedy determinism; slightly <1 tolerates minor CCL-async numeric drift.
PCC_TOP1_THRESHOLD = float(os.environ.get("GLM4_MOE_TEST_PCC_TOP1", "0.97"))
PCC_MAX_NEW = int(os.environ.get("GLM4_MOE_TEST_PCC_MAX_NEW", "32"))
_GOLDEN = Path(__file__).parent / "golden" / "glm4_pcc_golden.json"


@pytest.mark.timeout(3600)  # 218B build + decode far exceeds the default 300s
def test_glm4_moe_pcc_e2e(glm4_model):
    res = glm4_model.generate(PCC_MAX_NEW, enable_trace=False, sampling=False, warmup=False)
    got = res["generated"]
    logger.info(f"[pcc] generated {len(got)} tokens; text head: {res['text'][:160]!r}")

    if not _GOLDEN.exists():
        _GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        _GOLDEN.write_text(
            json.dumps(
                {
                    "model_id": os.environ.get("GLM4_MOE_HF_MODEL", ""),
                    "prompt_len": glm4_model.prompt_len,
                    "max_new": PCC_MAX_NEW,
                    "tokens": got,
                },
                indent=2,
            )
        )
        logger.warning(
            f"[pcc] golden BOOTSTRAPPED -> {_GOLDEN}. This run passes trivially; "
            f"review the text below for coherence and commit the golden. Regression "
            f"comparison is active from the next run on."
        )
        print(f"top1_accuracy=1.000000 (bootstrap)", flush=True)
        print(f"[pcc][BOOTSTRAP] text: {res['text']!r}", flush=True)
        return

    golden = json.loads(_GOLDEN.read_text())["tokens"]
    n = min(len(golden), len(got))
    matches = sum(1 for i in range(n) if golden[i] == got[i])
    top1 = matches / max(1, n)
    logger.info(f"[pcc] top1_accuracy={top1:.4f} ({matches}/{n}) threshold={PCC_TOP1_THRESHOLD}")
    print(f"top1_accuracy={top1:.6f}", flush=True)  # machine-parseable for the optimize tool
    assert top1 >= PCC_TOP1_THRESHOLD, (
        f"greedy top-1 agreement {top1:.4f} < {PCC_TOP1_THRESHOLD}; " f"golden[:8]={golden[:8]} got[:8]={got[:8]}"
    )
