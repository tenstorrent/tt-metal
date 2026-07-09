# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end pipeline test for `tencent/HunyuanImage-3.0` (Call-1:
`hunyuan_image3_transformer_prefill`).

Drives the ONE shared pipeline in `tt/pipeline.py` (the same code the demo runs)
and asserts:

  Gate 1 — every routed graduated stub runs native ttnn (source scan here;
           authoritative host-op check in test_host_op_selftest / Command 3).
  Gate 2 — all 3 graduated stubs (image3_decoder_layer, mo_e, top_k_gate) are
           INVOKED on the real forward path (real per-instance call counters).
  Gate 3 — final e2e PCC(last_hidden_state) vs the HF reference forward >= 0.95.

Run:  ./python_env/bin/python -m pytest \
        models/demos/vision/generative/hunyuanimage_3_0/tests/e2e/test_e2e_prefill.py -s
"""

from __future__ import annotations

import os
import re

import pytest
import torch

from models.demos.vision.generative.hunyuanimage_3_0.tt import pipeline as pl

HERE = os.path.dirname(os.path.abspath(__file__))
STUBS_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "_stubs"))
PROMPT = "A serene mountain lake at sunrise, photorealistic, ultra detailed."
PCC_TARGET = 0.95
GRADUATED = {"image3_decoder_layer", "mo_e", "top_k_gate"}

# Forbidden host-compute patterns in the graduated stub hot paths (Gate 1).
_FORBIDDEN = [
    r"\.generate\(",
    r"torch\.matmul",
    r"torch\.mm\(",
    r"torch\.bmm",
    r"torch\.einsum",
    r"torch\.softmax",
    r"torch\.log_softmax",
    r"torch\.layer_norm",
    r"torch\.rms_norm",
    r"F\.softmax",
    r"F\.linear",
    r"torch\.nn\.functional",
    r"F\.scaled_dot_product_attention",
    r"torch\.scaled_dot_product_attention",  # torch sdpa (ttnn.transformer.* is fine)
    r"torch\.argmax",
    r"torch\.topk",
    r"torch\.multinomial",
]

_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = pl.load_reference_model()
    return _MODEL


def test_gate1_stub_sources_are_native_ttnn():
    """Gate 1 (static): the graduated stub hot paths contain no torch host
    compute. Weight prep (`.to(torch.float32)` in `_to_ttnn`) is allowed."""
    offenders = []
    for name in GRADUATED:
        src = open(os.path.join(STUBS_DIR, f"{name}.py")).read()
        for pat in _FORBIDDEN:
            if re.search(pat, src):
                offenders.append(f"{name}: {pat}")
    assert not offenders, f"Gate 1 FAIL — torch host compute in graduated stubs: {offenders}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_e2e_prefill_gates(device_params, device):
    torch.manual_seed(0)
    model = _get_model()
    pipe = pl.build_pipeline(device, model)

    result = pipe.run_and_compare(PROMPT, pcc_target=PCC_TARGET)

    # ALWAYS print the achieved PCC (pass or fail), on its own line, before asserts.
    print(f"\ne2e PCC={result['pcc']}")
    print(f"e2e l_aux_tt={result['l_aux_tt']:.6f} l_aux_ref={result['l_aux_ref']:.6f}")
    print(f"e2e graduated invocations={result['invocations']}")
    print(f"e2e num_layers={pipe.num_layers} seq_len={pipe.seq_len}")

    # Gate 2 — every graduated stub invoked on the real forward path.
    inv = result["invocations"]
    missing = GRADUATED - {k for k, v in inv.items() if v and v > 0}
    assert not missing, f"Gate 2 FAIL — graduated stub(s) never invoked: {missing} (invocations={inv})"

    # Gate 3 — final e2e PCC >= 0.95.
    assert result["pcc_ok"] and result["pcc"] >= PCC_TARGET, f"Gate 3 FAIL — e2e PCC {result['pcc']} < {PCC_TARGET}"
