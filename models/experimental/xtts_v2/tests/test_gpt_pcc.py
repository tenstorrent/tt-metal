# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
PCC test for the TTNN XTTS-v2 GPT transformer core (Block 3) vs the CPU reference.

The reference (seeded input from the checkpoint's real embedding tables + HF GPT2 with
checkpoint weights) is computed live in-process — no golden files needed. Set
XTTS_GOLDEN_DIR to use stored fixtures instead (bit-identical to the live path).

Run:
    pytest -svv models/experimental/xtts_v2/tests/test_gpt_pcc.py
  or standalone:
    python models/experimental/xtts_v2/tests/test_gpt_pcc.py
"""

import os

import torch
import ttnn

from models.common.utility_functions import comp_pcc, comp_allclose
from models.experimental.xtts_v2.tests.reference_helpers import gpt_reference
from models.experimental.xtts_v2.tt.ttnn_xtts_gpt import TTNNGPTCore, preprocess_gpt_parameters

# Default gate for the native bf16 path (~0.9997). The fp32 + manual-attention path
# clears 0.9999 — use HIGH_ACCURACY=1 to gate at 0.9999 with that config.
TARGET_PCC = 0.999


def _load_golden():
    ref = gpt_reference()
    return ref["inputs_embeds"], ref["latents"]


def run_gpt_pcc(device, dtype=None, attention=None, pcc=None):
    # Default = native bf16 + SDPA (flash-attention): the fast path we intend to run on
    # TT (~0.9997 PCC, gated at 0.999). Set HIGH_ACCURACY=1 for the fp32 + manual-attention
    # path that clears 0.9999. See CLAUDE_XTTS_GPT.md.
    high_acc = os.environ.get("HIGH_ACCURACY", "0") == "1"
    if dtype is None:
        dtype = ttnn.float32 if high_acc else ttnn.bfloat16
    if attention is None:
        attention = "manual" if high_acc else "sdpa"
    if pcc is None:
        pcc = 0.9999 if high_acc else TARGET_PCC

    inputs_embeds, golden_latents = _load_golden()

    params = preprocess_gpt_parameters(device, dtype=dtype)
    model = TTNNGPTCore(device, params, activation_dtype=dtype, attention=attention)

    x = ttnn.from_torch(inputs_embeds, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    y = model(x)
    y_torch = ttnn.to_torch(y).to(torch.float32)

    passed, pcc_msg = comp_pcc(golden_latents, y_torch, pcc=pcc)
    _, allclose_msg = comp_allclose(golden_latents, y_torch)
    print(f"latents shape: golden={tuple(golden_latents.shape)} ttnn={tuple(y_torch.shape)}")
    print(f"pcc: {pcc_msg}")
    print(f"allclose: {allclose_msg}")
    return passed, pcc_msg


def test_gpt_core_pcc(device):
    passed, pcc_msg = run_gpt_pcc(device)
    assert passed, f"GPT core PCC below {TARGET_PCC}: {pcc_msg}"


if __name__ == "__main__":
    dev = ttnn.open_device(device_id=0)
    try:
        dev.enable_program_cache()
        ok, msg = run_gpt_pcc(dev)
    finally:
        ttnn.close_device(dev)
    print(("PASSED " if ok else "FAILED ") + str(msg))
    import sys

    sys.exit(0 if ok else 1)
