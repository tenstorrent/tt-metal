# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PCC test for the trace-captured KV-cached decode (TTNNGPTTracedDecoder).

Captures the decode step into a device trace, then replays it per token over the golden
`inputs_embeds`; the collected latents must match the prefill golden `latents.pt` (same
0.999 gate as the non-traced decode). Requires a device with a trace region.

Run:
    pytest -svv models/experimental/xtts_v2/tests/test_gpt_trace_pcc.py
"""

import os

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.xtts_v2.tt.ttnn_xtts_gpt import preprocess_gpt_parameters
from models.experimental.xtts_v2.tt.ttnn_xtts_gpt_decode import TTNNGPTTracedDecoder

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "..", "golden", "gpt")
TARGET_PCC = 0.999
TRACE_REGION = 50_000_000


def run_trace_pcc(device):
    inputs_embeds = torch.load(os.path.join(GOLDEN_DIR, "inputs_embeds.pt"))
    golden = torch.load(os.path.join(GOLDEN_DIR, "latents.pt"))
    S = inputs_embeds.shape[1]

    dec = TTNNGPTTracedDecoder(
        device, preprocess_gpt_parameters(device, dtype=ttnn.bfloat16), max_seq=((S + 31) // 32) * 32
    )
    dec.capture()
    latents = dec.decode_sequence(inputs_embeds)

    passed, pcc_msg = comp_pcc(golden, latents, pcc=TARGET_PCC)
    print(f"[traced decode] latents {tuple(latents.shape)} pcc: {pcc_msg}")
    return passed, pcc_msg


@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION}], indirect=True)
def test_gpt_trace_pcc(device):
    passed, pcc_msg = run_trace_pcc(device)
    assert passed, f"traced decode PCC below {TARGET_PCC}: {pcc_msg}"


if __name__ == "__main__":
    import sys

    dev = ttnn.open_device(device_id=0, trace_region_size=TRACE_REGION)
    try:
        dev.enable_program_cache()
        ok, msg = run_trace_pcc(dev)
    finally:
        ttnn.close_device(dev)
    print(("PASSED " if ok else "FAILED ") + str(msg))
    sys.exit(0 if ok else 1)
