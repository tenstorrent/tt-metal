# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PCC test for the KV-cached decode loop of the TTNN XTTS-v2 GPT core (Block 3).

Validation: feed the golden `inputs_embeds` one token at a time through the KV-cached
decode step; the stacked per-step latents must match the parallel prefill golden
`latents.pt` (causal attention => decode step t == prefill position t).

Run:
    pytest -svv models/experimental/xtts_v2/tests/test_gpt_decode_pcc.py
  or standalone:
    python models/experimental/xtts_v2/tests/test_gpt_decode_pcc.py
"""

import os

import torch
import ttnn

from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.xtts_v2.tt.ttnn_xtts_gpt import preprocess_gpt_parameters
from models.experimental.xtts_v2.tt.ttnn_xtts_gpt_decode import TTNNGPTDecoder

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "..", "golden", "gpt")
TARGET_PCC = 0.999  # native bf16 decode path


def _load_golden():
    inp = torch.load(os.path.join(GOLDEN_DIR, "inputs_embeds.pt"))
    latents = torch.load(os.path.join(GOLDEN_DIR, "latents.pt"))
    return inp, latents


def run_decode_pcc(device, max_seq_request=None):
    inputs_embeds, golden_latents = _load_golden()
    S = inputs_embeds.shape[1]

    # max_seq_request lets the caller exercise the BUG-1 path (odd tile counts). When None,
    # size tightly to S. The decoder rounds max_seq up to a multiple of 64 internally.
    req = max_seq_request if max_seq_request is not None else ((S + 31) // 32) * 32
    params = preprocess_gpt_parameters(device, dtype=ttnn.bfloat16)
    decoder = TTNNGPTDecoder(device, params, max_seq=req)

    latents = []
    for t in range(S):
        xt = ttnn.from_torch(
            inputs_embeds[:, t : t + 1, :], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        lt = decoder.decode_step(xt)
        latents.append(ttnn.to_torch(lt).to(torch.float32))

    dec = torch.cat(latents, dim=1)  # [1, S, 1024]
    passed, pcc_msg = comp_pcc(golden_latents, dec, pcc=TARGET_PCC)
    _, allclose_msg = comp_allclose(golden_latents, dec)
    print(f"decoded latents: golden={tuple(golden_latents.shape)} ttnn={tuple(dec.shape)} steps={S}")
    print(f"pcc: {pcc_msg}")
    print(f"allclose: {allclose_msg}")
    return passed, pcc_msg


def test_gpt_decode_pcc(device):
    passed, pcc_msg = run_decode_pcc(device)
    assert passed, f"GPT decode PCC below {TARGET_PCC}: {pcc_msg}"


def test_gpt_decode_pcc_large_odd_max_seq(device):
    """BUG-1 regression: requesting an odd-tile max_seq (736 = 23 tiles) used to collapse
    decode to PCC ~0.63. The decoder now rounds up to an even tile count (768), so PCC
    must stay above target."""
    passed, pcc_msg = run_decode_pcc(device, max_seq_request=736)
    assert passed, f"BUG-1 regression: odd-tile max_seq decode PCC below {TARGET_PCC}: {pcc_msg}"


if __name__ == "__main__":
    import sys

    dev = ttnn.open_device(device_id=0)
    try:
        dev.enable_program_cache()
        ok, msg = run_decode_pcc(dev)
        ok2, msg2 = run_decode_pcc(dev, max_seq_request=736)
    finally:
        ttnn.close_device(dev)
    print(("PASSED " if ok else "FAILED ") + str(msg))
    print(("PASSED " if ok2 else "FAILED ") + " [odd-max_seq regression] " + str(msg2))
    sys.exit(0 if (ok and ok2) else 1)
