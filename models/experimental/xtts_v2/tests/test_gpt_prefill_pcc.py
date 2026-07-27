# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PCC test for the one-shot parallel prefill (ttnn.fill_cache) of TTNNGPTTracedDecoder.

Validation: prefill the first SPLIT positions of the golden `inputs_embeds` in one batched pass
(filling the KV cache 0..SPLIT-1), then decode the remaining positions token-by-token from the
traced step. Because attention is causal, the decode latents at positions SPLIT..S-1 must match the
parallel-prefill golden `latents.pt` at those positions — which is only true if fill_cache seeded
the cache correctly (i.e. equivalent to token-by-token prefill).

Run:
    pytest -svv models/experimental/xtts_v2/tests/test_gpt_prefill_pcc.py
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


def run_prefill_pcc(device):
    inputs_embeds = torch.load(os.path.join(GOLDEN_DIR, "inputs_embeds.pt"))
    golden = torch.load(os.path.join(GOLDEN_DIR, "latents.pt"))
    S = inputs_embeds.shape[1]
    split = S // 2  # prefill the first half in one shot, decode the second half

    dec = TTNNGPTTracedDecoder(
        device, preprocess_gpt_parameters(device, dtype=ttnn.bfloat16), max_seq=((S + 31) // 32) * 32
    )
    dec.reset_caches()
    dec.prefill(inputs_embeds[:, :split, :].contiguous())  # fill cache 0..split-1 (one-shot)
    dec.capture()  # capture AFTER prefill; leaves the prefilled cache intact

    lat = []
    for t in range(split, S):
        emb = ttnn.from_torch(
            inputs_embeds[:, t : t + 1, :].contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=dec.mesh_mapper,
        )
        lat.append(ttnn.to_torch(dec.step_device(emb, t)).to(torch.float32))
    dec_lat = torch.cat(lat, dim=1)  # [1, S-split, 1024]

    passed, pcc_msg = comp_pcc(golden[:, split:, :], dec_lat, pcc=TARGET_PCC)
    print(f"[prefill+decode] split={split} decoded {tuple(dec_lat.shape)} pcc vs golden[{split}:]: {pcc_msg}")
    return passed, pcc_msg


@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION}], indirect=True)
def test_gpt_prefill_pcc(device):
    passed, pcc_msg = run_prefill_pcc(device)
    assert passed, f"parallel-prefill decode PCC below {TARGET_PCC}: {pcc_msg}"


if __name__ == "__main__":
    import sys

    dev = ttnn.open_device(device_id=0, trace_region_size=TRACE_REGION)
    try:
        dev.enable_program_cache()
        ok, msg = run_prefill_pcc(dev)
    finally:
        ttnn.close_device(dev)
    print(("PASSED " if ok else "FAILED ") + str(msg))
    sys.exit(0 if ok else 1)
