# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PCC test for the TTNN Perceiver resampler (tail of Block 1) vs coqui golden.
Feeds the golden conditioning-encoder output (enc_out) and checks perc_out."""
import os

import torch
import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.xtts_v2.tt.ttnn_xtts_cond import TTNNPerceiver, preprocess_perceiver_parameters

GOLDEN = os.path.join(os.path.dirname(__file__), "..", "golden", "cond")
TARGET_PCC = 0.999


def run_perceiver_pcc(device):
    enc = torch.load(os.path.join(GOLDEN, "enc_out.pt"))  # [1,1024,T]
    perc_g = torch.load(os.path.join(GOLDEN, "perc_out.pt"))  # [1,32,1024]
    frames = enc.permute(0, 2, 1).contiguous()  # [1,T,1024]
    T = frames.shape[1]
    S_pad = ((T + 31) // 32) * 32  # 505 -> 512
    frames = torch.nn.functional.pad(frames, (0, 0, 0, S_pad - T))  # pad seq to tile mult

    # additive key mask over [latents(32) + S_pad] keys: 0 for valid, -inf for padded frames
    ctx = 32 + S_pad
    mask = torch.zeros(1, 1, 1, ctx)
    mask[:, :, :, 32 + T :] = -1e9  # padded frame keys

    params = preprocess_perceiver_parameters(device, dtype=ttnn.float32)
    model = TTNNPerceiver(device, params)
    frames_tt = ttnn.from_torch(frames, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    mask_tt = ttnn.from_torch(mask, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    out = ttnn.to_torch(model(frames_tt, mask_tt)).to(torch.float32)
    passed, msg = comp_pcc(perc_g, out, pcc=TARGET_PCC)
    print(f"perceiver out {tuple(out.shape)} vs golden {tuple(perc_g.shape)}  pcc: {msg}")
    return passed, msg


def test_cond_perceiver_pcc(device):
    passed, msg = run_perceiver_pcc(device)
    assert passed, f"perceiver PCC below {TARGET_PCC}: {msg}"


if __name__ == "__main__":
    import sys

    dev = ttnn.open_device(device_id=0)
    try:
        dev.enable_program_cache()
        ok, msg = run_perceiver_pcc(dev)
    finally:
        ttnn.close_device(dev)
    print(("PASSED " if ok else "FAILED ") + str(msg))
    sys.exit(0 if ok else 1)
