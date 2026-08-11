# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end PCC test for the full TTNN conditioning branch (Block 1) vs the CPU reference:
mel_in [1,80,T] -> conditioning encoder -> Perceiver -> gpt_cond_latent [1,32,1024].

Input: deterministic synthetic voiced clip -> frontend.conditioning_mels. Reference:
reference/xtts_cond_ref.CondReference (validated PCC 1.0 vs coqui). Set XTTS_GOLDEN_DIR
to cross-check against stored coqui-captured fixtures instead."""
import torch
import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.xtts_v2.tests.reference_helpers import cond_reference
from models.experimental.xtts_v2.tt.ttnn_xtts_cond import (
    LATENTS,
    TTNNConditioningEncoder,
    TTNNPerceiver,
    preprocess_encoder_parameters,
    preprocess_perceiver_parameters,
)

TARGET_PCC = 0.999


def run_cond_pcc(device):
    ref = cond_reference()
    mel = ref["mel_in"]  # [1,80,T]
    gold = ref["gpt_cond_latent"]  # [1,32,1024]
    T = mel.shape[2]
    S = ((T + 31) // 32) * 32
    mel_f = torch.nn.functional.pad(mel.permute(0, 2, 1).contiguous(), (0, 0, 0, S - T))  # [1,S,80]

    enc = TTNNConditioningEncoder(device, preprocess_encoder_parameters(device, dtype=ttnn.float32), t_real=T, s_pad=S)
    perc = TTNNPerceiver(device, preprocess_perceiver_parameters(device, dtype=ttnn.float32))

    # perceiver key mask over [latents(32) + S] keys: -inf for padded frame positions
    km = torch.zeros(1, 1, 1, LATENTS + S)
    km[:, :, :, LATENTS + T :] = -1e9
    km_tt = ttnn.from_torch(km, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    mel_tt = ttnn.from_torch(mel_f, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    frames = enc(mel_tt)  # [1,S,1024]
    out = ttnn.to_torch(perc(frames, km_tt)).to(torch.float32)  # [1,32,1024]

    passed, msg = comp_pcc(gold, out, pcc=TARGET_PCC)
    print(f"gpt_cond_latent {tuple(out.shape)} vs golden {tuple(gold.shape)}  pcc: {msg}")
    return passed, msg


def test_cond_pcc(device):
    passed, msg = run_cond_pcc(device)
    assert passed, f"conditioning branch PCC below {TARGET_PCC}: {msg}"


if __name__ == "__main__":
    import sys

    dev = ttnn.open_device(device_id=0)
    try:
        dev.enable_program_cache()
        ok, msg = run_cond_pcc(dev)
    finally:
        ttnn.close_device(dev)
    print(("PASSED " if ok else "FAILED ") + str(msg))
    sys.exit(0 if ok else 1)
