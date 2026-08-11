# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""PCC test for the TTNN conditioning encoder (Block 1, first half) vs the CPU reference.

Input: a deterministic synthetic voiced clip pushed through the coqui-transcribed mel
front-end (frontend.conditioning_mels) -> mel_in [1,80,T]. Reference: reference/
xtts_cond_ref.CondReference (validated PCC 1.0 vs coqui) -> enc_out [1,1024,T].
Set XTTS_GOLDEN_DIR to cross-check against stored coqui-captured fixtures instead."""
import torch
import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.xtts_v2.tests.reference_helpers import cond_reference
from models.experimental.xtts_v2.tt.ttnn_xtts_cond import TTNNConditioningEncoder, preprocess_encoder_parameters

TARGET_PCC = 0.999


def run_encoder_pcc(device):
    ref = cond_reference()
    mel = ref["mel_in"]  # [1,80,T]
    enc_g = ref["enc_out"]  # [1,1024,T]
    T = mel.shape[2]
    S = ((T + 31) // 32) * 32  # 505 -> 512
    mel_f = mel.permute(0, 2, 1).contiguous()  # [1,T,80]
    mel_f = torch.nn.functional.pad(mel_f, (0, 0, 0, S - T))  # [1,S,80]

    params = preprocess_encoder_parameters(device, dtype=ttnn.float32)
    enc = TTNNConditioningEncoder(device, params, t_real=T, s_pad=S)
    mel_tt = ttnn.from_torch(mel_f, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(enc(mel_tt)).to(torch.float32)  # [1,S,1024]
    out = out[:, :T, :].permute(0, 2, 1).contiguous()  # [1,1024,T]

    passed, msg = comp_pcc(enc_g, out, pcc=TARGET_PCC)
    print(f"enc_out {tuple(out.shape)} vs golden {tuple(enc_g.shape)}  pcc: {msg}")
    return passed, msg


def test_cond_encoder_pcc(device):
    passed, msg = run_encoder_pcc(device)
    assert passed, f"conditioning encoder PCC below {TARGET_PCC}: {msg}"


if __name__ == "__main__":
    import sys

    dev = ttnn.open_device(device_id=0)
    try:
        dev.enable_program_cache()
        ok, msg = run_encoder_pcc(dev)
    finally:
        ttnn.close_device(dev)
    print(("PASSED " if ok else "FAILED ") + str(msg))
    sys.exit(0 if ok else 1)
