# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""PCC tests for the TTNN conditioning branch vs the CPU reference.

Three levels, so a failure localises itself: the conditioning encoder alone (mel_in -> enc_out),
the Perceiver resampler alone (fed the REFERENCE enc_out, so a fault there cannot be the
encoder's), and the two chained on device (mel_in -> gpt_cond_latent [1,32,1024] — the 32 rows the
GPT reads as "speak in this voice").

Input: a deterministic synthetic voiced clip through the mel front-end
(frontend.conditioning_mels). Reference: reference/xtts_cond_ref.CondReference, validated at PCC
1.0 against coqui activations during bringup. Set XTTS_GOLDEN_DIR to cross-check against stored
coqui-captured fixtures instead.

Run:
    pytest -svv models/experimental/xtts_v2/tests/pcc/test_cond_pcc.py
"""
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


def _s_pad(T):  # both modules want the sequence padded to a tile multiple (505 -> 512)
    return ((T + 31) // 32) * 32


def _frames(x, T, device):
    """[1,C,T] -> device [1,S,C], frames-major and tile-padded, as both modules take their input."""
    padded = torch.nn.functional.pad(x.permute(0, 2, 1).contiguous(), (0, 0, 0, _s_pad(T) - T))
    return ttnn.from_torch(padded, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)


def _key_mask(T, device):
    """Additive Perceiver key mask over [LATENTS + S] keys: -inf on the padded frame positions,
    so the resampler cannot attend to padding."""
    km = torch.zeros(1, 1, 1, LATENTS + _s_pad(T))
    km[:, :, :, LATENTS + T :] = -1e9
    return ttnn.from_torch(km, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)


def _encoder(device, T):
    params = preprocess_encoder_parameters(device, dtype=ttnn.float32)
    return TTNNConditioningEncoder(device, params, t_real=T, s_pad=_s_pad(T))


def _perceiver(device):
    return TTNNPerceiver(device, preprocess_perceiver_parameters(device, dtype=ttnn.float32))


def run_encoder_pcc(device):
    ref = cond_reference()
    mel, gold = ref["mel_in"], ref["enc_out"]  # [1,80,T] -> [1,1024,T]
    T = mel.shape[2]
    out = ttnn.to_torch(_encoder(device, T)(_frames(mel, T, device))).to(torch.float32)  # [1,S,1024]
    out = out[:, :T, :].permute(0, 2, 1).contiguous()  # back to [1,1024,T]
    passed, msg = comp_pcc(gold, out, pcc=TARGET_PCC)
    print(f"  encoder    enc_out         {tuple(out.shape)}  pcc: {msg}")
    return passed, msg


def run_perceiver_pcc(device):
    ref = cond_reference()
    enc, gold = ref["enc_out"], ref["perc_out"]  # [1,1024,T] -> [1,32,1024]
    T = enc.shape[2]
    out = ttnn.to_torch(_perceiver(device)(_frames(enc, T, device), _key_mask(T, device))).to(torch.float32)
    passed, msg = comp_pcc(gold, out, pcc=TARGET_PCC)
    print(f"  perceiver  perc_out        {tuple(out.shape)}  pcc: {msg}")
    return passed, msg


def run_cond_pcc(device):
    ref = cond_reference()
    mel, gold = ref["mel_in"], ref["gpt_cond_latent"]  # [1,80,T] -> [1,32,1024]
    T = mel.shape[2]
    frames = _encoder(device, T)(_frames(mel, T, device))  # stays on device between the two
    out = ttnn.to_torch(_perceiver(device)(frames, _key_mask(T, device))).to(torch.float32)
    passed, msg = comp_pcc(gold, out, pcc=TARGET_PCC)
    print(f"  chained    gpt_cond_latent {tuple(out.shape)}  pcc: {msg}")
    return passed, msg


def test_cond_encoder_pcc(device):
    passed, msg = run_encoder_pcc(device)
    assert passed, f"conditioning encoder PCC below {TARGET_PCC}: {msg}"


def test_cond_perceiver_pcc(device):
    passed, msg = run_perceiver_pcc(device)
    assert passed, f"perceiver PCC below {TARGET_PCC}: {msg}"


def test_cond_pcc(device):
    passed, msg = run_cond_pcc(device)
    assert passed, f"conditioning branch PCC below {TARGET_PCC}: {msg}"


if __name__ == "__main__":
    import sys

    dev = ttnn.open_device(device_id=0)
    try:
        dev.enable_program_cache()
        results = [run_encoder_pcc(dev), run_perceiver_pcc(dev), run_cond_pcc(dev)]
    finally:
        ttnn.close_device(dev)
    ok = all(r[0] for r in results)
    print(("PASSED " if ok else "FAILED ") + "; ".join(str(r[1]) for r in results))
    sys.exit(0 if ok else 1)
