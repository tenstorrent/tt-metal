# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""PCC test for the SeamlessM4T-v2 TTNN ``CodeHifiGanVocoder``.

Loads the golden tensors from
``models/demos/facebook_seamless_m4t_v2_large/reference/golden/code_hifigan_vocoder.pt``,
runs the TTNN ``CodeHifiGanVocoder`` on the open p150 (blackhole) device,
and asserts PCC > 0.99 against the saved reference output.

Can also be run as a standalone script which opens its own device, runs the
PCC comparison, prints a single-line JSON, and exits 0 on pass / 1 on fail.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.facebook_seamless_m4t_v2_large.tt.code_hifigan_vocoder import CodeHifiGanVocoder

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "reference" / "golden" / "code_hifigan_vocoder.pt"


def _pcc_from_message(passing: bool, pcc_message) -> float:
    msg_str = str(pcc_message).strip()
    try:
        return float(msg_str)
    except ValueError:
        match = re.search(r"-?\d+\.\d+(?:[eE][+-]?\d+)?", msg_str)
        return float(match.group(0)) if match else float("nan")


def _run_pcc(device) -> float:
    golden = torch.load(GOLDEN_PATH, weights_only=False)
    input_ids: torch.Tensor = golden["input_ids"]
    speaker_id: torch.Tensor = golden["speaker_id"]
    lang_id: torch.Tensor = golden["lang_id"]
    sd = golden["state_dict"]
    ref_waveform: torch.Tensor = golden["waveform"]  # [B, T_out] float32
    ref_lengths = golden["lengths"]
    ref_dur_out: torch.Tensor = golden["dur_out"]
    cfg = golden.get("config", {})

    pad_token_id = int(cfg.get("t2u_pad_token_id", 1))
    kernel_size = int(cfg.get("variance_predictor_kernel_size", 3))
    upsample_rates = tuple(cfg.get("upsample_rates", (5, 4, 4, 2, 2)))
    upsample_kernel_sizes = tuple(cfg.get("upsample_kernel_sizes", (11, 8, 8, 4, 4)))
    resblock_kernel_sizes = tuple(cfg.get("resblock_kernel_sizes", (3, 7, 11)))
    resblock_dilation_sizes = tuple(tuple(d) for d in cfg.get("resblock_dilation_sizes", ((1, 3, 5),) * 3))
    leaky_relu_slope = float(cfg.get("leaky_relu_slope", 0.1))

    tt_block = CodeHifiGanVocoder(
        device=device,
        state_dict=sd,
        pad_token_id=pad_token_id,
        variance_predictor_kernel_size=kernel_size,
        upsample_rates=upsample_rates,
        upsample_kernel_sizes=upsample_kernel_sizes,
        resblock_kernel_sizes=resblock_kernel_sizes,
        resblock_dilation_sizes=resblock_dilation_sizes,
        leaky_relu_slope=leaky_relu_slope,
        weight_dtype=ttnn.bfloat16,
    )

    tt_waveform = tt_block(input_ids=input_ids, speaker_id=speaker_id, lang_id=lang_id)
    tt_waveform_torch = ttnn.to_torch(tt_waveform).to(torch.float32).reshape(ref_waveform.shape)

    # Cross-check dur_out matches reference (integer must match exactly).
    dur_out = tt_block.last_dur_out
    assert dur_out is not None and torch.equal(
        dur_out.long(), ref_dur_out.long()
    ), f"dur_out mismatch: tt={dur_out.tolist()} ref={ref_dur_out.tolist()}"
    # And lengths.
    tt_lengths = tt_block.last_lengths
    assert torch.equal(
        tt_lengths.long().reshape(-1), torch.as_tensor(ref_lengths).long().reshape(-1)
    ), f"lengths mismatch: tt={tt_lengths} ref={ref_lengths}"

    passing, pcc_message = comp_pcc(ref_waveform, tt_waveform_torch, 0.99)
    print(f"comp_pcc(code_hifigan_vocoder): passing={passing}, message={pcc_message}")
    print(
        f"code_hifigan_vocoder shapes: ref={tuple(ref_waveform.shape)} tt={tuple(tt_waveform_torch.shape)} "
        f"max_abs_diff={(ref_waveform - tt_waveform_torch).abs().max().item():.3e}"
    )
    return _pcc_from_message(passing, pcc_message)


def test_tt_code_hifigan_vocoder():
    device = ttnn.open_device(device_id=0, l1_small_size=32768)
    try:
        pcc = _run_pcc(device)
    finally:
        ttnn.close_device(device)
    assert pcc > 0.99, f"PCC {pcc} <= 0.99"


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0, l1_small_size=32768)
    pcc = float("nan")
    try:
        pcc = _run_pcc(device)
    finally:
        ttnn.close_device(device)
    result = {
        "block": "code_hifigan_vocoder",
        "pcc": pcc,
        "passed": pcc > 0.99,
    }
    print(json.dumps(result))
    sys.exit(0 if result["passed"] else 1)
