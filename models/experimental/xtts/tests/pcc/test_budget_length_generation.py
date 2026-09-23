# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The traced pipeline must vocode the longest generation a pass can produce.

``xtts_demo.py`` samples (temperature 0.65), so the number of codes it vocodes changes from run
to run, and a pass can spend its whole ``GENERATION.max_tokens`` budget without emitting STOP.
The scheduled e2e job crashed on 2026-09-16 at one sampled length: a HiFi-GAN residual ``conv1d``
on the interleaved path built a program whose static circular buffers clashed with the L1 buffers
the traced pipeline keeps alive around the vocoder ("Statically allocated circular buffers in
program 1429 clash with L1 buffers"). An exact-length sweep over budgets 150..239 reproduced it
at budgets 223, 224 and 225 (222-224 codes) and nowhere else. The perf test (188 codes) and the
empty-generation test never reach those lengths, so this pins the path deterministically with
``min_new_tokens == max_new_tokens`` at a clashing budget (the full budget, 239 codes, passes on main).
"""
import pytest
import torch
import ttnn

from models.experimental.xtts.config import L1_SMALL_SIZE, SESSION_TRACE_REGION
from models.experimental.xtts.tests.pcc.test_empty_generation import SAMPLING, _inputs, _max_seq


@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": L1_SMALL_SIZE, "trace_region_size": SESSION_TRACE_REGION}], indirect=True
)
def test_budget_length_generation_vocodes(device, xtts_state_dict, reset_seeds):
    """A pass that runs out a clashing code budget still vocodes through the traced path."""
    tt, wav, spk_tt, padded, real_len, pad_to = _inputs(device, xtts_state_dict)
    budget = 223  # 222 codes: the residual conv1d 32->32 k=7 d=5 at L=247296 clashes without the fallback
    wav_dev, codes, perf = tt.inference_fully_traced(
        padded,
        wav,
        spk_tt,
        _max_seq(pad_to, budget),
        max_new_tokens=budget,
        text_real_len=real_len,
        **dict(SAMPLING, min_new_tokens=budget),
    )
    # n codes need n+1 replayed steps, so a budget of N steps owns N-1 latents at most.
    assert codes.shape[1] >= budget - 1, f"budget {budget} produced only {codes.shape[1]} codes"
    wav_t = ttnn.to_torch(wav_dev).float()
    assert wav_t.shape[1] > 0, "vocoder returned empty audio for a full-budget generation"
    assert torch.isfinite(wav_t).all(), "vocoder output has non-finite samples"
    assert perf["vocoder_replay_s"] > 0.0, "vocoder trace did not run"
