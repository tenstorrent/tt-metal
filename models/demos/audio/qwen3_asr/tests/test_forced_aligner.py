# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Qwen3-ForcedAligner-0.6B on TT against the qwen-asr package on CPU (fp32).

Golden files come from reference/dump_aligner_golden.py, one <name>.pt per clip. Run on
one N150:
  QWEN3ALIGNER_GOLDEN_DIR=/data/aligner_golden \
  pytest models/demos/audio/qwen3_asr/tests/test_forced_aligner.py
"""
import glob
import os
import time

import pytest
import torch

from models.demos.audio.qwen3_asr.tt.forced_aligner import Qwen3ForcedAligner
from tests.ttnn.utils_for_testing import comp_pcc

MIN_PCC = 0.97
MIN_BIN_MATCH = 0.9
MAX_MEAN_ABS_ERROR_MS = 40


def _goldens():
    root = os.environ.get("QWEN3ALIGNER_GOLDEN_DIR")
    files = sorted(glob.glob(os.path.join(root or "", "*.pt")))
    if not files:
        pytest.skip("set QWEN3ALIGNER_GOLDEN_DIR to the output of reference/dump_aligner_golden.py")
    return files


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_forced_aligner_matches_cpu(device):
    aligner = Qwen3ForcedAligner.build(device)
    for path in _goldens():
        g = torch.load(path)
        mel = g["input_features"][:, g["feature_attention_mask"].bool()]
        aligner.timestamp_logits(g["input_ids"], mel)
        start = time.perf_counter()
        logits = aligner.timestamp_logits(g["input_ids"], mel)
        warm_ms = (time.perf_counter() - start) * 1e3
        ref = g["timestamp_logits"]
        pcc = comp_pcc(ref, logits, MIN_PCC)[1]
        bins, ref_bins = logits.argmax(-1), ref.argmax(-1)
        match = (bins == ref_bins).float().mean().item()
        mae = ((bins - ref_bins).abs().float().mean() * g["timestamp_segment_time"]).item()
        print(
            f"{os.path.basename(path)}: {len(ref)} timestamps, PCC {pcc:.4f}, bin match {match:.3f}, MAE {mae:.1f} ms, {warm_ms:.0f} ms warm"
        )
        assert pcc >= MIN_PCC, f"{path}: PCC {pcc}"
        assert match >= MIN_BIN_MATCH, f"{path}: bin match {match}"
        assert mae <= MAX_MEAN_ABS_ERROR_MS, f"{path}: MAE {mae} ms"
