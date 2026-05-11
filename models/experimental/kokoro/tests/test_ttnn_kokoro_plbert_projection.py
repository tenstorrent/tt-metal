# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC: TTNN `bert_encoder` projection vs full PyTorch reference."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_TT_METAL_ROOT = Path(__file__).resolve().parents[4]
if str(_TT_METAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_TT_METAL_ROOT))

from models.common.utility_functions import comp_pcc
from models.experimental.kokoro.reference import KokoroConfig, load_plbert_from_huggingface
from models.experimental.kokoro.tt import TtKokoroPlBert

from loguru import logger


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_plbert_projection_matches_torch(mesh_device):
    """Full PL-BERT on device — ``d_en`` should match reference PCC."""
    torch_ref = load_plbert_from_huggingface(repo_id=KokoroConfig.repo_id, device="cpu")
    tt_plbert = TtKokoroPlBert(mesh_device, torch_ref)

    torch.manual_seed(0)
    input_ids = torch.randint(0, 50, (1, 32), dtype=torch.long)

    ref_out = torch_ref(input_ids)
    tt_out = tt_plbert(input_ids)

    assert ref_out.d_en.shape == tt_out.d_en.shape
    ok, reported_pcc = comp_pcc(ref_out.d_en, tt_out.d_en, pcc=0.94)
    logger.info(f"d_en PCC: {reported_pcc}")
    assert ok, f"d_en PCC too low: {reported_pcc}"
