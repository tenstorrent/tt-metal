# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Prefill PCC of ``DeepSeekV41PrefillEngram`` against the checkpoint's ``Engram``, on real weights.

The device half of Engram -- ``wkv``, the gate and the residual write -- over a whole chunk of tokens. The
host half (token ids -> n-gram rows) is covered by ``test_engram_lookup.py`` and the decode test, so the rows
here are random bf16 fed to both sides (``GivenRows`` stands in for the reference's table lookup), which
spares the ~101 GB table. Random rows give gates over the whole (0, 1) range, so the output differs from the
input by O(1) and the PCC is not met by the residual alone.

Checked by PCC on the output and on what Engram adds to the streams (``out - x``), as the decode test does.

The converted ttnn weights are cached under ``CACHE_DIR`` (``$DEEPSEEK_V4_CACHE_DIR``, default ``../cache``, per checkpoint).

    pytest models/experimental/deepseek_v41_flash/tests/prefill/test_engram.py
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.deepseek_v41_flash.tests.reference import (
    GivenRows,
    load_reference,
    reference_engram_with_embed,
)
from models.experimental.deepseek_v41_flash.tt.config import CACHE_DIR, DEFAULT_MODEL_DIR, engram_weights, load_config
from models.experimental.deepseek_v41_flash.tt.prefill.engram import DeepSeekV41PrefillEngram
from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader, resolve_snapshot_dir

PCC = 0.995
DELTA_PCC = 0.98


def _snapshot():
    try:
        return resolve_snapshot_dir(DEFAULT_MODEL_DIR)
    except FileNotFoundError:
        return None


def _pcc(expected, actual, threshold, what):
    passing, pcc = comp_pcc(expected.reshape(-1), actual.reshape(-1), threshold)
    logger.info(f"{what}: PCC {pcc}")
    assert passing, f"{what}: PCC {pcc} < {threshold}"


@pytest.mark.skipif(_snapshot() is None, reason=f"V4.1-Flash checkpoint not found under {DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("seq_len", (128, 1000, 2048))  # one tile column of rows, a ragged length, a long chunk
@pytest.mark.parametrize("layer", (1, 14))
@torch.no_grad()
def test_engram_prefill(device, layer, seq_len):
    snapshot = _snapshot()
    config = load_config(snapshot)
    loader = DeepseekV4WeightLoader(snapshot)
    engram = DeepSeekV41PrefillEngram(
        config, layer, engram_weights(loader, layer), device, cache=WeightCache(CACHE_DIR).sub(f"layers.{layer}.engram")
    )

    torch.manual_seed(layer + seq_len)
    rows = torch.randn(1, seq_len, engram.kin).bfloat16().float()
    reference = reference_engram_with_embed(
        load_reference(snapshot), snapshot, loader, layer, GivenRows(rows, config.engram_head_dim)
    )
    streams = torch.randn(1, seq_len, config.hc_mult, config.hidden_size).bfloat16().float()
    expected = reference(streams, torch.zeros(1, seq_len, 1, dtype=torch.long))  # the hash ids are ignored

    def to_device(t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    out = ttnn.to_torch(engram(to_device(streams), to_device(rows.unsqueeze(0)))).float()
    assert out.shape == expected.shape, (tuple(out.shape), tuple(expected.shape))
    what = f"layer {layer} T {seq_len}"
    _pcc(expected, out, PCC, what)
    _pcc(expected - streams, out - streams, DELTA_PCC, f"{what} (out - x)")
