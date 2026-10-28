# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""D3 R1-1: the folded q @ R / k @ R weight columns, as the model stores them on the device, are bitwise the permuted
and negated stored q / k columns, in both presets (bf16 and bfp8), so RoPE stays order-only."""
import pytest
import torch

import ttnn
from models.experimental.kairos_forecast.reference import dataflow as DF
from models.experimental.kairos_forecast.tt.common import Precision
from models.experimental.kairos_forecast.tt.model import D, TtKairos, rotate_cols


def _bits(t):
    return t.view({torch.bfloat16: torch.int16, torch.float32: torch.int32}[t.dtype])


def _groups(w):
    return ttnn.to_torch(w).float().reshape(-1, w.shape[-1]).split(D, -1)


@pytest.mark.parametrize("prec", ["accuracy", "performance"])
def test_folded_columns_bitwise(device, prec):
    m = TtKairos(device, DF.load_weights(), getattr(Precision, prec)(), layers=(1, 1))
    enc, dec = m.enc[0], m.dec[0]
    pairs = []
    for blk in (enc, dec):
        q, k, v = _groups(blk["qkv"])
        qr, kr, vr = _groups(blk["qkvr"])
        pairs += [(qr, rotate_cols(q)), (kr, rotate_cols(k)), (vr, v)]
    cq, cqr, cq2 = _groups(dec["q"])
    ck, cv, ckr = _groups(dec["kv"])
    pairs += [(cqr, rotate_cols(cq)), (cq2, cq), (ckr, rotate_cols(ck))]
    for i, (got, exp) in enumerate(pairs):
        assert torch.equal(_bits(got), _bits(exp)), (prec, i)
    assert enc["qkvr"].dtype == enc["qkv"].dtype and dec["kv"].dtype == m.dec[0]["qkv"].dtype
