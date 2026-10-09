# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""On-device token embedding vs HF ``embed_tokens``, real table, real prompt ids."""

import pytest
import torch

import ttnn
from models.demos.pplx_decider_v1_27b.reference import hf_reference as ref
from models.demos.pplx_decider_v1_27b.tests.test_utils import (
    DEVICE_PARAMS,
    SEQ_LENS,
    build_optimizations,
    check_pcc,
    golden_tensor,
    reader,
    seq_ids,
    to_host,
)
from models.demos.pplx_decider_v1_27b.tt.embedding import PplxEmbedding
from models.demos.pplx_decider_v1_27b.tt.weight_adapter import build_embedding_weight


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("seq_len", SEQ_LENS, ids=seq_ids())
def test_embedding(device, seq_len):
    hf = ref.build_embedding(reader(), dtype=torch.bfloat16)
    ids = golden_tensor("ids", seq_len)
    with torch.no_grad():
        expected = hf(ids).float()
    tt = PplxEmbedding(
        build_embedding_weight(hf.weight.detach(), build_optimizations(device).policy), mesh_device=device
    )
    tt_ids = ttnn.from_torch(ids.to(torch.int32), device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    out = to_host(tt(tt_ids), expected.shape)
    check_pcc(expected, out, module="embedding", seq_len=seq_len)
    assert torch.equal(out, expected), f"S{seq_len}: a BF16 row gather must be exact"
