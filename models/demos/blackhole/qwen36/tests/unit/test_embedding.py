# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Component PCC: token embedding (the framework ``tt_transformers`` Embedding the
model actually uses — see tt/model.py) vs a torch reference, with a random table.

Uses the bespoke single-device ``device`` fixture from tests/unit/conftest.py.
"""
import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.tests.test_factory import compute_pcc, get_pcc_threshold
from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs

from .conftest import DEVICE_PARAMS

pytestmark = [run_for_blackhole(), pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)]


def test_embedding_pcc(device, request):
    from models.tt_transformers.tt.embedding import Embedding

    args = Qwen36ModelArgs(mesh_device=device, max_batch_size=1, max_seq_len=2048)
    args.dummy_weights = True  # random table below; lets Embedding skip the weight cache path
    vocab, dim = args.vocab_size, args.dim
    table = torch.randn(vocab, dim, dtype=torch.bfloat16)
    sd = {"tok_embeddings.weight": table}
    emb = Embedding(mesh_device=device, args=args, weight_cache_path=None, state_dict=sd, dtype=ttnn.bfloat16)
    ids = torch.tensor([[1, 5, 9, 13]], dtype=torch.int32)
    ref = torch.nn.functional.embedding(ids.long(), table.float())
    ids_tt = ttnn.from_torch(ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    out = ttnn.to_torch(emb(ids_tt)).float()[..., :dim].reshape(ref.shape)
    pcc = compute_pcc(ref, out)
    logger.info(f"Embedding PCC: {pcc:.6f}")
    assert pcc > get_pcc_threshold(request), f"Embedding PCC too low: {pcc}"
