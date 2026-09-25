# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""LM head vs a torch reference with the vocab shard layout the model uses (vocab column-parallel over
TP, 32768 per chip; sequence over SP), at the spec's default weight dtype. Also checks the top-1 token
per position, which is what a consumer of the prefill logits reads."""

import pytest
import torch

from models.demos.mistral_medium_3_5_128b.reference.model import random_linear
from models.demos.mistral_medium_3_5_128b.tt.lm_head import LMHead

from .common import CFG, assert_pcc, randn, residual_to_torch, spec_dtypes, to_full


@pytest.mark.timeout(900)
@pytest.mark.parametrize("seq_len", [1024, 256])
def test_lm_head_vs_ref(galaxy_mesh, mesh_config, seq_len):
    w = random_linear(CFG.vocab_size, CFG.hidden_size, torch.Generator().manual_seed(111))
    x = randn(1, 1, seq_len, CFG.hidden_size, seed=112)
    ref = (x @ w.t()).float()

    head = LMHead(galaxy_mesh, mesh_config, w, dtype=spec_dtypes()["weights_default"])
    out = residual_to_torch(head(to_full(x, galaxy_mesh, mesh_config)), galaxy_mesh, mesh_config)
    assert out.shape == (1, 1, seq_len, CFG.vocab_size)
    assert_pcc(f"lm_head[{seq_len}]", ref, out)
    agree = (out.argmax(-1) == ref.argmax(-1)).float().mean().item()
    assert agree >= 0.9, f"top-1 agreement {agree:.3f}"
