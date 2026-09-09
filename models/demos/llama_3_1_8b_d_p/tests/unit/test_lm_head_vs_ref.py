# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""LM-head projection vs a torch reference, in the vocab shard layout the model uses.

Target mesh (8, 4), random weights. Structure follows `gemma4/tests/unit/test_lm_head.py`.

Column-parallel over the vocab: each TP column produces `vocab/tp = 32064` logits from a full-emb
input, and the logits stay sharded — nothing in prefill needs them gathered.

Llama-3.1's 128256 vocab needs no padding (4008 tiles; 1002 tiles per TP column), which is worth
asserting rather than assuming: most LM heads pad first, and a checkpoint whose vocab is ragged would
silently mis-slice the last column here.
"""

import pytest
import torch

import ttnn
from models.demos.llama_3_1_8b_d_p.reference.model import REF_DTYPE
from models.demos.llama_3_1_8b_d_p.tt.lm_head import LMHead

from ..test_factory import ACT_DTYPE, WEIGHT_DTYPE, assert_pcc, parametrize_target_mesh

SEQ = 512


@parametrize_target_mesh()
def test_lm_head_vs_ref(mesh_device, device_params, config, hf_config, mesh_config, topology_name):
    """Logits vs a torch reference, with the vocab sharded across the TP columns."""
    torch.manual_seed(0)
    x = torch.randn(1, 1, SEQ, config.hidden_size, dtype=REF_DTYPE)
    weight = (torch.randn(config.vocab_size, config.hidden_size) * 0.02).to(REF_DTYPE)
    golden = torch.nn.functional.linear(x, weight)  # [1, 1, SEQ, vocab]

    head = LMHead(
        mesh_device,
        hf_config,
        {"weight": weight},
        mesh_config=mesh_config,
        weight_dtype=WEIGHT_DTYPE,
    )
    assert head.vocab_local == config.vocab_size // mesh_config.tp == 32064

    tt_x = ttnn.from_torch(
        x,
        device=mesh_device,
        dtype=ACT_DTYPE,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=[2, None]),
    )
    out = head(tt_x)
    ttnn.synchronize_device(mesh_device)

    # Sequence over the SP rows, vocab over the TP cols.
    got = ttnn.to_torch(
        out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 3), mesh_shape=tuple(mesh_device.shape))
    ).to(REF_DTYPE)
    assert got.shape[-1] == config.vocab_size, f"gathered vocab width {got.shape[-1]} != {config.vocab_size}"
    assert_pcc("lm_head", golden, got, topology_name)


@parametrize_target_mesh()
def test_lm_head_rejects_tied_embeddings(mesh_device, device_params, hf_config, mesh_config):
    """A config claiming tied embeddings must be rejected: this checkpoint's lm_head is separate."""
    hf_config.tie_word_embeddings = True
    with pytest.raises(AssertionError, match="tie_word_embeddings"):
        LMHead(mesh_device, hf_config, {}, mesh_config=mesh_config)


@parametrize_target_mesh()
def test_lm_head_rejects_unaligned_vocab(mesh_device, device_params, hf_config, mesh_config):
    """A vocab that does not tile- and shard-align must fail loudly rather than be mis-sliced."""
    hf_config.vocab_size = 128256 + 1
    with pytest.raises(AssertionError):
        LMHead(mesh_device, hf_config, {}, mesh_config=mesh_config)
