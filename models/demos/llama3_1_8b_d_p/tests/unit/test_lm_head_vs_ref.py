# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PCC: final norm + lm_head projection vs the torch reference, with the vocab shard layout used.

The lm_head weight is padded from 128256 to ``per_device_pow2 * tp`` (32768 * 4 = 131072) BEFORE the
column-parallel shard, so each chip owns a contiguous 32768-wide vocab slice and all the padding
lands in the tail of the LAST chip. This test reassembles those slices and asserts the first 128256
columns are the real logits in order, and that the tail is zero.

A padding mistake here is completely invisible in a KV-only prefill run — prefill never reads logits
— and catastrophic the moment a decode stage does. It is tested against ``load_lm_head_weight``, the
same function ``Model`` calls, so the test cannot drift from the model.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.llama3_1_8b_d_p.reference.model import LlamaRMSNorm
from models.demos.llama3_1_8b_d_p.tt.model import compute_per_device_vocab, load_lm_head_weight, padded_vocab_size
from models.demos.llama3_1_8b_d_p.tt.rms_norm import RMSNorm

from ..test_factory import llama_config, make_mesh_config, parametrize_mesh_with_fabric, shard_seq_on_sp

PCC = 0.98  # bf8 lm_head weights over a 128k vocab, against an fp32 reference


def test_per_device_vocab_padding_math():
    """Host-only: the padded width is a power of 2 and covers the real vocab."""
    cfg = llama_config()
    per_dev = compute_per_device_vocab(cfg.vocab_size, 4)
    assert per_dev == 32768, per_dev
    assert per_dev & (per_dev - 1) == 0, "topk's multi-core path needs a power-of-2 width"
    assert padded_vocab_size(cfg.vocab_size, 4) == 131072
    assert padded_vocab_size(cfg.vocab_size, 4) >= cfg.vocab_size


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)])
@pytest.mark.parametrize("seq_len", [512], ids=["s512"])
def test_lm_head_vs_ref(mesh_device, device_params, seq_len, reset_seeds):
    cfg = llama_config()
    mesh_config = make_mesh_config(mesh_device)
    rows, cols = tuple(mesh_device.shape)
    tp = mesh_config.tp

    x = torch.randn(1, 1, seq_len, cfg.hidden_size) * 0.1
    norm_w = torch.randn(cfg.hidden_size) * 0.1
    lm_w = torch.randn(cfg.vocab_size, cfg.hidden_size, dtype=torch.bfloat16) * 0.02

    ref_norm = LlamaRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
    ref_norm.weight.data = norm_w.clone()
    reference = ref_norm(x).float() @ lm_w.float().t()  # [1, 1, S, vocab]

    tt_norm = RMSNorm(mesh_device, cfg, {"weight": norm_w}, tensor_cache_path=None, mesh_config=mesh_config)
    tt_lm = load_lm_head_weight(
        mesh_device, lm_w, vocab_size=cfg.vocab_size, mesh_config=mesh_config, dtype=ttnn.bfloat8_b
    )

    x_tt = shard_seq_on_sp(mesh_device, x, mesh_config)
    logits_tt = ttnn.matmul(tt_norm(x_tt), tt_lm, dtype=ttnn.bfloat16)

    # Reassemble: SP row r holds seq shard r; TP col c holds vocab slice c.
    dts = ttnn.get_device_tensors(logits_tt)
    per_row = []
    for r in range(rows):
        per_row.append(torch.cat([ttnn.to_torch(dts[r * cols + c]).float() for c in range(cols)], dim=-1))
    full = torch.cat(per_row, dim=2)  # [1, 1, S, padded_vocab]
    assert full.shape[-1] == padded_vocab_size(cfg.vocab_size, tp), full.shape

    tail = full[..., cfg.vocab_size :]
    assert tail.abs().max() == 0, "lm_head padding must be exactly zero and live only in the tail"

    passing, pcc = comp_pcc(reference, full[..., : cfg.vocab_size], PCC)
    logger.info(f"lm_head s={seq_len}: {pcc}")
    assert passing, f"PCC fail: {pcc}"
