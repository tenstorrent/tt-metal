# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC: :class:`~models.experimental.kokoro.tt.tt_custom_albert.TTCustomAlbert`
vs reference ``CustomAlbert`` (``transformers.AlbertModel`` returning
``last_hidden_state``)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AlbertConfig

_TT_METAL_ROOT = Path(__file__).resolve().parents[4]
if str(_TT_METAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_TT_METAL_ROOT))

import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.kokoro.reference.modules import CustomAlbert
from models.experimental.kokoro.tt import TTCustomAlbert, preprocess_tt_custom_albert


# Mirrors the Kokoro ``plbert`` block from ``config.json`` but shrunk for fast CI.
def _small_albert_config(vocab_size: int = 178) -> AlbertConfig:
    return AlbertConfig(
        vocab_size=vocab_size,
        embedding_size=128,
        hidden_size=256,
        num_attention_heads=8,
        intermediate_size=512,
        num_hidden_layers=2,
        num_hidden_groups=1,
        inner_group_num=1,
        max_position_embeddings=128,
        type_vocab_size=2,
        layer_norm_eps=1e-12,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        hidden_act="gelu_new",
        attn_implementation="eager",
    )


def _kokoro_albert_config() -> AlbertConfig:
    return AlbertConfig(
        vocab_size=178,
        embedding_size=128,
        hidden_size=768,
        num_attention_heads=12,
        intermediate_size=2048,
        num_hidden_layers=12,
        num_hidden_groups=1,
        inner_group_num=1,
        max_position_embeddings=512,
        type_vocab_size=2,
        layer_norm_eps=1e-12,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        hidden_act="gelu_new",
        attn_implementation="eager",
    )


def _run_pcc(
    device, *, cfg: AlbertConfig, B: int, T: int, seed: int, pad_from: int | None = None, threshold: float = 0.99
):
    torch.manual_seed(seed)

    ref = CustomAlbert(cfg).eval()
    params = preprocess_tt_custom_albert(ref, device)
    tt_mod = TTCustomAlbert(device, params)

    input_ids = torch.randint(1, cfg.vocab_size, (B, T), dtype=torch.long)
    attention_mask = torch.ones((B, T), dtype=torch.int32)
    if pad_from is not None:
        # Pad the tail of every row past ``pad_from``.
        attention_mask[:, pad_from:] = 0
        input_ids[:, pad_from:] = 0

    with torch.no_grad():
        ref_out = ref(input_ids, attention_mask=attention_mask)

    tt_out = tt_mod(input_ids, attention_mask=attention_mask)
    tt_torch = ttnn.to_torch(tt_out).float()
    ttnn.deallocate(tt_out)

    assert tt_torch.shape == ref_out.shape, (tt_torch.shape, ref_out.shape)

    if pad_from is not None:
        # Only valid positions are required to match — pad outputs are unobserved downstream.
        ref_view = ref_out[:, :pad_from, :]
        tt_view = tt_torch[:, :pad_from, :]
    else:
        ref_view = ref_out
        tt_view = tt_torch

    _, pcc = comp_pcc(ref_view, tt_view, pcc=0.0)
    print(f"TTCustomAlbert (B={B}, T={T}, pad_from={pad_from}, layers={cfg.num_hidden_layers}) PCC: {pcc:.6f}")
    assert pcc > threshold, f"PCC too low: {pcc}"


def test_tt_custom_albert_matches_torch_small(device):
    """Small ALBERT config, no padding — verifies the math end-to-end."""
    _run_pcc(device, cfg=_small_albert_config(), B=1, T=32, seed=0)


def test_tt_custom_albert_matches_torch_with_padding(device):
    """Same small config with attention mask covering only a prefix."""
    _run_pcc(device, cfg=_small_albert_config(), B=2, T=32, seed=1, pad_from=20)


def test_tt_custom_albert_matches_torch_kokoro_config(device):
    """Full Kokoro PLBERT shape (12 layers, hidden=768) with tile-aligned ``T``."""
    _run_pcc(device, cfg=_kokoro_albert_config(), B=1, T=64, seed=2)
