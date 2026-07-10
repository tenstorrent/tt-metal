# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Devstral-style decoder-layer decode PCC helpers (layer 0, no prefill, random hiddens)."""

from __future__ import annotations

from typing import NamedTuple

import torch
from transformers.cache_utils import DynamicCache

from models.experimental.vibevoice.tests.pcc.lm_pcc_common import (
    DECODE_GENERATION_LENGTH,
    PCC_THRESHOLD,
    _TTVibeVoiceLMLayerProbe,
    _get_hf_reference_model,
    as_layer_probe,
    build_tt_lm,
    compare_decode_hidden_pcc,
    print_decode_pcc_summary,
)

DECODE_LAYER_IDX = 0
DECODE_BATCH_SIZE = 1


class DecoderLayerPccContext(NamedTuple):
    hidden_size: int
    hf_layer: torch.nn.Module
    hf_rotary_emb: torch.nn.Module
    tt_probe: _TTVibeVoiceLMLayerProbe


def build_decoder_layer_pcc_context(mesh_device, lm_state, vv_config) -> DecoderLayerPccContext:
    """Layer-0 fixtures for decode PCC (empty KV cache, positions 0 … N-1)."""
    cfg = vv_config.decoder
    model = _get_hf_reference_model(lm_state, vv_config)
    lm_tt = build_tt_lm(lm_state, mesh_device, cfg)
    return DecoderLayerPccContext(
        hidden_size=cfg.hidden_size,
        hf_layer=model.layers[DECODE_LAYER_IDX],
        hf_rotary_emb=model.rotary_emb,
        tt_probe=as_layer_probe(lm_tt),
    )


def reference_decoder_layer_decode_forward(
    layer: torch.nn.Module,
    rotary_emb: torch.nn.Module,
    hidden: torch.Tensor,
    *,
    position: int,
    cache: DynamicCache,
) -> torch.Tensor:
    """Single decode step on HF ``Qwen2DecoderLayer`` with ``DynamicCache``."""
    pos = torch.tensor([[position]], dtype=torch.long, device=hidden.device)
    cache_position = torch.tensor([position], dtype=torch.long, device=hidden.device)
    cos, sin = rotary_emb(hidden, pos)
    with torch.no_grad():
        # HF decoder layers return a tuple (hidden_states, ...); cache kwarg is singular.
        out = layer(
            hidden,
            position_ids=pos,
            past_key_value=cache,
            use_cache=True,
            cache_position=cache_position,
            position_embeddings=(cos, sin),
        )
    hidden_out = out[0] if isinstance(out, (tuple, list)) else out
    if hidden_out.dim() == 2:
        hidden_out = hidden_out.unsqueeze(1)
    return hidden_out.float()


def tt_decoder_layer_decode_forward(
    probe: _TTVibeVoiceLMLayerProbe,
    hidden: torch.Tensor,
    *,
    position: int,
    kv_cache,
    layer_idx: int = DECODE_LAYER_IDX,
) -> torch.Tensor:
    """Single decode step on TT decoder layer (``hidden`` [B, 1, H] bf16)."""
    return probe.forward_decoder_layer_hidden(hidden, position, kv_cache, layer_idx=layer_idx)


def run_decoder_layer_decode_pcc_sweep(
    mesh_device,
    lm_state,
    vv_config,
    *,
    num_steps: int = DECODE_GENERATION_LENGTH,
) -> list[float]:
    """10 decode steps at positions 0–9 with random hidden states and empty KV cache."""
    ctx = build_decoder_layer_pcc_context(mesh_device, lm_state, vv_config)
    hf_cache = DynamicCache()
    kv_cache = ctx.tt_probe.alloc_kv_cache(num_steps + 8)

    failures: list[str] = []
    step_pccs: list[float] = []

    print(
        f"[decoder layer decode PCC] layer={DECODE_LAYER_IDX} batch={DECODE_BATCH_SIZE} "
        f"hidden={ctx.hidden_size} steps={num_steps} positions=0–{num_steps - 1} "
        f"(no prefill, random hiddens)"
    )

    for step in range(num_steps):
        hidden = (torch.rand(DECODE_BATCH_SIZE, 1, ctx.hidden_size, dtype=torch.bfloat16) * 2) - 1

        ref_out = reference_decoder_layer_decode_forward(
            ctx.hf_layer,
            ctx.hf_rotary_emb,
            hidden,
            position=step,
            cache=hf_cache,
        )
        tt_out = tt_decoder_layer_decode_forward(
            ctx.tt_probe,
            hidden,
            position=step,
            kv_cache=kv_cache,
        )

        passed_d, pcc_d = compare_decode_hidden_pcc(ref_out, tt_out)
        step_pccs.append(pcc_d)
        print(f"Decode step {step}  position={step}  PCC={pcc_d:.5f}")

        if not passed_d:
            failures.append(f"decode step={step} position={step} measured_pcc={pcc_d:.6f} threshold={PCC_THRESHOLD}")

    print_decode_pcc_summary(step_pccs)
    if failures:
        raise AssertionError("Decoder layer decode PCC below threshold:\n" + "\n".join(failures))

    return step_pccs
