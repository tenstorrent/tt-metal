# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for Devstral-2 decoder-layer PCC tests (prefill + decode)."""

from __future__ import annotations

import os
from typing import NamedTuple

import torch
from loguru import logger
from transformers.cache_utils import DynamicCache
from transformers.models.ministral3.modeling_ministral3 import (
    Ministral3DecoderLayer,
    Ministral3RotaryEmbedding,
)

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.devstral2_123B_instruct.tests._devstral_weights import (
    decode_tt_to_torch,
    devstral2_weight_cache_seq_len,
    load_ministral3_decoder_layer_weights,
    replicated_tt_to_torch,
    require_layer_weights,
    require_text_config,
    resolve_devstral2_weight_cache_path,
)
from models.experimental.devstral2_123B_instruct.tt.mem_config import pad_to_tile
from models.experimental.devstral2_123B_instruct.tt.model_args import (
    DEVSTRAL2_LARGE_L1_SMALL_SIZE,
    Devstral2Args,
)
from models.experimental.devstral2_123B_instruct.tt.tt_ministral3_decoder_layer import TtDecoderLayer
from models.experimental.devstral2_123B_instruct.tt.tt_ministral_rotary_emb import TtRotaryEmbedding
from models.tt_transformers.tt.ccl import TT_CCL

PCC_REQUIRED = 0.99
DECODE_GENERATION_LENGTH = 10
PCC_BATCH_SIZE = 1
LAYER_IDX = 0

# Prefill input length sweep (powers of two 32 … 256K). Batch size is fixed at ``PCC_BATCH_SIZE``.
PREFILL_SWEEP_SEQ_LENGTHS = [
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
]

PREFILL_SANITY_SEQ_LENGTHS = [32, 128]


def pcc_layer_max_seq_len() -> int:
    """KV / RoPE budget and on-disk weight cache key (``seq_{N}``). Matches demo / ``text_demo.py``."""
    return devstral2_weight_cache_seq_len()


def mesh_device_param() -> tuple[int, int]:
    return {
        "N150": (1, 1),
        "N300": (1, 2),
        "N150x4": (1, 4),
        "P150x4": (1, 4),
        "T3K": (1, 8),
        "TG": (8, 4),
    }.get(os.environ.get("MESH_DEVICE"), (1, 8))


def device_params() -> dict:
    return {
        "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        "l1_small_size": DEVSTRAL2_LARGE_L1_SMALL_SIZE,
    }


def log_pcc_step(label: str, passing: bool, pcc_message: str) -> None:
    if passing:
        logger.info(f"{label}: PASS — {pcc_message}")
    else:
        logger.warning(f"{label}: FAIL — {pcc_message}")


def build_causal_attn_mask(seq_len: int, *, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    return torch.triu(torch.full((seq_len, seq_len), float("-inf"), dtype=dtype), diagonal=1).reshape(
        1, 1, seq_len, seq_len
    )


class DecodePccContext(NamedTuple):
    text_cfg: object
    args: Devstral2Args
    ref: Ministral3DecoderLayer
    ref_rope: Ministral3RotaryEmbedding
    tt_layer: TtDecoderLayer


class PrefillPccContext(NamedTuple):
    text_cfg: object
    args: Devstral2Args
    ref: Ministral3DecoderLayer
    ref_rope: Ministral3RotaryEmbedding
    tt_layer: TtDecoderLayer


def _build_decoder_pcc_fixtures(
    mesh_device,
) -> tuple[object, Devstral2Args, Ministral3DecoderLayer, Ministral3RotaryEmbedding, TtDecoderLayer]:
    """Layer 0 fixtures with fixed ``max_seq_len`` and shared on-disk weight cache (``seq_262144``)."""
    text_cfg = require_text_config()
    state_dict = require_layer_weights(LAYER_IDX)
    layer_max_seq_len = pcc_layer_max_seq_len()
    weight_cache_path = resolve_devstral2_weight_cache_path(
        mesh_device, text_cfg, num_layers=int(text_cfg.num_hidden_layers)
    )

    ref = Ministral3DecoderLayer(text_cfg, layer_idx=LAYER_IDX).to(torch.bfloat16).eval()
    load_ministral3_decoder_layer_weights(ref, state_dict, LAYER_IDX)
    ref_rope = Ministral3RotaryEmbedding(text_cfg).eval()

    args = Devstral2Args.from_hf_config(
        text_cfg,
        mesh_shape=tuple(mesh_device.shape),
        max_seq_len=layer_max_seq_len,
        max_batch_size=PCC_BATCH_SIZE,
    )
    tt_ccl = TT_CCL(mesh_device)
    rope = TtRotaryEmbedding(
        args,
        mesh_device,
        max_position_embeddings=layer_max_seq_len,
        weight_cache_path=weight_cache_path,
    )
    tt_layer = TtDecoderLayer(
        args,
        mesh_device,
        state_dict,
        layer_idx=LAYER_IDX,
        tt_ccl=tt_ccl,
        rotary_emb=rope,
        weight_cache_path=weight_cache_path,
    )
    logger.info(
        f"Decoder PCC fixtures: layer_max_seq_len={layer_max_seq_len}, " f"weight_cache_path={weight_cache_path}"
    )
    return text_cfg, args, ref, ref_rope, tt_layer


def build_decode_pcc_context(mesh_device) -> DecodePccContext:
    """Build decode fixtures; reuses ``seq_{pcc_layer_max_seq_len()}`` weight cache."""
    text_cfg, args, ref, ref_rope, tt_layer = _build_decoder_pcc_fixtures(mesh_device)
    return DecodePccContext(text_cfg=text_cfg, args=args, ref=ref, ref_rope=ref_rope, tt_layer=tt_layer)


def build_prefill_pcc_context(mesh_device) -> PrefillPccContext:
    """Build prefill fixtures; input ``seq_len`` is swept separately in the test loop."""
    text_cfg, args, ref, ref_rope, tt_layer = _build_decoder_pcc_fixtures(mesh_device)
    return PrefillPccContext(text_cfg=text_cfg, args=args, ref=ref, ref_rope=ref_rope, tt_layer=tt_layer)


def hidden_to_tt_prefill(x: torch.Tensor, mesh_device) -> ttnn.Tensor:
    """``x``: ``[batch, seq, hidden]`` → TT ``(1, 1, seq, hidden)``."""
    batch, seq_len, hidden = x.shape
    return ttnn.from_torch(
        x.reshape(1, 1, seq_len, hidden),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def hidden_to_tt_decode(x: torch.Tensor, mesh_device, *, batch_size: int = PCC_BATCH_SIZE) -> ttnn.Tensor:
    """``x``: ``[batch, 1, hidden]`` → TT decode layout with tile-padded height."""
    assert batch_size == PCC_BATCH_SIZE, f"PCC tests only support batch_size={PCC_BATCH_SIZE}, got {batch_size}"
    hidden = x.shape[-1]
    tile_h = pad_to_tile(batch_size)
    x4 = torch.zeros(1, 1, tile_h, hidden, dtype=x.dtype)
    x4[0, 0, :batch_size] = x[0, 0]
    return ttnn.from_torch(
        x4,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def current_pos_to_tt(positions: torch.Tensor, mesh_device) -> ttnn.Tensor:
    pos = positions.reshape(-1).to(torch.int32)
    return ttnn.from_torch(
        pos,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def hf_decode_forward(
    ref: Ministral3DecoderLayer,
    ref_rope: Ministral3RotaryEmbedding,
    hidden: torch.Tensor,
    *,
    position: int,
    cache: DynamicCache,
) -> torch.Tensor:
    """Single decode step on HF layer 0 with ``DynamicCache``."""
    pos = torch.tensor([[position]], dtype=torch.long)
    cos, sin = ref_rope(hidden.unsqueeze(1).float(), pos)
    out = ref(
        hidden,
        position_ids=pos,
        position_embeddings=(cos.to(torch.bfloat16), sin.to(torch.bfloat16)),
        past_key_values=cache,
        use_cache=True,
    )
    if out.dim() == 2:
        out = out.unsqueeze(1)
    return out


def hf_prefill_forward(
    ref: Ministral3DecoderLayer,
    ref_rope: Ministral3RotaryEmbedding,
    hidden: torch.Tensor,
) -> torch.Tensor:
    """Full-sequence prefill on HF layer 0 (short sequences only; use chunked path for long seq)."""
    seq_len = hidden.shape[1]
    positions = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    cos, sin = ref_rope(hidden.unsqueeze(1).float(), positions)
    causal_mask = build_causal_attn_mask(seq_len)
    out = ref(
        hidden,
        attention_mask=causal_mask,
        position_ids=positions,
        position_embeddings=(cos.to(torch.bfloat16), sin.to(torch.bfloat16)),
    )
    return out


def hf_prefill_forward_chunk(
    ref: Ministral3DecoderLayer,
    ref_rope: Ministral3RotaryEmbedding,
    hidden: torch.Tensor,
    *,
    chunk_start: int,
    cache: DynamicCache,
) -> torch.Tensor:
    """One prefill chunk on HF layer 0; ``cache`` accumulates K/V across chunks."""
    chunk_len = hidden.shape[1]
    positions = torch.arange(chunk_start, chunk_start + chunk_len, dtype=torch.long).unsqueeze(0)
    cos, sin = ref_rope(hidden.unsqueeze(1).float(), positions)
    rope_emb = (cos.to(torch.bfloat16), sin.to(torch.bfloat16))
    if chunk_start == 0:
        out = ref(
            hidden,
            attention_mask=build_causal_attn_mask(chunk_len),
            position_ids=positions,
            position_embeddings=rope_emb,
            past_key_values=cache,
            use_cache=True,
        )
    else:
        out = ref(
            hidden,
            position_ids=positions,
            position_embeddings=rope_emb,
            past_key_values=cache,
            use_cache=True,
        )
    return out


def tt_prefill_forward_chunk(
    tt_layer: TtDecoderLayer,
    mesh_device,
    hidden: torch.Tensor,
    *,
    chunk_start: int,
    block: int,
    hidden_size: int,
) -> torch.Tensor:
    """One TT prefill chunk; pads partial blocks to ``block`` (matches ``tt_prefill_forward``)."""
    chunk_len = hidden.shape[1]
    chunk = hidden
    if chunk_len < block:
        pad = torch.zeros(1, block - chunk_len, hidden_size, dtype=chunk.dtype)
        chunk = torch.cat([chunk, pad], dim=1)

    tt_in = hidden_to_tt_prefill(chunk, mesh_device)
    tt_out = tt_layer(tt_in, mode="prefill", start_pos=chunk_start)
    tt_torch = replicated_tt_to_torch(tt_out, reshape=(1, block, hidden_size))
    tt_out.deallocate(True)
    return tt_torch[:, :chunk_len]


def run_prefill_pcc_at_seq_len(
    ctx: PrefillPccContext,
    mesh_device,
    *,
    seq_len: int,
) -> None:
    """Chunked prefill PCC: O(block) host memory per step (default block=128), not O(seq_len)."""
    hidden_size = ctx.args.hidden_size
    layer_max_seq_len = ctx.args.max_seq_len
    block = ctx.args.kv_block_size
    cache = DynamicCache()
    all_pass = True
    min_pcc = 1.0

    logger.info(
        f"Prefill PCC: batch={PCC_BATCH_SIZE}, seq_len={seq_len}, hidden_size={hidden_size}, "
        f"layer_max_seq_len={layer_max_seq_len}, chunk={block}, pcc≥{PCC_REQUIRED}"
    )

    for chunk_start in range(0, seq_len, block):
        chunk_end = min(chunk_start + block, seq_len)
        chunk_len = chunk_end - chunk_start
        hidden = (torch.rand(PCC_BATCH_SIZE, chunk_len, hidden_size, dtype=torch.bfloat16) * 2) - 1

        ref_out = hf_prefill_forward_chunk(ctx.ref, ctx.ref_rope, hidden, chunk_start=chunk_start, cache=cache)
        tt_out = tt_prefill_forward_chunk(
            ctx.tt_layer,
            mesh_device,
            hidden,
            chunk_start=chunk_start,
            block=block,
            hidden_size=hidden_size,
        )

        passing, pcc_val = comp_pcc(ref_out, tt_out, PCC_REQUIRED)
        if passing:
            min_pcc = min(min_pcc, float(pcc_val))
        else:
            logger.info(comp_allclose(ref_out, tt_out))
            log_pcc_step(
                f"prefill seq_len={seq_len} chunk=[{chunk_start}:{chunk_end})",
                passing,
                str(pcc_val),
            )
            all_pass = False

    if all_pass:
        log_pcc_step(f"prefill seq_len={seq_len} (all chunks)", True, f"min PCC {min_pcc}")

    assert all_pass, (
        f"Prefill PCC below {PCC_REQUIRED} for seq_len={seq_len} "
        f"(layer_max_seq_len={layer_max_seq_len}). Check warnings."
    )


def tt_decode_forward(
    tt_layer: TtDecoderLayer,
    mesh_device,
    hidden: torch.Tensor,
    *,
    position: int,
    hidden_size: int,
    batch_size: int = PCC_BATCH_SIZE,
) -> torch.Tensor:
    assert batch_size == PCC_BATCH_SIZE, f"decode PCC tests only support batch_size={PCC_BATCH_SIZE}, got {batch_size}"
    tt_in = hidden_to_tt_decode(hidden, mesh_device, batch_size=batch_size)
    pos_tt = current_pos_to_tt(torch.tensor([position], dtype=torch.long), mesh_device)
    tt_out = tt_layer(tt_in, mode="decode", current_pos=pos_tt)
    return decode_tt_to_torch(tt_out, hidden_size=hidden_size, batch_size=batch_size)


def tt_prefill_forward(
    tt_layer: TtDecoderLayer,
    mesh_device,
    hidden: torch.Tensor,
    *,
    args: Devstral2Args,
) -> torch.Tensor:
    """Chunked prefill when ``seq_len > kv_block_size``; pad last partial chunk to block size."""
    seq_len = hidden.shape[1]
    hidden_size = hidden.shape[2]
    block = args.kv_block_size
    chunks: list[torch.Tensor] = []

    for chunk_start in range(0, seq_len, block):
        chunk_end = min(chunk_start + block, seq_len)
        chunk_len = chunk_end - chunk_start
        chunk = hidden[:, chunk_start:chunk_end]
        chunks.append(
            tt_prefill_forward_chunk(
                tt_layer,
                mesh_device,
                chunk,
                chunk_start=chunk_start,
                block=block,
                hidden_size=hidden_size,
            )
        )

    return torch.cat(chunks, dim=1)
