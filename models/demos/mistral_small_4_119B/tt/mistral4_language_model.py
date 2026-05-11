# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Full Mistral Small 4 language model on TT: embed → decoder stack → final norm → lm_head.

Orchestrates:
    1. ``Mistral4Embedding1D`` / ``Mistral4Embedding2D`` — token embedding lookup
    2. ``forward_mistral4_text_stack_decode`` / ``forward_mistral4_text_stack_prefill`` — N decoder layers
    3. ``Mistral4LMHead`` — final RMSNorm + linear projection to vocab logits

This module provides build/forward helpers analogous to DeepSeek V3's ``RowBatchedModel``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from transformers.models.mistral4.configuration_mistral4 import Mistral4Config

import ttnn
from models.demos.mistral_small_4_119B.tt.embedding.mistral4_embedding_1d import Mistral4Embedding1D
from models.demos.mistral_small_4_119B.tt.lm_head import Mistral4LMHead
from models.demos.mistral_small_4_119B.tt.mistral4_text_stack import (
    build_mistral4_text_stack_decode_run_config,
    forward_mistral4_text_stack_decode,
)
from models.demos.mistral_small_4_119B.tt_utils.ccl import CCL
from models.demos.mistral_small_4_119B.tt_utils.run_config import create_run_config, deallocate_weight_config_tensors
from models.tt_transformers.tt.common import PagedAttentionConfig


def build_mistral4_language_model_decode(
    hf_config: Mistral4Config,
    mesh_device: ttnn.MeshDevice,
    fabric_config: ttnn.FabricConfig,
    batch_size_per_row: int,
    tmp_path: Path,
    paged_config: PagedAttentionConfig,
    ccl: CCL | None,
    *,
    num_layers: int | None = None,
    kvpe_seq_len: int,
    reference_batch_size: int,
    layer_state_dicts: Sequence[dict[str, torch.Tensor]] | None = None,
    embedding_state_dict: dict[str, torch.Tensor] | None = None,
    lm_head_state_dict: dict[str, torch.Tensor] | None = None,
) -> dict:
    """Build the full language model decode configuration.

    Returns a dict with:
        - ``embedding_run_config``: Config for embedding forward
        - ``text_stack_run_config``: Config for decoder layers forward
        - ``lm_head_run_config``: Config for final norm + lm_head forward
        - ``layer_block_classes``: Tuple of block classes per layer
        - ``weight_configs``: Dict of all weight configs for deallocation
        - ``page_table_mapping``: Shared page table mapping tensor
    """
    # ── Embedding ────────────────────────────────────────────────────────
    if embedding_state_dict is None:
        from transformers.models.mistral4.modeling_mistral4 import Mistral4Model

        dummy_model = Mistral4Model(hf_config).eval().to(torch.bfloat16)
        embedding_state_dict = {
            k.replace("embed_tokens.", ""): v.detach().clone() for k, v in dummy_model.embed_tokens.state_dict().items()
        }
        del dummy_model

    embedding_weight_cfg = Mistral4Embedding1D.convert_weights(
        hf_config, (embedding_state_dict,), tmp_path / "embedding", mesh_device
    )
    embedding_model_cfg = Mistral4Embedding1D.decode_model_config(hf_config, mesh_device)
    embedding_state = Mistral4Embedding1D.create_state(hf_config, mesh_device, ccl)
    embedding_run_cfg = create_run_config(embedding_model_cfg, embedding_weight_cfg, embedding_state, {})

    # ── Text stack (decoder layers) ──────────────────────────────────────
    (
        text_stack_run_cfg,
        layer_classes,
        text_stack_weight_cfg,
        page_mapping,
    ) = build_mistral4_text_stack_decode_run_config(
        hf_config,
        mesh_device,
        fabric_config,
        batch_size_per_row,
        tmp_path / "text_stack",
        paged_config,
        ccl,
        num_layers=num_layers,
        kvpe_seq_len=kvpe_seq_len,
        reference_batch_size=reference_batch_size,
        layer_state_dicts=layer_state_dicts,
    )

    # ── LM Head (final norm + projection) ────────────────────────────────
    if lm_head_state_dict is None:
        from transformers.models.mistral4.modeling_mistral4 import Mistral4ForCausalLM

        dummy_lm = Mistral4ForCausalLM(hf_config).eval().to(torch.bfloat16)
        lm_head_state_dict = {}
        for k, v in dummy_lm.model.norm.state_dict().items():
            lm_head_state_dict[f"norm.{k}"] = v.detach().clone()
        lm_head_state_dict["lm_head.weight"] = dummy_lm.lm_head.weight.detach().clone()
        del dummy_lm

    lm_head_weight_cfg = Mistral4LMHead.convert_weights(
        hf_config, (lm_head_state_dict,), tmp_path / "lm_head", mesh_device
    )
    lm_head_model_cfg = Mistral4LMHead.decode_model_config(hf_config, mesh_device, batch_size_per_row)
    if ccl is None:
        ccl = CCL(mesh_device)
    lm_head_state = Mistral4LMHead.create_state(hf_config, mesh_device, ccl)
    lm_head_run_cfg = create_run_config(lm_head_model_cfg, lm_head_weight_cfg, lm_head_state, {})

    return {
        "embedding_run_config": embedding_run_cfg,
        "text_stack_run_config": text_stack_run_cfg,
        "lm_head_run_config": lm_head_run_cfg,
        "layer_block_classes": layer_classes,
        "weight_configs": {
            "embedding": embedding_weight_cfg,
            "text_stack": text_stack_weight_cfg,
            "lm_head": lm_head_weight_cfg,
        },
        "page_table_mapping": page_mapping,
    }


def forward_mistral4_language_model_decode(
    token_ids: ttnn.Tensor,
    position_idxs: ttnn.Tensor,
    model_config: dict,
    rope_tensors: dict,
    page_table: ttnn.Tensor,
) -> ttnn.Tensor:
    """Full language model decode forward: embed → decoder stack → lm_head.

    Args:
        token_ids: Token indices [1, 1, batch_size] on device.
        position_idxs: Position indices for RoPE.
        model_config: Dict returned by ``build_mistral4_language_model_decode``.
        rope_tensors: RoPE cos/sin/trans_matrix tensors on device.
        page_table: Paged KV cache table on device.

    Returns:
        Logits tensor [1, 1, batch_size, vocab_size] on device.
    """
    # 1. Embedding lookup
    x = Mistral4Embedding1D.forward_decode(token_ids, model_config["embedding_run_config"])

    # 2. Decoder stack
    x = forward_mistral4_text_stack_decode(
        x,
        position_idxs,
        model_config["text_stack_run_config"],
        model_config["layer_block_classes"],
        rope_tensors,
        page_table,
    )

    # 3. Final norm + LM head
    logits = Mistral4LMHead.forward_decode(x, model_config["lm_head_run_config"])

    return logits


def deallocate_language_model_weights(model_config: dict) -> None:
    """Deallocate all weight tensors from a built language model config."""
    weight_configs = model_config.get("weight_configs", {})
    for key, wcfg in weight_configs.items():
        if wcfg is not None:
            deallocate_weight_config_tensors(wcfg)
