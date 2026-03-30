# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
import logging

import torch
from transformers import AutoTokenizer

from models.demos.speculative_deepseek_r1_broad.base_runtime import resolve_dtype
from models.demos.speculative_deepseek_r1_broad.reference.configuration_deepseek_r1 import (
    DeepSeekR1ReferenceConfig,
    load_reference_config,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bitonic sort / topk (mirrors deepseek_v3/reference/reference_utils.py)
# ---------------------------------------------------------------------------

def bitonic_sort(a, indices, up=True):
    def comp_and_swap(i, j, dir):
        if dir == (a[i] > a[j]):
            a[i], a[j] = a[j], a[i]
            indices[i], indices[j] = indices[j], indices[i]

    def bitonic_merge(low, cnt, dir):
        if cnt > 1:
            k = cnt // 2
            for i in range(low, low + k):
                comp_and_swap(i, i + k, dir)
            bitonic_merge(low, k, dir)
            bitonic_merge(low + k, k, dir)

    def bitonic_sort_rec(low, cnt, dir):
        if cnt > 1:
            k = cnt // 2
            bitonic_sort_rec(low, k, True)
            bitonic_sort_rec(low + k, k, False)
            bitonic_merge(low, cnt, dir)

    bitonic_sort_rec(0, len(a), up)


def topk_bitonic(input_tensor, k, dim=-1, largest=True, sorted=True):
    assert dim == -1, "This custom bitonic topK only supports last dimension"
    orig_shape = input_tensor.shape
    last_dim = orig_shape[dim]

    if (last_dim & (last_dim - 1)) != 0:
        raise ValueError("The last dimension must be a power of 2.")

    flat = input_tensor.reshape(-1, last_dim)
    topk_vals = []
    topk_indices = []

    for row in flat:
        row_vals = row.tolist()
        row_indices = list(range(len(row_vals)))

        bitonic_sort(row_vals, row_indices, up=not largest)

        selected = list(zip(row_vals[:k], row_indices[:k]))
        if sorted:
            selected.sort(reverse=largest, key=lambda x: x[0])
        vals_row, inds_row = zip(*selected)

        topk_vals.append(torch.tensor(vals_row, dtype=input_tensor.dtype))
        topk_indices.append(torch.tensor(inds_row, dtype=torch.long))

    topk_vals = torch.stack(topk_vals).reshape(*orig_shape[:-1], k)
    topk_indices = torch.stack(topk_indices).reshape(*orig_shape[:-1], k)
    return topk_vals, topk_indices


# ---------------------------------------------------------------------------
# Reference bundle (backward-compatible API)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReferenceBundle:
    tokenizer: object
    model: torch.nn.Module
    config: DeepSeekR1ReferenceConfig


def build_reference_bundle(
    model_id: str,
    *,
    device: str,
    torch_dtype: str,
    trust_remote_code: bool,
) -> ReferenceBundle:
    logger.info(
        "Loading reference bundle model_id=%s device=%s dtype=%s trust_remote_code=%s",
        model_id,
        device,
        torch_dtype,
        trust_remote_code,
    )
    from models.demos.speculative_deepseek_r1_broad.reference.modeling_deepseek_r1 import DeepSeekR1ReferenceForCausalLM

    cfg = load_reference_config(model_id, trust_remote_code=trust_remote_code)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model = DeepSeekR1ReferenceForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        torch_dtype=resolve_dtype(torch_dtype),
        low_cpu_mem_usage=True,
    ).to(torch.device(device))
    model.eval()
    return ReferenceBundle(tokenizer=tokenizer, model=model, config=cfg)


def summarize_model_structure(bundle: ReferenceBundle) -> str:
    cfg = bundle.config
    parts = [
        f"architecture={cfg.architecture}",
        f"hidden={cfg.hidden_size}",
        f"layers={cfg.num_hidden_layers}",
        f"heads={cfg.num_attention_heads}",
        f"kv_heads={cfg.num_key_value_heads}",
        f"vocab={cfg.vocab_size}",
        f"max_pos={cfg.max_position_embeddings}",
    ]
    if cfg.n_routed_experts:
        parts.extend(
            [
                f"routed_experts={cfg.n_routed_experts}",
                f"shared_experts={cfg.n_shared_experts}",
                f"experts_per_tok={cfg.num_experts_per_tok}",
                f"moe_inter={cfg.moe_intermediate_size}",
            ]
        )
    if cfg.kv_lora_rank:
        parts.extend(
            [
                f"kv_lora_rank={cfg.kv_lora_rank}",
                f"q_lora_rank={cfg.q_lora_rank}",
                f"qk_nope_dim={cfg.qk_nope_head_dim}",
                f"qk_rope_dim={cfg.qk_rope_head_dim}",
                f"v_dim={cfg.v_head_dim}",
            ]
        )
    return ", ".join(parts)
